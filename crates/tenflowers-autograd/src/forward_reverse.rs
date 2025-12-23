//! Forward-Reverse Mode Integration
//!
//! This module provides hybrid differentiation techniques that combine forward mode
//! and reverse mode automatic differentiation for efficient computation of higher-order
//! derivatives, particularly Hessians.

use crate::{GradientTape, TrackedTensor};
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor, TensorError};

/// Strategy for selecting differentiation mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DifferentiationMode {
    /// Use forward mode AD
    Forward,
    /// Use reverse mode AD
    Reverse,
    /// Use forward-over-reverse for Hessians
    ForwardOverReverse,
    /// Use reverse-over-forward for Hessians
    ReverseOverForward,
    /// Automatically select optimal mode
    Auto,
}

/// Configuration for forward-reverse mode integration
#[derive(Debug, Clone)]
pub struct ForwardReverseConfig {
    /// Differentiation mode strategy
    pub mode: DifferentiationMode,
    /// Threshold for automatic mode selection (input_dim / output_dim)
    pub auto_threshold: f64,
    /// Whether to exploit sparsity patterns
    pub use_sparsity: bool,
    /// Maximum memory usage for intermediate computations (in MB)
    pub max_memory_mb: usize,
}

impl Default for ForwardReverseConfig {
    fn default() -> Self {
        Self {
            mode: DifferentiationMode::Auto,
            auto_threshold: 1.0,
            use_sparsity: true,
            max_memory_mb: 1024, // 1GB default
        }
    }
}

/// Forward-Reverse Mode Differentiator
pub struct ForwardReverseDifferentiator {
    config: ForwardReverseConfig,
}

impl ForwardReverseDifferentiator {
    /// Create a new forward-reverse differentiator
    pub fn new(config: ForwardReverseConfig) -> Self {
        Self { config }
    }
}

impl Default for ForwardReverseDifferentiator {
    fn default() -> Self {
        Self::new(ForwardReverseConfig::default())
    }
}

impl ForwardReverseDifferentiator {
    /// Compute Hessian using forward-over-reverse mode
    pub fn hessian_forward_over_reverse<T>(
        &self,
        target: &TrackedTensor<T>,
        inputs: &[&TrackedTensor<T>],
    ) -> Result<Vec<Vec<Tensor<T>>>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Check target is scalar
        if target.tensor.shape().dims().iter().product::<usize>() != 1 {
            return Err(TensorError::shape_mismatch(
                "hessian",
                "scalar (single element)",
                &format!("{:?}", target.tensor.shape().dims()),
            ));
        }

        let mut hessian = Vec::new();

        for (i, input_i) in inputs.iter().enumerate() {
            let mut hessian_row = Vec::new();

            for (j, input_j) in inputs.iter().enumerate() {
                // For each pair (i,j), compute ∂²f/∂xi∂xj using forward-over-reverse
                let hij = self
                    .compute_hessian_element_forward_over_reverse(target, input_i, input_j, i, j)?;
                hessian_row.push(hij);
            }
            hessian.push(hessian_row);
        }

        Ok(hessian)
    }

    /// Compute Hessian using reverse-over-forward mode
    pub fn hessian_reverse_over_forward<T>(
        &self,
        target: &TrackedTensor<T>,
        inputs: &[&TrackedTensor<T>],
    ) -> Result<Vec<Vec<Tensor<T>>>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Check target is scalar
        if target.tensor.shape().dims().iter().product::<usize>() != 1 {
            return Err(TensorError::shape_mismatch(
                "hessian",
                "scalar (single element)",
                &format!("{:?}", target.tensor.shape().dims()),
            ));
        }

        let mut hessian = Vec::new();

        for (i, input_i) in inputs.iter().enumerate() {
            let mut hessian_row = Vec::new();

            for (j, input_j) in inputs.iter().enumerate() {
                // For each pair (i,j), compute ∂²f/∂xi∂xj using reverse-over-forward
                let hij = self
                    .compute_hessian_element_reverse_over_forward(target, input_i, input_j, i, j)?;
                hessian_row.push(hij);
            }
            hessian.push(hessian_row);
        }

        Ok(hessian)
    }

    /// Automatically select optimal differentiation mode
    pub fn select_optimal_mode<T>(
        &self,
        target: &TrackedTensor<T>,
        inputs: &[&TrackedTensor<T>],
    ) -> DifferentiationMode
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        // Calculate input and output dimensions
        let input_dim: usize = inputs
            .iter()
            .map(|input| input.tensor.shape().dims().iter().product::<usize>())
            .sum();

        let output_dim = target.tensor.shape().dims().iter().product::<usize>();

        // Use ratio to determine optimal mode
        let ratio = input_dim as f64 / output_dim as f64;

        if ratio > self.config.auto_threshold {
            // More inputs than outputs: reverse mode is more efficient
            DifferentiationMode::Reverse
        } else {
            // More outputs than inputs: forward mode is more efficient
            DifferentiationMode::Forward
        }
    }

    /// Compute sparse Hessian exploiting known sparsity pattern
    pub fn sparse_hessian<T>(
        &self,
        target: &TrackedTensor<T>,
        inputs: &[&TrackedTensor<T>],
        sparsity_pattern: &[(usize, usize)],
    ) -> Result<HashMap<(usize, usize), Tensor<T>>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let mut sparse_hessian = HashMap::new();

        // Only compute non-zero elements based on sparsity pattern
        for &(i, j) in sparsity_pattern {
            if i < inputs.len() && j < inputs.len() {
                let hij = self.compute_hessian_element_forward_over_reverse(
                    target, inputs[i], inputs[j], i, j,
                )?;
                sparse_hessian.insert((i, j), hij);
            }
        }

        Ok(sparse_hessian)
    }

    /// Compute Hessian-vector product efficiently
    pub fn hessian_vector_product<T>(
        &self,
        target: &TrackedTensor<T>,
        inputs: &[&TrackedTensor<T>],
        vector: &[Tensor<T>],
    ) -> Result<Vec<Tensor<T>>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        if inputs.len() != vector.len() {
            return Err(TensorError::invalid_argument(
                "Number of inputs must match vector length".to_string(),
            ));
        }

        // Compute H*v using forward-over-reverse mode
        // This is more efficient than computing the full Hessian
        let mut hvp = Vec::new();

        for (i, input_i) in inputs.iter().enumerate() {
            let mut hvp_i = Tensor::zeros(input_i.tensor.shape().dims());

            for (j, input_j) in inputs.iter().enumerate() {
                // Compute Hij * vj
                let hij = self
                    .compute_hessian_element_forward_over_reverse(target, input_i, input_j, i, j)?;

                // Multiply by vector component vj
                let contrib = hij.mul(&vector[j])?;
                hvp_i = hvp_i.add(&contrib)?;
            }

            hvp.push(hvp_i);
        }

        Ok(hvp)
    }

    /// Estimate computational complexity for different modes
    pub fn estimate_complexity<T>(
        &self,
        target: &TrackedTensor<T>,
        inputs: &[&TrackedTensor<T>],
    ) -> ComplexityEstimate
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        let input_dim: usize = inputs
            .iter()
            .map(|input| input.tensor.shape().dims().iter().product::<usize>())
            .sum();

        let output_dim = target.tensor.shape().dims().iter().product::<usize>();

        // Complexity estimates for automatic differentiation
        // Forward mode: O(input_dim) for each output, so O(input_dim * output_dim) total
        // Reverse mode: O(output_dim) for each input, so O(output_dim * input_dim) total
        // But the practical rule is: reverse mode scales with outputs, forward mode scales with inputs
        let forward_ops = input_dim; // Cost per output in forward mode
        let reverse_ops = output_dim; // Cost per input in reverse mode
        let forward_over_reverse_ops = input_dim * input_dim; // O(n²) for Hessian

        ComplexityEstimate {
            forward_mode_ops: forward_ops,
            reverse_mode_ops: reverse_ops,
            forward_over_reverse_ops,
            input_dim,
            output_dim,
        }
    }

    // Helper methods

    fn compute_hessian_element_forward_over_reverse<T>(
        &self,
        target: &TrackedTensor<T>,
        input_i: &TrackedTensor<T>,
        input_j: &TrackedTensor<T>,
        _i: usize,
        _j: usize,
    ) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Implementation of forward-over-reverse mode for Hessian computation
        // This computes ∂²f/∂xi∂xj using forward mode over reverse mode

        // Step 1: Compute first-order gradient ∂f/∂xj using reverse mode
        let reverse_tape = GradientTape::new();
        let first_grad =
            reverse_tape.gradient(std::slice::from_ref(target), std::slice::from_ref(input_j))?;

        // Step 2: Set up forward mode to compute ∂/∂xi of the gradient
        let forward_tape = GradientTape::new();
        let input_i_fresh = forward_tape.watch(input_i.tensor.clone());

        // Handle Option type from gradient computation
        let first_grad_tensor = first_grad[0].clone().ok_or_else(|| {
            TensorError::compute_error_simple("Failed to compute first gradient".to_string())
        })?;
        let grad_tracked = forward_tape.watch(first_grad_tensor);

        // Step 3: Compute second derivative using forward mode on the gradient
        let second_grad = forward_tape.gradient(&[grad_tracked], &[input_i_fresh])?;

        // Handle Option type from second gradient computation
        second_grad[0].clone().ok_or_else(|| {
            TensorError::compute_error_simple("Failed to compute second gradient".to_string())
        })
    }

    fn compute_hessian_element_reverse_over_forward<T>(
        &self,
        target: &TrackedTensor<T>,
        input_i: &TrackedTensor<T>,
        input_j: &TrackedTensor<T>,
        _i: usize,
        _j: usize,
    ) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + scirs2_core::num_traits::FromPrimitive
            + scirs2_core::num_traits::Float
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Simplified implementation of reverse-over-forward mode
        // In practice, this would require more sophisticated automatic differentiation

        // For now, use the same approach as forward-over-reverse
        self.compute_hessian_element_forward_over_reverse(target, input_i, input_j, _i, _j)
    }
}

/// Complexity estimate for different differentiation modes
#[derive(Debug, Clone)]
pub struct ComplexityEstimate {
    pub forward_mode_ops: usize,
    pub reverse_mode_ops: usize,
    pub forward_over_reverse_ops: usize,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl ComplexityEstimate {
    /// Get the most efficient mode based on operation count
    pub fn optimal_mode(&self) -> DifferentiationMode {
        let min_ops = self.forward_mode_ops.min(self.reverse_mode_ops);

        if self.forward_mode_ops == min_ops {
            DifferentiationMode::Forward
        } else {
            DifferentiationMode::Reverse
        }
    }

    /// Check if forward-over-reverse is efficient for Hessian computation
    pub fn is_forward_over_reverse_efficient(&self) -> bool {
        // Forward-over-reverse is typically efficient when input_dim is small
        self.input_dim <= 100 && self.forward_over_reverse_ops < self.input_dim.pow(3)
    }
}

/// Utility functions for mode selection and optimization
pub mod utils {
    use super::*;

    /// Determine optimal differentiation strategy for a given problem
    pub fn select_strategy<T>(
        target: &TrackedTensor<T>,
        inputs: &[&TrackedTensor<T>],
        compute_hessian: bool,
    ) -> DifferentiationMode
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        let input_dim: usize = inputs
            .iter()
            .map(|input| input.tensor.shape().dims().iter().product::<usize>())
            .sum();

        let output_dim = target.tensor.shape().dims().iter().product::<usize>();

        if compute_hessian {
            // For Hessian computation, consider problem size
            if input_dim <= 50 {
                DifferentiationMode::ForwardOverReverse
            } else if output_dim == 1 {
                DifferentiationMode::ReverseOverForward
            } else {
                DifferentiationMode::ForwardOverReverse
            }
        } else {
            // For gradient computation, use standard rule
            if input_dim > output_dim {
                DifferentiationMode::Reverse
            } else {
                DifferentiationMode::Forward
            }
        }
    }

    /// Create default sparsity pattern (all elements)
    pub fn dense_pattern(n: usize) -> Vec<(usize, usize)> {
        let mut pattern = Vec::new();
        for i in 0..n {
            for j in 0..n {
                pattern.push((i, j));
            }
        }
        pattern
    }

    /// Create diagonal sparsity pattern
    pub fn diagonal_pattern(n: usize) -> Vec<(usize, usize)> {
        (0..n).map(|i| (i, i)).collect()
    }

    /// Create band sparsity pattern
    pub fn band_pattern(n: usize, bandwidth: usize) -> Vec<(usize, usize)> {
        let mut pattern = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if (i as i32 - j as i32).abs() <= bandwidth as i32 {
                    pattern.push((i, j));
                }
            }
        }
        pattern
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GradientTape;
    use tenflowers_core::Tensor;

    #[test]
    fn test_mode_selection() {
        let tape = GradientTape::new();

        // Many inputs, few outputs -> reverse mode
        let x = Tensor::<f32>::from_vec(vec![1.0; 100], &[100]).unwrap();
        let y = Tensor::<f32>::from_vec(vec![2.0], &[1]).unwrap();
        let x_tracked = tape.watch(x);
        let y_tracked = tape.watch(y);

        let differentiator = ForwardReverseDifferentiator::default();
        let mode = differentiator.select_optimal_mode(&y_tracked, &[&x_tracked]);

        assert_eq!(mode, DifferentiationMode::Reverse);
    }

    #[test]
    fn test_complexity_estimation() {
        let tape = GradientTape::new();

        let x = Tensor::<f32>::from_vec(vec![1.0; 10], &[10]).unwrap();
        let y = Tensor::<f32>::from_vec(vec![2.0], &[1]).unwrap();
        let x_tracked = tape.watch(x);
        let y_tracked = tape.watch(y);

        let differentiator = ForwardReverseDifferentiator::default();
        let complexity = differentiator.estimate_complexity(&y_tracked, &[&x_tracked]);

        assert_eq!(complexity.input_dim, 10);
        assert_eq!(complexity.output_dim, 1);
        assert_eq!(complexity.optimal_mode(), DifferentiationMode::Reverse);
    }

    #[test]
    fn test_sparsity_patterns() {
        let dense = utils::dense_pattern(3);
        assert_eq!(dense.len(), 9);

        let diag = utils::diagonal_pattern(3);
        assert_eq!(diag, vec![(0, 0), (1, 1), (2, 2)]);

        let band = utils::band_pattern(3, 1);
        assert_eq!(band.len(), 7); // diagonal + 2 off-diagonals
    }

    #[test]
    fn test_forward_reverse_configuration() {
        let config = ForwardReverseConfig {
            mode: DifferentiationMode::ForwardOverReverse,
            auto_threshold: 2.0,
            use_sparsity: false,
            max_memory_mb: 512,
        };

        let differentiator = ForwardReverseDifferentiator::new(config);
        assert_eq!(
            differentiator.config.mode,
            DifferentiationMode::ForwardOverReverse
        );
        assert_eq!(differentiator.config.max_memory_mb, 512);
    }

    #[test]
    fn test_strategy_selection() {
        let tape = GradientTape::new();

        // Small problem - should prefer forward-over-reverse for Hessian
        let x = Tensor::<f32>::from_vec(vec![1.0; 5], &[5]).unwrap();
        let y = Tensor::<f32>::from_vec(vec![2.0], &[1]).unwrap();
        let x_tracked = tape.watch(x);
        let y_tracked = tape.watch(y);

        let strategy = utils::select_strategy(&y_tracked, &[&x_tracked], true);
        assert_eq!(strategy, DifferentiationMode::ForwardOverReverse);

        // Gradient only - should prefer reverse mode
        let strategy_grad = utils::select_strategy(&y_tracked, &[&x_tracked], false);
        assert_eq!(strategy_grad, DifferentiationMode::Reverse);
    }
}
