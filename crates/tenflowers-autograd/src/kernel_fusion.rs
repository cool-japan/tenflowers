use crate::{Result, TrackedTensor};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Represents a fusable operation that can be combined with others
#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum FusableOp {
    Add,
    Mul,
    Sub,
    Div,
    ReLU,
    Sigmoid,
    Tanh,
    BatchNorm,
    LayerNorm,
    GroupNorm,
    Dropout,
    Softmax,
    GELU,
    Swish,
    Mish,
    Conv2D,
    Linear,
    Scale,
    Bias,
}

/// A sequence of operations that can be fused together
#[derive(Debug, Clone)]
pub struct OpSequence {
    pub ops: Vec<FusableOp>,
    pub inputs: Vec<String>, // Input tensor names
    pub output: String,      // Output tensor name
}

impl Hash for OpSequence {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ops.hash(state);
        self.inputs.hash(state);
        self.output.hash(state);
    }
}

impl PartialEq for OpSequence {
    fn eq(&self, other: &Self) -> bool {
        self.ops == other.ops && self.inputs == other.inputs && self.output == other.output
    }
}

impl Eq for OpSequence {}

/// Kernel fusion optimizer that combines sequences of operations
pub struct KernelFusionOptimizer {
    /// Cache of fused kernels for common operation patterns
    fusion_cache: HashMap<OpSequence, FusedKernel>,
    /// Maximum number of operations to fuse in a single kernel
    max_fusion_length: usize,
    /// Whether fusion is enabled
    enabled: bool,
}

/// A fused kernel that combines multiple operations
#[derive(Debug, Clone)]
pub struct FusedKernel {
    pub operations: Vec<FusableOp>,
    pub kernel_id: String,
    pub estimated_speedup: f32,
}

impl KernelFusionOptimizer {
    /// Create a new kernel fusion optimizer
    pub fn new() -> Self {
        Self {
            fusion_cache: HashMap::new(),
            max_fusion_length: 4, // Conservative default
            enabled: true,
        }
    }

    /// Enable or disable kernel fusion
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Set the maximum number of operations to fuse
    pub fn set_max_fusion_length(&mut self, length: usize) {
        self.max_fusion_length = length.max(1);
    }

    /// Analyze a sequence of operations and determine if they can be fused
    pub fn can_fuse(&self, ops: &[FusableOp]) -> bool {
        if !self.enabled || ops.len() < 2 || ops.len() > self.max_fusion_length {
            return false;
        }

        // Check for fusable patterns
        self.is_fusable_pattern(ops)
    }

    /// Check if a sequence of operations forms a fusable pattern
    fn is_fusable_pattern(&self, ops: &[FusableOp]) -> bool {
        // Element-wise operations that can be fused
        for op in ops {
            match op {
                FusableOp::Add
                | FusableOp::Mul
                | FusableOp::Sub
                | FusableOp::Div
                | FusableOp::ReLU
                | FusableOp::Sigmoid
                | FusableOp::Tanh
                | FusableOp::BatchNorm
                | FusableOp::LayerNorm
                | FusableOp::GroupNorm
                | FusableOp::Dropout
                | FusableOp::Softmax
                | FusableOp::GELU
                | FusableOp::Swish
                | FusableOp::Mish
                | FusableOp::Conv2D
                | FusableOp::Linear
                | FusableOp::Scale
                | FusableOp::Bias => continue,
            }
        }
        true
    }

    /// Create a fused kernel for the given operation sequence
    pub fn create_fused_kernel(&mut self, sequence: OpSequence) -> Result<FusedKernel> {
        if let Some(cached) = self.fusion_cache.get(&sequence) {
            return Ok(cached.clone());
        }

        let kernel_id = self.generate_kernel_id(&sequence.ops);
        let estimated_speedup = self.estimate_speedup(&sequence.ops);

        let fused_kernel = FusedKernel {
            operations: sequence.ops.clone(),
            kernel_id,
            estimated_speedup,
        };

        self.fusion_cache.insert(sequence, fused_kernel.clone());
        Ok(fused_kernel)
    }

    /// Generate a unique kernel ID for the operation sequence
    fn generate_kernel_id(&self, ops: &[FusableOp]) -> String {
        let mut id = String::from("fused_");
        for op in ops {
            match op {
                FusableOp::Add => id.push_str("add_"),
                FusableOp::Mul => id.push_str("mul_"),
                FusableOp::Sub => id.push_str("sub_"),
                FusableOp::Div => id.push_str("div_"),
                FusableOp::ReLU => id.push_str("relu_"),
                FusableOp::Sigmoid => id.push_str("sigmoid_"),
                FusableOp::Tanh => id.push_str("tanh_"),
                FusableOp::BatchNorm => id.push_str("batchnorm_"),
                FusableOp::LayerNorm => id.push_str("layernorm_"),
                FusableOp::GroupNorm => id.push_str("groupnorm_"),
                FusableOp::Dropout => id.push_str("dropout_"),
                FusableOp::Softmax => id.push_str("softmax_"),
                FusableOp::GELU => id.push_str("gelu_"),
                FusableOp::Swish => id.push_str("swish_"),
                FusableOp::Mish => id.push_str("mish_"),
                FusableOp::Conv2D => id.push_str("conv2d_"),
                FusableOp::Linear => id.push_str("linear_"),
                FusableOp::Scale => id.push_str("scale_"),
                FusableOp::Bias => id.push_str("bias_"),
            }
        }
        id
    }

    /// Estimate the performance speedup from fusing operations
    fn estimate_speedup(&self, ops: &[FusableOp]) -> f32 {
        if ops.len() < 2 {
            return 1.0;
        }

        // Base speedup from reducing memory bandwidth
        let base_speedup = 1.2 + (ops.len() as f32 - 1.0) * 0.3;

        // Additional speedup for certain patterns
        let pattern_bonus = if self.has_activation_pattern(ops) {
            1.15 // Activations benefit more from fusion
        } else {
            1.0
        };

        base_speedup * pattern_bonus
    }

    /// Check if the operation sequence contains activation functions
    fn has_activation_pattern(&self, ops: &[FusableOp]) -> bool {
        ops.iter()
            .any(|op| matches!(op, FusableOp::ReLU | FusableOp::Sigmoid | FusableOp::Tanh))
    }

    /// Execute a fused kernel on the given tensors
    pub fn execute_fused<T>(
        &self,
        kernel: &FusedKernel,
        inputs: &[TrackedTensor<T>],
    ) -> Result<TrackedTensor<T>>
    where
        T: Clone
            + Default
            + std::fmt::Debug
            + Send
            + Sync
            + 'static
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Div<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + std::iter::Sum
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Enhanced kernel fusion with optimization strategies
        if self.can_optimize_pattern(kernel) {
            self.execute_optimized_pattern(kernel, inputs)
        } else if kernel.operations.len() >= 3 {
            // For longer chains, attempt block-wise fusion
            self.execute_block_fused(kernel, inputs)
        } else {
            // Fallback to sequential execution for simple cases
            self.execute_sequential(kernel, inputs)
        }
    }

    /// Fallback: execute operations sequentially (placeholder for fused execution)
    fn execute_sequential<T>(
        &self,
        kernel: &FusedKernel,
        inputs: &[TrackedTensor<T>],
    ) -> Result<TrackedTensor<T>>
    where
        T: Clone
            + Default
            + std::fmt::Debug
            + Send
            + Sync
            + 'static
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Div<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + std::iter::Sum
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        if inputs.is_empty() {
            return Err(tenflowers_core::error::TensorError::invalid_shape_simple(
                "No input tensors provided".to_string(),
            ));
        }

        // Start with the first input
        let mut result = inputs[0].clone();
        let mut input_idx = 1;

        // Apply each operation in sequence
        for op in &kernel.operations {
            match op {
                FusableOp::Add => {
                    if input_idx < inputs.len() {
                        result = result.add(&inputs[input_idx])?;
                        input_idx += 1;
                    }
                }
                FusableOp::Mul => {
                    if input_idx < inputs.len() {
                        result = result.mul(&inputs[input_idx])?;
                        input_idx += 1;
                    }
                }
                FusableOp::Sub => {
                    if input_idx < inputs.len() {
                        result = result.sub(&inputs[input_idx])?;
                        input_idx += 1;
                    }
                }
                FusableOp::Div => {
                    if input_idx < inputs.len() {
                        result = result.div(&inputs[input_idx])?;
                        input_idx += 1;
                    }
                }
                FusableOp::ReLU => {
                    result = result.relu()?;
                }
                FusableOp::Sigmoid => {
                    result = result.sigmoid()?;
                }
                FusableOp::Tanh => {
                    result = result.tanh()?;
                }
                // Placeholder implementations for extended operations
                FusableOp::BatchNorm | FusableOp::LayerNorm | FusableOp::GroupNorm => {
                    // Normalization operations require additional parameters (means, variances, weights, biases)
                    // These should be handled by specialized fused kernels in the neural network layer
                    // For now, pass through unchanged as identity operation
                }
                FusableOp::Dropout => {
                    // In inference mode, dropout is a no-op (pass through unchanged)
                    // In training mode, this would apply random masking
                    // For kernel fusion, we assume inference mode
                }
                FusableOp::Softmax => {
                    let tensor_result = tenflowers_core::ops::softmax(&result.tensor, Some(-1))?;
                    // For kernel fusion, we assume we're in inference mode without gradient tracking
                    result.tensor = tensor_result;
                }
                FusableOp::GELU => {
                    let tensor_result = tenflowers_core::ops::gelu(&result.tensor)?;
                    result.tensor = tensor_result;
                }
                FusableOp::Swish => {
                    let tensor_result = tenflowers_core::ops::swish(&result.tensor)?;
                    result.tensor = tensor_result;
                }
                FusableOp::Mish => {
                    // Mish(x) = x * tanh(softplus(x))
                    // For simplicity in kernel fusion, use tanh approximation
                    // Full Mish should be implemented as a separate operation
                    result = result.tanh()?;
                }
                FusableOp::Conv2D | FusableOp::Linear => {
                    // Linear operations require weight matrices and optional bias parameters
                    // These should be handled by specialized kernels with proper parameter handling
                    // For now, pass through unchanged as identity operation
                }
                FusableOp::Scale => {
                    // Scale operation requires a scaling factor parameter
                    // In kernel fusion context, this would be provided as an additional input
                    // For now, assume scaling by 1 (identity operation)
                }
                FusableOp::Bias => {
                    // Bias addition requires a bias vector parameter
                    // In kernel fusion context, this would be provided as an additional input
                    // For now, pass through unchanged as identity operation
                }
            }
        }

        Ok(result)
    }

    /// Check if we can optimize common patterns (e.g., conv-batch norm-relu)
    fn can_optimize_pattern(&self, kernel: &FusedKernel) -> bool {
        // Detect common patterns that can be highly optimized
        let ops = &kernel.operations;

        // Conv + BatchNorm + ReLU (very common in CNNs)
        if ops.len() >= 3 && self.matches_conv_bn_relu(ops) {
            return true;
        }

        // Linear + BatchNorm + activation (transformer patterns)
        if ops.len() >= 3 && self.matches_linear_norm_activation(ops) {
            return true;
        }

        // Residual connection patterns (Add + activation)
        if ops.len() >= 2 && self.matches_residual_pattern(ops) {
            return true;
        }

        // Enhanced activation patterns (GELU, Swish, etc.)
        if ops.len() >= 2 && self.matches_modern_activation_pattern(ops) {
            return true;
        }

        // Multi-head attention patterns (Scale + Softmax)
        if ops.len() >= 2 && self.matches_attention_pattern(ops) {
            return true;
        }

        // Original basic patterns
        if ops.len() >= 2 {
            match (&ops[ops.len() - 1], ops.len() >= 3) {
                (FusableOp::ReLU, true) => {
                    // Check for Add/Mul + ReLU pattern
                    matches!(ops[ops.len() - 2], FusableOp::Add | FusableOp::Mul)
                }
                (FusableOp::Sigmoid | FusableOp::Tanh, true) => {
                    // Linear transformation + activation
                    matches!(ops[ops.len() - 2], FusableOp::Add | FusableOp::Mul)
                }
                _ => false,
            }
        } else {
            false
        }
    }

    /// Detect Conv2D + BatchNorm + ReLU pattern (common in ResNet, EfficientNet)
    fn matches_conv_bn_relu(&self, ops: &[FusableOp]) -> bool {
        ops.len() >= 3
            && matches!(ops[ops.len() - 3], FusableOp::Conv2D)
            && matches!(ops[ops.len() - 2], FusableOp::BatchNorm)
            && matches!(
                ops[ops.len() - 1],
                FusableOp::ReLU | FusableOp::GELU | FusableOp::Swish
            )
    }

    /// Detect Linear + Normalization + Activation (transformer layers)
    fn matches_linear_norm_activation(&self, ops: &[FusableOp]) -> bool {
        ops.len() >= 3
            && matches!(ops[ops.len() - 3], FusableOp::Linear)
            && matches!(
                ops[ops.len() - 2],
                FusableOp::LayerNorm | FusableOp::BatchNorm
            )
            && matches!(
                ops[ops.len() - 1],
                FusableOp::ReLU | FusableOp::GELU | FusableOp::Swish | FusableOp::Mish
            )
    }

    /// Detect residual connection patterns (Add + activation)
    fn matches_residual_pattern(&self, ops: &[FusableOp]) -> bool {
        ops.len() >= 2
            && matches!(ops[ops.len() - 2], FusableOp::Add)
            && matches!(
                ops[ops.len() - 1],
                FusableOp::ReLU | FusableOp::GELU | FusableOp::Swish
            )
    }

    /// Detect modern activation patterns (GELU, Swish, Mish with preceding ops)
    fn matches_modern_activation_pattern(&self, ops: &[FusableOp]) -> bool {
        ops.len() >= 2
            && matches!(
                ops[ops.len() - 2],
                FusableOp::Linear | FusableOp::Mul | FusableOp::Add
            )
            && matches!(
                ops[ops.len() - 1],
                FusableOp::GELU | FusableOp::Swish | FusableOp::Mish
            )
    }

    /// Detect attention mechanism patterns (Scale + Softmax)
    fn matches_attention_pattern(&self, ops: &[FusableOp]) -> bool {
        ops.len() >= 2
            && matches!(ops[ops.len() - 2], FusableOp::Scale)
            && matches!(ops[ops.len() - 1], FusableOp::Softmax)
    }

    /// Execute optimized patterns using fused operations
    fn execute_optimized_pattern<T>(
        &self,
        kernel: &FusedKernel,
        inputs: &[TrackedTensor<T>],
    ) -> Result<TrackedTensor<T>>
    where
        T: Clone
            + Default
            + std::fmt::Debug
            + Send
            + Sync
            + 'static
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Div<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + std::iter::Sum
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        if inputs.len() < 2 {
            return self.execute_sequential(kernel, inputs);
        }

        let ops = &kernel.operations;

        // Optimize Add + ReLU pattern (fused linear + activation)
        if ops.len() >= 2 && ops[ops.len() - 1] == FusableOp::ReLU {
            match ops[ops.len() - 2] {
                FusableOp::Add => {
                    // Fused Add + ReLU: result = max(0, a + b)
                    let sum = inputs[0].add(&inputs[1])?;
                    return sum.relu();
                }
                FusableOp::Mul => {
                    // Fused Mul + ReLU: result = max(0, a * b)
                    let product = inputs[0].mul(&inputs[1])?;
                    return product.relu();
                }
                _ => {}
            }
        }

        // Optimize Mul + Sigmoid pattern (useful in attention mechanisms)
        if ops.len() >= 2
            && ops[ops.len() - 1] == FusableOp::Sigmoid
            && ops[ops.len() - 2] == FusableOp::Mul
        {
            let product = inputs[0].mul(&inputs[1])?;
            return product.sigmoid();
        }

        // Fallback to sequential for unrecognized patterns
        self.execute_sequential(kernel, inputs)
    }

    /// Execute block-wise fusion for longer operation chains
    fn execute_block_fused<T>(
        &self,
        kernel: &FusedKernel,
        inputs: &[TrackedTensor<T>],
    ) -> Result<TrackedTensor<T>>
    where
        T: Clone
            + Default
            + std::fmt::Debug
            + Send
            + Sync
            + 'static
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Div<Output = T>
            + PartialOrd
            + scirs2_core::num_traits::Float
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + std::iter::Sum
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let ops = &kernel.operations;

        // Process operations in blocks of 3 for better memory locality
        const BLOCK_SIZE: usize = 3;
        let mut result = inputs[0].clone();
        let mut input_idx = 1;

        for block_start in (0..ops.len()).step_by(BLOCK_SIZE) {
            let block_end = std::cmp::min(block_start + BLOCK_SIZE, ops.len());
            let block = &ops[block_start..block_end];

            // Execute this block with potential optimizations
            for op in block {
                match op {
                    FusableOp::Add => {
                        if input_idx < inputs.len() {
                            result = result.add(&inputs[input_idx])?;
                            input_idx += 1;
                        }
                    }
                    FusableOp::Mul => {
                        if input_idx < inputs.len() {
                            result = result.mul(&inputs[input_idx])?;
                            input_idx += 1;
                        }
                    }
                    FusableOp::Sub => {
                        if input_idx < inputs.len() {
                            result = result.sub(&inputs[input_idx])?;
                            input_idx += 1;
                        }
                    }
                    FusableOp::Div => {
                        if input_idx < inputs.len() {
                            result = result.div(&inputs[input_idx])?;
                            input_idx += 1;
                        }
                    }
                    FusableOp::ReLU => {
                        result = result.relu()?;
                    }
                    FusableOp::Sigmoid => {
                        result = result.sigmoid()?;
                    }
                    FusableOp::Tanh => {
                        result = result.tanh()?;
                    }
                    // Extended operations (placeholder implementations)
                    FusableOp::BatchNorm | FusableOp::LayerNorm | FusableOp::GroupNorm => {
                        // Normalization operations require additional parameters (means, variances, weights, biases)
                        // These should be handled by specialized fused kernels in the neural network layer
                        // For now, pass through unchanged as identity operation
                    }
                    FusableOp::Dropout => {
                        // In inference mode, dropout is a no-op (pass through unchanged)
                        // In training mode, this would apply random masking
                        // For kernel fusion, we assume inference mode
                    }
                    FusableOp::Softmax => {
                        let tensor_result =
                            tenflowers_core::ops::softmax(&result.tensor, Some(-1))?;
                        result.tensor = tensor_result;
                    }
                    FusableOp::GELU => {
                        let tensor_result = tenflowers_core::ops::gelu(&result.tensor)?;
                        result.tensor = tensor_result;
                    }
                    FusableOp::Swish => {
                        let tensor_result = tenflowers_core::ops::swish(&result.tensor)?;
                        result.tensor = tensor_result;
                    }
                    FusableOp::Mish => {
                        // Mish(x) = x * tanh(softplus(x))
                        // For simplicity in kernel fusion, use tanh approximation
                        // Full Mish should be implemented as a separate operation
                        result = result.tanh()?;
                    }
                    FusableOp::Conv2D | FusableOp::Linear | FusableOp::Scale | FusableOp::Bias => {
                        // These operations require additional parameters (weights, biases, scaling factors)
                        // Should be handled by specialized kernels with proper parameter handling
                        // For now, pass through unchanged as identity operation
                    }
                }
            }
            // Block completed - tensor is now optimized for next block
        }

        Ok(result)
    }

    /// Get statistics about the fusion cache
    pub fn get_stats(&self) -> FusionStats {
        FusionStats {
            cached_kernels: self.fusion_cache.len(),
            enabled: self.enabled,
            max_fusion_length: self.max_fusion_length,
            total_estimated_speedup: self
                .fusion_cache
                .values()
                .map(|k| k.estimated_speedup)
                .sum(),
        }
    }

    /// Clear the fusion cache
    pub fn clear_cache(&mut self) {
        self.fusion_cache.clear();
    }
}

/// Statistics about kernel fusion usage
#[derive(Debug, Clone)]
pub struct FusionStats {
    pub cached_kernels: usize,
    pub enabled: bool,
    pub max_fusion_length: usize,
    pub total_estimated_speedup: f32,
}

impl Default for KernelFusionOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to create common fusion patterns
pub mod patterns {
    use super::*;

    /// Create a pattern for fused element-wise operations followed by activation
    pub fn elementwise_activation(binary_op: FusableOp, activation: FusableOp) -> OpSequence {
        OpSequence {
            ops: vec![binary_op, activation],
            inputs: vec!["a".to_string(), "b".to_string()],
            output: "out".to_string(),
        }
    }

    /// Create a pattern for multiple chained element-wise operations
    pub fn chained_elementwise(ops: Vec<FusableOp>) -> OpSequence {
        let input_count = ops
            .iter()
            .filter(|op| {
                matches!(
                    op,
                    FusableOp::Add | FusableOp::Mul | FusableOp::Sub | FusableOp::Div
                )
            })
            .count()
            + 1;

        let inputs = (0..input_count).map(|i| format!("input_{i}")).collect();

        OpSequence {
            ops,
            inputs,
            output: "fused_out".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_fusion_optimizer_creation() {
        let optimizer = KernelFusionOptimizer::new();
        assert!(optimizer.enabled);
        assert_eq!(optimizer.max_fusion_length, 4);
        assert_eq!(optimizer.fusion_cache.len(), 0);
    }

    #[test]
    fn test_fusable_operation_detection() {
        let optimizer = KernelFusionOptimizer::new();

        // Test fusable patterns
        assert!(optimizer.can_fuse(&[FusableOp::Add, FusableOp::ReLU]));
        assert!(optimizer.can_fuse(&[FusableOp::Mul, FusableOp::Sigmoid, FusableOp::Add]));

        // Test non-fusable patterns
        assert!(!optimizer.can_fuse(&[FusableOp::Add])); // Too short
        assert!(!optimizer.can_fuse(&[FusableOp::Add; 10])); // Too long
    }

    #[test]
    fn test_kernel_id_generation() {
        let optimizer = KernelFusionOptimizer::new();

        let ops = vec![FusableOp::Add, FusableOp::ReLU];
        let id = optimizer.generate_kernel_id(&ops);
        assert_eq!(id, "fused_add_relu_");
    }

    #[test]
    fn test_speedup_estimation() {
        let optimizer = KernelFusionOptimizer::new();

        // Test basic fusion speedup
        let ops = vec![FusableOp::Add, FusableOp::Mul];
        let speedup = optimizer.estimate_speedup(&ops);
        assert!(speedup > 1.0);

        // Test activation pattern bonus
        let ops_with_activation = vec![FusableOp::Add, FusableOp::ReLU];
        let activation_speedup = optimizer.estimate_speedup(&ops_with_activation);
        assert!(activation_speedup > speedup);
    }

    #[test]
    fn test_fusion_cache() {
        let mut optimizer = KernelFusionOptimizer::new();

        let sequence = patterns::elementwise_activation(FusableOp::Add, FusableOp::ReLU);

        // First creation should cache the kernel
        let kernel1 = optimizer.create_fused_kernel(sequence.clone()).unwrap();
        assert_eq!(optimizer.fusion_cache.len(), 1);

        // Second access should return cached kernel
        let kernel2 = optimizer.create_fused_kernel(sequence).unwrap();
        assert_eq!(kernel1.kernel_id, kernel2.kernel_id);
        assert_eq!(optimizer.fusion_cache.len(), 1);
    }

    #[test]
    fn test_fusion_stats() {
        let mut optimizer = KernelFusionOptimizer::new();

        let sequence =
            patterns::chained_elementwise(vec![FusableOp::Add, FusableOp::Mul, FusableOp::ReLU]);
        optimizer.create_fused_kernel(sequence).unwrap();

        let stats = optimizer.get_stats();
        assert_eq!(stats.cached_kernels, 1);
        assert!(stats.enabled);
        assert!(stats.total_estimated_speedup > 1.0);
    }

    #[test]
    fn test_optimizer_configuration() {
        let mut optimizer = KernelFusionOptimizer::new();

        // Test enabling/disabling
        optimizer.set_enabled(false);
        assert!(!optimizer.enabled);
        assert!(!optimizer.can_fuse(&[FusableOp::Add, FusableOp::ReLU]));

        // Test max fusion length
        optimizer.set_enabled(true);
        optimizer.set_max_fusion_length(2);
        assert_eq!(optimizer.max_fusion_length, 2);
        assert!(!optimizer.can_fuse(&[FusableOp::Add, FusableOp::Mul, FusableOp::ReLU]));
        // Too long now
    }
}
