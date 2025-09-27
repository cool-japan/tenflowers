//! GradientTape implementation for automatic differentiation
//!
//! This module provides the main GradientTape implementation that records operations
//! and computes gradients through the computation graph.

use num_traits::{One, Zero};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tenflowers_core::{Result, Tensor, TensorError};

use super::helpers::{compare_gradients, compute_numerical_gradient};
use super::structures::{GradientTape, GradientTapeInner};
use super::{Operation, TensorId, TrackedTensor};

// TODO: Move complete GradientTape implementation from original tape.rs
// This includes:
// - GradientTape constructor and basic methods (lines 435-500, ~65 lines)
// - Record operation functionality (lines 501-700, ~200 lines)
// - Gradient computation core (lines 701-1500, ~800 lines)
// - Operation-specific gradient implementations (lines 1501-2583, ~1083 lines)
// - Helper and utility methods (lines 2584-2767, ~184 lines)
// Total: ~2332 lines of core gradient tape functionality

impl GradientTape {
    /// Create a new gradient tape
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(GradientTapeInner {
                nodes: Vec::new(),
                tensor_values: HashMap::new(),
                next_id: 0,
                is_recording: true,
            })),
        }
    }

    /// Create a gradient tape from an existing inner reference
    pub(crate) fn from_inner(inner: Arc<Mutex<GradientTapeInner>>) -> Self {
        Self { inner }
    }

    /// Start recording operations
    pub fn start_recording(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.is_recording = true;
        }
    }

    /// Stop recording operations
    pub fn stop_recording(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.is_recording = false;
        }
    }

    /// Check if the tape is currently recording
    pub fn is_recording(&self) -> bool {
        if let Ok(inner) = self.inner.lock() {
            inner.is_recording
        } else {
            false
        }
    }

    /// Clear the tape by removing all recorded operations and tensor values
    pub fn clear(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.nodes.clear();
            inner.tensor_values.clear();
            inner.next_id = 0;
        }
    }

    /// Record an operation in the tape
    pub fn record_op<T>(&self, operation: Operation, result: Tensor<T>) -> TrackedTensor<T>
    where
        T: Clone + Send + Sync + 'static,
    {
        let mut inner = self.inner.lock().unwrap();
        inner.record_op(operation, result, &self.inner)
    }

    /// Watch a tensor for gradient computation
    pub fn watch<T>(&self, tensor: Tensor<T>) -> TrackedTensor<T>
    where
        T: Clone + Send + Sync + 'static,
    {
        let mut inner = self.inner.lock().unwrap();
        let tensor_id = inner.next_id;
        inner.next_id += 1;

        // Store tensor value for gradient computation
        inner
            .tensor_values
            .insert(tensor_id, Box::new(tensor.clone()));

        TrackedTensor {
            tensor,
            id: tensor_id,
            tape: Arc::downgrade(&self.inner),
        }
    }

    /// Get the number of operations recorded in the tape
    pub fn len(&self) -> usize {
        if let Ok(inner) = self.inner.lock() {
            inner.nodes.len()
        } else {
            0
        }
    }

    /// Check if the tape is empty
    pub fn is_empty(&self) -> bool {
        if let Ok(inner) = self.inner.lock() {
            inner.nodes.is_empty() && inner.tensor_values.is_empty()
        } else {
            true
        }
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> usize {
        if let Ok(inner) = self.inner.lock() {
            let mut total_memory = 0;

            // Memory from nodes (operations)
            total_memory += inner.nodes.len() * std::mem::size_of::<super::structures::TapeNode>();

            // Memory from tensor values storage (rough estimate)
            total_memory += inner.tensor_values.len() * 1024; // Assume ~1KB per tensor on average

            // Memory from the HashMap overhead
            total_memory += inner.tensor_values.capacity()
                * std::mem::size_of::<(TensorId, Box<dyn std::any::Any + Send + Sync>)>();

            total_memory
        } else {
            0
        }
    }

    /// Get the number of nodes in the computation graph
    pub fn node_count(&self) -> usize {
        if let Ok(inner) = self.inner.lock() {
            inner.nodes.len()
        } else {
            0
        }
    }

    /// Get estimated memory usage in bytes
    pub fn memory_usage_estimate(&self) -> usize {
        // Placeholder implementation - estimate based on node count
        self.node_count() * 1024 // Rough estimate of 1KB per node
    }

    /// Access inner tape with mutable reference
    pub(crate) fn with_inner_mut<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut GradientTapeInner) -> R,
    {
        let mut inner = self.inner.lock().unwrap();
        f(&mut inner)
    }

    /// Get a tape node by its ID
    pub fn get_node(&self, id: TensorId) -> Option<super::structures::TapeNode> {
        if let Ok(inner) = self.inner.lock() {
            inner.nodes.iter().find(|node| node.id == id).cloned()
        } else {
            None
        }
    }

    /// Get all nodes in the tape
    pub fn get_all_nodes(&self) -> Vec<super::structures::TapeNode> {
        if let Ok(inner) = self.inner.lock() {
            inner.nodes.clone()
        } else {
            Vec::new()
        }
    }

    /// Get parent IDs for a given node
    pub fn get_parent_ids(&self, id: TensorId) -> Vec<TensorId> {
        if let Ok(inner) = self.inner.lock() {
            if let Some(node) = inner.nodes.iter().find(|node| node.id == id) {
                node.parents.clone()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    }

    /// Optimize the tape by removing unused operations
    pub fn optimize(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            // Mark all nodes that are reachable from the final outputs
            let mut reachable_nodes = std::collections::HashSet::new();
            let mut stack = Vec::new();

            // Start from all leaf nodes (nodes with no children in subsequent operations)
            for (i, node) in inner.nodes.iter().enumerate() {
                let is_leaf = !inner
                    .nodes
                    .iter()
                    .skip(i + 1)
                    .any(|later_node| self.node_depends_on(&later_node.operation, node.id));
                if is_leaf {
                    stack.push(i);
                }
            }

            // Mark all reachable nodes using DFS
            while let Some(node_idx) = stack.pop() {
                if reachable_nodes.insert(node_idx) {
                    let node = &inner.nodes[node_idx];

                    // Add parent nodes to the stack for processing
                    for parent_id in &node.parents {
                        if let Some(parent_idx) =
                            inner.nodes.iter().position(|n| n.id == *parent_id)
                        {
                            stack.push(parent_idx);
                        }
                    }
                }
            }

            // Remove unreachable nodes
            let original_count = inner.nodes.len();
            let mut new_nodes = Vec::new();
            let mut removed_tensor_ids = std::collections::HashSet::new();

            for (i, node) in inner.nodes.iter().enumerate() {
                if reachable_nodes.contains(&i) {
                    new_nodes.push(node.clone());
                } else {
                    removed_tensor_ids.insert(node.id);
                }
            }

            inner.nodes = new_nodes;

            // Remove tensor values for removed nodes
            inner
                .tensor_values
                .retain(|id, _| !removed_tensor_ids.contains(id));

            // Log optimization results (if logging is available)
            let removed_count = original_count - inner.nodes.len();
            if removed_count > 0 {
                eprintln!(
                    "Tape optimization: removed {} unused operations",
                    removed_count
                );
            }
        }
    }

    /// Export the computation graph for visualization
    pub fn export_graph(&self) -> Result<String> {
        // Export the computation graph in DOT format for visualization
        let inner = self.inner.lock().unwrap();
        let mut dot_graph = String::from("digraph ComputationGraph {\n");
        dot_graph.push_str("  rankdir=TB;\n");
        dot_graph.push_str("  node [shape=ellipse];\n");

        // Export all nodes in the computation graph
        for (i, node) in inner.nodes.iter().enumerate() {
            let node_label = match &node.operation {
                Operation::Add { .. } => "Add",
                Operation::Sub { .. } => "Sub",
                Operation::Mul { .. } => "Mul",
                Operation::Div { .. } => "Div",
                Operation::MatMul { .. } => "MatMul",
                Operation::Relu { .. } => "ReLU",
                Operation::Sigmoid { .. } => "Sigmoid",
                Operation::Tanh { .. } => "Tanh",
                Operation::Gelu { .. } => "GELU",
                Operation::Swish { .. } => "Swish",
                Operation::Mish { .. } => "Mish",
                Operation::LeakyRelu { .. } => "LeakyReLU",
                Operation::Elu { .. } => "ELU",
                Operation::Prelu { .. } => "PReLU",
                Operation::Softmax { .. } => "Softmax",
                Operation::Pow { .. } => "Pow",
                Operation::Sum { .. } => "Sum",
                Operation::Mean { .. } => "Mean",
                Operation::Max { .. } => "Max",
                Operation::Min { .. } => "Min",
                Operation::Var { .. } => "Var",
                Operation::Std { .. } => "Std",
                Operation::Neg { .. } => "Neg",
                Operation::Identity { .. } => "Identity",
                Operation::Constant => "Constant",
                Operation::Reshape { .. } => "Reshape",
                Operation::Transpose { .. } => "Transpose",
                Operation::Squeeze { .. } => "Squeeze",
                Operation::Unsqueeze { .. } => "Unsqueeze",
                Operation::Slice { .. } => "Slice",
                Operation::Concat { .. } => "Concat",
                Operation::Stack { .. } => "Stack",
                _ => "Operation",
            };

            dot_graph.push_str(&format!("  node_{} [label=\"{}\"];\n", i, node_label));

            // Add edges from input tensors to this operation
            let input_ids = match &node.operation {
                Operation::Add { lhs, rhs }
                | Operation::Sub { lhs, rhs }
                | Operation::Mul { lhs, rhs }
                | Operation::Div { lhs, rhs }
                | Operation::MatMul { lhs, rhs }
                | Operation::Pow { lhs, rhs } => vec![*lhs, *rhs],
                Operation::Relu { input }
                | Operation::Sigmoid { input }
                | Operation::Tanh { input }
                | Operation::Gelu { input }
                | Operation::Swish { input }
                | Operation::Mish { input }
                | Operation::Neg { input }
                | Operation::Identity { input } => vec![*input],
                Operation::LeakyRelu { input, .. } | Operation::Elu { input, .. } => vec![*input],
                Operation::Prelu { input, alpha } => vec![*input, *alpha],
                Operation::Softmax { input, .. } => vec![*input],
                Operation::Sum { input, .. }
                | Operation::Mean { input, .. }
                | Operation::Max { input, .. }
                | Operation::Min { input, .. }
                | Operation::Var { input, .. }
                | Operation::Std { input, .. } => vec![*input],
                Operation::Reshape { input, .. }
                | Operation::Transpose { input, .. }
                | Operation::Squeeze { input, .. }
                | Operation::Unsqueeze { input, .. }
                | Operation::Slice { input, .. } => vec![*input],
                Operation::Concat { inputs, .. } | Operation::Stack { inputs, .. } => {
                    inputs.clone()
                }
                Operation::Constant => vec![],
                _ => vec![],
            };

            for input_id in input_ids {
                // Find the node that produces this input
                if let Some(input_node_idx) = inner.nodes.iter().position(|n| n.id == input_id) {
                    dot_graph.push_str(&format!("  node_{} -> node_{};\n", input_node_idx, i));
                }
            }
        }

        dot_graph.push_str("}\n");
        Ok(dot_graph)
    }

    /// Numerical gradient checking for validating automatic differentiation
    ///
    /// This method compares analytical gradients computed by the tape with numerical gradients
    /// computed using finite differences. It's essential for debugging and validating
    /// gradient implementations.
    ///
    /// # Arguments
    /// * `function` - Function that takes TrackedTensor inputs and returns a TrackedTensor output
    /// * `inputs` - Input tensors at which to check gradients
    /// * `epsilon` - Step size for finite differences (default: 1e-5)
    /// * `relative_tolerance` - Relative tolerance for gradient comparison
    /// * `absolute_tolerance` - Absolute tolerance for gradient comparison
    ///
    /// # Returns
    /// `Ok(())` if gradients match within tolerance, `Err` otherwise
    pub fn numerical_gradient_check<T, F>(
        function: F,
        inputs: &[Tensor<T>],
        epsilon: T,
        relative_tolerance: T,
        absolute_tolerance: T,
    ) -> Result<()>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + PartialOrd
            + num_traits::Float
            + num_traits::FromPrimitive
            + bytemuck::Pod
            + bytemuck::Zeroable,
        F: Fn(&[&TrackedTensor<T>]) -> Result<TrackedTensor<T>>,
    {
        if inputs.is_empty() {
            return Err(TensorError::invalid_argument(
                "At least one input tensor is required for gradient checking".to_string(),
            ));
        }

        // For each input, compute analytical gradient using a fresh tape
        for (i, _input) in inputs.iter().enumerate() {
            // Create fresh tape for this gradient computation
            let tape = GradientTape::new();
            let tracked_inputs: Vec<TrackedTensor<T>> =
                inputs.iter().map(|inp| tape.watch(inp.clone())).collect();

            // Call function with tracked tensors for gradient computation
            let tracked_input_refs: Vec<&TrackedTensor<T>> = tracked_inputs.iter().collect();
            let output_tracked = function(&tracked_input_refs)?;

            // Compute analytical gradient for current input
            let analytical_gradients = tape.gradient(
                std::slice::from_ref(&output_tracked),
                std::slice::from_ref(&tracked_inputs[i]),
            )?;

            if let Some(analytical_grad) = &analytical_gradients[0] {
                // Compute numerical gradient using a wrapper that converts the TrackedTensor function
                let wrapper_fn = |tensor_inputs: &[Tensor<T>]| -> Result<Tensor<T>> {
                    // Create a temporary tape for the numerical computation
                    let temp_tape = GradientTape::new();
                    let temp_tracked: Vec<TrackedTensor<T>> = tensor_inputs
                        .iter()
                        .map(|t| temp_tape.watch(t.clone()))
                        .collect();
                    let temp_refs: Vec<&TrackedTensor<T>> = temp_tracked.iter().collect();
                    let result = function(&temp_refs)?;
                    Ok(result.tensor.clone())
                };
                let numerical_grad = compute_numerical_gradient(&wrapper_fn, inputs, i, epsilon)?;

                // Compare gradients
                compare_gradients(
                    analytical_grad,
                    &numerical_grad,
                    relative_tolerance,
                    absolute_tolerance,
                    i,
                )?;
            } else {
                return Err(TensorError::other(format!(
                    "No analytical gradient computed for input {}",
                    i
                )));
            }
        }

        Ok(())
    }

    /// Helper method to check if an operation depends on a given tensor ID
    fn node_depends_on(&self, operation: &Operation, tensor_id: TensorId) -> bool {
        match operation {
            Operation::Add { lhs, rhs }
            | Operation::Sub { lhs, rhs }
            | Operation::Mul { lhs, rhs }
            | Operation::Div { lhs, rhs }
            | Operation::MatMul { lhs, rhs }
            | Operation::Pow { lhs, rhs } => *lhs == tensor_id || *rhs == tensor_id,

            Operation::Relu { input }
            | Operation::Sigmoid { input }
            | Operation::Tanh { input }
            | Operation::Gelu { input }
            | Operation::Swish { input }
            | Operation::Mish { input }
            | Operation::Neg { input }
            | Operation::Identity { input }
            | Operation::Sum { input, .. }
            | Operation::Mean { input, .. }
            | Operation::Max { input, .. }
            | Operation::Min { input, .. }
            | Operation::Var { input, .. }
            | Operation::Std { input, .. }
            | Operation::Reshape { input, .. }
            | Operation::Transpose { input, .. }
            | Operation::Squeeze { input, .. }
            | Operation::Unsqueeze { input, .. }
            | Operation::Slice { input, .. } => *input == tensor_id,

            Operation::LeakyRelu { input, .. }
            | Operation::Elu { input, .. }
            | Operation::Softmax { input, .. } => *input == tensor_id,

            Operation::Prelu { input, alpha } => *input == tensor_id || *alpha == tensor_id,

            Operation::Concat { inputs, .. } | Operation::Stack { inputs, .. } => {
                inputs.contains(&tensor_id)
            }

            Operation::Split { input, .. } => *input == tensor_id,

            Operation::BatchNorm {
                input,
                gamma,
                beta,
                running_mean,
                running_var,
                ..
            } => {
                *input == tensor_id
                    || *gamma == tensor_id
                    || *beta == tensor_id
                    || *running_mean == tensor_id
                    || *running_var == tensor_id
            }

            Operation::LayerNorm {
                input, gamma, beta, ..
            }
            | Operation::GroupNorm {
                input, gamma, beta, ..
            }
            | Operation::InstanceNorm {
                input, gamma, beta, ..
            } => *input == tensor_id || *gamma == tensor_id || *beta == tensor_id,

            Operation::Conv2D {
                input,
                weight,
                bias,
                ..
            }
            | Operation::Conv3D {
                input,
                weight,
                bias,
                ..
            }
            | Operation::ConvTranspose2D {
                input,
                weight,
                bias,
                ..
            }
            | Operation::DepthwiseConv2D {
                input,
                weight,
                bias,
                ..
            }
            | Operation::GroupedConv2D {
                input,
                weight,
                bias,
                ..
            } => {
                *input == tensor_id
                    || *weight == tensor_id
                    || bias.map(|b| b == tensor_id).unwrap_or(false)
            }

            Operation::MaxPool2D { input, .. }
            | Operation::AvgPool2D { input, .. }
            | Operation::GlobalAvgPool2D { input }
            | Operation::GlobalMaxPool2D { input }
            | Operation::AdaptiveAvgPool2D { input, .. }
            | Operation::AdaptiveMaxPool2D { input, .. } => *input == tensor_id,

            Operation::BooleanMask { input, mask } => *input == tensor_id || *mask == tensor_id,

            Operation::Where { condition, x, z } => {
                *condition == tensor_id || *x == tensor_id || *z == tensor_id
            }

            Operation::IntegerArrayIndexing { input, indices, .. } => {
                *input == tensor_id || *indices == tensor_id
            }

            Operation::Fft { input, .. }
            | Operation::Ifft { input, .. }
            | Operation::Rfft { input, .. }
            | Operation::Fft2 { input, .. }
            | Operation::Ifft2 { input, .. }
            | Operation::Fft3 { input, .. }
            | Operation::Ifft3 { input, .. } => *input == tensor_id,

            Operation::Eig { input }
            | Operation::Svd { input }
            | Operation::Inv { input }
            | Operation::Det { input }
            | Operation::Cholesky { input }
            | Operation::Lu { input }
            | Operation::Pinv { input }
            | Operation::Gamma { input }
            | Operation::Lgamma { input }
            | Operation::Digamma { input }
            | Operation::Erf { input }
            | Operation::Erfc { input }
            | Operation::BesselJ0 { input }
            | Operation::BesselJ1 { input }
            | Operation::StopGradient { input } => *input == tensor_id,

            Operation::Beta { a, b } => *a == tensor_id || *b == tensor_id,

            Operation::Einsum { inputs, .. } => inputs.contains(&tensor_id),

            Operation::FusedAddReLU { lhs, rhs } => *lhs == tensor_id || *rhs == tensor_id,

            Operation::FusedDense {
                input,
                weight,
                bias,
            } => {
                *input == tensor_id
                    || *weight == tensor_id
                    || bias.map(|b| b == tensor_id).unwrap_or(false)
            }

            Operation::FusedConvBatchNorm {
                input,
                weight,
                bias,
                gamma,
                beta,
                running_mean,
                running_var,
                ..
            } => {
                *input == tensor_id
                    || *weight == tensor_id
                    || bias.map(|b| b == tensor_id).unwrap_or(false)
                    || *gamma == tensor_id
                    || *beta == tensor_id
                    || *running_mean == tensor_id
                    || *running_var == tensor_id
            }

            Operation::Constant => false,
        }
    }
}
