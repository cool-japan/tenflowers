//! Core optimization passes and trait definitions
//!
//! This module provides the fundamental optimization pass trait and basic passes
//! like constant folding, common subexpression elimination, and dead code elimination.

use crate::graph::{Graph, NodeId};
use crate::{Result, TensorError};
use std::collections::{HashMap, HashSet};

/// Helper function to get input node IDs for a given node
pub(crate) fn get_node_inputs(graph: &Graph, node_id: NodeId) -> Vec<NodeId> {
    if let Some(node) = graph.get_node(node_id) {
        node.inputs
            .iter()
            .filter_map(|&edge_id| graph.get_edge(edge_id).map(|edge| edge.from_node))
            .collect()
    } else {
        Vec::new()
    }
}

/// Helper function to get output node IDs for a given node
pub(crate) fn get_node_outputs(graph: &Graph, node_id: NodeId) -> Vec<NodeId> {
    if let Some(node) = graph.get_node(node_id) {
        node.outputs
            .iter()
            .filter_map(|&edge_id| graph.get_edge(edge_id).map(|edge| edge.to_node))
            .collect()
    } else {
        Vec::new()
    }
}

/// Graph optimization pass trait
pub trait OptimizationPass {
    /// Apply the optimization pass to the graph
    fn apply(&self, graph: &mut Graph) -> Result<bool>;

    /// Get the name of this optimization pass
    fn name(&self) -> &str;

    /// Check if this pass can be safely applied
    fn is_applicable(&self, graph: &Graph) -> bool;

    /// Get the pass priority (higher = run first)
    fn priority(&self) -> u32 {
        100
    }
}

/// Constant folding optimization pass
/// Evaluates constant expressions at compile time
pub struct ConstantFoldingPass;

impl OptimizationPass for ConstantFoldingPass {
    fn apply(&self, graph: &mut Graph) -> Result<bool> {
        let mut changed = false;
        let mut nodes_to_remove = Vec::new();
        let mut constant_nodes = HashSet::new();

        // Find all constant nodes (nodes with no inputs or only constant inputs)
        for node in graph.nodes() {
            if self.is_constant_node(graph, node.id) {
                constant_nodes.insert(node.id);
            }
        }

        // For each constant node that has arithmetic operations, try to fold
        for &node_id in &constant_nodes {
            if let Some(node) = graph.get_node(node_id) {
                if let crate::graph::NodeType::Operation(op_name) = &node.op_type {
                    match op_name.as_str() {
                        "Add" | "Mul" | "Sub" | "Div" => {
                            if self.can_fold_binary_op(graph, node_id) {
                                // Replace the operation with its constant result
                                // In a real implementation, we'd evaluate the operation
                                // For now, just mark it as foldable
                                nodes_to_remove.push(node_id);
                                changed = true;
                            }
                        }
                        "MatMul" => {
                            if self.can_fold_matmul(graph, node_id) {
                                nodes_to_remove.push(node_id);
                                changed = true;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Evaluate and replace folded nodes with constants
        for node_id in nodes_to_remove {
            if let Some(result_tensor) = self.evaluate_constant_operation(graph, node_id)? {
                graph.replace_with_constant(node_id, result_tensor)?;
            }
        }

        Ok(changed)
    }

    fn name(&self) -> &str {
        "ConstantFolding"
    }

    fn is_applicable(&self, graph: &Graph) -> bool {
        // Always applicable if there are nodes
        graph.node_count() > 0
    }

    fn priority(&self) -> u32 {
        200 // High priority - run early
    }
}

impl Default for ConstantFoldingPass {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstantFoldingPass {
    pub fn new() -> Self {
        Self
    }

    #[allow(clippy::only_used_in_recursion)]
    fn is_constant_node(&self, graph: &Graph, node_id: NodeId) -> bool {
        if let Some(node) = graph.get_node(node_id) {
            // Check if node is a constant or all inputs are constants
            if matches!(node.op_type, crate::graph::NodeType::Constant) {
                return true;
            }

            let inputs = get_node_inputs(graph, node_id);
            for input_id in inputs {
                if !self.is_constant_node(graph, input_id) {
                    return false;
                }
            }
            !node.inputs.is_empty()
        } else {
            false
        }
    }

    fn can_fold_binary_op(&self, graph: &Graph, node_id: NodeId) -> bool {
        let inputs = get_node_inputs(graph, node_id);
        inputs.len() == 2 && inputs.iter().all(|&id| self.is_constant_node(graph, id))
    }

    fn can_fold_matmul(&self, graph: &Graph, node_id: NodeId) -> bool {
        let inputs = get_node_inputs(graph, node_id);
        inputs.len() == 2 && inputs.iter().all(|&id| self.is_constant_node(graph, id))
    }

    fn evaluate_constant_operation(
        &self,
        graph: &Graph,
        node_id: NodeId,
    ) -> Result<Option<crate::tensor::Tensor<f32>>> {
        use crate::ops::{binary, matmul};

        let node = graph.get_node(node_id).ok_or_else(|| {
            TensorError::invalid_argument(format!("Node {node_id} does not exist"))
        })?;

        if let crate::graph::NodeType::Operation(op_name) = &node.op_type {
            let input_node_ids = get_node_inputs(graph, node_id);

            // Get input tensors
            let input_tensors: std::result::Result<Vec<_>, crate::error::TensorError> =
                input_node_ids
                    .iter()
                    .map(|&input_id| self.get_constant_tensor(graph, input_id))
                    .collect();

            let inputs = input_tensors?;

            match op_name.as_str() {
                "Add" if inputs.len() == 2 => {
                    let result = binary::add(&inputs[0], &inputs[1])?;
                    Ok(Some(result))
                }
                "Sub" if inputs.len() == 2 => {
                    let result = binary::sub(&inputs[0], &inputs[1])?;
                    Ok(Some(result))
                }
                "Mul" if inputs.len() == 2 => {
                    let result = binary::mul(&inputs[0], &inputs[1])?;
                    Ok(Some(result))
                }
                "Div" if inputs.len() == 2 => {
                    let result = binary::div(&inputs[0], &inputs[1])?;
                    Ok(Some(result))
                }
                "MatMul" if inputs.len() == 2 => {
                    let result = matmul::matmul(&inputs[0], &inputs[1])?;
                    Ok(Some(result))
                }
                _ => Ok(None), // Operation not supported for constant folding
            }
        } else {
            Ok(None)
        }
    }

    fn get_constant_tensor(
        &self,
        graph: &Graph,
        node_id: NodeId,
    ) -> std::result::Result<crate::tensor::Tensor<f32>, crate::error::TensorError> {
        let node = graph.get_node(node_id).ok_or_else(|| {
            TensorError::invalid_argument(format!("Node {node_id} does not exist"))
        })?;

        if let crate::graph::NodeType::Constant = &node.op_type {
            if let Some(crate::graph::AttributeValue::Tensor(tensor)) = node.attributes.get("value")
            {
                Ok(tensor.clone())
            } else {
                Err(TensorError::invalid_argument(
                    "Constant node missing tensor value".to_string(),
                ))
            }
        } else {
            Err(TensorError::invalid_argument(format!(
                "Node {node_id} is not a constant"
            )))
        }
    }
}

/// Common Subexpression Elimination pass
/// Removes duplicate computations
pub struct CSEPass;

impl OptimizationPass for CSEPass {
    fn apply(&self, graph: &mut Graph) -> Result<bool> {
        let mut changed = false;
        let mut expression_map: HashMap<String, NodeId> = HashMap::new();
        let mut nodes_to_redirect = Vec::new();

        // Find duplicate expressions
        for node in graph.nodes() {
            let expr_key = self.compute_expression_key(graph, node.id);

            if let Some(&existing_node_id) = expression_map.get(&expr_key) {
                // Found duplicate - redirect this node's outputs to the existing one
                nodes_to_redirect.push((node.id, existing_node_id));
                changed = true;
            } else {
                expression_map.insert(expr_key, node.id);
            }
        }

        // Redirect duplicate nodes
        for (duplicate_node, canonical_node) in nodes_to_redirect {
            // Redirect all outputs from duplicate to canonical node
            graph.redirect_node_outputs(duplicate_node, canonical_node)?;
            // Remove the duplicate node
            graph.remove_node(duplicate_node)?;
        }

        Ok(changed)
    }

    fn name(&self) -> &str {
        "CommonSubexpressionElimination"
    }

    fn is_applicable(&self, graph: &Graph) -> bool {
        graph.node_count() > 1
    }

    fn priority(&self) -> u32 {
        150 // Medium-high priority
    }
}

impl Default for CSEPass {
    fn default() -> Self {
        Self::new()
    }
}

impl CSEPass {
    pub fn new() -> Self {
        Self
    }

    #[allow(clippy::only_used_in_recursion)]
    fn compute_expression_key(&self, graph: &Graph, node_id: NodeId) -> String {
        if let Some(node) = graph.get_node(node_id) {
            let inputs = get_node_inputs(graph, node_id);
            let input_keys: Vec<String> = inputs
                .iter()
                .map(|&id| self.compute_expression_key(graph, id))
                .collect();

            format!("{:?}({})", node.op_type, input_keys.join(","))
        } else {
            format!("node_{node_id}")
        }
    }
}

/// Dead code elimination pass
/// Removes nodes that don't contribute to any output
pub struct DeadCodeEliminationPass;

impl OptimizationPass for DeadCodeEliminationPass {
    fn apply(&self, graph: &mut Graph) -> Result<bool> {
        let mut changed = false;
        let mut reachable = HashSet::new();

        // Mark all nodes reachable from outputs
        for node in graph.nodes() {
            if self.is_output_node(graph, node.id) {
                self.mark_reachable(graph, node.id, &mut reachable);
            }
        }

        // Remove unreachable nodes
        let mut nodes_to_remove = Vec::new();
        for node in graph.nodes() {
            if !reachable.contains(&node.id) {
                nodes_to_remove.push(node.id);
                changed = true;
            }
        }

        for node_id in nodes_to_remove {
            graph.remove_node(node_id)?;
        }

        Ok(changed)
    }

    fn name(&self) -> &str {
        "DeadCodeElimination"
    }

    fn is_applicable(&self, graph: &Graph) -> bool {
        graph.node_count() > 0
    }

    fn priority(&self) -> u32 {
        50 // Low priority - run last
    }
}

impl Default for DeadCodeEliminationPass {
    fn default() -> Self {
        Self::new()
    }
}

impl DeadCodeEliminationPass {
    pub fn new() -> Self {
        Self
    }

    fn is_output_node(&self, graph: &Graph, node_id: NodeId) -> bool {
        // A node is an output if it has no successors or is explicitly marked as output
        let outputs = get_node_outputs(graph, node_id);
        outputs.is_empty() || self.is_marked_as_output(graph, node_id)
    }

    fn is_marked_as_output(&self, _graph: &Graph, _node_id: NodeId) -> bool {
        // In a real implementation, check if node is marked as a graph output
        false
    }

    #[allow(clippy::only_used_in_recursion)]
    fn mark_reachable(&self, graph: &Graph, node_id: NodeId, reachable: &mut HashSet<NodeId>) {
        if reachable.contains(&node_id) {
            return;
        }

        reachable.insert(node_id);

        // Mark all inputs as reachable
        for input_id in get_node_inputs(graph, node_id) {
            self.mark_reachable(graph, input_id, reachable);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_folding_pass() {
        let pass = ConstantFoldingPass::new();
        assert_eq!(pass.name(), "ConstantFolding");
        assert_eq!(pass.priority(), 200);

        let graph = Graph::new();
        assert!(!pass.is_applicable(&graph));
    }

    #[test]
    fn test_cse_pass() {
        let pass = CSEPass::new();
        assert_eq!(pass.name(), "CommonSubexpressionElimination");
        assert_eq!(pass.priority(), 150);
    }

    #[test]
    fn test_dead_code_elimination_pass() {
        let pass = DeadCodeEliminationPass::new();
        assert_eq!(pass.name(), "DeadCodeElimination");
        assert_eq!(pass.priority(), 50);
    }
}
