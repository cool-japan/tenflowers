//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::graph::{Graph, NodeId};
use crate::{Result, TensorError};

use super::pass_support::{get_node_inputs, OptimizationPass};

/// Constant folding optimization pass
/// Evaluates constant expressions at compile time
pub struct ConstantFoldingPass;
impl ConstantFoldingPass {
    pub fn new() -> Self {
        Self
    }
    /// True only when `node_id` is itself a literal `Constant` node (not
    /// merely "part of a constant subtree" — see the comment in `apply`).
    pub(super) fn is_literal_constant(&self, graph: &Graph, node_id: NodeId) -> bool {
        graph
            .get_node(node_id)
            .is_some_and(|node| matches!(node.op_type, crate::graph::NodeType::Constant))
    }
    pub(super) fn can_fold_binary_op(&self, graph: &Graph, node_id: NodeId) -> bool {
        let inputs = get_node_inputs(graph, node_id);
        inputs.len() == 2 && inputs.iter().all(|&id| self.is_literal_constant(graph, id))
    }
    pub(super) fn can_fold_matmul(&self, graph: &Graph, node_id: NodeId) -> bool {
        let inputs = get_node_inputs(graph, node_id);
        inputs.len() == 2 && inputs.iter().all(|&id| self.is_literal_constant(graph, id))
    }
    pub(super) fn evaluate_constant_operation(
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
                _ => Ok(None),
            }
        } else {
            Ok(None)
        }
    }
    pub(super) fn get_constant_tensor(
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
impl OptimizationPass for ConstantFoldingPass {
    fn apply(&self, graph: &mut Graph) -> Result<bool> {
        let mut changed = false;
        let mut nodes_to_fold = Vec::new();
        for node in graph.nodes() {
            if let crate::graph::NodeType::Operation(op_name) = &node.op_type {
                let foldable = match op_name.as_str() {
                    "Add" | "Mul" | "Sub" | "Div" => self.can_fold_binary_op(graph, node.id),
                    "MatMul" => self.can_fold_matmul(graph, node.id),
                    _ => false,
                };
                if foldable {
                    nodes_to_fold.push(node.id);
                }
            }
        }
        for node_id in nodes_to_fold {
            if let Some(result_tensor) = self.evaluate_constant_operation(graph, node_id)? {
                graph.replace_with_constant(node_id, result_tensor)?;
                changed = true;
            }
        }
        Ok(changed)
    }
    fn name(&self) -> &str {
        "ConstantFolding"
    }
    fn is_applicable(&self, graph: &Graph) -> bool {
        graph.node_count() > 0
    }
    fn priority(&self) -> u32 {
        200
    }
}
impl Default for ConstantFoldingPass {
    fn default() -> Self {
        Self::new()
    }
}
