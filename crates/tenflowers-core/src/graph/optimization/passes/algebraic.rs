//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::graph::{Graph, NodeId};
use crate::Result;
use std::cell::RefCell;
use std::collections::HashSet;

use super::pass_support::{
    describe_input_edge, get_node_inputs, insert_scalar_constant, rewrite_operation_node,
    OptimizationPass,
};

/// Algebraic simplification pass
/// Simplifies algebraic expressions using mathematical identities
pub struct AlgebraicSimplificationPass {
    pub(super) outputs: RefCell<HashSet<NodeId>>,
}
impl AlgebraicSimplificationPass {
    pub fn new() -> Self {
        Self {
            outputs: RefCell::new(HashSet::new()),
        }
    }
    pub(super) fn is_protected(&self, node_id: NodeId) -> bool {
        self.outputs.borrow().contains(&node_id)
    }
    pub(super) fn find_simplification(
        &self,
        graph: &Graph,
        node_id: NodeId,
    ) -> Option<SimplificationType> {
        let node = graph.get_node(node_id)?;
        if let crate::graph::NodeType::Operation(op_name) = &node.op_type {
            let inputs = get_node_inputs(graph, node_id);
            match op_name.as_str() {
                "Add" if inputs.len() == 2 => {
                    if self.is_zero_constant(graph, inputs[1]) {
                        return Some(SimplificationType::ReplaceWithInput(0));
                    }
                    if self.is_zero_constant(graph, inputs[0]) {
                        return Some(SimplificationType::ReplaceWithInput(1));
                    }
                    if inputs[0] == inputs[1] {
                        return Some(SimplificationType::ConvertToMultiply(2.0));
                    }
                }
                "Mul" if inputs.len() == 2 => {
                    if self.is_one_constant(graph, inputs[1]) {
                        return Some(SimplificationType::ReplaceWithInput(0));
                    }
                    if self.is_one_constant(graph, inputs[0]) {
                        return Some(SimplificationType::ReplaceWithInput(1));
                    }
                    if self.is_zero_constant(graph, inputs[0])
                        || self.is_zero_constant(graph, inputs[1])
                    {
                        return Some(SimplificationType::ReplaceWithConstant(0.0));
                    }
                }
                "Sub" if inputs.len() == 2 => {
                    if self.is_zero_constant(graph, inputs[1]) {
                        return Some(SimplificationType::ReplaceWithInput(0));
                    }
                    if inputs[0] == inputs[1] {
                        return Some(SimplificationType::ReplaceWithConstant(0.0));
                    }
                }
                "Div" if inputs.len() == 2 => {
                    if self.is_one_constant(graph, inputs[1]) {
                        return Some(SimplificationType::ReplaceWithInput(0));
                    }
                    if inputs[0] == inputs[1] {
                        return Some(SimplificationType::ReplaceWithConstant(1.0));
                    }
                }
                "Pow" if inputs.len() == 2 => {
                    if self.is_one_constant(graph, inputs[1]) {
                        return Some(SimplificationType::ReplaceWithInput(0));
                    }
                    if self.is_zero_constant(graph, inputs[1]) {
                        return Some(SimplificationType::ReplaceWithConstant(1.0));
                    }
                }
                _ => {}
            }
        }
        None
    }
    pub(super) fn is_zero_constant(&self, graph: &Graph, node_id: NodeId) -> bool {
        self.is_scalar_constant(graph, node_id, 0.0)
    }
    pub(super) fn is_one_constant(&self, graph: &Graph, node_id: NodeId) -> bool {
        self.is_scalar_constant(graph, node_id, 1.0)
    }
    pub(super) fn is_scalar_constant(&self, graph: &Graph, node_id: NodeId, value: f32) -> bool {
        if let Some(node) = graph.get_node(node_id) {
            if let crate::graph::NodeType::Constant = &node.op_type {
                if let Some(crate::graph::AttributeValue::Tensor(tensor)) =
                    node.attributes.get("value")
                {
                    if tensor.shape().size() == 1 {
                        if let Some(slice) = tensor.as_slice() {
                            return (slice[0] - value).abs() < 1e-6;
                        }
                    }
                }
            }
        }
        false
    }
    pub(super) fn apply_simplification(
        &self,
        graph: &mut Graph,
        node_id: NodeId,
        simplification: SimplificationType,
    ) -> Result<bool> {
        match simplification {
            SimplificationType::ReplaceWithInput(input_idx) => {
                if self.is_protected(node_id) {
                    return Ok(false);
                }
                let inputs = get_node_inputs(graph, node_id);
                if input_idx < inputs.len() {
                    graph.redirect_node_outputs(node_id, inputs[input_idx])?;
                    graph.remove_node(node_id)?;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            SimplificationType::ReplaceWithConstant(value) => {
                let const_tensor = crate::tensor::Tensor::from_scalar(value);
                graph.replace_with_constant(node_id, const_tensor)?;
                Ok(true)
            }
            SimplificationType::ConvertToMultiply(scale) => {
                let Some((x_node, x_output, dtype, shape)) = describe_input_edge(graph, node_id, 0)
                else {
                    return Ok(false);
                };
                let device = graph
                    .get_node(node_id)
                    .map(|node| node.device)
                    .unwrap_or(crate::device::Device::Cpu);
                let const_id = insert_scalar_constant(graph, scale, device, "double")?;
                rewrite_operation_node(
                    graph,
                    node_id,
                    "Mul",
                    &[
                        (x_node, x_output, dtype, shape),
                        (
                            const_id,
                            0,
                            crate::dtype::DType::Float32,
                            crate::shape::Shape::new(vec![]),
                        ),
                    ],
                )?;
                Ok(true)
            }
        }
    }
}
impl OptimizationPass for AlgebraicSimplificationPass {
    fn apply(&self, graph: &mut Graph) -> Result<bool> {
        let mut changed = false;
        let mut nodes_to_simplify = Vec::new();
        for node in graph.nodes() {
            if let Some(simplification) = self.find_simplification(graph, node.id) {
                nodes_to_simplify.push((node.id, simplification));
            }
        }
        for (node_id, simplification) in nodes_to_simplify {
            if self.apply_simplification(graph, node_id, simplification)? {
                changed = true;
            }
        }
        Ok(changed)
    }
    fn name(&self) -> &str {
        "AlgebraicSimplification"
    }
    fn is_applicable(&self, graph: &Graph) -> bool {
        graph.node_count() > 0
    }
    fn priority(&self) -> u32 {
        180
    }
    fn set_outputs(&self, outputs: &HashSet<NodeId>) {
        *self.outputs.borrow_mut() = outputs.clone();
    }
}
impl Default for AlgebraicSimplificationPass {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Debug, Clone)]
pub(super) enum SimplificationType {
    ReplaceWithInput(usize),
    ReplaceWithConstant(f32),
    ConvertToMultiply(f32),
}
