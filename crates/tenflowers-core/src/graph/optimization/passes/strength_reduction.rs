//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::graph::{Graph, NodeId};
use crate::Result;
use std::cell::RefCell;
use std::collections::HashSet;

use super::pass_support::{
    describe_input_edge, get_node_inputs, insert_scalar_constant, is_exact_power_of_two,
    rewrite_operation_node, OptimizationPass,
};

#[derive(Debug, Clone)]
pub(super) enum ReductionType {
    Square,
    DivToMul,
    ExpLogCancel,
    LogExpCancel,
}
/// Strength reduction pass
/// Replaces expensive operations with cheaper equivalents
pub struct StrengthReductionPass {
    pub(super) outputs: RefCell<HashSet<NodeId>>,
}
impl StrengthReductionPass {
    pub fn new() -> Self {
        Self {
            outputs: RefCell::new(HashSet::new()),
        }
    }
    pub(super) fn is_protected(&self, node_id: NodeId) -> bool {
        self.outputs.borrow().contains(&node_id)
    }
    pub(super) fn find_reduction(&self, graph: &Graph, node_id: NodeId) -> Option<ReductionType> {
        let node = graph.get_node(node_id)?;
        if let crate::graph::NodeType::Operation(op_name) = &node.op_type {
            let inputs = get_node_inputs(graph, node_id);
            match op_name.as_str() {
                "Pow" if inputs.len() == 2 && self.is_constant_value(graph, inputs[1], 2.0) => {
                    return Some(ReductionType::Square);
                }
                "Div"
                    if inputs.len() == 2
                        && self
                            .scalar_constant_value(graph, inputs[1])
                            .is_some_and(is_exact_power_of_two) =>
                {
                    return Some(ReductionType::DivToMul);
                }
                "Exp" if inputs.len() == 1 && self.is_log_operation(graph, inputs[0]) => {
                    return Some(ReductionType::ExpLogCancel);
                }
                "Log" if inputs.len() == 1 && self.is_exp_operation(graph, inputs[0]) => {
                    return Some(ReductionType::LogExpCancel);
                }
                _ => {}
            }
        }
        None
    }
    pub(super) fn is_constant_value(&self, graph: &Graph, node_id: NodeId, value: f32) -> bool {
        self.scalar_constant_value(graph, node_id)
            .is_some_and(|v| (v - value).abs() < 1e-6)
    }
    /// Returns the value held by `node_id` if it is a scalar `Constant`
    /// node (a size-1 tensor), else `None`.
    pub(super) fn scalar_constant_value(&self, graph: &Graph, node_id: NodeId) -> Option<f32> {
        let node = graph.get_node(node_id)?;
        if let crate::graph::NodeType::Constant = &node.op_type {
            if let Some(crate::graph::AttributeValue::Tensor(tensor)) = node.attributes.get("value")
            {
                if tensor.shape().size() == 1 {
                    if let Some(slice) = tensor.as_slice() {
                        return slice.first().copied();
                    }
                }
            }
        }
        None
    }
    pub(super) fn is_log_operation(&self, graph: &Graph, node_id: NodeId) -> bool {
        if let Some(node) = graph.get_node(node_id) {
            matches!(
                node.op_type, crate ::graph::NodeType::Operation(ref op) if op == "Log"
            )
        } else {
            false
        }
    }
    pub(super) fn is_exp_operation(&self, graph: &Graph, node_id: NodeId) -> bool {
        if let Some(node) = graph.get_node(node_id) {
            matches!(
                node.op_type, crate ::graph::NodeType::Operation(ref op) if op == "Exp"
            )
        } else {
            false
        }
    }
    pub(super) fn apply_reduction(
        &self,
        graph: &mut Graph,
        node_id: NodeId,
        reduction: ReductionType,
    ) -> Result<bool> {
        match reduction {
            ReductionType::Square => {
                let Some((x_node, x_output, dtype, shape)) = describe_input_edge(graph, node_id, 0)
                else {
                    return Ok(false);
                };
                rewrite_operation_node(
                    graph,
                    node_id,
                    "Mul",
                    &[
                        (x_node, x_output, dtype, shape.clone()),
                        (x_node, x_output, dtype, shape),
                    ],
                )?;
                Ok(true)
            }
            ReductionType::DivToMul => {
                let Some((x_node, x_output, dtype, shape)) = describe_input_edge(graph, node_id, 0)
                else {
                    return Ok(false);
                };
                let Some(divisor_id) = get_node_inputs(graph, node_id).get(1).copied() else {
                    return Ok(false);
                };
                let Some(c) = self.scalar_constant_value(graph, divisor_id) else {
                    return Ok(false);
                };
                let reciprocal = 1.0f32 / c;
                if !reciprocal.is_finite() {
                    return Ok(false);
                }
                let device = graph
                    .get_node(node_id)
                    .map(|node| node.device)
                    .unwrap_or(crate::device::Device::Cpu);
                let const_id = insert_scalar_constant(graph, reciprocal, device, "reciprocal")?;
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
            ReductionType::ExpLogCancel | ReductionType::LogExpCancel => {
                if self.is_protected(node_id) {
                    return Ok(false);
                }
                let inputs = get_node_inputs(graph, node_id);
                if let Some(&inner_node) = inputs.first() {
                    let inner_inputs = get_node_inputs(graph, inner_node);
                    if let Some(&innermost) = inner_inputs.first() {
                        graph.redirect_node_outputs(node_id, innermost)?;
                        graph.remove_node(node_id)?;
                        return Ok(true);
                    }
                }
                Ok(false)
            }
        }
    }
}
impl OptimizationPass for StrengthReductionPass {
    fn apply(&self, graph: &mut Graph) -> Result<bool> {
        let mut changed = false;
        let mut reductions = Vec::new();
        for node in graph.nodes() {
            if let Some(reduction) = self.find_reduction(graph, node.id) {
                reductions.push((node.id, reduction));
            }
        }
        for (node_id, reduction) in reductions {
            if self.apply_reduction(graph, node_id, reduction)? {
                changed = true;
            }
        }
        Ok(changed)
    }
    fn name(&self) -> &str {
        "StrengthReduction"
    }
    fn is_applicable(&self, graph: &Graph) -> bool {
        graph.node_count() > 0
    }
    fn priority(&self) -> u32 {
        140
    }
    fn set_outputs(&self, outputs: &HashSet<NodeId>) {
        *self.outputs.borrow_mut() = outputs.clone();
    }
}
impl Default for StrengthReductionPass {
    fn default() -> Self {
        Self::new()
    }
}
