//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::graph::{EdgeId, Graph, NodeId};
use crate::Result;
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
/// Generate a node name guaranteed not to collide with any existing node
/// name in `graph`. Used by rewrites that must insert a brand-new constant
/// node (e.g. `x + x -> 2 * x`).
pub(super) fn unique_node_name(graph: &Graph, prefix: &str) -> String {
    let mut counter = graph.next_node_id;
    loop {
        let candidate = format!("__optpass_{prefix}_{counter}");
        if !graph.name_to_node.contains_key(&candidate) {
            return candidate;
        }
        counter += 1;
    }
}
/// Insert a new scalar `Constant` node holding `value`, returning its id.
pub(super) fn insert_scalar_constant(
    graph: &mut Graph,
    value: f32,
    device: crate::device::Device,
    name_hint: &str,
) -> Result<NodeId> {
    let name = unique_node_name(graph, name_hint);
    let mut attributes = HashMap::new();
    attributes.insert(
        "value".to_string(),
        crate::graph::AttributeValue::Tensor(crate::tensor::Tensor::from_scalar(value)),
    );
    graph.add_node(name, crate::graph::NodeType::Constant, device, attributes)
}
/// Describe the edge feeding input slot `input_index` of `node_id`, as
/// `(from_node, from_output, dtype, shape)`. Returns `None` if the node or
/// input slot does not exist.
pub(super) fn describe_input_edge(
    graph: &Graph,
    node_id: NodeId,
    input_index: usize,
) -> Option<(NodeId, usize, crate::dtype::DType, crate::shape::Shape)> {
    let node = graph.get_node(node_id)?;
    let edge_id = *node.inputs.get(input_index)?;
    let edge = graph.get_edge(edge_id)?;
    Some((
        edge.from_node,
        edge.from_output,
        edge.dtype,
        edge.shape.clone(),
    ))
}
/// Replace `node_id`'s current inputs with `new_inputs` (in order) and
/// change its operation name, while preserving the node's identity (its id
/// and name are untouched). This lets strength-reduction / algebraic
/// rewrites swap in a cheaper equivalent operation (e.g.
/// `pow(x, 2) -> mul(x, x)`) without disturbing consumer edges or any
/// fetch/output binding that references `node_id` directly.
pub(super) fn rewrite_operation_node(
    graph: &mut Graph,
    node_id: NodeId,
    new_op_name: &str,
    new_inputs: &[(NodeId, usize, crate::dtype::DType, crate::shape::Shape)],
) -> Result<()> {
    let old_input_edges: Vec<EdgeId> = graph
        .get_node(node_id)
        .map(|node| node.inputs.clone())
        .unwrap_or_default();
    for edge_id in old_input_edges {
        graph.remove_edge(edge_id)?;
    }
    for (input_index, (from_node, from_output, dtype, shape)) in new_inputs.iter().enumerate() {
        graph.add_edge(
            *from_node,
            node_id,
            *from_output,
            input_index,
            *dtype,
            shape.clone(),
            false,
        )?;
    }
    if let Some(node) = graph.get_node_mut(node_id) {
        node.op_type = crate::graph::NodeType::Operation(new_op_name.to_string());
    }
    Ok(())
}
/// Returns true if `value` is a finite, nonzero power of two (including
/// negative exponents such as 0.5 or 0.25). Such values have an exactly
/// representable reciprocal in IEEE-754 binary floating point, which is what
/// makes `x * (1/value)` bit-for-bit identical to `x / value`.
pub(super) fn is_exact_power_of_two(value: f32) -> bool {
    if !value.is_finite() || value == 0.0 {
        return false;
    }
    let bits = value.to_bits();
    let mantissa = bits & 0x007F_FFFF;
    let exponent = (bits >> 23) & 0xFF;
    mantissa == 0 && exponent != 0 && exponent != 0xFF
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
    /// Inform this pass which node ids are explicit graph outputs (e.g.
    /// pending fetch targets in the eager `Session` executor) that must keep
    /// their identity across optimization, even if the pass would otherwise
    /// eliminate or merge them away.
    ///
    /// Most passes never destroy a node's identity (they either leave nodes
    /// untouched or mutate them in place) and can rely on this default
    /// no-op. Passes that do remove/merge node ids — dead code elimination,
    /// common subexpression elimination, and any rewrite that redirects a
    /// node's consumers elsewhere before deleting it — override this to
    /// treat the given ids as protected.
    fn set_outputs(&self, _outputs: &HashSet<NodeId>) {}
}
