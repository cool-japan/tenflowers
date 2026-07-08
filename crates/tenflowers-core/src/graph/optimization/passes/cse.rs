//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::graph::{Graph, NodeId};
use crate::Result;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use super::pass_support::{get_node_inputs, OptimizationPass};

/// Common Subexpression Elimination pass
/// Removes duplicate computations
pub struct CSEPass {
    pub(super) outputs: RefCell<HashSet<NodeId>>,
}
impl CSEPass {
    pub fn new() -> Self {
        Self {
            outputs: RefCell::new(HashSet::new()),
        }
    }
    pub(super) fn is_protected(&self, node_id: NodeId) -> bool {
        self.outputs.borrow().contains(&node_id)
    }
    /// Compute a key such that two nodes sharing a key are guaranteed to
    /// always produce the same value, and are therefore safe to merge.
    ///
    /// This requires special-casing two node kinds where a bare
    /// `{:?}`-of-`op_type` key (the pre-fix behavior) is unsound:
    /// - `Constant`: the `NodeType::Constant` enum variant carries no
    ///   payload (the value lives in `attributes["value"]`), so every
    ///   constant's bare type tag is the literal string `"Constant"`
    ///   regardless of value. Without folding the actual value into the
    ///   key, two *different* constants (e.g. a literal `2.0` and a
    ///   literal `5.0`) would be treated as duplicates and merged,
    ///   silently substituting one value for the other.
    /// - `Placeholder` / `Variable`: these are externally supplied,
    ///   per-execution inputs. Two distinct placeholders with the same
    ///   dtype/shape (e.g. two different named inputs) are NOT
    ///   interchangeable even though they look identical structurally —
    ///   each can receive a different value via the feed dict. They must
    ///   never be merged with each other, so they are keyed uniquely by
    ///   node id.
    ///
    /// Plain `Operation` nodes remain keyed by `(op_name, input keys)`:
    /// this codebase's session executor only implements pure, deterministic
    /// operations (no randomness, no hidden state), so two structurally
    /// identical operation subtrees are always safe to merge.
    pub(super) fn compute_expression_key(&self, graph: &Graph, node_id: NodeId) -> String {
        let Some(node) = graph.get_node(node_id) else {
            return format!("node_{node_id}");
        };
        match &node.op_type {
            crate::graph::NodeType::Placeholder { .. }
            | crate::graph::NodeType::Variable { .. } => {
                format!("{:?}#{node_id}", node.op_type)
            }
            crate::graph::NodeType::Constant => {
                format!("Constant:{}", Self::constant_value_key(node))
            }
            crate::graph::NodeType::Operation(op_name) => {
                let inputs = get_node_inputs(graph, node_id);
                let input_keys: Vec<String> = inputs
                    .iter()
                    .map(|&id| self.compute_expression_key(graph, id))
                    .collect();
                format!("Operation({op_name})({})", input_keys.join(","))
            }
        }
    }
    /// Encode a constant node's shape and raw element bits so that
    /// bit-identical constants (including matching NaN/inf patterns) share
    /// a key, while any difference in shape or value does not.
    pub(super) fn constant_value_key(node: &crate::graph::GraphNode) -> String {
        match node.attributes.get("value") {
            Some(crate::graph::AttributeValue::Tensor(tensor)) => {
                let shape_key = format!("{:?}", tensor.shape().dims());
                let data_key = match tensor.as_slice() {
                    Some(slice) => slice
                        .iter()
                        .map(|v| v.to_bits().to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                    None => format!("non_contiguous#{}", node.id),
                };
                format!("{shape_key}:{data_key}")
            }
            _ => "no_value".to_string(),
        }
    }
}
impl OptimizationPass for CSEPass {
    fn apply(&self, graph: &mut Graph) -> Result<bool> {
        let mut changed = false;
        let mut expression_map: HashMap<String, NodeId> = HashMap::new();
        let mut nodes_to_redirect = Vec::new();
        for node in graph.nodes() {
            let expr_key = self.compute_expression_key(graph, node.id);
            if let Some(&existing_node_id) = expression_map.get(&expr_key) {
                nodes_to_redirect.push((node.id, existing_node_id));
            } else {
                expression_map.insert(expr_key, node.id);
            }
        }
        for (duplicate_node, canonical_node) in nodes_to_redirect {
            if self.is_protected(duplicate_node) {
                continue;
            }
            graph.redirect_node_outputs(duplicate_node, canonical_node)?;
            graph.remove_node(duplicate_node)?;
            changed = true;
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
        150
    }
    fn set_outputs(&self, outputs: &HashSet<NodeId>) {
        *self.outputs.borrow_mut() = outputs.clone();
    }
}
impl Default for CSEPass {
    fn default() -> Self {
        Self::new()
    }
}
