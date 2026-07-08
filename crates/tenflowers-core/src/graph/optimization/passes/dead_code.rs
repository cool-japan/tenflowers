//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::graph::{Graph, NodeId};
use crate::Result;
use std::cell::RefCell;
use std::collections::HashSet;

use super::pass_support::{get_node_inputs, get_node_outputs, OptimizationPass};

/// Dead code elimination pass
/// Removes nodes that don't contribute to any output
pub struct DeadCodeEliminationPass {
    pub(super) outputs: RefCell<HashSet<NodeId>>,
}
impl DeadCodeEliminationPass {
    pub fn new() -> Self {
        Self {
            outputs: RefCell::new(HashSet::new()),
        }
    }
    /// Construct a pass that additionally treats `outputs` as GC roots, so
    /// nodes reachable only from these ids are preserved even if nothing
    /// else about them would otherwise mark them as reachable.
    pub fn with_outputs(outputs: HashSet<NodeId>) -> Self {
        Self {
            outputs: RefCell::new(outputs),
        }
    }
    pub(super) fn is_output_node(&self, graph: &Graph, node_id: NodeId) -> bool {
        let outputs = get_node_outputs(graph, node_id);
        outputs.is_empty() || self.is_marked_as_output(node_id)
    }
    pub(super) fn is_marked_as_output(&self, node_id: NodeId) -> bool {
        self.outputs.borrow().contains(&node_id)
    }
    #[allow(clippy::only_used_in_recursion)]
    pub(super) fn mark_reachable(
        &self,
        graph: &Graph,
        node_id: NodeId,
        reachable: &mut HashSet<NodeId>,
    ) {
        if reachable.contains(&node_id) {
            return;
        }
        reachable.insert(node_id);
        for input_id in get_node_inputs(graph, node_id) {
            self.mark_reachable(graph, input_id, reachable);
        }
    }
}
impl OptimizationPass for DeadCodeEliminationPass {
    fn apply(&self, graph: &mut Graph) -> Result<bool> {
        let mut changed = false;
        let mut reachable = HashSet::new();
        for node in graph.nodes() {
            if self.is_output_node(graph, node.id) {
                self.mark_reachable(graph, node.id, &mut reachable);
            }
        }
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
        50
    }
    fn set_outputs(&self, outputs: &HashSet<NodeId>) {
        *self.outputs.borrow_mut() = outputs.clone();
    }
}
impl Default for DeadCodeEliminationPass {
    fn default() -> Self {
        Self::new()
    }
}
