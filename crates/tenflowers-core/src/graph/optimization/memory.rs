//! Memory optimization pass
//!
//! This module provides memory optimization capabilities including tensor lifetime
//! analysis, in-place operation identification, and memory layout optimization.

use super::passes::{get_node_inputs, get_node_outputs, OptimizationPass};
use crate::graph::{Graph, NodeId};
use crate::Result;
use std::collections::{HashMap, HashSet};

/// Memory optimization pass
/// Minimizes memory usage by reducing tensor lifetime and enabling in-place operations
pub struct MemoryOptimizationPass;

impl Default for MemoryOptimizationPass {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryOptimizationPass {
    pub fn new() -> Self {
        Self
    }

    /// Analyzes tensor lifetimes in the graph
    fn analyze_tensor_lifetimes(&self, graph: &Graph) -> HashMap<NodeId, (usize, usize)> {
        let mut lifetimes = HashMap::new();

        // Perform topological sort to determine execution order
        let execution_order = self.topological_sort(graph);

        for (position, &node_id) in execution_order.iter().enumerate() {
            // Track when this tensor is first created (birth)
            lifetimes.insert(node_id, (position, position));

            // Update lifetime end for all inputs (last usage)
            let inputs = get_node_inputs(graph, node_id);
            for input_id in inputs {
                if let Some((birth, _)) = lifetimes.get(&input_id) {
                    lifetimes.insert(input_id, (*birth, position));
                }
            }
        }

        lifetimes
    }

    /// Performs topological sort on the graph
    fn topological_sort(&self, graph: &Graph) -> Vec<NodeId> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut temp_visited = HashSet::new();

        // Start from nodes with no inputs (source nodes)
        for node in graph.nodes() {
            let node_id = node.id;
            if !visited.contains(&node_id) && get_node_inputs(graph, node_id).is_empty() {
                self.dfs_visit(graph, node_id, &mut visited, &mut temp_visited, &mut result);
            }
        }

        // Handle remaining nodes (in case of cycles or disconnected components)
        for node in graph.nodes() {
            let node_id = node.id;
            if !visited.contains(&node_id) {
                self.dfs_visit(graph, node_id, &mut visited, &mut temp_visited, &mut result);
            }
        }

        result
    }

    /// DFS visit for topological sort
    #[allow(clippy::only_used_in_recursion)]
    fn dfs_visit(
        &self,
        graph: &Graph,
        node_id: NodeId,
        visited: &mut HashSet<NodeId>,
        temp_visited: &mut HashSet<NodeId>,
        result: &mut Vec<NodeId>,
    ) {
        if temp_visited.contains(&node_id) {
            // Cycle detected, skip for now
            return;
        }

        if visited.contains(&node_id) {
            return;
        }

        temp_visited.insert(node_id);

        let outputs = get_node_outputs(graph, node_id);
        for output_id in outputs {
            self.dfs_visit(graph, output_id, visited, temp_visited, result);
        }

        temp_visited.remove(&node_id);
        visited.insert(node_id);
        result.push(node_id);
    }

    /// Identifies opportunities for in-place operations
    fn identify_inplace_opportunities(&self, graph: &Graph) -> Vec<(NodeId, NodeId)> {
        let mut opportunities = Vec::new();

        for node in graph.nodes() {
            let node_id = node.id;
            // Check if this operation can be done in-place
            if let crate::graph::NodeType::Operation(ref op_name) = node.op_type {
                if self.can_be_inplace(op_name) {
                    let inputs = get_node_inputs(graph, node_id);
                    let outputs = get_node_outputs(graph, node_id);

                    // For in-place operation, we need exactly one input and output
                    // and the input should not be used elsewhere
                    if inputs.len() == 1 && outputs.len() == 1 {
                        let input_id = inputs[0];

                        // Check if input is only used by this node
                        if self.is_only_consumer(graph, input_id, node_id) {
                            opportunities.push((node_id, input_id));
                        }
                    }
                }
            }
        }

        opportunities
    }

    /// Checks if an operation can be performed in-place
    fn can_be_inplace(&self, operation: &str) -> bool {
        matches!(
            operation,
            "add"
                | "sub"
                | "mul"
                | "div"
                | "relu"
                | "sigmoid"
                | "tanh"
                | "abs"
                | "neg"
                | "exp"
                | "log"
                | "sqrt"
                | "square"
        )
    }

    /// Checks if a node is the only consumer of an input
    fn is_only_consumer(&self, graph: &Graph, input_id: NodeId, consumer_id: NodeId) -> bool {
        let outputs = get_node_outputs(graph, input_id);
        outputs.len() == 1 && outputs[0] == consumer_id
    }

    /// Optimizes memory layout by grouping operations
    fn optimize_memory_layout(&self, graph: &mut Graph) -> bool {
        let mut changed = false;

        // Group operations that can share memory
        let memory_groups = self.find_memory_sharing_groups(graph);

        for group in memory_groups {
            if group.len() > 1 {
                // Mark operations in this group for memory sharing
                for &node_id in &group {
                    if let Some(node) = graph.get_node_mut(node_id) {
                        // Add memory sharing hint to node attributes
                        node.attributes.insert(
                            "memory_group".to_string(),
                            crate::graph::AttributeValue::String(format!("{group:?}")),
                        );
                        changed = true;
                    }
                }
            }
        }

        changed
    }

    /// Finds groups of operations that can share memory
    fn find_memory_sharing_groups(&self, graph: &Graph) -> Vec<Vec<NodeId>> {
        let mut groups = Vec::new();
        let mut visited = HashSet::new();

        for node in graph.nodes() {
            let node_id = node.id;
            if visited.contains(&node_id) {
                continue;
            }

            let mut group = Vec::new();
            self.build_memory_group(graph, node_id, &mut group, &mut visited);

            if group.len() > 1 {
                groups.push(group);
            }
        }

        groups
    }

    /// Recursively builds a memory sharing group
    fn build_memory_group(
        &self,
        graph: &Graph,
        node_id: NodeId,
        group: &mut Vec<NodeId>,
        visited: &mut HashSet<NodeId>,
    ) {
        if visited.contains(&node_id) {
            return;
        }

        visited.insert(node_id);
        group.push(node_id);

        // Only group with immediate successors that can share memory
        let outputs = get_node_outputs(graph, node_id);
        for output_id in outputs {
            if let Some(node) = graph.get_node(output_id) {
                // Only group operations that can safely share memory
                if let crate::graph::NodeType::Operation(ref op_name) = node.op_type {
                    if self.can_share_memory(op_name) {
                        self.build_memory_group(graph, output_id, group, visited);
                    }
                }
            }
        }
    }

    /// Checks if operations can safely share memory
    fn can_share_memory(&self, operation: &str) -> bool {
        // Operations that don't change tensor shape can often share memory
        matches!(
            operation,
            "add"
                | "sub"
                | "mul"
                | "div"
                | "relu"
                | "sigmoid"
                | "tanh"
                | "abs"
                | "neg"
                | "exp"
                | "log"
                | "sqrt"
                | "square"
                | "dropout"
        )
    }
}

impl OptimizationPass for MemoryOptimizationPass {
    fn apply(&self, graph: &mut Graph) -> Result<bool> {
        let mut changed = false;

        // Step 1: Analyze tensor lifetimes
        let lifetimes = self.analyze_tensor_lifetimes(graph);

        // Step 2: Identify in-place operation opportunities
        let inplace_ops = self.identify_inplace_opportunities(graph);

        // Step 3: Apply in-place optimizations
        for (node_id, input_id) in inplace_ops {
            if let Some(node) = graph.get_node_mut(node_id) {
                // Mark this operation as in-place
                node.attributes.insert(
                    "inplace".to_string(),
                    crate::graph::AttributeValue::Bool(true),
                );
                node.attributes.insert(
                    "inplace_input".to_string(),
                    crate::graph::AttributeValue::String(input_id.to_string()),
                );
                changed = true;
            }
        }

        // Step 4: Optimize memory layout
        if self.optimize_memory_layout(graph) {
            changed = true;
        }

        // Step 5: Add memory management hints
        for (node_id, (birth, death)) in lifetimes {
            if let Some(node) = graph.get_node_mut(node_id) {
                node.attributes.insert(
                    "lifetime_start".to_string(),
                    crate::graph::AttributeValue::Int(birth as i64),
                );
                node.attributes.insert(
                    "lifetime_end".to_string(),
                    crate::graph::AttributeValue::Int(death as i64),
                );
                changed = true;
            }
        }

        Ok(changed)
    }

    fn name(&self) -> &str {
        "MemoryOptimization"
    }

    fn is_applicable(&self, graph: &Graph) -> bool {
        // Memory optimization is applicable if there are at least 2 nodes
        graph.nodes().count() >= 2
    }

    fn priority(&self) -> u32 {
        80 // Run after CSE and fusion but before dead code elimination
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_optimization_pass() {
        let pass = MemoryOptimizationPass::new();
        assert_eq!(pass.name(), "MemoryOptimization");
        assert_eq!(pass.priority(), 80);

        let graph = Graph::new();
        assert!(!pass.is_applicable(&graph));
    }

    #[test]
    fn test_inplace_operations() {
        let pass = MemoryOptimizationPass::new();

        // Test that various operations can be in-place
        assert!(pass.can_be_inplace("add"));
        assert!(pass.can_be_inplace("relu"));
        assert!(pass.can_be_inplace("exp"));

        // Test that some operations cannot be in-place
        assert!(!pass.can_be_inplace("matmul"));
        assert!(!pass.can_be_inplace("conv2d"));
    }

    #[test]
    fn test_memory_sharing() {
        let pass = MemoryOptimizationPass::new();

        // Test that element-wise operations can share memory
        assert!(pass.can_share_memory("add"));
        assert!(pass.can_share_memory("relu"));
        assert!(pass.can_share_memory("sigmoid"));

        // Test that shape-changing operations cannot share memory
        assert!(!pass.can_share_memory("matmul"));
        assert!(!pass.can_share_memory("conv2d"));
        assert!(!pass.can_share_memory("reshape"));
    }
}
