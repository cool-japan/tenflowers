//! Control Dependency Management
//!
//! This module provides functionality for managing control dependencies
//! between nodes in the computation graph.

use super::core::*;
use crate::error::TensorError;

impl Graph {
    /// Add a control dependency between two nodes
    pub fn add_control_dependency(
        &mut self,
        from_node: NodeId,
        to_node: NodeId,
    ) -> Result<EdgeId, TensorError> {
        self.add_edge(
            from_node,
            to_node,
            0,
            0,
            crate::dtype::DType::Float32, // Control dependencies don't have meaningful types
            crate::shape::Shape::new(vec![]),
            true, // This is a control edge
        )
    }

    /// Add control dependencies from one node to multiple nodes
    pub fn add_control_dependencies(
        &mut self,
        from_node: NodeId,
        to_nodes: &[NodeId],
    ) -> Result<Vec<EdgeId>, TensorError> {
        let mut edge_ids = Vec::new();
        for &to_node in to_nodes {
            let edge_id = self.add_control_dependency(from_node, to_node)?;
            edge_ids.push(edge_id);
        }
        Ok(edge_ids)
    }

    /// Get all nodes that this node has control dependencies on
    pub fn get_control_dependencies(&self, node_id: NodeId) -> Vec<NodeId> {
        let node = match self.nodes.get(&node_id) {
            Some(node) => node,
            None => return Vec::new(),
        };

        node.inputs
            .iter()
            .filter_map(|&edge_id| self.edges.get(&edge_id))
            .filter(|edge| edge.is_control)
            .map(|edge| edge.from_node)
            .collect()
    }

    /// Get all nodes that depend on this node via control dependencies
    pub fn get_control_dependents(&self, node_id: NodeId) -> Vec<NodeId> {
        let node = match self.nodes.get(&node_id) {
            Some(node) => node,
            None => return Vec::new(),
        };

        node.outputs
            .iter()
            .filter_map(|&edge_id| self.edges.get(&edge_id))
            .filter(|edge| edge.is_control)
            .map(|edge| edge.to_node)
            .collect()
    }

    /// Check if there's a control dependency between two nodes
    pub fn has_control_dependency(&self, from_node: NodeId, to_node: NodeId) -> bool {
        self.edges
            .values()
            .any(|edge| edge.is_control && edge.from_node == from_node && edge.to_node == to_node)
    }

    /// Remove all control dependencies for a node (both incoming and outgoing)
    pub fn remove_control_dependencies(&mut self, node_id: NodeId) -> Result<usize, TensorError> {
        let node = self
            .nodes
            .get(&node_id)
            .ok_or_else(|| TensorError::invalid_argument(format!("Node {} not found", node_id)))?;

        let mut control_edges_to_remove = Vec::new();

        // Find incoming control edges
        for &edge_id in &node.inputs {
            if let Some(edge) = self.edges.get(&edge_id) {
                if edge.is_control {
                    control_edges_to_remove.push(edge_id);
                }
            }
        }

        // Find outgoing control edges
        for &edge_id in &node.outputs {
            if let Some(edge) = self.edges.get(&edge_id) {
                if edge.is_control {
                    control_edges_to_remove.push(edge_id);
                }
            }
        }

        let removed_count = control_edges_to_remove.len();

        // Remove the control edges
        for edge_id in control_edges_to_remove {
            self.remove_edge(edge_id)?;
        }

        Ok(removed_count)
    }

    /// Create a control context - ensure all nodes in the list execute in order
    pub fn create_control_context(&mut self, context_nodes: &[NodeId]) -> Result<(), TensorError> {
        // Validate all nodes exist
        for &node_id in context_nodes {
            if !self.nodes.contains_key(&node_id) {
                return Err(TensorError::invalid_argument(format!(
                    "Node {} not found",
                    node_id
                )));
            }
        }

        // Create control dependencies between consecutive nodes
        for window in context_nodes.windows(2) {
            self.add_control_dependency(window[0], window[1])?;
        }

        Ok(())
    }
}
