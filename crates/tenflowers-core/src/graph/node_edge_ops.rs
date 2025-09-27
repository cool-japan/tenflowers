//! Node and Edge Management Operations
//!
//! This module provides functionality for managing individual nodes and edges,
//! including removal, replacement, and redirection operations.

use super::core::*;
use crate::error::TensorError;

impl Graph {
    /// Remove a node and all its associated edges
    pub fn remove_node(&mut self, node_id: NodeId) -> Result<(), TensorError> {
        let node = self
            .nodes
            .get(&node_id)
            .ok_or_else(|| TensorError::invalid_argument(format!("Node {} not found", node_id)))?;

        // Collect all edges to remove
        let mut edges_to_remove = Vec::new();
        edges_to_remove.extend(node.inputs.iter());
        edges_to_remove.extend(node.outputs.iter());

        // Remove the node from name mapping
        self.name_to_node.remove(&node.name);

        // Remove all associated edges
        for &edge_id in &edges_to_remove {
            if let Some(edge) = self.edges.remove(&edge_id) {
                // Update the other node's edge lists
                if edge.from_node != node_id {
                    if let Some(from_node) = self.nodes.get_mut(&edge.from_node) {
                        from_node.outputs.retain(|&id| id != edge_id);
                    }
                }
                if edge.to_node != node_id {
                    if let Some(to_node) = self.nodes.get_mut(&edge.to_node) {
                        to_node.inputs.retain(|&id| id != edge_id);
                    }
                }
            }
        }

        // Remove the node
        self.nodes.remove(&node_id);

        self.topological_order = None; // Invalidate cached order
        self.version += 1;

        Ok(())
    }

    /// Remove an edge
    pub fn remove_edge(&mut self, edge_id: EdgeId) -> Result<(), TensorError> {
        let edge = self
            .edges
            .remove(&edge_id)
            .ok_or_else(|| TensorError::invalid_argument(format!("Edge {} not found", edge_id)))?;

        // Update node edge lists
        if let Some(from_node) = self.nodes.get_mut(&edge.from_node) {
            from_node.outputs.retain(|&id| id != edge_id);
        }
        if let Some(to_node) = self.nodes.get_mut(&edge.to_node) {
            to_node.inputs.retain(|&id| id != edge_id);
        }

        self.topological_order = None; // Invalidate cached order
        self.version += 1;

        Ok(())
    }

    /// Replace a node with a constant value
    pub fn replace_with_constant(
        &mut self,
        node_id: NodeId,
        constant_value: crate::tensor::Tensor<f32>,
    ) -> Result<(), TensorError> {
        let node = self
            .nodes
            .get_mut(&node_id)
            .ok_or_else(|| TensorError::invalid_argument(format!("Node {} not found", node_id)))?;

        // Update node type to constant
        node.op_type = NodeType::Constant;

        // Add the constant value as an attribute
        node.attributes
            .insert("value".to_string(), AttributeValue::Tensor(constant_value));

        // Remove all input edges since constants don't have inputs
        let input_edges: Vec<EdgeId> = node.inputs.clone();
        node.inputs.clear();

        for edge_id in input_edges {
            self.remove_edge(edge_id)?;
        }

        self.version += 1;

        Ok(())
    }

    /// Redirect all outputs from one node to another
    pub fn redirect_node_outputs(
        &mut self,
        from_node: NodeId,
        to_node: NodeId,
    ) -> Result<usize, TensorError> {
        if !self.nodes.contains_key(&from_node) {
            return Err(TensorError::invalid_argument(format!(
                "Source node {} not found",
                from_node
            )));
        }
        if !self.nodes.contains_key(&to_node) {
            return Err(TensorError::invalid_argument(format!(
                "Target node {} not found",
                to_node
            )));
        }

        let output_edges: Vec<EdgeId> = self.nodes.get(&from_node).unwrap().outputs.clone();

        let mut redirected_count = 0;

        for edge_id in output_edges {
            if let Some(edge) = self.edges.get_mut(&edge_id) {
                edge.from_node = to_node;
                redirected_count += 1;

                // Update node edge lists
                self.nodes.get_mut(&to_node).unwrap().outputs.push(edge_id);
            }
        }

        // Clear output edges from the original node
        self.nodes.get_mut(&from_node).unwrap().outputs.clear();

        if redirected_count > 0 {
            self.topological_order = None; // Invalidate cached order
            self.version += 1;
        }

        Ok(redirected_count)
    }
}
