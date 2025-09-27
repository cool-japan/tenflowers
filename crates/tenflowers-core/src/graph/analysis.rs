//! Graph Analysis Operations
//!
//! This module provides graph analysis functionality including topological sorting,
//! validation, and identification of input/output nodes.

use super::core::*;
use crate::error::TensorError;
use std::collections::{HashMap, HashSet, VecDeque};

impl Graph {
    /// Compute and cache the topological order of nodes
    pub fn compute_topological_order(&mut self) -> Result<&[NodeId], TensorError> {
        if self.topological_order.is_some() {
            return Ok(self.topological_order.as_ref().unwrap());
        }

        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        // Initialize in-degree count and adjacency list
        for node in self.nodes.values() {
            in_degree.insert(node.id, 0);
            adjacency.insert(node.id, Vec::new());
        }

        // Build adjacency list and count in-degrees
        for edge in self.edges.values() {
            if !edge.is_control {
                // Only consider data dependencies for topological order
                adjacency
                    .get_mut(&edge.from_node)
                    .unwrap()
                    .push(edge.to_node);
                *in_degree.get_mut(&edge.to_node).unwrap() += 1;
            }
        }

        // Kahn's algorithm
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        let mut result: Vec<NodeId> = Vec::new();

        // Start with nodes that have no incoming edges
        for (&node_id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(node_id);
            }
        }

        while let Some(node_id) = queue.pop_front() {
            result.push(node_id);

            // Remove this node from the graph and update in-degrees
            for &neighbor in adjacency.get(&node_id).unwrap() {
                let neighbor_degree = in_degree.get_mut(&neighbor).unwrap();
                *neighbor_degree -= 1;
                if *neighbor_degree == 0 {
                    queue.push_back(neighbor);
                }
            }
        }

        // Check for cycles
        if result.len() != self.nodes.len() {
            return Err(TensorError::invalid_argument(
                "Graph contains cycles".to_string(),
            ));
        }

        self.topological_order = Some(result);
        Ok(self.topological_order.as_ref().unwrap())
    }

    /// Validate the graph structure
    pub fn validate(&self) -> Result<(), TensorError> {
        // Check that all edge endpoints reference valid nodes
        for edge in self.edges.values() {
            if !self.nodes.contains_key(&edge.from_node) {
                return Err(TensorError::invalid_argument(format!(
                    "Edge {} references non-existent source node {}",
                    edge.id, edge.from_node
                )));
            }
            if !self.nodes.contains_key(&edge.to_node) {
                return Err(TensorError::invalid_argument(format!(
                    "Edge {} references non-existent destination node {}",
                    edge.id, edge.to_node
                )));
            }
        }

        // Check that node edge lists are consistent with actual edges
        for node in self.nodes.values() {
            for &edge_id in &node.inputs {
                if let Some(edge) = self.edges.get(&edge_id) {
                    if edge.to_node != node.id {
                        return Err(TensorError::invalid_argument(format!(
                            "Node {} lists edge {} as input, but edge points to node {}",
                            node.id, edge_id, edge.to_node
                        )));
                    }
                } else {
                    return Err(TensorError::invalid_argument(format!(
                        "Node {} references non-existent input edge {}",
                        node.id, edge_id
                    )));
                }
            }

            for &edge_id in &node.outputs {
                if let Some(edge) = self.edges.get(&edge_id) {
                    if edge.from_node != node.id {
                        return Err(TensorError::invalid_argument(format!(
                            "Node {} lists edge {} as output, but edge comes from node {}",
                            node.id, edge_id, edge.from_node
                        )));
                    }
                } else {
                    return Err(TensorError::invalid_argument(format!(
                        "Node {} references non-existent output edge {}",
                        node.id, edge_id
                    )));
                }
            }
        }

        // Check for name uniqueness
        let mut seen_names = HashSet::new();
        for node in self.nodes.values() {
            if !seen_names.insert(&node.name) {
                return Err(TensorError::invalid_argument(format!(
                    "Duplicate node name: '{}'",
                    node.name
                )));
            }
        }

        Ok(())
    }

    /// Find all input nodes (nodes with no incoming data edges)
    pub fn input_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .values()
            .filter(|node| {
                !node.inputs.iter().any(|&edge_id| {
                    self.edges
                        .get(&edge_id)
                        .is_some_and(|edge| !edge.is_control)
                })
            })
            .map(|node| node.id)
            .collect()
    }

    /// Find all output nodes (nodes with no outgoing data edges)
    pub fn output_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .values()
            .filter(|node| {
                !node.outputs.iter().any(|&edge_id| {
                    self.edges
                        .get(&edge_id)
                        .is_some_and(|edge| !edge.is_control)
                })
            })
            .map(|node| node.id)
            .collect()
    }
}
