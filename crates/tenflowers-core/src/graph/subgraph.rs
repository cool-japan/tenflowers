//! Subgraph Operations
//!
//! This module provides functionality for extracting and creating subgraphs
//! based on various criteria and traversal patterns.

use super::core::*;
use crate::{device::Device, error::TensorError};
use std::collections::{HashMap, HashSet, VecDeque};

impl Graph {
    /// Create a subgraph containing only the specified nodes
    pub fn subgraph(&self, node_ids: &[NodeId]) -> Result<Graph, TensorError> {
        let node_set: HashSet<NodeId> = node_ids.iter().cloned().collect();
        let mut subgraph = Graph::new();
        let mut id_mapping: HashMap<NodeId, NodeId> = HashMap::new();

        // Add nodes to subgraph
        for &node_id in node_ids {
            if let Some(node) = self.nodes.get(&node_id) {
                let new_id = subgraph.add_node(
                    node.name.clone(),
                    node.op_type.clone(),
                    node.device,
                    node.attributes.clone(),
                )?;
                id_mapping.insert(node_id, new_id);
            } else {
                return Err(TensorError::invalid_argument(format!(
                    "Node {} not found in graph",
                    node_id
                )));
            }
        }

        // Add edges between included nodes
        for edge in self.edges.values() {
            if node_set.contains(&edge.from_node) && node_set.contains(&edge.to_node) {
                let new_from = *id_mapping.get(&edge.from_node).unwrap();
                let new_to = *id_mapping.get(&edge.to_node).unwrap();

                subgraph.add_edge(
                    new_from,
                    new_to,
                    edge.from_output,
                    edge.to_input,
                    edge.dtype,
                    edge.shape.clone(),
                    edge.is_control,
                )?;
            }
        }

        Ok(subgraph)
    }

    /// Create a subgraph containing nodes of specific operation types
    pub fn subgraph_by_op_types(&self, op_types: &[&str]) -> Result<Graph, TensorError> {
        let node_ids: Vec<NodeId> = self
            .nodes
            .values()
            .filter(|node| match &node.op_type {
                NodeType::Operation(op_name) => op_types.contains(&op_name.as_str()),
                _ => false,
            })
            .map(|node| node.id)
            .collect();

        self.subgraph(&node_ids)
    }

    /// Create a subgraph containing nodes on a specific device
    pub fn subgraph_by_device(&self, device: Device) -> Result<Graph, TensorError> {
        let node_ids: Vec<NodeId> = self
            .nodes
            .values()
            .filter(|node| node.device == device)
            .map(|node| node.id)
            .collect();

        self.subgraph(&node_ids)
    }

    /// Create a subgraph with all dependencies of the specified nodes
    pub fn subgraph_with_dependencies(
        &self,
        root_nodes: &[NodeId],
        include_control_deps: bool,
    ) -> Result<Graph, TensorError> {
        let mut included_nodes = HashSet::new();
        let mut queue = VecDeque::new();

        // Start with root nodes
        for &node_id in root_nodes {
            if self.nodes.contains_key(&node_id) {
                queue.push_back(node_id);
                included_nodes.insert(node_id);
            } else {
                return Err(TensorError::invalid_argument(format!(
                    "Node {} not found in graph",
                    node_id
                )));
            }
        }

        // Traverse backwards through dependencies
        while let Some(node_id) = queue.pop_front() {
            if let Some(node) = self.nodes.get(&node_id) {
                for &edge_id in &node.inputs {
                    if let Some(edge) = self.edges.get(&edge_id) {
                        if (!edge.is_control || include_control_deps)
                            && !included_nodes.contains(&edge.from_node)
                        {
                            included_nodes.insert(edge.from_node);
                            queue.push_back(edge.from_node);
                        }
                    }
                }
            }
        }

        let node_ids: Vec<NodeId> = included_nodes.into_iter().collect();
        self.subgraph(&node_ids)
    }

    /// Find the connected component containing the specified node
    pub fn connected_component(&self, start_node: NodeId) -> Result<Graph, TensorError> {
        if !self.nodes.contains_key(&start_node) {
            return Err(TensorError::invalid_argument(format!(
                "Node {} not found in graph",
                start_node
            )));
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(start_node);
        visited.insert(start_node);

        // BFS to find all connected nodes
        while let Some(node_id) = queue.pop_front() {
            if let Some(node) = self.nodes.get(&node_id) {
                // Check all input edges
                for &edge_id in &node.inputs {
                    if let Some(edge) = self.edges.get(&edge_id) {
                        if !visited.contains(&edge.from_node) {
                            visited.insert(edge.from_node);
                            queue.push_back(edge.from_node);
                        }
                    }
                }
                // Check all output edges
                for &edge_id in &node.outputs {
                    if let Some(edge) = self.edges.get(&edge_id) {
                        if !visited.contains(&edge.to_node) {
                            visited.insert(edge.to_node);
                            queue.push_back(edge.to_node);
                        }
                    }
                }
            }
        }

        let node_ids: Vec<NodeId> = visited.into_iter().collect();
        self.subgraph(&node_ids)
    }

    /// Create a forward slice from the given nodes (includes all nodes reachable forward)
    pub fn forward_slice(
        &self,
        start_nodes: &[NodeId],
        include_control_deps: bool,
    ) -> Result<Graph, TensorError> {
        let mut included_nodes = HashSet::new();
        let mut queue = VecDeque::new();

        // Start with the specified nodes
        for &node_id in start_nodes {
            if self.nodes.contains_key(&node_id) {
                queue.push_back(node_id);
                included_nodes.insert(node_id);
            } else {
                return Err(TensorError::invalid_argument(format!(
                    "Node {} not found in graph",
                    node_id
                )));
            }
        }

        // Traverse forward through the graph
        while let Some(node_id) = queue.pop_front() {
            if let Some(node) = self.nodes.get(&node_id) {
                for &edge_id in &node.outputs {
                    if let Some(edge) = self.edges.get(&edge_id) {
                        if (!edge.is_control || include_control_deps)
                            && !included_nodes.contains(&edge.to_node)
                        {
                            included_nodes.insert(edge.to_node);
                            queue.push_back(edge.to_node);
                        }
                    }
                }
            }
        }

        let node_ids: Vec<NodeId> = included_nodes.into_iter().collect();
        self.subgraph(&node_ids)
    }

    /// Create a subgraph based on a custom predicate function
    pub fn subgraph_by_predicate<F>(&self, predicate: F) -> Result<Graph, TensorError>
    where
        F: Fn(&GraphNode) -> bool,
    {
        let node_ids: Vec<NodeId> = self
            .nodes
            .values()
            .filter(|node| predicate(node))
            .map(|node| node.id)
            .collect();

        self.subgraph(&node_ids)
    }
}
