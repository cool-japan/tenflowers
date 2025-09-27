//! Graph Manipulation Operations
//!
//! This module provides functionality for manipulating and transforming graphs,
//! including extending, merging, and replacing graph structures.

use super::core::*;
use crate::{device::Device, error::TensorError};
use std::collections::HashMap;

impl Graph {
    /// Extend this graph with another graph
    pub fn extend_with_graph(
        &mut self,
        other: &Graph,
        node_prefix: Option<&str>,
    ) -> Result<HashMap<NodeId, NodeId>, TensorError> {
        let mut id_mapping = HashMap::new();
        let prefix = node_prefix.unwrap_or("");

        // Add all nodes from the other graph
        for node in other.nodes.values() {
            let new_name = if prefix.is_empty() {
                node.name.clone()
            } else {
                format!("{prefix}_{}", node.name)
            };

            let new_id = self.add_node(
                new_name,
                node.op_type.clone(),
                node.device,
                node.attributes.clone(),
            )?;
            id_mapping.insert(node.id, new_id);
        }

        // Add all edges from the other graph
        for edge in other.edges.values() {
            let new_from = *id_mapping.get(&edge.from_node).unwrap();
            let new_to = *id_mapping.get(&edge.to_node).unwrap();

            self.add_edge(
                new_from,
                new_to,
                edge.from_output,
                edge.to_input,
                edge.dtype,
                edge.shape.clone(),
                edge.is_control,
            )?;
        }

        Ok(id_mapping)
    }

    /// Integrate a subgraph into this graph at specific connection points
    pub fn integrate_subgraph(
        &mut self,
        subgraph: &Graph,
        input_connections: &[(NodeId, usize, NodeId, usize)], // (external_node, output_idx, subgraph_input, input_idx)
        output_connections: &[(NodeId, usize, NodeId, usize)], // (subgraph_output, output_idx, external_node, input_idx)
        node_prefix: Option<&str>,
    ) -> Result<HashMap<NodeId, NodeId>, TensorError> {
        // First extend the graph with the subgraph
        let id_mapping = self.extend_with_graph(subgraph, node_prefix)?;

        // Create input connections
        for &(external_node, output_idx, subgraph_input, input_idx) in input_connections {
            if !self.nodes.contains_key(&external_node) {
                return Err(TensorError::invalid_argument(format!(
                    "External node {} not found",
                    external_node
                )));
            }

            let mapped_subgraph_node = *id_mapping.get(&subgraph_input).ok_or_else(|| {
                TensorError::invalid_argument(format!(
                    "Subgraph input node {} not found",
                    subgraph_input
                ))
            })?;

            // Create edge from external node to subgraph input
            // Note: This is a simplified version - in practice, we'd need to infer types and shapes
            self.add_edge(
                external_node,
                mapped_subgraph_node,
                output_idx,
                input_idx,
                crate::dtype::DType::Float32, // Default type
                crate::shape::Shape::new(vec![]),
                false,
            )?;
        }

        // Create output connections
        for &(subgraph_output, output_idx, external_node, input_idx) in output_connections {
            if !self.nodes.contains_key(&external_node) {
                return Err(TensorError::invalid_argument(format!(
                    "External node {} not found",
                    external_node
                )));
            }

            let mapped_subgraph_node = *id_mapping.get(&subgraph_output).ok_or_else(|| {
                TensorError::invalid_argument(format!(
                    "Subgraph output node {} not found",
                    subgraph_output
                ))
            })?;

            // Create edge from subgraph output to external node
            self.add_edge(
                mapped_subgraph_node,
                external_node,
                output_idx,
                input_idx,
                crate::dtype::DType::Float32, // Default type
                crate::shape::Shape::new(vec![]),
                false,
            )?;
        }

        Ok(id_mapping)
    }

    /// Merge multiple graphs into a single graph
    pub fn merge_graphs(graphs: &[&Graph]) -> Result<Graph, TensorError> {
        let mut merged = Graph::new();

        for (i, graph) in graphs.iter().enumerate() {
            let prefix = format!("graph_{}", i);
            merged.extend_with_graph(graph, Some(&prefix))?;
        }

        Ok(merged)
    }

    /// Add a node with automatically generated unique name
    pub fn add_node_auto_name(
        &mut self,
        base_name: &str,
        op_type: NodeType,
        device: Device,
        attributes: HashMap<String, AttributeValue>,
    ) -> Result<NodeId, TensorError> {
        let mut counter = 0;
        let mut name = base_name.to_string();

        while self.name_to_node.contains_key(&name) {
            counter += 1;
            name = format!("{}_{}", base_name, counter);
        }

        self.add_node(name, op_type, device, attributes)
    }

    /// Add a complete operation subgraph (operation with inputs and outputs)
    pub fn add_operation_subgraph(
        &mut self,
        op_name: &str,
        inputs: &[NodeId],
        output_shapes: &[crate::shape::Shape],
        output_dtypes: &[crate::dtype::DType],
        device: Device,
        attributes: HashMap<String, AttributeValue>,
    ) -> Result<Vec<NodeId>, TensorError> {
        // Create the operation node
        let op_node = self.add_node_auto_name(
            op_name,
            NodeType::Operation(op_name.to_string()),
            device,
            attributes,
        )?;

        // Connect inputs to the operation
        for (input_idx, &input_node) in inputs.iter().enumerate() {
            if !self.nodes.contains_key(&input_node) {
                return Err(TensorError::invalid_argument(format!(
                    "Input node {} not found",
                    input_node
                )));
            }

            self.add_edge(
                input_node,
                op_node,
                0, // Assume single output from input node
                input_idx,
                crate::dtype::DType::Float32, // Default - should be inferred
                crate::shape::Shape::new(vec![]),
                false,
            )?;
        }

        // Create output nodes if multiple outputs
        let mut output_nodes = vec![op_node];
        if output_shapes.len() > 1 {
            for (output_idx, (shape, dtype)) in output_shapes
                .iter()
                .zip(output_dtypes.iter())
                .enumerate()
                .skip(1)
            {
                let output_node_name = format!("{}_output_{}", op_name, output_idx);
                let output_node = self.add_node(
                    output_node_name,
                    NodeType::Operation("Identity".to_string()),
                    device,
                    HashMap::new(),
                )?;

                self.add_edge(
                    op_node,
                    output_node,
                    output_idx,
                    0,
                    *dtype,
                    shape.clone(),
                    false,
                )?;

                output_nodes.push(output_node);
            }
        }

        Ok(output_nodes)
    }

    /// Insert a new node between two existing connected nodes
    pub fn insert_node_between(
        &mut self,
        from_node: NodeId,
        to_node: NodeId,
        new_node_name: String,
        new_node_type: NodeType,
        device: Device,
        attributes: HashMap<String, AttributeValue>,
    ) -> Result<NodeId, TensorError> {
        // Find the edge to replace
        let edge_to_replace = self
            .edges
            .values()
            .find(|edge| edge.from_node == from_node && edge.to_node == to_node && !edge.is_control)
            .cloned();

        let edge = edge_to_replace.ok_or_else(|| {
            TensorError::invalid_argument(format!(
                "No data edge found between nodes {} and {}",
                from_node, to_node
            ))
        })?;

        // Create the new node
        let new_node = self.add_node(new_node_name, new_node_type, device, attributes)?;

        // Remove the original edge
        self.remove_edge(edge.id)?;

        // Create new edges: from_node -> new_node -> to_node
        self.add_edge(
            from_node,
            new_node,
            edge.from_output,
            0,
            edge.dtype,
            edge.shape.clone(),
            false,
        )?;

        self.add_edge(
            new_node,
            to_node,
            0,
            edge.to_input,
            edge.dtype,
            edge.shape,
            false,
        )?;

        Ok(new_node)
    }

    /// Replace a node with a subgraph
    pub fn replace_node_with_subgraph(
        &mut self,
        node_to_replace: NodeId,
        replacement_graph: &Graph,
        input_mapping: &HashMap<usize, NodeId>, // input_index -> replacement_node_id
        output_mapping: &HashMap<usize, NodeId>, // output_index -> replacement_node_id
    ) -> Result<HashMap<NodeId, NodeId>, TensorError> {
        let node = self
            .nodes
            .get(&node_to_replace)
            .ok_or_else(|| {
                TensorError::invalid_argument(format!("Node {} not found", node_to_replace))
            })?
            .clone();

        // Store incoming and outgoing edges
        let incoming_edges: Vec<_> = node
            .inputs
            .iter()
            .filter_map(|&edge_id| self.edges.get(&edge_id))
            .cloned()
            .collect();

        let outgoing_edges: Vec<_> = node
            .outputs
            .iter()
            .filter_map(|&edge_id| self.edges.get(&edge_id))
            .cloned()
            .collect();

        // Remove the original node
        self.remove_node(node_to_replace)?;

        // Add the replacement graph
        let id_mapping = self.extend_with_graph(replacement_graph, Some(&node.name))?;

        // Reconnect incoming edges
        for edge in incoming_edges {
            if let Some(&replacement_input) = input_mapping.get(&edge.to_input) {
                if let Some(&mapped_node) = id_mapping.get(&replacement_input) {
                    self.add_edge(
                        edge.from_node,
                        mapped_node,
                        edge.from_output,
                        0, // Connect to first input of replacement node
                        edge.dtype,
                        edge.shape,
                        edge.is_control,
                    )?;
                }
            }
        }

        // Reconnect outgoing edges
        for edge in outgoing_edges {
            if let Some(&replacement_output) = output_mapping.get(&edge.from_output) {
                if let Some(&mapped_node) = id_mapping.get(&replacement_output) {
                    self.add_edge(
                        mapped_node,
                        edge.to_node,
                        0, // Connect from first output of replacement node
                        edge.to_input,
                        edge.dtype,
                        edge.shape,
                        edge.is_control,
                    )?;
                }
            }
        }

        Ok(id_mapping)
    }
}
