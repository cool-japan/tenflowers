//! Core Graph Types and Basic Operations
//!
//! This module contains the fundamental data structures for the computation graph
//! and basic operations for creating and accessing graph elements.

use crate::{device::Device, dtype::DType, error::TensorError, shape::Shape, tensor::Tensor};
use std::collections::HashMap;

/// Unique identifier for nodes in the graph
pub type NodeId = u64;

/// Unique identifier for edges in the graph
pub type EdgeId = u64;

/// Graph node representing an operation, variable, placeholder, or constant
#[derive(Clone, Debug)]
pub struct GraphNode {
    pub id: NodeId,
    pub name: String,
    pub op_type: NodeType,
    pub device: Device,
    pub inputs: Vec<EdgeId>,
    pub outputs: Vec<EdgeId>,
    pub attributes: HashMap<String, AttributeValue>,
}

/// Types of nodes in the computation graph
#[derive(Clone, Debug, PartialEq)]
pub enum NodeType {
    /// Operation node that performs computation
    Operation(String), // op name
    /// Variable node that holds mutable state
    Variable { dtype: DType, shape: Shape },
    /// Placeholder node for feeding inputs
    Placeholder { dtype: DType, shape: Shape },
    /// Constant node with fixed value
    Constant,
}

/// Edge representing data flow between nodes
#[derive(Clone, Debug)]
pub struct GraphEdge {
    pub id: EdgeId,
    pub from_node: NodeId,
    pub to_node: NodeId,
    pub from_output: usize, // output index from source node
    pub to_input: usize,    // input index to destination node
    pub dtype: DType,
    pub shape: Shape,
    pub is_control: bool, // true for control dependencies
}

/// Attribute values that can be attached to nodes
#[derive(Clone, Debug)]
pub enum AttributeValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    IntList(Vec<i64>),
    FloatList(Vec<f64>),
    Shape(Shape),
    Tensor(Tensor<f32>), // For constants
}

/// Main computation graph structure
#[derive(Debug, Clone)]
pub struct Graph {
    pub(crate) nodes: HashMap<NodeId, GraphNode>,
    pub(crate) edges: HashMap<EdgeId, GraphEdge>,
    pub(crate) next_node_id: NodeId,
    pub(crate) next_edge_id: EdgeId,
    pub(crate) name_to_node: HashMap<String, NodeId>,
    pub(crate) topological_order: Option<Vec<NodeId>>,
    pub(crate) version: u64,
}

impl Graph {
    /// Create a new empty graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            next_node_id: 0,
            next_edge_id: 0,
            name_to_node: HashMap::new(),
            topological_order: None,
            version: 0,
        }
    }

    /// Add a new node to the graph
    pub fn add_node(
        &mut self,
        name: String,
        op_type: NodeType,
        device: Device,
        attributes: HashMap<String, AttributeValue>,
    ) -> Result<NodeId, TensorError> {
        // Ensure unique names
        if self.name_to_node.contains_key(&name) {
            return Err(TensorError::invalid_argument(format!(
                "Node name '{name}' already exists"
            )));
        }

        let node_id = self.next_node_id;
        self.next_node_id += 1;

        let node = GraphNode {
            id: node_id,
            name: name.clone(),
            op_type,
            device,
            inputs: Vec::new(),
            outputs: Vec::new(),
            attributes,
        };

        self.nodes.insert(node_id, node);
        self.name_to_node.insert(name, node_id);
        self.topological_order = None; // Invalidate cached order
        self.version += 1;

        Ok(node_id)
    }

    /// Add a new edge to the graph
    #[allow(clippy::too_many_arguments)]
    pub fn add_edge(
        &mut self,
        from_node: NodeId,
        to_node: NodeId,
        from_output: usize,
        to_input: usize,
        dtype: DType,
        shape: Shape,
        is_control: bool,
    ) -> Result<EdgeId, TensorError> {
        // Validate nodes exist
        if !self.nodes.contains_key(&from_node) {
            return Err(TensorError::invalid_argument(format!(
                "Source node {from_node} not found"
            )));
        }
        if !self.nodes.contains_key(&to_node) {
            return Err(TensorError::invalid_argument(format!(
                "Destination node {to_node} not found"
            )));
        }

        // Check for cycles (simplified - just direct cycle)
        if from_node == to_node {
            return Err(TensorError::invalid_argument(
                "Self-loops are not allowed".to_string(),
            ));
        }

        let edge_id = self.next_edge_id;
        self.next_edge_id += 1;

        let edge = GraphEdge {
            id: edge_id,
            from_node,
            to_node,
            from_output,
            to_input,
            dtype,
            shape,
            is_control,
        };

        self.edges.insert(edge_id, edge);

        // Update node edge lists
        self.nodes
            .get_mut(&from_node)
            .unwrap()
            .outputs
            .push(edge_id);
        self.nodes.get_mut(&to_node).unwrap().inputs.push(edge_id);

        self.topological_order = None; // Invalidate cached order
        self.version += 1;

        Ok(edge_id)
    }

    /// Get a node by ID
    pub fn get_node(&self, node_id: NodeId) -> Option<&GraphNode> {
        self.nodes.get(&node_id)
    }

    /// Get a node by name
    pub fn get_node_by_name(&self, name: &str) -> Option<&GraphNode> {
        self.name_to_node
            .get(name)
            .and_then(|&id| self.nodes.get(&id))
    }

    /// Get an edge by ID
    pub fn get_edge(&self, edge_id: EdgeId) -> Option<&GraphEdge> {
        self.edges.get(&edge_id)
    }

    /// Iterate over all nodes
    pub fn nodes(&self) -> impl Iterator<Item = &GraphNode> {
        self.nodes.values()
    }

    /// Iterate over all edges
    pub fn edges(&self) -> impl Iterator<Item = &GraphEdge> {
        self.edges.values()
    }

    /// Get a mutable reference to a node
    pub fn get_node_mut(&mut self, node_id: NodeId) -> Option<&mut GraphNode> {
        self.nodes.get_mut(&node_id)
    }

    /// Get a mutable reference to an edge
    pub fn get_edge_mut(&mut self, edge_id: EdgeId) -> Option<&mut GraphEdge> {
        self.edges.get_mut(&edge_id)
    }

    /// Get the number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
