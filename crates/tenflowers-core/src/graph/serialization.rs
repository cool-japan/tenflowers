//! Graph Serialization and Persistence
//!
//! This module provides functionality for serializing and deserializing graphs
//! for persistence, transfer, and interoperability.

use super::core::*;
use crate::error::TensorError;
use std::collections::HashMap;

/// Serializable representation of a graph for persistence and transfer
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphDef {
    pub nodes: Vec<NodeDef>,
    pub edges: Vec<EdgeDef>,
    pub version: u64,
}

/// Serializable representation of a graph node
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct NodeDef {
    pub id: NodeId,
    pub name: String,
    pub op_type: String,
    pub device: String,
    pub attributes: HashMap<String, AttributeValueDef>,
}

/// Serializable representation of a graph edge
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct EdgeDef {
    pub id: EdgeId,
    pub from_node: NodeId,
    pub to_node: NodeId,
    pub from_output: usize,
    pub to_input: usize,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub is_control: bool,
}

/// Serializable representation of attribute values
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum AttributeValueDef {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    IntList(Vec<i64>),
    FloatList(Vec<f64>),
    Shape(Vec<usize>),
    Tensor(Vec<f32>), // Flattened tensor data
}

impl From<AttributeValue> for AttributeValueDef {
    fn from(value: AttributeValue) -> Self {
        match value {
            AttributeValue::String(s) => AttributeValueDef::String(s),
            AttributeValue::Int(i) => AttributeValueDef::Int(i),
            AttributeValue::Float(f) => AttributeValueDef::Float(f),
            AttributeValue::Bool(b) => AttributeValueDef::Bool(b),
            AttributeValue::IntList(list) => AttributeValueDef::IntList(list),
            AttributeValue::FloatList(list) => AttributeValueDef::FloatList(list),
            AttributeValue::Shape(shape) => AttributeValueDef::Shape(shape.dims().to_vec()),
            AttributeValue::Tensor(tensor) => {
                // Flatten tensor data for serialization
                let data = tensor.as_slice().unwrap_or(&[]).to_vec();
                AttributeValueDef::Tensor(data)
            }
        }
    }
}

impl TryFrom<AttributeValueDef> for AttributeValue {
    type Error = TensorError;

    fn try_from(def: AttributeValueDef) -> Result<Self, Self::Error> {
        match def {
            AttributeValueDef::String(s) => Ok(AttributeValue::String(s)),
            AttributeValueDef::Int(i) => Ok(AttributeValue::Int(i)),
            AttributeValueDef::Float(f) => Ok(AttributeValue::Float(f)),
            AttributeValueDef::Bool(b) => Ok(AttributeValue::Bool(b)),
            AttributeValueDef::IntList(list) => Ok(AttributeValue::IntList(list)),
            AttributeValueDef::FloatList(list) => Ok(AttributeValue::FloatList(list)),
            AttributeValueDef::Shape(dims) => {
                Ok(AttributeValue::Shape(crate::shape::Shape::new(dims)))
            }
            AttributeValueDef::Tensor(data) => {
                // Reconstruct tensor from flattened data
                // Note: This is simplified - in practice we'd need to store shape info
                use crate::tensor::Tensor;
                let shape = vec![data.len()];
                let tensor = Tensor::from_vec(data, &shape)?;
                Ok(AttributeValue::Tensor(tensor))
            }
        }
    }
}

impl Graph {
    /// Convert graph to serializable format
    pub fn to_graph_def(&self) -> GraphDef {
        let nodes = self
            .nodes
            .values()
            .map(|node| NodeDef {
                id: node.id,
                name: node.name.clone(),
                op_type: match &node.op_type {
                    NodeType::Operation(op) => op.clone(),
                    NodeType::Variable { dtype, shape: _ } => format!("Variable:{:?}", dtype),
                    NodeType::Placeholder { dtype, shape: _ } => format!("Placeholder:{:?}", dtype),
                    NodeType::Constant => "Constant".to_string(),
                },
                device: format!("{:?}", node.device),
                attributes: node
                    .attributes
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone().into()))
                    .collect(),
            })
            .collect();

        let edges = self
            .edges
            .values()
            .map(|edge| EdgeDef {
                id: edge.id,
                from_node: edge.from_node,
                to_node: edge.to_node,
                from_output: edge.from_output,
                to_input: edge.to_input,
                dtype: format!("{:?}", edge.dtype),
                shape: edge.shape.dims().to_vec(),
                is_control: edge.is_control,
            })
            .collect();

        GraphDef {
            nodes,
            edges,
            version: self.version,
        }
    }

    /// Create graph from serializable format
    pub fn from_graph_def(graph_def: &GraphDef) -> Result<Self, TensorError> {
        let mut graph = Graph::new();
        let mut id_mapping: HashMap<NodeId, NodeId> = HashMap::new();

        // Add all nodes
        for node_def in &graph_def.nodes {
            let op_type = if node_def.op_type.starts_with("Variable:") {
                NodeType::Variable {
                    dtype: crate::dtype::DType::Float32, // Simplified
                    shape: crate::shape::Shape::new(vec![]),
                }
            } else if node_def.op_type.starts_with("Placeholder:") {
                NodeType::Placeholder {
                    dtype: crate::dtype::DType::Float32, // Simplified
                    shape: crate::shape::Shape::new(vec![]),
                }
            } else if node_def.op_type == "Constant" {
                NodeType::Constant
            } else {
                NodeType::Operation(node_def.op_type.clone())
            };

            let device = crate::device::Device::Cpu; // Simplified

            let attributes: Result<HashMap<String, AttributeValue>, TensorError> = node_def
                .attributes
                .iter()
                .map(|(k, v)| Ok((k.clone(), v.clone().try_into()?)))
                .collect();

            let new_id = graph.add_node(node_def.name.clone(), op_type, device, attributes?)?;

            id_mapping.insert(node_def.id, new_id);
        }

        // Add all edges
        for edge_def in &graph_def.edges {
            let from_node = *id_mapping.get(&edge_def.from_node).ok_or_else(|| {
                TensorError::invalid_argument(format!(
                    "Node {} not found in mapping",
                    edge_def.from_node
                ))
            })?;

            let to_node = *id_mapping.get(&edge_def.to_node).ok_or_else(|| {
                TensorError::invalid_argument(format!(
                    "Node {} not found in mapping",
                    edge_def.to_node
                ))
            })?;

            graph.add_edge(
                from_node,
                to_node,
                edge_def.from_output,
                edge_def.to_input,
                crate::dtype::DType::Float32, // Simplified
                crate::shape::Shape::new(edge_def.shape.clone()),
                edge_def.is_control,
            )?;
        }

        graph.version = graph_def.version;
        Ok(graph)
    }

    /// Save graph to file
    #[cfg(feature = "serialize")]
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), TensorError> {
        let graph_def = self.to_graph_def();
        let serialized = bincode::serialize(&graph_def)
            .map_err(|e| TensorError::invalid_argument(format!("Serialization failed: {}", e)))?;

        std::fs::write(path, serialized)
            .map_err(|e| TensorError::invalid_argument(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Load graph from file
    #[cfg(feature = "serialize")]
    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, TensorError> {
        let data = std::fs::read(path)
            .map_err(|e| TensorError::invalid_argument(format!("Failed to read file: {}", e)))?;

        let graph_def: GraphDef = bincode::deserialize(&data)
            .map_err(|e| TensorError::invalid_argument(format!("Deserialization failed: {}", e)))?;

        Self::from_graph_def(&graph_def)
    }

    /// Convert graph to JSON string
    #[cfg(feature = "serialize")]
    pub fn to_json(&self) -> Result<String, TensorError> {
        let graph_def = self.to_graph_def();
        serde_json::to_string_pretty(&graph_def)
            .map_err(|e| TensorError::invalid_argument(format!("JSON serialization failed: {}", e)))
    }

    /// Create graph from JSON string
    #[cfg(feature = "serialize")]
    pub fn from_json(json: &str) -> Result<Self, TensorError> {
        let graph_def: GraphDef = serde_json::from_str(json).map_err(|e| {
            TensorError::invalid_argument(format!("JSON deserialization failed: {}", e))
        })?;

        Self::from_graph_def(&graph_def)
    }
}
