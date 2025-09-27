//! Sequential Model ONNX Implementation
//!
//! This module provides ONNX export and import implementations specifically
//! for Sequential models, including layer conversion and model graph construction.

use super::data::{OnnxAttribute, OnnxGraph, OnnxNode, OnnxValueInfo};
use super::model::OnnxModel;
use super::traits::{OnnxExport, OnnxImport};
use super::types::OnnxDataType;
use crate::{layers::LayerType, Sequential};
use std::collections::HashMap;
use tenflowers_core::{Result, TensorError};

impl<T> OnnxExport<T> for Sequential<T>
where
    T: Clone + Default,
{
    fn to_onnx(&self, input_shape: &[usize]) -> Result<OnnxModel> {
        let mut nodes = Vec::new();
        let mut current_shape = input_shape.to_vec();
        let mut value_counter = 0;

        // Input value info
        let input_name = format!("input_{}", value_counter);
        value_counter += 1;
        let mut current_output = input_name.clone();

        let inputs = vec![OnnxValueInfo::new(
            input_name,
            OnnxDataType::Float32,
            current_shape.iter().map(|&x| x as i64).collect(),
        )];

        // Convert each layer to ONNX nodes
        for (layer_idx, layer) in self.layers().iter().enumerate() {
            let layer_name = format!("layer_{}", layer_idx);
            let output_name = format!("output_{}", value_counter);
            value_counter += 1;

            let (node, output_shape) = layer_to_onnx_node(
                &layer.layer_type(),
                &layer_name,
                &current_output,
                &output_name,
                &current_shape,
            )?;
            nodes.push(node);
            current_output = output_name;
            current_shape = output_shape;
        }

        // Output value info
        let outputs = vec![OnnxValueInfo::new(
            current_output,
            OnnxDataType::Float32,
            current_shape.iter().map(|&x| x as i64).collect(),
        )];

        let graph = OnnxGraph {
            name: "sequential_model".to_string(),
            nodes,
            inputs,
            outputs,
            initializers: Vec::new(), // Simplified - would need to include weights
        };

        Ok(OnnxModel::new(graph))
    }
}

impl<T> OnnxImport<T> for Sequential<T>
where
    T: Clone + Default,
{
    fn from_onnx(onnx_model: &OnnxModel) -> Result<Self> {
        // Simplified implementation
        // In a full implementation, this would parse the ONNX graph and reconstruct layers

        // For now, create an empty sequential model
        Ok(Sequential::new(Vec::new()))
    }
}

/// Convert a layer to an ONNX node
fn layer_to_onnx_node(
    layer: &LayerType,
    node_name: &str,
    input_name: &str,
    output_name: &str,
    input_shape: &[usize],
) -> Result<(OnnxNode, Vec<usize>)> {
    match layer {
        LayerType::Dense => {
            let mut attributes = HashMap::new();

            // TODO: Add weight and bias information as attributes when layer data is available
            // For now, this is a stub implementation

            let node = OnnxNode {
                name: node_name.to_string(),
                op_type: "MatMul".to_string(), // Dense layer is essentially a matrix multiplication
                inputs: vec![input_name.to_string()],
                outputs: vec![output_name.to_string()],
                attributes,
            };

            // Calculate output shape (simplified stub)
            let output_shape = if input_shape.len() >= 2 {
                let mut shape = input_shape.to_vec();
                // TODO: Use actual output features when layer data is available
                let last_idx = shape.len() - 1;
                shape[last_idx] = 128; // Default stub value
                shape
            } else {
                vec![128] // Default stub value
            };

            Ok((node, output_shape))
        }
        _ => {
            // For unsupported layers, create a generic node
            let node = OnnxNode {
                name: node_name.to_string(),
                op_type: "Identity".to_string(), // Pass-through
                inputs: vec![input_name.to_string()],
                outputs: vec![output_name.to_string()],
                attributes: HashMap::new(),
            };

            Ok((node, input_shape.to_vec()))
        }
    }
}
