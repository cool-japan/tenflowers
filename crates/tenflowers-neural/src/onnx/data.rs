//! ONNX Data Structures
//!
//! This module contains the core ONNX data structures including graphs,
//! nodes, value info, tensors, and attributes that represent the fundamental
//! building blocks of ONNX models.

use super::types::OnnxDataType;
use std::collections::HashMap;
use tenflowers_core::Result;

#[cfg(feature = "onnx")]
use super::proto;

/// Simplified ONNX graph representation
#[derive(Debug, Clone)]
pub struct OnnxGraph {
    pub name: String,
    pub nodes: Vec<OnnxNode>,
    pub inputs: Vec<OnnxValueInfo>,
    pub outputs: Vec<OnnxValueInfo>,
    pub initializers: Vec<OnnxTensor>,
}

#[cfg(feature = "onnx")]
impl OnnxGraph {
    /// Convert to protobuf format
    pub fn to_protobuf(&self) -> Result<proto::GraphProto> {
        let nodes = self
            .nodes
            .iter()
            .map(|n| n.to_protobuf())
            .collect::<Result<Vec<_>>>()?;

        let inputs = self
            .inputs
            .iter()
            .map(|i| i.to_protobuf())
            .collect::<Result<Vec<_>>>()?;

        let outputs = self
            .outputs
            .iter()
            .map(|o| o.to_protobuf())
            .collect::<Result<Vec<_>>>()?;

        let initializers = self
            .initializers
            .iter()
            .map(|init| init.to_protobuf())
            .collect::<Result<Vec<_>>>()?;

        Ok(proto::GraphProto {
            node: nodes,
            name: Some(self.name.clone()),
            initializer: initializers,
            input: inputs,
            output: outputs,
            value_info: Vec::new(), // Optional value info
        })
    }

    /// Convert from protobuf format
    pub fn from_protobuf(proto: &proto::GraphProto) -> Result<Self> {
        let nodes = proto
            .node
            .iter()
            .map(OnnxNode::from_protobuf)
            .collect::<Result<Vec<_>>>()?;

        let inputs = proto
            .input
            .iter()
            .map(OnnxValueInfo::from_protobuf)
            .collect::<Result<Vec<_>>>()?;

        let outputs = proto
            .output
            .iter()
            .map(OnnxValueInfo::from_protobuf)
            .collect::<Result<Vec<_>>>()?;

        let initializers = proto
            .initializer
            .iter()
            .map(OnnxTensor::from_protobuf)
            .collect::<Result<Vec<_>>>()?;

        Ok(OnnxGraph {
            name: proto.name.clone().unwrap_or_else(|| "graph".to_string()),
            nodes,
            inputs,
            outputs,
            initializers,
        })
    }
}

/// ONNX node representation
#[derive(Debug, Clone)]
pub struct OnnxNode {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, OnnxAttribute>,
}

#[cfg(feature = "onnx")]
impl OnnxNode {
    /// Convert to protobuf format
    pub fn to_protobuf(&self) -> Result<proto::NodeProto> {
        let attributes = self
            .attributes
            .iter()
            .map(|(k, v)| v.to_protobuf(k))
            .collect::<Result<Vec<_>>>()?;

        Ok(proto::NodeProto {
            input: self.inputs.clone(),
            output: self.outputs.clone(),
            name: Some(self.name.clone()),
            op_type: Some(self.op_type.clone()),
            attribute: attributes,
        })
    }

    /// Convert from protobuf format
    pub fn from_protobuf(proto: &proto::NodeProto) -> Result<Self> {
        let mut attributes = HashMap::new();
        for attr in &proto.attribute {
            if let Some(name) = &attr.name {
                attributes.insert(name.clone(), OnnxAttribute::from_protobuf(attr)?);
            }
        }

        Ok(OnnxNode {
            name: proto.name.clone().unwrap_or_else(|| "node".to_string()),
            op_type: proto
                .op_type
                .clone()
                .unwrap_or_else(|| "Unknown".to_string()),
            inputs: proto.input.clone(),
            outputs: proto.output.clone(),
            attributes,
        })
    }
}

impl OnnxNode {
    /// Create a new ONNX node
    pub fn new(name: String, op_type: String, inputs: Vec<String>, outputs: Vec<String>) -> Self {
        Self {
            name,
            op_type,
            inputs,
            outputs,
            attributes: HashMap::new(),
        }
    }
}

/// ONNX value info (for inputs/outputs)
#[derive(Debug, Clone)]
pub struct OnnxValueInfo {
    pub name: String,
    pub elem_type: OnnxDataType,
    pub shape: Vec<i64>,
}

#[cfg(feature = "onnx")]
impl OnnxValueInfo {
    /// Convert to protobuf format
    pub fn to_protobuf(&self) -> Result<proto::ValueInfoProto> {
        use proto::tensor_shape_proto::{dimension::Value as DimValue, Dimension};
        use proto::type_proto::{Tensor as TypeTensor, Value as TypeValue};

        // Create tensor shape
        let dims = self
            .shape
            .iter()
            .map(|&dim| Dimension {
                value: Some(DimValue::DimValue(dim)),
            })
            .collect();

        let tensor_shape = proto::TensorShapeProto { dim: dims };

        // Create tensor type
        let tensor_type = TypeTensor {
            elem_type: Some(self.elem_type as i32),
            shape: Some(tensor_shape),
        };

        // Create type proto
        let type_proto = proto::TypeProto {
            value: Some(TypeValue::TensorType(tensor_type)),
        };

        Ok(proto::ValueInfoProto {
            name: Some(self.name.clone()),
            r#type: Some(type_proto),
            doc_string: None,
        })
    }

    /// Convert from protobuf format
    pub fn from_protobuf(proto: &proto::ValueInfoProto) -> Result<Self> {
        // Extract shape and type information from the complex TypeProto structure
        // This is simplified - in a full implementation you'd need to traverse the type structure

        Ok(OnnxValueInfo {
            name: proto.name.clone().unwrap_or_else(|| "value".to_string()),
            elem_type: OnnxDataType::Float32, // Default type for simplification
            shape: vec![],                    // Simplified - would need to extract from TypeProto
        })
    }
}

impl OnnxValueInfo {
    /// Create a new ONNX value info
    pub fn new(name: String, elem_type: OnnxDataType, shape: Vec<i64>) -> Self {
        Self {
            name,
            elem_type,
            shape,
        }
    }
}

/// ONNX tensor (for weights/initializers)
#[derive(Debug, Clone)]
pub struct OnnxTensor {
    pub name: String,
    pub data_type: OnnxDataType,
    pub dims: Vec<i64>,
    pub raw_data: Vec<u8>,
}

#[cfg(feature = "onnx")]
impl OnnxTensor {
    /// Convert to protobuf format
    pub fn to_protobuf(&self) -> Result<proto::TensorProto> {
        Ok(proto::TensorProto {
            dims: self.dims.clone(),
            data_type: Some(self.data_type as i32),
            name: Some(self.name.clone()),
            raw_data: Some(self.raw_data.clone()),
            float_data: Vec::new(),
            int32_data: Vec::new(),
            int64_data: Vec::new(),
        })
    }

    /// Convert from protobuf format
    pub fn from_protobuf(proto: &proto::TensorProto) -> Result<Self> {
        let data_type = match proto.data_type.unwrap_or(1) {
            1 => OnnxDataType::Float32,
            6 => OnnxDataType::Int32,
            7 => OnnxDataType::Int64,
            11 => OnnxDataType::Float64,
            _ => OnnxDataType::Float32, // Default fallback
        };

        let raw_data = proto.raw_data.clone().unwrap_or_default();

        Ok(OnnxTensor {
            name: proto.name.clone().unwrap_or_else(|| "tensor".to_string()),
            data_type,
            dims: proto.dims.clone(),
            raw_data,
        })
    }
}

impl OnnxTensor {
    /// Create a new ONNX tensor
    pub fn new(name: String, data_type: OnnxDataType, dims: Vec<i64>, raw_data: Vec<u8>) -> Self {
        Self {
            name,
            data_type,
            dims,
            raw_data,
        }
    }
}

/// ONNX attribute value
#[derive(Debug, Clone)]
pub enum OnnxAttribute {
    Float(f32),
    Int(i64),
    String(String),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
}

#[cfg(feature = "onnx")]
impl OnnxAttribute {
    /// Convert to protobuf format
    pub fn to_protobuf(&self, name: &str) -> Result<proto::AttributeProto> {
        let mut attr = proto::AttributeProto {
            name: Some(name.to_string()),
            doc_string: None,
            r#type: None,
            f: None,
            i: None,
            s: None,
            floats: Vec::new(),
            ints: Vec::new(),
            strings: Vec::new(),
        };

        match self {
            OnnxAttribute::Float(value) => {
                attr.f = Some(*value);
                attr.r#type = Some(proto::AttributeType::Float as i32);
            }
            OnnxAttribute::Int(value) => {
                attr.i = Some(*value);
                attr.r#type = Some(proto::AttributeType::Int as i32);
            }
            OnnxAttribute::String(value) => {
                attr.s = Some(value.as_bytes().to_vec());
                attr.r#type = Some(proto::AttributeType::String as i32);
            }
            OnnxAttribute::Floats(values) => {
                attr.floats = values.clone();
                attr.r#type = Some(proto::AttributeType::Floats as i32);
            }
            OnnxAttribute::Ints(values) => {
                attr.ints = values.clone();
                attr.r#type = Some(proto::AttributeType::Ints as i32);
            }
            OnnxAttribute::Strings(values) => {
                attr.strings = values.iter().map(|s| s.as_bytes().to_vec()).collect();
                attr.r#type = Some(proto::AttributeType::Strings as i32);
            }
        }

        Ok(attr)
    }

    /// Convert from protobuf format
    pub fn from_protobuf(proto: &proto::AttributeProto) -> Result<Self> {
        match proto.r#type.unwrap_or(0) {
            x if x == proto::AttributeType::Float as i32 => {
                Ok(OnnxAttribute::Float(proto.f.unwrap_or(0.0)))
            }
            x if x == proto::AttributeType::Int as i32 => {
                Ok(OnnxAttribute::Int(proto.i.unwrap_or(0)))
            }
            x if x == proto::AttributeType::String as i32 => {
                let s = proto
                    .s
                    .as_ref()
                    .map(|bytes| String::from_utf8_lossy(bytes).to_string())
                    .unwrap_or_default();
                Ok(OnnxAttribute::String(s))
            }
            x if x == proto::AttributeType::Floats as i32 => {
                Ok(OnnxAttribute::Floats(proto.floats.clone()))
            }
            x if x == proto::AttributeType::Ints as i32 => {
                Ok(OnnxAttribute::Ints(proto.ints.clone()))
            }
            x if x == proto::AttributeType::Strings as i32 => {
                let strings = proto
                    .strings
                    .iter()
                    .map(|bytes| String::from_utf8_lossy(bytes).to_string())
                    .collect();
                Ok(OnnxAttribute::Strings(strings))
            }
            _ => Ok(OnnxAttribute::Float(0.0)), // Default fallback
        }
    }
}
