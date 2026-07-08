//! Bidirectional conversions between the hand-rolled `proto::*` protobuf
//! schema (the ONNX wire format) and TenfloweRS's own in-memory
//! `onnx_interop::types::Onnx*` structures.
//!
//! This module is where `ModelProto::decode` / `.encode_to_vec()` are
//! actually invoked (via the `to_protobuf` / `from_protobuf` inherent
//! methods added to each `types::Onnx*` struct below), mirroring the proven
//! pattern already used at `tenflowers-neural::onnx::{model, data}` but
//! adapted to `tenflowers-core`'s richer type layer, which additionally
//! models `value_info`, dynamic dimensions, and tensor-valued attributes
//! for real instead of stubbing them out.
//!
//! Only compiled when the `onnx` feature is enabled.
//!
//! ## Honest-error conventions
//!
//! Every decode path here is written to fail loudly on data it does not
//! understand rather than silently substituting a plausible-looking
//! default: unrecognized ONNX attribute-type codes and tensor data-type
//! codes are rejected with a descriptive `Err`, never coerced to
//! `Float(0.0)` or `Float32` (the two silently-wrong-mapping bugs present
//! in the `tenflowers-neural` reference implementation this module
//! replaces). See `data_type_from_i32` and `OnnxAttribute::from_protobuf`.

use super::proto;
use super::types::{
    OnnxAttribute, OnnxDataType, OnnxDimension, OnnxGraph, OnnxModel, OnnxModelMetadata, OnnxNode,
    OnnxOpsetImport, OnnxTensor, OnnxTensorShapeProto, OnnxTensorTypeProto, OnnxTypeProto,
    OnnxValueInfo,
};
use crate::{Result, TensorError};
use prost::Message;
use std::collections::HashMap;

/// Metadata-property key used to preserve `ModelProto.ir_version` across a
/// decode/re-encode round trip. `types::OnnxModel` has no dedicated
/// `ir_version` field (it is not otherwise part of TenfloweRS's in-memory
/// model representation), so it is stashed in the general-purpose
/// `metadata.metadata_props` map instead of being discarded silently.
const IR_VERSION_METADATA_KEY: &str = "onnx_ir_version";

/// Fallback IR version used when exporting a model that was not itself
/// decoded from protobuf bytes (so has no stashed `onnx_ir_version`
/// metadata property). This is a reasonable, stable, valid ONNX IR version
/// number, not a precise function of the model's opset imports; a full
/// opset-to-IR-version compatibility table is out of scope for this slice.
const DEFAULT_IR_VERSION: i64 = 8;

/// Converts a wire-level ONNX tensor data-type code to the in-memory
/// `OnnxDataType` enum.
///
/// This is a *total* match over the full officially-defined range
/// (`0..=16`, `Undefined` through `BFloat16`); any other code is a genuine
/// decode error, never silently coerced to a default type (which would
/// mislabel the accompanying raw bytes and corrupt interpretation).
pub(super) fn data_type_from_i32(code: i32) -> Result<OnnxDataType> {
    match code {
        0 => Ok(OnnxDataType::Undefined),
        1 => Ok(OnnxDataType::Float),
        2 => Ok(OnnxDataType::Uint8),
        3 => Ok(OnnxDataType::Int8),
        4 => Ok(OnnxDataType::Uint16),
        5 => Ok(OnnxDataType::Int16),
        6 => Ok(OnnxDataType::Int32),
        7 => Ok(OnnxDataType::Int64),
        8 => Ok(OnnxDataType::String),
        9 => Ok(OnnxDataType::Bool),
        10 => Ok(OnnxDataType::Float16),
        11 => Ok(OnnxDataType::Double),
        12 => Ok(OnnxDataType::Uint32),
        13 => Ok(OnnxDataType::Uint64),
        14 => Ok(OnnxDataType::Complex64),
        15 => Ok(OnnxDataType::Complex128),
        16 => Ok(OnnxDataType::BFloat16),
        other => Err(TensorError::invalid_argument(format!(
            "Unrecognized ONNX tensor data_type code {other}; valid codes are 0..=16 \
             (Undefined..BFloat16 per the ONNX TensorProto.DataType enum)"
        ))),
    }
}

impl OnnxModel {
    /// Serializes this model to the official ONNX `ModelProto` protobuf
    /// message shape. Callers that want raw bytes can `use prost::Message;`
    /// and call `.encode_to_vec()` on the result, or go through
    /// `OnnxExporter::export_to_bytes`, which additionally applies the
    /// exporter's configured opset version as a fallback when
    /// `self.opset_imports` is empty.
    pub fn to_protobuf(&self) -> Result<proto::ModelProto> {
        let graph = self.graph.to_protobuf()?;

        let ir_version = self
            .metadata
            .metadata_props
            .get(IR_VERSION_METADATA_KEY)
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(DEFAULT_IR_VERSION);

        let opset_import = self
            .opset_imports
            .iter()
            .map(|o| proto::OperatorSetIdProto {
                domain: Some(o.domain.clone()),
                version: Some(o.version),
            })
            .collect();

        Ok(proto::ModelProto {
            ir_version: Some(ir_version),
            opset_import,
            producer_name: Some(self.producer_name.clone()),
            producer_version: Some(self.producer_version.clone()),
            domain: Some(self.metadata.domain.clone()),
            model_version: Some(self.metadata.model_version),
            doc_string: Some(self.metadata.description.clone()),
            graph: Some(graph),
        })
    }

    /// Deserializes a model from raw ONNX protobuf bytes.
    pub fn from_protobuf(data: &[u8]) -> Result<Self> {
        let proto_model = proto::ModelProto::decode(data).map_err(|e| {
            TensorError::serialization_error_simple(format!("ONNX protobuf decode failed: {e}"))
        })?;

        let graph_proto = proto_model.graph.as_ref().ok_or_else(|| {
            TensorError::serialization_error_simple(
                "ONNX ModelProto is missing its required `graph` field".to_string(),
            )
        })?;
        let graph = OnnxGraph::from_protobuf(graph_proto)?;

        let mut metadata_props = HashMap::new();
        if let Some(ir_version) = proto_model.ir_version {
            metadata_props.insert(IR_VERSION_METADATA_KEY.to_string(), ir_version.to_string());
        }

        let opset_imports = proto_model
            .opset_import
            .iter()
            .map(|o| OnnxOpsetImport {
                domain: o.domain.clone().unwrap_or_default(),
                version: o.version.unwrap_or_default(),
            })
            .collect();

        Ok(OnnxModel {
            graph,
            metadata: OnnxModelMetadata {
                description: proto_model.doc_string.unwrap_or_default(),
                domain: proto_model.domain.unwrap_or_default(),
                model_version: proto_model.model_version.unwrap_or_default(),
                metadata_props,
            },
            opset_imports,
            producer_name: proto_model.producer_name.unwrap_or_default(),
            producer_version: proto_model.producer_version.unwrap_or_default(),
        })
    }
}

impl OnnxGraph {
    pub fn to_protobuf(&self) -> Result<proto::GraphProto> {
        let node = self
            .nodes
            .iter()
            .map(|n| n.to_protobuf())
            .collect::<Result<Vec<_>>>()?;
        let input = self
            .inputs
            .iter()
            .map(|i| i.to_protobuf())
            .collect::<Result<Vec<_>>>()?;
        let output = self
            .outputs
            .iter()
            .map(|o| o.to_protobuf())
            .collect::<Result<Vec<_>>>()?;
        let initializer = self
            .initializers
            .iter()
            .map(|t| t.to_protobuf())
            .collect::<Result<Vec<_>>>()?;
        // Unlike the tenflowers-neural reference (which always emits an
        // empty value_info list), this genuinely converts the graph's
        // intermediate-value type/shape information for real.
        let value_info = self
            .value_info
            .iter()
            .map(|v| v.to_protobuf())
            .collect::<Result<Vec<_>>>()?;

        Ok(proto::GraphProto {
            node,
            name: Some(self.name.clone()),
            initializer,
            value_info,
            input,
            output,
        })
    }

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
        let value_info = proto
            .value_info
            .iter()
            .map(OnnxValueInfo::from_protobuf)
            .collect::<Result<Vec<_>>>()?;

        Ok(OnnxGraph {
            nodes,
            inputs,
            outputs,
            initializers,
            value_info,
            name: proto.name.clone().unwrap_or_else(|| "graph".to_string()),
        })
    }
}

impl OnnxNode {
    pub fn to_protobuf(&self) -> Result<proto::NodeProto> {
        let attribute = self
            .attributes
            .iter()
            .map(|(k, v)| v.to_protobuf(k))
            .collect::<Result<Vec<_>>>()?;

        Ok(proto::NodeProto {
            input: self.inputs.clone(),
            output: self.outputs.clone(),
            name: Some(self.name.clone()),
            op_type: Some(self.op_type.clone()),
            attribute,
        })
    }

    pub fn from_protobuf(proto: &proto::NodeProto) -> Result<Self> {
        let mut attributes = HashMap::with_capacity(proto.attribute.len());
        for attr in &proto.attribute {
            // Unlike cosmetic fields such as `name`/`op_type` (which default
            // to an empty string when absent), an attribute with no name
            // cannot be stored in the `HashMap<String, OnnxAttribute>` at
            // all; silently skipping it (as the tenflowers-neural reference
            // implementation does) would silently drop real graph data, so
            // this is a genuine decode error instead.
            let key = attr.name.clone().ok_or_else(|| {
                TensorError::invalid_argument(
                    "ONNX node attribute is missing its required `name` field".to_string(),
                )
            })?;
            attributes.insert(key, OnnxAttribute::from_protobuf(attr)?);
        }

        Ok(OnnxNode {
            name: proto.name.clone().unwrap_or_default(),
            op_type: proto.op_type.clone().unwrap_or_default(),
            inputs: proto.input.clone(),
            outputs: proto.output.clone(),
            attributes,
        })
    }
}

impl OnnxValueInfo {
    pub fn to_protobuf(&self) -> Result<proto::ValueInfoProto> {
        use proto::tensor_shape_proto::{dimension::Value as DimValue, Dimension};
        use proto::type_proto::{Tensor as TypeTensor, Value as TypeValue};

        let r#type = match &self.value_type.tensor_type {
            Some(tensor_type) => {
                let dim = tensor_type
                    .shape
                    .dims
                    .iter()
                    .map(|d| {
                        let value = match d {
                            OnnxDimension::Value(v) => DimValue::DimValue(*v),
                            OnnxDimension::Param(s) => DimValue::DimParam(s.clone()),
                        };
                        Dimension { value: Some(value) }
                    })
                    .collect();

                let tensor = TypeTensor {
                    elem_type: Some(tensor_type.elem_type as i32),
                    shape: Some(proto::TensorShapeProto { dim }),
                };

                Some(proto::TypeProto {
                    value: Some(TypeValue::TensorType(tensor)),
                })
            }
            None => None,
        };

        Ok(proto::ValueInfoProto {
            name: Some(self.name.clone()),
            r#type,
            doc_string: None,
        })
    }

    /// Decodes a `ValueInfoProto`'s full `TypeProto` structure into
    /// `types::OnnxTypeProto`, genuinely walking
    /// `r#type.tensor_type.{elem_type, shape.dim[]}` instead of stubbing it
    /// out with a hardcoded `Float32`/`vec![]` default (as the
    /// `tenflowers-neural` reference implementation does). Each dimension's
    /// oneof is mapped faithfully: `Some(DimValue(v))` ->
    /// `OnnxDimension::Value(v)`, `Some(DimParam(s))` ->
    /// `OnnxDimension::Param(s)`. A dimension whose oneof is entirely unset
    /// (`None`) represents an ONNX "unknown dimension with no symbolic
    /// name"; this is documented here as `OnnxDimension::Param(String::new())`
    /// since `types::OnnxDimension` has no separate "unknown" variant.
    pub fn from_protobuf(proto: &proto::ValueInfoProto) -> Result<Self> {
        let value_type = match proto.r#type.as_ref().and_then(|t| t.value.as_ref()) {
            Some(proto::type_proto::Value::TensorType(tensor_type)) => {
                let elem_type = data_type_from_i32(tensor_type.elem_type.unwrap_or(0))?;

                // Absence of `shape` means "unranked" in the ONNX spec; both
                // "unranked" and "explicitly rank-0 (scalar)" are
                // represented as an empty `dims` vec here, since
                // `OnnxTensorShapeProto` has no separate unranked marker.
                let dims = tensor_type
                    .shape
                    .as_ref()
                    .map(|shape| {
                        shape
                            .dim
                            .iter()
                            .map(|dimension| match &dimension.value {
                                Some(proto::tensor_shape_proto::dimension::Value::DimValue(v)) => {
                                    OnnxDimension::Value(*v)
                                }
                                Some(proto::tensor_shape_proto::dimension::Value::DimParam(s)) => {
                                    OnnxDimension::Param(s.clone())
                                }
                                None => OnnxDimension::Param(String::new()),
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                OnnxTypeProto {
                    tensor_type: Some(OnnxTensorTypeProto {
                        elem_type,
                        shape: OnnxTensorShapeProto { dims },
                    }),
                }
            }
            _ => OnnxTypeProto { tensor_type: None },
        };

        Ok(OnnxValueInfo {
            name: proto.name.clone().unwrap_or_default(),
            value_type,
        })
    }
}

impl OnnxTensor {
    pub fn to_protobuf(&self) -> Result<proto::TensorProto> {
        Ok(proto::TensorProto {
            dims: self.dims.clone(),
            data_type: Some(self.data_type as i32),
            name: Some(self.name.clone()),
            raw_data: Some(self.data.clone()),
            float_data: Vec::new(),
            int32_data: Vec::new(),
            int64_data: Vec::new(),
        })
    }

    pub fn from_protobuf(proto: &proto::TensorProto) -> Result<Self> {
        // `0` (Undefined) is the ONNX spec's own documented sentinel for
        // "not specified"; using it here for an absent `data_type` field is
        // an honest reflection of the wire data, not a fabricated guess
        // (unlike silently assuming Float32 as the reference implementation
        // does).
        let data_type = data_type_from_i32(proto.data_type.unwrap_or(0))?;

        Ok(OnnxTensor {
            name: proto.name.clone().unwrap_or_default(),
            data_type,
            dims: proto.dims.clone(),
            data: proto.raw_data.clone().unwrap_or_default(),
        })
    }
}

impl OnnxAttribute {
    pub fn to_protobuf(&self, name: &str) -> Result<proto::AttributeProto> {
        let mut attr = proto::AttributeProto {
            name: Some(name.to_string()),
            doc_string: None,
            r#type: None,
            f: None,
            i: None,
            s: None,
            t: None,
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
            OnnxAttribute::Tensor(tensor) => {
                attr.t = Some(tensor.to_protobuf()?);
                attr.r#type = Some(proto::AttributeType::Tensor as i32);
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

    /// Decodes an `AttributeProto`. Every branch is a genuine `Err` when the
    /// declared type is missing, unrecognized, or its corresponding payload
    /// field is absent — never a fabricated `Float(0.0)` fallback (the bug
    /// present in `tenflowers-neural::onnx::data::OnnxAttribute::from_protobuf`,
    /// which this replaces).
    pub fn from_protobuf(proto: &proto::AttributeProto) -> Result<Self> {
        let attr_name = proto
            .name
            .clone()
            .unwrap_or_else(|| "<unnamed>".to_string());

        let ty = proto.r#type.ok_or_else(|| {
            TensorError::invalid_argument(format!(
                "ONNX attribute '{attr_name}' has no declared `type`; cannot decode"
            ))
        })?;

        if ty == proto::AttributeType::Float as i32 {
            let v = proto.f.ok_or_else(|| {
                TensorError::invalid_argument(format!(
                    "ONNX attribute '{attr_name}' declares type Float but field `f` is absent"
                ))
            })?;
            Ok(OnnxAttribute::Float(v))
        } else if ty == proto::AttributeType::Int as i32 {
            let v = proto.i.ok_or_else(|| {
                TensorError::invalid_argument(format!(
                    "ONNX attribute '{attr_name}' declares type Int but field `i` is absent"
                ))
            })?;
            Ok(OnnxAttribute::Int(v))
        } else if ty == proto::AttributeType::String as i32 {
            let bytes = proto.s.as_ref().ok_or_else(|| {
                TensorError::invalid_argument(format!(
                    "ONNX attribute '{attr_name}' declares type String but field `s` is absent"
                ))
            })?;
            Ok(OnnxAttribute::String(
                String::from_utf8_lossy(bytes).to_string(),
            ))
        } else if ty == proto::AttributeType::Tensor as i32 {
            let tensor_proto = proto.t.as_ref().ok_or_else(|| {
                TensorError::invalid_argument(format!(
                    "ONNX attribute '{attr_name}' declares type Tensor but field `t` is absent"
                ))
            })?;
            Ok(OnnxAttribute::Tensor(OnnxTensor::from_protobuf(
                tensor_proto,
            )?))
        } else if ty == proto::AttributeType::Floats as i32 {
            Ok(OnnxAttribute::Floats(proto.floats.clone()))
        } else if ty == proto::AttributeType::Ints as i32 {
            Ok(OnnxAttribute::Ints(proto.ints.clone()))
        } else if ty == proto::AttributeType::Strings as i32 {
            Ok(OnnxAttribute::Strings(
                proto
                    .strings
                    .iter()
                    .map(|b| String::from_utf8_lossy(b).to_string())
                    .collect(),
            ))
        } else {
            Err(TensorError::not_implemented_simple(format!(
                "ONNX attribute '{attr_name}' has type code {ty}, which is not supported for \
                 decoding by this crate (supported: Float=1, Int=2, String=3, Tensor=4, \
                 Floats=6, Ints=7, Strings=8; Graph/SparseTensor/TypeProto and their plural \
                 forms are out of scope for this slice)"
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_interop::{OnnxConfig, OnnxExporter, OnnxImporter};

    fn sample_model() -> OnnxModel {
        let mut relu_attributes = HashMap::new();
        relu_attributes.insert(
            "reference_tensor".to_string(),
            OnnxAttribute::Tensor(OnnxTensor {
                name: "ref_const".to_string(),
                data_type: OnnxDataType::Float,
                dims: vec![2],
                data: [1.5f32, -2.5f32]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            }),
        );

        let add_node = OnnxNode {
            name: "add1".to_string(),
            op_type: "Add".to_string(),
            inputs: vec!["input".to_string(), "bias".to_string()],
            outputs: vec!["added".to_string()],
            attributes: HashMap::new(),
        };
        let relu_node = OnnxNode {
            name: "relu1".to_string(),
            op_type: "Relu".to_string(),
            inputs: vec!["added".to_string()],
            outputs: vec!["output".to_string()],
            attributes: relu_attributes,
        };

        let input = OnnxValueInfo {
            name: "input".to_string(),
            value_type: OnnxTypeProto {
                tensor_type: Some(OnnxTensorTypeProto {
                    elem_type: OnnxDataType::Float,
                    shape: OnnxTensorShapeProto {
                        dims: vec![
                            OnnxDimension::Param("batch".to_string()),
                            OnnxDimension::Value(3),
                        ],
                    },
                }),
            },
        };
        let output = OnnxValueInfo {
            name: "output".to_string(),
            value_type: OnnxTypeProto {
                tensor_type: Some(OnnxTensorTypeProto {
                    elem_type: OnnxDataType::Float,
                    shape: OnnxTensorShapeProto {
                        dims: vec![OnnxDimension::Value(3)],
                    },
                }),
            },
        };

        let bias_initializer = OnnxTensor {
            name: "bias".to_string(),
            data_type: OnnxDataType::Float,
            dims: vec![3],
            data: [1.0f32, 2.0f32, 3.0f32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        };

        let graph = OnnxGraph {
            nodes: vec![add_node, relu_node],
            inputs: vec![input],
            outputs: vec![output],
            initializers: vec![bias_initializer],
            value_info: vec![],
            name: "test_graph".to_string(),
        };

        OnnxModel {
            graph,
            metadata: OnnxModelMetadata {
                description: "round trip test model".to_string(),
                domain: "ai.tenflowers.test".to_string(),
                model_version: 42,
                metadata_props: HashMap::new(),
            },
            opset_imports: vec![OnnxOpsetImport {
                domain: String::new(),
                version: 18,
            }],
            producer_name: "tenflowers-core-tests".to_string(),
            producer_version: "0.1.2".to_string(),
        }
    }

    #[test]
    fn model_round_trips_through_protobuf_bytes() {
        let model = sample_model();

        let proto_model = model.to_protobuf().expect("to_protobuf should succeed");
        let bytes = proto_model.encode_to_vec();

        // Prove the bytes are genuine ONNX wire format by decoding them
        // with the raw prost struct directly, independent of our own
        // `from_protobuf` convenience wrapper.
        let redecoded = proto::ModelProto::decode(bytes.as_slice())
            .expect("hand decoding the encoded bytes should succeed");
        assert_eq!(
            redecoded.producer_name.as_deref(),
            Some("tenflowers-core-tests")
        );

        let round_tripped = OnnxModel::from_protobuf(&bytes).expect("from_protobuf should succeed");

        assert_eq!(round_tripped.producer_name, model.producer_name);
        assert_eq!(round_tripped.producer_version, model.producer_version);
        assert_eq!(
            round_tripped.metadata.description,
            model.metadata.description
        );
        assert_eq!(round_tripped.metadata.domain, model.metadata.domain);
        assert_eq!(
            round_tripped.metadata.model_version,
            model.metadata.model_version
        );
        assert_eq!(round_tripped.opset_imports.len(), 1);
        assert_eq!(round_tripped.opset_imports[0].version, 18);

        assert_eq!(round_tripped.graph.name, model.graph.name);
        assert_eq!(round_tripped.graph.nodes.len(), 2);
        for (expected, actual) in model
            .graph
            .nodes
            .iter()
            .zip(round_tripped.graph.nodes.iter())
        {
            assert_eq!(actual.name, expected.name);
            assert_eq!(actual.op_type, expected.op_type);
            assert_eq!(actual.inputs, expected.inputs);
            assert_eq!(actual.outputs, expected.outputs);
            assert_eq!(actual.attributes.len(), expected.attributes.len());
        }

        // The Tensor-valued attribute on relu1 must have survived via the
        // new AttributeProto.t field.
        let relu = &round_tripped.graph.nodes[1];
        match relu.attributes.get("reference_tensor") {
            Some(OnnxAttribute::Tensor(t)) => {
                assert_eq!(t.name, "ref_const");
                assert_eq!(t.dims, vec![2]);
                assert_eq!(t.data.len(), 8);
                let v0 = f32::from_le_bytes(t.data[0..4].try_into().expect("4 bytes"));
                let v1 = f32::from_le_bytes(t.data[4..8].try_into().expect("4 bytes"));
                assert_eq!(v0, 1.5);
                assert_eq!(v1, -2.5);
            }
            other => panic!("expected a Tensor attribute, got {other:?}"),
        }

        // Initializer data survives byte-for-byte.
        assert_eq!(round_tripped.graph.initializers.len(), 1);
        assert_eq!(round_tripped.graph.initializers[0].dims, vec![3]);
        assert_eq!(
            round_tripped.graph.initializers[0].data,
            model.graph.initializers[0].data
        );

        // value_info / dynamic-dimension shape extraction: `input` has a
        // Param("batch") dim followed by a Value(3) dim.
        let in_dims = &round_tripped.graph.inputs[0]
            .value_type
            .tensor_type
            .as_ref()
            .expect("tensor_type")
            .shape
            .dims;
        assert_eq!(in_dims.len(), 2);
        match &in_dims[0] {
            OnnxDimension::Param(s) => assert_eq!(s, "batch"),
            other => panic!("expected Param(\"batch\"), got {other:?}"),
        }
        match &in_dims[1] {
            OnnxDimension::Value(v) => assert_eq!(*v, 3),
            other => panic!("expected Value(3), got {other:?}"),
        }

        let out_dims = &round_tripped.graph.outputs[0]
            .value_type
            .tensor_type
            .as_ref()
            .expect("tensor_type")
            .shape
            .dims;
        assert_eq!(out_dims.len(), 1);
        match &out_dims[0] {
            OnnxDimension::Value(v) => assert_eq!(*v, 3),
            other => panic!("expected Value(3), got {other:?}"),
        }
    }

    #[test]
    fn import_from_bytes_decodes_hand_crafted_protobuf() {
        let proto_model = proto::ModelProto {
            ir_version: Some(9),
            opset_import: vec![proto::OperatorSetIdProto {
                domain: Some(String::new()),
                version: Some(17),
            }],
            producer_name: Some("hand-crafted".to_string()),
            producer_version: Some("1.0".to_string()),
            domain: Some("test.domain".to_string()),
            model_version: Some(7),
            doc_string: Some("hand crafted model".to_string()),
            graph: Some(proto::GraphProto {
                node: vec![proto::NodeProto {
                    input: vec!["x".to_string()],
                    output: vec!["y".to_string()],
                    name: Some("identity1".to_string()),
                    op_type: Some("Identity".to_string()),
                    attribute: vec![],
                }],
                name: Some("hand_graph".to_string()),
                initializer: vec![],
                value_info: vec![],
                input: vec![proto::ValueInfoProto {
                    name: Some("x".to_string()),
                    r#type: None,
                    doc_string: None,
                }],
                output: vec![proto::ValueInfoProto {
                    name: Some("y".to_string()),
                    r#type: None,
                    doc_string: None,
                }],
            }),
        };

        let bytes = proto_model.encode_to_vec();

        let importer = OnnxImporter::new();
        let model = importer
            .import_from_bytes(&bytes)
            .expect("importing hand-crafted bytes should succeed");

        assert_eq!(model.graph.name, "hand_graph");
        assert_eq!(model.graph.nodes.len(), 1);
        assert_eq!(model.graph.nodes[0].name, "identity1");
        assert_eq!(model.graph.nodes[0].op_type, "Identity");
        assert_eq!(model.graph.nodes[0].inputs, vec!["x".to_string()]);
        assert_eq!(model.graph.nodes[0].outputs, vec!["y".to_string()]);
        assert_eq!(model.producer_name, "hand-crafted");
        assert_eq!(model.metadata.model_version, 7);
        assert_eq!(model.metadata.description, "hand crafted model");
        assert_eq!(model.opset_imports.len(), 1);
        assert_eq!(model.opset_imports[0].version, 17);
    }

    #[test]
    fn unrecognized_attribute_type_code_is_an_error() {
        let attr = proto::AttributeProto {
            name: Some("bogus".to_string()),
            doc_string: None,
            r#type: Some(999),
            f: None,
            i: None,
            s: None,
            t: None,
            floats: Vec::new(),
            ints: Vec::new(),
            strings: Vec::new(),
        };
        let result = OnnxAttribute::from_protobuf(&attr);
        assert!(
            result.is_err(),
            "unrecognized attribute type code must error, not fabricate Float(0.0)"
        );
    }

    #[test]
    fn missing_attribute_type_code_is_an_error() {
        let attr = proto::AttributeProto {
            name: Some("no_type".to_string()),
            doc_string: None,
            r#type: None,
            f: None,
            i: None,
            s: None,
            t: None,
            floats: Vec::new(),
            ints: Vec::new(),
            strings: Vec::new(),
        };
        let result = OnnxAttribute::from_protobuf(&attr);
        assert!(result.is_err());
    }

    #[test]
    fn unrecognized_tensor_data_type_code_is_an_error() {
        let tensor = proto::TensorProto {
            dims: vec![1],
            data_type: Some(999),
            name: Some("bogus".to_string()),
            raw_data: Some(vec![0, 0, 0, 0]),
            float_data: Vec::new(),
            int32_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let result = OnnxTensor::from_protobuf(&tensor);
        assert!(
            result.is_err(),
            "unrecognized tensor data_type code must error, not fabricate Float32"
        );
    }

    #[test]
    fn attribute_tensor_round_trips_via_t_field() {
        let inner = OnnxTensor {
            name: "w".to_string(),
            data_type: OnnxDataType::Int64,
            dims: vec![1],
            data: 42i64.to_le_bytes().to_vec(),
        };
        let attr = OnnxAttribute::Tensor(inner);
        let proto_attr = attr.to_protobuf("weights").expect("to_protobuf");
        assert!(
            proto_attr.t.is_some(),
            "AttributeProto.t must be populated for Tensor attributes"
        );
        let decoded = OnnxAttribute::from_protobuf(&proto_attr).expect("from_protobuf");
        match decoded {
            OnnxAttribute::Tensor(t) => {
                assert_eq!(t.name, "w");
                assert_eq!(t.data_type, OnnxDataType::Int64);
                assert_eq!(t.dims, vec![1]);
            }
            other => panic!("expected Tensor, got {other:?}"),
        }
    }

    #[test]
    fn exporter_falls_back_to_configured_opset_version_when_model_has_none() {
        let mut model = sample_model();
        model.opset_imports.clear();

        let config = OnnxConfig {
            opset_version: 21,
            ..OnnxConfig::default()
        };
        let exporter = OnnxExporter::with_config(config);
        let bytes = exporter
            .export_to_bytes(&model)
            .expect("export should succeed");
        let decoded = proto::ModelProto::decode(bytes.as_slice()).expect("decode should succeed");
        assert_eq!(decoded.opset_import.len(), 1);
        assert_eq!(decoded.opset_import[0].version, Some(21));
    }

    #[test]
    fn exporter_respects_models_own_opset_imports() {
        // `sample_model()` already sets opset_imports = [{"", 18}]; a
        // differently-configured exporter must not override that explicit
        // model data with its own fallback version.
        let model = sample_model();
        let config = OnnxConfig {
            opset_version: 99,
            ..OnnxConfig::default()
        };
        let exporter = OnnxExporter::with_config(config);
        let bytes = exporter
            .export_to_bytes(&model)
            .expect("export should succeed");
        let decoded = proto::ModelProto::decode(bytes.as_slice()).expect("decode should succeed");
        assert_eq!(decoded.opset_import.len(), 1);
        assert_eq!(decoded.opset_import[0].version, Some(18));
    }
}
