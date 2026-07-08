//! Hand-written protobuf schema for the ONNX wire format.
//!
//! This mirrors the subset of the official ONNX `onnx.proto3` schema that
//! TenfloweRS needs in order to read and write `ModelProto` messages. It is
//! a hand-rolled `prost` schema (no `.proto` file / `prost-build` code
//! generation involved), following the same proven pattern already used at
//! `tenflowers-neural::onnx::proto`.
//!
//! This module is only compiled when the `onnx` feature is enabled (see the
//! `#[cfg(feature = "onnx")] pub mod proto;` declaration in
//! `onnx_interop::mod`).
//!
//! Field tag numbers for `ModelProto`, `GraphProto`, `NodeProto`,
//! `TensorProto`, `ValueInfoProto`, `TypeProto`, `TensorShapeProto` and
//! `OperatorSetIdProto` match the official ONNX wire format. `AttributeProto`
//! additionally carries the official `t` (tensor) field at tag 5 so that
//! `OnnxAttribute::Tensor` can round-trip through the wire for real (the
//! `tenflowers-neural` reference implementation omits this field); its
//! `floats`/`ints`/`strings` tags (6/7/8) mirror that existing hand-rolled
//! reference rather than the official onnx.proto3 numbering (7/8/9, with `g`
//! at 6). This schema is self-consistent for TenfloweRS's own encode/decode
//! round trips, but is not guaranteed byte-for-byte compatible with
//! `AttributeProto` messages that set `g`/`floats`/`ints`/`strings` produced
//! by other ONNX toolchains.

use prost::{Enumeration, Message};

/// Top-level ONNX model container.
#[derive(Clone, PartialEq, Message)]
pub struct ModelProto {
    /// The version of the IR this model targets. See the ONNX `Version` enum.
    #[prost(int64, optional, tag = "1")]
    pub ir_version: Option<i64>,

    /// The OperatorSets this model relies on.
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetIdProto>,

    /// The name of the framework or tool used to generate this model.
    #[prost(string, optional, tag = "2")]
    pub producer_name: Option<String>,

    /// The version of the framework or tool used to generate this model.
    #[prost(string, optional, tag = "3")]
    pub producer_version: Option<String>,

    /// Domain name of the model.
    #[prost(string, optional, tag = "4")]
    pub domain: Option<String>,

    /// The version of the model itself.
    #[prost(int64, optional, tag = "5")]
    pub model_version: Option<i64>,

    /// Human-readable documentation for this model.
    #[prost(string, optional, tag = "6")]
    pub doc_string: Option<String>,

    /// The parameterized graph that is evaluated to execute the model.
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
}

/// ONNX computation graph.
#[derive(Clone, PartialEq, Message)]
pub struct GraphProto {
    /// The nodes in the graph, topologically sorted.
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,

    /// The name of the graph.
    #[prost(string, optional, tag = "2")]
    pub name: Option<String>,

    /// A list of named tensor values, used to specify constant inputs.
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,

    /// Type/shape information for intermediate values in the graph.
    #[prost(message, repeated, tag = "13")]
    pub value_info: Vec<ValueInfoProto>,

    /// The inputs of the graph.
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,

    /// The outputs of the graph.
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,
}

/// A single computation node (operator instance) in an ONNX graph.
#[derive(Clone, PartialEq, Message)]
pub struct NodeProto {
    /// The inputs to this node.
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,

    /// The outputs of this node.
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,

    /// An optional identifier for this node in a graph.
    #[prost(string, optional, tag = "3")]
    pub name: Option<String>,

    /// The symbolic identifier of the operator to execute (e.g. "Add").
    #[prost(string, optional, tag = "4")]
    pub op_type: Option<String>,

    /// Named attributes.
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,
}

/// A tensor's shape, data type, and raw content (used for initializers and
/// for `AttributeProto.t`).
#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    /// The shape of the tensor.
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,

    /// The data type of the tensor; see `TensorDataType` /
    /// `onnx_interop::types::OnnxDataType` for the code meanings.
    #[prost(int32, optional, tag = "2")]
    pub data_type: Option<i32>,

    /// The name of the tensor.
    #[prost(string, optional, tag = "8")]
    pub name: Option<String>,

    /// Tensor content, organized in row-major order, in `data_type`'s
    /// native little-endian byte layout. This crate always writes tensor
    /// data via `raw_data` (never the field-based encodings below).
    #[prost(bytes, optional, tag = "9")]
    pub raw_data: Option<Vec<u8>>,

    /// For float and complex64 values. Not written by this crate's
    /// exporter (which always emits `raw_data`); present for schema
    /// completeness / reading tensors produced by other ONNX toolchains
    /// that use the field-based encoding instead of `raw_data`.
    #[prost(float, repeated, packed = "false", tag = "4")]
    pub float_data: Vec<f32>,

    /// For int32, uint8, int8, uint16, int16, bool, and float16 values.
    #[prost(int32, repeated, packed = "false", tag = "5")]
    pub int32_data: Vec<i32>,

    /// For int64 values.
    #[prost(int64, repeated, packed = "false", tag = "7")]
    pub int64_data: Vec<i64>,
}

/// Type and shape information for a named graph value (an input, output, or
/// intermediate value).
#[derive(Clone, PartialEq, Message)]
pub struct ValueInfoProto {
    /// The name of the tensor.
    #[prost(string, optional, tag = "1")]
    pub name: Option<String>,

    /// This field MUST be present for this version of the IR.
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,

    /// Human-readable documentation for this value.
    #[prost(string, optional, tag = "3")]
    pub doc_string: Option<String>,
}

/// A value's type. Only the tensor-type oneof arm is modeled; ONNX's
/// sequence/map/optional/sparse-tensor type variants are out of scope for
/// this slice.
#[derive(Clone, PartialEq, Message)]
pub struct TypeProto {
    #[prost(oneof = "type_proto::Value", tags = "1")]
    pub value: Option<type_proto::Value>,
}

pub mod type_proto {
    use prost::{Message, Oneof};

    #[derive(Clone, PartialEq, Oneof)]
    pub enum Value {
        #[prost(message, tag = "1")]
        TensorType(Tensor),
    }

    /// Tensor-shaped type information: element type plus shape.
    #[derive(Clone, PartialEq, Message)]
    pub struct Tensor {
        /// The basic element type of the tensor.
        #[prost(int32, optional, tag = "1")]
        pub elem_type: Option<i32>,

        /// The shape of the tensor. Absence means "unranked" per the ONNX
        /// spec (as opposed to an explicit rank-0 / scalar shape, which is
        /// `Some(TensorShapeProto { dim: vec![] })`).
        #[prost(message, optional, tag = "2")]
        pub shape: Option<super::TensorShapeProto>,
    }
}

/// A tensor shape: a sequence of dimensions, each either a fixed value or a
/// named symbolic parameter (for dynamic axes, e.g. a batch-size dimension
/// named `"batch"`).
#[derive(Clone, PartialEq, Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<tensor_shape_proto::Dimension>,
}

pub mod tensor_shape_proto {
    use prost::{Message, Oneof};

    #[derive(Clone, PartialEq, Message)]
    pub struct Dimension {
        #[prost(oneof = "dimension::Value", tags = "1, 2")]
        pub value: Option<dimension::Value>,
    }

    pub mod dimension {
        use prost::Oneof;

        #[derive(Clone, PartialEq, Oneof)]
        pub enum Value {
            #[prost(int64, tag = "1")]
            DimValue(i64),
            #[prost(string, tag = "2")]
            DimParam(String),
        }
    }
}

/// A single named attribute on a `NodeProto`.
#[derive(Clone, PartialEq, Message)]
pub struct AttributeProto {
    /// The name field MUST be present for this version of the IR.
    #[prost(string, optional, tag = "1")]
    pub name: Option<String>,

    /// Human-readable documentation for this attribute.
    #[prost(string, optional, tag = "13")]
    pub doc_string: Option<String>,

    /// The type field MUST be present for this version of the IR.
    #[prost(enumeration = "AttributeType", optional, tag = "20")]
    pub r#type: Option<i32>,

    /// Exactly one of the fields below is populated, matching whichever
    /// variant `type` declares.
    #[prost(float, optional, tag = "2")]
    pub f: Option<f32>,

    #[prost(int64, optional, tag = "3")]
    pub i: Option<i64>,

    #[prost(bytes, optional, tag = "4")]
    pub s: Option<Vec<u8>>,

    /// Tensor-valued attribute payload. Added over the hand-rolled
    /// `tenflowers-neural` reference implementation (which omits this
    /// field entirely) so that `OnnxAttribute::Tensor` round-trips through
    /// the wire for real instead of needing a fake fallback. Matches the
    /// official ONNX `AttributeProto.t` field at tag 5.
    #[prost(message, optional, tag = "5")]
    pub t: Option<TensorProto>,

    #[prost(float, repeated, packed = "false", tag = "6")]
    pub floats: Vec<f32>,

    #[prost(int64, repeated, packed = "false", tag = "7")]
    pub ints: Vec<i64>,

    #[prost(bytes, repeated, tag = "8")]
    pub strings: Vec<Vec<u8>>,
}

/// Identifies one (domain, version) operator set that a model depends on.
#[derive(Clone, PartialEq, Message)]
pub struct OperatorSetIdProto {
    /// The domain of the operator set being identified (empty string means
    /// the default ONNX domain).
    #[prost(string, optional, tag = "1")]
    pub domain: Option<String>,

    /// The version of the operator set being identified.
    #[prost(int64, optional, tag = "2")]
    pub version: Option<i64>,
}

/// Mirrors the official ONNX `TensorProto.DataType` enum. Not referenced
/// directly by `TensorProto::data_type` (which stores a raw `i32` for
/// forward-compatibility with data-type codes from newer ONNX versions,
/// per standard protobuf practice for wire-level enums); provided for
/// documentation and for callers that want a typed view of the wire-level
/// codes. See `onnx_interop::types::OnnxDataType` for the enum actually
/// used by TenfloweRS's in-memory representation, and
/// `onnx_interop::convert::data_type_from_i32` for the exhaustive,
/// honest-error conversion between the two.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum TensorDataType {
    Undefined = 0,
    Float = 1,
    Uint8 = 2,
    Int8 = 3,
    Uint16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11,
    Uint32 = 12,
    Uint64 = 13,
    Complex64 = 14,
    Complex128 = 15,
    Bfloat16 = 16,
}

/// Mirrors the official ONNX `AttributeProto.AttributeType` enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum AttributeType {
    Undefined = 0,
    Float = 1,
    Int = 2,
    String = 3,
    Tensor = 4,
    Graph = 5,
    SparseTensor = 11,
    TypeProto = 13,
    Floats = 6,
    Ints = 7,
    Strings = 8,
    Tensors = 9,
    Graphs = 10,
    SparseTensors = 12,
    TypeProtos = 14,
}
