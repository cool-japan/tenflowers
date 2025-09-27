//! ONNX Protobuf Schema Definitions
//!
//! This module contains the protobuf definitions for ONNX format.
//! Based on the official ONNX protobuf schema.
//! Only available when the "onnx" feature is enabled.

#[cfg(feature = "onnx")]
pub mod proto {
    //! ONNX Protobuf message definitions
    //!
    //! This module contains the protobuf definitions for ONNX format.
    //! Based on the official ONNX protobuf schema.

    use prost::{Enumeration, Message};

    /// ONNX Model protobuf message
    #[derive(Clone, PartialEq, Message)]
    pub struct ModelProto {
        /// The version of the IR this model targets. See Version enum.
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

    /// ONNX Graph protobuf message
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

        /// Information for the values in the graph.
        #[prost(message, repeated, tag = "13")]
        pub value_info: Vec<ValueInfoProto>,

        /// The inputs and outputs of the graph.
        #[prost(message, repeated, tag = "11")]
        pub input: Vec<ValueInfoProto>,

        #[prost(message, repeated, tag = "12")]
        pub output: Vec<ValueInfoProto>,
    }

    /// ONNX Node protobuf message
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

        /// The symbolic identifier of the Operator to execute.
        #[prost(string, optional, tag = "4")]
        pub op_type: Option<String>,

        /// Named attributes.
        #[prost(message, repeated, tag = "5")]
        pub attribute: Vec<AttributeProto>,
    }

    /// ONNX Tensor protobuf message
    #[derive(Clone, PartialEq, Message)]
    pub struct TensorProto {
        /// The shape of the tensor.
        #[prost(int64, repeated, tag = "1")]
        pub dims: Vec<i64>,

        /// The data type of the tensor.
        #[prost(int32, optional, tag = "2")]
        pub data_type: Option<i32>,

        /// The name of the tensor.
        #[prost(string, optional, tag = "8")]
        pub name: Option<String>,

        /// Tensor content must be organized in row-major order.
        #[prost(bytes, optional, tag = "9")]
        pub raw_data: Option<Vec<u8>>,

        /// For float and complex64 values
        #[prost(float, repeated, packed = "false", tag = "4")]
        pub float_data: Vec<f32>,

        /// For int32, uint8, int8, uint16, int16, bool, and float16 values
        #[prost(int32, repeated, packed = "false", tag = "5")]
        pub int32_data: Vec<i32>,

        /// For int64 values
        #[prost(int64, repeated, packed = "false", tag = "7")]
        pub int64_data: Vec<i64>,
    }

    /// ONNX ValueInfo protobuf message
    #[derive(Clone, PartialEq, Message)]
    pub struct ValueInfoProto {
        /// The name of the tensor.
        #[prost(string, optional, tag = "1")]
        pub name: Option<String>,

        /// This field MUST be present for this version of the IR.
        #[prost(message, optional, tag = "2")]
        pub r#type: Option<TypeProto>,

        /// A human-readable documentation for this value.
        #[prost(string, optional, tag = "3")]
        pub doc_string: Option<String>,
    }

    /// ONNX Type protobuf message
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

        /// ONNX Tensor Type protobuf message
        #[derive(Clone, PartialEq, Message)]
        pub struct Tensor {
            /// The basic element type of a tensor.
            #[prost(int32, optional, tag = "1")]
            pub elem_type: Option<i32>,

            /// The shape of the tensor.
            #[prost(message, optional, tag = "2")]
            pub shape: Option<super::TensorShapeProto>,
        }
    }

    /// ONNX TensorShape protobuf message
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

    /// ONNX Attribute protobuf message
    #[derive(Clone, PartialEq, Message)]
    pub struct AttributeProto {
        /// The name field MUST be present for this version of the IR.
        #[prost(string, optional, tag = "1")]
        pub name: Option<String>,

        /// A human-readable documentation for this attribute.
        #[prost(string, optional, tag = "13")]
        pub doc_string: Option<String>,

        /// The type field MUST be present for this version of the IR.
        #[prost(enumeration = "AttributeType", optional, tag = "20")]
        pub r#type: Option<i32>,

        /// Exactly ONE of the following fields must be present for this version of the IR
        #[prost(float, optional, tag = "2")]
        pub f: Option<f32>,

        #[prost(int64, optional, tag = "3")]
        pub i: Option<i64>,

        #[prost(bytes, optional, tag = "4")]
        pub s: Option<Vec<u8>>,

        #[prost(float, repeated, packed = "false", tag = "6")]
        pub floats: Vec<f32>,

        #[prost(int64, repeated, packed = "false", tag = "7")]
        pub ints: Vec<i64>,

        #[prost(bytes, repeated, tag = "8")]
        pub strings: Vec<Vec<u8>>,
    }

    /// ONNX OperatorSetId protobuf message
    #[derive(Clone, PartialEq, Message)]
    pub struct OperatorSetIdProto {
        /// The domain of the operator set being identified.
        #[prost(string, optional, tag = "1")]
        pub domain: Option<String>,

        /// The version of the operator set being identified.
        #[prost(int64, optional, tag = "2")]
        pub version: Option<i64>,
    }

    /// ONNX Data Type enumeration
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
    #[repr(i32)]
    pub enum TensorDataType {
        Undefined = 0,
        Float = 1,       // float32
        Uint8 = 2,       // uint8
        Int8 = 3,        // int8
        Uint16 = 4,      // uint16
        Int16 = 5,       // int16
        Int32 = 6,       // int32
        Int64 = 7,       // int64
        String = 8,      // string
        Bool = 9,        // bool
        Float16 = 10,    // float16
        Double = 11,     // float64/double
        Uint32 = 12,     // uint32
        Uint64 = 13,     // uint64
        Complex64 = 14,  // complex64
        Complex128 = 15, // complex128
        Bfloat16 = 16,   // bfloat16
    }

    /// ONNX Attribute Type enumeration
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
}

#[cfg(feature = "onnx")]
pub use proto::*;
