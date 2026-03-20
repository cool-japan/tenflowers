//! ONNX core data structures and types
//!
//! This module defines all the fundamental ONNX data structures used for
//! model representation, graph structure, and type information.

use crate::{Result, Tensor};
use std::collections::HashMap;

/// ONNX model representation for TenfloweRS
#[derive(Debug, Clone)]
pub struct OnnxModel {
    /// Model graph containing nodes and connections
    pub graph: OnnxGraph,
    /// Model metadata
    pub metadata: OnnxModelMetadata,
    /// Opset version information
    pub opset_imports: Vec<OnnxOpsetImport>,
    /// Producer information
    pub producer_name: String,
    /// Producer version
    pub producer_version: String,
}

/// ONNX computation graph
#[derive(Debug, Clone)]
pub struct OnnxGraph {
    /// Graph nodes (operations)
    pub nodes: Vec<OnnxNode>,
    /// Graph inputs
    pub inputs: Vec<OnnxValueInfo>,
    /// Graph outputs
    pub outputs: Vec<OnnxValueInfo>,
    /// Initializers (weights and constants)
    pub initializers: Vec<OnnxTensor>,
    /// Value information for intermediate values
    pub value_info: Vec<OnnxValueInfo>,
    /// Graph name
    pub name: String,
}

/// ONNX node representing an operation
#[derive(Debug, Clone)]
pub struct OnnxNode {
    /// Node name
    pub name: String,
    /// Operation type
    pub op_type: String,
    /// Input tensor names
    pub inputs: Vec<String>,
    /// Output tensor names
    pub outputs: Vec<String>,
    /// Node attributes
    pub attributes: HashMap<String, OnnxAttribute>,
}

/// ONNX value information
#[derive(Debug, Clone)]
pub struct OnnxValueInfo {
    /// Value name
    pub name: String,
    /// Value type
    pub value_type: OnnxTypeProto,
}

/// ONNX tensor data
#[derive(Debug, Clone)]
pub struct OnnxTensor {
    /// Tensor name
    pub name: String,
    /// Data type
    pub data_type: OnnxDataType,
    /// Tensor dimensions
    pub dims: Vec<i64>,
    /// Raw tensor data
    pub data: Vec<u8>,
}

/// ONNX attribute value
#[derive(Debug, Clone)]
pub enum OnnxAttribute {
    /// Float attribute
    Float(f32),
    /// Integer attribute
    Int(i64),
    /// String attribute
    String(String),
    /// Tensor attribute
    Tensor(OnnxTensor),
    /// Float array attribute
    Floats(Vec<f32>),
    /// Integer array attribute
    Ints(Vec<i64>),
    /// String array attribute
    Strings(Vec<String>),
}

/// ONNX data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxDataType {
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
    BFloat16 = 16,
}

/// ONNX type information
#[derive(Debug, Clone)]
pub struct OnnxTypeProto {
    /// Tensor type information
    pub tensor_type: Option<OnnxTensorTypeProto>,
}

/// ONNX tensor type information
#[derive(Debug, Clone)]
pub struct OnnxTensorTypeProto {
    /// Element data type
    pub elem_type: OnnxDataType,
    /// Tensor shape
    pub shape: OnnxTensorShapeProto,
}

/// ONNX tensor shape information
#[derive(Debug, Clone)]
pub struct OnnxTensorShapeProto {
    /// Shape dimensions
    pub dims: Vec<OnnxDimension>,
}

/// ONNX dimension information
#[derive(Debug, Clone)]
pub enum OnnxDimension {
    /// Fixed dimension size
    Value(i64),
    /// Dynamic dimension with parameter name
    Param(String),
}

/// ONNX model metadata
#[derive(Debug, Clone)]
pub struct OnnxModelMetadata {
    /// Model description
    pub description: String,
    /// Model domain
    pub domain: String,
    /// Model version
    pub model_version: i64,
    /// Custom metadata properties
    pub metadata_props: HashMap<String, String>,
}

/// ONNX opset import information
#[derive(Debug, Clone)]
pub struct OnnxOpsetImport {
    /// Domain name (empty for default ONNX domain)
    pub domain: String,
    /// Opset version
    pub version: i64,
}

/// ONNX import/export configuration
#[derive(Debug)]
pub struct OnnxConfig {
    /// ONNX opset version to use for export
    pub opset_version: i64,
    /// Whether to optimize the exported model
    pub optimize: bool,
    /// Export format options
    pub format_options: OnnxFormatOptions,
}

/// ONNX format options
#[derive(Debug, Clone)]
pub struct OnnxFormatOptions {
    /// Use external data for large tensors
    pub use_external_data: bool,
    /// External data threshold (bytes)
    pub external_data_threshold: usize,
    /// Compress large tensors
    pub compress_large_tensors: bool,
    /// Include documentation strings
    pub include_docs: bool,
}

/// Custom ONNX operator trait
pub trait OnnxCustomOp: Send + Sync {
    /// Get operator name
    fn name(&self) -> &str;

    /// Get operator domain
    fn domain(&self) -> &str;

    /// Export operation to ONNX
    fn export(
        &self,
        inputs: &[String],
        outputs: &[String],
        attrs: &HashMap<String, OnnxAttribute>,
    ) -> Result<OnnxNode>;

    /// Import operation from ONNX
    fn import(&self, node: &OnnxNode) -> Result<Box<dyn TenfloweRSOperation>>;
}

/// TenfloweRS operation trait for ONNX import
pub trait TenfloweRSOperation: Send + Sync {
    /// Execute the operation
    fn execute(&self, inputs: &[Tensor<f32>]) -> Result<Vec<Tensor<f32>>>;

    /// Get operation name
    fn name(&self) -> &str;
}

/// ONNX operation mapping trait
pub trait OnnxOpMapping: Send + Sync {
    /// Convert ONNX node to TenfloweRS operation
    fn map_operation(
        &self,
        node: &OnnxNode,
        initializers: &HashMap<String, Tensor<f32>>,
    ) -> Result<Box<dyn TenfloweRSOperation>>;

    /// Get supported operation names
    fn supported_ops(&self) -> Vec<String>;
}

impl Default for OnnxConfig {
    fn default() -> Self {
        Self {
            opset_version: 18,
            optimize: true,
            format_options: OnnxFormatOptions::default(),
        }
    }
}

impl Default for OnnxFormatOptions {
    fn default() -> Self {
        Self {
            use_external_data: true,
            external_data_threshold: 1024 * 1024, // 1MB
            compress_large_tensors: true,
            include_docs: true,
        }
    }
}
