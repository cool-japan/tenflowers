//! ONNX Model Integration
//!
//! This module provides basic ONNX model loading and conversion capabilities,
//! enabling interoperability with ONNX models from other frameworks.

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tenflowers_core::{Device, Result, Tensor, TensorError};

use super::{LoadResult, ModelMetadata, SemanticVersion};

/// The real, prost-based protobuf ONNX parser this loader delegates to.
/// Only referenced from behind `#[cfg(feature = "onnx")]`, since none of its
/// protobuf-decoding functionality exists without that feature.
#[cfg(feature = "onnx")]
use crate::onnx as onnx_impl;

/// ONNX data types
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxDataType {
    Float32,
    Float64,
    Int32,
    Int64,
    Uint8,
    Int8,
    Uint16,
    Int16,
    Bool,
    Float16,
    BFloat16,
    Complex64,
    Complex128,
}

impl OnnxDataType {
    /// Convert ONNX data type to TenfloweRS dtype string
    pub fn to_dtype_string(&self) -> String {
        match self {
            OnnxDataType::Float32 => "f32".to_string(),
            OnnxDataType::Float64 => "f64".to_string(),
            OnnxDataType::Int32 => "i32".to_string(),
            OnnxDataType::Int64 => "i64".to_string(),
            OnnxDataType::Uint8 => "u8".to_string(),
            OnnxDataType::Int8 => "i8".to_string(),
            OnnxDataType::Uint16 => "u16".to_string(),
            OnnxDataType::Int16 => "i16".to_string(),
            OnnxDataType::Bool => "bool".to_string(),
            OnnxDataType::Float16 => "f16".to_string(),
            OnnxDataType::BFloat16 => "bf16".to_string(),
            OnnxDataType::Complex64 => "c64".to_string(),
            OnnxDataType::Complex128 => "c128".to_string(),
        }
    }

    /// Parse ONNX data type from integer code
    pub fn from_type_code(code: i32) -> Result<Self> {
        match code {
            1 => Ok(OnnxDataType::Float32),
            2 => Ok(OnnxDataType::Uint8),
            3 => Ok(OnnxDataType::Int8),
            4 => Ok(OnnxDataType::Uint16),
            5 => Ok(OnnxDataType::Int16),
            6 => Ok(OnnxDataType::Int32),
            7 => Ok(OnnxDataType::Int64),
            9 => Ok(OnnxDataType::Bool),
            10 => Ok(OnnxDataType::Float16),
            11 => Ok(OnnxDataType::Float64),
            14 => Ok(OnnxDataType::Complex64),
            15 => Ok(OnnxDataType::Complex128),
            16 => Ok(OnnxDataType::BFloat16),
            _ => Err(TensorError::serialization_error_simple(format!(
                "Unsupported ONNX data type code: {}",
                code
            ))),
        }
    }
}

/// ONNX tensor information
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct OnnxTensorInfo {
    /// Tensor name
    pub name: String,
    /// Data type
    pub dtype: OnnxDataType,
    /// Tensor shape (None for dynamic dimensions)
    pub shape: Vec<Option<i64>>,
    /// Raw data bytes
    pub data: Option<Vec<u8>>,
}

impl OnnxTensorInfo {
    /// Create new ONNX tensor info
    pub fn new(name: String, dtype: OnnxDataType, shape: Vec<Option<i64>>) -> Self {
        Self {
            name,
            dtype,
            shape,
            data: None,
        }
    }

    /// Check if shape is fully defined (no dynamic dimensions)
    pub fn is_shape_static(&self) -> bool {
        self.shape.iter().all(|dim| dim.is_some())
    }

    /// Get static shape if available
    pub fn static_shape(&self) -> Option<Vec<i64>> {
        if self.is_shape_static() {
            Some(self.shape.iter().filter_map(|&dim| dim).collect())
        } else {
            None
        }
    }
}

/// ONNX operator/node type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OnnxOpType {
    // Tensor operations
    Reshape,
    Transpose,
    Squeeze,
    Unsqueeze,
    Concat,
    Split,
    Slice,
    Gather,
    Scatter,

    // Math operations
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Gemm,

    // Activation functions
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
    Gelu,
    Swish,

    // Neural network layers
    Conv,
    BatchNormalization,
    LayerNormalization,
    Dropout,
    MaxPool,
    AveragePool,
    GlobalAveragePool,

    // Recurrent layers
    LSTM,
    GRU,

    // Other
    Constant,
    Identity,
    Cast,
    Unknown(String),
}

impl OnnxOpType {
    /// Parse operator type from string
    pub fn parse_op_type(s: &str) -> Self {
        match s {
            "Reshape" => OnnxOpType::Reshape,
            "Transpose" => OnnxOpType::Transpose,
            "Squeeze" => OnnxOpType::Squeeze,
            "Unsqueeze" => OnnxOpType::Unsqueeze,
            "Concat" => OnnxOpType::Concat,
            "Split" => OnnxOpType::Split,
            "Slice" => OnnxOpType::Slice,
            "Gather" => OnnxOpType::Gather,
            "Scatter" => OnnxOpType::Scatter,
            "Add" => OnnxOpType::Add,
            "Sub" => OnnxOpType::Sub,
            "Mul" => OnnxOpType::Mul,
            "Div" => OnnxOpType::Div,
            "MatMul" => OnnxOpType::MatMul,
            "Gemm" => OnnxOpType::Gemm,
            "Relu" => OnnxOpType::Relu,
            "Sigmoid" => OnnxOpType::Sigmoid,
            "Tanh" => OnnxOpType::Tanh,
            "Softmax" => OnnxOpType::Softmax,
            "Gelu" => OnnxOpType::Gelu,
            "Swish" => OnnxOpType::Swish,
            "Conv" => OnnxOpType::Conv,
            "BatchNormalization" => OnnxOpType::BatchNormalization,
            "LayerNormalization" => OnnxOpType::LayerNormalization,
            "Dropout" => OnnxOpType::Dropout,
            "MaxPool" => OnnxOpType::MaxPool,
            "AveragePool" => OnnxOpType::AveragePool,
            "GlobalAveragePool" => OnnxOpType::GlobalAveragePool,
            "LSTM" => OnnxOpType::LSTM,
            "GRU" => OnnxOpType::GRU,
            "Constant" => OnnxOpType::Constant,
            "Identity" => OnnxOpType::Identity,
            "Cast" => OnnxOpType::Cast,
            _ => OnnxOpType::Unknown(s.to_string()),
        }
    }

    /// Get operator type name
    pub fn name(&self) -> String {
        match self {
            OnnxOpType::Unknown(name) => name.clone(),
            _ => format!("{:?}", self),
        }
    }
}

/// ONNX node/operation
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct OnnxNode {
    /// Node name
    pub name: String,
    /// Operator type
    pub op_type: String,
    /// Input tensor names
    pub inputs: Vec<String>,
    /// Output tensor names
    pub outputs: Vec<String>,
    /// Node attributes
    pub attributes: HashMap<String, OnnxAttribute>,
}

impl OnnxNode {
    /// Create new ONNX node
    pub fn new(name: String, op_type: String) -> Self {
        Self {
            name,
            op_type,
            inputs: Vec::new(),
            outputs: Vec::new(),
            attributes: HashMap::new(),
        }
    }

    /// Get parsed operator type
    pub fn parsed_op_type(&self) -> OnnxOpType {
        OnnxOpType::parse_op_type(&self.op_type)
    }

    /// Add input tensor
    pub fn add_input(&mut self, input: String) {
        self.inputs.push(input);
    }

    /// Add output tensor
    pub fn add_output(&mut self, output: String) {
        self.outputs.push(output);
    }

    /// Set attribute
    pub fn set_attribute(&mut self, name: String, value: OnnxAttribute) {
        self.attributes.insert(name, value);
    }

    /// Get attribute by name
    pub fn get_attribute(&self, name: &str) -> Option<&OnnxAttribute> {
        self.attributes.get(name)
    }
}

/// ONNX attribute value
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub enum OnnxAttribute {
    Int(i64),
    Float(f32),
    String(String),
    Tensor(OnnxTensorInfo),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Strings(Vec<String>),
}

impl OnnxAttribute {
    /// Get as integer
    pub fn as_int(&self) -> Option<i64> {
        match self {
            OnnxAttribute::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as float
    pub fn as_float(&self) -> Option<f32> {
        match self {
            OnnxAttribute::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            OnnxAttribute::String(v) => Some(v),
            _ => None,
        }
    }

    /// Get as integer array
    pub fn as_ints(&self) -> Option<&[i64]> {
        match self {
            OnnxAttribute::Ints(v) => Some(v),
            _ => None,
        }
    }

    /// Get as float array
    pub fn as_floats(&self) -> Option<&[f32]> {
        match self {
            OnnxAttribute::Floats(v) => Some(v),
            _ => None,
        }
    }
}

/// ONNX graph representation
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct OnnxGraph {
    /// Graph name
    pub name: String,
    /// Graph nodes/operations
    pub nodes: Vec<OnnxNode>,
    /// Input tensors
    pub inputs: Vec<OnnxTensorInfo>,
    /// Output tensors
    pub outputs: Vec<OnnxTensorInfo>,
    /// Initializer tensors (weights)
    pub initializers: Vec<OnnxTensorInfo>,
    /// Value info (intermediate tensors)
    pub value_info: Vec<OnnxTensorInfo>,
}

impl OnnxGraph {
    /// Create new ONNX graph
    pub fn new(name: String) -> Self {
        Self {
            name,
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            initializers: Vec::new(),
            value_info: Vec::new(),
        }
    }

    /// Add node to graph
    pub fn add_node(&mut self, node: OnnxNode) {
        self.nodes.push(node);
    }

    /// Add input tensor
    pub fn add_input(&mut self, input: OnnxTensorInfo) {
        self.inputs.push(input);
    }

    /// Add output tensor
    pub fn add_output(&mut self, output: OnnxTensorInfo) {
        self.outputs.push(output);
    }

    /// Add initializer (weight) tensor
    pub fn add_initializer(&mut self, initializer: OnnxTensorInfo) {
        self.initializers.push(initializer);
    }

    /// Get initializer by name
    pub fn get_initializer(&self, name: &str) -> Option<&OnnxTensorInfo> {
        self.initializers.iter().find(|init| init.name == name)
    }

    /// Get all parameter names
    pub fn parameter_names(&self) -> Vec<String> {
        self.initializers
            .iter()
            .map(|init| init.name.clone())
            .collect()
    }

    /// Validate graph structure
    pub fn validate(&self) -> Result<()> {
        // Check that all node inputs are available
        let available_tensors: std::collections::HashSet<_> = self
            .inputs
            .iter()
            .map(|t| t.name.clone())
            .chain(self.initializers.iter().map(|t| t.name.clone()))
            .collect();

        for node in &self.nodes {
            for input in &node.inputs {
                if !available_tensors.contains(input) {
                    return Err(TensorError::serialization_error_simple(format!(
                        "Node '{}' references undefined input '{}'",
                        node.name, input
                    )));
                }
            }
        }

        Ok(())
    }
}

/// ONNX model metadata
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct OnnxModelMetadata {
    /// Model version
    pub model_version: i64,
    /// Producer name
    pub producer_name: String,
    /// Producer version
    pub producer_version: String,
    /// Domain
    pub domain: String,
    /// ONNX IR version
    pub ir_version: i64,
    /// Opset imports
    pub opset_imports: Vec<(String, i64)>,
}

impl Default for OnnxModelMetadata {
    fn default() -> Self {
        Self {
            model_version: 1,
            producer_name: "TenfloweRS".to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            domain: "".to_string(),
            ir_version: 8,
            opset_imports: vec![("".to_string(), 15)],
        }
    }
}

/// ONNX model
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct OnnxModel {
    /// Model metadata
    pub metadata: OnnxModelMetadata,
    /// Model graph
    pub graph: OnnxGraph,
}

impl OnnxModel {
    /// Create new ONNX model
    pub fn new(graph: OnnxGraph) -> Self {
        Self {
            metadata: OnnxModelMetadata::default(),
            graph,
        }
    }

    /// Validate model
    pub fn validate(&self) -> Result<()> {
        self.graph.validate()
    }

    /// Convert to TenfloweRS ModelMetadata
    pub fn to_tenflowers_metadata(&self) -> ModelMetadata {
        use super::{HardwareRequirements, TrainingInfo};

        ModelMetadata {
            model_type: "ONNX".to_string(),
            version: SemanticVersion::new(self.metadata.model_version as u32, 0, 0),
            framework_version: format!("ONNX-{}", self.metadata.ir_version),
            created_at: chrono::Utc::now().to_rfc3339(),
            architecture_hash: format!("{:x}", self.calculate_architecture_hash()),
            parameter_count: self.graph.initializers.len(),
            model_size: self.estimate_model_size() as usize,
            training_info: TrainingInfo {
                epochs: None,
                final_loss: None,
                validation_accuracy: None,
                optimizer: None,
                learning_rate: None,
                dataset_info: None,
            },
            hardware_requirements: HardwareRequirements {
                min_memory: self.estimate_model_size() * 2,
                recommended_memory: self.estimate_model_size() * 4,
                gpu_required: false,
                cpu_features: vec![],
                target_device: "cpu".to_string(),
            },
            custom: HashMap::new(),
        }
    }

    /// Calculate architecture hash
    fn calculate_architecture_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.graph.name.hash(&mut hasher);
        self.graph.nodes.len().hash(&mut hasher);
        self.graph.inputs.len().hash(&mut hasher);
        self.graph.outputs.len().hash(&mut hasher);

        hasher.finish()
    }

    /// Estimate model size in bytes
    fn estimate_model_size(&self) -> u64 {
        let mut size = 0u64;

        for init in &self.graph.initializers {
            if let Some(shape) = init.static_shape() {
                let elements: i64 = shape.iter().product();
                // Assume 4 bytes per element (f32)
                size += (elements * 4) as u64;
            }
        }

        size
    }
}

/// ONNX loader configuration
#[derive(Debug, Clone)]
pub struct OnnxLoadConfig {
    /// Target device for loaded tensors
    pub device: Device,
    /// Whether to perform strict validation
    pub strict_validation: bool,
    /// Whether to optimize graph
    pub optimize_graph: bool,
    /// Custom operator mappings
    pub custom_op_mappings: HashMap<String, String>,
}

impl Default for OnnxLoadConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            strict_validation: true,
            optimize_graph: false,
            custom_op_mappings: HashMap::new(),
        }
    }
}

impl OnnxLoadConfig {
    /// Create new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set target device
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Enable/disable strict validation
    pub fn with_strict_validation(mut self, strict: bool) -> Self {
        self.strict_validation = strict;
        self
    }

    /// Enable/disable graph optimization
    pub fn with_optimization(mut self, optimize: bool) -> Self {
        self.optimize_graph = optimize;
        self
    }

    /// Add custom operator mapping
    pub fn add_op_mapping(mut self, onnx_op: String, tenflowers_op: String) -> Self {
        self.custom_op_mappings.insert(onnx_op, tenflowers_op);
        self
    }
}

/// Decode raw ONNX protobuf bytes into the top-level `ModelProto` message.
///
/// This is the single real prost-based decode step that both the metadata-only
/// path (`utils::get_onnx_info`) and the full-load path (`parse_protobuf_model`)
/// build on, so bytes are only parsed once per call either way (rather than,
/// say, decoding once via `onnx::model::OnnxModel::from_protobuf` for the graph
/// and a second time just to recover `opset_import`, which that struct doesn't
/// retain).
#[cfg(feature = "onnx")]
fn decode_model_proto(bytes: &[u8]) -> Result<onnx_impl::onnx_proto::ModelProto> {
    use prost::Message;

    onnx_impl::onnx_proto::ModelProto::decode(bytes).map_err(|e| {
        TensorError::serialization_error_simple(format!("Failed to decode ONNX protobuf: {e}"))
    })
}

/// Build the target [`OnnxModelMetadata`] directly from a decoded `ModelProto`,
/// without touching its `graph` field at all. Field defaults intentionally
/// mirror `crate::onnx::model::OnnxModel::from_protobuf` exactly, so metadata
/// read this way is identical to metadata read via a full model load.
#[cfg(feature = "onnx")]
fn metadata_from_proto_model(proto_model: &onnx_impl::onnx_proto::ModelProto) -> OnnxModelMetadata {
    OnnxModelMetadata {
        model_version: proto_model.model_version.unwrap_or(1),
        producer_name: proto_model
            .producer_name
            .clone()
            .unwrap_or_else(|| "Unknown".to_string()),
        producer_version: proto_model
            .producer_version
            .clone()
            .unwrap_or_else(|| "Unknown".to_string()),
        domain: proto_model.domain.clone().unwrap_or_default(),
        ir_version: proto_model.ir_version.unwrap_or(7),
        opset_imports: proto_model
            .opset_import
            .iter()
            .map(|op| {
                (
                    op.domain.clone().unwrap_or_default(),
                    op.version.unwrap_or_default(),
                )
            })
            .collect(),
    }
}

/// Map the source (4-variant) ONNX dtype onto the target (13-variant) dtype.
#[cfg(feature = "onnx")]
fn convert_source_dtype(dtype: onnx_impl::types::OnnxDataType) -> OnnxDataType {
    match dtype {
        onnx_impl::types::OnnxDataType::Float32 => OnnxDataType::Float32,
        onnx_impl::types::OnnxDataType::Float64 => OnnxDataType::Float64,
        onnx_impl::types::OnnxDataType::Int32 => OnnxDataType::Int32,
        onnx_impl::types::OnnxDataType::Int64 => OnnxDataType::Int64,
    }
}

/// Map a source attribute value onto the target attribute representation.
#[cfg(feature = "onnx")]
fn convert_attribute(real_attr: &onnx_impl::data::OnnxAttribute) -> OnnxAttribute {
    match real_attr {
        onnx_impl::data::OnnxAttribute::Float(v) => OnnxAttribute::Float(*v),
        onnx_impl::data::OnnxAttribute::Int(v) => OnnxAttribute::Int(*v),
        onnx_impl::data::OnnxAttribute::String(v) => OnnxAttribute::String(v.clone()),
        onnx_impl::data::OnnxAttribute::Floats(v) => OnnxAttribute::Floats(v.clone()),
        onnx_impl::data::OnnxAttribute::Ints(v) => OnnxAttribute::Ints(v.clone()),
        onnx_impl::data::OnnxAttribute::Strings(v) => OnnxAttribute::Strings(v.clone()),
    }
}

/// Map a source node onto the target node representation.
#[cfg(feature = "onnx")]
fn convert_node(real_node: &onnx_impl::data::OnnxNode) -> OnnxNode {
    let mut node = OnnxNode::new(real_node.name.clone(), real_node.op_type.clone());
    node.inputs = real_node.inputs.clone();
    node.outputs = real_node.outputs.clone();
    for (k, v) in &real_node.attributes {
        node.attributes.insert(k.clone(), convert_attribute(v));
    }
    node
}

/// ONNX represents unknown/dynamic dimensions with a negative sentinel value
/// (see the `DimParam` handling in `onnx::data::OnnxValueInfo::from_protobuf`);
/// map those onto `None` for the target's `Vec<Option<i64>>` shape representation.
#[cfg(feature = "onnx")]
fn dim_to_optional(d: i64) -> Option<i64> {
    if d < 0 {
        None
    } else {
        Some(d)
    }
}

/// Map a source value-info (graph input/output) onto the target tensor-info representation.
#[cfg(feature = "onnx")]
fn convert_value_info(real_vi: &onnx_impl::data::OnnxValueInfo) -> OnnxTensorInfo {
    let dtype = convert_source_dtype(real_vi.elem_type);
    let shape = real_vi.shape.iter().map(|&d| dim_to_optional(d)).collect();
    OnnxTensorInfo::new(real_vi.name.clone(), dtype, shape)
    // .data intentionally stays None: OnnxValueInfo carries no tensor data on the source side.
}

/// Map a source initializer tensor (with raw weight bytes) onto the target tensor-info representation.
#[cfg(feature = "onnx")]
fn convert_tensor(real_tensor: &onnx_impl::data::OnnxTensor) -> OnnxTensorInfo {
    let dtype = convert_source_dtype(real_tensor.data_type);
    let shape = real_tensor
        .dims
        .iter()
        .map(|&d| dim_to_optional(d))
        .collect();
    let mut info = OnnxTensorInfo::new(real_tensor.name.clone(), dtype, shape);
    info.data = Some(real_tensor.raw_data.clone());
    info
}

/// Map a whole source graph onto the target graph representation.
#[cfg(feature = "onnx")]
fn convert_graph(real_graph: &onnx_impl::data::OnnxGraph) -> OnnxGraph {
    OnnxGraph {
        name: real_graph.name.clone(),
        nodes: real_graph.nodes.iter().map(convert_node).collect(),
        inputs: real_graph.inputs.iter().map(convert_value_info).collect(),
        outputs: real_graph.outputs.iter().map(convert_value_info).collect(),
        initializers: real_graph.initializers.iter().map(convert_tensor).collect(),
        // `onnx::data::OnnxGraph` has no `value_info` field at all -- its own
        // `from_protobuf` never reads `GraphProto.value_info` either, so this
        // is genuinely absent on the source side rather than dropped here.
        value_info: Vec::new(),
    }
}

/// Parse real ONNX protobuf bytes into the target [`OnnxModel`] representation
/// by delegating to the working prost-based parser in `crate::onnx`: one
/// top-level decode, then the real `OnnxGraph::from_protobuf` graph conversion
/// (the same two steps `onnx::model::OnnxModel::from_protobuf` performs
/// internally -- reusing them directly here avoids decoding the bytes twice).
#[cfg(feature = "onnx")]
fn parse_protobuf_model(bytes: &[u8]) -> Result<OnnxModel> {
    let proto_model = decode_model_proto(bytes)?;
    let metadata = metadata_from_proto_model(&proto_model);

    let graph_proto = proto_model.graph.as_ref().ok_or_else(|| {
        TensorError::serialization_error_simple("Missing graph in model".to_string())
    })?;
    let real_graph = onnx_impl::data::OnnxGraph::from_protobuf(graph_proto)?;
    let graph = convert_graph(&real_graph);

    Ok(OnnxModel { metadata, graph })
}

/// ONNX model loader
pub struct OnnxLoader {
    config: OnnxLoadConfig,
}

impl OnnxLoader {
    /// Create new ONNX loader with default configuration
    pub fn new() -> Self {
        Self {
            config: OnnxLoadConfig::default(),
        }
    }

    /// Create ONNX loader with custom configuration
    pub fn with_config(config: OnnxLoadConfig) -> Self {
        Self { config }
    }

    /// Load ONNX model from file
    pub fn load_from_file<P: AsRef<Path>>(&self, path: P) -> Result<OnnxModel> {
        #[cfg(feature = "onnx")]
        {
            let bytes = std::fs::read(path.as_ref()).map_err(|e| {
                TensorError::serialization_error_simple(format!(
                    "Failed to read ONNX file '{}': {e}",
                    path.as_ref().display()
                ))
            })?;
            self.load_from_bytes(&bytes)
        }
        #[cfg(not(feature = "onnx"))]
        {
            Err(TensorError::serialization_error_simple(format!(
                "ONNX protobuf loading requires the 'onnx' feature to be enabled (path: '{}')",
                path.as_ref().display()
            )))
        }
    }

    /// Load ONNX model from bytes
    pub fn load_from_bytes(&self, bytes: &[u8]) -> Result<OnnxModel> {
        #[cfg(feature = "onnx")]
        {
            let model = parse_protobuf_model(bytes)?;
            if self.config.strict_validation {
                model.validate()?;
            }
            Ok(model)
        }
        #[cfg(not(feature = "onnx"))]
        {
            Err(TensorError::serialization_error_simple(format!(
                "ONNX protobuf loading requires the 'onnx' feature to be enabled ({} bytes provided)",
                bytes.len()
            )))
        }
    }

    /// Convert ONNX weights to TenfloweRS format
    pub fn convert_weights<T>(&self, model: &OnnxModel) -> Result<HashMap<String, Tensor<T>>>
    where
        T: Clone + Default + bytemuck::Pod + bytemuck::Zeroable + 'static,
    {
        let mut weights = HashMap::new();

        for initializer in &model.graph.initializers {
            let tensor = Self::tensor_from_initializer::<T>(initializer)?;
            weights.insert(initializer.name.clone(), tensor);
        }

        Ok(weights)
    }

    /// Check whether the requested Rust element type matches an ONNX dtype's
    /// on-disk representation (byte layout), so raw bytes can be soundly
    /// reinterpreted as `[T]`.
    fn dtype_matches<T: 'static>(dtype: OnnxDataType) -> bool {
        use std::any::TypeId;

        let target = TypeId::of::<T>();
        match dtype {
            OnnxDataType::Float32 => target == TypeId::of::<f32>(),
            OnnxDataType::Float64 => target == TypeId::of::<f64>(),
            OnnxDataType::Int32 => target == TypeId::of::<i32>(),
            OnnxDataType::Int64 => target == TypeId::of::<i64>(),
            OnnxDataType::Uint8 => target == TypeId::of::<u8>(),
            OnnxDataType::Int8 => target == TypeId::of::<i8>(),
            OnnxDataType::Uint16 => target == TypeId::of::<u16>(),
            OnnxDataType::Int16 => target == TypeId::of::<i16>(),
            // ONNX bool tensors store one raw byte per element.
            OnnxDataType::Bool => target == TypeId::of::<u8>(),
            OnnxDataType::Float16
            | OnnxDataType::BFloat16
            | OnnxDataType::Complex64
            | OnnxDataType::Complex128 => false,
        }
    }

    /// Reinterpret an initializer's raw bytes as a `Tensor<T>`, validating
    /// shape, dtype, and byte-length along the way. Never fabricates data:
    /// any mismatch results in a typed `Err`.
    fn tensor_from_initializer<T>(info: &OnnxTensorInfo) -> Result<Tensor<T>>
    where
        T: Clone + Default + bytemuck::Pod + bytemuck::Zeroable + 'static,
    {
        let shape_i64 = info.static_shape().ok_or_else(|| {
            TensorError::serialization_error_simple(format!(
                "initializer '{}' has a dynamic shape {:?}; weights must be fully static",
                info.name, info.shape
            ))
        })?;

        let mut shape = Vec::with_capacity(shape_i64.len());
        for d in shape_i64 {
            if d < 0 {
                return Err(TensorError::serialization_error_simple(format!(
                    "initializer '{}' has an invalid negative dimension {d}",
                    info.name
                )));
            }
            shape.push(d as usize);
        }

        if !Self::dtype_matches::<T>(info.dtype) {
            return Err(TensorError::serialization_error_simple(format!(
                "initializer '{}' has ONNX dtype {:?} which does not match the requested Rust \
                 element type (size {} bytes); call convert_weights::<T>() with a T matching \
                 the tensor's declared dtype",
                info.name,
                info.dtype,
                std::mem::size_of::<T>()
            )));
        }

        let raw = info.data.as_deref().ok_or_else(|| {
            TensorError::serialization_error_simple(format!(
                "initializer '{}' has no raw tensor data to convert",
                info.name
            ))
        })?;

        let count: usize = shape.iter().product();
        let expected_bytes = count * std::mem::size_of::<T>();
        if raw.len() != expected_bytes {
            return Err(TensorError::serialization_error_simple(format!(
                "initializer '{}' raw data is {} bytes but shape {:?} with dtype {:?} expects {} bytes",
                info.name,
                raw.len(),
                shape,
                info.dtype,
                expected_bytes
            )));
        }

        let values: &[T] = bytemuck::try_cast_slice(raw).map_err(|e| {
            TensorError::serialization_error_simple(format!(
                "initializer '{}' raw bytes could not be reinterpreted as the target element type: {e}",
                info.name
            ))
        })?;

        Tensor::from_vec(values.to_vec(), &shape)
    }

    /// Get configuration
    pub fn config(&self) -> &OnnxLoadConfig {
        &self.config
    }
}

impl Default for OnnxLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// ONNX integration utilities
pub mod utils {
    use super::*;

    /// Check if file is an ONNX model
    pub fn is_onnx_file<P: AsRef<Path>>(path: P) -> bool {
        path.as_ref()
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("onnx"))
            .unwrap_or(false)
    }

    /// Get ONNX file info without loading the full model
    pub fn get_onnx_info<P: AsRef<Path>>(path: P) -> Result<OnnxModelMetadata> {
        #[cfg(feature = "onnx")]
        {
            let bytes = std::fs::read(path.as_ref()).map_err(|e| {
                TensorError::serialization_error_simple(format!(
                    "Failed to read ONNX file '{}': {e}",
                    path.as_ref().display()
                ))
            })?;
            // Decode only the top-level `ModelProto` and read metadata straight
            // off it -- this never runs the `OnnxGraph::from_protobuf` graph
            // conversion (node/tensor/attribute allocation), so it is genuinely
            // lighter than a full `load_from_bytes` call.
            let proto_model = super::decode_model_proto(&bytes)?;
            Ok(super::metadata_from_proto_model(&proto_model))
        }
        #[cfg(not(feature = "onnx"))]
        {
            Err(TensorError::serialization_error_simple(format!(
                "ONNX metadata reading requires the 'onnx' feature to be enabled (path: '{}')",
                path.as_ref().display()
            )))
        }
    }

    /// Convert ONNX data type to TenfloweRS dtype
    pub fn convert_dtype(onnx_dtype: OnnxDataType) -> String {
        onnx_dtype.to_dtype_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_data_type_conversion() {
        assert_eq!(OnnxDataType::Float32.to_dtype_string(), "f32");
        assert_eq!(OnnxDataType::Float64.to_dtype_string(), "f64");
        assert_eq!(OnnxDataType::Int32.to_dtype_string(), "i32");
    }

    #[test]
    fn test_onnx_data_type_from_code() {
        assert_eq!(
            OnnxDataType::from_type_code(1).expect("test: operation should succeed"),
            OnnxDataType::Float32
        );
        assert_eq!(
            OnnxDataType::from_type_code(11).expect("test: operation should succeed"),
            OnnxDataType::Float64
        );
        assert!(OnnxDataType::from_type_code(999).is_err());
    }

    #[test]
    fn test_onnx_op_type_parsing() {
        assert_eq!(OnnxOpType::parse_op_type("Conv"), OnnxOpType::Conv);
        assert_eq!(OnnxOpType::parse_op_type("Relu"), OnnxOpType::Relu);
        assert_eq!(OnnxOpType::parse_op_type("MatMul"), OnnxOpType::MatMul);

        if let OnnxOpType::Unknown(name) = OnnxOpType::parse_op_type("CustomOp") {
            assert_eq!(name, "CustomOp");
        } else {
            panic!("Expected Unknown variant");
        }
    }

    #[test]
    fn test_onnx_tensor_info() {
        let tensor = OnnxTensorInfo::new(
            "test".to_string(),
            OnnxDataType::Float32,
            vec![Some(3), Some(224), Some(224)],
        );

        assert!(tensor.is_shape_static());
        assert_eq!(tensor.static_shape(), Some(vec![3, 224, 224]));

        let dynamic_tensor = OnnxTensorInfo::new(
            "dynamic".to_string(),
            OnnxDataType::Float32,
            vec![None, Some(224), Some(224)],
        );

        assert!(!dynamic_tensor.is_shape_static());
        assert_eq!(dynamic_tensor.static_shape(), None);
    }

    #[test]
    fn test_onnx_node() {
        let mut node = OnnxNode::new("conv1".to_string(), "Conv".to_string());
        node.add_input("input".to_string());
        node.add_output("output".to_string());
        node.set_attribute("kernel_size".to_string(), OnnxAttribute::Ints(vec![3, 3]));

        assert_eq!(node.inputs.len(), 1);
        assert_eq!(node.outputs.len(), 1);
        assert!(node.get_attribute("kernel_size").is_some());

        if let Some(OnnxAttribute::Ints(kernel)) = node.get_attribute("kernel_size") {
            assert_eq!(kernel, &vec![3, 3]);
        } else {
            panic!("Expected Ints attribute");
        }
    }

    #[test]
    fn test_onnx_attribute() {
        let int_attr = OnnxAttribute::Int(42);
        assert_eq!(int_attr.as_int(), Some(42));
        assert_eq!(int_attr.as_float(), None);

        let float_attr = OnnxAttribute::Float(3.15);
        assert_eq!(float_attr.as_float(), Some(3.15));
        assert_eq!(float_attr.as_int(), None);

        let ints_attr = OnnxAttribute::Ints(vec![1, 2, 3]);
        assert_eq!(ints_attr.as_ints(), Some(&[1, 2, 3][..]));
    }

    #[test]
    fn test_onnx_graph_creation() {
        let mut graph = OnnxGraph::new("test_graph".to_string());

        let input = OnnxTensorInfo::new(
            "input".to_string(),
            OnnxDataType::Float32,
            vec![Some(1), Some(3), Some(224), Some(224)],
        );
        graph.add_input(input);

        let weight = OnnxTensorInfo::new(
            "weight".to_string(),
            OnnxDataType::Float32,
            vec![Some(64), Some(3), Some(7), Some(7)],
        );
        graph.add_initializer(weight);

        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.initializers.len(), 1);
        assert_eq!(graph.parameter_names().len(), 1);
    }

    #[test]
    fn test_onnx_graph_validation() {
        let mut graph = OnnxGraph::new("valid_graph".to_string());

        // Add input
        graph.add_input(OnnxTensorInfo::new(
            "input".to_string(),
            OnnxDataType::Float32,
            vec![Some(1), Some(3)],
        ));

        // Add initializer
        graph.add_initializer(OnnxTensorInfo::new(
            "weight".to_string(),
            OnnxDataType::Float32,
            vec![Some(3), Some(3)],
        ));

        // Add valid node
        let mut node = OnnxNode::new("matmul".to_string(), "MatMul".to_string());
        node.add_input("input".to_string());
        node.add_input("weight".to_string());
        node.add_output("output".to_string());
        graph.add_node(node);

        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_onnx_graph_validation_failure() {
        let mut graph = OnnxGraph::new("invalid_graph".to_string());

        // Add node with undefined input
        let mut node = OnnxNode::new("matmul".to_string(), "MatMul".to_string());
        node.add_input("undefined_input".to_string());
        graph.add_node(node);

        assert!(graph.validate().is_err());
    }

    #[test]
    fn test_onnx_model_metadata() {
        let metadata = OnnxModelMetadata::default();
        assert_eq!(metadata.producer_name, "TenfloweRS");
        assert_eq!(metadata.ir_version, 8);
        assert!(!metadata.opset_imports.is_empty());
    }

    #[test]
    fn test_onnx_model_creation() {
        let graph = OnnxGraph::new("test_model".to_string());
        let model = OnnxModel::new(graph);

        assert_eq!(model.metadata.producer_name, "TenfloweRS");
        assert_eq!(model.graph.name, "test_model");
    }

    #[test]
    fn test_onnx_load_config() {
        let config = OnnxLoadConfig::new()
            .with_device(Device::Cpu)
            .with_strict_validation(false)
            .with_optimization(true);

        assert_eq!(config.device, Device::Cpu);
        assert!(!config.strict_validation);
        assert!(config.optimize_graph);
    }

    #[test]
    fn test_onnx_loader_creation() {
        let loader = OnnxLoader::new();
        assert!(loader.config().strict_validation);

        let custom_config = OnnxLoadConfig::new().with_strict_validation(false);
        let custom_loader = OnnxLoader::with_config(custom_config);
        assert!(!custom_loader.config().strict_validation);
    }

    #[test]
    fn test_is_onnx_file() {
        assert!(utils::is_onnx_file("model.onnx"));
        assert!(utils::is_onnx_file("model.ONNX"));
        assert!(!utils::is_onnx_file("model.pt"));
        assert!(!utils::is_onnx_file("model.json"));
    }

    #[test]
    fn test_convert_dtype() {
        assert_eq!(utils::convert_dtype(OnnxDataType::Float32), "f32");
        assert_eq!(utils::convert_dtype(OnnxDataType::Int64), "i64");
    }

    // ---- Real loader tests -------------------------------------------------

    #[cfg(feature = "onnx")]
    #[test]
    fn test_onnx_loader_round_trip_from_bytes_and_file() {
        use crate::onnx::onnx_proto;
        use prost::Message;

        let node = onnx_proto::NodeProto {
            input: vec!["x".to_string(), "w".to_string()],
            output: vec!["y".to_string()],
            name: Some("matmul_node".to_string()),
            op_type: Some("MatMul".to_string()),
            attribute: vec![onnx_proto::AttributeProto {
                name: Some("axes".to_string()),
                r#type: Some(onnx_proto::AttributeType::Ints as i32),
                ints: vec![0, 1],
                ..Default::default()
            }],
        };

        // Mixed shape: one dynamic (DimParam) dim + one static (DimValue) dim,
        // to exercise the dynamic -> None mapping end to end.
        let input_shape = onnx_proto::TensorShapeProto {
            dim: vec![
                onnx_proto::tensor_shape_proto::Dimension {
                    value: Some(onnx_proto::tensor_shape_proto::dimension::Value::DimParam(
                        "batch".to_string(),
                    )),
                },
                onnx_proto::tensor_shape_proto::Dimension {
                    value: Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
                        3,
                    )),
                },
            ],
        };
        let input_vi = onnx_proto::ValueInfoProto {
            name: Some("x".to_string()),
            r#type: Some(onnx_proto::TypeProto {
                value: Some(onnx_proto::type_proto::Value::TensorType(
                    onnx_proto::type_proto::Tensor {
                        elem_type: Some(onnx_proto::TensorDataType::Float as i32),
                        shape: Some(input_shape),
                    },
                )),
            }),
            doc_string: None,
        };

        let output_shape = onnx_proto::TensorShapeProto {
            dim: vec![onnx_proto::tensor_shape_proto::Dimension {
                value: Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
                    2,
                )),
            }],
        };
        let output_vi = onnx_proto::ValueInfoProto {
            name: Some("y".to_string()),
            r#type: Some(onnx_proto::TypeProto {
                value: Some(onnx_proto::type_proto::Value::TensorType(
                    onnx_proto::type_proto::Tensor {
                        elem_type: Some(onnx_proto::TensorDataType::Float as i32),
                        shape: Some(output_shape),
                    },
                )),
            }),
            doc_string: None,
        };

        let weight_values: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight_bytes = bytemuck::cast_slice(&weight_values).to_vec();
        let initializer = onnx_proto::TensorProto {
            dims: vec![2, 3],
            data_type: Some(onnx_proto::TensorDataType::Float as i32),
            name: Some("w".to_string()),
            raw_data: Some(weight_bytes),
            ..Default::default()
        };

        let graph = onnx_proto::GraphProto {
            node: vec![node],
            name: Some("test_graph".to_string()),
            initializer: vec![initializer],
            input: vec![input_vi],
            output: vec![output_vi],
            value_info: Vec::new(),
        };

        let model_proto = onnx_proto::ModelProto {
            ir_version: Some(9),
            opset_import: vec![onnx_proto::OperatorSetIdProto {
                domain: Some("".to_string()),
                version: Some(17),
            }],
            producer_name: Some("pytest-onnx".to_string()),
            producer_version: Some("1.2.3".to_string()),
            domain: Some("ai.example".to_string()),
            model_version: Some(42),
            doc_string: Some("a test model".to_string()),
            graph: Some(graph),
        };

        let bytes = model_proto.encode_to_vec();

        let loader = OnnxLoader::new();
        let model = loader
            .load_from_bytes(&bytes)
            .expect("test: load_from_bytes should succeed on a well-formed model");

        // Metadata
        assert_eq!(model.metadata.ir_version, 9);
        assert_eq!(model.metadata.producer_name, "pytest-onnx");
        assert_eq!(model.metadata.producer_version, "1.2.3");
        assert_eq!(model.metadata.domain, "ai.example");
        assert_eq!(model.metadata.model_version, 42);
        assert_eq!(model.metadata.opset_imports, vec![("".to_string(), 17)]);

        // Graph / node
        assert_eq!(model.graph.name, "test_graph");
        assert_eq!(model.graph.nodes.len(), 1);
        let n = &model.graph.nodes[0];
        assert_eq!(n.name, "matmul_node");
        assert_eq!(n.op_type, "MatMul");
        assert_eq!(n.inputs, vec!["x".to_string(), "w".to_string()]);
        assert_eq!(n.outputs, vec!["y".to_string()]);
        match n.attributes.get("axes") {
            Some(OnnxAttribute::Ints(v)) => assert_eq!(v, &vec![0, 1]),
            other => panic!("expected Ints attribute, got {other:?}"),
        }

        // Input: dynamic dim mapped to None, static dim mapped to Some
        assert_eq!(model.graph.inputs.len(), 1);
        assert_eq!(model.graph.inputs[0].shape, vec![None, Some(3)]);
        assert_eq!(model.graph.inputs[0].dtype, OnnxDataType::Float32);

        // Output
        assert_eq!(model.graph.outputs.len(), 1);
        assert_eq!(model.graph.outputs[0].shape, vec![Some(2)]);

        // Initializer / weight bytes
        assert_eq!(model.graph.initializers.len(), 1);
        let init = &model.graph.initializers[0];
        assert_eq!(init.name, "w");
        assert_eq!(init.dtype, OnnxDataType::Float32);
        assert_eq!(init.static_shape(), Some(vec![2, 3]));
        let data_bytes = init
            .data
            .as_ref()
            .expect("test: initializer should carry raw data");
        let floats: &[f32] = bytemuck::cast_slice(data_bytes.as_slice());
        assert_eq!(floats, &weight_values);

        // File round-trip through the same bytes
        let file_path = std::env::temp_dir().join(format!(
            "tenflowers_onnx_roundtrip_test_{}.onnx",
            std::process::id()
        ));
        std::fs::write(&file_path, &bytes).expect("test: writing temp onnx file should succeed");
        let model_from_file = loader
            .load_from_file(&file_path)
            .expect("test: load_from_file should succeed");
        assert_eq!(
            model_from_file.metadata.ir_version,
            model.metadata.ir_version
        );
        assert_eq!(model_from_file.graph.name, model.graph.name);
        assert_eq!(model_from_file.graph.nodes.len(), model.graph.nodes.len());

        // convert_weights
        let weights = loader
            .convert_weights::<f32>(&model)
            .expect("test: convert_weights should succeed");
        let tensor = weights
            .get("w")
            .expect("test: weight 'w' should be present");
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.data(), &weight_values);

        // get_onnx_info (lightweight metadata read)
        let info = utils::get_onnx_info(&file_path).expect("test: get_onnx_info should succeed");
        assert_eq!(info.ir_version, 9);
        assert_eq!(info.producer_name, "pytest-onnx");
        assert_eq!(info.opset_imports, vec![("".to_string(), 17)]);

        let _ = std::fs::remove_file(&file_path);
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn test_onnx_load_from_bytes_garbage_returns_err() {
        let loader = OnnxLoader::new();
        let result = loader.load_from_bytes(&[0xFF, 0x00, 0xAB]);
        assert!(result.is_err());
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn test_onnx_load_from_bytes_missing_graph_returns_err() {
        use crate::onnx::onnx_proto;
        use prost::Message;

        let model_proto = onnx_proto::ModelProto {
            ir_version: Some(9),
            graph: None,
            ..Default::default()
        };
        let bytes = model_proto.encode_to_vec();

        let loader = OnnxLoader::new();
        let result = loader.load_from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn test_convert_weights_dtype_mismatch_returns_err() {
        let mut graph = OnnxGraph::new("g".to_string());
        let mut info = OnnxTensorInfo::new("w".to_string(), OnnxDataType::Float32, vec![Some(2)]);
        info.data = Some(bytemuck::cast_slice(&[1.0f32, 2.0f32]).to_vec());
        graph.add_initializer(info);
        let model = OnnxModel::new(graph);

        let loader = OnnxLoader::new();
        let result = loader.convert_weights::<i64>(&model);
        assert!(result.is_err());
    }

    #[cfg(not(feature = "onnx"))]
    #[test]
    fn test_onnx_load_from_bytes_without_onnx_feature_returns_err() {
        let loader = OnnxLoader::new();
        let result = loader.load_from_bytes(b"not real onnx data");
        assert!(result.is_err());
        let message = result
            .err()
            .expect("test: error should be present")
            .to_string();
        assert!(
            message.contains("feature"),
            "error message should mention the missing feature: {message}"
        );
    }

    #[cfg(not(feature = "onnx"))]
    #[test]
    fn test_onnx_load_from_file_without_onnx_feature_returns_err() {
        let loader = OnnxLoader::new();
        let result = loader.load_from_file("/nonexistent/path/model.onnx");
        assert!(result.is_err());
        let message = result
            .err()
            .expect("test: error should be present")
            .to_string();
        assert!(
            message.contains("feature"),
            "error message should mention the missing feature: {message}"
        );
    }

    #[cfg(not(feature = "onnx"))]
    #[test]
    fn test_get_onnx_info_without_onnx_feature_returns_err() {
        let result = utils::get_onnx_info("/nonexistent/path/model.onnx");
        assert!(result.is_err());
        let message = result
            .err()
            .expect("test: error should be present")
            .to_string();
        assert!(
            message.contains("feature"),
            "error message should mention the missing feature: {message}"
        );
    }

    /// Exercises `convert_weights` using only hand-built target types (no
    /// protobuf involved at all), so it runs -- and gives real coverage --
    /// in both the default build and the `onnx`-feature build.
    #[test]
    fn test_convert_weights_pure_target_types_f32() {
        let mut graph = OnnxGraph::new("g".to_string());
        let values: [f32; 4] = [10.0, 20.0, 30.0, 40.0];
        let mut info = OnnxTensorInfo::new(
            "weight".to_string(),
            OnnxDataType::Float32,
            vec![Some(2), Some(2)],
        );
        info.data = Some(bytemuck::cast_slice(&values).to_vec());
        graph.add_initializer(info);
        let model = OnnxModel::new(graph);

        let loader = OnnxLoader::new();
        let weights = loader
            .convert_weights::<f32>(&model)
            .expect("test: convert_weights should succeed");
        let tensor = weights
            .get("weight")
            .expect("test: 'weight' should be present");
        assert_eq!(tensor.shape().dims(), &[2, 2]);
        assert_eq!(tensor.data(), &values);
    }
}
