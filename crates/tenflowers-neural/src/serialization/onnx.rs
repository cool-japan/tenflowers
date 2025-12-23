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
    pub fn load_from_file<P: AsRef<Path>>(&self, _path: P) -> Result<OnnxModel> {
        // TODO: Implement actual ONNX protobuf parsing
        // This requires the prost or similar protobuf library
        Err(TensorError::serialization_error_simple(
            "ONNX loading not yet implemented - requires protobuf parsing".to_string(),
        ))
    }

    /// Load ONNX model from bytes
    pub fn load_from_bytes(&self, _bytes: &[u8]) -> Result<OnnxModel> {
        // TODO: Implement actual ONNX protobuf parsing
        Err(TensorError::serialization_error_simple(
            "ONNX loading not yet implemented - requires protobuf parsing".to_string(),
        ))
    }

    /// Convert ONNX weights to TenfloweRS format
    pub fn convert_weights<T>(&self, model: &OnnxModel) -> Result<HashMap<String, Tensor<T>>>
    where
        T: Clone + Default,
    {
        let mut weights = HashMap::new();

        for initializer in &model.graph.initializers {
            // TODO: Implement actual weight conversion
            // This would parse the raw bytes in initializer.data
            // and create Tensor<T> objects
            let _name = &initializer.name;
            let _dtype = &initializer.dtype;
            let _shape = &initializer.shape;
        }

        Ok(weights)
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
    pub fn get_onnx_info<P: AsRef<Path>>(_path: P) -> Result<OnnxModelMetadata> {
        // TODO: Implement lightweight metadata extraction
        Ok(OnnxModelMetadata::default())
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
            OnnxDataType::from_type_code(1).unwrap(),
            OnnxDataType::Float32
        );
        assert_eq!(
            OnnxDataType::from_type_code(11).unwrap(),
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

        let float_attr = OnnxAttribute::Float(3.14);
        assert_eq!(float_attr.as_float(), Some(3.14));
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
}
