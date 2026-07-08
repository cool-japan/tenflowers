use crate::layers::{BatchNorm, Conv2D, Dense, Layer};
use crate::model::Sequential;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
/// TensorFlow SavedModel compatibility layer for TenfloweRS
///
/// This module provides functionality to load and convert TensorFlow SavedModel
/// formats to TenfloweRS models, enabling easy migration and interoperability.
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tenflowers_core::{DType, Result, TensorError};

/// Representation of a TensorFlow SavedModel
#[derive(Debug, Clone)]
pub struct SavedModel {
    /// Model metadata
    pub metadata: SavedModelMetadata,
    /// Function signatures
    pub signatures: HashMap<String, FunctionSignature>,
    /// Model graph definition
    pub graph_def: GraphDef,
    /// Variable values (weights and biases)
    pub variables: HashMap<String, VariableInfo>,
}

/// Metadata information from SavedModel
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SavedModelMetadata {
    /// TensorFlow version used to create the model
    pub tensorflow_version: String,
    /// Model creation timestamp
    pub created_time: Option<i64>,
    /// Model description or name
    pub description: Option<String>,
    /// Tags used when saving the model
    pub tags: Vec<String>,
    /// Input/output tensor specifications
    pub tensor_specs: HashMap<String, TensorSpec>,
}

/// Function signature information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct FunctionSignature {
    /// Input tensor specifications
    pub inputs: HashMap<String, TensorSpec>,
    /// Output tensor specifications  
    pub outputs: HashMap<String, TensorSpec>,
    /// Method name (e.g., "serving_default")
    pub method_name: String,
}

/// Tensor specification
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TensorSpec {
    /// Tensor shape (-1 for dynamic dimensions)
    pub shape: Vec<i64>,
    /// Data type
    pub dtype: String,
    /// Tensor name
    pub name: String,
}

/// Graph definition containing operations and their connections
#[derive(Debug, Clone)]
pub struct GraphDef {
    /// List of operations in the graph
    pub operations: Vec<Operation>,
    /// Input/output mappings
    pub io_mapping: HashMap<String, String>,
}

/// Individual operation in the TensorFlow graph
#[derive(Debug, Clone)]
pub struct Operation {
    /// Operation name
    pub name: String,
    /// Operation type (e.g., "MatMul", "Add", "Conv2D")
    pub op_type: String,
    /// Input tensor names
    pub inputs: Vec<String>,
    /// Output tensor names  
    pub outputs: Vec<String>,
    /// Operation attributes
    pub attributes: HashMap<String, AttributeValue>,
}

/// Attribute value for operations
#[derive(Debug, Clone)]
pub enum AttributeValue {
    String(String),
    Int(i64),
    Float(f32),
    Bool(bool),
    IntList(Vec<i64>),
    FloatList(Vec<f32>),
    StringList(Vec<String>),
}

/// Variable information (weights, biases, etc.)
#[derive(Debug, Clone)]
pub struct VariableInfo {
    /// Variable name
    pub name: String,
    /// Variable shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DType,
    /// Variable data
    pub data: Vec<f32>,
}

/// SavedModel loader and converter
pub struct SavedModelLoader {
    /// Whether to enable verbose logging during conversion
    pub verbose: bool,
    /// Mapping from TensorFlow ops to TenfloweRS ops
    op_mapping: HashMap<String, String>,
}

impl SavedModelLoader {
    /// Create a new SavedModel loader
    pub fn new() -> Self {
        let mut op_mapping = HashMap::new();

        // Basic operation mappings
        op_mapping.insert("MatMul".to_string(), "matmul".to_string());
        op_mapping.insert("Add".to_string(), "add".to_string());
        op_mapping.insert("Sub".to_string(), "sub".to_string());
        op_mapping.insert("Mul".to_string(), "mul".to_string());
        op_mapping.insert("Div".to_string(), "div".to_string());
        op_mapping.insert("Relu".to_string(), "relu".to_string());
        op_mapping.insert("Sigmoid".to_string(), "sigmoid".to_string());
        op_mapping.insert("Tanh".to_string(), "tanh".to_string());
        op_mapping.insert("Softmax".to_string(), "softmax".to_string());
        op_mapping.insert("Conv2D".to_string(), "conv2d".to_string());
        op_mapping.insert("MaxPool".to_string(), "max_pool2d".to_string());
        op_mapping.insert("AvgPool".to_string(), "avg_pool2d".to_string());
        op_mapping.insert("BatchNorm".to_string(), "batch_norm".to_string());
        op_mapping.insert("Reshape".to_string(), "reshape".to_string());
        op_mapping.insert("Transpose".to_string(), "transpose".to_string());

        Self {
            verbose: false,
            op_mapping,
        }
    }

    /// Enable verbose logging
    pub fn with_verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    /// Load a SavedModel from directory
    pub fn load_saved_model<P: AsRef<Path>>(&self, model_dir: P) -> Result<SavedModel> {
        let model_path = model_dir.as_ref();

        if !model_path.exists() {
            return Err(TensorError::invalid_argument(format!(
                "SavedModel directory does not exist: {}",
                model_path.display()
            )));
        }

        if self.verbose {
            println!("Loading SavedModel from: {}", model_path.display());
        }

        // Look for saved_model.pb or saved_model.pbtxt
        let pb_path = model_path.join("saved_model.pb");
        let pbtxt_path = model_path.join("saved_model.pbtxt");

        if pb_path.exists() {
            self.load_from_pb(&pb_path)
        } else if pbtxt_path.exists() {
            self.load_from_pbtxt(&pbtxt_path)
        } else {
            Err(TensorError::invalid_argument(
                "No saved_model.pb or saved_model.pbtxt found in directory".to_string(),
            ))
        }
    }

    /// Load from binary protobuf file.
    ///
    /// Parsing a real TensorFlow `saved_model.pb` requires decoding the
    /// `tensorflow.SavedModel` protobuf schema (MetaGraphDef, GraphDef, NodeDefs,
    /// the attribute map, and the embedded checkpoint variable values). That is a
    /// large subsystem that is on the roadmap but not yet implemented here.
    ///
    /// Returning a hardcoded `SavedModel` (e.g. version "2.8.0") that ignores the
    /// actual bytes on disk would silently fabricate a model and mislead callers
    /// into thinking their file was imported, so an honest error is returned instead.
    fn load_from_pb<P: AsRef<Path>>(&self, pb_path: P) -> Result<SavedModel> {
        let path = pb_path.as_ref();
        if self.verbose {
            println!(
                "TensorFlow binary protobuf import requested for: {}",
                path.display()
            );
        }

        Err(TensorError::not_implemented_simple(format!(
            "TensorFlow SavedModel binary protobuf import is not yet implemented \
             (file: {}). Decoding the tensorflow.SavedModel / GraphDef protobuf schema \
             and its embedded checkpoint variables is a planned subsystem; no model can \
             be produced from this file yet.",
            path.display()
        )))
    }

    /// Load from text protobuf file
    fn load_from_pbtxt<P: AsRef<Path>>(&self, pbtxt_path: P) -> Result<SavedModel> {
        if self.verbose {
            println!("Loading from protobuf format (text)...");
        }

        let content = fs::read_to_string(pbtxt_path).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to read pbtxt file: {e}"))
        })?;

        // Basic text parsing (in real implementation, would use proper protobuf parser)
        self.parse_pbtxt_content(&content)
    }

    /// Parse protobuf text content.
    ///
    /// A faithful parser would tokenize the protobuf text format and reconstruct the
    /// `tensorflow.SavedModel` message (meta graphs, node defs, signatures, and
    /// variable values). That parser is not yet implemented.
    ///
    /// Returning a fabricated `SavedModel` with invented metadata (version "2.8.0",
    /// timestamp `1234567890`) that ignores `content` would misrepresent the input,
    /// so an honest error is returned instead.
    fn parse_pbtxt_content(&self, content: &str) -> Result<SavedModel> {
        if self.verbose {
            println!(
                "TensorFlow text protobuf parse requested ({} bytes); parser not yet \
                 implemented.",
                content.len()
            );
        }

        Err(TensorError::not_implemented_simple(
            "TensorFlow SavedModel text protobuf (pbtxt) parsing is not yet implemented. \
             Reconstructing the tensorflow.SavedModel message from the text format is a \
             planned subsystem; the file's contents cannot be converted into a model yet."
                .to_string(),
        ))
    }

    /// Create default function signatures
    fn create_default_signatures(&self) -> HashMap<String, FunctionSignature> {
        let mut signatures = HashMap::new();

        // Default serving signature
        signatures.insert(
            "serving_default".to_string(),
            FunctionSignature {
                inputs: {
                    let mut inputs = HashMap::new();
                    inputs.insert(
                        "input".to_string(),
                        TensorSpec {
                            shape: vec![-1, 224, 224, 3], // Common image input shape
                            dtype: "float32".to_string(),
                            name: "input".to_string(),
                        },
                    );
                    inputs
                },
                outputs: {
                    let mut outputs = HashMap::new();
                    outputs.insert(
                        "output".to_string(),
                        TensorSpec {
                            shape: vec![-1, 1000], // Common classification output
                            dtype: "float32".to_string(),
                            name: "output".to_string(),
                        },
                    );
                    outputs
                },
                method_name: "serving_default".to_string(),
            },
        );

        signatures
    }

    /// Convert SavedModel to TenfloweRS Sequential model
    pub fn convert_to_sequential(&self, saved_model: &SavedModel) -> Result<Sequential<f32>> {
        if self.verbose {
            println!("Converting SavedModel to TenfloweRS Sequential model...");
        }

        let mut layers: Vec<Box<dyn Layer<f32>>> = Vec::new();

        // Analyze the graph and create corresponding layers
        for operation in &saved_model.graph_def.operations {
            if let Some(layer) =
                self.convert_operation_to_layer(operation, &saved_model.variables)?
            {
                layers.push(layer);
            }
        }

        // If no convertible operations were found we must NOT fabricate an example
        // model (e.g. an arbitrary 224*224*3 -> 128 -> 64 -> 1000 stack); doing so
        // would misrepresent the source graph. Surface an honest error instead.
        if layers.is_empty() {
            return Err(TensorError::not_implemented_simple(
                "No convertible operations were found in the SavedModel graph. \
                 Fabricating a placeholder example model would misrepresent the \
                 source graph, so conversion is reported as failed."
                    .to_string(),
            ));
        }

        Ok(Sequential::new(layers))
    }

    /// Find the variable whose shape describes the weights of an operation.
    ///
    /// TensorFlow lists the weight tensor among an operation's inputs, so we look up
    /// each input name in the variable table and return the first real match. This
    /// lets us derive layer dimensions from actual data rather than fabricating them.
    fn find_weight_variable<'a>(
        &self,
        operation: &Operation,
        variables: &'a HashMap<String, VariableInfo>,
    ) -> Option<&'a VariableInfo> {
        operation
            .inputs
            .iter()
            .find_map(|input_name| variables.get(input_name))
    }

    /// Convert a TensorFlow operation to a TenfloweRS layer.
    ///
    /// Layer dimensions are derived from the operation's actual weight variable shape
    /// (when present in `variables`). If the real dimensions cannot be determined we
    /// return an honest error rather than inventing placeholder sizes that would not
    /// match the source model.
    fn convert_operation_to_layer(
        &self,
        operation: &Operation,
        variables: &HashMap<String, VariableInfo>,
    ) -> Result<Option<Box<dyn Layer<f32>>>> {
        match operation.op_type.as_str() {
            "MatMul" | "Dense" => {
                // A Dense/MatMul weight is 2-D with shape [input_features, output_features].
                let weight = self
                    .find_weight_variable(operation, variables)
                    .ok_or_else(|| {
                        TensorError::not_implemented_simple(format!(
                            "Cannot convert operation '{}' (MatMul/Dense): no weight variable \
                         was found among its inputs, so the input/output dimensions are \
                         unknown. Inventing placeholder dimensions would not match the \
                         source model.",
                            operation.name
                        ))
                    })?;

                if weight.shape.len() != 2 {
                    return Err(TensorError::invalid_argument(format!(
                        "Operation '{}' (MatMul/Dense) has a weight variable '{}' with \
                         shape {:?}; expected a 2-D [in, out] weight.",
                        operation.name, weight.name, weight.shape
                    )));
                }

                let input_features = weight.shape[0];
                let output_features = weight.shape[1];

                if self.verbose {
                    println!(
                        "Converting {} to Dense layer ({} -> {})",
                        operation.name, input_features, output_features
                    );
                }

                Ok(Some(Box::new(Dense::new(
                    input_features,
                    output_features,
                    true,
                ))))
            }
            "Conv2D" => {
                // A Conv2D kernel is 4-D. TensorFlow stores it as
                // [kernel_h, kernel_w, in_channels, out_channels].
                let weight = self
                    .find_weight_variable(operation, variables)
                    .ok_or_else(|| {
                        TensorError::not_implemented_simple(format!(
                            "Cannot convert operation '{}' (Conv2D): no kernel variable was \
                         found among its inputs, so the channel/kernel dimensions are \
                         unknown. Inventing placeholder dimensions would not match the \
                         source model.",
                            operation.name
                        ))
                    })?;

                if weight.shape.len() != 4 {
                    return Err(TensorError::invalid_argument(format!(
                        "Operation '{}' (Conv2D) has a kernel variable '{}' with shape \
                         {:?}; expected a 4-D [kh, kw, in, out] kernel.",
                        operation.name, weight.name, weight.shape
                    )));
                }

                let kernel_h = weight.shape[0];
                let kernel_w = weight.shape[1];
                let in_channels = weight.shape[2];
                let out_channels = weight.shape[3];

                if self.verbose {
                    println!(
                        "Converting {} to Conv2D layer ({}x{}, {} -> {})",
                        operation.name, kernel_h, kernel_w, in_channels, out_channels
                    );
                }

                Ok(Some(Box::new(Conv2D::new(
                    in_channels,
                    out_channels,
                    (kernel_h, kernel_w),
                    (1, 1),             // stride (real strides require attribute decoding)
                    "same".to_string(), // padding
                    true,               // use_bias
                ))))
            }
            "BatchNorm" => {
                // BatchNorm parameters (gamma/beta/mean/variance) are 1-D vectors whose
                // length equals the number of features.
                let weight = self
                    .find_weight_variable(operation, variables)
                    .ok_or_else(|| {
                        TensorError::not_implemented_simple(format!(
                            "Cannot convert operation '{}' (BatchNorm): no parameter variable \
                         was found among its inputs, so the feature count is unknown. \
                         Inventing a placeholder feature count would not match the source \
                         model.",
                            operation.name
                        ))
                    })?;

                let num_features = weight.shape.iter().copied().product::<usize>();
                if num_features == 0 {
                    return Err(TensorError::invalid_argument(format!(
                        "Operation '{}' (BatchNorm) has a parameter variable '{}' with an \
                         empty shape {:?}; cannot determine the feature count.",
                        operation.name, weight.name, weight.shape
                    )));
                }

                if self.verbose {
                    println!(
                        "Converting {} to BatchNorm layer ({} features)",
                        operation.name, num_features
                    );
                }

                Ok(Some(Box::new(BatchNorm::new(num_features))))
            }
            "Relu" | "Sigmoid" | "Tanh" | "Softmax" => {
                // Activation functions are typically handled as part of other layers
                // or as separate functional operations in TenfloweRS
                if self.verbose {
                    println!(
                        "Skipping activation function {} (handled separately)",
                        operation.name
                    );
                }
                Ok(None)
            }
            _ => {
                if self.verbose {
                    println!(
                        "Unsupported operation type: {} ({})",
                        operation.op_type, operation.name
                    );
                }
                Ok(None)
            }
        }
    }

    /// Load variables (weights/biases) from TensorFlow checkpoint files.
    ///
    /// TensorFlow checkpoints are stored across a `.index` file and one or more
    /// sharded `.data-*` files using the BundleHeader/BundleEntry protobuf format.
    /// Decoding that format and the raw tensor payloads is a planned subsystem.
    ///
    /// Returning an empty `HashMap` would silently claim the checkpoint contained no
    /// variables, so an honest error is returned instead.
    pub fn load_variables<P: AsRef<Path>>(
        &self,
        checkpoint_dir: P,
    ) -> Result<HashMap<String, VariableInfo>> {
        let checkpoint_path = checkpoint_dir.as_ref();

        if self.verbose {
            println!(
                "TensorFlow checkpoint variable load requested for: {}",
                checkpoint_path.display()
            );
        }

        Err(TensorError::not_implemented_simple(format!(
            "TensorFlow checkpoint variable loading is not yet implemented (path: {}). \
             Decoding the checkpoint bundle (.index / .data-* BundleEntry protobufs) and \
             the raw tensor payloads is a planned subsystem; returning empty variables \
             would falsely report an empty checkpoint.",
            checkpoint_path.display()
        )))
    }
}

impl Default for SavedModelLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level API for loading TensorFlow SavedModels
pub fn load_tensorflow_model<P: AsRef<Path>>(model_dir: P) -> Result<Sequential<f32>> {
    let loader = SavedModelLoader::new().with_verbose();
    let saved_model = loader.load_saved_model(model_dir)?;
    loader.convert_to_sequential(&saved_model)
}

/// Load TensorFlow SavedModel with custom configuration
pub fn load_tensorflow_model_with_config<P: AsRef<Path>>(
    model_dir: P,
    verbose: bool,
) -> Result<(Sequential<f32>, SavedModelMetadata)> {
    let loader = if verbose {
        SavedModelLoader::new().with_verbose()
    } else {
        SavedModelLoader::new()
    };

    let saved_model = loader.load_saved_model(model_dir)?;
    let metadata = saved_model.metadata.clone();
    let model = loader.convert_to_sequential(&saved_model)?;

    Ok((model, metadata))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_saved_model_loader_creation() {
        let loader = SavedModelLoader::new();
        assert!(!loader.verbose);
        assert!(loader.op_mapping.contains_key("MatMul"));
        assert!(loader.op_mapping.contains_key("Conv2D"));
    }

    #[test]
    fn test_saved_model_loader_verbose() {
        let loader = SavedModelLoader::new().with_verbose();
        assert!(loader.verbose);
    }

    #[test]
    fn test_tensor_spec_creation() {
        let spec = TensorSpec {
            shape: vec![-1, 224, 224, 3],
            dtype: "float32".to_string(),
            name: "input_image".to_string(),
        };

        assert_eq!(spec.shape, vec![-1, 224, 224, 3]);
        assert_eq!(spec.dtype, "float32");
        assert_eq!(spec.name, "input_image");
    }

    #[test]
    fn test_function_signature_creation() {
        let loader = SavedModelLoader::new();
        let signatures = loader.create_default_signatures();

        assert!(signatures.contains_key("serving_default"));
        let sig = &signatures["serving_default"];
        assert!(sig.inputs.contains_key("input"));
        assert!(sig.outputs.contains_key("output"));
    }

    #[test]
    fn test_load_nonexistent_model() {
        let loader = SavedModelLoader::new();
        let result = loader.load_saved_model("/nonexistent/path");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_pbtxt_returns_honest_error() {
        // A real saved_model.pbtxt must NOT be silently turned into a fabricated model.
        // Loading it should now surface an honest "not implemented" error rather than
        // returning an invented version/timestamp that ignores the file contents.
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let model_dir = temp_dir.path().join("test_model");
        fs::create_dir_all(&model_dir).expect("test: directory creation should succeed");

        let pbtxt_content = r#"
meta_graphs {
  meta_info_def {
    tags: "serve"
    tensorflow_version: "2.8.0"
  }
}
"#;
        fs::write(model_dir.join("saved_model.pbtxt"), pbtxt_content)
            .expect("test: file write should succeed");

        let loader = SavedModelLoader::new();
        let result = loader.load_saved_model(&model_dir);
        assert!(
            result.is_err(),
            "pbtxt import must return an honest error, not a fabricated SavedModel"
        );
    }

    #[test]
    fn test_load_pb_returns_honest_error() {
        // A binary saved_model.pb must also surface an honest error rather than a
        // hardcoded model that ignores the bytes on disk.
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let model_dir = temp_dir.path().join("test_model_pb");
        fs::create_dir_all(&model_dir).expect("test: directory creation should succeed");

        // Write arbitrary bytes; the importer must not pretend to parse them.
        fs::write(model_dir.join("saved_model.pb"), [0u8, 1, 2, 3, 4])
            .expect("test: file write should succeed");

        let loader = SavedModelLoader::new();
        let result = loader.load_saved_model(&model_dir);
        assert!(
            result.is_err(),
            "pb import must return an honest error, not a fabricated SavedModel"
        );
    }

    #[test]
    fn test_convert_to_sequential_without_weights_errors() {
        // Converting a graph whose operations lack weight variables must NOT fabricate
        // an example model with invented dimensions.
        let loader = SavedModelLoader::new();

        let saved_model = SavedModel {
            metadata: SavedModelMetadata {
                tensorflow_version: "unknown".to_string(),
                created_time: None,
                description: Some("Test model".to_string()),
                tags: vec!["serve".to_string()],
                tensor_specs: HashMap::new(),
            },
            signatures: HashMap::new(),
            graph_def: GraphDef {
                operations: vec![Operation {
                    name: "dense1".to_string(),
                    op_type: "MatMul".to_string(),
                    inputs: vec!["input".to_string()],
                    outputs: vec!["dense1_output".to_string()],
                    attributes: HashMap::new(),
                }],
                io_mapping: HashMap::new(),
            },
            variables: HashMap::new(),
        };

        let result = loader.convert_to_sequential(&saved_model);
        assert!(
            result.is_err(),
            "conversion without real weight dimensions must error, not fabricate sizes"
        );
    }

    #[test]
    fn test_convert_to_sequential_with_real_weights() {
        // When a real weight variable is present, the Dense layer must be built from
        // its actual [in, out] shape rather than placeholder dimensions.
        let loader = SavedModelLoader::new();

        let mut variables = HashMap::new();
        variables.insert(
            "dense1/kernel".to_string(),
            VariableInfo {
                name: "dense1/kernel".to_string(),
                shape: vec![16, 7],
                dtype: DType::Float32,
                data: vec![0.0_f32; 16 * 7],
            },
        );

        let saved_model = SavedModel {
            metadata: SavedModelMetadata {
                tensorflow_version: "unknown".to_string(),
                created_time: None,
                description: Some("Test model".to_string()),
                tags: vec!["serve".to_string()],
                tensor_specs: HashMap::new(),
            },
            signatures: HashMap::new(),
            graph_def: GraphDef {
                operations: vec![Operation {
                    name: "dense1".to_string(),
                    op_type: "MatMul".to_string(),
                    inputs: vec!["input".to_string(), "dense1/kernel".to_string()],
                    outputs: vec!["dense1_output".to_string()],
                    attributes: HashMap::new(),
                }],
                io_mapping: HashMap::new(),
            },
            variables,
        };

        let model = loader
            .convert_to_sequential(&saved_model)
            .expect("test: conversion with real weights should succeed");
        // The single Dense layer must expose a weight tensor matching the source shape.
        let params = model.parameters();
        assert!(
            params.iter().any(|p| p.shape().dims() == [16, 7]),
            "Dense layer dimensions must be derived from the real [16, 7] weight"
        );
    }

    #[test]
    fn test_operation_conversion() {
        let loader = SavedModelLoader::new();

        // Without a weight variable the MatMul conversion must error (no fabrication).
        let empty_variables = HashMap::new();
        let matmul_op = Operation {
            name: "dense1".to_string(),
            op_type: "MatMul".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            attributes: HashMap::new(),
        };
        let result = loader.convert_operation_to_layer(&matmul_op, &empty_variables);
        assert!(
            result.is_err(),
            "MatMul conversion without a weight variable must error, not fabricate dims"
        );

        // With a real weight variable, the conversion yields a layer.
        let mut variables = HashMap::new();
        variables.insert(
            "kernel".to_string(),
            VariableInfo {
                name: "kernel".to_string(),
                shape: vec![5, 3],
                dtype: DType::Float32,
                data: vec![0.0_f32; 15],
            },
        );
        let matmul_with_weight = Operation {
            name: "dense2".to_string(),
            op_type: "MatMul".to_string(),
            inputs: vec!["input".to_string(), "kernel".to_string()],
            outputs: vec!["output".to_string()],
            attributes: HashMap::new(),
        };
        let layer = loader
            .convert_operation_to_layer(&matmul_with_weight, &variables)
            .expect("test: conversion should succeed with real weight");
        assert!(layer.is_some());

        // Activation ops remain unconverted (handled separately).
        let relu_op = Operation {
            name: "relu1".to_string(),
            op_type: "Relu".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            attributes: HashMap::new(),
        };
        let result = loader.convert_operation_to_layer(&relu_op, &variables);
        assert!(result.is_ok());
        assert!(result.expect("test: result should be valid").is_none()); // ReLU is handled separately
    }

    #[test]
    fn test_load_variables_returns_honest_error() {
        // Checkpoint loading is not implemented; it must error rather than silently
        // returning an empty variable map (which would falsely report no weights).
        let loader = SavedModelLoader::new();
        let result = loader.load_variables("/nonexistent/checkpoint");
        assert!(result.is_err());
    }

    #[test]
    fn test_high_level_api() {
        // Test would require actual SavedModel files
        // For now, just test that the function exists and handles errors gracefully
        let result = load_tensorflow_model("/nonexistent/path");
        assert!(result.is_err());
    }
}
