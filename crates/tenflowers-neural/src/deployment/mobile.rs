use crate::model::{Model, Sequential};
use num_traits::{Float, One, Zero};
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
/// Mobile backend support for CoreML and TensorFlow Lite export.
///
/// This module provides functionality to export TenfloweRS models to mobile-optimized formats
/// for deployment on iOS (CoreML) and Android (TensorFlow Lite) devices.
use std::collections::HashMap;
use tenflowers_core::Result;

/// Mobile backend types for model export
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum MobileBackend {
    /// Apple CoreML format for iOS deployment
    CoreML,
    /// TensorFlow Lite format for Android/edge deployment
    TensorFlowLite,
    /// ONNX Runtime Mobile format
    OnnxMobile,
}

/// Configuration for mobile model export
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MobileExportConfig {
    /// Target mobile backend
    pub backend: MobileBackend,
    /// Target device architecture (e.g., "arm64", "x86_64")
    pub target_arch: String,
    /// Optimization level for mobile deployment
    pub optimization_level: OptimizationLevel,
    /// Whether to quantize weights for smaller model size
    pub quantize_weights: bool,
    /// Input tensor shapes for static optimization
    pub input_shapes: Vec<Vec<usize>>,
    /// Output tensor names
    pub output_names: Vec<String>,
    /// Model metadata for deployment
    pub metadata: MobileModelMetadata,
}

/// Optimization levels for mobile deployment
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum OptimizationLevel {
    /// No optimization - preserve full precision and functionality
    None,
    /// Basic optimization - safe optimizations that don't affect accuracy
    Basic,
    /// Aggressive optimization - may trade some accuracy for performance
    Aggressive,
    /// Ultra optimization - maximum performance optimizations
    Ultra,
}

impl Default for OptimizationLevel {
    fn default() -> Self {
        Self::Basic
    }
}

/// Metadata for mobile model deployment
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MobileModelMetadata {
    /// Model name for identification
    pub name: String,
    /// Model version string
    pub version: String,
    /// Model description
    pub description: String,
    /// Author/organization
    pub author: String,
    /// License information
    pub license: String,
    /// Input preprocessing requirements
    pub preprocessing: Vec<PreprocessingStep>,
    /// Post-processing requirements
    pub postprocessing: Vec<PostprocessingStep>,
}

impl Default for MobileModelMetadata {
    fn default() -> Self {
        Self {
            name: "TenfloweRS_Model".to_string(),
            version: "1.0.0".to_string(),
            description: "Model exported from TenfloweRS".to_string(),
            author: "TenfloweRS".to_string(),
            license: "MIT".to_string(),
            preprocessing: Vec::new(),
            postprocessing: Vec::new(),
        }
    }
}

/// Preprocessing step for input data
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PreprocessingStep {
    /// Type of preprocessing (e.g., "normalize", "resize", "crop")
    pub step_type: String,
    /// Parameters for the preprocessing step
    pub parameters: HashMap<String, String>,
}

/// Post-processing step for output data
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PostprocessingStep {
    /// Type of post-processing (e.g., "softmax", "argmax", "threshold")
    pub step_type: String,
    /// Parameters for the post-processing step
    pub parameters: HashMap<String, String>,
}

/// Exported mobile model container
#[derive(Debug)]
pub struct MobileModel {
    /// Serialized model data in target format
    pub model_data: Vec<u8>,
    /// Export configuration used
    pub config: MobileExportConfig,
    /// Export statistics
    pub stats: MobileExportStats,
}

/// Statistics about mobile model export
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MobileExportStats {
    /// Original model size in bytes
    pub original_size: usize,
    /// Exported model size in bytes
    pub exported_size: usize,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Estimated inference time on target device (ms)
    pub estimated_inference_time: Option<f32>,
    /// Estimated accuracy impact from optimizations
    pub accuracy_impact: Option<f32>,
}

impl MobileExportStats {
    /// Calculate size reduction percentage
    pub fn size_reduction_percent(&self) -> f32 {
        if self.original_size == 0 {
            0.0
        } else {
            ((self.original_size - self.exported_size) as f32 / self.original_size as f32) * 100.0
        }
    }
}

/// CoreML model exporter
pub struct CoreMLExporter {
    config: MobileExportConfig,
}

impl CoreMLExporter {
    /// Create a new CoreML exporter
    pub fn new(config: MobileExportConfig) -> Self {
        Self { config }
    }

    /// Export a Sequential model to CoreML format
    pub fn export<T>(&self, model: &Sequential<T>) -> Result<MobileModel>
    where
        T: Float
            + Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // CoreML export implementation
        let original_size = self.estimate_model_size(model);

        // In a full implementation, this would:
        // 1. Convert TenfloweRS layers to CoreML layer specifications
        // 2. Create CoreML model protobuf
        // 3. Apply CoreML-specific optimizations
        // 4. Serialize to .mlmodel format

        let exported_data = self.create_coreml_model(model)?;
        let exported_size = exported_data.len();

        let stats = MobileExportStats {
            original_size,
            exported_size,
            compression_ratio: if exported_size > 0 {
                original_size as f32 / exported_size as f32
            } else {
                1.0
            },
            estimated_inference_time: Some(self.estimate_coreml_inference_time(model)),
            accuracy_impact: Some(self.estimate_accuracy_impact()),
        };

        Ok(MobileModel {
            model_data: exported_data,
            config: self.config.clone(),
            stats,
        })
    }

    /// Create CoreML model data
    fn create_coreml_model<T>(&self, model: &Sequential<T>) -> Result<Vec<u8>>
    where
        T: Float
            + Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Simplified CoreML model generation
        // In practice, this would use Apple's CoreML protobuf definitions

        let mut coreml_data = Vec::new();

        // CoreML header
        coreml_data.extend_from_slice(b"COREML");
        coreml_data.extend_from_slice(&[1, 0, 0, 0]); // Version

        // Model metadata (simplified serialization)
        let metadata = format!(
            "{{\"name\":\"{}\",\"version\":\"{}\",\"description\":\"{}\"}}",
            self.config.metadata.name,
            self.config.metadata.version,
            self.config.metadata.description
        );
        let metadata_bytes = metadata.as_bytes();
        coreml_data.extend_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        coreml_data.extend_from_slice(metadata_bytes);

        // Layer definitions (simplified)
        let param_count = model.parameters().len();
        coreml_data.extend_from_slice(&(param_count as u32).to_le_bytes());

        // Serialize layer information
        for (i, param) in model.parameters().iter().enumerate() {
            // Layer type identifier (simplified)
            coreml_data.push(1); // Dense layer type

            // Layer index
            coreml_data.extend_from_slice(&(i as u32).to_le_bytes());

            // Parameter shape (simplified - just record size)
            let param_size = param.shape().elements();
            coreml_data.extend_from_slice(&(param_size as u32).to_le_bytes());
        }

        // Apply quantization if enabled
        if self.config.quantize_weights {
            self.apply_weight_quantization(&mut coreml_data)?;
        }

        Ok(coreml_data)
    }

    /// Estimate inference time for CoreML model
    fn estimate_coreml_inference_time<T>(&self, model: &Sequential<T>) -> f32
    where
        T: Float
            + Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Rough estimation based on model complexity
        let param_count = model.parameters().len();
        let base_time = match self.config.target_arch.as_str() {
            "arm64" => 2.0,  // iPhone/iPad
            "x86_64" => 1.5, // Mac
            _ => 2.5,
        };

        base_time + (param_count as f32 * 0.001) // 1μs per parameter roughly
    }

    /// Estimate accuracy impact from optimizations
    fn estimate_accuracy_impact(&self) -> f32 {
        match self.config.optimization_level {
            OptimizationLevel::None => 0.0,
            OptimizationLevel::Basic => -0.01, // 1% accuracy loss
            OptimizationLevel::Aggressive => -0.03, // 3% accuracy loss
            OptimizationLevel::Ultra => -0.05, // 5% accuracy loss
        }
    }

    /// Apply weight quantization to model data
    fn apply_weight_quantization(&self, model_data: &mut Vec<u8>) -> Result<()> {
        // Simplified quantization - in practice would quantize actual weights
        // For now, just add quantization metadata
        model_data.extend_from_slice(b"QUANTIZED");
        Ok(())
    }

    /// Estimate model size
    fn estimate_model_size<T>(&self, model: &Sequential<T>) -> usize
    where
        T: Float
            + Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        model.parameters().len() * std::mem::size_of::<f32>()
    }
}

/// TensorFlow Lite model exporter
pub struct TensorFlowLiteExporter {
    config: MobileExportConfig,
}

impl TensorFlowLiteExporter {
    /// Create a new TensorFlow Lite exporter
    pub fn new(config: MobileExportConfig) -> Self {
        Self { config }
    }

    /// Export a Sequential model to TensorFlow Lite format
    pub fn export<T>(&self, model: &Sequential<T>) -> Result<MobileModel>
    where
        T: Float
            + Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // TFLite export implementation
        let original_size = self.estimate_model_size(model);

        // In a full implementation, this would:
        // 1. Convert TenfloweRS layers to TFLite flatbuffer format
        // 2. Apply TFLite optimization passes
        // 3. Generate .tflite file with quantization if enabled

        let exported_data = self.create_tflite_model(model)?;
        let exported_size = exported_data.len();

        let stats = MobileExportStats {
            original_size,
            exported_size,
            compression_ratio: if exported_size > 0 {
                original_size as f32 / exported_size as f32
            } else {
                1.0
            },
            estimated_inference_time: Some(self.estimate_tflite_inference_time(model)),
            accuracy_impact: Some(self.estimate_accuracy_impact()),
        };

        Ok(MobileModel {
            model_data: exported_data,
            config: self.config.clone(),
            stats,
        })
    }

    /// Create TensorFlow Lite model data
    fn create_tflite_model<T>(&self, model: &Sequential<T>) -> Result<Vec<u8>>
    where
        T: Float
            + Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Simplified TFLite model generation
        // In practice, this would use TensorFlow Lite's flatbuffer definitions

        let mut tflite_data = Vec::new();

        // TFLite header (simplified)
        tflite_data.extend_from_slice(b"TFL3"); // Magic bytes
        tflite_data.extend_from_slice(&[0, 0, 0, 1]); // Version

        // Model schema version
        tflite_data.extend_from_slice(&[3, 0, 0, 0]); // Schema version 3

        // Subgraphs (simplified - single subgraph)
        tflite_data.extend_from_slice(&[1, 0, 0, 0]); // Number of subgraphs

        // Operators and tensors (simplified)
        let param_count = model.parameters().len();
        tflite_data.extend_from_slice(&(param_count as u32).to_le_bytes());

        // Serialize layer information in TFLite format
        for (i, param) in model.parameters().iter().enumerate() {
            // Operator code
            tflite_data.push(1); // FULLY_CONNECTED op code

            // Tensor info
            tflite_data.extend_from_slice(&(i as u32).to_le_bytes());

            // Shape info
            let param_size = param.shape().elements();
            tflite_data.extend_from_slice(&(param_size as u32).to_le_bytes());

            // Type info (float32)
            tflite_data.push(1); // FLOAT32 type
        }

        // Apply TFLite-specific optimizations
        if self.config.quantize_weights {
            self.apply_tflite_quantization(&mut tflite_data)?;
        }

        // Optimization passes based on level
        self.apply_optimization_passes(&mut tflite_data)?;

        Ok(tflite_data)
    }

    /// Estimate inference time for TFLite model
    fn estimate_tflite_inference_time<T>(&self, model: &Sequential<T>) -> f32
    where
        T: Float
            + Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Rough estimation based on model complexity
        let param_count = model.parameters().len();
        let base_time = match self.config.target_arch.as_str() {
            "arm64" => 3.0,  // Android ARM
            "x86_64" => 2.0, // x86 Android emulator
            _ => 4.0,
        };

        base_time + (param_count as f32 * 0.002) // 2μs per parameter roughly
    }

    /// Estimate accuracy impact from optimizations
    fn estimate_accuracy_impact(&self) -> f32 {
        let mut impact = match self.config.optimization_level {
            OptimizationLevel::None => 0.0,
            OptimizationLevel::Basic => -0.005, // 0.5% accuracy loss
            OptimizationLevel::Aggressive => -0.02, // 2% accuracy loss
            OptimizationLevel::Ultra => -0.04,  // 4% accuracy loss
        };

        // Additional impact from quantization
        if self.config.quantize_weights {
            impact -= 0.01; // Additional 1% from quantization
        }

        impact
    }

    /// Apply TFLite-specific quantization
    fn apply_tflite_quantization(&self, model_data: &mut Vec<u8>) -> Result<()> {
        // Add quantization metadata
        model_data.extend_from_slice(b"QUANTIZED_INT8");
        Ok(())
    }

    /// Apply TFLite optimization passes
    fn apply_optimization_passes(&self, model_data: &mut Vec<u8>) -> Result<()> {
        match self.config.optimization_level {
            OptimizationLevel::None => {}
            OptimizationLevel::Basic => {
                model_data.extend_from_slice(b"OPT_BASIC");
            }
            OptimizationLevel::Aggressive => {
                model_data.extend_from_slice(b"OPT_AGGRESSIVE");
            }
            OptimizationLevel::Ultra => {
                model_data.extend_from_slice(b"OPT_ULTRA");
            }
        }
        Ok(())
    }

    /// Estimate model size
    fn estimate_model_size<T>(&self, model: &Sequential<T>) -> usize
    where
        T: Float
            + Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        model.parameters().len() * std::mem::size_of::<f32>()
    }
}

/// ONNX Mobile model exporter
pub struct OnnxMobileExporter {
    config: MobileExportConfig,
}

impl OnnxMobileExporter {
    /// Create a new ONNX Mobile exporter
    pub fn new(config: MobileExportConfig) -> Self {
        Self { config }
    }

    /// Export a Sequential model to ONNX Mobile format
    pub fn export<T>(&self, model: &Sequential<T>) -> Result<MobileModel>
    where
        T: Float
            + Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // ONNX Mobile export implementation
        let original_size = self.estimate_model_size(model);

        // In a full implementation, this would:
        // 1. Convert TenfloweRS layers to ONNX operator graph
        // 2. Apply ONNX Runtime Mobile optimization passes
        // 3. Generate optimized .onnx file with quantization if enabled

        let exported_data = self.create_onnx_mobile_model(model)?;
        let exported_size = exported_data.len();

        let stats = MobileExportStats {
            original_size,
            exported_size,
            compression_ratio: if exported_size > 0 {
                original_size as f32 / exported_size as f32
            } else {
                1.0
            },
            estimated_inference_time: Some(self.estimate_onnx_inference_time(model)),
            accuracy_impact: Some(self.estimate_accuracy_impact()),
        };

        Ok(MobileModel {
            model_data: exported_data,
            config: self.config.clone(),
            stats,
        })
    }

    /// Create ONNX Mobile model data
    fn create_onnx_mobile_model<T>(&self, model: &Sequential<T>) -> Result<Vec<u8>>
    where
        T: Float
            + Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // Simplified ONNX Mobile model generation
        // In practice, this would use ONNX protobuf definitions and Runtime Mobile optimizations

        let mut onnx_data = Vec::new();

        // ONNX protobuf header (simplified)
        onnx_data.extend_from_slice(b"ONNX");
        onnx_data.extend_from_slice(&[1, 12, 0, 0]); // ONNX version 1.12

        // Producer info for ONNX Mobile
        let producer_info = format!(
            "{{\"producer_name\":\"TenfloweRS\",\"producer_version\":\"0.1.0\",\"model_version\":\"{}\"}}",
            self.config.metadata.version
        );
        let producer_bytes = producer_info.as_bytes();
        onnx_data.extend_from_slice(&(producer_bytes.len() as u32).to_le_bytes());
        onnx_data.extend_from_slice(producer_bytes);

        // Model metadata for ONNX
        let model_metadata = format!(
            "{{\"name\":\"{}\",\"description\":\"{}\",\"domain\":\"ai.tenflowers\"}}",
            self.config.metadata.name, self.config.metadata.description
        );
        let metadata_bytes = model_metadata.as_bytes();
        onnx_data.extend_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        onnx_data.extend_from_slice(metadata_bytes);

        // Graph definition (simplified)
        onnx_data.extend_from_slice(b"GRAPH");

        // Input definitions
        onnx_data.extend_from_slice(&(self.config.input_shapes.len() as u32).to_le_bytes());
        for (i, shape) in self.config.input_shapes.iter().enumerate() {
            let input_name = format!("input_{i}");
            onnx_data.extend_from_slice(&(input_name.len() as u32).to_le_bytes());
            onnx_data.extend_from_slice(input_name.as_bytes());

            // Shape information
            onnx_data.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for &dim in shape {
                onnx_data.extend_from_slice(&(dim as u32).to_le_bytes());
            }
        }

        // Node definitions (operators)
        let param_count = model.parameters().len();
        onnx_data.extend_from_slice(&(param_count as u32).to_le_bytes());

        for (i, param) in model.parameters().iter().enumerate() {
            // ONNX operator type (simplified)
            let op_type = "MatMul"; // Dense layers become MatMul + Add operations
            onnx_data.extend_from_slice(&(op_type.len() as u32).to_le_bytes());
            onnx_data.extend_from_slice(op_type.as_bytes());

            // Node name
            let node_name = format!("layer_{i}");
            onnx_data.extend_from_slice(&(node_name.len() as u32).to_le_bytes());
            onnx_data.extend_from_slice(node_name.as_bytes());

            // Input/output connections (simplified)
            let input_name = if i == 0 {
                "input_0".to_string()
            } else {
                format!("layer_{}_output", i - 1)
            };
            let output_name = format!("layer_{i}_output");

            onnx_data.extend_from_slice(&(input_name.len() as u32).to_le_bytes());
            onnx_data.extend_from_slice(input_name.as_bytes());
            onnx_data.extend_from_slice(&(output_name.len() as u32).to_le_bytes());
            onnx_data.extend_from_slice(output_name.as_bytes());

            // Weight tensor information
            let param_size = param.shape().elements();
            onnx_data.extend_from_slice(&(param_size as u32).to_le_bytes());
        }

        // Output definitions
        onnx_data.extend_from_slice(&(self.config.output_names.len() as u32).to_le_bytes());
        for output_name in &self.config.output_names {
            onnx_data.extend_from_slice(&(output_name.len() as u32).to_le_bytes());
            onnx_data.extend_from_slice(output_name.as_bytes());
        }

        // Apply ONNX Runtime Mobile optimizations
        self.apply_onnx_mobile_optimizations(&mut onnx_data)?;

        // Apply quantization if enabled
        if self.config.quantize_weights {
            self.apply_onnx_quantization(&mut onnx_data)?;
        }

        Ok(onnx_data)
    }

    /// Apply ONNX Runtime Mobile specific optimizations
    fn apply_onnx_mobile_optimizations(&self, model_data: &mut Vec<u8>) -> Result<()> {
        match self.config.optimization_level {
            OptimizationLevel::None => {}
            OptimizationLevel::Basic => {
                // Basic optimizations: constant folding, redundant node elimination
                model_data.extend_from_slice(b"ORT_MOBILE_BASIC");
            }
            OptimizationLevel::Aggressive => {
                // Aggressive optimizations: operator fusion, layout optimizations
                model_data.extend_from_slice(b"ORT_MOBILE_AGGRESSIVE");
                model_data.extend_from_slice(b"FUSION_ENABLED");
            }
            OptimizationLevel::Ultra => {
                // Ultra optimizations: all above + precision optimizations
                model_data.extend_from_slice(b"ORT_MOBILE_ULTRA");
                model_data.extend_from_slice(b"FUSION_ENABLED");
                model_data.extend_from_slice(b"PRECISION_OPT");
            }
        }
        Ok(())
    }

    /// Apply ONNX-specific quantization
    fn apply_onnx_quantization(&self, model_data: &mut Vec<u8>) -> Result<()> {
        // ONNX Runtime supports various quantization schemes
        model_data.extend_from_slice(b"ONNX_QDQ_INT8"); // Quantize-Dequantize INT8
        Ok(())
    }

    /// Estimate inference time for ONNX Mobile model
    fn estimate_onnx_inference_time<T>(&self, model: &Sequential<T>) -> f32
    where
        T: Float
            + Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // ONNX Runtime Mobile has excellent optimization, typically faster than TFLite
        let param_count = model.parameters().len();
        let base_time = match self.config.target_arch.as_str() {
            "arm64" => 1.8,  // ONNX Runtime Mobile ARM64 optimizations
            "x86_64" => 1.2, // x86 optimizations
            _ => 2.5,
        };

        let per_param_time = match self.config.optimization_level {
            OptimizationLevel::None => 0.003,
            OptimizationLevel::Basic => 0.002,
            OptimizationLevel::Aggressive => 0.0015,
            OptimizationLevel::Ultra => 0.001,
        };

        base_time + (param_count as f32 * per_param_time)
    }

    /// Estimate accuracy impact from optimizations
    fn estimate_accuracy_impact(&self) -> f32 {
        let mut impact = match self.config.optimization_level {
            OptimizationLevel::None => 0.0,
            OptimizationLevel::Basic => -0.002, // 0.2% accuracy loss (ONNX RT is very precise)
            OptimizationLevel::Aggressive => -0.008, // 0.8% accuracy loss
            OptimizationLevel::Ultra => -0.015, // 1.5% accuracy loss
        };

        // ONNX quantization is typically more accurate than other frameworks
        if self.config.quantize_weights {
            impact -= 0.005; // Only 0.5% additional loss from quantization
        }

        impact
    }

    /// Estimate model size
    fn estimate_model_size<T>(&self, model: &Sequential<T>) -> usize
    where
        T: Float
            + Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        // ONNX models tend to be slightly larger due to metadata but compress well
        let base_size = model.parameters().len() * std::mem::size_of::<f32>();
        let metadata_overhead = base_size / 20; // ~5% metadata overhead
        base_size + metadata_overhead
    }
}

/// High-level mobile export API
pub struct MobileExporter;

impl MobileExporter {
    /// Export a model to the specified mobile backend
    pub fn export<T>(model: &Sequential<T>, config: MobileExportConfig) -> Result<MobileModel>
    where
        T: Float
            + Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        match config.backend {
            MobileBackend::CoreML => {
                let exporter = CoreMLExporter::new(config);
                exporter.export(model)
            }
            MobileBackend::TensorFlowLite => {
                let exporter = TensorFlowLiteExporter::new(config);
                exporter.export(model)
            }
            MobileBackend::OnnxMobile => {
                let exporter = OnnxMobileExporter::new(config);
                exporter.export(model)
            }
        }
    }

    /// Create a default mobile export configuration
    pub fn default_config(backend: MobileBackend) -> MobileExportConfig {
        MobileExportConfig {
            backend,
            target_arch: "arm64".to_string(),
            optimization_level: OptimizationLevel::Basic,
            quantize_weights: true,
            input_shapes: vec![vec![1, 224, 224, 3]], // Default image input
            output_names: vec!["output".to_string()],
            metadata: MobileModelMetadata::default(),
        }
    }

    /// Create an optimized mobile export configuration
    pub fn optimized_config(backend: MobileBackend) -> MobileExportConfig {
        MobileExportConfig {
            backend,
            target_arch: "arm64".to_string(),
            optimization_level: OptimizationLevel::Aggressive,
            quantize_weights: true,
            input_shapes: vec![vec![1, 224, 224, 3]],
            output_names: vec!["output".to_string()],
            metadata: MobileModelMetadata::default(),
        }
    }
}

/// Builder for mobile export configurations
pub struct MobileExportConfigBuilder {
    config: MobileExportConfig,
}

impl MobileExportConfigBuilder {
    /// Create a new builder with specified backend
    pub fn new(backend: MobileBackend) -> Self {
        Self {
            config: MobileExporter::default_config(backend),
        }
    }

    /// Set target architecture
    pub fn target_arch(mut self, arch: &str) -> Self {
        self.config.target_arch = arch.to_string();
        self
    }

    /// Set optimization level
    pub fn optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.config.optimization_level = level;
        self
    }

    /// Enable or disable weight quantization
    pub fn quantize_weights(mut self, quantize: bool) -> Self {
        self.config.quantize_weights = quantize;
        self
    }

    /// Set input shapes
    pub fn input_shapes(mut self, shapes: Vec<Vec<usize>>) -> Self {
        self.config.input_shapes = shapes;
        self
    }

    /// Set output names
    pub fn output_names(mut self, names: Vec<String>) -> Self {
        self.config.output_names = names;
        self
    }

    /// Set model metadata
    pub fn metadata(mut self, metadata: MobileModelMetadata) -> Self {
        self.config.metadata = metadata;
        self
    }

    /// Build the configuration
    pub fn build(self) -> MobileExportConfig {
        self.config
    }
}

/// Convenience functions for common mobile export scenarios
pub mod presets {
    use super::*;

    /// Export model for iOS deployment with CoreML
    pub fn export_for_ios<T>(model: &Sequential<T>) -> Result<MobileModel>
    where
        T: Float
            + Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let config = MobileExportConfigBuilder::new(MobileBackend::CoreML)
            .target_arch("arm64")
            .optimization_level(OptimizationLevel::Basic)
            .quantize_weights(true)
            .build();

        MobileExporter::export(model, config)
    }

    /// Export model for Android deployment with TensorFlow Lite
    pub fn export_for_android<T>(model: &Sequential<T>) -> Result<MobileModel>
    where
        T: Float
            + Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let config = MobileExportConfigBuilder::new(MobileBackend::TensorFlowLite)
            .target_arch("arm64")
            .optimization_level(OptimizationLevel::Basic)
            .quantize_weights(true)
            .build();

        MobileExporter::export(model, config)
    }

    /// Export highly optimized model for edge devices
    pub fn export_for_edge<T>(model: &Sequential<T>, backend: MobileBackend) -> Result<MobileModel>
    where
        T: Float
            + Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let config = MobileExportConfigBuilder::new(backend)
            .target_arch("arm64")
            .optimization_level(OptimizationLevel::Ultra)
            .quantize_weights(true)
            .build();

        MobileExporter::export(model, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;
    use crate::model::Sequential;

    #[test]
    fn test_mobile_export_config_builder() {
        let config = MobileExportConfigBuilder::new(MobileBackend::CoreML)
            .target_arch("x86_64")
            .optimization_level(OptimizationLevel::Aggressive)
            .quantize_weights(false)
            .build();

        assert_eq!(config.backend, MobileBackend::CoreML);
        assert_eq!(config.target_arch, "x86_64");
        assert_eq!(config.optimization_level, OptimizationLevel::Aggressive);
        assert!(!config.quantize_weights);
    }

    #[test]
    fn test_coreml_export() {
        let model = Sequential::new(vec![
            Box::new(Dense::<f32>::new(10, 5, true)),
            Box::new(Dense::<f32>::new(5, 1, true)),
        ]);

        let config = MobileExporter::default_config(MobileBackend::CoreML);
        let exporter = CoreMLExporter::new(config);
        let result = exporter.export(&model);

        assert!(result.is_ok());
        let mobile_model = result.unwrap();
        assert!(!mobile_model.model_data.is_empty());
        assert_eq!(mobile_model.config.backend, MobileBackend::CoreML);
    }

    #[test]
    fn test_tflite_export() {
        let model = Sequential::new(vec![
            Box::new(Dense::<f32>::new(10, 5, true)),
            Box::new(Dense::<f32>::new(5, 1, true)),
        ]);

        let config = MobileExporter::default_config(MobileBackend::TensorFlowLite);
        let exporter = TensorFlowLiteExporter::new(config);
        let result = exporter.export(&model);

        assert!(result.is_ok());
        let mobile_model = result.unwrap();
        assert!(!mobile_model.model_data.is_empty());
        assert_eq!(mobile_model.config.backend, MobileBackend::TensorFlowLite);
    }

    #[test]
    fn test_mobile_export_stats() {
        let stats = MobileExportStats {
            original_size: 1000,
            exported_size: 750,
            compression_ratio: 1.33,
            estimated_inference_time: Some(5.0),
            accuracy_impact: Some(-0.02),
        };

        assert_eq!(stats.size_reduction_percent(), 25.0);
    }

    #[test]
    fn test_onnx_mobile_export() {
        let model = Sequential::new(vec![
            Box::new(Dense::<f32>::new(10, 5, true)),
            Box::new(Dense::<f32>::new(5, 1, true)),
        ]);

        let config = MobileExporter::default_config(MobileBackend::OnnxMobile);
        let exporter = OnnxMobileExporter::new(config);
        let result = exporter.export(&model);

        assert!(result.is_ok());
        let mobile_model = result.unwrap();
        assert!(!mobile_model.model_data.is_empty());
        assert_eq!(mobile_model.config.backend, MobileBackend::OnnxMobile);

        // Verify ONNX-specific features
        assert!(mobile_model.model_data.starts_with(b"ONNX"));
        assert!(mobile_model.stats.estimated_inference_time.is_some());
        assert!(mobile_model.stats.accuracy_impact.is_some());
    }

    #[test]
    fn test_onnx_mobile_optimizations() {
        let model = Sequential::new(vec![Box::new(Dense::<f32>::new(5, 1, true))]);

        // Test different optimization levels
        for opt_level in [
            OptimizationLevel::None,
            OptimizationLevel::Basic,
            OptimizationLevel::Aggressive,
            OptimizationLevel::Ultra,
        ] {
            let config = MobileExportConfigBuilder::new(MobileBackend::OnnxMobile)
                .optimization_level(opt_level)
                .build();

            let exporter = OnnxMobileExporter::new(config);
            let result = exporter.export(&model);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_preset_exports() {
        let model = Sequential::new(vec![Box::new(Dense::<f32>::new(5, 1, true))]);

        // Test iOS export preset
        let ios_result = presets::export_for_ios(&model);
        assert!(ios_result.is_ok());

        // Test Android export preset
        let android_result = presets::export_for_android(&model);
        assert!(android_result.is_ok());

        // Test edge export preset with TensorFlow Lite
        let edge_result = presets::export_for_edge(&model, MobileBackend::TensorFlowLite);
        assert!(edge_result.is_ok());

        // Test edge export preset with ONNX Mobile
        let onnx_edge_result = presets::export_for_edge(&model, MobileBackend::OnnxMobile);
        assert!(onnx_edge_result.is_ok());
    }
}
