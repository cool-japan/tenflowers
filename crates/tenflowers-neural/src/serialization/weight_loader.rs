//! Weight Loading System - Standardized Pretrained Weight Loading
//!
//! This module provides a comprehensive system for loading pretrained weights
//! from various formats into TenfloweRS models.
//!
//! ## Supported Formats
//!
//! - **SafeTensors**: Safe, efficient binary format (recommended)
//! - **JSON**: Human-readable format for small models
//! - **Binary**: Custom binary format with version support
//! - **NumPy**: Load weights from .npz files
//!
//! ## Features
//!
//! - **Partial Loading**: Load only specific layers/parameters
//! - **Strict Validation**: Ensure shapes and types match
//! - **Weight Mapping**: Map weight names between formats
//! - **Type Conversion**: Automatic type conversion when safe
//! - **Progress Tracking**: Monitor loading progress for large models
//!
//! ## Example
//!
//! ```rust,ignore
//! use tenflowers_neural::serialization::weight_loader::{WeightLoader, LoadConfig};
//!
//! // Create a weight loader
//! let loader = WeightLoader::new();
//!
//! // Load weights with default configuration
//! let weights = loader.load_from_file("model_weights.bin")?;
//!
//! // Load with custom configuration
//! let config = LoadConfig::new()
//!     .with_strict(false)  // Allow partial loading
//!     .with_device(Device::Cpu);
//!
//! let weights = loader.load_with_config("weights.json", config)?;
//! ```

use std::collections::HashMap;
use std::path::Path;
use tenflowers_core::{Device, Result, Tensor, TensorError};

/// On-disk representation of a single named weight tensor.
///
/// The JSON weight format stores, for every parameter, its logical shape and a
/// flat (row-major) buffer of values. This is intentionally simple and
/// human-readable; binary formats are recommended for large production models.
#[cfg(feature = "serialize")]
#[derive(serde::Serialize, serde::Deserialize)]
struct StoredTensor<T> {
    /// Logical dimensions of the tensor.
    shape: Vec<usize>,
    /// Flat row-major data buffer.
    data: Vec<T>,
}

/// Top-level on-disk document for the JSON weight format.
#[cfg(feature = "serialize")]
#[derive(serde::Serialize, serde::Deserialize)]
struct WeightDocument<T> {
    /// Format version for forward-compatibility checks.
    version: u32,
    /// Map of weight name to its stored tensor.
    tensors: HashMap<String, StoredTensor<T>>,
}

/// Current version tag written into the JSON weight format header.
#[cfg(feature = "serialize")]
const JSON_WEIGHT_FORMAT_VERSION: u32 = 1;

/// Weight format for serialization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// SafeTensors format (recommended for production)
    SafeTensors,
    /// JSON format (human-readable, good for debugging)
    Json,
    /// Custom binary format with versioning
    Binary,
    /// NumPy .npz format
    NumPy,
    /// Auto-detect format from file extension
    Auto,
}

impl WeightFormat {
    /// Detect format from file extension
    pub fn from_path(path: &Path) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some("safetensors") => WeightFormat::SafeTensors,
            Some("json") => WeightFormat::Json,
            Some("bin") | Some("pt") => WeightFormat::Binary,
            Some("npz") | Some("npy") => WeightFormat::NumPy,
            _ => WeightFormat::Binary, // Default
        }
    }
}

/// Configuration for weight loading
#[derive(Debug, Clone)]
pub struct LoadConfig {
    /// Strict mode: fail if any weights are missing or mismatched
    pub strict: bool,

    /// Target device for loaded weights
    pub device: Device,

    /// Allow type conversion (e.g., f64 -> f32)
    pub allow_type_conversion: bool,

    /// Prefix to add to all weight names
    pub prefix: Option<String>,

    /// Suffix to add to all weight names
    pub suffix: Option<String>,

    /// Weight name mapping (old_name -> new_name)
    pub name_mapping: HashMap<String, String>,

    /// Weights to exclude from loading
    pub exclude_patterns: Vec<String>,

    /// Weights to include (if empty, include all)
    pub include_patterns: Vec<String>,
}

impl LoadConfig {
    /// Create a new load configuration with defaults
    pub fn new() -> Self {
        Self {
            strict: true,
            device: Device::Cpu,
            allow_type_conversion: true,
            prefix: None,
            suffix: None,
            name_mapping: HashMap::new(),
            exclude_patterns: Vec::new(),
            include_patterns: Vec::new(),
        }
    }

    /// Set strict mode
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Set target device
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Allow type conversion
    pub fn with_type_conversion(mut self, allow: bool) -> Self {
        self.allow_type_conversion = allow;
        self
    }

    /// Add a name prefix
    pub fn with_prefix(mut self, prefix: String) -> Self {
        self.prefix = Some(prefix);
        self
    }

    /// Add a name suffix
    pub fn with_suffix(mut self, suffix: String) -> Self {
        self.suffix = Some(suffix);
        self
    }

    /// Add weight name mapping
    pub fn with_mapping(mut self, old_name: String, new_name: String) -> Self {
        self.name_mapping.insert(old_name, new_name);
        self
    }

    /// Exclude weights matching pattern
    pub fn exclude(mut self, pattern: String) -> Self {
        self.exclude_patterns.push(pattern);
        self
    }

    /// Include only weights matching pattern
    pub fn include(mut self, pattern: String) -> Self {
        self.include_patterns.push(pattern);
        self
    }

    /// Apply name transformation
    pub fn transform_name(&self, name: &str) -> String {
        let mut result = name.to_string();

        // Apply mapping
        if let Some(mapped) = self.name_mapping.get(name) {
            result = mapped.clone();
        }

        // Apply prefix
        if let Some(prefix) = &self.prefix {
            result = format!("{}{}", prefix, result);
        }

        // Apply suffix
        if let Some(suffix) = &self.suffix {
            result = format!("{}{}", result, suffix);
        }

        result
    }

    /// Check if a weight name should be included
    pub fn should_include(&self, name: &str) -> bool {
        // Check exclude patterns
        for pattern in &self.exclude_patterns {
            if name.contains(pattern) {
                return false;
            }
        }

        // Check include patterns (if any specified)
        if !self.include_patterns.is_empty() {
            return self.include_patterns.iter().any(|p| name.contains(p));
        }

        true
    }
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Weight loading result with metadata
#[derive(Debug)]
pub struct LoadResult<T> {
    /// Loaded weights (name -> tensor)
    pub weights: HashMap<String, Tensor<T>>,

    /// Number of weights loaded
    pub num_loaded: usize,

    /// Number of weights skipped
    pub num_skipped: usize,

    /// Warnings encountered during loading
    pub warnings: Vec<String>,

    /// Total size in bytes
    pub total_bytes: usize,
}

impl<T> LoadResult<T> {
    /// Create a new load result
    pub fn new(weights: HashMap<String, Tensor<T>>) -> Self {
        let num_loaded = weights.len();
        let total_bytes = 0; // NOTE(v0.2): Calculate total_bytes from tensor sizes

        Self {
            weights,
            num_loaded,
            num_skipped: 0,
            warnings: Vec::new(),
            total_bytes,
        }
    }

    /// Add a warning message
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Get a weight by name
    pub fn get(&self, name: &str) -> Option<&Tensor<T>> {
        self.weights.get(name)
    }

    /// Check if loading was successful
    pub fn is_success(&self) -> bool {
        self.num_loaded > 0
    }
}

/// Weight loader for managing pretrained weight loading
#[derive(Debug)]
pub struct WeightLoader {
    /// Default load configuration
    default_config: LoadConfig,
}

impl WeightLoader {
    /// Create a new weight loader
    pub fn new() -> Self {
        Self {
            default_config: LoadConfig::new(),
        }
    }

    /// Create with custom default configuration
    pub fn with_config(config: LoadConfig) -> Self {
        Self {
            default_config: config,
        }
    }

    /// Load weights from file with default configuration.
    ///
    /// Requires the `serialize` feature for the JSON text format. Binary,
    /// SafeTensors, and NumPy formats are not yet supported and return an
    /// explicit error rather than silently producing an empty result.
    #[cfg(feature = "serialize")]
    pub fn load_from_file<T>(&self, path: impl AsRef<Path>) -> Result<LoadResult<T>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + serde::Serialize
            + serde::de::DeserializeOwned
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        self.load_with_config(path, self.default_config.clone())
    }

    /// Load weights from file with custom configuration.
    #[cfg(feature = "serialize")]
    pub fn load_with_config<T>(
        &self,
        path: impl AsRef<Path>,
        config: LoadConfig,
    ) -> Result<LoadResult<T>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + serde::Serialize
            + serde::de::DeserializeOwned
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let path = path.as_ref();

        // Detect format
        let format = WeightFormat::from_path(path);

        match format {
            // The JSON document is self-describing, so an auto-detected file is
            // routed through the JSON loader.
            WeightFormat::Json | WeightFormat::Auto => self.load_json(path, config),
            WeightFormat::Binary => Err(TensorError::not_implemented_simple(
                "binary weight loading is not implemented; use the JSON format".to_string(),
            )),
            WeightFormat::SafeTensors => Err(TensorError::not_implemented_simple(
                "SafeTensors weight loading requires the `safetensors` crate dependency; \
                 use the JSON format"
                    .to_string(),
            )),
            WeightFormat::NumPy => Err(TensorError::not_implemented_simple(
                "NumPy (.npz/.npy) weight loading requires a NumPy reader dependency; \
                 use the JSON format"
                    .to_string(),
            )),
        }
    }

    /// Load weights from file (no-`serialize` build): returns an explicit error.
    #[cfg(not(feature = "serialize"))]
    pub fn load_from_file<T>(&self, _path: impl AsRef<Path>) -> Result<LoadResult<T>>
    where
        T: Clone + Default + 'static,
    {
        Err(TensorError::not_implemented_simple(
            "weight loading requires the `serialize` feature to be enabled".to_string(),
        ))
    }

    /// Load weights with config (no-`serialize` build): returns an explicit error.
    #[cfg(not(feature = "serialize"))]
    pub fn load_with_config<T>(
        &self,
        _path: impl AsRef<Path>,
        _config: LoadConfig,
    ) -> Result<LoadResult<T>>
    where
        T: Clone + Default + 'static,
    {
        Err(TensorError::not_implemented_simple(
            "weight loading requires the `serialize` feature to be enabled".to_string(),
        ))
    }

    /// Load weights from the JSON text format.
    ///
    /// Deserializes a [`WeightDocument`], reconstructs each [`Tensor`] from its
    /// shape and flat buffer, and applies the configured name transformation and
    /// include/exclude filtering. In strict mode, a structurally invalid file
    /// (e.g. a buffer whose length does not match its shape) is a hard error.
    #[cfg(feature = "serialize")]
    fn load_json<T>(&self, path: &Path, config: LoadConfig) -> Result<LoadResult<T>>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + serde::Serialize
            + serde::de::DeserializeOwned
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let file = std::fs::File::open(path).map_err(|e| {
            TensorError::io_error_simple(format!(
                "failed to open weight file {}: {}",
                path.display(),
                e
            ))
        })?;
        let reader = std::io::BufReader::new(file);

        let document: WeightDocument<T> = serde_json::from_reader(reader).map_err(|e| {
            TensorError::serialization_error_simple(format!(
                "failed to deserialize JSON weights from {}: {}",
                path.display(),
                e
            ))
        })?;

        if document.version != JSON_WEIGHT_FORMAT_VERSION {
            return Err(TensorError::serialization_error_simple(format!(
                "unsupported JSON weight format version {} (expected {})",
                document.version, JSON_WEIGHT_FORMAT_VERSION
            )));
        }

        let mut weights: HashMap<String, Tensor<T>> = HashMap::new();
        let mut warnings: Vec<String> = Vec::new();
        let mut num_skipped = 0usize;
        let mut total_bytes = 0usize;

        for (raw_name, stored) in document.tensors {
            if !config.should_include(&raw_name) {
                num_skipped += 1;
                continue;
            }

            let name = config.transform_name(&raw_name);

            let expected: usize = stored.shape.iter().product();
            if stored.data.len() != expected {
                let message = format!(
                    "weight '{}' has {} values but shape {:?} requires {}",
                    raw_name,
                    stored.data.len(),
                    stored.shape,
                    expected
                );
                if config.strict {
                    return Err(TensorError::serialization_error_simple(message));
                }
                warnings.push(message);
                num_skipped += 1;
                continue;
            }

            total_bytes += stored.data.len() * std::mem::size_of::<T>();

            let tensor = Tensor::from_vec(stored.data, &stored.shape)?;

            // Place on the requested device when GPU support is compiled in.
            let tensor = self.place_on_device(tensor, &config.device)?;

            weights.insert(name, tensor);
        }

        let mut result = LoadResult::new(weights);
        result.num_skipped = num_skipped;
        result.total_bytes = total_bytes;
        for warning in warnings {
            result.add_warning(warning);
        }
        Ok(result)
    }

    /// Move a freshly loaded tensor onto the configured device.
    ///
    /// On CPU-only builds the tensor is already in the right place; when GPU or
    /// ROCm support is compiled in, non-CPU targets trigger a device transfer.
    #[cfg(feature = "serialize")]
    fn place_on_device<T>(&self, tensor: Tensor<T>, device: &Device) -> Result<Tensor<T>>
    where
        T: Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
    {
        if device.is_cpu() {
            return Ok(tensor);
        }

        #[cfg(feature = "gpu")]
        {
            tensor.to(*device)
        }

        #[cfg(not(feature = "gpu"))]
        {
            // No non-CPU device variants exist in this build configuration.
            Ok(tensor)
        }
    }

    /// Save weights to file.
    ///
    /// Requires the `serialize` feature for the JSON text format. Binary,
    /// SafeTensors, and NumPy formats are not yet supported and return an
    /// explicit error rather than silently claiming success without writing.
    #[cfg(feature = "serialize")]
    pub fn save_to_file<T>(
        &self,
        weights: &HashMap<String, Tensor<T>>,
        path: impl AsRef<Path>,
        format: WeightFormat,
    ) -> Result<()>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + serde::Serialize
            + serde::de::DeserializeOwned
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let path = path.as_ref();

        match format {
            WeightFormat::Json | WeightFormat::Auto => self.save_json(weights, path),
            WeightFormat::Binary => Err(TensorError::not_implemented_simple(
                "binary weight saving is not implemented; use the JSON format".to_string(),
            )),
            WeightFormat::SafeTensors => Err(TensorError::not_implemented_simple(
                "SafeTensors weight saving requires the `safetensors` crate dependency; \
                 use the JSON format"
                    .to_string(),
            )),
            WeightFormat::NumPy => Err(TensorError::not_implemented_simple(
                "NumPy (.npz/.npy) weight saving requires a NumPy writer dependency; \
                 use the JSON format"
                    .to_string(),
            )),
        }
    }

    /// Save weights to file (no-`serialize` build): returns an explicit error.
    #[cfg(not(feature = "serialize"))]
    pub fn save_to_file<T>(
        &self,
        _weights: &HashMap<String, Tensor<T>>,
        _path: impl AsRef<Path>,
        _format: WeightFormat,
    ) -> Result<()>
    where
        T: Clone + Default + 'static,
    {
        Err(TensorError::not_implemented_simple(
            "weight saving requires the `serialize` feature to be enabled".to_string(),
        ))
    }

    /// Save weights to the JSON text format.
    ///
    /// Serializes every tensor as a shape plus a flat row-major buffer into a
    /// [`WeightDocument`] and writes it to `path`. The file is always written
    /// (or an error is returned); it never silently succeeds without writing.
    #[cfg(feature = "serialize")]
    fn save_json<T>(&self, weights: &HashMap<String, Tensor<T>>, path: &Path) -> Result<()>
    where
        T: Clone
            + Default
            + Send
            + Sync
            + 'static
            + scirs2_core::num_traits::Zero
            + scirs2_core::num_traits::One
            + serde::Serialize
            + serde::de::DeserializeOwned
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let mut tensors: HashMap<String, StoredTensor<T>> = HashMap::with_capacity(weights.len());

        for (name, tensor) in weights {
            let shape = tensor.shape().dims().to_vec();
            let data = tensor.to_vec()?;
            tensors.insert(name.clone(), StoredTensor { shape, data });
        }

        let document = WeightDocument {
            version: JSON_WEIGHT_FORMAT_VERSION,
            tensors,
        };

        // Ensure the parent directory exists before writing.
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    TensorError::io_error_simple(format!(
                        "failed to create directory {}: {}",
                        parent.display(),
                        e
                    ))
                })?;
            }
        }

        let file = std::fs::File::create(path).map_err(|e| {
            TensorError::io_error_simple(format!(
                "failed to create weight file {}: {}",
                path.display(),
                e
            ))
        })?;
        let writer = std::io::BufWriter::new(file);

        serde_json::to_writer_pretty(writer, &document).map_err(|e| {
            TensorError::serialization_error_simple(format!(
                "failed to serialize JSON weights to {}: {}",
                path.display(),
                e
            ))
        })?;

        Ok(())
    }
}

impl Default for WeightLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to create a weight loader
pub fn loader() -> WeightLoader {
    WeightLoader::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_format_detection() {
        assert_eq!(
            WeightFormat::from_path(Path::new("model.safetensors")),
            WeightFormat::SafeTensors
        );
        assert_eq!(
            WeightFormat::from_path(Path::new("weights.json")),
            WeightFormat::Json
        );
        assert_eq!(
            WeightFormat::from_path(Path::new("model.bin")),
            WeightFormat::Binary
        );
        assert_eq!(
            WeightFormat::from_path(Path::new("data.npz")),
            WeightFormat::NumPy
        );
    }

    #[test]
    fn test_load_config_creation() {
        let config = LoadConfig::new();
        assert!(config.strict);
        assert!(config.allow_type_conversion);
        assert!(config.name_mapping.is_empty());
    }

    #[test]
    fn test_load_config_builder() {
        let config = LoadConfig::new()
            .with_strict(false)
            .with_prefix("layer.".to_string())
            .with_suffix(".weight".to_string());

        assert!(!config.strict);
        assert_eq!(config.prefix, Some("layer.".to_string()));
        assert_eq!(config.suffix, Some(".weight".to_string()));
    }

    #[test]
    fn test_name_transformation() {
        let config = LoadConfig::new()
            .with_prefix("model.".to_string())
            .with_suffix(".data".to_string());

        let transformed = config.transform_name("conv1");
        assert_eq!(transformed, "model.conv1.data");
    }

    #[test]
    fn test_name_mapping() {
        let config = LoadConfig::new().with_mapping("old_name".to_string(), "new_name".to_string());

        let transformed = config.transform_name("old_name");
        assert_eq!(transformed, "new_name");
    }

    #[test]
    fn test_should_include_no_filters() {
        let config = LoadConfig::new();
        assert!(config.should_include("any_weight"));
    }

    #[test]
    fn test_should_include_with_exclude() {
        let config = LoadConfig::new().exclude("bias".to_string());

        assert!(!config.should_include("layer.bias"));
        assert!(config.should_include("layer.weight"));
    }

    #[test]
    fn test_should_include_with_include() {
        let config = LoadConfig::new().include("weight".to_string());

        assert!(config.should_include("layer.weight"));
        assert!(!config.should_include("layer.bias"));
    }

    #[test]
    fn test_load_result_creation() {
        let weights = HashMap::new();
        let result = LoadResult::<f32>::new(weights);

        assert_eq!(result.num_loaded, 0);
        assert_eq!(result.num_skipped, 0);
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_load_result_warnings() {
        let weights = HashMap::new();
        let mut result = LoadResult::<f32>::new(weights);

        result.add_warning("Test warning".to_string());
        assert_eq!(result.warnings.len(), 1);
        assert_eq!(result.warnings[0], "Test warning");
    }

    #[test]
    fn test_weight_loader_creation() {
        let loader = WeightLoader::new();
        assert!(loader.default_config.strict);
    }

    #[test]
    fn test_weight_loader_with_config() {
        let config = LoadConfig::new().with_strict(false);
        let loader = WeightLoader::with_config(config);
        assert!(!loader.default_config.strict);
    }

    #[test]
    fn test_loader_helper() {
        let loader = loader();
        assert!(loader.default_config.strict);
    }

    #[cfg(feature = "serialize")]
    #[test]
    fn test_json_save_load_roundtrip() {
        let loader = WeightLoader::new();

        let mut weights: HashMap<String, Tensor<f32>> = HashMap::new();
        weights.insert(
            "layer1.weight".to_string(),
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2])
                .expect("test: tensor creation should succeed"),
        );
        weights.insert(
            "layer1.bias".to_string(),
            Tensor::from_vec(vec![0.5_f32, -0.5], &[2])
                .expect("test: tensor creation should succeed"),
        );

        let path = std::env::temp_dir().join("tenflowers_weight_loader_roundtrip.json");

        loader
            .save_to_file(&weights, &path, WeightFormat::Json)
            .expect("test: saving weights should succeed");

        // The file must actually exist and be non-empty.
        let metadata = std::fs::metadata(&path).expect("test: saved file should exist");
        assert!(metadata.len() > 0, "saved weight file should not be empty");

        let result: LoadResult<f32> = loader
            .load_from_file(&path)
            .expect("test: loading weights should succeed");

        assert_eq!(result.num_loaded, 2);

        let weight = result
            .get("layer1.weight")
            .expect("test: weight should be present");
        assert_eq!(weight.shape().dims(), &[2, 2]);
        assert_eq!(
            weight.to_vec().expect("test: to_vec should succeed"),
            vec![1.0_f32, 2.0, 3.0, 4.0]
        );

        let bias = result
            .get("layer1.bias")
            .expect("test: bias should be present");
        assert_eq!(bias.shape().dims(), &[2]);

        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "serialize")]
    #[test]
    fn test_json_load_respects_include_filter() {
        let loader = WeightLoader::with_config(LoadConfig::new().include("weight".to_string()));

        let mut weights: HashMap<String, Tensor<f32>> = HashMap::new();
        weights.insert(
            "conv.weight".to_string(),
            Tensor::from_vec(vec![1.0_f32, 2.0], &[2]).expect("test: creation should succeed"),
        );
        weights.insert(
            "conv.bias".to_string(),
            Tensor::from_vec(vec![3.0_f32], &[1]).expect("test: creation should succeed"),
        );

        let path = std::env::temp_dir().join("tenflowers_weight_loader_filter.json");
        loader
            .save_to_file(&weights, &path, WeightFormat::Json)
            .expect("test: saving should succeed");

        let result: LoadResult<f32> = loader
            .load_from_file(&path)
            .expect("test: loading should succeed");

        assert_eq!(result.num_loaded, 1);
        assert_eq!(result.num_skipped, 1);
        assert!(result.get("conv.weight").is_some());
        assert!(result.get("conv.bias").is_none());

        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "serialize")]
    #[test]
    fn test_binary_save_returns_honest_error() {
        let loader = WeightLoader::new();
        let weights: HashMap<String, Tensor<f32>> = HashMap::new();
        let path = std::env::temp_dir().join("tenflowers_weight_loader_binary.bin");

        let outcome = loader.save_to_file(&weights, &path, WeightFormat::Binary);
        assert!(
            outcome.is_err(),
            "binary save must return an error instead of silently succeeding"
        );
        // No file should have been created.
        assert!(!path.exists());
    }
}
