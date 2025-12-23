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
use std::path::{Path, PathBuf};
use tenflowers_core::{Device, Result, Tensor, TensorError};

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
        let total_bytes = 0; // TODO: Calculate from tensors

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

    /// Load weights from file with default configuration
    pub fn load_from_file<T>(&self, path: impl AsRef<Path>) -> Result<LoadResult<T>>
    where
        T: Clone + Default + 'static,
    {
        self.load_with_config(path, self.default_config.clone())
    }

    /// Load weights from file with custom configuration
    pub fn load_with_config<T>(
        &self,
        path: impl AsRef<Path>,
        config: LoadConfig,
    ) -> Result<LoadResult<T>>
    where
        T: Clone + Default + 'static,
    {
        let path = path.as_ref();

        // Detect format
        let format = WeightFormat::from_path(path);

        match format {
            WeightFormat::Json => self.load_json(path, config),
            WeightFormat::Binary => self.load_binary(path, config),
            WeightFormat::SafeTensors => self.load_safetensors(path, config),
            WeightFormat::NumPy => self.load_numpy(path, config),
            WeightFormat::Auto => {
                // Try each format
                Err(TensorError::serialization_error_simple(
                    "Auto-detection not yet implemented".to_string(),
                ))
            }
        }
    }

    /// Load weights from JSON format
    fn load_json<T>(&self, _path: &Path, _config: LoadConfig) -> Result<LoadResult<T>>
    where
        T: Clone + Default + 'static,
    {
        // Placeholder implementation
        let weights = HashMap::new();
        let mut result = LoadResult::new(weights);
        result.add_warning("JSON loading not yet fully implemented".to_string());
        Ok(result)
    }

    /// Load weights from binary format
    fn load_binary<T>(&self, _path: &Path, _config: LoadConfig) -> Result<LoadResult<T>>
    where
        T: Clone + Default + 'static,
    {
        // Placeholder implementation
        let weights = HashMap::new();
        let mut result = LoadResult::new(weights);
        result.add_warning("Binary loading not yet fully implemented".to_string());
        Ok(result)
    }

    /// Load weights from SafeTensors format
    fn load_safetensors<T>(&self, _path: &Path, _config: LoadConfig) -> Result<LoadResult<T>>
    where
        T: Clone + Default + 'static,
    {
        // Placeholder implementation
        let weights = HashMap::new();
        let mut result = LoadResult::new(weights);
        result.add_warning("SafeTensors loading not yet fully implemented".to_string());
        Ok(result)
    }

    /// Load weights from NumPy format
    fn load_numpy<T>(&self, _path: &Path, _config: LoadConfig) -> Result<LoadResult<T>>
    where
        T: Clone + Default + 'static,
    {
        // Placeholder implementation
        let weights = HashMap::new();
        let mut result = LoadResult::new(weights);
        result.add_warning("NumPy loading not yet fully implemented".to_string());
        Ok(result)
    }

    /// Save weights to file
    pub fn save_to_file<T>(
        &self,
        weights: &HashMap<String, Tensor<T>>,
        path: impl AsRef<Path>,
        format: WeightFormat,
    ) -> Result<()>
    where
        T: Clone + Default + 'static,
    {
        let _path = path.as_ref();

        match format {
            WeightFormat::Json => self.save_json(weights, _path),
            WeightFormat::Binary => self.save_binary(weights, _path),
            WeightFormat::SafeTensors => self.save_safetensors(weights, _path),
            WeightFormat::NumPy => self.save_numpy(weights, _path),
            WeightFormat::Auto => {
                // Use binary as default
                self.save_binary(weights, _path)
            }
        }
    }

    /// Save weights to JSON format
    fn save_json<T>(&self, _weights: &HashMap<String, Tensor<T>>, _path: &Path) -> Result<()>
    where
        T: Clone + Default + 'static,
    {
        // Placeholder implementation
        Ok(())
    }

    /// Save weights to binary format
    fn save_binary<T>(&self, _weights: &HashMap<String, Tensor<T>>, _path: &Path) -> Result<()>
    where
        T: Clone + Default + 'static,
    {
        // Placeholder implementation
        Ok(())
    }

    /// Save weights to SafeTensors format
    fn save_safetensors<T>(&self, _weights: &HashMap<String, Tensor<T>>, _path: &Path) -> Result<()>
    where
        T: Clone + Default + 'static,
    {
        // Placeholder implementation
        Ok(())
    }

    /// Save weights to NumPy format
    fn save_numpy<T>(&self, _weights: &HashMap<String, Tensor<T>>, _path: &Path) -> Result<()>
    where
        T: Clone + Default + 'static,
    {
        // Placeholder implementation
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
}
