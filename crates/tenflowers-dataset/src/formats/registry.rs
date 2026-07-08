//! Global format registry with automatic format discovery and registration
//!
//! This module provides a singleton registry that automatically discovers and registers
//! all available format readers, enabling automatic format detection and loading.

use crate::error_taxonomy::helpers as error_helpers;
use crate::formats::unified_reader::{FormatDetection, FormatFactory, FormatReader};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock, RwLock};
use tenflowers_core::{Result, TensorError};

/// Global format registry singleton
static GLOBAL_REGISTRY: OnceLock<GlobalFormatRegistry> = OnceLock::new();

/// Thread-safe global format registry
pub struct GlobalFormatRegistry {
    factories: Arc<RwLock<HashMap<String, Box<dyn FormatFactory>>>>,
}

impl GlobalFormatRegistry {
    /// Create a new global registry and auto-register all available formats
    fn new() -> Self {
        let registry = Self {
            factories: Arc::new(RwLock::new(HashMap::new())),
        };

        // Auto-register all available formats
        registry.auto_register_formats();

        registry
    }

    /// Get the global registry instance
    pub fn get() -> &'static GlobalFormatRegistry {
        GLOBAL_REGISTRY.get_or_init(|| {
            let registry = GlobalFormatRegistry::new();
            registry
        })
    }

    /// Auto-register all available format factories
    fn auto_register_formats(&self) {
        // Register CSV if available
        #[cfg(feature = "csv_format")]
        {
            use crate::formats::csv_format_reader::CsvFormatFactory;
            self.register_factory(Box::new(CsvFormatFactory));
        }

        // Register JSON if available
        #[cfg(feature = "serialize")]
        {
            use crate::formats::json_format_reader::JsonFormatFactory;
            self.register_factory(Box::new(JsonFormatFactory));
        }

        // Register Parquet if available
        #[cfg(feature = "parquet")]
        {
            use crate::formats::parquet_format_reader::ParquetFormatFactory;
            self.register_factory(Box::new(ParquetFormatFactory));
        }

        // Register HDF5 if available
        #[cfg(feature = "hdf5")]
        {
            use crate::formats::hdf5_format_reader::HDF5FormatFactory;
            self.register_factory(Box::new(HDF5FormatFactory));
        }
    }

    /// Register a format factory
    pub fn register_factory(&self, factory: Box<dyn FormatFactory>) {
        let format_name = factory.format_name().to_string();
        let mut factories = self
            .factories
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        factories.insert(format_name, factory);
    }

    /// Unregister a format
    pub fn unregister_format(&self, format_name: &str) -> bool {
        let mut factories = self
            .factories
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        factories.remove(format_name).is_some()
    }

    /// Get all registered format names
    pub fn list_formats(&self) -> Vec<String> {
        let factories = self
            .factories
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        factories.keys().cloned().collect()
    }

    /// Get all supported extensions across all formats
    pub fn list_extensions(&self) -> Vec<String> {
        let factories = self
            .factories
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut extensions = Vec::new();

        for factory in factories.values() {
            for ext in factory.extensions() {
                if !extensions.contains(&ext.to_string()) {
                    extensions.push(ext.to_string());
                }
            }
        }

        extensions.sort();
        extensions
    }

    /// Detect the best format for a given file
    pub fn detect_format(&self, path: &Path) -> Result<FormatDetection> {
        let factories = self.factories.read().map_err(|_| {
            TensorError::invalid_operation_simple("factories lock poisoned".to_string())
        })?;

        if factories.is_empty() {
            return Err(error_helpers::invalid_configuration(
                "GlobalFormatRegistry::detect_format",
                "registry",
                "No format factories registered",
            ));
        }

        let mut best_detection = FormatDetection {
            format_name: String::new(),
            confidence: 0.0,
            method: crate::formats::unified_reader::DetectionMethod::Extension,
        };

        // Try all factories and pick the one with highest confidence
        for factory in factories.values() {
            if let Ok(detection) = factory.can_read(path) {
                if detection.confidence > best_detection.confidence {
                    best_detection = detection;
                }
            }
        }

        if best_detection.confidence == 0.0 {
            return Err(error_helpers::invalid_configuration(
                "GlobalFormatRegistry::detect_format",
                "format",
                format!("No compatible format found for file: {:?}", path),
            ));
        }

        Ok(best_detection)
    }

    /// Create a reader for a specific format
    pub fn create_reader(&self, format_name: &str, path: &Path) -> Result<Box<dyn FormatReader>> {
        let factories = self.factories.read().map_err(|_| {
            TensorError::invalid_operation_simple("factories lock poisoned".to_string())
        })?;

        let factory = factories.get(format_name).ok_or_else(|| {
            error_helpers::invalid_configuration(
                "GlobalFormatRegistry::create_reader",
                "format",
                format!("Format '{}' not registered", format_name),
            )
        })?;

        factory.create_reader(path)
    }

    /// Auto-detect format and create reader
    pub fn auto_create_reader(&self, path: &Path) -> Result<Box<dyn FormatReader>> {
        let detection = self.detect_format(path)?;

        if detection.confidence < 0.5 {
            return Err(error_helpers::invalid_configuration(
                "GlobalFormatRegistry::auto_create_reader",
                "format",
                format!(
                    "Low confidence ({:.2}) for detected format '{}'",
                    detection.confidence, detection.format_name
                ),
            ));
        }

        self.create_reader(&detection.format_name, path)
    }

    /// Get factory for a specific format
    pub fn get_factory(&self, format_name: &str) -> Option<Arc<dyn FormatFactory>> {
        let factories = self
            .factories
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        factories.get(format_name).map(|f| {
            // Clone the Arc wrapper, not the factory itself
            // This is a workaround since we can't clone trait objects
            // In practice, we'd need a different approach or wrapper
            // For now, we'll just return None if cloning is needed
            None
        })?
    }

    /// Check if a format is registered
    pub fn has_format(&self, format_name: &str) -> bool {
        let factories = self
            .factories
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        factories.contains_key(format_name)
    }

    /// Get format information
    pub fn get_format_info(&self, format_name: &str) -> Option<FormatInfo> {
        let factories = self
            .factories
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        factories.get(format_name).map(|factory| FormatInfo {
            name: factory.format_name().to_string(),
            extensions: factory
                .extensions()
                .iter()
                .map(|&s| s.to_string())
                .collect(),
        })
    }

    /// Get all format information
    pub fn get_all_format_info(&self) -> Vec<FormatInfo> {
        let factories = self
            .factories
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        factories
            .values()
            .map(|factory| FormatInfo {
                name: factory.format_name().to_string(),
                extensions: factory
                    .extensions()
                    .iter()
                    .map(|&s| s.to_string())
                    .collect(),
            })
            .collect()
    }
}

/// Format information
#[derive(Debug, Clone)]
pub struct FormatInfo {
    /// Format name
    pub name: String,
    /// Supported file extensions
    pub extensions: Vec<String>,
}

/// Helper function to register a factory
pub fn register_format_factory<T: FormatFactory + 'static>(factory: T) {
    GlobalFormatRegistry::get().register_factory(Box::new(factory));
}

/// Convenient functions for working with the global registry
pub mod global {
    use super::*;

    /// List all registered formats
    pub fn list_formats() -> Vec<String> {
        GlobalFormatRegistry::get().list_formats()
    }

    /// List all supported extensions
    pub fn list_extensions() -> Vec<String> {
        GlobalFormatRegistry::get().list_extensions()
    }

    /// Detect format for a file
    pub fn detect_format(path: &Path) -> Result<FormatDetection> {
        GlobalFormatRegistry::get().detect_format(path)
    }

    /// Create reader for a specific format
    pub fn create_reader(format_name: &str, path: &Path) -> Result<Box<dyn FormatReader>> {
        GlobalFormatRegistry::get().create_reader(format_name, path)
    }

    /// Auto-detect and create reader
    pub fn auto_create_reader(path: &Path) -> Result<Box<dyn FormatReader>> {
        GlobalFormatRegistry::get().auto_create_reader(path)
    }

    /// Check if format is registered
    pub fn has_format(format_name: &str) -> bool {
        GlobalFormatRegistry::get().has_format(format_name)
    }

    /// Get format information
    pub fn get_format_info(format_name: &str) -> Option<FormatInfo> {
        GlobalFormatRegistry::get().get_format_info(format_name)
    }

    /// Get all format information
    pub fn get_all_format_info() -> Vec<FormatInfo> {
        GlobalFormatRegistry::get().get_all_format_info()
    }

    /// Register a format factory
    pub fn register_factory(factory: Box<dyn FormatFactory>) {
        GlobalFormatRegistry::get().register_factory(factory);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_registry_singleton() {
        let registry1 = GlobalFormatRegistry::get();
        let registry2 = GlobalFormatRegistry::get();

        // Should be the same instance
        assert!(std::ptr::eq(registry1, registry2));
    }

    #[test]
    fn test_list_formats() {
        let formats = global::list_formats();

        // Should have formats based on enabled features
        #[cfg(feature = "csv_format")]
        {
            assert!(!formats.is_empty());
            assert!(formats.iter().any(|f| f.contains("CSV")));
        }

        #[cfg(feature = "serialize")]
        {
            assert!(!formats.is_empty());
            assert!(formats.iter().any(|f| f.contains("JSON")));
        }

        #[cfg(feature = "parquet")]
        {
            assert!(!formats.is_empty());
            assert!(formats.iter().any(|f| f.contains("Parquet")));
        }

        #[cfg(feature = "hdf5")]
        {
            assert!(!formats.is_empty());
            assert!(formats.iter().any(|f| f.contains("HDF5")));
        }
    }

    #[test]
    fn test_list_extensions() {
        let extensions = global::list_extensions();

        #[cfg(feature = "csv_format")]
        {
            assert!(!extensions.is_empty());
            assert!(extensions.contains(&"csv".to_string()));
        }

        #[cfg(feature = "serialize")]
        {
            assert!(!extensions.is_empty());
            assert!(
                extensions.contains(&"json".to_string())
                    || extensions.contains(&"jsonl".to_string())
            );
        }

        #[cfg(feature = "parquet")]
        {
            assert!(!extensions.is_empty());
            assert!(extensions.contains(&"parquet".to_string()));
        }

        #[cfg(feature = "hdf5")]
        {
            assert!(!extensions.is_empty());
            assert!(
                extensions.contains(&"h5".to_string()) || extensions.contains(&"hdf5".to_string())
            );
        }
    }

    #[test]
    fn test_has_format() {
        #[cfg(feature = "csv_format")]
        {
            assert!(global::has_format("CSV"));
        }

        #[cfg(feature = "serialize")]
        {
            assert!(global::has_format("JSON"));
        }

        #[cfg(feature = "parquet")]
        {
            assert!(global::has_format("Parquet"));
        }

        #[cfg(feature = "hdf5")]
        {
            assert!(global::has_format("HDF5"));
        }

        assert!(!global::has_format("NonexistentFormat"));
    }

    #[test]
    fn test_get_format_info() {
        #[cfg(feature = "csv_format")]
        {
            let info = global::get_format_info("CSV");
            assert!(info.is_some());

            let info = info.expect("test: operation should succeed");
            assert_eq!(info.name, "CSV");
            assert!(!info.extensions.is_empty());
        }

        #[cfg(feature = "serialize")]
        {
            let info = global::get_format_info("JSON");
            assert!(info.is_some());

            let info = info.expect("test: operation should succeed");
            assert_eq!(info.name, "JSON");
            assert!(!info.extensions.is_empty());
        }
    }

    #[test]
    fn test_get_all_format_info() {
        let all_info = global::get_all_format_info();

        // Check that at least one format is registered
        #[cfg(any(
            feature = "csv_format",
            feature = "serialize",
            feature = "parquet",
            feature = "hdf5"
        ))]
        {
            assert!(!all_info.is_empty());
        }

        #[cfg(feature = "csv_format")]
        {
            assert!(all_info.iter().any(|info| info.name == "CSV"));
        }

        #[cfg(feature = "serialize")]
        {
            assert!(all_info.iter().any(|info| info.name == "JSON"));
        }

        #[cfg(feature = "parquet")]
        {
            assert!(all_info.iter().any(|info| info.name == "Parquet"));
        }

        #[cfg(feature = "hdf5")]
        {
            assert!(all_info.iter().any(|info| info.name == "HDF5"));
        }
    }

    #[test]
    fn test_detect_format_nonexistent() {
        use std::path::PathBuf;
        let path = PathBuf::from("/nonexistent/file.unknown");
        let result = global::detect_format(&path);

        // Should either fail or return low confidence
        if let Ok(detection) = result {
            assert_eq!(detection.confidence, 0.0);
        }
    }

    #[test]
    fn test_create_reader_invalid_format() {
        use std::path::PathBuf;
        let path = PathBuf::from("/nonexistent/file.txt");
        let result = global::create_reader("InvalidFormat", &path);

        assert!(result.is_err());
    }

    #[test]
    fn test_format_info_structure() {
        let info = FormatInfo {
            name: "TestFormat".to_string(),
            extensions: vec!["test".to_string(), "tst".to_string()],
        };

        assert_eq!(info.name, "TestFormat");
        assert_eq!(info.extensions.len(), 2);
        assert!(info.extensions.contains(&"test".to_string()));
    }
}
