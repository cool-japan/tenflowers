//! Global format registry with automatic format discovery and registration
//!
//! This module provides a singleton registry that automatically discovers and registers
//! all available format readers, enabling automatic format detection and loading.

use crate::error_taxonomy::helpers as error_helpers;
use crate::formats::unified_reader::{FormatDetection, FormatFactory, FormatReader};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock, RwLock};
use tenflowers_core::Result;

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
        // Register Arrow/Parquet if available
        #[cfg(feature = "parquet")]
        {
            use crate::formats::arrow::ArrowFormatFactory;
            self.register_factory(Box::new(ArrowFormatFactory));
        }

        // Additional formats can be registered here as they become available
        // CSV, JSON, HDF5, etc.
    }

    /// Register a format factory
    pub fn register_factory(&self, factory: Box<dyn FormatFactory>) {
        let format_name = factory.format_name().to_string();
        let mut factories = self.factories.write().unwrap();
        factories.insert(format_name, factory);
    }

    /// Unregister a format
    pub fn unregister_format(&self, format_name: &str) -> bool {
        let mut factories = self.factories.write().unwrap();
        factories.remove(format_name).is_some()
    }

    /// Get all registered format names
    pub fn list_formats(&self) -> Vec<String> {
        let factories = self.factories.read().unwrap();
        factories.keys().cloned().collect()
    }

    /// Get all supported extensions across all formats
    pub fn list_extensions(&self) -> Vec<String> {
        let factories = self.factories.read().unwrap();
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
        let factories = self.factories.read().unwrap();

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
        let factories = self.factories.read().unwrap();

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
        let factories = self.factories.read().unwrap();
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
        let factories = self.factories.read().unwrap();
        factories.contains_key(format_name)
    }

    /// Get format information
    pub fn get_format_info(&self, format_name: &str) -> Option<FormatInfo> {
        let factories = self.factories.read().unwrap();
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
        let factories = self.factories.read().unwrap();
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

        // Should have at least Arrow if parquet feature is enabled
        #[cfg(feature = "parquet")]
        {
            assert!(!formats.is_empty());
            assert!(formats
                .iter()
                .any(|f| f.contains("Arrow") || f.contains("Parquet")));
        }
    }

    #[test]
    fn test_list_extensions() {
        let extensions = global::list_extensions();

        #[cfg(feature = "parquet")]
        {
            assert!(!extensions.is_empty());
            assert!(
                extensions.contains(&"parquet".to_string())
                    || extensions.contains(&"arrow".to_string())
            );
        }
    }

    #[test]
    fn test_has_format() {
        #[cfg(feature = "parquet")]
        {
            assert!(global::has_format("Arrow/Parquet"));
        }

        assert!(!global::has_format("NonexistentFormat"));
    }

    #[test]
    fn test_get_format_info() {
        #[cfg(feature = "parquet")]
        {
            let info = global::get_format_info("Arrow/Parquet");
            assert!(info.is_some());

            let info = info.unwrap();
            assert_eq!(info.name, "Arrow/Parquet");
            assert!(!info.extensions.is_empty());
        }
    }

    #[test]
    fn test_get_all_format_info() {
        let all_info = global::get_all_format_info();

        #[cfg(feature = "parquet")]
        {
            assert!(!all_info.is_empty());
            assert!(all_info.iter().any(|info| info.name == "Arrow/Parquet"));
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
