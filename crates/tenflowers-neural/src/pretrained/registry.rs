//! Model Registry - Centralized Pretrained Model Management
//!
//! This module provides a comprehensive model registry system for managing,
//! discovering, and loading pretrained neural network models.
//!
//! ## Features
//!
//! - **Model Registration**: Register pretrained models with metadata
//! - **Version Management**: Support multiple versions of the same model
//! - **Model Discovery**: List and search available models
//! - **Efficient Loading**: Cache and load models efficiently
//! - **Metadata Rich**: Store comprehensive model information
//!
//! ## Example
//!
//! ```rust,ignore
//! use tenflowers_neural::pretrained::registry::{ModelRegistry, ModelMetadata};
//!
//! // Get the global registry
//! let registry = ModelRegistry::global();
//!
//! // List all available models
//! let models = registry.list_models();
//!
//! // Load a specific model
//! let model_info = registry.get_model("resnet50", Some("v1.0"))?;
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Model domain classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelDomain {
    /// Computer vision models
    Vision,
    /// Natural language processing models
    NLP,
    /// Audio processing models
    Audio,
    /// Multimodal models
    Multimodal,
    /// Reinforcement learning models
    RL,
    /// Custom domain
    Custom(String),
}

impl std::fmt::Display for ModelDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelDomain::Vision => write!(f, "vision"),
            ModelDomain::NLP => write!(f, "nlp"),
            ModelDomain::Audio => write!(f, "audio"),
            ModelDomain::Multimodal => write!(f, "multimodal"),
            ModelDomain::RL => write!(f, "rl"),
            ModelDomain::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Model architecture type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelArchitecture {
    /// Convolutional Neural Network
    CNN,
    /// Transformer architecture
    Transformer,
    /// Recurrent Neural Network
    RNN,
    /// State Space Model
    SSM,
    /// Mixture of Experts
    MoE,
    /// Custom architecture
    Custom(String),
}

/// Model metadata containing all information about a pretrained model
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Unique model identifier
    pub id: String,

    /// Human-readable model name
    pub name: String,

    /// Model version
    pub version: String,

    /// Model domain
    pub domain: ModelDomain,

    /// Model architecture type
    pub architecture: ModelArchitecture,

    /// Model description
    pub description: String,

    /// Model author/organization
    pub author: String,

    /// Model license
    pub license: String,

    /// Training dataset information
    pub dataset: Option<String>,

    /// Model input shape (if fixed)
    pub input_shape: Option<Vec<usize>>,

    /// Number of parameters
    pub num_parameters: Option<usize>,

    /// Model size in bytes
    pub size_bytes: Option<usize>,

    /// Model file path (local or URL)
    pub model_path: Option<PathBuf>,

    /// Additional tags for searching
    pub tags: Vec<String>,

    /// Creation timestamp
    pub created_at: Option<String>,

    /// Last update timestamp
    pub updated_at: Option<String>,
}

impl ModelMetadata {
    /// Create a new model metadata entry
    pub fn new(id: impl Into<String>, name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            version: version.into(),
            domain: ModelDomain::Custom("unknown".to_string()),
            architecture: ModelArchitecture::Custom("unknown".to_string()),
            description: String::new(),
            author: String::new(),
            license: "Unknown".to_string(),
            dataset: None,
            input_shape: None,
            num_parameters: None,
            size_bytes: None,
            model_path: None,
            tags: Vec::new(),
            created_at: None,
            updated_at: None,
        }
    }

    /// Set the model domain
    pub fn with_domain(mut self, domain: ModelDomain) -> Self {
        self.domain = domain;
        self
    }

    /// Set the model architecture
    pub fn with_architecture(mut self, architecture: ModelArchitecture) -> Self {
        self.architecture = architecture;
        self
    }

    /// Set the model description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set the model author
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = author.into();
        self
    }

    /// Set the model license
    pub fn with_license(mut self, license: impl Into<String>) -> Self {
        self.license = license.into();
        self
    }

    /// Set the training dataset
    pub fn with_dataset(mut self, dataset: impl Into<String>) -> Self {
        self.dataset = Some(dataset.into());
        self
    }

    /// Set the input shape
    pub fn with_input_shape(mut self, shape: Vec<usize>) -> Self {
        self.input_shape = Some(shape);
        self
    }

    /// Set the number of parameters
    pub fn with_num_parameters(mut self, num_params: usize) -> Self {
        self.num_parameters = Some(num_params);
        self
    }

    /// Set the model size
    pub fn with_size_bytes(mut self, size: usize) -> Self {
        self.size_bytes = Some(size);
        self
    }

    /// Set the model file path
    pub fn with_model_path(mut self, path: PathBuf) -> Self {
        self.model_path = Some(path);
        self
    }

    /// Add tags for searching
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Get the unique model key (id:version)
    pub fn key(&self) -> String {
        format!("{}:{}", self.id, self.version)
    }
}

/// Model Registry for managing pretrained models
#[derive(Debug, Clone)]
pub struct ModelRegistry {
    models: Arc<RwLock<HashMap<String, ModelMetadata>>>,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get the global model registry instance
    pub fn global() -> &'static Self {
        use std::sync::OnceLock;
        static REGISTRY: OnceLock<ModelRegistry> = OnceLock::new();
        REGISTRY.get_or_init(|| {
            let registry = ModelRegistry::new();
            registry.register_builtin_models();
            registry
        })
    }

    /// Register a model in the registry
    pub fn register(&self, metadata: ModelMetadata) -> Result<(), String> {
        let mut models = self.models.write().map_err(|e| e.to_string())?;
        let key = metadata.key();

        if models.contains_key(&key) {
            return Err(format!("Model {} already registered", key));
        }

        models.insert(key, metadata);
        Ok(())
    }

    /// Get a model by ID and optional version
    pub fn get_model(&self, id: &str, version: Option<&str>) -> Result<ModelMetadata, String> {
        let models = self.models.read().map_err(|e| e.to_string())?;

        if let Some(ver) = version {
            let key = format!("{}:{}", id, ver);
            models
                .get(&key)
                .cloned()
                .ok_or_else(|| format!("Model {} not found", key))
        } else {
            // Find the latest version
            let matching: Vec<_> = models.values().filter(|m| m.id == id).collect();

            if matching.is_empty() {
                return Err(format!("No versions of model {} found", id));
            }

            // Return the first match (in a real implementation, sort by version)
            Ok(matching[0].clone())
        }
    }

    /// List all registered models
    pub fn list_models(&self) -> Vec<ModelMetadata> {
        self.models
            .read()
            .map(|models| models.values().cloned().collect())
            .unwrap_or_default()
    }

    /// List models by domain
    pub fn list_models_by_domain(&self, domain: &ModelDomain) -> Vec<ModelMetadata> {
        self.models
            .read()
            .map(|models| {
                models
                    .values()
                    .filter(|m| &m.domain == domain)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Search models by tag
    pub fn search_by_tag(&self, tag: &str) -> Vec<ModelMetadata> {
        self.models
            .read()
            .map(|models| {
                models
                    .values()
                    .filter(|m| m.tags.iter().any(|t| t.contains(tag)))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Search models by name (case-insensitive partial match)
    pub fn search_by_name(&self, query: &str) -> Vec<ModelMetadata> {
        let query_lower = query.to_lowercase();
        self.models
            .read()
            .map(|models| {
                models
                    .values()
                    .filter(|m| m.name.to_lowercase().contains(&query_lower))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Remove a model from the registry
    pub fn unregister(&self, id: &str, version: &str) -> Result<ModelMetadata, String> {
        let mut models = self.models.write().map_err(|e| e.to_string())?;
        let key = format!("{}:{}", id, version);
        models
            .remove(&key)
            .ok_or_else(|| format!("Model {} not found", key))
    }

    /// Get the total number of registered models
    pub fn count(&self) -> usize {
        self.models.read().map(|m| m.len()).unwrap_or(0)
    }

    /// Register built-in models
    fn register_builtin_models(&self) {
        // Register popular vision models
        let _ = self.register(
            ModelMetadata::new("resnet50", "ResNet-50", "v1.0")
                .with_domain(ModelDomain::Vision)
                .with_architecture(ModelArchitecture::CNN)
                .with_description("ResNet-50 model pretrained on ImageNet")
                .with_author("Microsoft Research")
                .with_license("MIT")
                .with_dataset("ImageNet-1K")
                .with_input_shape(vec![3, 224, 224])
                .with_num_parameters(25_557_032)
                .with_tags(vec![
                    "cnn".to_string(),
                    "imagenet".to_string(),
                    "classification".to_string(),
                ]),
        );

        let _ = self.register(
            ModelMetadata::new("vit_base", "Vision Transformer Base", "v1.0")
                .with_domain(ModelDomain::Vision)
                .with_architecture(ModelArchitecture::Transformer)
                .with_description("Vision Transformer (ViT-B/16) pretrained on ImageNet-21K")
                .with_author("Google Research")
                .with_license("Apache-2.0")
                .with_dataset("ImageNet-21K")
                .with_input_shape(vec![3, 224, 224])
                .with_num_parameters(86_567_656)
                .with_tags(vec![
                    "transformer".to_string(),
                    "imagenet".to_string(),
                    "classification".to_string(),
                ]),
        );

        // Register NLP models
        let _ = self.register(
            ModelMetadata::new("bert_base", "BERT Base", "v1.0")
                .with_domain(ModelDomain::NLP)
                .with_architecture(ModelArchitecture::Transformer)
                .with_description("BERT Base model with 12 layers")
                .with_author("Google Research")
                .with_license("Apache-2.0")
                .with_dataset("BooksCorpus + Wikipedia")
                .with_num_parameters(110_000_000)
                .with_tags(vec![
                    "transformer".to_string(),
                    "bert".to_string(),
                    "language-model".to_string(),
                ]),
        );

        let _ = self.register(
            ModelMetadata::new("gpt2", "GPT-2", "v1.0")
                .with_domain(ModelDomain::NLP)
                .with_architecture(ModelArchitecture::Transformer)
                .with_description("GPT-2 language model")
                .with_author("OpenAI")
                .with_license("MIT")
                .with_dataset("WebText")
                .with_num_parameters(117_000_000)
                .with_tags(vec![
                    "transformer".to_string(),
                    "gpt".to_string(),
                    "language-model".to_string(),
                    "generation".to_string(),
                ]),
        );
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_metadata_creation() {
        let metadata = ModelMetadata::new("test_model", "Test Model", "1.0")
            .with_domain(ModelDomain::Vision)
            .with_architecture(ModelArchitecture::CNN)
            .with_description("A test model")
            .with_num_parameters(1_000_000);

        assert_eq!(metadata.id, "test_model");
        assert_eq!(metadata.name, "Test Model");
        assert_eq!(metadata.version, "1.0");
        assert_eq!(metadata.domain, ModelDomain::Vision);
        assert_eq!(metadata.num_parameters, Some(1_000_000));
    }

    #[test]
    fn test_model_key() {
        let metadata = ModelMetadata::new("model1", "Model 1", "2.0");
        assert_eq!(metadata.key(), "model1:2.0");
    }

    #[test]
    fn test_registry_registration() {
        let registry = ModelRegistry::new();
        let metadata = ModelMetadata::new("test1", "Test 1", "1.0");

        assert!(registry.register(metadata).is_ok());
        assert_eq!(registry.count(), 1);
    }

    #[test]
    fn test_registry_duplicate_registration() {
        let registry = ModelRegistry::new();
        let metadata1 = ModelMetadata::new("test1", "Test 1", "1.0");
        let metadata2 = ModelMetadata::new("test1", "Test 1", "1.0");

        assert!(registry.register(metadata1).is_ok());
        assert!(registry.register(metadata2).is_err());
    }

    #[test]
    fn test_registry_get_model() {
        let registry = ModelRegistry::new();
        let metadata =
            ModelMetadata::new("test1", "Test 1", "1.0").with_domain(ModelDomain::Vision);

        registry.register(metadata).unwrap();

        let retrieved = registry.get_model("test1", Some("1.0"));
        assert!(retrieved.is_ok());
        assert_eq!(retrieved.unwrap().id, "test1");
    }

    #[test]
    fn test_registry_list_models() {
        let registry = ModelRegistry::new();
        registry
            .register(ModelMetadata::new("model1", "Model 1", "1.0"))
            .unwrap();
        registry
            .register(ModelMetadata::new("model2", "Model 2", "1.0"))
            .unwrap();

        let models = registry.list_models();
        assert_eq!(models.len(), 2);
    }

    #[test]
    fn test_registry_list_by_domain() {
        let registry = ModelRegistry::new();
        registry
            .register(
                ModelMetadata::new("vision1", "Vision 1", "1.0").with_domain(ModelDomain::Vision),
            )
            .unwrap();
        registry
            .register(ModelMetadata::new("nlp1", "NLP 1", "1.0").with_domain(ModelDomain::NLP))
            .unwrap();

        let vision_models = registry.list_models_by_domain(&ModelDomain::Vision);
        assert_eq!(vision_models.len(), 1);
        assert_eq!(vision_models[0].id, "vision1");
    }

    #[test]
    fn test_registry_search_by_tag() {
        let registry = ModelRegistry::new();
        registry
            .register(
                ModelMetadata::new("model1", "Model 1", "1.0")
                    .with_tags(vec!["cnn".to_string(), "classification".to_string()]),
            )
            .unwrap();

        let results = registry.search_by_tag("cnn");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "model1");
    }

    #[test]
    fn test_registry_search_by_name() {
        let registry = ModelRegistry::new();
        registry
            .register(ModelMetadata::new("resnet", "ResNet-50", "1.0"))
            .unwrap();
        registry
            .register(ModelMetadata::new("vgg", "VGG-16", "1.0"))
            .unwrap();

        let results = registry.search_by_name("resnet");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "resnet");
    }

    #[test]
    fn test_registry_unregister() {
        let registry = ModelRegistry::new();
        let metadata = ModelMetadata::new("test1", "Test 1", "1.0");

        registry.register(metadata).unwrap();
        assert_eq!(registry.count(), 1);

        let removed = registry.unregister("test1", "1.0");
        assert!(removed.is_ok());
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_global_registry() {
        let registry = ModelRegistry::global();

        // Global registry should have builtin models
        assert!(registry.count() > 0);

        // Should be able to retrieve builtin models
        let resnet = registry.get_model("resnet50", Some("v1.0"));
        assert!(resnet.is_ok());
    }

    #[test]
    fn test_model_domain_display() {
        assert_eq!(ModelDomain::Vision.to_string(), "vision");
        assert_eq!(ModelDomain::NLP.to_string(), "nlp");
        assert_eq!(ModelDomain::Custom("test".to_string()).to_string(), "test");
    }
}
