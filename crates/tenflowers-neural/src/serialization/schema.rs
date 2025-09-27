//! Schema validation for model serialization
//!
//! This module provides comprehensive schema validation for serialized models,
//! ensuring data integrity and compatibility.

use super::{AdvancedModelState, ModelMetadata, ParameterInfo, SemanticVersion};
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tenflowers_core::{Result, TensorError};

/// Schema validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
    /// Schema version used for validation
    pub schema_version: SemanticVersion,
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error type
    pub error_type: ValidationErrorType,
    /// Error message
    pub message: String,
    /// Field path where error occurred
    pub field_path: Option<String>,
    /// Expected value
    pub expected: Option<String>,
    /// Actual value
    pub actual: Option<String>,
}

/// Validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Warning type
    pub warning_type: ValidationWarningType,
    /// Warning message
    pub message: String,
    /// Field path where warning occurred
    pub field_path: Option<String>,
}

/// Types of validation errors
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationErrorType {
    /// Missing required field
    MissingField,
    /// Invalid field type
    InvalidType,
    /// Invalid field value
    InvalidValue,
    /// Schema version mismatch
    VersionMismatch,
    /// Checksum mismatch
    ChecksumMismatch,
    /// Invalid parameter shape
    InvalidShape,
    /// Unsupported format
    UnsupportedFormat,
    /// Corrupted data
    CorruptedData,
}

/// Types of validation warnings
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationWarningType {
    /// Deprecated field
    DeprecatedField,
    /// Missing optional field
    MissingOptionalField,
    /// Unusual value
    UnusualValue,
    /// Performance concern
    PerformanceConcern,
    /// Compatibility issue
    CompatibilityIssue,
}

/// Schema definition for model serialization
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct ModelSchema {
    /// Schema version
    pub version: SemanticVersion,
    /// Required fields
    pub required_fields: Vec<FieldSchema>,
    /// Optional fields
    pub optional_fields: Vec<FieldSchema>,
    /// Parameter constraints
    pub parameter_constraints: ParameterConstraints,
    /// Metadata constraints
    pub metadata_constraints: MetadataConstraints,
}

/// Field schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSchema {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: FieldType,
    /// Field description
    pub description: Option<String>,
    /// Validation rules
    pub validation_rules: Vec<ValidationRule>,
}

/// Supported field types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FieldType {
    String,
    Integer,
    Float,
    Boolean,
    Array(Box<FieldType>),
    Object(HashMap<String, FieldType>),
    Version,
    Bytes,
}

/// Validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule type
    pub rule_type: ValidationRuleType,
    /// Rule value
    pub value: String,
    /// Error message if rule fails
    pub error_message: Option<String>,
}

/// Types of validation rules
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationRuleType {
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Minimum length
    MinLength,
    /// Maximum length
    MaxLength,
    /// Regular expression pattern
    Pattern,
    /// Allowed values
    Enum,
    /// Custom validation function
    Custom,
}

/// Parameter constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterConstraints {
    /// Maximum number of parameters
    pub max_parameters: Option<usize>,
    /// Maximum parameter size
    pub max_parameter_size: Option<usize>,
    /// Allowed data types
    pub allowed_dtypes: Vec<String>,
    /// Allowed devices
    pub allowed_devices: Vec<String>,
    /// Maximum tensor rank
    pub max_tensor_rank: Option<usize>,
}

/// Metadata constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataConstraints {
    /// Required metadata fields
    pub required_fields: Vec<String>,
    /// Maximum metadata size
    pub max_metadata_size: Option<usize>,
    /// Allowed model types
    pub allowed_model_types: Vec<String>,
}

/// Schema validator
pub struct SchemaValidator {
    /// Registered schemas by version
    schemas: HashMap<SemanticVersion, ModelSchema>,
    /// Current schema version
    current_version: SemanticVersion,
}

impl SchemaValidator {
    /// Create a new schema validator
    pub fn new() -> Self {
        let mut validator = Self {
            schemas: HashMap::new(),
            current_version: SemanticVersion::new(0, 1, 0),
        };

        // Register default schemas
        validator.register_default_schemas();
        validator
    }

    /// Register default schemas
    fn register_default_schemas(&mut self) {
        // Schema for version 0.1.0
        let v01_schema = ModelSchema {
            version: SemanticVersion::new(0, 1, 0),
            required_fields: vec![
                FieldSchema {
                    name: "metadata".to_string(),
                    field_type: FieldType::Object(HashMap::new()),
                    description: Some("Model metadata".to_string()),
                    validation_rules: vec![],
                },
                FieldSchema {
                    name: "parameters_info".to_string(),
                    field_type: FieldType::Array(Box::new(FieldType::Object(HashMap::new()))),
                    description: Some("Parameter information".to_string()),
                    validation_rules: vec![],
                },
                FieldSchema {
                    name: "parameters_data".to_string(),
                    field_type: FieldType::Bytes,
                    description: Some("Serialized parameter data".to_string()),
                    validation_rules: vec![],
                },
                FieldSchema {
                    name: "schema_hash".to_string(),
                    field_type: FieldType::String,
                    description: Some("Schema validation hash".to_string()),
                    validation_rules: vec![ValidationRule {
                        rule_type: ValidationRuleType::MinLength,
                        value: "1".to_string(),
                        error_message: Some("Schema hash cannot be empty".to_string()),
                    }],
                },
            ],
            optional_fields: vec![FieldSchema {
                name: "compression_info".to_string(),
                field_type: FieldType::Object(HashMap::new()),
                description: Some("Compression information".to_string()),
                validation_rules: vec![],
            }],
            parameter_constraints: ParameterConstraints {
                max_parameters: Some(1_000_000),
                max_parameter_size: Some(1024 * 1024 * 1024), // 1GB
                allowed_dtypes: vec![
                    "f32".to_string(),
                    "f64".to_string(),
                    "i32".to_string(),
                    "i64".to_string(),
                ],
                allowed_devices: vec!["CPU".to_string(), "GPU".to_string()],
                max_tensor_rank: Some(8),
            },
            metadata_constraints: MetadataConstraints {
                required_fields: vec!["model_type".to_string(), "version".to_string()],
                max_metadata_size: Some(1024 * 1024), // 1MB
                allowed_model_types: vec!["Sequential".to_string(), "Functional".to_string()],
            },
        };

        self.schemas
            .insert(SemanticVersion::new(0, 1, 0), v01_schema);
    }

    /// Register a schema for a specific version
    pub fn register_schema(&mut self, version: SemanticVersion, schema: ModelSchema) {
        self.schemas.insert(version, schema);
    }

    /// Validate a model state against the schema
    pub fn validate(&self, model_state: &AdvancedModelState) -> ValidationResult {
        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            schema_version: self.current_version.clone(),
        };

        // Get schema for the model version
        let schema = match self.schemas.get(&model_state.metadata.version) {
            Some(schema) => schema,
            None => {
                result.errors.push(ValidationError {
                    error_type: ValidationErrorType::VersionMismatch,
                    message: format!(
                        "No schema found for version {}",
                        model_state.metadata.version
                    ),
                    field_path: Some("metadata.version".to_string()),
                    expected: Some("Supported version".to_string()),
                    actual: Some(model_state.metadata.version.to_string()),
                });
                result.is_valid = false;
                return result;
            }
        };

        // Validate metadata
        self.validate_metadata(&model_state.metadata, schema, &mut result);

        // Validate parameters
        self.validate_parameters(&model_state.parameters_info, schema, &mut result);

        // Validate schema hash
        self.validate_schema_hash(&model_state.schema_hash, &mut result);

        // Validate compression info if present
        if let Some(compression_info) = &model_state.compression_info {
            self.validate_compression_info(compression_info, &mut result);
        }

        result
    }

    /// Validate metadata
    fn validate_metadata(
        &self,
        metadata: &ModelMetadata,
        schema: &ModelSchema,
        result: &mut ValidationResult,
    ) {
        // Check required metadata fields
        for required_field in &schema.metadata_constraints.required_fields {
            match required_field.as_str() {
                "model_type" => {
                    if metadata.model_type.is_empty() {
                        result.errors.push(ValidationError {
                            error_type: ValidationErrorType::MissingField,
                            message: "Model type is required".to_string(),
                            field_path: Some("metadata.model_type".to_string()),
                            expected: Some("Non-empty string".to_string()),
                            actual: Some("Empty string".to_string()),
                        });
                        result.is_valid = false;
                    }
                }
                "version" => {
                    // Version validation is handled separately
                }
                _ => {
                    // Check custom metadata fields
                    if !metadata.custom.contains_key(required_field) {
                        result.warnings.push(ValidationWarning {
                            warning_type: ValidationWarningType::MissingOptionalField,
                            message: format!("Missing optional metadata field: {}", required_field),
                            field_path: Some(format!("metadata.custom.{}", required_field)),
                        });
                    }
                }
            }
        }

        // Check allowed model types
        if !schema
            .metadata_constraints
            .allowed_model_types
            .contains(&metadata.model_type)
        {
            result.warnings.push(ValidationWarning {
                warning_type: ValidationWarningType::UnusualValue,
                message: format!("Unusual model type: {}", metadata.model_type),
                field_path: Some("metadata.model_type".to_string()),
            });
        }

        // Check parameter count
        if let Some(max_params) = schema.parameter_constraints.max_parameters {
            if metadata.parameter_count > max_params {
                result.errors.push(ValidationError {
                    error_type: ValidationErrorType::InvalidValue,
                    message: format!(
                        "Too many parameters: {} > {}",
                        metadata.parameter_count, max_params
                    ),
                    field_path: Some("metadata.parameter_count".to_string()),
                    expected: Some(format!("<= {}", max_params)),
                    actual: Some(metadata.parameter_count.to_string()),
                });
                result.is_valid = false;
            }
        }

        // Check model size
        if metadata.model_size == 0 {
            result.warnings.push(ValidationWarning {
                warning_type: ValidationWarningType::UnusualValue,
                message: "Model size is zero".to_string(),
                field_path: Some("metadata.model_size".to_string()),
            });
        }

        // Check framework version format
        if !metadata.framework_version.starts_with("TenfloweRS-") {
            result.warnings.push(ValidationWarning {
                warning_type: ValidationWarningType::UnusualValue,
                message: "Unusual framework version format".to_string(),
                field_path: Some("metadata.framework_version".to_string()),
            });
        }
    }

    /// Validate parameters
    fn validate_parameters(
        &self,
        parameters_info: &[ParameterInfo],
        schema: &ModelSchema,
        result: &mut ValidationResult,
    ) {
        // Check maximum number of parameters
        if let Some(max_params) = schema.parameter_constraints.max_parameters {
            if parameters_info.len() > max_params {
                result.errors.push(ValidationError {
                    error_type: ValidationErrorType::InvalidValue,
                    message: format!(
                        "Too many parameters: {} > {}",
                        parameters_info.len(),
                        max_params
                    ),
                    field_path: Some("parameters_info".to_string()),
                    expected: Some(format!("<= {}", max_params)),
                    actual: Some(parameters_info.len().to_string()),
                });
                result.is_valid = false;
            }
        }

        // Validate each parameter
        for (i, param) in parameters_info.iter().enumerate() {
            let field_path = format!("parameters_info[{}]", i);

            // Check parameter name
            if param.name.is_empty() {
                result.errors.push(ValidationError {
                    error_type: ValidationErrorType::MissingField,
                    message: "Parameter name is required".to_string(),
                    field_path: Some(format!("{}.name", field_path)),
                    expected: Some("Non-empty string".to_string()),
                    actual: Some("Empty string".to_string()),
                });
                result.is_valid = false;
            }

            // Check data type
            if !schema
                .parameter_constraints
                .allowed_dtypes
                .contains(&param.dtype)
            {
                result.errors.push(ValidationError {
                    error_type: ValidationErrorType::InvalidValue,
                    message: format!("Unsupported data type: {}", param.dtype),
                    field_path: Some(format!("{}.dtype", field_path)),
                    expected: Some(format!(
                        "One of: {:?}",
                        schema.parameter_constraints.allowed_dtypes
                    )),
                    actual: Some(param.dtype.clone()),
                });
                result.is_valid = false;
            }

            // Check device
            if !schema
                .parameter_constraints
                .allowed_devices
                .contains(&param.device)
            {
                result.warnings.push(ValidationWarning {
                    warning_type: ValidationWarningType::UnusualValue,
                    message: format!("Unusual device: {}", param.device),
                    field_path: Some(format!("{}.device", field_path)),
                });
            }

            // Check tensor rank
            if let Some(max_rank) = schema.parameter_constraints.max_tensor_rank {
                if param.shape.len() > max_rank {
                    result.errors.push(ValidationError {
                        error_type: ValidationErrorType::InvalidShape,
                        message: format!(
                            "Tensor rank too high: {} > {}",
                            param.shape.len(),
                            max_rank
                        ),
                        field_path: Some(format!("{}.shape", field_path)),
                        expected: Some(format!("<= {}", max_rank)),
                        actual: Some(param.shape.len().to_string()),
                    });
                    result.is_valid = false;
                }
            }

            // Check shape validity
            if param.shape.is_empty() {
                result.errors.push(ValidationError {
                    error_type: ValidationErrorType::InvalidShape,
                    message: "Parameter shape cannot be empty".to_string(),
                    field_path: Some(format!("{}.shape", field_path)),
                    expected: Some("Non-empty shape".to_string()),
                    actual: Some("Empty shape".to_string()),
                });
                result.is_valid = false;
            }

            // Check for zero dimensions
            if param.shape.iter().any(|&dim| dim == 0) {
                result.errors.push(ValidationError {
                    error_type: ValidationErrorType::InvalidShape,
                    message: "Parameter shape contains zero dimension".to_string(),
                    field_path: Some(format!("{}.shape", field_path)),
                    expected: Some("All dimensions > 0".to_string()),
                    actual: Some(format!("{:?}", param.shape)),
                });
                result.is_valid = false;
            }
        }
    }

    /// Validate schema hash
    fn validate_schema_hash(&self, schema_hash: &str, result: &mut ValidationResult) {
        if schema_hash.is_empty() {
            result.errors.push(ValidationError {
                error_type: ValidationErrorType::MissingField,
                message: "Schema hash is required".to_string(),
                field_path: Some("schema_hash".to_string()),
                expected: Some("Non-empty string".to_string()),
                actual: Some("Empty string".to_string()),
            });
            result.is_valid = false;
        }
    }

    /// Validate compression info
    fn validate_compression_info(
        &self,
        compression_info: &super::CompressionInfo,
        result: &mut ValidationResult,
    ) {
        // Check compression ratio
        if compression_info.compression_ratio > 1.0 {
            result.warnings.push(ValidationWarning {
                warning_type: ValidationWarningType::UnusualValue,
                message: format!(
                    "Unusual compression ratio: {}",
                    compression_info.compression_ratio
                ),
                field_path: Some("compression_info.compression_ratio".to_string()),
            });
        }

        // Check if compressed size is valid
        if compression_info.compressed_size > compression_info.original_size {
            result.warnings.push(ValidationWarning {
                warning_type: ValidationWarningType::PerformanceConcern,
                message: "Compressed size larger than original size".to_string(),
                field_path: Some("compression_info".to_string()),
            });
        }
    }

    /// Get schema for a specific version
    pub fn get_schema(&self, version: &SemanticVersion) -> Option<&ModelSchema> {
        self.schemas.get(version)
    }

    /// Get all supported schema versions
    pub fn get_supported_versions(&self) -> Vec<&SemanticVersion> {
        self.schemas.keys().collect()
    }

    /// Generate schema hash for a model state
    pub fn generate_schema_hash(&self, model_state: &AdvancedModelState) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash metadata
        model_state.metadata.model_type.hash(&mut hasher);
        model_state.metadata.version.hash(&mut hasher);

        // Hash parameter info
        for param in &model_state.parameters_info {
            param.name.hash(&mut hasher);
            param.shape.hash(&mut hasher);
            param.dtype.hash(&mut hasher);
        }

        format!("{:x}", hasher.finish())
    }
}

impl Default for SchemaValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for schema validation
pub mod utils {
    use super::*;

    /// Create a basic validation result
    pub fn create_validation_result(is_valid: bool) -> ValidationResult {
        ValidationResult {
            is_valid,
            errors: Vec::new(),
            warnings: Vec::new(),
            schema_version: SemanticVersion::new(0, 1, 0),
        }
    }

    /// Check if a field type is compatible with a value
    pub fn is_type_compatible(field_type: &FieldType, value: &str) -> bool {
        match field_type {
            FieldType::String => true,
            FieldType::Integer => value.parse::<i64>().is_ok(),
            FieldType::Float => value.parse::<f64>().is_ok(),
            FieldType::Boolean => value.parse::<bool>().is_ok(),
            FieldType::Version => SemanticVersion::new(0, 0, 0).to_string().contains('.'),
            FieldType::Bytes => true,
            FieldType::Array(_) => value.starts_with('[') && value.ends_with(']'),
            FieldType::Object(_) => value.starts_with('{') && value.ends_with('}'),
        }
    }

    /// Validate a field against its schema
    pub fn validate_field_value(
        field_schema: &FieldSchema,
        value: &str,
    ) -> Result<Vec<ValidationError>> {
        let mut errors = Vec::new();

        // Check type compatibility
        if !is_type_compatible(&field_schema.field_type, value) {
            errors.push(ValidationError {
                error_type: ValidationErrorType::InvalidType,
                message: format!("Invalid type for field '{}'", field_schema.name),
                field_path: Some(field_schema.name.clone()),
                expected: Some(format!("{:?}", field_schema.field_type)),
                actual: Some(value.to_string()),
            });
        }

        // Validate rules
        for rule in &field_schema.validation_rules {
            if let Err(error) = validate_rule(rule, value, &field_schema.name) {
                errors.push(error);
            }
        }

        Ok(errors)
    }

    /// Validate a single rule
    fn validate_rule(
        rule: &ValidationRule,
        value: &str,
        field_name: &str,
    ) -> std::result::Result<(), ValidationError> {
        match rule.rule_type {
            ValidationRuleType::MinLength => {
                let min_len: usize = rule.value.parse().map_err(|_| ValidationError {
                    error_type: ValidationErrorType::InvalidValue,
                    message: "Invalid min length rule".to_string(),
                    field_path: Some(field_name.to_string()),
                    expected: None,
                    actual: None,
                })?;

                if value.len() < min_len {
                    return Err(ValidationError {
                        error_type: ValidationErrorType::InvalidValue,
                        message: rule.error_message.clone().unwrap_or_else(|| {
                            format!("Value too short for field '{}'", field_name)
                        }),
                        field_path: Some(field_name.to_string()),
                        expected: Some(format!(">= {} characters", min_len)),
                        actual: Some(format!("{} characters", value.len())),
                    });
                }
            }
            ValidationRuleType::MaxLength => {
                let max_len: usize = rule.value.parse().map_err(|_| ValidationError {
                    error_type: ValidationErrorType::InvalidValue,
                    message: "Invalid max length rule".to_string(),
                    field_path: Some(field_name.to_string()),
                    expected: None,
                    actual: None,
                })?;

                if value.len() > max_len {
                    return Err(ValidationError {
                        error_type: ValidationErrorType::InvalidValue,
                        message: rule.error_message.clone().unwrap_or_else(|| {
                            format!("Value too long for field '{}'", field_name)
                        }),
                        field_path: Some(field_name.to_string()),
                        expected: Some(format!("<= {} characters", max_len)),
                        actual: Some(format!("{} characters", value.len())),
                    });
                }
            }
            ValidationRuleType::Pattern => {
                // Simple pattern matching (full regex support would require regex crate)
                if rule.value == "non_empty" && value.is_empty() {
                    return Err(ValidationError {
                        error_type: ValidationErrorType::InvalidValue,
                        message: rule
                            .error_message
                            .clone()
                            .unwrap_or_else(|| format!("Field '{}' cannot be empty", field_name)),
                        field_path: Some(field_name.to_string()),
                        expected: Some("Non-empty value".to_string()),
                        actual: Some("Empty value".to_string()),
                    });
                }
            }
            ValidationRuleType::Enum => {
                let allowed_values: Vec<&str> = rule.value.split(',').collect();
                if !allowed_values.contains(&value) {
                    return Err(ValidationError {
                        error_type: ValidationErrorType::InvalidValue,
                        message: rule
                            .error_message
                            .clone()
                            .unwrap_or_else(|| format!("Invalid value for field '{}'", field_name)),
                        field_path: Some(field_name.to_string()),
                        expected: Some(format!("One of: {}", rule.value)),
                        actual: Some(value.to_string()),
                    });
                }
            }
            _ => {
                // Other rule types not implemented yet
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::{HardwareRequirements, TrainingInfo};
    use super::*;

    #[test]
    fn test_schema_validator_creation() {
        let validator = SchemaValidator::new();
        assert!(!validator.schemas.is_empty());
        assert!(validator
            .get_schema(&SemanticVersion::new(0, 1, 0))
            .is_some());
    }

    #[test]
    fn test_validation_result_creation() {
        let result = utils::create_validation_result(true);
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_field_type_compatibility() {
        assert!(utils::is_type_compatible(&FieldType::String, "hello"));
        assert!(utils::is_type_compatible(&FieldType::Integer, "42"));
        assert!(!utils::is_type_compatible(&FieldType::Integer, "hello"));
        assert!(utils::is_type_compatible(&FieldType::Float, "3.14"));
        assert!(utils::is_type_compatible(&FieldType::Boolean, "true"));
    }

    #[test]
    fn test_validation_error_creation() {
        let error = ValidationError {
            error_type: ValidationErrorType::MissingField,
            message: "Test error".to_string(),
            field_path: Some("test.field".to_string()),
            expected: Some("value".to_string()),
            actual: Some("empty".to_string()),
        };

        assert_eq!(error.error_type, ValidationErrorType::MissingField);
        assert_eq!(error.message, "Test error");
    }

    #[test]
    fn test_validation_warning_creation() {
        let warning = ValidationWarning {
            warning_type: ValidationWarningType::DeprecatedField,
            message: "Test warning".to_string(),
            field_path: Some("test.field".to_string()),
        };

        assert_eq!(warning.warning_type, ValidationWarningType::DeprecatedField);
        assert_eq!(warning.message, "Test warning");
    }

    #[test]
    fn test_model_validation() {
        let validator = SchemaValidator::new();

        // Create a valid model state
        let metadata = ModelMetadata {
            model_type: "Sequential".to_string(),
            version: SemanticVersion::new(0, 1, 0),
            framework_version: "TenfloweRS-0.1.0".to_string(),
            created_at: "2023-01-01T00:00:00Z".to_string(),
            architecture_hash: "test_hash".to_string(),
            parameter_count: 1000,
            model_size: 4000,
            training_info: TrainingInfo {
                epochs: Some(10),
                final_loss: Some(0.1),
                validation_accuracy: Some(0.9),
                optimizer: Some("Adam".to_string()),
                learning_rate: Some(0.001),
                dataset_info: Some("MNIST".to_string()),
            },
            hardware_requirements: HardwareRequirements {
                min_memory: 1024,
                recommended_memory: 2048,
                gpu_required: false,
                cpu_features: vec![],
                target_device: "CPU".to_string(),
            },
            custom: HashMap::new(),
        };

        let parameter_info = ParameterInfo {
            name: "weight".to_string(),
            shape: vec![784, 10],
            dtype: "f32".to_string(),
            device: "CPU".to_string(),
            requires_grad: true,
            checksum: 12345,
        };

        let model_state = AdvancedModelState {
            metadata,
            parameters_info: vec![parameter_info],
            parameters_data: vec![1, 2, 3, 4],
            compression_info: None,
            schema_hash: "test_schema_hash".to_string(),
        };

        let result = validator.validate(&model_state);
        assert!(result.is_valid);
    }
}
