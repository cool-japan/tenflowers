//! Enhanced schema validation system
//!
//! This module provides comprehensive schema validation capabilities for ensuring
//! data integrity across different formats and operations.

use crate::error_taxonomy::helpers as error_helpers;
use crate::formats::unified_reader::{DataType, FieldInfo, FormatMetadata};
use tenflowers_core::{DType, Result, TensorError};

/// Schema validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Whether to enforce strict type matching
    pub strict_types: bool,
    /// Whether to enforce field order
    pub enforce_field_order: bool,
    /// Whether to allow nullable fields
    pub allow_nullable: bool,
    /// Whether to validate shapes
    pub validate_shapes: bool,
    /// Maximum allowed field count
    pub max_fields: Option<usize>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strict_types: true,
            enforce_field_order: false,
            allow_nullable: true,
            validate_shapes: true,
            max_fields: None,
        }
    }
}

/// Schema validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
}

impl ValidationResult {
    /// Create a successful validation result
    pub fn success() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Create a failed validation result
    pub fn failure(error: ValidationError) -> Self {
        Self {
            is_valid: false,
            errors: vec![error],
            warnings: Vec::new(),
        }
    }

    /// Add an error
    pub fn add_error(&mut self, error: ValidationError) {
        self.is_valid = false;
        self.errors.push(error);
    }

    /// Add a warning
    pub fn add_warning(&mut self, warning: ValidationWarning) {
        self.warnings.push(warning);
    }
}

/// Schema validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error category
    pub category: ValidationErrorCategory,
    /// Field name (if applicable)
    pub field_name: Option<String>,
    /// Error message
    pub message: String,
}

/// Validation error categories
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationErrorCategory {
    /// Type mismatch
    TypeMismatch,
    /// Missing required field
    MissingField,
    /// Shape mismatch
    ShapeMismatch,
    /// Nullable violation
    NullableViolation,
    /// Field count mismatch
    FieldCountMismatch,
    /// Field order mismatch
    FieldOrderMismatch,
    /// Unsupported type
    UnsupportedType,
}

/// Schema validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Warning message
    pub message: String,
    /// Field name (if applicable)
    pub field_name: Option<String>,
}

/// Enhanced schema validator
pub struct SchemaValidator {
    config: ValidationConfig,
}

impl SchemaValidator {
    /// Create a new schema validator with default configuration
    pub fn new() -> Self {
        Self {
            config: ValidationConfig::default(),
        }
    }

    /// Create a schema validator with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Validate schema against expected fields
    pub fn validate(
        &self,
        actual_metadata: &FormatMetadata,
        expected_fields: &[FieldInfo],
    ) -> ValidationResult {
        let mut result = ValidationResult::success();

        // Validate field count
        if let Err(e) = self.validate_field_count(actual_metadata, expected_fields) {
            result.add_error(e);
            return result; // Early return on field count mismatch
        }

        // Validate each field
        for (i, expected_field) in expected_fields.iter().enumerate() {
            if let Some(actual_field) = self.find_field(&actual_metadata.fields, expected_field, i)
            {
                // Validate field
                if let Err(errors) = self.validate_field(expected_field, actual_field) {
                    for error in errors {
                        result.add_error(error);
                    }
                }
            } else {
                result.add_error(ValidationError {
                    category: ValidationErrorCategory::MissingField,
                    field_name: Some(expected_field.name.clone()),
                    message: format!("Required field '{}' not found", expected_field.name),
                });
            }
        }

        result
    }

    /// Validate metadata structure
    pub fn validate_metadata(&self, metadata: &FormatMetadata) -> ValidationResult {
        let mut result = ValidationResult::success();

        // Check maximum field count
        if let Some(max_fields) = self.config.max_fields {
            if metadata.fields.len() > max_fields {
                result.add_error(ValidationError {
                    category: ValidationErrorCategory::FieldCountMismatch,
                    field_name: None,
                    message: format!(
                        "Too many fields: {} (max: {})",
                        metadata.fields.len(),
                        max_fields
                    ),
                });
            }
        }

        // Validate each field
        for field in &metadata.fields {
            if let Err(errors) = self.validate_field_structure(field) {
                for error in errors {
                    result.add_error(error);
                }
            }
        }

        result
    }

    /// Validate field count
    fn validate_field_count(
        &self,
        actual_metadata: &FormatMetadata,
        expected_fields: &[FieldInfo],
    ) -> std::result::Result<(), ValidationError> {
        if actual_metadata.fields.len() != expected_fields.len() {
            return Err(ValidationError {
                category: ValidationErrorCategory::FieldCountMismatch,
                field_name: None,
                message: format!(
                    "Field count mismatch: expected {}, got {}",
                    expected_fields.len(),
                    actual_metadata.fields.len()
                ),
            });
        }
        Ok(())
    }

    /// Find field in actual fields
    fn find_field<'a>(
        &self,
        actual_fields: &'a [FieldInfo],
        expected_field: &FieldInfo,
        index: usize,
    ) -> Option<&'a FieldInfo> {
        if self.config.enforce_field_order {
            // Use positional matching
            actual_fields.get(index)
        } else {
            // Use name matching
            actual_fields.iter().find(|f| f.name == expected_field.name)
        }
    }

    /// Validate a single field
    fn validate_field(
        &self,
        expected: &FieldInfo,
        actual: &FieldInfo,
    ) -> std::result::Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        // Validate type
        if let Err(e) = self.validate_type(&expected.dtype, &actual.dtype, &expected.name) {
            errors.push(e);
        }

        // Validate nullable
        if !self.config.allow_nullable && actual.nullable && !expected.nullable {
            errors.push(ValidationError {
                category: ValidationErrorCategory::NullableViolation,
                field_name: Some(expected.name.clone()),
                message: format!("Field '{}' is nullable but should not be", expected.name),
            });
        }

        // Validate shape
        if self.config.validate_shapes {
            if let Err(e) = self.validate_shape(&expected.shape, &actual.shape, &expected.name) {
                errors.push(e);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Validate data type
    fn validate_type(
        &self,
        expected: &DataType,
        actual: &DataType,
        field_name: &str,
    ) -> std::result::Result<(), ValidationError> {
        if self.config.strict_types {
            // Exact match required
            if expected != actual {
                return Err(ValidationError {
                    category: ValidationErrorCategory::TypeMismatch,
                    field_name: Some(field_name.to_string()),
                    message: format!(
                        "Type mismatch for field '{}': expected {:?}, got {:?}",
                        field_name, expected, actual
                    ),
                });
            }
        } else {
            // Compatible types allowed
            if !self.are_types_compatible(expected, actual) {
                return Err(ValidationError {
                    category: ValidationErrorCategory::TypeMismatch,
                    field_name: Some(field_name.to_string()),
                    message: format!(
                        "Incompatible type for field '{}': expected {:?}, got {:?}",
                        field_name, expected, actual
                    ),
                });
            }
        }
        Ok(())
    }

    /// Check if types are compatible
    fn are_types_compatible(&self, expected: &DataType, actual: &DataType) -> bool {
        match (expected, actual) {
            // Exact matches
            (a, b) if a == b => true,

            // Numeric type compatibility
            (
                DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64,
                DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64,
            ) => true,

            // List compatibility
            (DataType::List(inner1), DataType::List(inner2)) => {
                self.are_types_compatible(inner1, inner2)
            }

            _ => false,
        }
    }

    /// Validate shape
    fn validate_shape(
        &self,
        expected: &Option<Vec<usize>>,
        actual: &Option<Vec<usize>>,
        field_name: &str,
    ) -> std::result::Result<(), ValidationError> {
        match (expected, actual) {
            (Some(exp), Some(act)) if exp != act => {
                return Err(ValidationError {
                    category: ValidationErrorCategory::ShapeMismatch,
                    field_name: Some(field_name.to_string()),
                    message: format!(
                        "Shape mismatch for field '{}': expected {:?}, got {:?}",
                        field_name, exp, act
                    ),
                });
            }
            (Some(_), None) => {
                return Err(ValidationError {
                    category: ValidationErrorCategory::ShapeMismatch,
                    field_name: Some(field_name.to_string()),
                    message: format!("Shape expected for field '{}' but not provided", field_name),
                });
            }
            _ => {}
        }
        Ok(())
    }

    /// Validate field structure
    fn validate_field_structure(
        &self,
        field: &FieldInfo,
    ) -> std::result::Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        // Check for empty field name
        if field.name.is_empty() {
            errors.push(ValidationError {
                category: ValidationErrorCategory::MissingField,
                field_name: None,
                message: "Field name cannot be empty".to_string(),
            });
        }

        // Validate nested types
        if let Err(e) = self.validate_type_structure(&field.dtype, &field.name) {
            errors.push(e);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Validate type structure
    fn validate_type_structure(
        &self,
        dtype: &DataType,
        field_name: &str,
    ) -> std::result::Result<(), ValidationError> {
        match dtype {
            DataType::Struct(fields) if fields.is_empty() => {
                return Err(ValidationError {
                    category: ValidationErrorCategory::UnsupportedType,
                    field_name: Some(field_name.to_string()),
                    message: format!("Struct type for field '{}' has no fields", field_name),
                });
            }
            DataType::List(inner) => {
                self.validate_type_structure(inner, field_name)?;
            }
            _ => {}
        }
        Ok(())
    }
}

impl Default for SchemaValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_success() {
        let validator = SchemaValidator::new();

        let expected = vec![FieldInfo {
            name: "feature".to_string(),
            dtype: DataType::Float32,
            shape: Some(vec![10]),
            nullable: false,
            description: None,
        }];

        let metadata = FormatMetadata {
            format_name: "Test".to_string(),
            version: None,
            num_samples: 100,
            fields: expected.clone(),
            metadata: std::collections::HashMap::new(),
            supports_random_access: true,
            supports_streaming: true,
        };

        let result = validator.validate(&metadata, &expected);
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_type_mismatch() {
        let validator = SchemaValidator::new();

        let expected = vec![FieldInfo {
            name: "feature".to_string(),
            dtype: DataType::Float32,
            shape: None,
            nullable: false,
            description: None,
        }];

        let metadata = FormatMetadata {
            format_name: "Test".to_string(),
            version: None,
            num_samples: 100,
            fields: vec![FieldInfo {
                name: "feature".to_string(),
                dtype: DataType::String,
                shape: None,
                nullable: false,
                description: None,
            }],
            metadata: std::collections::HashMap::new(),
            supports_random_access: true,
            supports_streaming: true,
        };

        let result = validator.validate(&metadata, &expected);
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
        assert_eq!(
            result.errors[0].category,
            ValidationErrorCategory::TypeMismatch
        );
    }

    #[test]
    fn test_missing_field() {
        let validator = SchemaValidator::new();

        let expected = vec![
            FieldInfo {
                name: "feature1".to_string(),
                dtype: DataType::Float32,
                shape: None,
                nullable: false,
                description: None,
            },
            FieldInfo {
                name: "feature2".to_string(),
                dtype: DataType::Float32,
                shape: None,
                nullable: false,
                description: None,
            },
        ];

        let metadata = FormatMetadata {
            format_name: "Test".to_string(),
            version: None,
            num_samples: 100,
            fields: vec![FieldInfo {
                name: "feature1".to_string(),
                dtype: DataType::Float32,
                shape: None,
                nullable: false,
                description: None,
            }],
            metadata: std::collections::HashMap::new(),
            supports_random_access: true,
            supports_streaming: true,
        };

        let result = validator.validate(&metadata, &expected);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_compatible_numeric_types() {
        let config = ValidationConfig {
            strict_types: false,
            ..Default::default()
        };
        let validator = SchemaValidator::with_config(config);

        assert!(validator.are_types_compatible(&DataType::Float32, &DataType::Float64));
        assert!(validator.are_types_compatible(&DataType::Int32, &DataType::Int64));
    }
}
