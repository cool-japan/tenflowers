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

// ──────────────────────────────────────────────────────────────────────────────
// Extended schema comparison: FieldDiff + ValidationReport + validate_full
// ──────────────────────────────────────────────────────────────────────────────

/// Per-field outcome of a full schema comparison.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldDiff {
    /// Both schemas agree on name, type, and required-ness.
    Compatible,
    /// The actual type is a safe widening of the expected type (e.g. Int32 → Int64).
    Widening {
        /// The expected (narrower) type.
        from: DataType,
        /// The actual (wider) type.
        to: DataType,
    },
    /// Actual type is incompatible with expected type — a hard mismatch.
    TypeMismatch {
        /// Expected type.
        expected: DataType,
        /// Actual type.
        got: DataType,
    },
    /// A field that was required in the expected schema is absent from the actual schema.
    MissingRequired {
        /// Name of the absent field.
        field: String,
    },
    /// A field present in the actual schema is not listed in the expected schema.
    UnexpectedExtra {
        /// Name of the unexpected field.
        field: String,
    },
}

/// Structured outcome of a full schema comparison.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Per-field diff results (one entry per unique field name across both schemas).
    pub diffs: Vec<(String, FieldDiff)>,
    /// `true` iff no `TypeMismatch` or `MissingRequired` entries exist.
    pub compatible: bool,
    /// Non-fatal informational warnings (e.g. widening detected).
    pub warnings: Vec<ValidationWarning>,
    /// Hard errors (type mismatch, missing required fields).
    pub errors: Vec<ValidationError>,
}

impl ValidationReport {
    fn new() -> Self {
        Self {
            diffs: Vec::new(),
            compatible: true,
            warnings: Vec::new(),
            errors: Vec::new(),
        }
    }

    fn push_diff(&mut self, field_name: String, diff: FieldDiff) {
        match &diff {
            FieldDiff::TypeMismatch { expected, got } => {
                self.compatible = false;
                self.errors.push(ValidationError {
                    category: ValidationErrorCategory::TypeMismatch,
                    field_name: Some(field_name.clone()),
                    message: format!(
                        "Type mismatch for '{}': expected {:?}, got {:?}",
                        field_name, expected, got
                    ),
                });
            }
            FieldDiff::MissingRequired { field } => {
                self.compatible = false;
                self.errors.push(ValidationError {
                    category: ValidationErrorCategory::MissingField,
                    field_name: Some(field.clone()),
                    message: format!("Required field '{}' is missing", field),
                });
            }
            FieldDiff::Widening { from, to } => {
                self.warnings.push(ValidationWarning {
                    field_name: Some(field_name.clone()),
                    message: format!("Type widening for '{}': {:?} → {:?}", field_name, from, to),
                });
            }
            FieldDiff::UnexpectedExtra { field } => {
                self.warnings.push(ValidationWarning {
                    field_name: Some(field.clone()),
                    message: format!("Unexpected extra field '{}' in actual schema", field),
                });
            }
            FieldDiff::Compatible => {}
        }
        self.diffs.push((field_name, diff));
    }
}

/// Validation policy used by `validate_full`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationPolicy {
    /// Strict: type widening is treated as an error.
    Strict,
    /// Lenient: type widening is allowed and produces only a warning.
    Lenient,
}

impl SchemaValidator {
    /// Create a strict validator: widening is treated as a hard error.
    pub fn strict() -> Self {
        Self {
            config: ValidationConfig {
                strict_types: true,
                ..ValidationConfig::default()
            },
        }
    }

    /// Create a lenient validator: widening between numeric types is allowed.
    pub fn lenient() -> Self {
        Self {
            config: ValidationConfig {
                strict_types: false,
                ..ValidationConfig::default()
            },
        }
    }

    /// Perform a full structured comparison of `actual` against `expected` fields.
    pub fn validate_full(
        &self,
        actual: &FormatMetadata,
        expected: &[FieldInfo],
    ) -> ValidationReport {
        let policy = if self.config.strict_types {
            ValidationPolicy::Strict
        } else {
            ValidationPolicy::Lenient
        };

        let mut report = ValidationReport::new();

        let actual_map: std::collections::HashMap<&str, &FieldInfo> =
            actual.fields.iter().map(|f| (f.name.as_str(), f)).collect();

        let mut seen_expected: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for exp in expected {
            seen_expected.insert(exp.name.as_str());
            match actual_map.get(exp.name.as_str()) {
                None => {
                    report.push_diff(
                        exp.name.clone(),
                        FieldDiff::MissingRequired {
                            field: exp.name.clone(),
                        },
                    );
                }
                Some(act) => {
                    let diff = Self::classify_type_diff(&exp.dtype, &act.dtype, policy);
                    report.push_diff(exp.name.clone(), diff);
                }
            }
        }

        for act in &actual.fields {
            if !seen_expected.contains(act.name.as_str()) {
                report.push_diff(
                    act.name.clone(),
                    FieldDiff::UnexpectedExtra {
                        field: act.name.clone(),
                    },
                );
            }
        }

        report
    }

    fn classify_type_diff(
        expected: &DataType,
        actual: &DataType,
        policy: ValidationPolicy,
    ) -> FieldDiff {
        if expected == actual {
            return FieldDiff::Compatible;
        }

        if Self::is_widening(expected, actual) {
            return match policy {
                ValidationPolicy::Lenient => FieldDiff::Widening {
                    from: expected.clone(),
                    to: actual.clone(),
                },
                ValidationPolicy::Strict => FieldDiff::TypeMismatch {
                    expected: expected.clone(),
                    got: actual.clone(),
                },
            };
        }

        FieldDiff::TypeMismatch {
            expected: expected.clone(),
            got: actual.clone(),
        }
    }

    fn is_widening(expected: &DataType, actual: &DataType) -> bool {
        use DataType::*;
        matches!(
            (expected, actual),
            (Int8, Int16)
                | (Int8, Int32)
                | (Int8, Int64)
                | (Int16, Int32)
                | (Int16, Int64)
                | (Int32, Int64)
                | (UInt8, UInt16)
                | (UInt8, UInt32)
                | (UInt8, UInt64)
                | (UInt16, UInt32)
                | (UInt16, UInt64)
                | (UInt32, UInt64)
                | (Float32, Float64)
                | (Int8, Float64)
                | (Int16, Float64)
                | (Int32, Float64)
                | (Int64, Float64)
                | (UInt8, Float64)
                | (UInt16, Float64)
                | (UInt32, Float64)
                | (UInt64, Float64)
        )
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

    // ── validate_full tests ──────────────────────────────────────────────────

    fn make_metadata(fields: Vec<FieldInfo>) -> FormatMetadata {
        FormatMetadata {
            format_name: "test".to_string(),
            version: None,
            num_samples: 10,
            fields,
            metadata: std::collections::HashMap::new(),
            supports_random_access: true,
            supports_streaming: true,
        }
    }

    fn field(name: &str, dtype: DataType) -> FieldInfo {
        FieldInfo {
            name: name.to_string(),
            dtype,
            shape: None,
            nullable: false,
            description: None,
        }
    }

    #[test]
    fn test_field_diff_compatible() {
        let validator = SchemaValidator::lenient();
        let expected = vec![field("x", DataType::Float32)];
        let actual = make_metadata(vec![field("x", DataType::Float32)]);
        let report = validator.validate_full(&actual, &expected);
        assert!(report.compatible);
        assert_eq!(report.diffs.len(), 1);
        assert_eq!(report.diffs[0].1, FieldDiff::Compatible);
    }

    #[test]
    fn test_field_diff_widening_lenient() {
        let validator = SchemaValidator::lenient();
        let expected = vec![field("x", DataType::Int32)];
        let actual = make_metadata(vec![field("x", DataType::Int64)]);
        let report = validator.validate_full(&actual, &expected);
        assert!(report.compatible, "lenient widening should be compatible");
        assert!(!report.warnings.is_empty());
        assert!(matches!(
            &report.diffs[0].1,
            FieldDiff::Widening {
                from: DataType::Int32,
                to: DataType::Int64
            }
        ));
    }

    #[test]
    fn test_field_diff_widening_strict() {
        let validator = SchemaValidator::strict();
        let expected = vec![field("x", DataType::Int32)];
        let actual = make_metadata(vec![field("x", DataType::Int64)]);
        let report = validator.validate_full(&actual, &expected);
        assert!(!report.compatible);
        assert!(matches!(
            &report.diffs[0].1,
            FieldDiff::TypeMismatch {
                expected: DataType::Int32,
                got: DataType::Int64
            }
        ));
    }

    #[test]
    fn test_field_diff_type_mismatch() {
        let validator = SchemaValidator::lenient();
        let expected = vec![field("y", DataType::Float64)];
        let actual = make_metadata(vec![field("y", DataType::String)]);
        let report = validator.validate_full(&actual, &expected);
        assert!(!report.compatible);
        assert!(matches!(&report.diffs[0].1, FieldDiff::TypeMismatch { .. }));
    }

    #[test]
    fn test_field_diff_missing_required() {
        let validator = SchemaValidator::lenient();
        let expected = vec![field("a", DataType::Int32), field("b", DataType::Float32)];
        let actual = make_metadata(vec![field("a", DataType::Int32)]);
        let report = validator.validate_full(&actual, &expected);
        assert!(!report.compatible);
        let has_missing = report
            .diffs
            .iter()
            .any(|(_, d)| matches!(d, FieldDiff::MissingRequired { field } if field == "b"));
        assert!(has_missing, "expected MissingRequired for 'b'");
    }

    #[test]
    fn test_field_diff_unexpected_extra() {
        let validator = SchemaValidator::lenient();
        let expected = vec![field("a", DataType::Int32)];
        let actual = make_metadata(vec![
            field("a", DataType::Int32),
            field("z", DataType::Bool),
        ]);
        let report = validator.validate_full(&actual, &expected);
        assert!(
            report.compatible,
            "unexpected extra should not break compatibility"
        );
        let has_extra = report
            .diffs
            .iter()
            .any(|(_, d)| matches!(d, FieldDiff::UnexpectedExtra { field } if field == "z"));
        assert!(has_extra, "expected UnexpectedExtra for 'z'");
    }

    #[test]
    fn test_float32_to_float64_widening() {
        let validator = SchemaValidator::lenient();
        let expected = vec![field("v", DataType::Float32)];
        let actual = make_metadata(vec![field("v", DataType::Float64)]);
        let report = validator.validate_full(&actual, &expected);
        assert!(report.compatible);
        assert!(matches!(
            &report.diffs[0].1,
            FieldDiff::Widening {
                from: DataType::Float32,
                to: DataType::Float64
            }
        ));
    }
}
