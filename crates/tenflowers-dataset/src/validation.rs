use crate::Dataset;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};

// Type aliases for complex types
#[allow(dead_code)]
type SampleData<T> = (usize, (Tensor<T>, Tensor<T>));
type SampleList<T> = [(usize, (Tensor<T>, Tensor<T>))];

#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub check_schema: bool,
    pub check_ranges: bool,
    pub check_duplicates: bool,
    pub check_outliers: bool,
    pub outlier_threshold: f64, // Z-score threshold for outlier detection
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            check_schema: true,
            check_ranges: true,
            check_duplicates: true,
            check_outliers: true,
            outlier_threshold: 3.0, // 3 standard deviations
        }
    }
}

#[derive(Debug, Clone)]
pub struct SchemaInfo {
    pub feature_shape: Vec<usize>,
    pub label_shape: Vec<usize>,
    pub expected_feature_type: String,
    pub expected_label_type: String,
}

#[derive(Debug, Clone)]
pub struct RangeConstraint<T> {
    pub min_value: Option<T>,
    pub max_value: Option<T>,
}

impl<T> RangeConstraint<T> {
    pub fn new(min_value: Option<T>, max_value: Option<T>) -> Self {
        Self {
            min_value,
            max_value,
        }
    }

    pub fn min(min_value: T) -> Self {
        Self {
            min_value: Some(min_value),
            max_value: None,
        }
    }

    pub fn max(max_value: T) -> Self {
        Self {
            min_value: None,
            max_value: Some(max_value),
        }
    }

    pub fn range(min_value: T, max_value: T) -> Self {
        Self {
            min_value: Some(min_value),
            max_value: Some(max_value),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub schema_errors: Vec<String>,
    pub range_errors: Vec<String>,
    pub duplicate_indices: Vec<usize>,
    pub outlier_indices: Vec<usize>,
}

impl ValidationResult {
    pub fn new() -> Self {
        Self {
            is_valid: true,
            schema_errors: Vec::new(),
            range_errors: Vec::new(),
            duplicate_indices: Vec::new(),
            outlier_indices: Vec::new(),
        }
    }

    pub fn has_errors(&self) -> bool {
        !self.schema_errors.is_empty()
            || !self.range_errors.is_empty()
            || !self.duplicate_indices.is_empty()
            || !self.outlier_indices.is_empty()
    }

    pub fn add_schema_error(&mut self, error: String) {
        self.schema_errors.push(error);
        self.is_valid = false;
    }

    pub fn add_range_error(&mut self, error: String) {
        self.range_errors.push(error);
        self.is_valid = false;
    }

    pub fn add_duplicate(&mut self, index: usize) {
        self.duplicate_indices.push(index);
        self.is_valid = false;
    }

    pub fn add_outlier(&mut self, index: usize) {
        self.outlier_indices.push(index);
        self.is_valid = false;
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

pub struct DataValidator<T> {
    config: ValidationConfig,
    schema_info: Option<SchemaInfo>,
    feature_range: Option<RangeConstraint<T>>,
    label_range: Option<RangeConstraint<T>>,
}

impl<T> DataValidator<T>
where
    T: Clone
        + Default
        + PartialEq
        + PartialOrd
        + std::fmt::Display
        + scirs2_core::numeric::Float
        + Send
        + Sync
        + 'static,
{
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            schema_info: None,
            feature_range: None,
            label_range: None,
        }
    }

    pub fn with_schema(mut self, schema: SchemaInfo) -> Self {
        self.schema_info = Some(schema);
        self
    }

    pub fn with_feature_range(mut self, range: RangeConstraint<T>) -> Self {
        self.feature_range = Some(range);
        self
    }

    pub fn with_label_range(mut self, range: RangeConstraint<T>) -> Self {
        self.label_range = Some(range);
        self
    }

    pub fn validate<D: Dataset<T>>(&self, dataset: &D) -> Result<ValidationResult> {
        let mut result = ValidationResult::new();

        if dataset.is_empty() {
            result.add_schema_error("Dataset is empty".to_string());
            return Ok(result);
        }

        // Collect all samples for validation
        let mut samples = Vec::new();
        for i in 0..dataset.len() {
            match dataset.get(i) {
                Ok(sample) => samples.push((i, sample)),
                Err(e) => {
                    result.add_schema_error(format!("Failed to get sample {i}: {e:?}"));
                }
            }
        }

        if self.config.check_schema {
            self.validate_schema(&samples, &mut result)?;
        }

        if self.config.check_ranges {
            self.validate_ranges(&samples, &mut result)?;
        }

        if self.config.check_duplicates {
            self.validate_duplicates(&samples, &mut result)?;
        }

        if self.config.check_outliers {
            self.validate_outliers(&samples, &mut result)?;
        }

        Ok(result)
    }

    fn validate_schema(
        &self,
        samples: &SampleList<T>,
        result: &mut ValidationResult,
    ) -> Result<()> {
        if let Some(ref schema) = self.schema_info {
            for (index, (features, labels)) in samples {
                // Check feature shape
                if features.shape().dims() != schema.feature_shape {
                    result.add_schema_error(format!(
                        "Sample {}: Feature shape mismatch. Expected {:?}, got {:?}",
                        index,
                        schema.feature_shape,
                        features.shape().dims()
                    ));
                }

                // Check label shape
                if labels.shape().dims() != schema.label_shape {
                    result.add_schema_error(format!(
                        "Sample {}: Label shape mismatch. Expected {:?}, got {:?}",
                        index,
                        schema.label_shape,
                        labels.shape().dims()
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_ranges(
        &self,
        samples: &SampleList<T>,
        result: &mut ValidationResult,
    ) -> Result<()> {
        for (index, (features, labels)) in samples {
            // Validate feature ranges
            if let Some(ref range) = self.feature_range {
                if let Some(feature_data) = features.as_slice() {
                    for (i, &value) in feature_data.iter().enumerate() {
                        if let Some(min_val) = &range.min_value {
                            if value < *min_val {
                                result.add_range_error(format!(
                                    "Sample {index}: Feature {i} value {value} below minimum {min_val}"
                                ));
                            }
                        }
                        if let Some(max_val) = &range.max_value {
                            if value > *max_val {
                                result.add_range_error(format!(
                                    "Sample {index}: Feature {i} value {value} above maximum {max_val}"
                                ));
                            }
                        }
                    }
                }
            }

            // Validate label ranges
            if let Some(ref range) = self.label_range {
                if let Some(label_data) = labels.as_slice() {
                    for (i, &value) in label_data.iter().enumerate() {
                        if let Some(min_val) = &range.min_value {
                            if value < *min_val {
                                result.add_range_error(format!(
                                    "Sample {index}: Label {i} value {value} below minimum {min_val}"
                                ));
                            }
                        }
                        if let Some(max_val) = &range.max_value {
                            if value > *max_val {
                                result.add_range_error(format!(
                                    "Sample {index}: Label {i} value {value} above maximum {max_val}"
                                ));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_duplicates(
        &self,
        samples: &SampleList<T>,
        result: &mut ValidationResult,
    ) -> Result<()> {
        let mut seen_features: HashMap<Vec<String>, Vec<usize>> = HashMap::new();

        for (index, (features, _)) in samples {
            if let Some(feature_data) = features.as_slice() {
                // Convert to string representation for comparison
                let feature_key: Vec<String> = feature_data
                    .iter()
                    .map(|&x| format!("{x:.6}")) // Use 6 decimal places for float comparison
                    .collect();

                seen_features.entry(feature_key).or_default().push(*index);
            }
        }

        // Find duplicates
        for (_, indices) in seen_features {
            if indices.len() > 1 {
                for &index in &indices[1..] {
                    // Skip first occurrence
                    result.add_duplicate(index);
                }
            }
        }

        Ok(())
    }

    fn validate_outliers(
        &self,
        samples: &SampleList<T>,
        result: &mut ValidationResult,
    ) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }

        // Collect feature values for statistical analysis
        let mut feature_values: Vec<Vec<T>> = Vec::new();
        let feature_size = if let Some((_, (features, _))) = samples.first() {
            if let Some(data) = features.as_slice() {
                data.len()
            } else {
                return Ok(()); // Can't analyze GPU tensors
            }
        } else {
            return Ok(());
        };

        // Initialize feature value vectors
        for _ in 0..feature_size {
            feature_values.push(Vec::new());
        }

        // Collect all feature values
        for (_, (features, _)) in samples {
            if let Some(data) = features.as_slice() {
                for (i, &value) in data.iter().enumerate() {
                    if i < feature_values.len() {
                        feature_values[i].push(value);
                    }
                }
            }
        }

        // Calculate mean and std for each feature
        let mut means = Vec::new();
        let mut stds = Vec::new();

        for values in &feature_values {
            if values.is_empty() {
                continue;
            }

            let mean = values.iter().copied().fold(T::zero(), |acc, x| acc + x)
                / T::from(values.len()).unwrap();
            means.push(mean);

            let variance = values
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .fold(T::zero(), |acc, x| acc + x)
                / T::from(values.len()).unwrap();

            let std = variance.sqrt();
            stds.push(std);
        }

        // Check for outliers using Z-score
        let threshold = T::from(self.config.outlier_threshold).unwrap();

        for (index, (features, _)) in samples {
            if let Some(data) = features.as_slice() {
                for (i, &value) in data.iter().enumerate() {
                    if i < means.len() && i < stds.len() {
                        let mean = means[i];
                        let std = stds[i];

                        if std > T::zero() {
                            let z_score = ((value - mean) / std).abs();
                            if z_score > threshold {
                                result.add_outlier(*index);
                                break; // One outlier per sample is enough
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

pub trait DatasetValidationExt<T> {
    fn validate(&self, validator: &DataValidator<T>) -> Result<ValidationResult>;
    fn validate_with_config(&self, config: ValidationConfig) -> Result<ValidationResult>;
    fn is_valid(&self) -> Result<bool>;
}

impl<T, D: Dataset<T>> DatasetValidationExt<T> for D
where
    T: Clone
        + Default
        + PartialEq
        + PartialOrd
        + std::fmt::Display
        + scirs2_core::numeric::Float
        + Send
        + Sync
        + 'static,
{
    fn validate(&self, validator: &DataValidator<T>) -> Result<ValidationResult> {
        validator.validate(self)
    }

    fn validate_with_config(&self, config: ValidationConfig) -> Result<ValidationResult> {
        let validator = DataValidator::new(config);
        validator.validate(self)
    }

    fn is_valid(&self) -> Result<bool> {
        let config = ValidationConfig::default();
        let result = self.validate_with_config(config)?;
        Ok(!result.has_errors())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorDataset;
    use tenflowers_core::Tensor;

    #[test]
    fn test_validation_config() {
        let config = ValidationConfig::default();
        assert!(config.check_schema);
        assert!(config.check_ranges);
        assert!(config.check_duplicates);
        assert!(config.check_outliers);
        assert_eq!(config.outlier_threshold, 3.0);
    }

    #[test]
    fn test_range_constraint() {
        let range = RangeConstraint::range(0.0f32, 1.0f32);
        assert_eq!(range.min_value, Some(0.0));
        assert_eq!(range.max_value, Some(1.0));

        let min_only = RangeConstraint::min(-1.0f32);
        assert_eq!(min_only.min_value, Some(-1.0));
        assert_eq!(min_only.max_value, None);

        let max_only = RangeConstraint::max(10.0f32);
        assert_eq!(max_only.min_value, None);
        assert_eq!(max_only.max_value, Some(10.0));
    }

    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult::new();
        assert!(result.is_valid);
        assert!(!result.has_errors());

        result.add_schema_error("Schema error".to_string());
        assert!(!result.is_valid);
        assert!(result.has_errors());
        assert_eq!(result.schema_errors.len(), 1);

        result.add_duplicate(5);
        assert_eq!(result.duplicate_indices.len(), 1);
        assert_eq!(result.duplicate_indices[0], 5);
    }

    #[test]
    fn test_schema_validation() {
        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let schema = SchemaInfo {
            feature_shape: vec![2], // Expect shape [2] after squeezing
            label_shape: vec![],    // Expect scalar after squeezing
            expected_feature_type: "f32".to_string(),
            expected_label_type: "f32".to_string(),
        };

        let validator = DataValidator::new(ValidationConfig::default()).with_schema(schema);

        let result = validator.validate(&dataset).unwrap();
        assert!(result.is_valid);
        assert!(!result.has_errors());
    }

    #[test]
    fn test_range_validation() {
        let features = Tensor::<f32>::from_vec(
            vec![0.5, 0.8, 1.2, 0.3], // 1.2 is above range [0, 1]
            &[2, 2],
        )
        .unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let feature_range = RangeConstraint::range(0.0f32, 1.0f32);
        let validator =
            DataValidator::new(ValidationConfig::default()).with_feature_range(feature_range);

        let result = validator.validate(&dataset).unwrap();
        assert!(!result.is_valid);
        assert!(result.has_errors());
        assert!(!result.range_errors.is_empty());
    }

    #[test]
    fn test_duplicate_detection() {
        let features = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0], // First two samples are duplicates
            &[3, 2],
        )
        .unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0, 2.0], &[3]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let config = ValidationConfig {
            check_schema: false,
            check_ranges: false,
            check_duplicates: true,
            check_outliers: false,
            outlier_threshold: 3.0,
        };

        let validator = DataValidator::new(config);
        let result = validator.validate(&dataset).unwrap();

        assert!(!result.is_valid);
        assert!(result.has_errors());
        assert!(!result.duplicate_indices.is_empty());
    }

    #[test]
    fn test_outlier_detection() {
        let features = Tensor::<f32>::from_vec(
            vec![1.0, 1.0, 1.1, 1.0, 1.2, 1.0, 1.0, 1.0, 100.0, 1.0], // 100.0 is an outlier with more stable baseline
            &[5, 2],
        )
        .unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0], &[5]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let config = ValidationConfig {
            check_schema: false,
            check_ranges: false,
            check_duplicates: false,
            check_outliers: true,
            outlier_threshold: 1.0, // Very low threshold to catch the outlier
        };

        let validator = DataValidator::new(config);
        let result = validator.validate(&dataset).unwrap();

        assert!(!result.is_valid);
        assert!(result.has_errors());
        assert!(!result.outlier_indices.is_empty());
    }

    #[test]
    fn test_dataset_validation_ext() {
        let features = Tensor::<f32>::from_vec(vec![0.5, 0.8, 0.3, 0.7], &[2, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let is_valid = dataset.is_valid().unwrap();
        assert!(is_valid);

        let config = ValidationConfig::default();
        let result = dataset.validate_with_config(config).unwrap();
        assert!(result.is_valid);
    }
}
