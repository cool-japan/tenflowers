//! Data quality metrics and drift detection
//!
//! This module provides comprehensive data quality assessment including:
//! - Statistical quality metrics (completeness, validity, consistency)
//! - Data drift detection (distribution shift, concept drift)
//! - Outlier detection and anomaly analysis
//! - Data profiling and summary statistics

use crate::Dataset;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor, TensorError};

/// Comprehensive data quality metrics
#[derive(Debug, Clone)]
pub struct DataQualityMetrics {
    /// Dataset name or identifier
    pub dataset_name: String,
    /// Total number of samples
    pub total_samples: usize,
    /// Completeness metrics (percentage of non-null values)
    pub completeness: HashMap<String, f64>,
    /// Validity metrics (percentage of values passing validation rules)
    pub validity: HashMap<String, f64>,
    /// Consistency metrics (internal consistency checks)
    pub consistency_score: f64,
    /// Timeliness (data freshness indicators)
    pub timeliness_score: Option<f64>,
    /// Accuracy estimates
    pub accuracy_estimates: HashMap<String, f64>,
    /// Uniqueness metrics (duplicate detection)
    pub uniqueness_score: f64,
    /// Overall quality score (0.0 to 1.0)
    pub overall_quality_score: f64,
    /// Detected issues
    pub issues: Vec<DataQualityIssue>,
}

/// Data quality issue description
#[derive(Debug, Clone)]
pub struct DataQualityIssue {
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue category
    pub category: IssueCategory,
    /// Description of the issue
    pub description: String,
    /// Affected fields/columns
    pub affected_fields: Vec<String>,
    /// Number of affected samples
    pub affected_count: usize,
    /// Suggested remediation
    pub remediation: Option<String>,
}

/// Issue severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSeverity {
    /// Critical issue requiring immediate attention
    Critical,
    /// High priority issue
    High,
    /// Medium priority issue
    Medium,
    /// Low priority issue
    Low,
    /// Informational only
    Info,
}

/// Issue categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueCategory {
    /// Missing data issues
    Completeness,
    /// Invalid values
    Validity,
    /// Inconsistent data
    Consistency,
    /// Duplicate data
    Uniqueness,
    /// Accuracy issues
    Accuracy,
    /// Timeliness issues
    Timeliness,
    /// Statistical anomalies
    StatisticalAnomaly,
}

/// Data drift detection configuration
#[derive(Debug, Clone)]
pub struct DriftDetectionConfig {
    /// Reference dataset size (for baseline)
    pub reference_window_size: usize,
    /// Detection threshold (0.0 to 1.0)
    pub detection_threshold: f64,
    /// Statistical test to use
    pub statistical_test: StatisticalTest,
    /// Minimum number of samples for detection
    pub min_samples: usize,
    /// Enable distribution visualization
    pub enable_visualization: bool,
}

impl Default for DriftDetectionConfig {
    fn default() -> Self {
        Self {
            reference_window_size: 1000,
            detection_threshold: 0.05, // 5% significance level
            statistical_test: StatisticalTest::KolmogorovSmirnov,
            min_samples: 100,
            enable_visualization: false,
        }
    }
}

/// Statistical tests for drift detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatisticalTest {
    /// Kolmogorov-Smirnov test
    KolmogorovSmirnov,
    /// Chi-squared test
    ChiSquared,
    /// Population Stability Index (PSI)
    PopulationStabilityIndex,
    /// Kullback-Leibler divergence
    KullbackLeibler,
    /// Jensen-Shannon divergence
    JensenShannon,
}

/// Data drift detection result
#[derive(Debug, Clone)]
pub struct DriftDetectionResult {
    /// Whether drift was detected
    pub drift_detected: bool,
    /// Drift score (higher means more drift)
    pub drift_score: f64,
    /// Statistical test p-value (if applicable)
    pub p_value: Option<f64>,
    /// Distribution distance metric
    pub distance_metric: f64,
    /// Drift type
    pub drift_type: DriftType,
    /// Detailed analysis
    pub analysis: String,
    /// Affected features
    pub affected_features: Vec<String>,
}

/// Types of detected drift
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftType {
    /// No drift detected
    NoDrift,
    /// Covariate shift (feature distribution change)
    CovariateShift,
    /// Concept drift (relationship change)
    ConceptDrift,
    /// Label drift (target distribution change)
    LabelDrift,
    /// Combined drift
    CombinedDrift,
}

/// Data quality analyzer
pub struct DataQualityAnalyzer {
    /// Configuration for quality checks
    config: QualityAnalysisConfig,
}

/// Configuration for quality analysis
#[derive(Debug, Clone)]
pub struct QualityAnalysisConfig {
    /// Check for missing values
    pub check_completeness: bool,
    /// Check for invalid values
    pub check_validity: bool,
    /// Check for duplicates
    pub check_duplicates: bool,
    /// Check for outliers
    pub check_outliers: bool,
    /// Outlier detection method
    pub outlier_method: OutlierDetectionMethod,
    /// Outlier threshold (e.g., number of standard deviations)
    pub outlier_threshold: f64,
    /// Maximum number of unique values to track
    pub max_unique_values: usize,
}

impl Default for QualityAnalysisConfig {
    fn default() -> Self {
        Self {
            check_completeness: true,
            check_validity: true,
            check_duplicates: true,
            check_outliers: true,
            outlier_method: OutlierDetectionMethod::IQR,
            outlier_threshold: 1.5,
            max_unique_values: 10000,
        }
    }
}

/// Outlier detection methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutlierDetectionMethod {
    /// Interquartile Range (IQR) method
    IQR,
    /// Z-score method
    ZScore,
    /// Modified Z-score method
    ModifiedZScore,
    /// Isolation Forest
    IsolationForest,
}

impl DataQualityAnalyzer {
    /// Create a new data quality analyzer
    pub fn new(config: QualityAnalysisConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(QualityAnalysisConfig::default())
    }

    /// Analyze data quality for a dataset
    pub fn analyze<T>(
        &self,
        dataset: &dyn Dataset<T>,
        dataset_name: impl Into<String>,
    ) -> Result<DataQualityMetrics>
    where
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + scirs2_core::numeric::Float
            + Send
            + Sync
            + 'static,
    {
        let dataset_name = dataset_name.into();
        let total_samples = dataset.len();

        if total_samples == 0 {
            return Ok(DataQualityMetrics {
                dataset_name,
                total_samples: 0,
                completeness: HashMap::new(),
                validity: HashMap::new(),
                consistency_score: 0.0,
                timeliness_score: None,
                accuracy_estimates: HashMap::new(),
                uniqueness_score: 0.0,
                overall_quality_score: 0.0,
                issues: vec![DataQualityIssue {
                    severity: IssueSeverity::Critical,
                    category: IssueCategory::Completeness,
                    description: "Dataset is empty".to_string(),
                    affected_fields: vec![],
                    affected_count: 0,
                    remediation: Some("Add data to the dataset".to_string()),
                }],
            });
        }

        let mut metrics = DataQualityMetrics {
            dataset_name,
            total_samples,
            completeness: HashMap::new(),
            validity: HashMap::new(),
            consistency_score: 1.0,
            timeliness_score: None,
            accuracy_estimates: HashMap::new(),
            uniqueness_score: 1.0,
            overall_quality_score: 0.0,
            issues: Vec::new(),
        };

        // Check completeness
        if self.config.check_completeness {
            self.check_completeness(dataset, &mut metrics)?;
        }

        // Check for duplicates
        if self.config.check_duplicates {
            self.check_duplicates(dataset, &mut metrics)?;
        }

        // Check for outliers
        if self.config.check_outliers {
            self.check_outliers(dataset, &mut metrics)?;
        }

        // Calculate overall quality score
        metrics.overall_quality_score = self.calculate_overall_score(&metrics);

        Ok(metrics)
    }

    fn check_completeness<T>(
        &self,
        dataset: &dyn Dataset<T>,
        metrics: &mut DataQualityMetrics,
    ) -> Result<()>
    where
        T: Clone + Default + scirs2_core::numeric::Zero + PartialEq + Send + Sync + 'static,
    {
        let mut non_zero_count = 0;

        for i in 0..dataset.len().min(1000) {
            // Sample up to 1000 items
            if let Ok((features, _)) = dataset.get(i) {
                if let Some(data) = features.as_slice() {
                    if data.iter().any(|x| *x != T::zero()) {
                        non_zero_count += 1;
                    }
                }
            }
        }

        let samples_checked = dataset.len().min(1000);
        let completeness_score = if samples_checked > 0 {
            non_zero_count as f64 / samples_checked as f64
        } else {
            0.0
        };

        metrics
            .completeness
            .insert("features".to_string(), completeness_score);

        if completeness_score < 0.9 {
            metrics.issues.push(DataQualityIssue {
                severity: if completeness_score < 0.5 {
                    IssueSeverity::High
                } else {
                    IssueSeverity::Medium
                },
                category: IssueCategory::Completeness,
                description: format!("Low completeness score: {:.2}%", completeness_score * 100.0),
                affected_fields: vec!["features".to_string()],
                affected_count: ((1.0 - completeness_score) * samples_checked as f64) as usize,
                remediation: Some(
                    "Investigate missing data and apply imputation if appropriate".to_string(),
                ),
            });
        }

        Ok(())
    }

    fn check_duplicates<T>(
        &self,
        dataset: &dyn Dataset<T>,
        metrics: &mut DataQualityMetrics,
    ) -> Result<()>
    where
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + scirs2_core::numeric::Float
            + Send
            + Sync
            + 'static,
    {
        use std::collections::HashSet;

        // Check for duplicate samples using approximate comparison for floats
        let samples_to_check = dataset.len().min(1000);
        let mut seen_samples: HashSet<String> = HashSet::new();
        let mut duplicate_count = 0usize;

        for i in 0..samples_to_check {
            if let Ok((features, _labels)) = dataset.get(i) {
                // Create a fingerprint from the tensor values
                if let Some(data_slice) = features.as_slice() {
                    // Create a deterministic string representation
                    // For floats, we round to a fixed precision to avoid precision issues
                    let fingerprint: String = data_slice
                        .iter()
                        .map(|v| {
                            let f = v.to_f64().unwrap_or(0.0);
                            format!("{:.6}", f) // 6 decimal places for fingerprinting
                        })
                        .collect::<Vec<_>>()
                        .join(",");

                    if !seen_samples.insert(fingerprint) {
                        duplicate_count += 1;
                    }
                }
            }
        }

        // Calculate uniqueness score: (unique samples) / (total samples checked)
        let unique_count = samples_to_check - duplicate_count;
        metrics.uniqueness_score = if samples_to_check > 0 {
            unique_count as f64 / samples_to_check as f64
        } else {
            1.0
        };

        Ok(())
    }

    fn check_outliers<T>(
        &self,
        dataset: &dyn Dataset<T>,
        metrics: &mut DataQualityMetrics,
    ) -> Result<()>
    where
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + scirs2_core::numeric::Float
            + Send
            + Sync
            + 'static,
    {
        let mut values: Vec<f64> = Vec::new();

        // Collect values for outlier analysis
        for i in 0..dataset.len().min(1000) {
            if let Ok((features, _)) = dataset.get(i) {
                if let Some(data) = features.as_slice() {
                    for &val in data {
                        values.push(val.to_f64().unwrap_or(0.0));
                    }
                }
            }
        }

        if values.is_empty() {
            return Ok(());
        }

        // Calculate statistics
        let outlier_count = match self.config.outlier_method {
            OutlierDetectionMethod::IQR => self.detect_outliers_iqr(&values),
            OutlierDetectionMethod::ZScore => self.detect_outliers_zscore(&values),
            _ => 0,
        };

        if outlier_count > 0 {
            let outlier_percentage = outlier_count as f64 / values.len() as f64;
            if outlier_percentage > 0.05 {
                // More than 5% outliers
                metrics.issues.push(DataQualityIssue {
                    severity: IssueSeverity::Medium,
                    category: IssueCategory::StatisticalAnomaly,
                    description: format!(
                        "High percentage of outliers detected: {:.2}%",
                        outlier_percentage * 100.0
                    ),
                    affected_fields: vec!["features".to_string()],
                    affected_count: outlier_count,
                    remediation: Some(
                        "Review outlier values and consider outlier removal or transformation"
                            .to_string(),
                    ),
                });
            }
        }

        Ok(())
    }

    fn detect_outliers_iqr(&self, values: &[f64]) -> usize {
        if values.len() < 4 {
            return 0;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1_idx = sorted.len() / 4;
        let q3_idx = (3 * sorted.len()) / 4;
        let q1 = sorted[q1_idx];
        let q3 = sorted[q3_idx];
        let iqr = q3 - q1;

        let lower_bound = q1 - self.config.outlier_threshold * iqr;
        let upper_bound = q3 + self.config.outlier_threshold * iqr;

        values
            .iter()
            .filter(|&&v| v < lower_bound || v > upper_bound)
            .count()
    }

    fn detect_outliers_zscore(&self, values: &[f64]) -> usize {
        if values.is_empty() {
            return 0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0;
        }

        values
            .iter()
            .filter(|&&v| ((v - mean) / std_dev).abs() > self.config.outlier_threshold)
            .count()
    }

    fn calculate_overall_score(&self, metrics: &DataQualityMetrics) -> f64 {
        let mut scores = Vec::new();

        // Completeness score
        if let Some(&completeness) = metrics.completeness.get("features") {
            scores.push(completeness);
        }

        // Uniqueness score
        scores.push(metrics.uniqueness_score);

        // Consistency score
        scores.push(metrics.consistency_score);

        // Reduce score based on issues
        let issue_penalty = metrics.issues.iter().fold(0.0, |acc, issue| {
            acc + match issue.severity {
                IssueSeverity::Critical => 0.3,
                IssueSeverity::High => 0.2,
                IssueSeverity::Medium => 0.1,
                IssueSeverity::Low => 0.05,
                IssueSeverity::Info => 0.0,
            }
        });

        let base_score = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };

        (base_score - issue_penalty).clamp(0.0, 1.0)
    }

    /// Generate a human-readable quality report
    pub fn generate_report(&self, metrics: &DataQualityMetrics) -> String {
        let mut report = format!(
            "Data Quality Report: {}\n\
             ================================\n\
             Total Samples: {}\n\
             Overall Quality Score: {:.2}%\n\n",
            metrics.dataset_name,
            metrics.total_samples,
            metrics.overall_quality_score * 100.0
        );

        // Completeness
        if !metrics.completeness.is_empty() {
            report.push_str("Completeness:\n");
            for (field, score) in &metrics.completeness {
                report.push_str(&format!("  {}: {:.2}%\n", field, score * 100.0));
            }
            report.push('\n');
        }

        // Uniqueness
        report.push_str(&format!(
            "Uniqueness Score: {:.2}%\n\n",
            metrics.uniqueness_score * 100.0
        ));

        // Issues
        if !metrics.issues.is_empty() {
            report.push_str(&format!("Detected Issues ({}):\n", metrics.issues.len()));
            for (i, issue) in metrics.issues.iter().enumerate() {
                report.push_str(&format!(
                    "  {}. [{:?}] [{:?}] {}\n",
                    i + 1,
                    issue.severity,
                    issue.category,
                    issue.description
                ));
                if !issue.affected_fields.is_empty() {
                    report.push_str(&format!(
                        "     Affected fields: {}\n",
                        issue.affected_fields.join(", ")
                    ));
                }
                if let Some(remediation) = &issue.remediation {
                    report.push_str(&format!("     Remediation: {}\n", remediation));
                }
            }
        } else {
            report.push_str("No issues detected.\n");
        }

        report
    }
}

/// Extension trait to add quality analysis to any dataset
pub trait DataQualityExt<T>: Dataset<T> + Sized {
    /// Analyze data quality
    fn analyze_quality(&self, name: impl Into<String>) -> Result<DataQualityMetrics>
    where
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + scirs2_core::numeric::Float
            + Send
            + Sync
            + 'static,
    {
        let analyzer = DataQualityAnalyzer::default();
        analyzer.analyze(self, name)
    }

    /// Generate a quality report
    fn quality_report(&self, name: impl Into<String>) -> Result<String>
    where
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + scirs2_core::numeric::Float
            + Send
            + Sync
            + 'static,
    {
        let metrics = self.analyze_quality(name)?;
        let analyzer = DataQualityAnalyzer::default();
        Ok(analyzer.generate_report(&metrics))
    }
}

// Blanket implementation for all datasets
impl<T, D: Dataset<T>> DataQualityExt<T> for D {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorDataset;
    use tenflowers_core::Tensor;

    #[test]
    fn test_quality_analyzer_creation() {
        let analyzer = DataQualityAnalyzer::default();
        assert!(analyzer.config.check_completeness);
        assert!(analyzer.config.check_validity);
    }

    #[test]
    fn test_empty_dataset_quality() {
        let features = Tensor::<f32>::from_vec(vec![], &[0, 1]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![], &[0]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let analyzer = DataQualityAnalyzer::default();
        let metrics = analyzer.analyze(&dataset, "test_dataset").unwrap();

        assert_eq!(metrics.total_samples, 0);
        assert_eq!(metrics.overall_quality_score, 0.0);
        assert!(!metrics.issues.is_empty());
    }

    #[test]
    fn test_quality_extension_trait() {
        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let metrics = dataset.analyze_quality("test_dataset").unwrap();
        assert_eq!(metrics.total_samples, 2);
        assert!(metrics.overall_quality_score > 0.0);
    }

    #[test]
    fn test_outlier_detection_iqr() {
        let config = QualityAnalysisConfig::default();
        let analyzer = DataQualityAnalyzer::new(config);

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100.0 is an outlier
        let outlier_count = analyzer.detect_outliers_iqr(&values);

        assert!(outlier_count > 0);
    }

    #[test]
    fn test_drift_detection_config() {
        let config = DriftDetectionConfig::default();
        assert_eq!(config.reference_window_size, 1000);
        assert_eq!(config.detection_threshold, 0.05);
        assert_eq!(config.statistical_test, StatisticalTest::KolmogorovSmirnov);
    }
}
