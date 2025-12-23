//! Tensor analysis and visualization tools
//!
//! This module provides comprehensive tensor analysis capabilities including
//! statistical analysis, distribution visualization, and property inspection.

// use scirs2_core::num_traits::ToPrimitive; // Unused for now
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

/// Tensor analysis and visualization tools
#[pyclass]
pub struct PyTensorAnalyzer {
    analyzer: TensorAnalyzer,
}

impl Default for PyTensorAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyTensorAnalyzer {
    #[new]
    pub fn new() -> Self {
        PyTensorAnalyzer {
            analyzer: TensorAnalyzer::new(),
        }
    }

    /// Analyze tensor properties and generate visualization data
    pub fn analyze_tensor(
        &self,
        py: Python,
        tensor_data: &Bound<'_, PyList>,
    ) -> PyResult<PyObject> {
        // Convert Python list to tensor data
        let tensor_values: Vec<f32> = tensor_data
            .iter()
            .map(|item| item.extract::<f32>())
            .collect::<Result<Vec<_>, _>>()?;

        let analysis = self
            .analyzer
            .analyze_values(&tensor_values)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        let py_dict = PyDict::new(py);
        py_dict.set_item("mean", analysis.mean)?;
        py_dict.set_item("std", analysis.std)?;
        py_dict.set_item("min", analysis.min)?;
        py_dict.set_item("max", analysis.max)?;
        py_dict.set_item("histogram", PyList::new(py, analysis.histogram)?)?;
        py_dict.set_item("percentiles", PyDict::new(py))?;

        // Add percentiles
        let percentiles_item = py_dict.get_item("percentiles")?.unwrap();
        let percentiles_dict = percentiles_item.downcast::<PyDict>()?;
        for (percentile, value) in analysis.percentiles.iter() {
            percentiles_dict.set_item(percentile.to_string(), *value)?;
        }

        // Add correlation analysis if multiple tensors provided
        if !analysis.correlations.is_empty() {
            let correlations_dict = PyDict::new(py);
            for ((name1, name2), correlation) in analysis.correlations.iter() {
                let key = format!("{}-{}", name1, name2);
                correlations_dict.set_item(key, *correlation)?;
            }
            py_dict.set_item("correlations", correlations_dict)?;
        }

        // Add similarity metrics
        if !analysis.similarities.is_empty() {
            let similarities_dict = PyDict::new(py);
            for ((name1, name2), similarity) in analysis.similarities.iter() {
                let key = format!("{}-{}", name1, name2);
                similarities_dict.set_item(key, *similarity)?;
            }
            py_dict.set_item("similarities", similarities_dict)?;
        }

        py_dict.set_item("outliers", PyList::new(py, analysis.outliers)?)?;
        py_dict.set_item("skewness", analysis.skewness)?;
        py_dict.set_item("kurtosis", analysis.kurtosis)?;
        py_dict.set_item("entropy", analysis.entropy)?;

        Ok(py_dict.into())
    }

    /// Generate histogram data for plotting
    pub fn generate_histogram(
        &self,
        py: Python,
        tensor_data: &Bound<'_, PyList>,
        bins: Option<usize>,
    ) -> PyResult<PyObject> {
        let tensor_values: Vec<f32> = tensor_data
            .iter()
            .map(|item| item.extract::<f32>())
            .collect::<Result<Vec<_>, _>>()?;

        let num_bins = bins.unwrap_or(50);
        let histogram = self
            .analyzer
            .generate_histogram(&tensor_values, num_bins)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        let py_dict = PyDict::new(py);
        py_dict.set_item("bins", PyList::new(py, histogram.bins)?)?;
        py_dict.set_item("counts", PyList::new(py, histogram.counts)?)?;
        py_dict.set_item("bin_edges", PyList::new(py, histogram.bin_edges)?)?;

        Ok(py_dict.into())
    }

    /// Analyze multiple tensors for comparison
    pub fn compare_tensors(
        &self,
        py: Python,
        tensor_dict: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        let mut tensor_data: HashMap<String, Vec<f32>> = HashMap::new();

        // Extract tensor data from Python dictionary
        for (key, value) in tensor_dict.iter() {
            let name: String = key.extract()?;
            let data_list: &Bound<'_, PyList> = value.downcast()?;
            let values: Vec<f32> = data_list
                .iter()
                .map(|item| item.extract::<f32>())
                .collect::<Result<Vec<_>, _>>()?;
            tensor_data.insert(name, values);
        }

        let comparison = self
            .analyzer
            .compare_multiple_tensors(&tensor_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        let py_dict = PyDict::new(py);

        // Individual statistics for each tensor
        let stats_dict = PyDict::new(py);
        for (name, stats) in comparison.individual_stats.iter() {
            let tensor_stats = PyDict::new(py);
            tensor_stats.set_item("mean", stats.mean)?;
            tensor_stats.set_item("std", stats.std)?;
            tensor_stats.set_item("min", stats.min)?;
            tensor_stats.set_item("max", stats.max)?;
            stats_dict.set_item(name, tensor_stats)?;
        }
        py_dict.set_item("individual_stats", stats_dict)?;

        // Cross-tensor correlations
        let correlations_dict = PyDict::new(py);
        for ((name1, name2), correlation) in comparison.correlations.iter() {
            let key = format!("{}-{}", name1, name2);
            correlations_dict.set_item(key, *correlation)?;
        }
        py_dict.set_item("correlations", correlations_dict)?;

        // Similarity metrics
        let similarities_dict = PyDict::new(py);
        for ((name1, name2), similarity) in comparison.similarities.iter() {
            let key = format!("{}-{}", name1, name2);
            similarities_dict.set_item(key, *similarity)?;
        }
        py_dict.set_item("similarities", similarities_dict)?;

        py_dict.set_item(
            "recommendations",
            PyList::new(py, comparison.recommendations)?,
        )?;

        Ok(py_dict.into())
    }
}

/// Internal tensor analyzer implementation
pub struct TensorAnalyzer {
    cache: std::sync::RwLock<HashMap<String, TensorAnalysisResult>>,
}

impl TensorAnalyzer {
    pub fn new() -> Self {
        TensorAnalyzer {
            cache: std::sync::RwLock::new(HashMap::new()),
        }
    }

    pub fn analyze_values(&self, values: &[f32]) -> Result<TensorAnalysisResult, String> {
        if values.is_empty() {
            return Err("Cannot analyze empty tensor".to_string());
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std = variance.sqrt();

        let min = sorted_values[0];
        let max = sorted_values[sorted_values.len() - 1];

        // Calculate percentiles
        let mut percentiles = HashMap::new();
        for p in [5, 25, 50, 75, 95].iter() {
            let index = (*p as f32 / 100.0 * (sorted_values.len() - 1) as f32) as usize;
            percentiles.insert(*p, sorted_values[index]);
        }

        // Generate histogram
        let histogram = self.create_histogram(values, 50);

        // Calculate skewness and kurtosis
        let skewness = self.calculate_skewness(values, mean, std);
        let kurtosis = self.calculate_kurtosis(values, mean, std);

        // Calculate entropy
        let entropy = self.calculate_entropy(values);

        // Detect outliers using IQR method
        let q1 = percentiles[&25];
        let q3 = percentiles[&75];
        let iqr = q3 - q1;
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;
        let outliers: Vec<f32> = values
            .iter()
            .filter(|&&x| x < lower_bound || x > upper_bound)
            .copied()
            .collect();

        Ok(TensorAnalysisResult {
            mean,
            std,
            min,
            max,
            histogram,
            percentiles,
            correlations: HashMap::new(),
            similarities: HashMap::new(),
            outliers,
            skewness,
            kurtosis,
            entropy,
        })
    }

    pub fn generate_histogram(
        &self,
        values: &[f32],
        num_bins: usize,
    ) -> Result<HistogramData, String> {
        if values.is_empty() {
            return Err("Cannot generate histogram for empty data".to_string());
        }

        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let bin_width = (max_val - min_val) / num_bins as f32;

        let mut counts = vec![0usize; num_bins];
        let mut bin_edges = Vec::with_capacity(num_bins + 1);

        // Create bin edges
        for i in 0..=num_bins {
            bin_edges.push(min_val + i as f32 * bin_width);
        }

        // Count values in each bin
        for &value in values {
            let bin_index = if value == max_val {
                num_bins - 1 // Put max value in last bin
            } else {
                ((value - min_val) / bin_width) as usize
            };
            if bin_index < num_bins {
                counts[bin_index] += 1;
            }
        }

        // Create bin centers
        let bins: Vec<f32> = (0..num_bins)
            .map(|i| min_val + (i as f32 + 0.5) * bin_width)
            .collect();

        Ok(HistogramData {
            bins,
            counts,
            bin_edges,
        })
    }

    pub fn compare_multiple_tensors(
        &self,
        tensor_data: &HashMap<String, Vec<f32>>,
    ) -> Result<TensorComparisonResult, String> {
        let mut individual_stats = HashMap::new();
        let mut correlations = HashMap::new();
        let mut similarities = HashMap::new();

        // Analyze each tensor individually
        for (name, values) in tensor_data {
            let analysis = self.analyze_values(values)?;
            individual_stats.insert(
                name.clone(),
                BasicStats {
                    mean: analysis.mean,
                    std: analysis.std,
                    min: analysis.min,
                    max: analysis.max,
                },
            );
        }

        // Calculate pairwise correlations and similarities
        let names: Vec<_> = tensor_data.keys().collect();
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                let name1 = names[i];
                let name2 = names[j];
                let values1 = &tensor_data[name1];
                let values2 = &tensor_data[name2];

                if values1.len() == values2.len() {
                    let correlation = self.calculate_correlation(values1, values2);
                    let similarity = self.calculate_cosine_similarity(values1, values2);

                    correlations.insert((name1.clone(), name2.clone()), correlation);
                    similarities.insert((name1.clone(), name2.clone()), similarity);
                }
            }
        }

        // Generate recommendations
        let recommendations =
            self.generate_comparison_recommendations(&individual_stats, &correlations);

        Ok(TensorComparisonResult {
            individual_stats,
            correlations,
            similarities,
            recommendations,
        })
    }

    fn create_histogram(&self, values: &[f32], num_bins: usize) -> Vec<usize> {
        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let bin_width = (max_val - min_val) / num_bins as f32;

        let mut histogram = vec![0; num_bins];
        for &value in values {
            let bin_index = if value == max_val {
                num_bins - 1
            } else {
                ((value - min_val) / bin_width) as usize
            };
            if bin_index < num_bins {
                histogram[bin_index] += 1;
            }
        }
        histogram
    }

    fn calculate_skewness(&self, values: &[f32], mean: f32, std: f32) -> f32 {
        if std == 0.0 {
            return 0.0;
        }
        let n = values.len() as f32;
        let skewness = values
            .iter()
            .map(|x| ((x - mean) / std).powi(3))
            .sum::<f32>()
            / n;
        skewness
    }

    fn calculate_kurtosis(&self, values: &[f32], mean: f32, std: f32) -> f32 {
        if std == 0.0 {
            return 0.0;
        }
        let n = values.len() as f32;
        let kurtosis = values
            .iter()
            .map(|x| ((x - mean) / std).powi(4))
            .sum::<f32>()
            / n
            - 3.0; // Excess kurtosis
        kurtosis
    }

    fn calculate_entropy(&self, values: &[f32]) -> f32 {
        // Simple entropy calculation based on histogram
        let histogram = self.create_histogram(values, 50);
        let total = values.len() as f32;

        histogram
            .iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f32 / total;
                -p * p.log2()
            })
            .sum()
    }

    fn calculate_correlation(&self, values1: &[f32], values2: &[f32]) -> f32 {
        let mean1 = values1.iter().sum::<f32>() / values1.len() as f32;
        let mean2 = values2.iter().sum::<f32>() / values2.len() as f32;

        let numerator: f32 = values1
            .iter()
            .zip(values2.iter())
            .map(|(x1, x2)| (x1 - mean1) * (x2 - mean2))
            .sum();

        let denominator1: f32 = values1.iter().map(|x| (x - mean1).powi(2)).sum();
        let denominator2: f32 = values2.iter().map(|x| (x - mean2).powi(2)).sum();

        if denominator1 == 0.0 || denominator2 == 0.0 {
            0.0
        } else {
            numerator / (denominator1.sqrt() * denominator2.sqrt())
        }
    }

    fn calculate_cosine_similarity(&self, values1: &[f32], values2: &[f32]) -> f32 {
        let dot_product: f32 = values1
            .iter()
            .zip(values2.iter())
            .map(|(x1, x2)| x1 * x2)
            .sum();
        let norm1: f32 = values1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = values2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }

    fn generate_comparison_recommendations(
        &self,
        stats: &HashMap<String, BasicStats>,
        correlations: &HashMap<(String, String), f32>,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check for high variance tensors
        for (name, stat) in stats {
            if stat.std > stat.mean.abs() * 2.0 {
                recommendations.push(format!(
                    "Tensor '{}' has high variance (std: {:.3}, mean: {:.3}) - consider normalization",
                    name, stat.std, stat.mean
                ));
            }
        }

        // Check for highly correlated tensors
        for ((name1, name2), correlation) in correlations {
            if correlation.abs() > 0.9 {
                recommendations.push(format!(
                    "Tensors '{}' and '{}' are highly correlated ({:.3}) - consider removing redundancy",
                    name1, name2, correlation
                ));
            }
        }

        if recommendations.is_empty() {
            recommendations
                .push("Tensor analysis looks good - no major issues detected".to_string());
        }

        recommendations
    }
}

impl Default for TensorAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// Data structures for analysis results
pub struct TensorAnalysisResult {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub histogram: Vec<usize>,
    pub percentiles: HashMap<u8, f32>,
    pub correlations: HashMap<(String, String), f32>,
    pub similarities: HashMap<(String, String), f32>,
    pub outliers: Vec<f32>,
    pub skewness: f32,
    pub kurtosis: f32,
    pub entropy: f32,
}

pub struct HistogramData {
    pub bins: Vec<f32>,
    pub counts: Vec<usize>,
    pub bin_edges: Vec<f32>,
}

pub struct BasicStats {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
}

pub struct TensorComparisonResult {
    pub individual_stats: HashMap<String, BasicStats>,
    pub correlations: HashMap<(String, String), f32>,
    pub similarities: HashMap<(String, String), f32>,
    pub recommendations: Vec<String>,
}
