//! CSV format factory and reader implementation
//!
//! This module implements the FormatFactory and FormatReader traits for CSV files,
//! enabling automatic format detection and unified data loading.

use crate::error_taxonomy::helpers as error_helpers;
use crate::formats::unified_reader::{
    DataType, DetectionMethod, FieldInfo, FormatDetection, FormatFactory, FormatMetadata,
    FormatReader, FormatSample,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use tenflowers_core::{Result, Tensor, TensorError};

/// CSV format factory for automatic detection and reader creation
pub struct CsvFormatFactory;

impl FormatFactory for CsvFormatFactory {
    fn format_name(&self) -> &str {
        "CSV"
    }

    fn extensions(&self) -> Vec<&str> {
        vec!["csv", "tsv", "txt"]
    }

    fn can_read(&self, path: &Path) -> Result<FormatDetection> {
        // Check file extension
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_lowercase());

        let mut confidence = 0.0;
        let mut method = DetectionMethod::Extension;

        match extension.as_deref() {
            Some("csv") => {
                confidence = 0.95;
                method = DetectionMethod::Extension;
            }
            Some("tsv") => {
                confidence = 0.9;
                method = DetectionMethod::Extension;
            }
            Some("txt") => {
                // For .txt files, check content
                if let Ok(is_csv) = Self::check_csv_content(path) {
                    if is_csv {
                        confidence = 0.7;
                        method = DetectionMethod::ContentAnalysis;
                    }
                }
            }
            _ => {
                // Try content analysis
                if let Ok(is_csv) = Self::check_csv_content(path) {
                    if is_csv {
                        confidence = 0.6;
                        method = DetectionMethod::ContentAnalysis;
                    }
                }
            }
        }

        Ok(FormatDetection {
            format_name: self.format_name().to_string(),
            confidence,
            method,
        })
    }

    fn create_reader(&self, path: &Path) -> Result<Box<dyn FormatReader>> {
        Ok(Box::new(CsvFormatReader::new(path)?))
    }
}

impl CsvFormatFactory {
    /// Check if file content looks like CSV
    fn check_csv_content(path: &Path) -> Result<bool> {
        let file = File::open(path).map_err(|_| {
            error_helpers::file_not_found("CsvFormatFactory::check_csv_content", path)
        })?;

        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Check first few lines
        let mut csv_like = false;
        for _ in 0..5 {
            if let Some(Ok(line)) = lines.next() {
                // CSV typically has commas, tabs, or semicolons
                let separators = [',', '\t', ';'];
                let has_separator = separators.iter().any(|&sep| line.contains(sep));

                if has_separator {
                    csv_like = true;
                    break;
                }
            } else {
                break;
            }
        }

        Ok(csv_like)
    }
}

/// CSV format reader implementation
pub struct CsvFormatReader {
    path: PathBuf,
    delimiter: u8,
    has_header: bool,
    metadata: FormatMetadata,
    samples: Vec<Vec<String>>,
    header: Vec<String>,
}

impl CsvFormatReader {
    /// Create a new CSV format reader
    pub fn new(path: &Path) -> Result<Self> {
        let delimiter = Self::detect_delimiter(path)?;
        let (has_header, header, samples) = Self::load_csv_data(path, delimiter)?;

        // Infer field types from data
        let fields = Self::infer_fields(&header, &samples);

        let metadata = FormatMetadata {
            format_name: "CSV".to_string(),
            version: None,
            num_samples: samples.len(),
            fields,
            metadata: HashMap::new(),
            supports_random_access: true,
            supports_streaming: true,
        };

        Ok(Self {
            path: path.to_path_buf(),
            delimiter,
            has_header,
            metadata,
            samples,
            header,
        })
    }

    /// Detect the delimiter used in the CSV file
    fn detect_delimiter(path: &Path) -> Result<u8> {
        let file = File::open(path).map_err(|_| {
            error_helpers::file_not_found("CsvFormatReader::detect_delimiter", path)
        })?;

        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        if let Some(Ok(first_line)) = lines.next() {
            // Count different delimiters
            let comma_count = first_line.matches(',').count();
            let tab_count = first_line.matches('\t').count();
            let semicolon_count = first_line.matches(';').count();

            // Return the most common delimiter
            if comma_count >= tab_count && comma_count >= semicolon_count {
                Ok(b',')
            } else if tab_count >= semicolon_count {
                Ok(b'\t')
            } else {
                Ok(b';')
            }
        } else {
            Ok(b',') // Default to comma
        }
    }

    /// Load CSV data from file
    fn load_csv_data(path: &Path, delimiter: u8) -> Result<(bool, Vec<String>, Vec<Vec<String>>)> {
        let file = File::open(path)
            .map_err(|_| error_helpers::file_not_found("CsvFormatReader::load_csv_data", path))?;

        let mut reader = csv::ReaderBuilder::new()
            .delimiter(delimiter)
            .from_reader(file);

        let mut has_header = false;
        let mut header = Vec::new();
        let mut samples = Vec::new();

        // Get headers
        if let Ok(headers) = reader.headers() {
            has_header = true;
            header = headers.iter().map(|s| s.to_string()).collect();
        }

        // Read records
        for result in reader.records() {
            let record = result.map_err(|e| {
                error_helpers::data_corruption(
                    "CsvFormatReader::load_csv_data",
                    format!("CSV parse error: {}", e),
                    Some(path.to_path_buf()),
                )
            })?;

            let row: Vec<String> = record.iter().map(|s| s.to_string()).collect();
            samples.push(row);
        }

        // If no header was detected, create default column names
        if !has_header && !samples.is_empty() {
            let num_cols = samples[0].len();
            header = (0..num_cols).map(|i| format!("col_{}", i)).collect();
        }

        Ok((has_header, header, samples))
    }

    /// Infer field types from sample data
    fn infer_fields(header: &[String], samples: &[Vec<String>]) -> Vec<FieldInfo> {
        let mut fields = Vec::new();

        for (i, name) in header.iter().enumerate() {
            let dtype = Self::infer_column_type(samples, i);

            fields.push(FieldInfo {
                name: name.clone(),
                dtype,
                shape: Some(vec![1]),
                nullable: true,
                description: None,
            });
        }

        fields
    }

    /// Infer the data type of a column
    fn infer_column_type(samples: &[Vec<String>], col_index: usize) -> DataType {
        let mut all_int = true;
        let mut all_float = true;
        let mut all_bool = true;

        for row in samples.iter().take(100) {
            // Sample first 100 rows
            if col_index >= row.len() {
                continue;
            }

            let value = &row[col_index];
            let trimmed = value.trim();

            if trimmed.is_empty() {
                continue;
            }

            // Check if it's a boolean
            if !["true", "false", "0", "1"].contains(&trimmed.to_lowercase().as_str()) {
                all_bool = false;
            }

            // Check if it's an integer
            if value.parse::<i64>().is_err() {
                all_int = false;
            }

            // Check if it's a float
            if value.parse::<f64>().is_err() {
                all_float = false;
            }
        }

        if all_bool {
            DataType::Bool
        } else if all_int {
            DataType::Int64
        } else if all_float {
            DataType::Float64
        } else {
            DataType::String
        }
    }

    /// Convert string value to f32
    fn parse_to_f32(value: &str) -> Result<f32> {
        value.trim().parse::<f32>().map_err(|e| {
            TensorError::invalid_argument(format!("Cannot parse '{}' as f32: {}", value, e))
        })
    }
}

impl FormatReader for CsvFormatReader {
    fn metadata(&self) -> Result<FormatMetadata> {
        Ok(self.metadata.clone())
    }

    fn get_sample(&self, index: usize) -> Result<FormatSample> {
        if index >= self.samples.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index,
                self.samples.len()
            )));
        }

        let row = &self.samples[index];

        // Split into features and labels (last column as label)
        let num_cols = row.len();
        if num_cols == 0 {
            return Err(TensorError::invalid_argument("Empty row".to_string()));
        }

        // Features: all columns except last
        let mut feature_data = Vec::new();
        for item in row.iter().take(num_cols - 1) {
            feature_data.push(Self::parse_to_f32(item)?);
        }

        // Label: last column
        let label_data = vec![Self::parse_to_f32(&row[num_cols - 1])?];

        let features = if feature_data.is_empty() {
            Tensor::<f32>::zeros(&[1])
        } else {
            Tensor::from_vec(feature_data, &[num_cols - 1])?
        };

        let labels = Tensor::from_vec(label_data, &[1])?;

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "CSV".to_string());
        metadata.insert("row_index".to_string(), index.to_string());

        Ok(FormatSample {
            features,
            labels,
            source_index: index,
            metadata,
        })
    }

    fn iter(&self) -> Box<dyn Iterator<Item = Result<FormatSample>> + '_> {
        Box::new((0..self.samples.len()).map(move |i| self.get_sample(i)))
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_csv_format_detection() {
        let factory = CsvFormatFactory;

        // Test .csv extension
        let csv_path = Path::new("data.csv");
        let detection = factory
            .can_read(csv_path)
            .expect("test: format detection should succeed");
        assert!(detection.confidence >= 0.9);
        assert_eq!(detection.format_name, "CSV");
    }

    #[test]
    fn test_csv_format_reader() {
        // Create a temporary CSV file
        let mut temp_file = NamedTempFile::new().expect("test: temp file creation should succeed");
        writeln!(temp_file, "feature1,feature2,label").expect("test: writeln should succeed");
        writeln!(temp_file, "1.0,2.0,0").expect("test: writeln should succeed");
        writeln!(temp_file, "3.0,4.0,1").expect("test: writeln should succeed");
        temp_file.flush().expect("test: flush should succeed");

        let reader =
            CsvFormatReader::new(temp_file.path()).expect("test: reader creation should succeed");

        assert_eq!(reader.len(), 2);

        let sample = reader
            .get_sample(0)
            .expect("test: get sample should succeed");
        assert_eq!(sample.source_index, 0);
    }

    #[test]
    fn test_delimiter_detection() {
        // Test comma delimiter
        let mut temp_file = NamedTempFile::new().expect("test: temp file creation should succeed");
        writeln!(temp_file, "a,b,c").expect("test: writeln should succeed");
        temp_file.flush().expect("test: flush should succeed");

        let delimiter = CsvFormatReader::detect_delimiter(temp_file.path())
            .expect("test: delimiter detection should succeed");
        assert_eq!(delimiter, b',');

        // Test tab delimiter
        let mut temp_file = NamedTempFile::new().expect("test: temp file creation should succeed");
        writeln!(temp_file, "a\tb\tc").expect("test: writeln should succeed");
        temp_file.flush().expect("test: flush should succeed");

        let delimiter = CsvFormatReader::detect_delimiter(temp_file.path())
            .expect("test: delimiter detection should succeed");
        assert_eq!(delimiter, b'\t');
    }

    #[test]
    fn test_type_inference() {
        let samples = vec![
            vec!["1".to_string(), "2.5".to_string(), "hello".to_string()],
            vec!["2".to_string(), "3.7".to_string(), "world".to_string()],
        ];

        let dtype0 = CsvFormatReader::infer_column_type(&samples, 0);
        let dtype1 = CsvFormatReader::infer_column_type(&samples, 1);
        let dtype2 = CsvFormatReader::infer_column_type(&samples, 2);

        assert_eq!(dtype0, DataType::Int64);
        assert_eq!(dtype1, DataType::Float64);
        assert_eq!(dtype2, DataType::String);
    }
}
