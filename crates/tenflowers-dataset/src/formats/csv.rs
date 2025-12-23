//! CSV dataset support
//!
//! This module provides dataset implementations for CSV files with various parsing options.

use crate::{
    error_taxonomy::helpers as error_helpers, formats::common::MissingValueStrategy, Dataset,
};
use std::path::Path;
use tenflowers_core::{Result, Tensor, TensorError};

/// CSV dataset implementation
#[derive(Debug)]
pub struct CsvDataset<T> {
    data: Vec<Vec<T>>,
    labels: Vec<T>,
}

impl<T> CsvDataset<T>
where
    T: Clone + Default + std::str::FromStr + Send + Sync + 'static,
{
    pub fn new(data: Vec<Vec<T>>, labels: Vec<T>) -> Self {
        Self { data, labels }
    }

    /// Load a CSV dataset from a file with default settings
    #[cfg(feature = "csv_format")]
    pub fn from_path(path: impl AsRef<Path>, label_column: Option<usize>) -> Result<Self>
    where
        <T as std::str::FromStr>::Err: std::fmt::Display,
    {
        let mut builder = CsvDatasetBuilder::new().from_path(path);
        if let Some(idx) = label_column {
            builder = builder.label_column(idx);
        }
        builder.build()
    }
}

impl<T> Dataset<T> for CsvDataset<T>
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
{
    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.data.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {index} out of bounds for dataset of length {}",
                self.data.len()
            )));
        }

        let features = Tensor::from_vec(self.data[index].clone(), &[self.data[index].len()])?;
        let label = Tensor::from_scalar(self.labels[index].clone());
        Ok((features, label))
    }
}

/// Builder for CSV datasets
pub struct CsvDatasetBuilder {
    missing_value_strategy: MissingValueStrategy,
    delimiter: char,
    has_header: bool,
    path: Option<std::path::PathBuf>,
    label_column_index: Option<usize>,
}

impl Default for CsvDatasetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CsvDatasetBuilder {
    pub fn new() -> Self {
        Self {
            missing_value_strategy: MissingValueStrategy::default(),
            delimiter: ',',
            has_header: true,
            path: None,
            label_column_index: None,
        }
    }

    pub fn missing_value_strategy(mut self, strategy: MissingValueStrategy) -> Self {
        self.missing_value_strategy = strategy;
        self
    }

    pub fn delimiter(mut self, delimiter: char) -> Self {
        self.delimiter = delimiter;
        self
    }

    pub fn has_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }

    pub fn from_path(mut self, path: impl AsRef<Path>) -> Self {
        self.path = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn label_column(mut self, index: usize) -> Self {
        self.label_column_index = Some(index);
        self
    }

    #[cfg(feature = "csv_format")]
    pub fn build<T>(self) -> Result<CsvDataset<T>>
    where
        T: Clone + Default + std::str::FromStr + Send + Sync + 'static,
        <T as std::str::FromStr>::Err: std::fmt::Display,
    {
        use std::fs::File;

        let path = self.path.ok_or_else(|| {
            error_helpers::invalid_configuration("build", "path", "CSV file path must be specified")
        })?;

        let file =
            File::open(&path).map_err(|_| error_helpers::file_not_found("build", path.clone()))?;

        let mut reader = csv::ReaderBuilder::new()
            .delimiter(self.delimiter as u8)
            .has_headers(self.has_header)
            .from_reader(file);

        let mut all_data: Vec<Vec<T>> = Vec::new();
        let mut all_labels: Vec<T> = Vec::new();

        for result in reader.records() {
            let record = result.map_err(|e| {
                error_helpers::data_corruption(
                    "build",
                    format!("CSV parse error: {}", e),
                    Some(path.clone()),
                )
            })?;

            let mut row_data: Vec<T> = Vec::new();
            let mut label: Option<T> = None;

            for (i, field) in record.iter().enumerate() {
                // Handle empty fields as potential missing values
                let field_trimmed = field.trim();
                if field_trimmed.is_empty() {
                    // Handle missing values based on strategy
                    match &self.missing_value_strategy {
                        MissingValueStrategy::SkipRow => {
                            // Skip entire row if any value is missing
                            continue;
                        }
                        MissingValueStrategy::FillValue(ref default_str) => {
                            let value: T = default_str.parse().map_err(|_| {
                                error_helpers::data_corruption(
                                    "build",
                                    format!("Failed to parse default value '{}'", default_str),
                                    Some(path.clone()),
                                )
                            })?;
                            if Some(i) == self.label_column_index {
                                label = Some(value);
                            } else {
                                row_data.push(value);
                            }
                            continue;
                        }
                        MissingValueStrategy::FillMean => {
                            // For now, use default value (mean calculation would require two passes)
                            let value = T::default();
                            if Some(i) == self.label_column_index {
                                label = Some(value);
                            } else {
                                row_data.push(value);
                            }
                            continue;
                        }
                        MissingValueStrategy::ForwardFill => {
                            // Use last value or default
                            let value = row_data.last().cloned().unwrap_or_default();
                            if Some(i) != self.label_column_index {
                                row_data.push(value);
                            }
                            continue;
                        }
                        MissingValueStrategy::BackwardFill => {
                            // Simplified: use default (proper backfill requires two passes)
                            let value = T::default();
                            if Some(i) == self.label_column_index {
                                label = Some(value);
                            } else {
                                row_data.push(value);
                            }
                            continue;
                        }
                    }
                }

                let value: T = field_trimmed.parse::<T>().map_err(|e| {
                    error_helpers::data_corruption(
                        "build",
                        format!("Failed to parse value '{}': {}", field_trimmed, e),
                        Some(path.clone()),
                    )
                })?;

                if Some(i) == self.label_column_index {
                    label = Some(value);
                } else {
                    row_data.push(value);
                }
            }

            let label_value = label.unwrap_or_default();
            all_data.push(row_data);
            all_labels.push(label_value);
        }

        Ok(CsvDataset::new(all_data, all_labels))
    }

    #[cfg(not(feature = "csv_format"))]
    pub fn build<T>(self) -> Result<CsvDataset<T>>
    where
        T: Clone + Default + std::str::FromStr + Send + Sync + 'static,
    {
        Err(TensorError::InvalidOperation {
            operation: "CsvDatasetBuilder::build".to_string(),
            reason:
                "CSV format feature not enabled. Enable 'csv_format' feature to use CSV datasets."
                    .to_string(),
            context: None,
        })
    }
}

/// Chunked CSV dataset for large files
pub struct ChunkedCsvDataset<T> {
    chunks: Vec<CsvDataset<T>>,
    chunk_size: usize,
}

impl<T> ChunkedCsvDataset<T>
where
    T: Clone + Default + std::str::FromStr + Send + Sync + 'static,
{
    pub fn new(chunks: Vec<CsvDataset<T>>, chunk_size: usize) -> Self {
        Self { chunks, chunk_size }
    }

    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }
}

impl<T> Dataset<T> for ChunkedCsvDataset<T>
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
{
    fn len(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.len()).sum()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        let mut current_index = index;
        for chunk in &self.chunks {
            if current_index < chunk.len() {
                return chunk.get(current_index);
            }
            current_index -= chunk.len();
        }
        Err(TensorError::invalid_argument(format!(
            "Index {index} out of bounds for chunked dataset"
        )))
    }
}

/// CSV chunk representation
pub struct CsvChunk {
    pub start_row: usize,
    pub end_row: usize,
    pub data: Vec<String>,
}

impl CsvChunk {
    pub fn new(start_row: usize, end_row: usize, data: Vec<String>) -> Self {
        Self {
            start_row,
            end_row,
            data,
        }
    }
}
