//! Cross-format operations and utilities
//!
//! This module provides utilities for working with multiple data formats,
//! including format conversion, unified batch reading, and cross-format concatenation.

use crate::error_taxonomy::helpers as error_helpers;
use crate::formats::unified_reader::{
    DataType, FieldInfo, FormatMetadata, FormatReader, FormatSample,
};
use std::path::Path;
use tenflowers_core::{Result, Tensor, TensorError};

/// Cross-format iterator that unifies iteration across different formats
pub struct CrossFormatIterator<'a> {
    readers: Vec<Box<dyn FormatReader + 'a>>,
    current_reader_idx: usize,
    current_sample_idx: usize,
}

impl<'a> CrossFormatIterator<'a> {
    /// Create a new cross-format iterator
    pub fn new(readers: Vec<Box<dyn FormatReader + 'a>>) -> Self {
        Self {
            readers,
            current_reader_idx: 0,
            current_sample_idx: 0,
        }
    }

    /// Get total number of samples across all readers
    pub fn total_samples(&self) -> usize {
        self.readers.iter().map(|r| r.len()).sum()
    }
}

impl<'a> Iterator for CrossFormatIterator<'a> {
    type Item = Result<FormatSample>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_reader_idx < self.readers.len() {
            let reader = &self.readers[self.current_reader_idx];

            if self.current_sample_idx < reader.len() {
                let result = reader.get_sample(self.current_sample_idx);
                self.current_sample_idx += 1;
                return Some(result);
            }

            // Move to next reader
            self.current_reader_idx += 1;
            self.current_sample_idx = 0;
        }

        None
    }
}

/// Unified batch reader for cross-format batch operations
pub struct UnifiedBatchReader {
    reader: Box<dyn FormatReader>,
    batch_size: usize,
    current_index: usize,
}

impl UnifiedBatchReader {
    /// Create a new unified batch reader
    pub fn new(reader: Box<dyn FormatReader>, batch_size: usize) -> Self {
        Self {
            reader,
            batch_size,
            current_index: 0,
        }
    }

    /// Get next batch
    pub fn next_batch(&mut self) -> Result<Option<Vec<FormatSample>>> {
        if self.current_index >= self.reader.len() {
            return Ok(None);
        }

        let end_index = (self.current_index + self.batch_size).min(self.reader.len());
        let batch = self
            .reader
            .get_samples(&(self.current_index..end_index).collect::<Vec<_>>())?;

        self.current_index = end_index;
        Ok(Some(batch))
    }

    /// Reset iterator to beginning
    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    /// Get total number of batches
    pub fn num_batches(&self) -> usize {
        (self.reader.len() + self.batch_size - 1) / self.batch_size
    }
}

/// Format converter for converting between different formats
pub struct FormatConverter;

impl FormatConverter {
    /// Convert samples from one format to tensor batches
    pub fn samples_to_tensors(samples: &[FormatSample]) -> Result<(Tensor<f32>, Tensor<f32>)> {
        if samples.is_empty() {
            return Err(TensorError::invalid_argument(
                "Empty sample list".to_string(),
            ));
        }

        // Stack features
        let feature_data: Vec<Vec<f32>> = samples
            .iter()
            .map(|s| {
                s.features
                    .as_slice()
                    .ok_or_else(|| {
                        TensorError::invalid_argument("Cannot access feature data".to_string())
                    })
                    .map(|slice| slice.to_vec())
            })
            .collect::<Result<Vec<_>>>()?;

        // Stack labels
        let label_data: Vec<Vec<f32>> = samples
            .iter()
            .map(|s| {
                s.labels
                    .as_slice()
                    .ok_or_else(|| {
                        TensorError::invalid_argument("Cannot access label data".to_string())
                    })
                    .map(|slice| slice.to_vec())
            })
            .collect::<Result<Vec<_>>>()?;

        let batch_size = samples.len();
        let feature_size = feature_data[0].len();
        let label_size = label_data[0].len();

        // Flatten and create tensors
        let flat_features: Vec<f32> = feature_data.into_iter().flatten().collect();
        let flat_labels: Vec<f32> = label_data.into_iter().flatten().collect();

        let features = Tensor::from_vec(flat_features, &[batch_size, feature_size])?;
        let labels = Tensor::from_vec(flat_labels, &[batch_size, label_size])?;

        Ok((features, labels))
    }

    /// Concatenate samples from multiple readers
    pub fn concatenate_samples(samples_list: Vec<Vec<FormatSample>>) -> Result<Vec<FormatSample>> {
        let mut all_samples = Vec::new();
        let mut global_index = 0;

        for samples in samples_list {
            for mut sample in samples {
                sample.source_index = global_index;
                global_index += 1;
                all_samples.push(sample);
            }
        }

        Ok(all_samples)
    }
}

/// Schema compatibility checker
pub struct SchemaCompatibility;

impl SchemaCompatibility {
    /// Check if two schemas are compatible for concatenation
    pub fn are_compatible(schema1: &FormatMetadata, schema2: &FormatMetadata) -> Result<bool> {
        // Check if field counts match
        if schema1.fields.len() != schema2.fields.len() {
            return Ok(false);
        }

        // Check if field types are compatible
        for (field1, field2) in schema1.fields.iter().zip(schema2.fields.iter()) {
            if !Self::are_types_compatible(&field1.dtype, &field2.dtype) {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Check if two data types are compatible
    pub fn are_types_compatible(type1: &DataType, type2: &DataType) -> bool {
        match (type1, type2) {
            // Exact matches
            (DataType::Bool, DataType::Bool) => true,
            (DataType::String, DataType::String) => true,
            (DataType::Binary, DataType::Binary) => true,

            // Numeric types are compatible with each other
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

            // List types need compatible element types
            (DataType::List(inner1), DataType::List(inner2)) => {
                Self::are_types_compatible(inner1, inner2)
            }

            // Struct types need all fields to be compatible
            (DataType::Struct(fields1), DataType::Struct(fields2)) => {
                if fields1.len() != fields2.len() {
                    return false;
                }
                fields1
                    .iter()
                    .zip(fields2.iter())
                    .all(|(f1, f2)| Self::are_types_compatible(&f1.dtype, &f2.dtype))
            }

            // Tensor types
            (DataType::Tensor(_), DataType::Tensor(_)) => true,

            // Everything else is incompatible
            _ => false,
        }
    }

    /// Merge two schemas into a unified schema
    pub fn merge_schemas(schemas: &[FormatMetadata]) -> Result<FormatMetadata> {
        if schemas.is_empty() {
            return Err(error_helpers::invalid_configuration(
                "merge_schemas",
                "schemas",
                "Cannot merge empty schema list",
            ));
        }

        let first = &schemas[0];

        // Check all schemas are compatible
        for schema in &schemas[1..] {
            if !Self::are_compatible(first, schema)? {
                return Err(error_helpers::schema_mismatch(
                    "merge_schemas",
                    "compatible schemas",
                    "incompatible schemas found",
                ));
            }
        }

        // Create merged schema
        let total_samples: usize = schemas.iter().map(|s| s.num_samples).sum();

        Ok(FormatMetadata {
            format_name: "Merged".to_string(),
            version: None,
            num_samples: total_samples,
            fields: first.fields.clone(),
            metadata: first.metadata.clone(),
            supports_random_access: schemas.iter().all(|s| s.supports_random_access),
            supports_streaming: schemas.iter().any(|s| s.supports_streaming),
        })
    }
}

/// Cross-format dataset concatenation
pub struct CrossFormatConcatenation {
    readers: Vec<Box<dyn FormatReader>>,
    cumulative_lengths: Vec<usize>,
    total_length: usize,
}

impl CrossFormatConcatenation {
    /// Create a new cross-format concatenation
    pub fn new(readers: Vec<Box<dyn FormatReader>>) -> Result<Self> {
        if readers.is_empty() {
            return Err(error_helpers::invalid_configuration(
                "CrossFormatConcatenation::new",
                "readers",
                "Cannot create concatenation with no readers",
            ));
        }

        // Verify schema compatibility
        let schemas: Vec<FormatMetadata> = readers
            .iter()
            .map(|r| r.metadata())
            .collect::<Result<Vec<_>>>()?;

        SchemaCompatibility::merge_schemas(&schemas)?;

        let mut cumulative_lengths = Vec::with_capacity(readers.len());
        let mut total_length = 0;

        for reader in &readers {
            total_length += reader.len();
            cumulative_lengths.push(total_length);
        }

        Ok(Self {
            readers,
            cumulative_lengths,
            total_length,
        })
    }

    /// Get total length
    pub fn len(&self) -> usize {
        self.total_length
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.total_length == 0
    }

    /// Get sample by global index
    pub fn get_sample(&self, global_index: usize) -> Result<FormatSample> {
        if global_index >= self.total_length {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for concatenated dataset of length {}",
                global_index, self.total_length
            )));
        }

        // Find which reader and local index
        for (reader_idx, &cumulative_len) in self.cumulative_lengths.iter().enumerate() {
            if global_index < cumulative_len {
                let local_index = if reader_idx == 0 {
                    global_index
                } else {
                    global_index - self.cumulative_lengths[reader_idx - 1]
                };

                return self.readers[reader_idx].get_sample(local_index);
            }
        }

        Err(TensorError::invalid_argument(format!(
            "Index {} could not be mapped to reader",
            global_index
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_compatibility() {
        assert!(SchemaCompatibility::are_types_compatible(
            &DataType::Float32,
            &DataType::Float64
        ));
        assert!(SchemaCompatibility::are_types_compatible(
            &DataType::Int32,
            &DataType::Int64
        ));
        assert!(!SchemaCompatibility::are_types_compatible(
            &DataType::Bool,
            &DataType::String
        ));
    }

    #[test]
    fn test_list_type_compatibility() {
        let list1 = DataType::List(Box::new(DataType::Float32));
        let list2 = DataType::List(Box::new(DataType::Int32));

        assert!(SchemaCompatibility::are_types_compatible(&list1, &list2));
    }

    #[test]
    fn test_format_converter_empty() {
        let result = FormatConverter::samples_to_tensors(&[]);
        assert!(result.is_err());
    }
}
