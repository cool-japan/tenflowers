//! JSON format factory and reader implementation
//!
//! This module implements the FormatFactory and FormatReader traits for JSON and JSONL files,
//! enabling automatic format detection and unified data loading.

use crate::error_taxonomy::helpers as error_helpers;
use crate::formats::unified_reader::{
    read_magic_bytes, DataType, DetectionMethod, FieldInfo, FormatDetection, FormatFactory,
    FormatMetadata, FormatReader, FormatSample,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use tenflowers_core::{Result, Tensor, TensorError};

/// JSON format factory for automatic detection and reader creation
pub struct JsonFormatFactory;

impl FormatFactory for JsonFormatFactory {
    fn format_name(&self) -> &str {
        "JSON"
    }

    fn extensions(&self) -> Vec<&str> {
        vec!["json", "jsonl", "ndjson"]
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
            Some("json") => {
                confidence = 0.95;
                method = DetectionMethod::Extension;
            }
            Some("jsonl") | Some("ndjson") => {
                confidence = 0.95;
                method = DetectionMethod::Extension;
            }
            _ => {
                // Try magic bytes detection
                if let Ok(is_json) = Self::check_json_content(path) {
                    if is_json {
                        confidence = 0.8;
                        method = DetectionMethod::MagicBytes;
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
        Ok(Box::new(JsonFormatReader::new(path)?))
    }
}

impl JsonFormatFactory {
    /// Check if file content looks like JSON
    fn check_json_content(path: &Path) -> Result<bool> {
        if let Ok(bytes) = read_magic_bytes(path, 10) {
            // JSON typically starts with '{' or '['
            let starts_with_brace = !bytes.is_empty() && (bytes[0] == b'{' || bytes[0] == b'[');

            // Check for whitespace before brace (common in formatted JSON)
            let starts_with_whitespace = bytes
                .iter()
                .take(5)
                .any(|&b| b == b' ' || b == b'\n' || b == b'\r' || b == b'\t');
            let has_brace_after = bytes.iter().any(|&b| b == b'{' || b == b'[');

            Ok(starts_with_brace || (starts_with_whitespace && has_brace_after))
        } else {
            Ok(false)
        }
    }
}

/// JSON format reader implementation
pub struct JsonFormatReader {
    path: PathBuf,
    metadata: FormatMetadata,
    samples: Vec<serde_json::Value>,
    is_jsonl: bool,
}

impl JsonFormatReader {
    /// Create a new JSON format reader
    pub fn new(path: &Path) -> Result<Self> {
        let (is_jsonl, samples) = Self::load_json_data(path)?;

        // Infer fields from first sample
        let fields = if !samples.is_empty() {
            Self::infer_fields(&samples[0])
        } else {
            Vec::new()
        };

        let metadata = FormatMetadata {
            format_name: if is_jsonl { "JSONL" } else { "JSON" }.to_string(),
            version: None,
            num_samples: samples.len(),
            fields,
            metadata: HashMap::new(),
            supports_random_access: true,
            supports_streaming: is_jsonl,
        };

        Ok(Self {
            path: path.to_path_buf(),
            metadata,
            samples,
            is_jsonl,
        })
    }

    /// Load JSON data from file
    fn load_json_data(path: &Path) -> Result<(bool, Vec<serde_json::Value>)> {
        let file = File::open(path)
            .map_err(|_| error_helpers::file_not_found("JsonFormatReader::load_json_data", path))?;

        let reader = BufReader::new(file);

        // Try JSONL format first (one JSON object per line)
        let mut lines = reader.lines();
        if let Some(Ok(first_line)) = lines.next() {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&first_line) {
                // Looks like JSONL
                let mut samples = vec![value];

                for line_result in lines {
                    let line = line_result.map_err(|e| {
                        error_helpers::data_corruption(
                            "JsonFormatReader::load_json_data",
                            format!("Failed to read line: {}", e),
                            Some(path.to_path_buf()),
                        )
                    })?;

                    if !line.trim().is_empty() {
                        let value = serde_json::from_str(&line).map_err(|e| {
                            error_helpers::data_corruption(
                                "JsonFormatReader::load_json_data",
                                format!("JSONL parse error: {}", e),
                                Some(path.to_path_buf()),
                            )
                        })?;
                        samples.push(value);
                    }
                }

                return Ok((true, samples));
            }
        }

        // Try regular JSON format
        let file = File::open(path)
            .map_err(|_| error_helpers::file_not_found("JsonFormatReader::load_json_data", path))?;

        let reader = BufReader::new(file);
        let json_value: serde_json::Value = serde_json::from_reader(reader).map_err(|e| {
            error_helpers::data_corruption(
                "JsonFormatReader::load_json_data",
                format!("JSON parse error: {}", e),
                Some(path.to_path_buf()),
            )
        })?;

        match json_value {
            serde_json::Value::Array(arr) => Ok((false, arr)),
            serde_json::Value::Object(_) => {
                // Single object, wrap in array
                Ok((false, vec![json_value]))
            }
            _ => Err(error_helpers::data_corruption(
                "JsonFormatReader::load_json_data",
                "JSON must be array or object",
                Some(path.to_path_buf()),
            )),
        }
    }

    /// Infer field information from a JSON value
    fn infer_fields(value: &serde_json::Value) -> Vec<FieldInfo> {
        let mut fields = Vec::new();

        if let serde_json::Value::Object(map) = value {
            for (key, val) in map {
                let dtype = Self::infer_type(val);
                fields.push(FieldInfo {
                    name: key.clone(),
                    dtype,
                    shape: None,
                    nullable: val.is_null(),
                    description: None,
                });
            }
        }

        fields
    }

    /// Infer data type from JSON value
    fn infer_type(value: &serde_json::Value) -> DataType {
        match value {
            serde_json::Value::Null => DataType::String,
            serde_json::Value::Bool(_) => DataType::Bool,
            serde_json::Value::Number(n) => {
                if n.is_i64() {
                    DataType::Int64
                } else if n.is_u64() {
                    DataType::UInt64
                } else {
                    DataType::Float64
                }
            }
            serde_json::Value::String(_) => DataType::String,
            serde_json::Value::Array(arr) => {
                if let Some(first) = arr.first() {
                    DataType::List(Box::new(Self::infer_type(first)))
                } else {
                    DataType::List(Box::new(DataType::String))
                }
            }
            serde_json::Value::Object(_) => {
                let nested_fields = Self::infer_fields(value);
                DataType::Struct(nested_fields)
            }
        }
    }

    /// Convert JSON value to tensor
    fn json_value_to_tensor(value: &serde_json::Value) -> Result<Tensor<f32>> {
        match value {
            serde_json::Value::Number(n) => {
                let val = n.as_f64().ok_or_else(|| {
                    TensorError::invalid_argument("Cannot convert to f64".to_string())
                })?;
                Ok(Tensor::from_scalar(val as f32))
            }
            serde_json::Value::Array(arr) => {
                let mut data = Vec::new();
                Self::flatten_json_array(arr, &mut data)?;
                let len = data.len();
                Tensor::from_vec(data, &[len])
            }
            serde_json::Value::Bool(b) => Ok(Tensor::from_scalar(if *b { 1.0 } else { 0.0 })),
            _ => Err(TensorError::invalid_argument(
                "Cannot convert JSON value to tensor".to_string(),
            )),
        }
    }

    /// Flatten JSON array recursively
    fn flatten_json_array(arr: &[serde_json::Value], data: &mut Vec<f32>) -> Result<()> {
        for val in arr {
            match val {
                serde_json::Value::Number(n) => {
                    let v = n.as_f64().ok_or_else(|| {
                        TensorError::invalid_argument("Cannot convert to f64".to_string())
                    })?;
                    data.push(v as f32);
                }
                serde_json::Value::Array(nested) => {
                    Self::flatten_json_array(nested, data)?;
                }
                serde_json::Value::Bool(b) => {
                    data.push(if *b { 1.0 } else { 0.0 });
                }
                _ => {
                    return Err(TensorError::invalid_argument(
                        "Cannot flatten non-numeric JSON array".to_string(),
                    ))
                }
            }
        }
        Ok(())
    }
}

impl FormatReader for JsonFormatReader {
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

        let sample = &self.samples[index];

        if let serde_json::Value::Object(map) = sample {
            // Extract features and labels
            let feature_val = map
                .get("features")
                .or_else(|| map.get("data"))
                .or_else(|| map.get("x"))
                .ok_or_else(|| {
                    TensorError::invalid_argument(
                        "No feature field found in JSON object".to_string(),
                    )
                })?;

            let label_val = map
                .get("label")
                .or_else(|| map.get("target"))
                .or_else(|| map.get("y"))
                .ok_or_else(|| {
                    TensorError::invalid_argument("No label field found in JSON object".to_string())
                })?;

            let features = Self::json_value_to_tensor(feature_val)?;
            let labels = Self::json_value_to_tensor(label_val)?;

            let mut metadata = HashMap::new();
            metadata.insert("source".to_string(), "JSON".to_string());
            metadata.insert("index".to_string(), index.to_string());

            Ok(FormatSample {
                features,
                labels,
                source_index: index,
                metadata,
            })
        } else {
            Err(TensorError::invalid_argument(
                "JSON sample must be an object".to_string(),
            ))
        }
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
    fn test_json_format_detection() {
        let factory = JsonFormatFactory;

        let json_path = Path::new("data.json");
        let detection = factory
            .can_read(json_path)
            .expect("test: format detection should succeed");
        assert!(detection.confidence >= 0.9);
        assert_eq!(detection.format_name, "JSON");
    }

    #[test]
    fn test_json_array_format() {
        let mut temp_file = NamedTempFile::new().expect("test: temp file creation should succeed");
        writeln!(
            temp_file,
            r#"[
                {{"features": [1.0, 2.0], "label": 0}},
                {{"features": [3.0, 4.0], "label": 1}}
            ]"#
        )
        .expect("test: operation should succeed");
        temp_file.flush().expect("test: flush should succeed");

        let reader =
            JsonFormatReader::new(temp_file.path()).expect("test: reader creation should succeed");
        assert_eq!(reader.len(), 2);

        let sample = reader
            .get_sample(0)
            .expect("test: get sample should succeed");
        assert_eq!(sample.source_index, 0);
    }

    #[test]
    fn test_jsonl_format() {
        let mut temp_file = NamedTempFile::new().expect("test: temp file creation should succeed");
        writeln!(temp_file, r#"{{"features": [1.0, 2.0], "label": 0}}"#)
            .expect("test: writeln should succeed");
        writeln!(temp_file, r#"{{"features": [3.0, 4.0], "label": 1}}"#)
            .expect("test: writeln should succeed");
        temp_file.flush().expect("test: flush should succeed");

        let reader =
            JsonFormatReader::new(temp_file.path()).expect("test: reader creation should succeed");
        assert_eq!(reader.len(), 2);
        assert!(reader.metadata.supports_streaming);
    }

    #[test]
    fn test_type_inference() {
        let json_str = r#"{"num": 42, "float": 3.14, "bool": true, "str": "hello"}"#;
        let value: serde_json::Value =
            serde_json::from_str(json_str).expect("test: JSON parsing should succeed");
        let fields = JsonFormatReader::infer_fields(&value);

        assert_eq!(fields.len(), 4);
        assert!(fields
            .iter()
            .any(|f| f.name == "num" && f.dtype == DataType::Int64));
        assert!(fields
            .iter()
            .any(|f| f.name == "float" && f.dtype == DataType::Float64));
        assert!(fields
            .iter()
            .any(|f| f.name == "bool" && f.dtype == DataType::Bool));
        assert!(fields
            .iter()
            .any(|f| f.name == "str" && f.dtype == DataType::String));
    }
}
