use crate::{Result, TrackedTensor};
use std::collections::HashMap;
use tenflowers_core::Tensor;

/// Different compression techniques for gradients
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionMethod {
    /// No compression applied
    None,
    /// Quantize gradients to fewer bits (8-bit, 16-bit)
    Quantization { bits: u8 },
    /// Apply sparsification by zeroing out small gradients
    Sparsification { threshold: f32 },
    /// Combined quantization and sparsification
    Hybrid {
        quantization_bits: u8,
        sparsification_threshold: f32,
    },
    /// Top-K sparsification - keep only K largest absolute values
    TopK { k: usize },
}

/// Configuration for gradient compression
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    pub method: CompressionMethod,
    /// Enable error feedback to compensate for compression
    pub error_feedback: bool,
    /// Minimum tensor size to apply compression (avoid overhead on small tensors)
    pub min_tensor_size: usize,
    /// Compression ratio target (0.0 = no compression, 1.0 = maximum compression)
    pub target_ratio: f32,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            method: CompressionMethod::Sparsification { threshold: 1e-4 },
            error_feedback: true,
            min_tensor_size: 1024, // Only compress tensors with 1K+ elements
            target_ratio: 0.9,     // 90% compression target
        }
    }
}

/// Gradient compression engine
pub struct GradientCompressor {
    config: CompressionConfig,
    /// Error accumulation for error feedback
    error_feedback: HashMap<String, Tensor<f32>>,
    /// Compression statistics
    stats: CompressionStats,
}

/// Statistics about compression performance
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    pub total_gradients_processed: usize,
    pub total_bytes_original: usize,
    pub total_bytes_compressed: usize,
    pub average_compression_ratio: f32,
    pub total_compression_time_ms: f32,
}

impl CompressionStats {
    /// Calculate current compression ratio
    pub fn compression_ratio(&self) -> f32 {
        if self.total_bytes_original == 0 {
            0.0
        } else {
            1.0 - (self.total_bytes_compressed as f32 / self.total_bytes_original as f32)
        }
    }

    /// Calculate average processing time per gradient
    pub fn average_processing_time_ms(&self) -> f32 {
        if self.total_gradients_processed == 0 {
            0.0
        } else {
            self.total_compression_time_ms / self.total_gradients_processed as f32
        }
    }
}

impl GradientCompressor {
    /// Create a new gradient compressor with default configuration
    pub fn new() -> Self {
        Self::with_config(CompressionConfig::default())
    }

    /// Create a new gradient compressor with specific configuration
    pub fn with_config(config: CompressionConfig) -> Self {
        Self {
            config,
            error_feedback: HashMap::new(),
            stats: CompressionStats::default(),
        }
    }

    /// Compress a gradient tensor
    pub fn compress(
        &mut self,
        gradient: &TrackedTensor<f32>,
        tensor_id: &str,
    ) -> Result<CompressedGradient> {
        let start_time = std::time::Instant::now();

        // Skip compression for small tensors
        if gradient.tensor().shape().size() < self.config.min_tensor_size {
            return Ok(CompressedGradient::uncompressed(gradient.clone()));
        }

        // Apply error feedback if enabled
        let mut grad_with_feedback = gradient.clone();
        if self.config.error_feedback {
            if let Some(error) = self.error_feedback.get(tensor_id) {
                // Create a tracked tensor from the error tensor
                let error_tracked = TrackedTensor::new(error.clone());
                grad_with_feedback = grad_with_feedback.add(&error_tracked)?;
            }
        }

        let original_size = grad_with_feedback.tensor().shape().size() * std::mem::size_of::<f32>();

        let compressed = match self.config.method {
            CompressionMethod::None => CompressedGradient::uncompressed(grad_with_feedback),
            CompressionMethod::Quantization { bits } => {
                self.quantize_gradient(&grad_with_feedback, bits)?
            }
            CompressionMethod::Sparsification { threshold } => {
                self.sparsify_gradient(&grad_with_feedback, threshold, tensor_id)?
            }
            CompressionMethod::Hybrid {
                quantization_bits,
                sparsification_threshold,
            } => {
                let sparsified = self.sparsify_gradient(
                    &grad_with_feedback,
                    sparsification_threshold,
                    tensor_id,
                )?;
                self.quantize_compressed_gradient(sparsified, quantization_bits)?
            }
            CompressionMethod::TopK { k } => {
                self.topk_sparsify(&grad_with_feedback, k, tensor_id)?
            }
        };

        let compressed_size = compressed.estimated_size_bytes();
        let compression_time = start_time.elapsed().as_secs_f32() * 1000.0;

        // Update statistics
        self.stats.total_gradients_processed += 1;
        self.stats.total_bytes_original += original_size;
        self.stats.total_bytes_compressed += compressed_size;
        self.stats.total_compression_time_ms += compression_time;
        self.stats.average_compression_ratio = self.stats.compression_ratio();

        Ok(compressed)
    }

    /// Decompress a gradient back to full representation
    pub fn decompress(
        &mut self,
        compressed: &CompressedGradient,
        _tensor_id: &str,
    ) -> Result<TrackedTensor<f32>> {
        let decompressed = match compressed {
            CompressedGradient::Uncompressed { gradient } => gradient.clone(),
            CompressedGradient::Quantized {
                values,
                scale,
                zero_point,
                shape,
            } => self.dequantize_gradient(values, *scale, *zero_point, shape)?,
            CompressedGradient::Sparse {
                indices,
                values,
                shape,
                ..
            } => self.densify_gradient(indices, values, shape)?,
            CompressedGradient::QuantizedSparse {
                indices,
                quantized_values,
                scale,
                zero_point,
                shape,
            } => {
                // First dequantize the values
                let values_tensor =
                    Tensor::<u8>::from_vec(quantized_values.clone(), &[quantized_values.len()])?;
                let dequantized = self.dequantize_values(&values_tensor, *scale, *zero_point)?;
                let values = dequantized
                    .as_slice()
                    .ok_or_else(|| {
                        tenflowers_core::error::TensorError::invalid_argument(
                            "Failed to get dequantized slice".to_string(),
                        )
                    })?
                    .to_vec();

                // Then densify
                self.densify_gradient(indices, &values, shape)?
            }
            CompressedGradient::TopK {
                indices,
                values,
                shape,
            } => self.densify_gradient(indices, values, shape)?,
        };

        Ok(decompressed)
    }

    /// Quantize gradient to fewer bits
    fn quantize_gradient(
        &self,
        gradient: &TrackedTensor<f32>,
        bits: u8,
    ) -> Result<CompressedGradient> {
        let data = gradient.tensor().as_slice().ok_or_else(|| {
            tenflowers_core::error::TensorError::invalid_argument(
                "Failed to get tensor slice".to_string(),
            )
        })?;
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let num_levels = (1u32 << bits) as f32;
        let scale = (max_val - min_val) / (num_levels - 1.0);
        let zero_point = -min_val / scale;

        let quantized: Vec<u8> = data
            .iter()
            .map(|&x| ((x - min_val) / scale).round().clamp(0.0, num_levels - 1.0) as u8)
            .collect();

        Ok(CompressedGradient::Quantized {
            values: quantized,
            scale,
            zero_point,
            shape: gradient.tensor().shape().dims().to_vec(),
        })
    }

    /// Sparsify gradient by zeroing out small values
    fn sparsify_gradient(
        &mut self,
        gradient: &TrackedTensor<f32>,
        threshold: f32,
        tensor_id: &str,
    ) -> Result<CompressedGradient> {
        let data = gradient.tensor().as_slice().ok_or_else(|| {
            tenflowers_core::error::TensorError::invalid_argument(
                "Failed to get tensor slice".to_string(),
            )
        })?;
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut error_accumulation = vec![0.0; data.len()];

        for (i, &val) in data.iter().enumerate() {
            if val.abs() >= threshold {
                indices.push(i);
                values.push(val);
            } else if self.config.error_feedback {
                // Store error for feedback
                error_accumulation[i] = val;
            }
        }

        // Update error feedback
        if self.config.error_feedback && !error_accumulation.iter().all(|&x| x == 0.0) {
            let error_tensor =
                Tensor::from_vec(error_accumulation, gradient.tensor().shape().dims())?;
            self.error_feedback
                .insert(tensor_id.to_string(), error_tensor);
        }

        let nnz = values.len();
        Ok(CompressedGradient::Sparse {
            indices,
            values,
            shape: gradient.tensor().shape().dims().to_vec(),
            nnz,
        })
    }

    /// Top-K sparsification - keep only K largest absolute values
    fn topk_sparsify(
        &mut self,
        gradient: &TrackedTensor<f32>,
        k: usize,
        tensor_id: &str,
    ) -> Result<CompressedGradient> {
        let data = gradient.tensor().as_slice().ok_or_else(|| {
            tenflowers_core::error::TensorError::invalid_argument(
                "Failed to get tensor slice".to_string(),
            )
        })?;
        let total_elements = data.len();
        let k = k.min(total_elements);

        // Create pairs of (absolute_value, index, original_value)
        let mut indexed_values: Vec<(f32, usize, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, &val)| (val.abs(), i, val))
            .collect();

        // Sort by absolute value in descending order
        indexed_values.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Take top K
        let mut indices = Vec::with_capacity(k);
        let mut values = Vec::with_capacity(k);
        let mut error_accumulation = vec![0.0; total_elements];

        for (i, (_, idx, val)) in indexed_values.iter().enumerate() {
            if i < k {
                indices.push(*idx);
                values.push(*val);
            } else if self.config.error_feedback {
                error_accumulation[*idx] = *val;
            }
        }

        // Update error feedback
        if self.config.error_feedback && !error_accumulation.iter().all(|&x| x == 0.0) {
            let error_tensor =
                Tensor::from_vec(error_accumulation, gradient.tensor().shape().dims())?;
            self.error_feedback
                .insert(tensor_id.to_string(), error_tensor);
        }

        Ok(CompressedGradient::TopK {
            indices,
            values,
            shape: gradient.tensor().shape().dims().to_vec(),
        })
    }

    /// Quantize an already compressed sparse gradient
    fn quantize_compressed_gradient(
        &self,
        compressed: CompressedGradient,
        bits: u8,
    ) -> Result<CompressedGradient> {
        match compressed {
            CompressedGradient::Sparse {
                indices,
                values,
                shape,
                ..
            } => {
                let values_tensor = Tensor::from_vec(values, &[indices.len()])?;
                let quantized = self.quantize_values(&values_tensor, bits)?;

                Ok(CompressedGradient::QuantizedSparse {
                    indices,
                    quantized_values: quantized.values,
                    scale: quantized.scale,
                    zero_point: quantized.zero_point,
                    shape,
                })
            }
            _ => Ok(compressed), // Already quantized or uncompressed
        }
    }

    /// Helper to quantize just the values
    fn quantize_values(&self, values: &Tensor<f32>, bits: u8) -> Result<QuantizedValues> {
        let data = values.as_slice().ok_or_else(|| {
            tenflowers_core::error::TensorError::invalid_argument(
                "Failed to get values slice".to_string(),
            )
        })?;
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let num_levels = (1u32 << bits) as f32;
        let scale = (max_val - min_val) / (num_levels - 1.0);
        let zero_point = -min_val / scale;

        let quantized: Vec<u8> = data
            .iter()
            .map(|&x| ((x - min_val) / scale).round().clamp(0.0, num_levels - 1.0) as u8)
            .collect();

        Ok(QuantizedValues {
            values: quantized,
            scale,
            zero_point,
        })
    }

    /// Dequantize gradient values
    fn dequantize_gradient(
        &self,
        values: &[u8],
        scale: f32,
        zero_point: f32,
        shape: &[usize],
    ) -> Result<TrackedTensor<f32>> {
        let dequantized: Vec<f32> = values
            .iter()
            .map(|&q| (q as f32 - zero_point) * scale)
            .collect();

        let tensor = Tensor::from_vec(dequantized, shape)?;
        Ok(TrackedTensor::new(tensor))
    }

    /// Dequantize just values (helper)
    fn dequantize_values(
        &self,
        values: &Tensor<u8>,
        scale: f32,
        zero_point: f32,
    ) -> Result<Tensor<f32>> {
        let data = values.as_slice().ok_or_else(|| {
            tenflowers_core::error::TensorError::invalid_shape_simple(
                "Failed to get slice".to_string(),
            )
        })?;
        let dequantized: Vec<f32> = data
            .iter()
            .map(|&q| (q as f32 - zero_point) * scale)
            .collect();

        Tensor::from_vec(dequantized, values.shape().dims())
    }

    /// Convert sparse representation back to dense
    fn densify_gradient(
        &self,
        indices: &[usize],
        values: &[f32],
        shape: &[usize],
    ) -> Result<TrackedTensor<f32>> {
        let total_elements: usize = shape.iter().product();
        let mut dense_data = vec![0.0; total_elements];

        for (&idx, &val) in indices.iter().zip(values.iter()) {
            if idx < total_elements {
                dense_data[idx] = val;
            }
        }

        let tensor = Tensor::from_vec(dense_data, shape)?;
        Ok(TrackedTensor::new(tensor))
    }

    /// Get compression statistics
    pub fn get_stats(&self) -> &CompressionStats {
        &self.stats
    }

    /// Reset compression statistics
    pub fn reset_stats(&mut self) {
        self.stats = CompressionStats::default();
    }

    /// Clear error feedback accumulation
    pub fn clear_error_feedback(&mut self) {
        self.error_feedback.clear();
    }

    /// Update compression configuration
    pub fn update_config(&mut self, config: CompressionConfig) {
        self.config = config;
        // Clear error feedback when config changes
        self.error_feedback.clear();
    }
}

/// Helper struct for quantized values
#[derive(Debug, Clone)]
struct QuantizedValues {
    values: Vec<u8>,
    scale: f32,
    zero_point: f32,
}

/// Compressed gradient representation
#[derive(Debug, Clone)]
pub enum CompressedGradient {
    /// No compression applied
    Uncompressed { gradient: TrackedTensor<f32> },
    /// Quantized to fewer bits
    Quantized {
        values: Vec<u8>,
        scale: f32,
        zero_point: f32,
        shape: Vec<usize>,
    },
    /// Sparsified gradient (only non-zero values stored)
    Sparse {
        indices: Vec<usize>,
        values: Vec<f32>,
        shape: Vec<usize>,
        nnz: usize, // number of non-zeros
    },
    /// Both quantized and sparsified
    QuantizedSparse {
        indices: Vec<usize>,
        quantized_values: Vec<u8>,
        scale: f32,
        zero_point: f32,
        shape: Vec<usize>,
    },
    /// Top-K sparsified
    TopK {
        indices: Vec<usize>,
        values: Vec<f32>,
        shape: Vec<usize>,
    },
}

impl CompressedGradient {
    /// Create an uncompressed gradient
    pub fn uncompressed(gradient: TrackedTensor<f32>) -> Self {
        Self::Uncompressed { gradient }
    }

    /// Estimate the size in bytes of the compressed representation
    pub fn estimated_size_bytes(&self) -> usize {
        match self {
            Self::Uncompressed { gradient } => {
                gradient.tensor().shape().size() * std::mem::size_of::<f32>()
            }
            Self::Quantized { values, .. } => values.len() + 3 * std::mem::size_of::<f32>(),
            Self::Sparse {
                indices, values, ..
            } => {
                indices.len() * std::mem::size_of::<usize>()
                    + values.len() * std::mem::size_of::<f32>()
            }
            Self::QuantizedSparse {
                indices,
                quantized_values,
                ..
            } => {
                indices.len() * std::mem::size_of::<usize>()
                    + quantized_values.len()
                    + 2 * std::mem::size_of::<f32>()
            }
            Self::TopK {
                indices, values, ..
            } => {
                indices.len() * std::mem::size_of::<usize>()
                    + values.len() * std::mem::size_of::<f32>()
            }
        }
    }

    /// Get the compression ratio compared to uncompressed size
    pub fn compression_ratio(&self, original_size: usize) -> f32 {
        let compressed_size = self.estimated_size_bytes();
        1.0 - (compressed_size as f32 / original_size as f32)
    }
}

impl Default for GradientCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_compressor_creation() {
        let compressor = GradientCompressor::new();
        assert_eq!(compressor.config.min_tensor_size, 1024);
        assert!(compressor.config.error_feedback);
    }

    #[test]
    fn test_compression_config() {
        let config = CompressionConfig {
            method: CompressionMethod::Quantization { bits: 8 },
            error_feedback: false,
            min_tensor_size: 512,
            target_ratio: 0.8,
        };

        let compressor = GradientCompressor::with_config(config);
        assert_eq!(compressor.config.min_tensor_size, 512);
        assert!(!compressor.config.error_feedback);
    }

    #[test]
    fn test_compression_stats() {
        let stats = CompressionStats {
            total_gradients_processed: 10,
            total_bytes_original: 1000,
            total_bytes_compressed: 200,
            ..Default::default()
        };

        assert_eq!(stats.compression_ratio(), 0.8); // 80% compression
    }

    #[test]
    fn test_compressed_gradient_size_estimation() {
        // Test uncompressed
        let gradient = TrackedTensor::new(Tensor::<f32>::zeros(&[100]));
        let uncompressed = CompressedGradient::uncompressed(gradient);
        assert_eq!(uncompressed.estimated_size_bytes(), 100 * 4); // 4 bytes per f32

        // Test quantized
        let quantized = CompressedGradient::Quantized {
            values: vec![0u8; 100],
            scale: 1.0,
            zero_point: 0.0,
            shape: vec![100],
        };
        assert_eq!(quantized.estimated_size_bytes(), 100 + 12); // 100 bytes + 3 f32s

        // Test sparse
        let sparse = CompressedGradient::Sparse {
            indices: vec![1, 2, 3],
            values: vec![1.0, 2.0, 3.0],
            shape: vec![100],
            nnz: 3,
        };
        assert_eq!(sparse.estimated_size_bytes(), 3 * 8 + 3 * 4); // 3 usize + 3 f32
    }
}
