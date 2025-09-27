//! Compression Engine and Codecs
//!
//! This module provides compression capabilities for cross-datacenter replication,
//! including various compression algorithms optimized for different network
//! conditions and adaptive compression strategies.

use super::config::{AdaptiveCompressionConfig, CompressionAlgorithm, CompressionConfig};
use std::collections::HashMap;
use std::time::Duration;
use tenflowers_core::TensorError;

/// Compression engine for parameter data
#[allow(dead_code)]
pub struct CompressionEngine {
    algorithms: HashMap<CompressionAlgorithm, Box<dyn CompressionCodec>>,
    current_algorithm: CompressionAlgorithm,
    adaptive_config: AdaptiveCompressionConfig,
}

/// Trait for compression algorithms
pub trait CompressionCodec: Send + Sync {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, TensorError>;
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, TensorError>;
    fn compression_ratio(&self) -> f64;
    fn compute_cost(&self) -> f64; // CPU cost for compression/decompression
}

/// Network conditions for adaptive compression
#[derive(Debug, Clone)]
pub struct NetworkConditions {
    pub available_bandwidth: f64,
    pub current_latency: Duration,
    pub packet_loss_rate: f64,
    pub congestion_level: f64,
}

/// Quantization compression codec
struct QuantizationCodec {
    bits: u8,
}

/// TopK sparsification compression codec
struct TopKCodec {
    k: usize,
}

/// Error feedback compression codec
struct ErrorFeedbackCodec {
    compression_ratio: f64,
}

impl CompressionEngine {
    /// Create a new compression engine with the given configuration
    pub fn new(config: CompressionConfig) -> Result<Self, TensorError> {
        let mut algorithms: HashMap<CompressionAlgorithm, Box<dyn CompressionCodec>> =
            HashMap::new();

        // Register compression algorithms
        algorithms.insert(
            CompressionAlgorithm::Quantization { bits: 8 },
            Box::new(QuantizationCodec::new(8)),
        );

        Ok(CompressionEngine {
            algorithms,
            current_algorithm: config.algorithm,
            adaptive_config: AdaptiveCompressionConfig::default(),
        })
    }

    /// Adapt compression algorithm and parameters based on network conditions
    pub fn adapt_compression(&mut self, conditions: &NetworkConditions) -> Result<(), TensorError> {
        let bandwidth_mbps = conditions.available_bandwidth;
        let latency_ms = conditions.current_latency.as_millis() as f64;
        let congestion = conditions.congestion_level;

        // Choose compression algorithm based on network conditions
        let new_algorithm = if bandwidth_mbps < 100.0 || congestion > 0.7 {
            // Low bandwidth or high congestion - use aggressive compression
            CompressionAlgorithm::Quantization { bits: 4 }
        } else if bandwidth_mbps < 500.0 || congestion > 0.4 {
            // Medium bandwidth - use moderate compression
            CompressionAlgorithm::Quantization { bits: 8 }
        } else if latency_ms > 100.0 {
            // High latency - use TopK for sparsity
            CompressionAlgorithm::TopK { k: 1000 }
        } else {
            // Good conditions - use light compression or error feedback
            CompressionAlgorithm::ErrorFeedback {
                compression_ratio: 0.8,
            }
        };

        // Update the current algorithm if it's different
        if self.current_algorithm != new_algorithm {
            self.current_algorithm = new_algorithm.clone();

            // Register new algorithm if not already present
            if !self.algorithms.contains_key(&new_algorithm) {
                let codec: Box<dyn CompressionCodec> = match &new_algorithm {
                    CompressionAlgorithm::Quantization { bits } => {
                        Box::new(QuantizationCodec::new(*bits))
                    }
                    CompressionAlgorithm::TopK { k } => Box::new(TopKCodec::new(*k)),
                    CompressionAlgorithm::ErrorFeedback { compression_ratio } => {
                        Box::new(ErrorFeedbackCodec::new(*compression_ratio))
                    }
                    _ => Box::new(QuantizationCodec::new(8)), // Default fallback
                };
                self.algorithms.insert(new_algorithm.clone(), codec);
            }
        }

        // Update adaptive configuration
        self.adaptive_config.target_bandwidth_utilization = if congestion > 0.5 {
            0.6 // Reduce target utilization under congestion
        } else {
            0.8 // Normal target utilization
        };

        Ok(())
    }

    /// Compress data using the current algorithm
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>, TensorError> {
        if let Some(codec) = self.algorithms.get(&self.current_algorithm) {
            codec.compress(data)
        } else {
            Err(TensorError::invalid_argument(
                "Compression algorithm not found".to_string(),
            ))
        }
    }

    /// Decompress data using the specified algorithm
    pub fn decompress(
        &self,
        data: &[u8],
        algorithm: &CompressionAlgorithm,
    ) -> Result<Vec<u8>, TensorError> {
        if let Some(codec) = self.algorithms.get(algorithm) {
            codec.decompress(data)
        } else {
            Err(TensorError::invalid_argument(
                "Compression algorithm not found".to_string(),
            ))
        }
    }
}

impl QuantizationCodec {
    fn new(bits: u8) -> Self {
        QuantizationCodec { bits }
    }
}

impl CompressionCodec for QuantizationCodec {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, TensorError> {
        // Implement quantization compression
        // Simple quantization: reduce precision by keeping only most significant bits
        let quantization_factor = 8 - self.bits;
        let compressed: Vec<u8> = data
            .iter()
            .map(|&byte| byte >> quantization_factor << quantization_factor)
            .collect();
        Ok(compressed)
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, TensorError> {
        // For quantization, decompression is just returning the data as-is
        // (precision is already lost in compression)
        Ok(data.to_vec())
    }

    fn compression_ratio(&self) -> f64 {
        32.0 / self.bits as f64
    }

    fn compute_cost(&self) -> f64 {
        0.1 // Low CPU cost
    }
}

impl TopKCodec {
    fn new(k: usize) -> Self {
        TopKCodec { k }
    }
}

impl CompressionCodec for TopKCodec {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, TensorError> {
        // TopK compression: keep only the K largest values
        if data.len() <= self.k {
            return Ok(data.to_vec());
        }

        // Convert bytes to values, find top-k, and create sparse representation
        let mut indexed_values: Vec<(usize, u8)> =
            data.iter().enumerate().map(|(i, &val)| (i, val)).collect();

        // Sort by value (descending)
        indexed_values.sort_by(|a, b| b.1.cmp(&a.1));

        // Keep only top-k
        indexed_values.truncate(self.k);

        // Create sparse representation: [num_elements, index1, value1, index2, value2, ...]
        let mut compressed = Vec::with_capacity(1 + self.k * 3);
        compressed.push(self.k as u8);

        for (index, value) in indexed_values {
            // Store index as 2 bytes (little endian) and value as 1 byte
            compressed.extend_from_slice(&(index as u16).to_le_bytes());
            compressed.push(value);
        }

        Ok(compressed)
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, TensorError> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let num_elements = data[0] as usize;
        let mut result = vec![0u8; self.k.max(1000)]; // Assume reasonable max size

        let mut pos = 1;
        for _ in 0..num_elements {
            if pos + 3 > data.len() {
                break;
            }

            let index = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
            let value = data[pos + 2];

            if index < result.len() {
                result[index] = value;
            }

            pos += 3;
        }

        Ok(result)
    }

    fn compression_ratio(&self) -> f64 {
        // Depends on the sparsity, but roughly self.k out of original data size
        10.0 // Approximate compression ratio
    }

    fn compute_cost(&self) -> f64 {
        0.5 // Medium CPU cost due to sorting
    }
}

impl ErrorFeedbackCodec {
    fn new(compression_ratio: f64) -> Self {
        ErrorFeedbackCodec { compression_ratio }
    }
}

impl CompressionCodec for ErrorFeedbackCodec {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, TensorError> {
        // Error feedback compression: maintain error state for better quality
        // Simplified implementation - would normally maintain error buffers
        let compressed_size = (data.len() as f64 * self.compression_ratio) as usize;
        if compressed_size >= data.len() {
            return Ok(data.to_vec());
        }

        // Simple subsampling with error feedback concept
        let step = data.len() / compressed_size;
        let compressed: Vec<u8> = data.iter().step_by(step.max(1)).copied().collect();

        Ok(compressed)
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, TensorError> {
        // Error feedback decompression: reconstruct with error correction
        // Simplified implementation - would normally apply error feedback
        Ok(data.to_vec())
    }

    fn compression_ratio(&self) -> f64 {
        1.0 / self.compression_ratio
    }

    fn compute_cost(&self) -> f64 {
        0.3 // Medium CPU cost
    }
}

// Implement PartialEq and Eq for CompressionAlgorithm to support HashMap usage
impl PartialEq for CompressionAlgorithm {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                CompressionAlgorithm::Quantization { bits: a },
                CompressionAlgorithm::Quantization { bits: b },
            ) => a == b,
            (
                CompressionAlgorithm::Sparsification { threshold: a },
                CompressionAlgorithm::Sparsification { threshold: b },
            ) => a == b,
            (CompressionAlgorithm::TopK { k: a }, CompressionAlgorithm::TopK { k: b }) => a == b,
            (
                CompressionAlgorithm::ErrorFeedback {
                    compression_ratio: a,
                },
                CompressionAlgorithm::ErrorFeedback {
                    compression_ratio: b,
                },
            ) => a == b,
            (CompressionAlgorithm::Adaptive, CompressionAlgorithm::Adaptive) => true,
            _ => false,
        }
    }
}

impl Eq for CompressionAlgorithm {}

impl std::hash::Hash for CompressionAlgorithm {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            CompressionAlgorithm::Quantization { bits } => {
                "quantization".hash(state);
                bits.hash(state);
            }
            CompressionAlgorithm::Sparsification { threshold } => {
                "sparsification".hash(state);
                threshold.to_bits().hash(state);
            }
            CompressionAlgorithm::TopK { k } => {
                "topk".hash(state);
                k.hash(state);
            }
            CompressionAlgorithm::ErrorFeedback { compression_ratio } => {
                "error_feedback".hash(state);
                compression_ratio.to_bits().hash(state);
            }
            CompressionAlgorithm::Adaptive => {
                "adaptive".hash(state);
            }
        }
    }
}
