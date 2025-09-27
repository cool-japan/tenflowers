//! Compression utilities for model serialization

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use tenflowers_core::{Result, TensorError};

/// Compression algorithms supported
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// Zstandard compression (fast and good compression)
    Zstd,
    /// Gzip compression (widely compatible)
    Gzip,
    /// LZ4 compression (very fast)
    Lz4,
}

impl Default for CompressionAlgorithm {
    fn default() -> Self {
        Self::Zstd
    }
}

/// Compression metadata
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct CompressionInfo {
    /// Algorithm used
    pub algorithm: CompressionAlgorithm,
    /// Compression level (0-9)
    pub level: u8,
    /// Original size before compression
    pub original_size: usize,
    /// Compressed size
    pub compressed_size: usize,
    /// Compression ratio
    pub compression_ratio: f32,
}

impl CompressionInfo {
    pub fn new(
        algorithm: CompressionAlgorithm,
        level: u8,
        original_size: usize,
        compressed_size: usize,
    ) -> Self {
        let compression_ratio = if original_size > 0 {
            compressed_size as f32 / original_size as f32
        } else {
            1.0
        };

        Self {
            algorithm,
            level,
            original_size,
            compressed_size,
            compression_ratio,
        }
    }
}

/// Compression utilities
pub struct Compressor;

impl Compressor {
    /// Compress data using specified algorithm
    pub fn compress(data: &[u8], algorithm: CompressionAlgorithm, level: u8) -> Result<Vec<u8>> {
        match algorithm {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            CompressionAlgorithm::Zstd => Self::compress_zstd(data, level),
            CompressionAlgorithm::Gzip => Self::compress_gzip(data, level),
            CompressionAlgorithm::Lz4 => Self::compress_lz4(data, level),
        }
    }

    /// Decompress data using specified algorithm
    pub fn decompress(data: &[u8], algorithm: CompressionAlgorithm) -> Result<Vec<u8>> {
        match algorithm {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            CompressionAlgorithm::Zstd => Self::decompress_zstd(data),
            CompressionAlgorithm::Gzip => Self::decompress_gzip(data),
            CompressionAlgorithm::Lz4 => Self::decompress_lz4(data),
        }
    }

    /// Compress with zstd
    fn compress_zstd(data: &[u8], level: u8) -> Result<Vec<u8>> {
        #[cfg(feature = "zstd")]
        {
            zstd::encode_all(data, level as i32).map_err(|e| {
                TensorError::serialization_error_simple(format!("Zstd compression failed: {}", e))
            })
        }
        #[cfg(not(feature = "zstd"))]
        {
            let _ = (data, level);
            Err(TensorError::serialization_error_simple(
                "Zstd compression not available".to_string(),
            ))
        }
    }

    /// Decompress with zstd
    fn decompress_zstd(data: &[u8]) -> Result<Vec<u8>> {
        #[cfg(feature = "zstd")]
        {
            zstd::decode_all(data).map_err(|e| {
                TensorError::serialization_error_simple(format!("Zstd decompression failed: {}", e))
            })
        }
        #[cfg(not(feature = "zstd"))]
        {
            let _ = data;
            Err(TensorError::serialization_error_simple(
                "Zstd decompression not available".to_string(),
            ))
        }
    }

    /// Compress with gzip
    fn compress_gzip(data: &[u8], level: u8) -> Result<Vec<u8>> {
        #[cfg(feature = "gzip")]
        {
            use flate2::write::GzEncoder;
            use flate2::Compression;

            let mut encoder = GzEncoder::new(Vec::new(), Compression::new(level as u32));
            encoder.write_all(data).map_err(|e| {
                TensorError::serialization_error_simple(format!("Gzip compression failed: {}", e))
            })?;
            encoder.finish().map_err(|e| {
                TensorError::serialization_error_simple(format!("Gzip compression failed: {}", e))
            })
        }
        #[cfg(not(feature = "gzip"))]
        {
            let _ = (data, level);
            Err(TensorError::serialization_error_simple(
                "Gzip compression not available".to_string(),
            ))
        }
    }

    /// Decompress with gzip
    fn decompress_gzip(data: &[u8]) -> Result<Vec<u8>> {
        #[cfg(feature = "gzip")]
        {
            use flate2::read::GzDecoder;

            let mut decoder = GzDecoder::new(data);
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed).map_err(|e| {
                TensorError::serialization_error_simple(format!("Gzip decompression failed: {}", e))
            })?;
            Ok(decompressed)
        }
        #[cfg(not(feature = "gzip"))]
        {
            let _ = data;
            Err(TensorError::serialization_error_simple(
                "Gzip decompression not available".to_string(),
            ))
        }
    }

    /// Compress with lz4
    fn compress_lz4(data: &[u8], level: u8) -> Result<Vec<u8>> {
        #[cfg(feature = "lz4")]
        {
            use lz4::block::{compress, CompressionMode};

            let mode = match level {
                0..=3 => CompressionMode::FAST(level as i32),
                4..=9 => CompressionMode::HIGHCOMPRESSION,
                _ => CompressionMode::DEFAULT,
            };

            compress(data, Some(mode), true).map_err(|e| {
                TensorError::serialization_error_simple(format!("LZ4 compression failed: {}", e))
            })
        }
        #[cfg(not(feature = "lz4"))]
        {
            let _ = (data, level);
            Err(TensorError::serialization_error_simple(
                "LZ4 compression not available".to_string(),
            ))
        }
    }

    /// Decompress with lz4
    fn decompress_lz4(data: &[u8]) -> Result<Vec<u8>> {
        #[cfg(feature = "lz4")]
        {
            use lz4::block::decompress;

            // For LZ4, we need to know the original size
            // In a real implementation, this would be stored in the header
            // For now, we'll use a reasonable maximum size
            let max_size = data.len() * 4; // Assume max 4x compression ratio

            decompress(data, Some(max_size as i32)).map_err(|e| {
                TensorError::serialization_error_simple(format!("LZ4 decompression failed: {}", e))
            })
        }
        #[cfg(not(feature = "lz4"))]
        {
            let _ = data;
            Err(TensorError::serialization_error_simple(
                "LZ4 decompression not available".to_string(),
            ))
        }
    }
}

/// Compression analysis utilities
pub struct CompressionAnalyzer;

impl CompressionAnalyzer {
    /// Analyze compression potential for data
    pub fn analyze_compression_potential(data: &[u8]) -> CompressionAnalysis {
        let mut byte_counts = [0u32; 256];
        for &byte in data {
            byte_counts[byte as usize] += 1;
        }

        // Calculate entropy
        let data_len = data.len() as f64;
        let entropy = byte_counts
            .iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f64 / data_len;
                -p * p.log2()
            })
            .sum::<f64>();

        // Estimate compression ratio based on entropy
        let theoretical_compression = entropy / 8.0;

        // Find most common patterns
        let mut patterns = std::collections::HashMap::new();
        if data.len() >= 4 {
            for window in data.windows(4) {
                *patterns.entry(window).or_insert(0) += 1;
            }
        }

        let max_pattern_count = patterns.values().max().cloned().unwrap_or(0);
        let pattern_ratio = max_pattern_count as f64 / data.len() as f64;

        CompressionAnalysis {
            entropy,
            theoretical_compression_ratio: theoretical_compression,
            pattern_repetition_ratio: pattern_ratio,
            recommended_algorithm: Self::recommend_algorithm(entropy, pattern_ratio),
        }
    }

    /// Recommend compression algorithm based on data characteristics
    fn recommend_algorithm(entropy: f64, pattern_ratio: f64) -> CompressionAlgorithm {
        // High entropy data doesn't compress well
        if entropy > 7.5 {
            return CompressionAlgorithm::None;
        }

        // High pattern repetition favors LZ4
        if pattern_ratio > 0.1 {
            return CompressionAlgorithm::Lz4;
        }

        // Medium entropy with some patterns favors Zstd
        if entropy > 4.0 && entropy < 7.0 {
            return CompressionAlgorithm::Zstd;
        }

        // Low entropy favors Gzip
        CompressionAlgorithm::Gzip
    }
}

/// Compression analysis result
#[derive(Debug)]
pub struct CompressionAnalysis {
    /// Data entropy (0-8 bits)
    pub entropy: f64,
    /// Theoretical compression ratio
    pub theoretical_compression_ratio: f64,
    /// Pattern repetition ratio
    pub pattern_repetition_ratio: f64,
    /// Recommended algorithm
    pub recommended_algorithm: CompressionAlgorithm,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_info() {
        let info = CompressionInfo::new(CompressionAlgorithm::Zstd, 3, 1000, 500);

        assert_eq!(info.algorithm, CompressionAlgorithm::Zstd);
        assert_eq!(info.level, 3);
        assert_eq!(info.original_size, 1000);
        assert_eq!(info.compressed_size, 500);
        assert_eq!(info.compression_ratio, 0.5);
    }

    #[test]
    fn test_no_compression() {
        let data = b"Hello, World!";
        let compressed = Compressor::compress(data, CompressionAlgorithm::None, 0).unwrap();
        let decompressed = Compressor::decompress(&compressed, CompressionAlgorithm::None).unwrap();

        assert_eq!(data, compressed.as_slice());
        assert_eq!(data, decompressed.as_slice());
    }

    #[test]
    fn test_compression_analysis() {
        // Test with repetitive data
        let repetitive_data = b"AAAAAAAAAA";
        let analysis = CompressionAnalyzer::analyze_compression_potential(repetitive_data);

        assert!(analysis.entropy < 1.0); // Low entropy
        assert!(analysis.pattern_repetition_ratio > 0.0);

        // Test with random data
        let random_data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let analysis = CompressionAnalyzer::analyze_compression_potential(&random_data);

        assert!(analysis.entropy > 7.0); // High entropy
        assert_eq!(analysis.recommended_algorithm, CompressionAlgorithm::None);
    }
}
