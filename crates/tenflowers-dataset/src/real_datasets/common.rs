//! Common utilities and constants for real dataset loaders

use std::path::PathBuf;

/// URLs for MNIST dataset files
pub const MNIST_TRAIN_IMAGES_URL: &str =
    "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
pub const MNIST_TRAIN_LABELS_URL: &str =
    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
pub const MNIST_TEST_IMAGES_URL: &str =
    "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz";
pub const MNIST_TEST_LABELS_URL: &str =
    "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz";

/// URLs for CIFAR-10 dataset files
pub const CIFAR10_URL: &str = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";

/// URLs for ImageNet dataset files (validation set - smaller for testing)
pub const IMAGENET_VAL_URL: &str = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar";
pub const IMAGENET_LABELS_URL: &str =
    "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_validation_ground_truth.txt";

/// URLs for IMDB dataset files
pub const IMDB_URL: &str = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";

/// URLs for AG News dataset files
pub const AG_NEWS_TRAIN_URL: &str =
    "https://github.com/mhjabreel/CharCnn_Keras/raw/master/data/ag_news_csv/train.csv";
pub const AG_NEWS_TEST_URL: &str =
    "https://github.com/mhjabreel/CharCnn_Keras/raw/master/data/ag_news_csv/test.csv";

/// Common dataset configuration trait
pub trait DatasetConfig {
    /// Get the root directory for data storage
    fn root(&self) -> &PathBuf;

    /// Whether to use training set (true) or test set (false)
    fn is_train(&self) -> bool;

    /// Whether to download if not found
    fn should_download(&self) -> bool;
}

/// Utility functions for byte order conversion
pub mod byte_utils {
    /// Convert 4 bytes to u32 (big-endian)
    pub fn bytes_to_u32_be(bytes: &[u8]) -> u32 {
        ((bytes[0] as u32) << 24)
            | ((bytes[1] as u32) << 16)
            | ((bytes[2] as u32) << 8)
            | (bytes[3] as u32)
    }

    /// Convert 4 bytes to u32 (little-endian)
    pub fn bytes_to_u32_le(bytes: &[u8]) -> u32 {
        (bytes[0] as u32)
            | ((bytes[1] as u32) << 8)
            | ((bytes[2] as u32) << 16)
            | ((bytes[3] as u32) << 24)
    }
}

/// Common validation functions
pub mod validation {
    use std::path::Path;
    use tenflowers_core::{Result, TensorError};

    /// Validate that a file exists
    pub fn validate_file_exists(path: &Path, description: &str) -> Result<()> {
        if !path.exists() {
            return Err(TensorError::invalid_argument(format!(
                "{} file not found: {}",
                description,
                path.display()
            )));
        }
        Ok(())
    }

    /// Validate directory exists or create it
    pub fn ensure_directory_exists(path: &Path, description: &str) -> Result<()> {
        if !path.exists() {
            std::fs::create_dir_all(path).map_err(|e| {
                TensorError::invalid_argument(format!(
                    "Failed to create {} directory: {}",
                    description, e
                ))
            })?;
        }
        Ok(())
    }

    /// Validate file size matches expected
    pub fn validate_file_size(
        path: &Path,
        expected_size: Option<u64>,
        description: &str,
    ) -> Result<()> {
        if let Some(expected) = expected_size {
            let metadata = path.metadata().map_err(|e| {
                TensorError::invalid_argument(format!(
                    "Failed to get {} file metadata: {}",
                    description, e
                ))
            })?;

            if metadata.len() != expected {
                return Err(TensorError::invalid_argument(format!(
                    "{} file has unexpected size: {} bytes, expected {} bytes",
                    description,
                    metadata.len(),
                    expected
                )));
            }
        }
        Ok(())
    }
}

/// Progress tracking for downloads and processing
pub struct ProgressTracker {
    total: u64,
    current: u64,
    description: String,
}

impl ProgressTracker {
    /// Create a new progress tracker
    pub fn new(total: u64, description: String) -> Self {
        Self {
            total,
            current: 0,
            description,
        }
    }

    /// Update progress
    pub fn update(&mut self, current: u64) {
        self.current = current;
        let percentage = if self.total > 0 {
            (self.current as f64 / self.total as f64) * 100.0
        } else {
            0.0
        };

        if self.current % (self.total / 20 + 1) == 0 || self.current == self.total {
            println!(
                "{}: {:.1}% ({}/{})",
                self.description, percentage, self.current, self.total
            );
        }
    }

    /// Mark as completed
    pub fn complete(&mut self) {
        self.current = self.total;
        println!("{}: 100% - Completed!", self.description);
    }
}

/// Error handling utilities
pub mod error_utils {
    use tenflowers_core::TensorError;

    /// Convert IO error to TensorError with context
    pub fn io_error_with_context(error: std::io::Error, context: &str) -> TensorError {
        TensorError::invalid_argument(format!("{}: {}", context, error))
    }

    /// Create a not found error for datasets
    pub fn dataset_not_found_error(dataset_name: &str, suggestion: &str) -> TensorError {
        TensorError::invalid_argument(format!("{} files not found. {}", dataset_name, suggestion))
    }

    /// Create an invalid format error
    pub fn invalid_format_error(dataset_name: &str, details: &str) -> TensorError {
        TensorError::invalid_argument(format!("Invalid {} format: {}", dataset_name, details))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_u32_be() {
        let bytes = [0x12, 0x34, 0x56, 0x78];
        assert_eq!(byte_utils::bytes_to_u32_be(&bytes), 0x12345678);
    }

    #[test]
    fn test_bytes_to_u32_le() {
        let bytes = [0x78, 0x56, 0x34, 0x12];
        assert_eq!(byte_utils::bytes_to_u32_le(&bytes), 0x12345678);
    }

    #[test]
    fn test_progress_tracker() {
        let mut tracker = ProgressTracker::new(100, "Test".to_string());
        tracker.update(50);
        tracker.complete();
        assert_eq!(tracker.current, tracker.total);
    }
}
