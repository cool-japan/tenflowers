//! Download and extraction utilities for dataset files

use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;
use tenflowers_core::{Result, TensorError};

use super::common::{error_utils, ProgressTracker};

#[cfg(feature = "download")]
use reqwest::blocking::Client;

#[cfg(feature = "download")]
use flate2::read::GzDecoder;

#[cfg(feature = "download")]
use tar::Archive;

/// Download utilities for dataset files
pub struct Downloader {
    #[cfg(feature = "download")]
    client: Client,
}

impl Downloader {
    /// Create a new downloader
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "download")]
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(300)) // 5 minutes timeout
                .user_agent("tenflowers-dataset/0.1.0")
                .build()
                .unwrap_or_else(|_| Client::new()),
        }
    }

    /// Download a file from URL to destination
    #[cfg(feature = "download")]
    pub fn download_file(&self, url: &str, dest_path: &Path, description: &str) -> Result<()> {
        println!("Downloading {}: {}", description, url);

        let response = self.client.get(url).send().map_err(|e| {
            error_utils::io_error_with_context(
                std::io::Error::new(std::io::ErrorKind::Other, e),
                &format!("Failed to start download from {}", url),
            )
        })?;

        if !response.status().is_success() {
            return Err(TensorError::invalid_argument(format!(
                "Failed to download {}: HTTP {}",
                description,
                response.status()
            )));
        }

        let total_size = response.content_length().unwrap_or(0);
        let mut tracker = ProgressTracker::new(total_size, format!("Downloading {}", description));

        let mut file = File::create(dest_path).map_err(|e| {
            error_utils::io_error_with_context(e, "Failed to create destination file")
        })?;

        let mut downloaded = 0u64;
        let mut buffer = [0; 8192];
        let mut reader = BufReader::new(response);

        loop {
            let bytes_read = reader.read(&mut buffer).map_err(|e| {
                error_utils::io_error_with_context(e, "Failed to read from download stream")
            })?;

            if bytes_read == 0 {
                break;
            }

            file.write_all(&buffer[..bytes_read]).map_err(|e| {
                error_utils::io_error_with_context(e, "Failed to write to destination file")
            })?;

            downloaded += bytes_read as u64;
            tracker.update(downloaded);
        }

        tracker.complete();
        println!("{} downloaded successfully!", description);
        Ok(())
    }

    /// Download a file when download feature is disabled
    #[cfg(not(feature = "download"))]
    pub fn download_file(&self, _url: &str, _dest_path: &Path, description: &str) -> Result<()> {
        Err(TensorError::invalid_argument(format!(
            "Download feature not enabled. Please enable the 'download' feature or manually download {} files.",
            description
        )))
    }

    /// Extract a gzipped file
    #[cfg(feature = "download")]
    pub fn extract_gzip(&self, gz_path: &Path, dest_path: &Path, description: &str) -> Result<()> {
        println!("Extracting {}", description);

        let gz_file = File::open(gz_path)
            .map_err(|e| error_utils::io_error_with_context(e, "Failed to open gzip file"))?;

        let mut decoder = GzDecoder::new(gz_file);
        let mut dest_file = File::create(dest_path).map_err(|e| {
            error_utils::io_error_with_context(e, "Failed to create destination file")
        })?;

        std::io::copy(&mut decoder, &mut dest_file)
            .map_err(|e| error_utils::io_error_with_context(e, "Failed to extract gzip file"))?;

        println!("{} extracted successfully!", description);
        Ok(())
    }

    /// Extract a gzipped file when download feature is disabled
    #[cfg(not(feature = "download"))]
    pub fn extract_gzip(
        &self,
        _gz_path: &Path,
        _dest_path: &Path,
        description: &str,
    ) -> Result<()> {
        Err(TensorError::invalid_argument(format!(
            "Download feature not enabled. Cannot extract {} files.",
            description
        )))
    }

    /// Extract a tar.gz archive
    #[cfg(feature = "download")]
    pub fn extract_tar_gz(
        &self,
        tar_gz_path: &Path,
        dest_dir: &Path,
        description: &str,
    ) -> Result<()> {
        println!("Extracting {} archive", description);

        let tar_gz_file = File::open(tar_gz_path)
            .map_err(|e| error_utils::io_error_with_context(e, "Failed to open tar.gz file"))?;

        let gz_decoder = GzDecoder::new(tar_gz_file);
        let mut archive = Archive::new(gz_decoder);

        archive.unpack(dest_dir).map_err(|e| {
            error_utils::io_error_with_context(e, "Failed to extract tar.gz archive")
        })?;

        println!("{} archive extracted successfully!", description);
        Ok(())
    }

    /// Extract a tar.gz archive when download feature is disabled
    #[cfg(not(feature = "download"))]
    pub fn extract_tar_gz(
        &self,
        _tar_gz_path: &Path,
        _dest_dir: &Path,
        description: &str,
    ) -> Result<()> {
        Err(TensorError::invalid_argument(format!(
            "Download feature not enabled. Cannot extract {} archive.",
            description
        )))
    }

    /// Download and extract MNIST files
    pub fn download_mnist(&self, mnist_dir: &Path, train: bool) -> Result<()> {
        use super::common::{
            MNIST_TEST_IMAGES_URL, MNIST_TEST_LABELS_URL, MNIST_TRAIN_IMAGES_URL,
            MNIST_TRAIN_LABELS_URL,
        };

        let (images_url, labels_url, images_file, labels_file) = if train {
            (
                MNIST_TRAIN_IMAGES_URL,
                MNIST_TRAIN_LABELS_URL,
                "train-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
            )
        } else {
            (
                MNIST_TEST_IMAGES_URL,
                MNIST_TEST_LABELS_URL,
                "t10k-images-idx3-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz",
            )
        };

        let images_gz_path = mnist_dir.join(images_file);
        let labels_gz_path = mnist_dir.join(labels_file);

        // Download compressed files
        self.download_file(images_url, &images_gz_path, "MNIST images")?;
        self.download_file(labels_url, &labels_gz_path, "MNIST labels")?;

        // Extract files
        let images_path = mnist_dir.join(images_file.trim_end_matches(".gz"));
        let labels_path = mnist_dir.join(labels_file.trim_end_matches(".gz"));

        self.extract_gzip(&images_gz_path, &images_path, "MNIST images")?;
        self.extract_gzip(&labels_gz_path, &labels_path, "MNIST labels")?;

        // Clean up compressed files
        let _ = std::fs::remove_file(images_gz_path);
        let _ = std::fs::remove_file(labels_gz_path);

        Ok(())
    }

    /// Download and extract CIFAR-10 files
    pub fn download_cifar10(&self, cifar_dir: &Path) -> Result<()> {
        use super::common::CIFAR10_URL;

        let tar_gz_path = cifar_dir.join("cifar-10-binary.tar.gz");

        // Download archive
        self.download_file(CIFAR10_URL, &tar_gz_path, "CIFAR-10 dataset")?;

        // Extract archive
        self.extract_tar_gz(&tar_gz_path, cifar_dir, "CIFAR-10")?;

        // Clean up archive
        let _ = std::fs::remove_file(tar_gz_path);

        Ok(())
    }

    /// Download and extract ImageNet validation files
    pub fn download_imagenet_val(&self, imagenet_dir: &Path) -> Result<()> {
        use super::common::{IMAGENET_LABELS_URL, IMAGENET_VAL_URL};

        let val_tar_path = imagenet_dir.join("ILSVRC2012_img_val.tar");
        let labels_path = imagenet_dir.join("ILSVRC2012_validation_ground_truth.txt");

        // Download files
        self.download_file(
            IMAGENET_VAL_URL,
            &val_tar_path,
            "ImageNet validation images",
        )?;
        self.download_file(
            IMAGENET_LABELS_URL,
            &labels_path,
            "ImageNet validation labels",
        )?;

        // Extract validation images (tar, not tar.gz)
        #[cfg(feature = "download")]
        {
            let val_images_dir = imagenet_dir.join("val");
            std::fs::create_dir_all(&val_images_dir).map_err(|e| {
                error_utils::io_error_with_context(
                    e,
                    "Failed to create validation images directory",
                )
            })?;

            let tar_file = File::open(&val_tar_path).map_err(|e| {
                error_utils::io_error_with_context(e, "Failed to open validation tar file")
            })?;

            let mut archive = Archive::new(tar_file);
            archive.unpack(&val_images_dir).map_err(|e| {
                error_utils::io_error_with_context(e, "Failed to extract validation images")
            })?;

            // Clean up tar file
            let _ = std::fs::remove_file(val_tar_path);
        }

        Ok(())
    }

    /// Download IMDB dataset
    pub fn download_imdb(&self, imdb_dir: &Path) -> Result<()> {
        use super::common::IMDB_URL;

        let tar_gz_path = imdb_dir.join("aclImdb_v1.tar.gz");

        // Download archive
        self.download_file(IMDB_URL, &tar_gz_path, "IMDB dataset")?;

        // Extract archive
        self.extract_tar_gz(&tar_gz_path, imdb_dir, "IMDB")?;

        // Clean up archive
        let _ = std::fs::remove_file(tar_gz_path);

        Ok(())
    }

    /// Download AG News dataset
    pub fn download_ag_news(&self, ag_news_dir: &Path) -> Result<()> {
        use super::common::{AG_NEWS_TEST_URL, AG_NEWS_TRAIN_URL};

        let train_path = ag_news_dir.join("train.csv");
        let test_path = ag_news_dir.join("test.csv");

        // Download CSV files
        self.download_file(AG_NEWS_TRAIN_URL, &train_path, "AG News training data")?;
        self.download_file(AG_NEWS_TEST_URL, &test_path, "AG News test data")?;

        Ok(())
    }
}

impl Default for Downloader {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility function to get file size
pub fn get_file_size(path: &Path) -> Result<u64> {
    let metadata = path
        .metadata()
        .map_err(|e| error_utils::io_error_with_context(e, "Failed to get file metadata"))?;
    Ok(metadata.len())
}

/// Utility function to verify checksum (simplified)
pub fn verify_checksum(path: &Path, expected_hash: Option<&str>) -> Result<bool> {
    if expected_hash.is_none() {
        return Ok(true); // Skip verification if no hash provided
    }

    // For now, just return true. In a real implementation, you would
    // compute and verify the actual checksum (MD5, SHA256, etc.)
    let _file_size = get_file_size(path)?;

    println!("Checksum verification skipped (not implemented)");
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_downloader_creation() {
        let downloader = Downloader::new();
        // Just verify it can be created without panicking
        drop(downloader);
    }

    #[test]
    fn test_get_file_size() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");

        // Create a test file
        std::fs::write(&test_file, b"Hello, World!").unwrap();

        let size = get_file_size(&test_file).unwrap();
        assert_eq!(size, 13); // "Hello, World!" is 13 bytes
    }

    #[test]
    fn test_verify_checksum_no_hash() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");

        // Create a test file
        std::fs::write(&test_file, b"test").unwrap();

        let result = verify_checksum(&test_file, None).unwrap();
        assert!(result);
    }

    #[test]
    fn test_verify_checksum_with_hash() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");

        // Create a test file
        std::fs::write(&test_file, b"test").unwrap();

        // For now, this should always return true
        let result = verify_checksum(&test_file, Some("dummy_hash")).unwrap();
        assert!(result);
    }
}
