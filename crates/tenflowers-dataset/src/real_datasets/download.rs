//! Download and extraction utilities for dataset files

use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;
use tenflowers_core::{Result, TensorError};

use super::common::{error_utils, ProgressTracker};

#[cfg(feature = "download")]
use reqwest::blocking::Client;

#[cfg(feature = "download")]
use oxiarc_archive::{GzipReader, TarReader};

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
                .user_agent(concat!("tenflowers-dataset/", env!("CARGO_PKG_VERSION")))
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

        let mut gzip_reader = GzipReader::new(gz_file).map_err(|e| {
            error_utils::io_error_with_context(
                std::io::Error::new(std::io::ErrorKind::Other, format!("{}", e)),
                "Failed to open gzip file",
            )
        })?;
        let decompressed = gzip_reader.decompress().map_err(|e| {
            error_utils::io_error_with_context(
                std::io::Error::new(std::io::ErrorKind::Other, format!("{}", e)),
                "Failed to extract gzip file",
            )
        })?;
        let mut dest_file = File::create(dest_path).map_err(|e| {
            error_utils::io_error_with_context(e, "Failed to create destination file")
        })?;
        dest_file
            .write_all(&decompressed)
            .map_err(|e| error_utils::io_error_with_context(e, "Failed to write extracted file"))?;

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

        let mut gzip_reader = GzipReader::new(tar_gz_file).map_err(|e| {
            error_utils::io_error_with_context(
                std::io::Error::new(std::io::ErrorKind::Other, format!("{}", e)),
                "Failed to open gzip file",
            )
        })?;
        let decompressed = gzip_reader.decompress().map_err(|e| {
            error_utils::io_error_with_context(
                std::io::Error::new(std::io::ErrorKind::Other, format!("{}", e)),
                "Failed to decompress tar.gz file",
            )
        })?;
        let mut tar_reader = TarReader::new(std::io::Cursor::new(decompressed)).map_err(|e| {
            error_utils::io_error_with_context(
                std::io::Error::new(std::io::ErrorKind::Other, format!("{}", e)),
                "Failed to parse tar archive",
            )
        })?;
        // Clone entries first to avoid borrow conflict (Entry: Clone)
        let entries = tar_reader.entries().to_vec();
        for entry in &entries {
            let dest_path = dest_dir.join(&entry.name);
            if entry.name.ends_with('/') {
                std::fs::create_dir_all(&dest_path).map_err(|e| {
                    error_utils::io_error_with_context(e, "Failed to create directory")
                })?;
            } else {
                if let Some(parent) = dest_path.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        error_utils::io_error_with_context(e, "Failed to create parent directory")
                    })?;
                }
                let data = tar_reader.extract_to_vec(entry).map_err(|e| {
                    error_utils::io_error_with_context(
                        std::io::Error::new(std::io::ErrorKind::Other, format!("{}", e)),
                        "Failed to extract tar entry",
                    )
                })?;
                std::fs::write(&dest_path, &data).map_err(|e| {
                    error_utils::io_error_with_context(e, "Failed to write extracted file")
                })?;
            }
        }

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

            let mut tar_reader = TarReader::new(tar_file).map_err(|e| {
                error_utils::io_error_with_context(
                    std::io::Error::new(std::io::ErrorKind::Other, format!("{}", e)),
                    "Failed to parse tar archive",
                )
            })?;
            // Clone entries first to avoid borrow conflict (Entry: Clone)
            let entries = tar_reader.entries().to_vec();
            for entry in &entries {
                let dest_path = val_images_dir.join(&entry.name);
                if entry.name.ends_with('/') {
                    std::fs::create_dir_all(&dest_path).map_err(|e| {
                        error_utils::io_error_with_context(e, "Failed to create directory")
                    })?;
                } else {
                    if let Some(parent) = dest_path.parent() {
                        std::fs::create_dir_all(parent).map_err(|e| {
                            error_utils::io_error_with_context(
                                e,
                                "Failed to create parent directory",
                            )
                        })?;
                    }
                    let data = tar_reader.extract_to_vec(entry).map_err(|e| {
                        error_utils::io_error_with_context(
                            std::io::Error::new(std::io::ErrorKind::Other, format!("{}", e)),
                            "Failed to extract tar entry",
                        )
                    })?;
                    std::fs::write(&dest_path, &data).map_err(|e| {
                        error_utils::io_error_with_context(e, "Failed to write extracted file")
                    })?;
                }
            }

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

/// Verify the integrity of a downloaded file against an expected checksum.
///
/// The expected checksum may optionally carry an algorithm prefix of the form
/// `"<algo>:<hex>"` (for example `"sha256:abc123..."`). When no prefix is
/// present, the algorithm is inferred from the hexadecimal digest length:
/// 64 hex characters is treated as SHA-256, and 8 hex characters is treated
/// as CRC-32.
///
/// The function computes the *real* digest of the file's bytes and compares it
/// (case-insensitively) against the expected value:
/// - returns `Ok(true)` when the digest matches,
/// - returns `Ok(false)` when the digest does not match,
/// - returns `Ok(true)` when `expected_hash` is `None` (nothing to verify),
/// - returns an honest `Err(...)` when the algorithm cannot be determined or is
///   not supported, so callers are never told an unverifiable file is valid.
///
/// Only pure-Rust hashing is used (the `sha2` and `crc32fast` crates), in
/// keeping with the project's pure-Rust dependency policy. CRC-32 support
/// relies on the optional `crc32fast` dependency, which is wired to this
/// crate's `tfrecord` feature (enabled by default); when that feature is
/// disabled, CRC-32 verification returns an honest `Err(...)` instead of
/// silently reporting success.
pub fn verify_checksum(path: &Path, expected_hash: Option<&str>) -> Result<bool> {
    let expected = match expected_hash {
        // Nothing was requested to be verified.
        None => return Ok(true),
        Some(value) => value.trim(),
    };

    if expected.is_empty() {
        return Err(TensorError::invalid_argument(
            "Checksum verification failed: an empty expected checksum was provided".to_string(),
        ));
    }

    // Split an optional "<algorithm>:<hex>" prefix.
    let (algorithm, expected_hex) = match expected.split_once(':') {
        Some((algo, hex)) => (algo.trim().to_ascii_lowercase(), hex.trim()),
        None => (String::new(), expected),
    };

    // Validate the expected digest is hexadecimal so we can compare reliably.
    if expected_hex.is_empty() || !expected_hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(TensorError::invalid_argument(format!(
            "Checksum verification failed: expected checksum '{expected}' is not a valid hexadecimal digest"
        )));
    }

    // Resolve the digest algorithm, inferring from length when unprefixed.
    let resolved_algorithm = if algorithm.is_empty() {
        match expected_hex.len() {
            64 => "sha256",
            8 => "crc32",
            other => {
                return Err(TensorError::invalid_argument(format!(
                    "Checksum verification failed: cannot determine hash algorithm for a \
                     {other}-character digest; prefix the expected checksum with an algorithm \
                     (e.g. \"sha256:...\")"
                )));
            }
        }
    } else {
        algorithm.as_str()
    };

    match resolved_algorithm {
        "sha256" | "sha-256" => {
            if expected_hex.len() != 64 {
                return Err(TensorError::invalid_argument(format!(
                    "Checksum verification failed: SHA-256 digest must be 64 hex characters, \
                     got {} characters",
                    expected_hex.len()
                )));
            }
            let actual_hex = compute_sha256_hex(path)?;
            Ok(actual_hex.eq_ignore_ascii_case(expected_hex))
        }
        "crc32" | "crc-32" => {
            if expected_hex.len() != 8 {
                return Err(TensorError::invalid_argument(format!(
                    "Checksum verification failed: CRC-32 digest must be 8 hex characters, \
                     got {} characters",
                    expected_hex.len()
                )));
            }
            let actual_hex = compute_crc32_hex(path)?;
            Ok(actual_hex.eq_ignore_ascii_case(expected_hex))
        }
        other => Err(TensorError::invalid_argument(format!(
            "Checksum verification failed: hash algorithm '{other}' is not supported \
             (supported: sha256, crc32)"
        ))),
    }
}

/// Compute the SHA-256 digest of a file's contents, returned as a lowercase hex string.
fn compute_sha256_hex(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};

    let file = File::open(path)
        .map_err(|e| error_utils::io_error_with_context(e, "Failed to open file for checksum"))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = reader.read(&mut buffer).map_err(|e| {
            error_utils::io_error_with_context(e, "Failed to read file while computing checksum")
        })?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        // Writing to a String never fails; surface any unexpected error honestly.
        write!(hex, "{byte:02x}").map_err(|e| {
            TensorError::invalid_argument(format!("Failed to format checksum digest: {e}"))
        })?;
    }
    Ok(hex)
}

/// Compute the CRC-32 (IEEE 802.3) checksum of a file's contents, returned as
/// a lowercase 8-character hex string.
///
/// Uses the pure-Rust `crc32fast` crate, which is wired to this crate's
/// `tfrecord` feature (enabled by default).
#[cfg(feature = "tfrecord")]
fn compute_crc32_hex(path: &Path) -> Result<String> {
    use crc32fast::Hasher;

    let file = File::open(path)
        .map_err(|e| error_utils::io_error_with_context(e, "Failed to open file for checksum"))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Hasher::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = reader.read(&mut buffer).map_err(|e| {
            error_utils::io_error_with_context(e, "Failed to read file while computing checksum")
        })?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let checksum = hasher.finalize();
    Ok(format!("{checksum:08x}"))
}

/// Compute the CRC-32 checksum when the `tfrecord` feature (which provides
/// the `crc32fast` dependency) is disabled.
#[cfg(not(feature = "tfrecord"))]
fn compute_crc32_hex(_path: &Path) -> Result<String> {
    Err(TensorError::invalid_argument(
        "CRC-32 checksum verification requires the 'tfrecord' feature to be enabled.".to_string(),
    ))
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
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let test_file = temp_dir.path().join("test.txt");

        // Create a test file
        std::fs::write(&test_file, b"Hello, World!").expect("test: write should succeed");

        let size = get_file_size(&test_file).expect("test: operation should succeed");
        assert_eq!(size, 13); // "Hello, World!" is 13 bytes
    }

    #[test]
    fn test_verify_checksum_no_hash() {
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let test_file = temp_dir.path().join("test.txt");

        // Create a test file
        std::fs::write(&test_file, b"test").expect("test: write should succeed");

        let result = verify_checksum(&test_file, None).expect("test: operation should succeed");
        assert!(result);
    }

    #[test]
    fn test_verify_checksum_invalid_hash_is_error() {
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let test_file = temp_dir.path().join("test.txt");

        std::fs::write(&test_file, b"test").expect("test: write should succeed");

        // A non-hex / unknown-length expected checksum can never be verified,
        // so the function must report an honest error rather than success.
        let result = verify_checksum(&test_file, Some("dummy_hash"));
        assert!(
            result.is_err(),
            "an unverifiable checksum must not be reported as valid"
        );
    }

    #[test]
    fn test_verify_checksum_correct_sha256_returns_true() {
        // Known SHA-256 of the bytes b"test".
        const TEST_SHA256: &str =
            "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";

        let base = std::env::temp_dir().join(format!(
            "tenflowers_checksum_ok_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&base).expect("test: temp dir creation should succeed");
        let test_file = base.join("known.bin");
        std::fs::write(&test_file, b"test").expect("test: write should succeed");

        // Bare 64-hex digest is inferred as SHA-256.
        let inferred =
            verify_checksum(&test_file, Some(TEST_SHA256)).expect("test: verification should run");
        assert!(inferred, "correct bare SHA-256 must verify as true");

        // Explicit algorithm prefix is also accepted (any casing).
        let prefixed = verify_checksum(&test_file, Some(&format!("sha256:{TEST_SHA256}")))
            .expect("test: verification should run");
        assert!(prefixed, "correct prefixed SHA-256 must verify as true");

        let upper = verify_checksum(&test_file, Some(&TEST_SHA256.to_uppercase()))
            .expect("test: verification should run");
        assert!(upper, "case-insensitive SHA-256 must verify as true");

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_verify_checksum_tampered_bytes_returns_false() {
        // SHA-256 of b"test", but the file actually holds different (tampered) bytes.
        const TEST_SHA256: &str =
            "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";

        let base = std::env::temp_dir().join(format!(
            "tenflowers_checksum_bad_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&base).expect("test: temp dir creation should succeed");
        let test_file = base.join("tampered.bin");
        std::fs::write(&test_file, b"tampered").expect("test: write should succeed");

        let result =
            verify_checksum(&test_file, Some(TEST_SHA256)).expect("test: verification should run");
        assert!(
            !result,
            "a file whose bytes do not match the expected hash must verify as false"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_verify_checksum_correct_crc32_returns_true() {
        let base = std::env::temp_dir().join(format!(
            "tenflowers_checksum_crc32_ok_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&base).expect("test: temp dir creation should succeed");
        let test_file = base.join("known.bin");
        std::fs::write(&test_file, b"test").expect("test: write should succeed");

        // Compute the digest directly so this test is self-consistent without
        // depending on a hand-copied known-answer constant.
        let actual_hex =
            compute_crc32_hex(&test_file).expect("test: crc32 computation should succeed");
        assert_eq!(
            actual_hex.len(),
            8,
            "CRC-32 hex digest must be 8 characters"
        );

        // Bare 8-hex digest is inferred as CRC-32.
        let inferred =
            verify_checksum(&test_file, Some(&actual_hex)).expect("test: verification should run");
        assert!(inferred, "correct bare CRC-32 must verify as true");

        // Explicit algorithm prefix is also accepted (any casing).
        let prefixed = verify_checksum(&test_file, Some(&format!("crc32:{actual_hex}")))
            .expect("test: verification should run");
        assert!(prefixed, "correct prefixed CRC-32 must verify as true");

        let upper = verify_checksum(&test_file, Some(&actual_hex.to_uppercase()))
            .expect("test: verification should run");
        assert!(upper, "case-insensitive CRC-32 must verify as true");

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_verify_checksum_crc32_tampered_bytes_returns_false() {
        let base = std::env::temp_dir().join(format!(
            "tenflowers_checksum_crc32_bad_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&base).expect("test: temp dir creation should succeed");

        // Digest of the *expected* content ("test"), computed independently
        // of the tampered file below.
        let expected_file = base.join("expected.bin");
        std::fs::write(&expected_file, b"test").expect("test: write should succeed");
        let expected_hex =
            compute_crc32_hex(&expected_file).expect("test: crc32 computation should succeed");

        // The actual file on disk holds different (tampered) bytes.
        let tampered_file = base.join("tampered.bin");
        std::fs::write(&tampered_file, b"tampered").expect("test: write should succeed");

        let result = verify_checksum(&tampered_file, Some(&expected_hex))
            .expect("test: verification should run");
        assert!(
            !result,
            "a file whose bytes do not match the expected CRC-32 must verify as false"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_verify_checksum_unsupported_algorithm_is_error() {
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let test_file = temp_dir.path().join("test.txt");
        std::fs::write(&test_file, b"test").expect("test: write should succeed");

        // md5 is valid hex but not supported by the pure-Rust hasher here.
        let result = verify_checksum(&test_file, Some("md5:098f6bcd4621d373cade4e832627b4f6"));
        assert!(
            result.is_err(),
            "unsupported algorithms must return an honest error, never Ok(true)"
        );
    }
}
