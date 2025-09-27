//! Real dataset loaders with automatic downloading
//!
//! This module provides loaders for popular ML datasets like MNIST, CIFAR-10, etc.
//! with automatic downloading from official sources.
//!
//! ## Module Structure
//!
//! - [`common`]: Common utilities and constants for real dataset loaders
//! - [`download`]: Download and extraction utilities for dataset files
//! - [`vision`]: Computer vision datasets: MNIST, CIFAR-10, ImageNet
//! - [`nlp`]: Natural Language Processing datasets: IMDB, AG News

pub mod common;
pub mod download;
pub mod nlp;
pub mod vision;

// Re-export all public types for backward compatibility

// Common utilities
pub use common::{
    DatasetConfig, ProgressTracker, AG_NEWS_TEST_URL, AG_NEWS_TRAIN_URL, CIFAR10_URL,
    IMAGENET_LABELS_URL, IMAGENET_VAL_URL, IMDB_URL, MNIST_TEST_IMAGES_URL, MNIST_TEST_LABELS_URL,
    MNIST_TRAIN_IMAGES_URL, MNIST_TRAIN_LABELS_URL,
};

// Download utilities
pub use download::{get_file_size, verify_checksum, Downloader};

// Vision datasets
pub use vision::{
    Cifar10Config, ImageNetConfig, MnistConfig, RealCifar10Builder, RealCifar10Dataset,
    RealImageNetBuilder, RealImageNetDataset, RealMnistBuilder, RealMnistDataset,
};

// NLP datasets
pub use nlp::{
    AgNewsConfig, ImdbConfig, RealAgNewsBuilder, RealAgNewsDataset, RealImdbBuilder,
    RealImdbDataset,
};

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_mnist_builder_integration() {
        let temp_dir = TempDir::new().unwrap();
        let builder = RealMnistBuilder::new()
            .root(temp_dir.path())
            .train(true)
            .download(false); // Don't actually download in tests

        // This should fail since we don't have files and download is disabled
        let result = builder.build::<f32>();
        assert!(result.is_err());
    }

    #[test]
    fn test_cifar10_builder_integration() {
        let temp_dir = TempDir::new().unwrap();
        let builder = RealCifar10Builder::new()
            .root(temp_dir.path())
            .train(true)
            .download(false); // Don't actually download in tests

        // This should fail since we don't have files and download is disabled
        let result = builder.build::<f32>();
        assert!(result.is_err());
    }

    #[test]
    fn test_imagenet_builder_integration() {
        let temp_dir = TempDir::new().unwrap();
        let builder = RealImageNetBuilder::new()
            .root(temp_dir.path())
            .train(false) // Use validation set
            .download(false); // Don't actually download in tests

        // This should fail since we don't have files and download is disabled
        let result = builder.build::<f32>();
        assert!(result.is_err());
    }

    #[test]
    fn test_imdb_builder_integration() {
        let temp_dir = TempDir::new().unwrap();
        let builder = RealImdbBuilder::new()
            .root(temp_dir.path())
            .train(true)
            .download(false); // Don't actually download in tests

        // This should fail since we don't have files and download is disabled
        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_ag_news_builder_integration() {
        let temp_dir = TempDir::new().unwrap();
        let builder = RealAgNewsBuilder::new()
            .root(temp_dir.path())
            .train(true)
            .download(false); // Don't actually download in tests

        // This should fail since we don't have files and download is disabled
        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_downloader_integration() {
        let downloader = Downloader::new();
        // Just verify it can be created without panicking
        drop(downloader);
    }

    #[test]
    fn test_constants_available() {
        // Test that all URL constants are accessible
        assert!(!MNIST_TRAIN_IMAGES_URL.is_empty());
        assert!(!MNIST_TRAIN_LABELS_URL.is_empty());
        assert!(!MNIST_TEST_IMAGES_URL.is_empty());
        assert!(!MNIST_TEST_LABELS_URL.is_empty());
        assert!(!CIFAR10_URL.is_empty());
        assert!(!IMAGENET_VAL_URL.is_empty());
        assert!(!IMAGENET_LABELS_URL.is_empty());
        assert!(!IMDB_URL.is_empty());
        assert!(!AG_NEWS_TRAIN_URL.is_empty());
        assert!(!AG_NEWS_TEST_URL.is_empty());
    }

    #[test]
    fn test_config_defaults() {
        let mnist_config = MnistConfig::default();
        assert!(mnist_config.train);
        assert!(mnist_config.download);

        let cifar10_config = Cifar10Config::default();
        assert!(cifar10_config.train);
        assert!(cifar10_config.download);

        let imagenet_config = ImageNetConfig::default();
        assert!(!imagenet_config.train); // Default to validation set
        assert!(!imagenet_config.download); // Default to no download (ImageNet is large)

        let imdb_config = ImdbConfig::default();
        assert!(imdb_config.train);
        assert!(imdb_config.download);

        let ag_news_config = AgNewsConfig::default();
        assert!(ag_news_config.train);
        assert!(ag_news_config.download);
    }

    #[test]
    fn test_class_names() {
        let mnist_classes = 10;
        let cifar10_classes = 10;
        let imagenet_classes = 1000;
        let imdb_classes = 2;
        let ag_news_classes = 4;

        assert_eq!(mnist_classes, 10);
        assert_eq!(cifar10_classes, 10);
        assert_eq!(imagenet_classes, 1000);
        assert_eq!(imdb_classes, 2);
        assert_eq!(ag_news_classes, 4);

        // Test actual class names
        let cifar10_names = vision::RealCifar10Dataset::<f32>::class_names();
        assert_eq!(cifar10_names.len(), 10);
        assert_eq!(cifar10_names[0], "airplane");

        let imdb_names = nlp::RealImdbDataset::class_names();
        assert_eq!(imdb_names.len(), 2);
        assert_eq!(imdb_names[0], "negative");
        assert_eq!(imdb_names[1], "positive");

        let ag_news_names = nlp::RealAgNewsDataset::class_names();
        assert_eq!(ag_news_names.len(), 4);
        assert_eq!(ag_news_names[0], "World");
        assert_eq!(ag_news_names[3], "Science/Tech");
    }
}
