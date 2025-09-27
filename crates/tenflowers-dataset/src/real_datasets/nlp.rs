//! Natural Language Processing datasets: IMDB, AG News

use crate::Dataset;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use tenflowers_core::{Result, Tensor, TensorError};

use super::common::{error_utils, validation, DatasetConfig};
use super::download::Downloader;

/// Real IMDB dataset for sentiment analysis
pub struct RealImdbDataset {
    texts: Vec<String>,
    labels: Vec<i32>,
    num_samples: usize,
    is_train: bool,
}

/// Configuration for IMDB dataset loading
#[derive(Debug, Clone)]
pub struct ImdbConfig {
    /// Root directory to store downloaded data
    pub root: PathBuf,
    /// Whether to use training set (true) or test set (false)
    pub train: bool,
    /// Whether to download if not found
    pub download: bool,
    /// Maximum number of samples to load (None for all)
    pub max_samples: Option<usize>,
}

impl Default for ImdbConfig {
    fn default() -> Self {
        Self {
            root: PathBuf::from("./data"),
            train: true,
            download: true,
            max_samples: None,
        }
    }
}

impl DatasetConfig for ImdbConfig {
    fn root(&self) -> &PathBuf {
        &self.root
    }

    fn is_train(&self) -> bool {
        self.train
    }

    fn should_download(&self) -> bool {
        self.download
    }
}

impl RealImdbDataset {
    /// Create a new IMDB dataset
    pub fn new(config: ImdbConfig) -> Result<Self> {
        let imdb_dir = config.root.join("aclImdb");

        // Create directory if it doesn't exist
        validation::ensure_directory_exists(&imdb_dir, "IMDB")?;

        // Check if data exists, download if needed
        let split_dir = if config.train { "train" } else { "test" };
        let data_dir = imdb_dir.join(split_dir);

        if config.download && !data_dir.exists() {
            let downloader = Downloader::new();
            downloader.download_imdb(&config.root)?;
        }

        // Verify directories exist
        validation::validate_file_exists(&data_dir.join("pos"), "IMDB positive reviews")?;
        validation::validate_file_exists(&data_dir.join("neg"), "IMDB negative reviews")?;

        // Load the data
        let (texts, labels) = Self::load_imdb_data(&data_dir, config.max_samples)?;
        let num_samples = texts.len();

        Ok(Self {
            texts,
            labels,
            num_samples,
            is_train: config.train,
        })
    }

    /// Load IMDB data from text files
    fn load_imdb_data(
        data_dir: &Path,
        max_samples: Option<usize>,
    ) -> Result<(Vec<String>, Vec<i32>)> {
        let mut texts = Vec::new();
        let mut labels = Vec::new();

        // Load positive reviews
        let pos_dir = data_dir.join("pos");
        Self::load_reviews_from_directory(
            &pos_dir,
            1,
            &mut texts,
            &mut labels,
            max_samples.map(|m| m / 2),
        )?;

        // Load negative reviews
        let neg_dir = data_dir.join("neg");
        Self::load_reviews_from_directory(
            &neg_dir,
            0,
            &mut texts,
            &mut labels,
            max_samples.map(|m| m / 2),
        )?;

        Ok((texts, labels))
    }

    /// Load reviews from a specific directory
    fn load_reviews_from_directory(
        dir: &Path,
        label: i32,
        texts: &mut Vec<String>,
        labels: &mut Vec<i32>,
        max_samples: Option<usize>,
    ) -> Result<()> {
        let entries = std::fs::read_dir(dir)
            .map_err(|e| error_utils::io_error_with_context(e, "Failed to read IMDB directory"))?;

        let mut count = 0;
        for entry in entries {
            if let Some(max) = max_samples {
                if count >= max {
                    break;
                }
            }

            let entry = entry.map_err(|e| {
                error_utils::io_error_with_context(e, "Failed to read directory entry")
            })?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("txt") {
                let content = std::fs::read_to_string(&path).map_err(|e| {
                    error_utils::io_error_with_context(e, "Failed to read review file")
                })?;

                // Clean up the text (remove newlines, extra spaces)
                let cleaned_text = content
                    .lines()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ");

                texts.push(cleaned_text);
                labels.push(label);
                count += 1;
            }
        }

        Ok(())
    }

    /// Get all texts
    pub fn texts(&self) -> &[String] {
        &self.texts
    }

    /// Get all labels
    pub fn labels(&self) -> &[i32] {
        &self.labels
    }

    /// Get whether this is training set
    pub fn is_train(&self) -> bool {
        self.is_train
    }

    /// Get the number of classes (2 for sentiment: positive, negative)
    pub fn num_classes(&self) -> usize {
        2
    }

    /// Get class names
    pub fn class_names() -> Vec<String> {
        vec!["negative".to_string(), "positive".to_string()]
    }
}

impl Dataset<f32> for RealImdbDataset {
    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, index: usize) -> Result<(Tensor<f32>, Tensor<f32>)> {
        if index >= self.num_samples {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index, self.num_samples
            )));
        }

        // For text data, return a simple representation
        // In a real implementation, you would tokenize and encode the text
        let text = &self.texts[index];
        let label = self.labels[index];

        // Simple features: text length, word count, etc.
        let word_count = text.split_whitespace().count() as f32;
        let char_count = text.len() as f32;
        let avg_word_length = if word_count > 0.0 {
            char_count / word_count
        } else {
            0.0
        };

        // Create feature tensor [word_count, char_count, avg_word_length]
        let features = Tensor::from_vec(vec![word_count, char_count, avg_word_length], &[3])?;

        // Create label tensor
        let label_tensor = Tensor::from_scalar(label as f32);

        Ok((features, label_tensor))
    }
}

/// Real AG News dataset for news categorization
pub struct RealAgNewsDataset {
    texts: Vec<String>,
    labels: Vec<i32>,
    num_samples: usize,
    is_train: bool,
}

/// Configuration for AG News dataset loading
#[derive(Debug, Clone)]
pub struct AgNewsConfig {
    /// Root directory to store downloaded data
    pub root: PathBuf,
    /// Whether to use training set (true) or test set (false)
    pub train: bool,
    /// Whether to download if not found
    pub download: bool,
    /// Maximum number of samples to load (None for all)
    pub max_samples: Option<usize>,
}

impl Default for AgNewsConfig {
    fn default() -> Self {
        Self {
            root: PathBuf::from("./data"),
            train: true,
            download: true,
            max_samples: None,
        }
    }
}

impl DatasetConfig for AgNewsConfig {
    fn root(&self) -> &PathBuf {
        &self.root
    }

    fn is_train(&self) -> bool {
        self.train
    }

    fn should_download(&self) -> bool {
        self.download
    }
}

impl RealAgNewsDataset {
    /// Create a new AG News dataset
    pub fn new(config: AgNewsConfig) -> Result<Self> {
        let ag_news_dir = config.root.join("ag_news");

        // Create directory if it doesn't exist
        validation::ensure_directory_exists(&ag_news_dir, "AG News")?;

        // Check if data exists, download if needed
        let csv_file = if config.train {
            "train.csv"
        } else {
            "test.csv"
        };

        let csv_path = ag_news_dir.join(csv_file);

        if config.download && !csv_path.exists() {
            let downloader = Downloader::new();
            downloader.download_ag_news(&ag_news_dir)?;
        }

        // Verify file exists
        validation::validate_file_exists(&csv_path, "AG News CSV file")?;

        // Load the data
        let (texts, labels) = Self::load_ag_news_data(&csv_path, config.max_samples)?;
        let num_samples = texts.len();

        Ok(Self {
            texts,
            labels,
            num_samples,
            is_train: config.train,
        })
    }

    /// Load AG News data from CSV file
    fn load_ag_news_data(
        csv_path: &Path,
        max_samples: Option<usize>,
    ) -> Result<(Vec<String>, Vec<i32>)> {
        let file = File::open(csv_path).map_err(|e| {
            error_utils::io_error_with_context(e, "Failed to open AG News CSV file")
        })?;

        let reader = BufReader::new(file);
        let mut texts = Vec::new();
        let mut labels = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            if let Some(max) = max_samples {
                if line_num >= max {
                    break;
                }
            }

            let line =
                line.map_err(|e| error_utils::io_error_with_context(e, "Failed to read CSV line"))?;

            // Parse CSV line: "label","title","description"
            if let Some((label_str, rest)) = Self::parse_csv_field(&line) {
                if let Some((title, description)) = Self::parse_csv_field(rest) {
                    // Parse label (1-based to 0-based)
                    if let Ok(label) = label_str.parse::<i32>() {
                        let combined_text =
                            if let Some((desc, _)) = Self::parse_csv_field(description) {
                                format!("{} {}", title, desc)
                            } else {
                                title.to_string()
                            };

                        texts.push(combined_text);
                        labels.push(label - 1); // Convert to 0-based indexing
                    }
                }
            }
        }

        Ok((texts, labels))
    }

    /// Simple CSV field parser (handles quoted fields)
    fn parse_csv_field(line: &str) -> Option<(&str, &str)> {
        let line = line.trim();
        if let Some(stripped) = line.strip_prefix('"') {
            // Find the closing quote
            if let Some(end_quote) = stripped.find('"') {
                let field = &stripped[..end_quote];
                // end_quote is relative to line[1..], so actual quote position is end_quote + 1
                let quote_pos = end_quote + 1;
                let after_quote = quote_pos + 1;

                // Check if there's a comma after the quote
                if after_quote < line.len() && line.chars().nth(after_quote) == Some(',') {
                    let rest = &line[after_quote + 1..]; // Skip the comma
                    Some((field, rest))
                } else {
                    // No comma, so this is the last field
                    Some((field, ""))
                }
            } else {
                None
            }
        } else {
            // Find the first comma
            if let Some(comma_pos) = line.find(',') {
                let field = &line[..comma_pos];
                let rest = &line[comma_pos + 1..];
                Some((field, rest))
            } else {
                Some((line, ""))
            }
        }
    }

    /// Get all texts
    pub fn texts(&self) -> &[String] {
        &self.texts
    }

    /// Get all labels
    pub fn labels(&self) -> &[i32] {
        &self.labels
    }

    /// Get whether this is training set
    pub fn is_train(&self) -> bool {
        self.is_train
    }

    /// Get the number of classes (4 for AG News)
    pub fn num_classes(&self) -> usize {
        4
    }

    /// Get class names
    pub fn class_names() -> Vec<String> {
        vec![
            "World".to_string(),
            "Sports".to_string(),
            "Business".to_string(),
            "Science/Tech".to_string(),
        ]
    }
}

impl Dataset<f32> for RealAgNewsDataset {
    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, index: usize) -> Result<(Tensor<f32>, Tensor<f32>)> {
        if index >= self.num_samples {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index, self.num_samples
            )));
        }

        // For text data, return a simple representation
        let text = &self.texts[index];
        let label = self.labels[index];

        // Simple features: text length, word count, etc.
        let word_count = text.split_whitespace().count() as f32;
        let char_count = text.len() as f32;
        let avg_word_length = if word_count > 0.0 {
            char_count / word_count
        } else {
            0.0
        };

        // Create feature tensor [word_count, char_count, avg_word_length]
        let features = Tensor::from_vec(vec![word_count, char_count, avg_word_length], &[3])?;

        // Create label tensor
        let label_tensor = Tensor::from_scalar(label as f32);

        Ok((features, label_tensor))
    }
}

/// Builder patterns for easy dataset construction
pub struct RealImdbBuilder {
    config: ImdbConfig,
}

impl RealImdbBuilder {
    /// Create a new IMDB builder
    pub fn new() -> Self {
        Self {
            config: ImdbConfig::default(),
        }
    }

    /// Set the root directory for data storage
    pub fn root<P: AsRef<Path>>(mut self, root: P) -> Self {
        self.config.root = root.as_ref().to_path_buf();
        self
    }

    /// Set whether to use training set (true) or test set (false)
    pub fn train(mut self, train: bool) -> Self {
        self.config.train = train;
        self
    }

    /// Set whether to download if not found
    pub fn download(mut self, download: bool) -> Self {
        self.config.download = download;
        self
    }

    /// Set maximum number of samples to load
    pub fn max_samples(mut self, max_samples: Option<usize>) -> Self {
        self.config.max_samples = max_samples;
        self
    }

    /// Build the IMDB dataset
    pub fn build(self) -> Result<RealImdbDataset> {
        RealImdbDataset::new(self.config)
    }
}

impl Default for RealImdbBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct RealAgNewsBuilder {
    config: AgNewsConfig,
}

impl RealAgNewsBuilder {
    /// Create a new AG News builder
    pub fn new() -> Self {
        Self {
            config: AgNewsConfig::default(),
        }
    }

    /// Set the root directory for data storage
    pub fn root<P: AsRef<Path>>(mut self, root: P) -> Self {
        self.config.root = root.as_ref().to_path_buf();
        self
    }

    /// Set whether to use training set (true) or test set (false)
    pub fn train(mut self, train: bool) -> Self {
        self.config.train = train;
        self
    }

    /// Set whether to download if not found
    pub fn download(mut self, download: bool) -> Self {
        self.config.download = download;
        self
    }

    /// Set maximum number of samples to load
    pub fn max_samples(mut self, max_samples: Option<usize>) -> Self {
        self.config.max_samples = max_samples;
        self
    }

    /// Build the AG News dataset
    pub fn build(self) -> Result<RealAgNewsDataset> {
        RealAgNewsDataset::new(self.config)
    }
}

impl Default for RealAgNewsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_imdb_builder() {
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
    fn test_imdb_config_default() {
        let config = ImdbConfig::default();
        assert_eq!(config.root, PathBuf::from("./data"));
        assert!(config.train);
        assert!(config.download);
        assert!(config.max_samples.is_none());
    }

    #[test]
    fn test_ag_news_builder() {
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
    fn test_ag_news_config_default() {
        let config = AgNewsConfig::default();
        assert_eq!(config.root, PathBuf::from("./data"));
        assert!(config.train);
        assert!(config.download);
        assert!(config.max_samples.is_none());
    }

    #[test]
    fn test_csv_field_parsing() {
        // Test quoted field
        let result = RealAgNewsDataset::parse_csv_field("\"World News\",\"Title\",\"Description\"");
        assert_eq!(result, Some(("World News", "\"Title\",\"Description\"")));

        // Test unquoted field
        let result = RealAgNewsDataset::parse_csv_field("1,\"Title\",\"Description\"");
        assert_eq!(result, Some(("1", "\"Title\",\"Description\"")));

        // Test single field
        let result = RealAgNewsDataset::parse_csv_field("\"Only field\"");
        assert_eq!(result, Some(("Only field", "")));
    }

    #[test]
    fn test_imdb_class_names() {
        let class_names = RealImdbDataset::class_names();
        assert_eq!(class_names.len(), 2);
        assert_eq!(class_names[0], "negative");
        assert_eq!(class_names[1], "positive");
    }

    #[test]
    fn test_ag_news_class_names() {
        let class_names = RealAgNewsDataset::class_names();
        assert_eq!(class_names.len(), 4);
        assert_eq!(class_names[0], "World");
        assert_eq!(class_names[3], "Science/Tech");
    }
}
