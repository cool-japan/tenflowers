//! Text dataset format support for NLP tasks
//!
//! This module provides support for loading text datasets commonly used in
//! natural language processing tasks such as classification, sentiment analysis,
//! and language modeling.

use crate::Dataset;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Configuration for text dataset loading
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct TextConfig {
    /// Path to the text file
    pub file_path: PathBuf,
    /// Maximum sequence length (truncate or pad to this length)
    pub max_sequence_length: usize,
    /// Character to use for padding sequences
    pub pad_token: String,
    /// Character to use for unknown tokens
    pub unk_token: String,
    /// Character to use for beginning of sequence
    pub bos_token: Option<String>,
    /// Character to use for end of sequence
    pub eos_token: Option<String>,
    /// Whether to convert text to lowercase
    pub lowercase: bool,
    /// Minimum token frequency (tokens below this are replaced with unk_token)
    pub min_frequency: usize,
    /// Maximum vocabulary size (keep most frequent tokens)
    pub max_vocab_size: Option<usize>,
    /// Whether to split on whitespace or use character-level tokenization
    pub tokenization_strategy: TokenizationStrategy,
    /// Label extraction strategy
    pub label_strategy: LabelStrategy,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            file_path: PathBuf::new(),
            max_sequence_length: 512,
            pad_token: "<pad>".to_string(),
            unk_token: "<unk>".to_string(),
            bos_token: Some("<bos>".to_string()),
            eos_token: Some("<eos>".to_string()),
            lowercase: true,
            min_frequency: 2,
            max_vocab_size: Some(10000),
            tokenization_strategy: TokenizationStrategy::WordLevel,
            label_strategy: LabelStrategy::FromFilename,
        }
    }
}

/// Tokenization strategy for text processing
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum TokenizationStrategy {
    /// Split on whitespace
    WordLevel,
    /// Character-level tokenization
    CharLevel,
    /// Subword tokenization (simple byte-pair encoding)
    Subword,
}

/// Strategy for extracting labels from text data
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum LabelStrategy {
    /// Extract label from filename (e.g., "pos_review1.txt" -> "pos")
    FromFilename,
    /// Extract label from first line of file
    FromFirstLine,
    /// Extract label from directory name
    FromDirectory,
    /// No labels (unsupervised learning)
    NoLabels,
    /// Labels provided separately
    External(Vec<String>),
}

/// Vocabulary for text tokenization
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Vocabulary {
    /// Token to ID mapping
    pub token_to_id: HashMap<String, usize>,
    /// ID to token mapping
    pub id_to_token: HashMap<usize, String>,
    /// Token frequencies
    pub token_counts: HashMap<String, usize>,
    /// Special tokens
    pub pad_token_id: usize,
    pub unk_token_id: usize,
    pub bos_token_id: Option<usize>,
    pub eos_token_id: Option<usize>,
}

impl Vocabulary {
    /// Create a new vocabulary from text data
    pub fn from_texts(texts: &[String], config: &TextConfig) -> Self {
        let mut token_counts = HashMap::new();

        // Count token frequencies
        for text in texts {
            let tokens = Self::tokenize_text(text, &config.tokenization_strategy, config.lowercase);
            for token in tokens {
                *token_counts.entry(token).or_insert(0) += 1;
            }
        }

        // Filter by minimum frequency
        token_counts.retain(|_, &mut count| count >= config.min_frequency);

        // Sort by frequency and limit vocabulary size
        let mut sorted_tokens: Vec<_> = token_counts.iter().collect();
        sorted_tokens.sort_by(|a, b| b.1.cmp(a.1));

        if let Some(max_size) = config.max_vocab_size {
            sorted_tokens.truncate(max_size.saturating_sub(4)); // Reserve space for special tokens
        }

        // Build vocabulary mappings
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut next_id = 0;

        // Add special tokens first
        let pad_token_id = next_id;
        token_to_id.insert(config.pad_token.clone(), next_id);
        id_to_token.insert(next_id, config.pad_token.clone());
        next_id += 1;

        let unk_token_id = next_id;
        token_to_id.insert(config.unk_token.clone(), next_id);
        id_to_token.insert(next_id, config.unk_token.clone());
        next_id += 1;

        let bos_token_id = if let Some(ref bos_token) = config.bos_token {
            let id = next_id;
            token_to_id.insert(bos_token.clone(), next_id);
            id_to_token.insert(next_id, bos_token.clone());
            next_id += 1;
            Some(id)
        } else {
            None
        };

        let eos_token_id = if let Some(ref eos_token) = config.eos_token {
            let id = next_id;
            token_to_id.insert(eos_token.clone(), next_id);
            id_to_token.insert(next_id, eos_token.clone());
            next_id += 1;
            Some(id)
        } else {
            None
        };

        // Add regular tokens
        for (token, _) in sorted_tokens {
            if !token_to_id.contains_key(token) {
                token_to_id.insert(token.clone(), next_id);
                id_to_token.insert(next_id, token.clone());
                next_id += 1;
            }
        }

        Self {
            token_to_id,
            id_to_token,
            token_counts,
            pad_token_id,
            unk_token_id,
            bos_token_id,
            eos_token_id,
        }
    }

    /// Tokenize text according to strategy
    fn tokenize_text(text: &str, strategy: &TokenizationStrategy, lowercase: bool) -> Vec<String> {
        let processed_text = if lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        match strategy {
            TokenizationStrategy::WordLevel => processed_text
                .split_whitespace()
                .map(|s| s.to_string())
                .collect(),
            TokenizationStrategy::CharLevel => {
                processed_text.chars().map(|c| c.to_string()).collect()
            }
            TokenizationStrategy::Subword => {
                // Simple subword tokenization (could be enhanced with BPE)
                Self::simple_subword_tokenize(&processed_text)
            }
        }
    }

    /// Simple subword tokenization (splits on punctuation and whitespace)
    fn simple_subword_tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();

        for ch in text.chars() {
            if ch.is_whitespace() || ch.is_ascii_punctuation() {
                if !current_token.is_empty() {
                    tokens.push(current_token.clone());
                    current_token.clear();
                }
                if !ch.is_whitespace() {
                    tokens.push(ch.to_string());
                }
            } else {
                current_token.push(ch);
            }
        }

        if !current_token.is_empty() {
            tokens.push(current_token);
        }

        tokens
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str, config: &TextConfig) -> Vec<usize> {
        let tokens = Self::tokenize_text(text, &config.tokenization_strategy, config.lowercase);
        let mut token_ids = Vec::new();

        // Add BOS token if configured
        if let Some(bos_id) = self.bos_token_id {
            token_ids.push(bos_id);
        }

        // Convert tokens to IDs
        for token in tokens {
            let id = self.token_to_id.get(&token).unwrap_or(&self.unk_token_id);
            token_ids.push(*id);

            // Stop if we're approaching max length (save space for EOS)
            if token_ids.len() >= config.max_sequence_length - 1 {
                break;
            }
        }

        // Add EOS token if configured
        if let Some(eos_id) = self.eos_token_id {
            if token_ids.len() < config.max_sequence_length {
                token_ids.push(eos_id);
            }
        }

        // Pad or truncate to max length
        if token_ids.len() < config.max_sequence_length {
            token_ids.resize(config.max_sequence_length, self.pad_token_id);
        } else {
            token_ids.truncate(config.max_sequence_length);
        }

        token_ids
    }

    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[usize]) -> String {
        let tokens: Vec<String> = token_ids
            .iter()
            .filter_map(|&id| self.id_to_token.get(&id))
            .filter(|token| {
                // Skip special tokens in output
                *token != &self.id_to_token[&self.pad_token_id]
                    && Some(*token)
                        != self
                            .bos_token_id
                            .as_ref()
                            .and_then(|id| self.id_to_token.get(id))
                    && Some(*token)
                        != self
                            .eos_token_id
                            .as_ref()
                            .and_then(|id| self.id_to_token.get(id))
            })
            .cloned()
            .collect();

        tokens.join(" ")
    }

    /// Get vocabulary size
    pub fn len(&self) -> usize {
        self.token_to_id.len()
    }

    /// Check if vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.token_to_id.is_empty()
    }
}

/// Text dataset for NLP tasks
#[derive(Debug, Clone)]
pub struct TextDataset<T> {
    samples: Vec<(Tensor<T>, Tensor<T>)>,
    vocabulary: Vocabulary,
    label_vocab: Option<HashMap<String, usize>>,
    config: TextConfig,
}

impl<T> TextDataset<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::ToPrimitive
        + scirs2_core::numeric::NumCast
        + Send
        + Sync
        + 'static,
{
    /// Create a new text dataset from configuration
    pub fn from_config(config: TextConfig) -> Result<Self> {
        let (texts, labels) = Self::load_text_data(&config)?;

        // Build vocabulary from texts
        let vocabulary = Vocabulary::from_texts(&texts, &config);

        // Build label vocabulary if needed
        let label_vocab = if !labels.is_empty() {
            let mut unique_labels: Vec<_> = labels.to_vec();
            unique_labels.sort();
            unique_labels.dedup();

            let label_vocab: HashMap<String, usize> = unique_labels
                .into_iter()
                .enumerate()
                .map(|(i, label)| (label, i))
                .collect();
            Some(label_vocab)
        } else {
            None
        };

        // Convert texts and labels to tensors
        let mut samples = Vec::new();
        for (text, label) in texts.iter().zip(labels.iter()) {
            let token_ids = vocabulary.encode(text, &config);
            let feature_tensor = Self::ids_to_tensor(&token_ids)?;

            let label_tensor = if let Some(ref label_vocab) = label_vocab {
                let label_id = label_vocab.get(label).unwrap_or(&0);
                Self::scalar_to_tensor(*label_id)?
            } else {
                // No labels case - use dummy label
                Self::scalar_to_tensor(0)?
            };

            samples.push((feature_tensor, label_tensor));
        }

        Ok(Self {
            samples,
            vocabulary,
            label_vocab,
            config,
        })
    }

    /// Create a new text dataset from file path
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config = TextConfig {
            file_path: path.as_ref().to_path_buf(),
            ..Default::default()
        };
        Self::from_config(config)
    }

    /// Load text data and labels from file
    fn load_text_data(config: &TextConfig) -> Result<(Vec<String>, Vec<String>)> {
        let file = File::open(&config.file_path)
            .map_err(|e| TensorError::invalid_argument(format!("Cannot open text file: {e}")))?;

        let reader = BufReader::new(file);
        let mut texts = Vec::new();
        let mut labels = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| {
                TensorError::invalid_argument(format!("Cannot read line {}: {}", line_num + 1, e))
            })?;

            if line.trim().is_empty() {
                continue;
            }

            match &config.label_strategy {
                LabelStrategy::FromFirstLine => {
                    if line_num == 0 {
                        // First line contains labels
                        continue;
                    }
                    texts.push(line);
                    labels.push("unlabeled".to_string());
                }
                LabelStrategy::FromFilename => {
                    texts.push(line);
                    let filename = config
                        .file_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown");

                    // Try to extract label from filename (e.g., "pos_review.txt" -> "pos")
                    let label = if let Some(pos) = filename.find('_') {
                        filename[..pos].to_string()
                    } else {
                        filename.to_string()
                    };
                    labels.push(label);
                }
                LabelStrategy::FromDirectory => {
                    texts.push(line);
                    let dir_name = config
                        .file_path
                        .parent()
                        .and_then(|p| p.file_name())
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown");
                    labels.push(dir_name.to_string());
                }
                LabelStrategy::NoLabels => {
                    texts.push(line);
                    // No labels needed
                }
                LabelStrategy::External(external_labels) => {
                    texts.push(line);
                    if line_num < external_labels.len() {
                        labels.push(external_labels[line_num].clone());
                    } else {
                        labels.push("unknown".to_string());
                    }
                }
            }
        }

        if texts.is_empty() {
            return Err(TensorError::invalid_argument(
                "No text data found in file".to_string(),
            ));
        }

        Ok((texts, labels))
    }

    /// Convert token IDs to tensor
    fn ids_to_tensor(ids: &[usize]) -> Result<Tensor<T>> {
        let data: Vec<T> = ids
            .iter()
            .map(|&id| scirs2_core::num_traits::NumCast::from(id).unwrap_or_else(T::default))
            .collect();
        Tensor::from_vec(data, &[ids.len()])
    }

    /// Convert scalar to tensor
    fn scalar_to_tensor(value: usize) -> Result<Tensor<T>> {
        let tensor_val = scirs2_core::num_traits::NumCast::from(value).unwrap_or_else(T::default);
        Ok(Tensor::from_scalar(tensor_val))
    }

    /// Get the vocabulary used by this dataset
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    /// Get the label vocabulary if available
    pub fn label_vocabulary(&self) -> Option<&HashMap<String, usize>> {
        self.label_vocab.as_ref()
    }

    /// Get the configuration used for this dataset
    pub fn config(&self) -> &TextConfig {
        &self.config
    }

    /// Get statistics about the loaded dataset
    pub fn info(&self) -> TextDatasetInfo {
        TextDatasetInfo {
            sample_count: self.samples.len(),
            vocabulary_size: self.vocabulary.len(),
            max_sequence_length: self.config.max_sequence_length,
            label_count: self.label_vocab.as_ref().map(|lv| lv.len()),
            file_path: self.config.file_path.clone(),
        }
    }
}

impl<T> Dataset<T> for TextDataset<T>
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
{
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.samples.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index,
                self.samples.len()
            )));
        }
        Ok(self.samples[index].clone())
    }
}

/// Tokenized text dataset that preprocesses all text into token sequences
pub type TokenizedDataset<T> = TextDataset<T>;

/// Information about a text dataset
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct TextDatasetInfo {
    pub sample_count: usize,
    pub vocabulary_size: usize,
    pub max_sequence_length: usize,
    pub label_count: Option<usize>,
    pub file_path: PathBuf,
}

/// Builder for text datasets
#[derive(Debug, Default)]
pub struct TextDatasetBuilder {
    config: TextConfig,
}

impl TextDatasetBuilder {
    /// Create a new text dataset builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the file path
    pub fn file_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.config.file_path = path.as_ref().to_path_buf();
        self
    }

    /// Set maximum sequence length
    pub fn max_sequence_length(mut self, length: usize) -> Self {
        self.config.max_sequence_length = length;
        self
    }

    /// Set tokenization strategy
    pub fn tokenization_strategy(mut self, strategy: TokenizationStrategy) -> Self {
        self.config.tokenization_strategy = strategy;
        self
    }

    /// Set label extraction strategy
    pub fn label_strategy(mut self, strategy: LabelStrategy) -> Self {
        self.config.label_strategy = strategy;
        self
    }

    /// Enable or disable lowercase conversion
    pub fn lowercase(mut self, lowercase: bool) -> Self {
        self.config.lowercase = lowercase;
        self
    }

    /// Set minimum token frequency
    pub fn min_frequency(mut self, freq: usize) -> Self {
        self.config.min_frequency = freq;
        self
    }

    /// Set maximum vocabulary size
    pub fn max_vocab_size(mut self, size: usize) -> Self {
        self.config.max_vocab_size = Some(size);
        self
    }

    /// Set pad token
    pub fn pad_token<S: Into<String>>(mut self, token: S) -> Self {
        self.config.pad_token = token.into();
        self
    }

    /// Set unknown token
    pub fn unk_token<S: Into<String>>(mut self, token: S) -> Self {
        self.config.unk_token = token.into();
        self
    }

    /// Build the text dataset
    pub fn build<T>(self) -> Result<TextDataset<T>>
    where
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + scirs2_core::numeric::ToPrimitive
            + scirs2_core::numeric::NumCast
            + Send
            + Sync
            + 'static,
    {
        TextDataset::from_config(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_text_config_default() {
        let config = TextConfig::default();
        assert_eq!(config.max_sequence_length, 512);
        assert_eq!(config.pad_token, "<pad>");
        assert_eq!(config.unk_token, "<unk>");
        assert!(config.lowercase);
        assert_eq!(config.min_frequency, 2);
    }

    #[test]
    fn test_vocabulary_creation() {
        let texts = vec![
            "hello world".to_string(),
            "world peace".to_string(),
            "hello peace".to_string(),
        ];

        let config = TextConfig::default();
        let vocab = Vocabulary::from_texts(&texts, &config);

        assert!(vocab.len() > 0);
        assert!(vocab.token_to_id.contains_key("hello"));
        assert!(vocab.token_to_id.contains_key("world"));
        assert!(vocab.token_to_id.contains_key("peace"));
        assert!(vocab.token_to_id.contains_key(&config.pad_token));
        assert!(vocab.token_to_id.contains_key(&config.unk_token));
    }

    #[test]
    fn test_tokenization_strategies() {
        let text = "Hello, world!";

        let word_tokens = Vocabulary::tokenize_text(text, &TokenizationStrategy::WordLevel, true);
        assert_eq!(word_tokens, vec!["hello,", "world!"]);

        let char_tokens = Vocabulary::tokenize_text(text, &TokenizationStrategy::CharLevel, true);
        assert!(char_tokens.len() > word_tokens.len());
        assert!(char_tokens.contains(&"h".to_string()));
        assert!(char_tokens.contains(&"!".to_string()));

        let subword_tokens = Vocabulary::tokenize_text(text, &TokenizationStrategy::Subword, true);
        assert!(subword_tokens.contains(&"hello".to_string()));
        assert!(subword_tokens.contains(&",".to_string()));
        assert!(subword_tokens.contains(&"world".to_string()));
        assert!(subword_tokens.contains(&"!".to_string()));
    }

    #[test]
    fn test_encode_decode() {
        let texts = vec!["hello world".to_string(), "world peace".to_string()];

        let mut config = TextConfig::default();
        config.min_frequency = 1; // Allow single occurrence tokens
        let vocab = Vocabulary::from_texts(&texts, &config);

        let encoded = vocab.encode("hello world", &config);
        assert_eq!(encoded.len(), config.max_sequence_length);

        let decoded = vocab.decode(&encoded);

        // The decoded text should contain the original words
        // Note: case might be different due to lowercase processing
        assert!(decoded.contains("hello") || decoded.contains("Hello"));
        assert!(decoded.contains("world") || decoded.contains("World"));
    }

    #[test]
    fn test_text_dataset_from_file() {
        // Create a temporary text file
        let mut temp_file = NamedTempFile::new().unwrap();
        let text_content =
            "This is a positive review\nThis is another positive review\nGreat product!";
        temp_file.write_all(text_content.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        let dataset = TextDataset::<f32>::from_file(temp_file.path()).unwrap();

        assert_eq!(dataset.len(), 3);
        assert!(dataset.vocabulary().len() > 0);

        let (features, label) = dataset.get(0).unwrap();
        assert_eq!(
            features.shape().dims()[0],
            dataset.config().max_sequence_length
        );
    }

    #[test]
    fn test_text_dataset_builder() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let text_content = "Short text\nAnother text";
        temp_file.write_all(text_content.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        let dataset = TextDatasetBuilder::new()
            .file_path(temp_file.path())
            .max_sequence_length(10)
            .tokenization_strategy(TokenizationStrategy::WordLevel)
            .lowercase(true)
            .min_frequency(1)
            .build::<f32>()
            .unwrap();

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.config().max_sequence_length, 10);

        let info = dataset.info();
        assert_eq!(info.sample_count, 2);
        assert_eq!(info.max_sequence_length, 10);
    }

    #[test]
    fn test_label_strategies() {
        // Test filename-based labeling
        let mut temp_file = NamedTempFile::with_prefix("positive_").unwrap();
        let text_content = "This is positive text";
        temp_file.write_all(text_content.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        let config = TextConfig {
            file_path: temp_file.path().to_path_buf(),
            label_strategy: LabelStrategy::FromFilename,
            ..Default::default()
        };

        let dataset = TextDataset::<f32>::from_config(config).unwrap();
        assert!(dataset.label_vocabulary().is_some());
        assert!(dataset.label_vocabulary().unwrap().contains_key("positive"));
    }

    #[test]
    fn test_vocabulary_size_limit() {
        let texts = vec![
            "a b c d e".to_string(),
            "f g h i j".to_string(),
            "k l m n o".to_string(),
        ];

        let config = TextConfig {
            max_vocab_size: Some(8), // Very small vocabulary
            min_frequency: 1,
            ..Default::default()
        };

        let vocab = Vocabulary::from_texts(&texts, &config);

        // Should have special tokens + limited regular tokens
        assert!(vocab.len() <= 8);
        assert!(vocab.token_to_id.contains_key(&config.pad_token));
        assert!(vocab.token_to_id.contains_key(&config.unk_token));
    }

    #[test]
    fn test_empty_text_file() {
        let temp_file = NamedTempFile::new().unwrap();
        // Empty file

        let result = TextDataset::<f32>::from_file(temp_file.path());
        assert!(result.is_err());
    }
}
