//! Text Corpus Generation
//!
//! This module contains functionality for generating synthetic text corpora
//! for natural language processing tasks and text-based machine learning.

use super::core::DatasetGenerator;
use crate::Dataset;
use tenflowers_core::{Result, Tensor};

/// Configuration for synthetic text corpus generation
#[derive(Debug, Clone)]
pub struct TextCorpusConfig {
    pub vocab_size: usize,
    pub min_sequence_length: usize,
    pub max_sequence_length: usize,
    pub language_model: bool,
    pub task_type: TextSynthesisTask,
    pub random_seed: Option<u64>,
}

#[derive(Debug, Clone)]
pub enum TextSynthesisTask {
    Classification,
    LanguageModeling,
    SequenceToSequence,
    NamedEntityRecognition,
    SentimentAnalysis,
}

impl Default for TextCorpusConfig {
    fn default() -> Self {
        Self {
            vocab_size: 1000,
            min_sequence_length: 10,
            max_sequence_length: 50,
            language_model: false,
            task_type: TextSynthesisTask::Classification,
            random_seed: None,
        }
    }
}

impl TextCorpusConfig {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            ..Default::default()
        }
    }

    pub fn with_sequence_length(mut self, min_len: usize, max_len: usize) -> Self {
        self.min_sequence_length = min_len;
        self.max_sequence_length = max_len;
        self
    }

    pub fn with_task(mut self, task_type: TextSynthesisTask) -> Self {
        self.task_type = task_type;
        self
    }

    pub fn with_language_model(mut self, is_lm: bool) -> Self {
        self.language_model = is_lm;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }
}

/// Synthetic text corpus for NLP tasks
pub struct SyntheticTextCorpus {
    sequences: Vec<Vec<usize>>,
    labels: Vec<usize>,
    vocab: Vec<String>,
    config: TextCorpusConfig,
}

impl SyntheticTextCorpus {
    pub fn new(config: TextCorpusConfig, _n_samples: usize) -> Result<Self> {
        let mut rng = if let Some(seed) = config.random_seed {
            scirs2_core::random::Random::seed(seed)
        } else {
            scirs2_core::random::Random::seed(0)
        };

        let vocab = Self::generate_vocabulary(&config, &mut rng);

        // Generate sequences and labels based on task type
        let (sequences, labels) = Self::generate_sequences_and_labels(&config, &vocab, &mut rng)?;

        Ok(Self {
            sequences,
            labels,
            vocab,
            config,
        })
    }

    fn generate_vocabulary(
        config: &TextCorpusConfig,
        rng: &mut impl scirs2_core::random::Rng,
    ) -> Vec<String> {
        let mut vocab = Vec::with_capacity(config.vocab_size);

        // Add special tokens
        vocab.push("<PAD>".to_string());
        vocab.push("<UNK>".to_string());
        vocab.push("<START>".to_string());
        vocab.push("<END>".to_string());

        // Generate synthetic words
        let syllables = [
            "ba", "be", "bi", "bo", "bu", "ca", "ce", "ci", "co", "cu", "da", "de", "di", "do",
            "du", "fa", "fe", "fi", "fo", "fu", "ga", "ge", "gi", "go", "gu", "ha", "he", "hi",
            "ho", "hu", "ka", "ke", "ki", "ko", "ku", "la", "le", "li", "lo", "lu", "ma", "me",
            "mi", "mo", "mu", "na", "ne", "ni", "no", "nu", "pa", "pe", "pi", "po", "pu", "ra",
            "re", "ri", "ro", "ru", "sa", "se", "si", "so", "su", "ta", "te", "ti", "to", "tu",
            "va", "ve", "vi", "vo", "vu", "wa", "we", "wi", "wo", "wu", "xa", "xe", "xi", "xo",
            "xu", "ya", "ye", "yi", "yo", "yu", "za", "ze", "zi", "zo", "zu",
        ];

        while vocab.len() < config.vocab_size {
            let word_length = rng.random_range(1..4);
            let mut word = String::new();

            for _ in 0..word_length {
                let syllable = syllables[rng.random_range(0..syllables.len())];
                word.push_str(syllable);
            }

            if !vocab.contains(&word) {
                vocab.push(word);
            }
        }

        vocab
    }

    fn generate_sequences_and_labels(
        config: &TextCorpusConfig,
        vocab: &[String],
        rng: &mut impl scirs2_core::random::Rng,
    ) -> Result<(Vec<Vec<usize>>, Vec<usize>)> {
        let n_samples = 1000; // Default sample count
        let mut sequences = Vec::with_capacity(n_samples);
        let mut labels = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let sequence = Self::generate_sequence(config, vocab, rng);
            let label = Self::generate_label(config, &sequence, vocab, rng);

            sequences.push(sequence);
            labels.push(label);
        }

        Ok((sequences, labels))
    }

    fn generate_sequence(
        config: &TextCorpusConfig,
        vocab: &[String],
        rng: &mut impl scirs2_core::random::Rng,
    ) -> Vec<usize> {
        let seq_length =
            rng.random_range(config.min_sequence_length..config.max_sequence_length + 1);
        let mut sequence = Vec::with_capacity(seq_length);

        if config.language_model {
            sequence.push(2); // <START> token
        }

        for _ in 0..seq_length - if config.language_model { 2 } else { 0 } {
            let token_id = rng.random_range(4..vocab.len()); // Skip special tokens
            sequence.push(token_id);
        }

        if config.language_model {
            sequence.push(3); // <END> token
        }

        sequence
    }

    fn generate_label(
        config: &TextCorpusConfig,
        sequence: &[usize],
        _vocab: &[String],
        rng: &mut impl scirs2_core::random::Rng,
    ) -> usize {
        match config.task_type {
            TextSynthesisTask::Classification => {
                // Simple rule: label based on sequence length
                if sequence.len() > 30 {
                    1 // Long sequence class
                } else {
                    0 // Short sequence class
                }
            }
            TextSynthesisTask::SentimentAnalysis => {
                // Random sentiment: 0=negative, 1=neutral, 2=positive
                rng.random_range(0..3)
            }
            TextSynthesisTask::LanguageModeling => {
                // For language modeling, next token prediction
                sequence.last().copied().unwrap_or(0)
            }
            TextSynthesisTask::SequenceToSequence => {
                // Simple transformation rule
                sequence.len() % 10
            }
            TextSynthesisTask::NamedEntityRecognition => {
                // Random entity type
                rng.random_range(0..5) // 5 entity types
            }
        }
    }

    pub fn vocab(&self) -> &[String] {
        &self.vocab
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

impl Dataset<f32> for SyntheticTextCorpus {
    fn len(&self) -> usize {
        self.sequences.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<f32>, Tensor<f32>)> {
        if index >= self.len() {
            return Err(tenflowers_core::TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index,
                self.len()
            )));
        }

        let sequence = &self.sequences[index];
        let label = self.labels[index];

        // Convert sequence to tensor (pad to max length)
        let max_len = self.config.max_sequence_length;
        let mut padded_sequence = vec![0.0f32; max_len];
        for (i, &token) in sequence.iter().enumerate() {
            if i < max_len {
                padded_sequence[i] = token as f32;
            }
        }

        let feature_tensor = Tensor::from_vec(padded_sequence, &[max_len])?;
        let label_tensor = Tensor::from_vec(vec![label as f32], &[])?;

        Ok((feature_tensor, label_tensor))
    }
}

impl DatasetGenerator {
    /// Generate a synthetic text corpus
    pub fn make_text_corpus(config: TextCorpusConfig) -> Result<SyntheticTextCorpus> {
        SyntheticTextCorpus::new(config, 1000)
    }
}
