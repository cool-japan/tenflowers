//! Data Augmentation Utilities
//!
//! This module provides comprehensive data augmentation techniques for neural networks.
//! Supports image, text, and sequence augmentation strategies.

use std::collections::HashMap;
use tenflowers_core::{Device, Result, Tensor, TensorError};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Augmentation configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct AugmentationConfig {
    /// Whether augmentation is enabled
    pub enabled: bool,
    /// Probability of applying augmentation (0.0-1.0)
    pub probability: f32,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Augmentation-specific parameters
    pub parameters: HashMap<String, f32>,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            probability: 0.5,
            seed: None,
            parameters: HashMap::new(),
        }
    }
}

impl AugmentationConfig {
    /// Create new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set enabled flag
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set probability
    pub fn with_probability(mut self, probability: f32) -> Self {
        self.probability = probability.clamp(0.0, 1.0);
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Add parameter
    pub fn with_parameter(mut self, key: String, value: f32) -> Self {
        self.parameters.insert(key, value);
        self
    }

    /// Get parameter value
    pub fn get_parameter(&self, key: &str) -> Option<f32> {
        self.parameters.get(key).copied()
    }
}

/// Image augmentation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageAugmentation {
    /// Horizontal flip
    HorizontalFlip,
    /// Vertical flip
    VerticalFlip,
    /// Random rotation
    Rotation,
    /// Random scaling
    Scaling,
    /// Random translation
    Translation,
    /// Random brightness adjustment
    Brightness,
    /// Random contrast adjustment
    Contrast,
    /// Random saturation adjustment
    Saturation,
    /// Random hue adjustment
    Hue,
    /// Random crop
    Crop,
    /// Random zoom
    Zoom,
    /// Random shear
    Shear,
    /// Gaussian noise
    GaussianNoise,
    /// Salt and pepper noise
    SaltPepperNoise,
    /// Random erasing
    RandomErasing,
    /// Cutout augmentation
    Cutout,
    /// Mixup augmentation
    Mixup,
    /// CutMix augmentation
    CutMix,
}

impl ImageAugmentation {
    /// Get augmentation name
    pub fn name(&self) -> &'static str {
        match self {
            ImageAugmentation::HorizontalFlip => "horizontal_flip",
            ImageAugmentation::VerticalFlip => "vertical_flip",
            ImageAugmentation::Rotation => "rotation",
            ImageAugmentation::Scaling => "scaling",
            ImageAugmentation::Translation => "translation",
            ImageAugmentation::Brightness => "brightness",
            ImageAugmentation::Contrast => "contrast",
            ImageAugmentation::Saturation => "saturation",
            ImageAugmentation::Hue => "hue",
            ImageAugmentation::Crop => "crop",
            ImageAugmentation::Zoom => "zoom",
            ImageAugmentation::Shear => "shear",
            ImageAugmentation::GaussianNoise => "gaussian_noise",
            ImageAugmentation::SaltPepperNoise => "salt_pepper_noise",
            ImageAugmentation::RandomErasing => "random_erasing",
            ImageAugmentation::Cutout => "cutout",
            ImageAugmentation::Mixup => "mixup",
            ImageAugmentation::CutMix => "cutmix",
        }
    }
}

/// Text/sequence augmentation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceAugmentation {
    /// Random word deletion
    WordDeletion,
    /// Random word insertion
    WordInsertion,
    /// Random word swap
    WordSwap,
    /// Random word substitution
    WordSubstitution,
    /// Back translation
    BackTranslation,
    /// Synonym replacement
    SynonymReplacement,
    /// Random insertion of noise
    NoiseInsertion,
    /// Sequence reversal
    Reversal,
    /// Time warping
    TimeWarping,
    /// Time masking
    TimeMasking,
    /// Frequency masking
    FrequencyMasking,
}

impl SequenceAugmentation {
    /// Get augmentation name
    pub fn name(&self) -> &'static str {
        match self {
            SequenceAugmentation::WordDeletion => "word_deletion",
            SequenceAugmentation::WordInsertion => "word_insertion",
            SequenceAugmentation::WordSwap => "word_swap",
            SequenceAugmentation::WordSubstitution => "word_substitution",
            SequenceAugmentation::BackTranslation => "back_translation",
            SequenceAugmentation::SynonymReplacement => "synonym_replacement",
            SequenceAugmentation::NoiseInsertion => "noise_insertion",
            SequenceAugmentation::Reversal => "reversal",
            SequenceAugmentation::TimeWarping => "time_warping",
            SequenceAugmentation::TimeMasking => "time_masking",
            SequenceAugmentation::FrequencyMasking => "frequency_masking",
        }
    }
}

/// Augmentation pipeline for composing multiple augmentations
#[derive(Debug)]
pub struct AugmentationPipeline {
    image_augmentations: Vec<(ImageAugmentation, AugmentationConfig)>,
    sequence_augmentations: Vec<(SequenceAugmentation, AugmentationConfig)>,
}

impl AugmentationPipeline {
    /// Create new augmentation pipeline
    pub fn new() -> Self {
        Self {
            image_augmentations: Vec::new(),
            sequence_augmentations: Vec::new(),
        }
    }

    /// Add image augmentation
    pub fn add_image_augmentation(
        mut self,
        aug: ImageAugmentation,
        config: AugmentationConfig,
    ) -> Self {
        self.image_augmentations.push((aug, config));
        self
    }

    /// Add sequence augmentation
    pub fn add_sequence_augmentation(
        mut self,
        aug: SequenceAugmentation,
        config: AugmentationConfig,
    ) -> Self {
        self.sequence_augmentations.push((aug, config));
        self
    }

    /// Apply image augmentations
    pub fn apply_image<T>(&self, input: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Clone + Default,
    {
        let mut result = input.clone();

        for (aug, config) in &self.image_augmentations {
            if !config.enabled {
                continue;
            }

            // TODO: Implement actual augmentation logic
            // This is a placeholder for the augmentation framework
            result = self.apply_single_image_augmentation(*aug, &result, config)?;
        }

        Ok(result)
    }

    /// Apply sequence augmentations
    pub fn apply_sequence<T>(&self, input: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Clone + Default,
    {
        let mut result = input.clone();

        for (aug, config) in &self.sequence_augmentations {
            if !config.enabled {
                continue;
            }

            // TODO: Implement actual augmentation logic
            result = self.apply_single_sequence_augmentation(*aug, &result, config)?;
        }

        Ok(result)
    }

    /// Apply single image augmentation (placeholder)
    fn apply_single_image_augmentation<T>(
        &self,
        aug: ImageAugmentation,
        input: &Tensor<T>,
        config: &AugmentationConfig,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default,
    {
        // Placeholder implementation
        // Real implementation would apply actual transformations
        Ok(input.clone())
    }

    /// Apply single sequence augmentation (placeholder)
    fn apply_single_sequence_augmentation<T>(
        &self,
        aug: SequenceAugmentation,
        input: &Tensor<T>,
        config: &AugmentationConfig,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default,
    {
        // Placeholder implementation
        Ok(input.clone())
    }

    /// Get number of image augmentations
    pub fn num_image_augmentations(&self) -> usize {
        self.image_augmentations.len()
    }

    /// Get number of sequence augmentations
    pub fn num_sequence_augmentations(&self) -> usize {
        self.sequence_augmentations.len()
    }

    /// Get total number of augmentations
    pub fn total_augmentations(&self) -> usize {
        self.num_image_augmentations() + self.num_sequence_augmentations()
    }
}

impl Default for AugmentationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Pre-configured augmentation pipelines for common use cases
pub mod presets {
    use super::*;

    /// Standard image augmentation pipeline for classification
    pub fn standard_image_classification() -> AugmentationPipeline {
        AugmentationPipeline::new()
            .add_image_augmentation(
                ImageAugmentation::HorizontalFlip,
                AugmentationConfig::new().with_probability(0.5),
            )
            .add_image_augmentation(
                ImageAugmentation::Rotation,
                AugmentationConfig::new()
                    .with_probability(0.3)
                    .with_parameter("max_angle".to_string(), 15.0),
            )
            .add_image_augmentation(
                ImageAugmentation::Brightness,
                AugmentationConfig::new()
                    .with_probability(0.3)
                    .with_parameter("factor".to_string(), 0.2),
            )
            .add_image_augmentation(
                ImageAugmentation::Contrast,
                AugmentationConfig::new()
                    .with_probability(0.3)
                    .with_parameter("factor".to_string(), 0.2),
            )
    }

    /// Aggressive image augmentation for small datasets
    pub fn aggressive_image_augmentation() -> AugmentationPipeline {
        AugmentationPipeline::new()
            .add_image_augmentation(
                ImageAugmentation::HorizontalFlip,
                AugmentationConfig::new().with_probability(0.5),
            )
            .add_image_augmentation(
                ImageAugmentation::VerticalFlip,
                AugmentationConfig::new().with_probability(0.3),
            )
            .add_image_augmentation(
                ImageAugmentation::Rotation,
                AugmentationConfig::new()
                    .with_probability(0.5)
                    .with_parameter("max_angle".to_string(), 30.0),
            )
            .add_image_augmentation(
                ImageAugmentation::Scaling,
                AugmentationConfig::new()
                    .with_probability(0.5)
                    .with_parameter("scale_range".to_string(), 0.2),
            )
            .add_image_augmentation(
                ImageAugmentation::Brightness,
                AugmentationConfig::new()
                    .with_probability(0.5)
                    .with_parameter("factor".to_string(), 0.3),
            )
            .add_image_augmentation(
                ImageAugmentation::Contrast,
                AugmentationConfig::new()
                    .with_probability(0.5)
                    .with_parameter("factor".to_string(), 0.3),
            )
            .add_image_augmentation(
                ImageAugmentation::GaussianNoise,
                AugmentationConfig::new()
                    .with_probability(0.3)
                    .with_parameter("std".to_string(), 0.1),
            )
            .add_image_augmentation(
                ImageAugmentation::RandomErasing,
                AugmentationConfig::new()
                    .with_probability(0.3)
                    .with_parameter("area_ratio".to_string(), 0.15),
            )
    }

    /// Cutout/Mixup augmentation for modern training
    pub fn modern_image_augmentation() -> AugmentationPipeline {
        AugmentationPipeline::new()
            .add_image_augmentation(
                ImageAugmentation::HorizontalFlip,
                AugmentationConfig::new().with_probability(0.5),
            )
            .add_image_augmentation(
                ImageAugmentation::Cutout,
                AugmentationConfig::new()
                    .with_probability(0.5)
                    .with_parameter("size".to_string(), 16.0),
            )
            .add_image_augmentation(
                ImageAugmentation::Mixup,
                AugmentationConfig::new()
                    .with_probability(0.5)
                    .with_parameter("alpha".to_string(), 0.2),
            )
    }

    /// Standard text/NLP augmentation
    pub fn standard_text_augmentation() -> AugmentationPipeline {
        AugmentationPipeline::new()
            .add_sequence_augmentation(
                SequenceAugmentation::SynonymReplacement,
                AugmentationConfig::new()
                    .with_probability(0.3)
                    .with_parameter("n_replacements".to_string(), 3.0),
            )
            .add_sequence_augmentation(
                SequenceAugmentation::WordDeletion,
                AugmentationConfig::new()
                    .with_probability(0.2)
                    .with_parameter("deletion_prob".to_string(), 0.1),
            )
            .add_sequence_augmentation(
                SequenceAugmentation::WordSwap,
                AugmentationConfig::new()
                    .with_probability(0.2)
                    .with_parameter("n_swaps".to_string(), 2.0),
            )
    }

    /// Audio/speech augmentation
    pub fn standard_audio_augmentation() -> AugmentationPipeline {
        AugmentationPipeline::new()
            .add_sequence_augmentation(
                SequenceAugmentation::TimeWarping,
                AugmentationConfig::new()
                    .with_probability(0.5)
                    .with_parameter("warp_factor".to_string(), 0.1),
            )
            .add_sequence_augmentation(
                SequenceAugmentation::TimeMasking,
                AugmentationConfig::new()
                    .with_probability(0.5)
                    .with_parameter("max_time_mask".to_string(), 100.0),
            )
            .add_sequence_augmentation(
                SequenceAugmentation::FrequencyMasking,
                AugmentationConfig::new()
                    .with_probability(0.5)
                    .with_parameter("max_freq_mask".to_string(), 27.0),
            )
    }
}

/// Augmentation statistics tracker
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct AugmentationStats {
    /// Total number of augmentations applied
    pub total_applied: usize,
    /// Count per augmentation type
    pub type_counts: HashMap<String, usize>,
    /// Average application time in milliseconds
    pub avg_time_ms: f64,
}

impl AugmentationStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self {
            total_applied: 0,
            type_counts: HashMap::new(),
            avg_time_ms: 0.0,
        }
    }

    /// Record augmentation application
    pub fn record(&mut self, aug_name: &str, time_ms: f64) {
        self.total_applied += 1;
        *self.type_counts.entry(aug_name.to_string()).or_insert(0) += 1;

        // Update running average
        let n = self.total_applied as f64;
        self.avg_time_ms = (self.avg_time_ms * (n - 1.0) + time_ms) / n;
    }

    /// Get count for specific augmentation
    pub fn get_count(&self, aug_name: &str) -> usize {
        self.type_counts.get(aug_name).copied().unwrap_or(0)
    }

    /// Get most used augmentation
    pub fn most_used(&self) -> Option<(String, usize)> {
        self.type_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(name, &count)| (name.clone(), count))
    }

    /// Get least used augmentation
    pub fn least_used(&self) -> Option<(String, usize)> {
        self.type_counts
            .iter()
            .min_by_key(|(_, &count)| count)
            .map(|(name, &count)| (name.clone(), count))
    }
}

impl Default for AugmentationStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_augmentation_config_default() {
        let config = AugmentationConfig::default();
        assert!(config.enabled);
        assert_eq!(config.probability, 0.5);
        assert!(config.seed.is_none());
    }

    #[test]
    fn test_augmentation_config_builder() {
        let config = AugmentationConfig::new()
            .with_enabled(false)
            .with_probability(0.8)
            .with_seed(42)
            .with_parameter("test".to_string(), 1.5);

        assert!(!config.enabled);
        assert_eq!(config.probability, 0.8);
        assert_eq!(config.seed, Some(42));
        assert_eq!(config.get_parameter("test"), Some(1.5));
    }

    #[test]
    fn test_augmentation_config_probability_clamping() {
        let config1 = AugmentationConfig::new().with_probability(1.5);
        assert_eq!(config1.probability, 1.0);

        let config2 = AugmentationConfig::new().with_probability(-0.5);
        assert_eq!(config2.probability, 0.0);
    }

    #[test]
    fn test_image_augmentation_names() {
        assert_eq!(ImageAugmentation::HorizontalFlip.name(), "horizontal_flip");
        assert_eq!(ImageAugmentation::Rotation.name(), "rotation");
        assert_eq!(ImageAugmentation::Mixup.name(), "mixup");
    }

    #[test]
    fn test_sequence_augmentation_names() {
        assert_eq!(SequenceAugmentation::WordDeletion.name(), "word_deletion");
        assert_eq!(SequenceAugmentation::TimeWarping.name(), "time_warping");
    }

    #[test]
    fn test_augmentation_pipeline_creation() {
        let pipeline = AugmentationPipeline::new();
        assert_eq!(pipeline.num_image_augmentations(), 0);
        assert_eq!(pipeline.num_sequence_augmentations(), 0);
        assert_eq!(pipeline.total_augmentations(), 0);
    }

    #[test]
    fn test_augmentation_pipeline_add_image() {
        let pipeline = AugmentationPipeline::new().add_image_augmentation(
            ImageAugmentation::HorizontalFlip,
            AugmentationConfig::default(),
        );

        assert_eq!(pipeline.num_image_augmentations(), 1);
        assert_eq!(pipeline.total_augmentations(), 1);
    }

    #[test]
    fn test_augmentation_pipeline_add_sequence() {
        let pipeline = AugmentationPipeline::new().add_sequence_augmentation(
            SequenceAugmentation::WordSwap,
            AugmentationConfig::default(),
        );

        assert_eq!(pipeline.num_sequence_augmentations(), 1);
        assert_eq!(pipeline.total_augmentations(), 1);
    }

    #[test]
    fn test_augmentation_pipeline_multiple() {
        let pipeline = AugmentationPipeline::new()
            .add_image_augmentation(
                ImageAugmentation::HorizontalFlip,
                AugmentationConfig::default(),
            )
            .add_image_augmentation(ImageAugmentation::Rotation, AugmentationConfig::default())
            .add_sequence_augmentation(
                SequenceAugmentation::WordSwap,
                AugmentationConfig::default(),
            );

        assert_eq!(pipeline.num_image_augmentations(), 2);
        assert_eq!(pipeline.num_sequence_augmentations(), 1);
        assert_eq!(pipeline.total_augmentations(), 3);
    }

    #[test]
    fn test_preset_standard_image_classification() {
        let pipeline = presets::standard_image_classification();
        assert!(pipeline.num_image_augmentations() > 0);
    }

    #[test]
    fn test_preset_aggressive_image_augmentation() {
        let pipeline = presets::aggressive_image_augmentation();
        assert!(pipeline.num_image_augmentations() > 4);
    }

    #[test]
    fn test_preset_modern_image_augmentation() {
        let pipeline = presets::modern_image_augmentation();
        assert!(pipeline.num_image_augmentations() > 0);
    }

    #[test]
    fn test_preset_standard_text_augmentation() {
        let pipeline = presets::standard_text_augmentation();
        assert!(pipeline.num_sequence_augmentations() > 0);
    }

    #[test]
    fn test_preset_standard_audio_augmentation() {
        let pipeline = presets::standard_audio_augmentation();
        assert!(pipeline.num_sequence_augmentations() > 0);
    }

    #[test]
    fn test_augmentation_stats_creation() {
        let stats = AugmentationStats::new();
        assert_eq!(stats.total_applied, 0);
        assert_eq!(stats.avg_time_ms, 0.0);
    }

    #[test]
    fn test_augmentation_stats_record() {
        let mut stats = AugmentationStats::new();

        stats.record("flip", 1.0);
        stats.record("rotation", 2.0);
        stats.record("flip", 3.0);

        assert_eq!(stats.total_applied, 3);
        assert_eq!(stats.get_count("flip"), 2);
        assert_eq!(stats.get_count("rotation"), 1);
        assert_eq!(stats.avg_time_ms, 2.0); // (1 + 2 + 3) / 3
    }

    #[test]
    fn test_augmentation_stats_most_used() {
        let mut stats = AugmentationStats::new();

        stats.record("flip", 1.0);
        stats.record("flip", 1.0);
        stats.record("flip", 1.0);
        stats.record("rotation", 1.0);

        let most_used = stats.most_used();
        assert!(most_used.is_some());
        let (name, count) = most_used.unwrap();
        assert_eq!(name, "flip");
        assert_eq!(count, 3);
    }

    #[test]
    fn test_augmentation_stats_least_used() {
        let mut stats = AugmentationStats::new();

        stats.record("flip", 1.0);
        stats.record("flip", 1.0);
        stats.record("flip", 1.0);
        stats.record("rotation", 1.0);

        let least_used = stats.least_used();
        assert!(least_used.is_some());
        let (name, count) = least_used.unwrap();
        assert_eq!(name, "rotation");
        assert_eq!(count, 1);
    }
}
