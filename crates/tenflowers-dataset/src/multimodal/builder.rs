//! Builder pattern implementation for multimodal datasets

use super::{
    config::MultimodalConfig,
    dataset::MultimodalDataset,
    sample::MultimodalSample,
    types::{FusionStrategy, Modality},
};
use std::collections::HashMap;
use std::path::Path;
use tenflowers_core::{Result, Tensor, TensorError};

/// Builder for creating multimodal datasets
#[derive(Debug, Clone)]
pub struct MultimodalDatasetBuilder<T> {
    samples: Vec<MultimodalSample<T>>,
    config: MultimodalConfig,
}

impl<T> MultimodalDatasetBuilder<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            config: MultimodalConfig::default(),
        }
    }

    /// Set required modalities
    pub fn with_required_modalities(mut self, modalities: Vec<Modality>) -> Self {
        self.config.required_modalities = modalities;
        self
    }

    /// Set optional modalities
    pub fn with_optional_modalities(mut self, modalities: Vec<Modality>) -> Self {
        self.config.optional_modalities = modalities;
        self
    }

    /// Set fusion strategy
    pub fn with_fusion_strategy(mut self, strategy: FusionStrategy) -> Self {
        self.config.fusion_strategy = strategy;
        self
    }

    /// Set maximum text length
    pub fn with_max_text_length(mut self, length: usize) -> Self {
        self.config.max_text_length = Some(length);
        self
    }

    /// Set target image size
    pub fn with_target_image_size(mut self, width: u32, height: u32) -> Self {
        self.config.target_image_size = Some((width, height));
        self
    }

    /// Set target audio sample rate
    pub fn with_target_audio_sample_rate(mut self, sample_rate: u32) -> Self {
        self.config.target_audio_sample_rate = Some(sample_rate);
        self
    }

    /// Enable/disable padding for missing modalities
    pub fn with_padding(mut self, pad: bool) -> Self {
        self.config.pad_missing_modalities = pad;
        self
    }

    /// Enable/disable strict validation
    pub fn with_strict_validation(mut self, strict: bool) -> Self {
        self.config.strict_validation = strict;
        self
    }

    /// Enable/disable result caching
    pub fn with_caching(mut self, cache: bool, max_size: usize) -> Self {
        self.config.cache_fused_results = cache;
        self.config.max_cache_size = max_size;
        self
    }

    /// Set the complete configuration
    pub fn with_config(mut self, config: MultimodalConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a sample
    pub fn add_sample(mut self, sample: MultimodalSample<T>) -> Self {
        self.samples.push(sample);
        self
    }

    /// Add multiple samples at once
    pub fn add_samples(mut self, samples: Vec<MultimodalSample<T>>) -> Self {
        self.samples.extend(samples);
        self
    }

    /// Clear all samples
    pub fn clear_samples(mut self) -> Self {
        self.samples.clear();
        self
    }

    /// Get the current number of samples
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Get a reference to the current configuration
    pub fn config(&self) -> &MultimodalConfig {
        &self.config
    }

    /// Get a reference to the current samples
    pub fn samples(&self) -> &[MultimodalSample<T>] {
        &self.samples
    }

    /// Validate all current samples against the configuration
    pub fn validate(&self) -> Result<()> {
        if !self.config.strict_validation {
            return Ok(());
        }

        for (i, sample) in self.samples.iter().enumerate() {
            // Check required modalities
            for modality in &self.config.required_modalities {
                if !sample.has_modality(modality) {
                    return Err(TensorError::invalid_argument(format!(
                        "Sample {} missing required modality: {:?}",
                        i, modality
                    )));
                }
            }

            // Validate modality expectations
            let available = sample.available_modalities();
            for modality in &available {
                if !self.config.is_expected_modality(modality) {
                    return Err(TensorError::invalid_argument(format!(
                        "Sample {} contains unexpected modality: {:?}",
                        i, modality
                    )));
                }
            }
        }

        Ok(())
    }

    /// Get statistics about the current samples
    pub fn statistics(&self) -> BuilderStatistics {
        let mut modality_counts: HashMap<Modality, usize> = HashMap::new();
        let mut total_size = 0usize;

        for sample in &self.samples {
            total_size += 1;
            for modality in sample.available_modalities() {
                *modality_counts.entry(modality).or_insert(0) += 1;
            }
        }

        let modality_coverage: HashMap<Modality, f64> = modality_counts
            .iter()
            .map(|(modality, count)| {
                let coverage = if total_size > 0 {
                    *count as f64 / total_size as f64
                } else {
                    0.0
                };
                (modality.clone(), coverage)
            })
            .collect();

        BuilderStatistics {
            total_samples: total_size,
            modality_counts,
            modality_coverage,
            required_modalities: self.config.required_modalities.clone(),
            optional_modalities: self.config.optional_modalities.clone(),
        }
    }

    /// Build the dataset
    pub fn build(self) -> Result<MultimodalDataset<T>> {
        // Validate before building if strict validation is enabled
        if self.config.strict_validation {
            self.validate()?;
        }

        MultimodalDataset::new(self.samples, self.config)
    }

    /// Build the dataset without validation (even if strict validation is enabled)
    pub fn build_unchecked(self) -> MultimodalDataset<T> {
        let mut config = self.config;
        config.strict_validation = false;
        MultimodalDataset::new(self.samples, config).expect("Unchecked build should not fail")
    }
}

impl<T> Default for MultimodalDatasetBuilder<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about samples in the builder
#[derive(Debug, Clone)]
pub struct BuilderStatistics {
    pub total_samples: usize,
    pub modality_counts: HashMap<Modality, usize>,
    pub modality_coverage: HashMap<Modality, f64>,
    pub required_modalities: Vec<Modality>,
    pub optional_modalities: Vec<Modality>,
}

impl std::fmt::Display for BuilderStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Multimodal Dataset Builder Statistics:")?;
        writeln!(f, "  Total samples: {}", self.total_samples)?;
        writeln!(f, "  Required modalities: {:?}", self.required_modalities)?;
        writeln!(f, "  Optional modalities: {:?}", self.optional_modalities)?;
        writeln!(f, "  Modality coverage:")?;

        for (modality, coverage) in &self.modality_coverage {
            let count = self.modality_counts.get(modality).unwrap_or(&0);
            writeln!(
                f,
                "    {}: {:.2}% ({} samples)",
                modality,
                coverage * 100.0,
                count
            )?;
        }

        Ok(())
    }
}

/// Builder for creating specific types of multimodal datasets
impl<T> MultimodalDatasetBuilder<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    /// Create a builder configured for text-only datasets
    pub fn text_only() -> Self {
        Self::new().with_config(MultimodalConfig::text_only())
    }

    /// Create a builder configured for vision-language datasets
    pub fn vision_language() -> Self {
        Self::new().with_config(MultimodalConfig::vision_language())
    }

    /// Create a builder with minimal configuration (most flexible)
    pub fn minimal() -> Self {
        Self::new().with_config(MultimodalConfig::minimal())
    }
}
