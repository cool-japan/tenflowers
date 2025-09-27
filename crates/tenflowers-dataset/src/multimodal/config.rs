//! Configuration for multimodal datasets

use super::types::{FusionStrategy, Modality};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Configuration for multimodal dataset fusion
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct MultimodalConfig {
    /// Required modalities (sample must have all of these)
    pub required_modalities: Vec<Modality>,
    /// Optional modalities (sample may have these)
    pub optional_modalities: Vec<Modality>,
    /// Fusion strategy for combining modalities
    pub fusion_strategy: FusionStrategy,
    /// Maximum sequence length for text modality
    pub max_text_length: Option<usize>,
    /// Target image size for image modality
    pub target_image_size: Option<(u32, u32)>,
    /// Sample rate for audio modality
    pub target_audio_sample_rate: Option<u32>,
    /// Whether to pad missing optional modalities with zeros
    pub pad_missing_modalities: bool,
    /// Validation mode - whether to strictly enforce required modalities
    pub strict_validation: bool,
    /// Cache fusion results for performance
    pub cache_fused_results: bool,
    /// Maximum cache size for fusion results
    pub max_cache_size: usize,
}

impl Default for MultimodalConfig {
    fn default() -> Self {
        Self {
            required_modalities: vec![Modality::Text],
            optional_modalities: vec![Modality::Image],
            fusion_strategy: FusionStrategy::Concatenation,
            max_text_length: Some(512),
            target_image_size: Some((224, 224)),
            target_audio_sample_rate: Some(16000),
            pad_missing_modalities: true,
            strict_validation: true,
            cache_fused_results: false,
            max_cache_size: 1000,
        }
    }
}

impl MultimodalConfig {
    /// Create a new configuration with minimal required settings
    pub fn minimal() -> Self {
        Self {
            required_modalities: vec![],
            optional_modalities: vec![Modality::Text, Modality::Image],
            fusion_strategy: FusionStrategy::Concatenation,
            max_text_length: None,
            target_image_size: None,
            target_audio_sample_rate: None,
            pad_missing_modalities: false,
            strict_validation: false,
            cache_fused_results: false,
            max_cache_size: 0,
        }
    }

    /// Create a configuration for text-only datasets
    pub fn text_only() -> Self {
        Self {
            required_modalities: vec![Modality::Text],
            optional_modalities: vec![],
            fusion_strategy: FusionStrategy::Concatenation,
            max_text_length: Some(512),
            target_image_size: None,
            target_audio_sample_rate: None,
            pad_missing_modalities: false,
            strict_validation: true,
            cache_fused_results: false,
            max_cache_size: 0,
        }
    }

    /// Create a configuration for vision-language datasets
    pub fn vision_language() -> Self {
        Self {
            required_modalities: vec![Modality::Text, Modality::Image],
            optional_modalities: vec![],
            fusion_strategy: FusionStrategy::Concatenation,
            max_text_length: Some(256),
            target_image_size: Some((224, 224)),
            target_audio_sample_rate: None,
            pad_missing_modalities: true,
            strict_validation: true,
            cache_fused_results: true,
            max_cache_size: 5000,
        }
    }

    /// Validate that all required modalities are present in available modalities
    pub fn validate_modalities(&self, available: &[Modality]) -> Result<(), String> {
        for required in &self.required_modalities {
            if !available.contains(required) {
                return Err(format!(
                    "Required modality {:?} not found in sample",
                    required
                ));
            }
        }
        Ok(())
    }

    /// Get all modalities (required + optional)
    pub fn all_modalities(&self) -> Vec<Modality> {
        let mut all = self.required_modalities.clone();
        all.extend(self.optional_modalities.clone());
        all.sort_by_key(|m| format!("{:?}", m)); // Sort for consistency
        all.dedup();
        all
    }

    /// Check if a modality is expected (required or optional)
    pub fn is_expected_modality(&self, modality: &Modality) -> bool {
        self.required_modalities.contains(modality) || self.optional_modalities.contains(modality)
    }
}
