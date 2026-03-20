//! Main multimodal dataset implementation

use super::{
    config::MultimodalConfig,
    sample::MultimodalSample,
    types::{FusionStrategy, Modality},
};
use crate::Dataset;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor, TensorError};

/// A dataset containing multimodal samples
#[derive(Debug, Clone)]
pub struct MultimodalDataset<T> {
    samples: Vec<MultimodalSample<T>>,
    config: MultimodalConfig,
}

impl<T> MultimodalDataset<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    /// Create a new multimodal dataset
    pub fn new(samples: Vec<MultimodalSample<T>>, config: MultimodalConfig) -> Result<Self> {
        // Validate that all samples have required modalities
        if config.strict_validation {
            for (i, sample) in samples.iter().enumerate() {
                for modality in &config.required_modalities {
                    if !sample.has_modality(modality) {
                        return Err(TensorError::invalid_argument(format!(
                            "Sample {} missing required modality: {:?}",
                            i, modality
                        )));
                    }
                }
            }
        }

        Ok(Self { samples, config })
    }

    /// Create an empty multimodal dataset
    pub fn empty(config: MultimodalConfig) -> Self {
        Self {
            samples: Vec::new(),
            config,
        }
    }

    /// Add a sample to the dataset
    pub fn add_sample(&mut self, sample: MultimodalSample<T>) -> Result<()> {
        // Validate required modalities if strict validation is enabled
        if self.config.strict_validation {
            for modality in &self.config.required_modalities {
                if !sample.has_modality(modality) {
                    return Err(TensorError::invalid_argument(format!(
                        "Sample missing required modality: {:?}",
                        modality
                    )));
                }
            }
        }

        self.samples.push(sample);
        Ok(())
    }

    /// Get configuration
    pub fn config(&self) -> &MultimodalConfig {
        &self.config
    }

    /// Get mutable configuration
    pub fn config_mut(&mut self) -> &mut MultimodalConfig {
        &mut self.config
    }

    /// Get sample statistics by modality
    pub fn modality_statistics(&self) -> HashMap<Modality, usize> {
        let mut stats = HashMap::new();

        for sample in &self.samples {
            for modality in sample.available_modalities() {
                *stats.entry(modality).or_insert(0) += 1;
            }
        }

        stats
    }

    /// Filter samples that have all specified modalities
    pub fn filter_by_modalities(&self, modalities: &[Modality]) -> Vec<usize> {
        self.samples
            .iter()
            .enumerate()
            .filter_map(|(i, sample)| {
                if modalities.iter().all(|m| sample.has_modality(m)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get multimodal sample by index
    pub fn get_multimodal(&self, index: usize) -> Result<&MultimodalSample<T>> {
        self.samples.get(index).ok_or_else(|| {
            TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index,
                self.samples.len()
            ))
        })
    }

    /// Get mutable multimodal sample by index
    pub fn get_multimodal_mut(&mut self, index: usize) -> Result<&mut MultimodalSample<T>> {
        let len = self.samples.len();
        self.samples.get_mut(index).ok_or_else(|| {
            TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index, len
            ))
        })
    }

    /// Remove a sample at the given index
    pub fn remove_sample(&mut self, index: usize) -> Result<MultimodalSample<T>> {
        if index >= self.samples.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index,
                self.samples.len()
            )));
        }
        Ok(self.samples.remove(index))
    }

    /// Get all samples
    pub fn samples(&self) -> &[MultimodalSample<T>] {
        &self.samples
    }

    /// Get all samples mutably
    pub fn samples_mut(&mut self) -> &mut [MultimodalSample<T>] {
        &mut self.samples
    }

    /// Extend dataset with additional samples
    pub fn extend(&mut self, other_samples: Vec<MultimodalSample<T>>) -> Result<()> {
        if self.config.strict_validation {
            for (i, sample) in other_samples.iter().enumerate() {
                for modality in &self.config.required_modalities {
                    if !sample.has_modality(modality) {
                        return Err(TensorError::invalid_argument(format!(
                            "Additional sample {} missing required modality: {:?}",
                            i, modality
                        )));
                    }
                }
            }
        }

        self.samples.extend(other_samples);
        Ok(())
    }

    /// Clear all samples
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

impl<T> Dataset<T> for MultimodalDataset<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::Float
        + Send
        + Sync
        + 'static,
{
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        let sample = self.get_multimodal(index)?;

        // Fuse modalities according to strategy
        let fused_features = self.fuse_modalities(sample)?;

        Ok((fused_features, sample.label.clone()))
    }
}

impl<T> MultimodalDataset<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::Float
        + Send
        + Sync
        + 'static,
{
    /// Fuse modalities according to the configured strategy
    pub fn fuse_modalities(&self, sample: &MultimodalSample<T>) -> Result<Tensor<T>> {
        match &self.config.fusion_strategy {
            FusionStrategy::Concatenation => self.concatenate_modalities(sample),
            FusionStrategy::Separate => {
                // For separate strategy, just return the first available modality
                for modality in &self.config.required_modalities {
                    if let Some(tensor) = sample.get_modality(modality) {
                        return Ok(tensor.clone());
                    }
                }
                Err(TensorError::invalid_argument(
                    "No required modality found".to_string(),
                ))
            }
            FusionStrategy::EarlyFusion => self.concatenate_modalities(sample), // Same as concatenation for now
            FusionStrategy::LateFusion => self.concatenate_modalities(sample), // Same as concatenation for now
            FusionStrategy::Attention => {
                // Simplified attention fusion - would need more sophisticated implementation
                self.concatenate_modalities(sample)
            }
        }
    }

    /// Concatenate features from all available modalities
    fn concatenate_modalities(&self, sample: &MultimodalSample<T>) -> Result<Tensor<T>> {
        let mut all_features = Vec::new();

        // Collect features from all available modalities in a consistent order
        let modalities = [
            Modality::Text,
            Modality::Image,
            Modality::Audio,
            Modality::Video,
            Modality::Embeddings,
        ];

        for modality in &modalities {
            if let Some(tensor) = sample.get_modality(modality) {
                // Flatten tensor to 1D for concatenation
                let flattened = tenflowers_core::ops::reshape(tensor, &[tensor.shape().size()])?;
                if let Some(data) = flattened.as_slice() {
                    all_features.extend_from_slice(data);
                } else {
                    return Err(TensorError::invalid_operation_simple(
                        "Cannot access tensor data (GPU tensor not supported in fusion)"
                            .to_string(),
                    ));
                }
            } else if self.config.pad_missing_modalities
                && self.config.optional_modalities.contains(modality)
            {
                // Add zeros for missing optional modalities
                let padding_size = self.get_expected_modality_size(modality);
                all_features.extend(vec![T::zero(); padding_size]);
            }
        }

        // Add custom modalities
        for tensor in sample.custom.values() {
            let flattened = tenflowers_core::ops::reshape(tensor, &[tensor.shape().size()])?;
            if let Some(data) = flattened.as_slice() {
                all_features.extend_from_slice(data);
            }
        }

        if all_features.is_empty() {
            return Err(TensorError::invalid_argument(
                "No features to fuse".to_string(),
            ));
        }

        let length = all_features.len();
        Tensor::from_vec(all_features, &[length])
    }

    /// Get expected size for a modality when padding
    fn get_expected_modality_size(&self, modality: &Modality) -> usize {
        match modality {
            Modality::Text => self.config.max_text_length.unwrap_or(512),
            Modality::Image => {
                if let Some((w, h)) = self.config.target_image_size {
                    (w as usize) * (h as usize) * 3 // RGB
                } else {
                    224 * 224 * 3
                }
            }
            Modality::Audio => self.config.target_audio_sample_rate.unwrap_or(16000) as usize,
            Modality::Video => 224 * 224 * 3 * 10, // 10 frames
            Modality::Embeddings => 768,           // Common embedding dimension
            Modality::Custom(_) => 256,            // Default size for custom modalities
        }
    }

    /// Get dataset summary statistics
    pub fn summary(&self) -> MultimodalDatasetSummary {
        let modality_stats = self.modality_statistics();
        let total_samples = self.len();

        let coverage = modality_stats
            .iter()
            .map(|(modality, count)| (modality.clone(), *count as f64 / total_samples as f64))
            .collect();

        MultimodalDatasetSummary {
            total_samples,
            modality_stats,
            modality_coverage: coverage,
            config: self.config.clone(),
        }
    }
}

/// Summary statistics for a multimodal dataset
#[derive(Debug, Clone)]
pub struct MultimodalDatasetSummary {
    pub total_samples: usize,
    pub modality_stats: HashMap<Modality, usize>,
    pub modality_coverage: HashMap<Modality, f64>,
    pub config: MultimodalConfig,
}

impl std::fmt::Display for MultimodalDatasetSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Multimodal Dataset Summary:")?;
        writeln!(f, "  Total samples: {}", self.total_samples)?;
        writeln!(f, "  Fusion strategy: {}", self.config.fusion_strategy)?;
        writeln!(
            f,
            "  Required modalities: {:?}",
            self.config.required_modalities
        )?;
        writeln!(
            f,
            "  Optional modalities: {:?}",
            self.config.optional_modalities
        )?;
        writeln!(f, "  Modality coverage:")?;

        for (modality, coverage) in &self.modality_coverage {
            writeln!(
                f,
                "    {}: {:.2}% ({} samples)",
                modality,
                coverage * 100.0,
                self.modality_stats[modality]
            )?;
        }

        Ok(())
    }
}
