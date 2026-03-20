//! Data Augmentation Utilities
//!
//! This module provides comprehensive data augmentation techniques for neural networks.
//! Supports image, text, and sequence augmentation strategies.
//!
//! Image augmentations expect tensors of shape `[C, H, W]` or `[N, C, H, W]`.
//! Sequence augmentations expect tensors of shape `[L]`, `[N, L]`, or `[N, F, L]`.

mod image_ops;
mod sequence_ops;

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use tenflowers_core::{Result, Tensor, TensorError};

use scirs2_core::num_traits::{Float, FromPrimitive, Zero};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

use image_ops::*;
use sequence_ops::*;

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

    /// Apply image augmentations to a tensor.
    ///
    /// The tensor is expected to have shape `[C, H, W]` or `[N, C, H, W]`.
    /// Each augmentation in the pipeline is applied sequentially; a random
    /// coin flip (per the config probability) determines whether each step
    /// is actually applied.
    pub fn apply_image<T>(&self, input: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Float + FromPrimitive + Clone + Default + Zero,
    {
        let mut result = input.clone();

        for (aug, config) in &self.image_augmentations {
            if !config.enabled {
                continue;
            }

            let mut rng = make_rng(config.seed);
            let coin: f64 = rng.random_range(0.0..1.0);
            if coin > config.probability as f64 {
                continue;
            }

            result = self.apply_single_image_augmentation(*aug, &result, config, &mut rng)?;
        }

        Ok(result)
    }

    /// Apply sequence augmentations to a tensor.
    ///
    /// The tensor may have shape `[L]`, `[N, L]`, or `[N, F, L]`.
    pub fn apply_sequence<T>(&self, input: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Float + FromPrimitive + Clone + Default + Zero,
    {
        let mut result = input.clone();

        for (aug, config) in &self.sequence_augmentations {
            if !config.enabled {
                continue;
            }

            let mut rng = make_rng(config.seed);
            let coin: f64 = rng.random_range(0.0..1.0);
            if coin > config.probability as f64 {
                continue;
            }

            result = self.apply_single_sequence_augmentation(*aug, &result, config, &mut rng)?;
        }

        Ok(result)
    }

    /// Apply a single image augmentation to the tensor.
    ///
    /// Supports tensors of shape `[C, H, W]` (single image) or `[N, C, H, W]` (batch).
    /// For batch tensors the same transformation parameters are applied to every sample.
    fn apply_single_image_augmentation<T>(
        &self,
        aug: ImageAugmentation,
        input: &Tensor<T>,
        config: &AugmentationConfig,
        rng: &mut StdRng,
    ) -> Result<Tensor<T>>
    where
        T: Float + FromPrimitive + Clone + Default + Zero,
    {
        let dims = input.shape().dims();
        let ndim = dims.len();

        // Determine spatial dimensions (H, W)
        let (height, width) = if ndim == 3 {
            (dims[1], dims[2])
        } else if ndim == 4 {
            (dims[2], dims[3])
        } else {
            return Err(TensorError::invalid_shape_simple(format!(
                "Image augmentation expects 3-D [C,H,W] or 4-D [N,C,H,W] tensor, got {ndim}-D"
            )));
        };

        match aug {
            ImageAugmentation::HorizontalFlip => apply_horizontal_flip(input, dims, height, width),
            ImageAugmentation::VerticalFlip => apply_vertical_flip(input, dims, height, width),
            ImageAugmentation::Rotation => {
                let max_angle = config.get_parameter("max_angle").unwrap_or(15.0);
                let angle: f64 = rng.random_range(-max_angle as f64..max_angle as f64);
                apply_rotation(input, dims, height, width, angle)
            }
            ImageAugmentation::Scaling => {
                let scale_range = config.get_parameter("scale_range").unwrap_or(0.2) as f64;
                let scale: f64 = 1.0 + rng.random_range(-scale_range..scale_range);
                apply_scaling(input, dims, height, width, scale)
            }
            ImageAugmentation::Translation => {
                let max_shift = config.get_parameter("max_shift").unwrap_or(4.0) as f64;
                let dx: f64 = rng.random_range(-max_shift..max_shift);
                let dy: f64 = rng.random_range(-max_shift..max_shift);
                apply_translation(input, dims, height, width, dx, dy)
            }
            ImageAugmentation::Brightness => {
                let factor = config.get_parameter("factor").unwrap_or(0.2) as f64;
                let delta: f64 = rng.random_range(-factor..factor);
                apply_brightness(input, delta)
            }
            ImageAugmentation::Contrast => {
                let factor = config.get_parameter("factor").unwrap_or(0.2) as f64;
                let alpha: f64 = 1.0 + rng.random_range(-factor..factor);
                apply_contrast(input, dims, height, width, alpha)
            }
            ImageAugmentation::Saturation => {
                let factor = config.get_parameter("factor").unwrap_or(0.3) as f64;
                let alpha: f64 = 1.0 + rng.random_range(-factor..factor);
                apply_saturation(input, dims, alpha)
            }
            ImageAugmentation::Hue => {
                let max_delta = config.get_parameter("max_delta").unwrap_or(0.1) as f64;
                let delta: f64 = rng.random_range(-max_delta..max_delta);
                apply_hue_shift(input, delta)
            }
            ImageAugmentation::Crop => {
                let crop_frac = config.get_parameter("crop_fraction").unwrap_or(0.8) as f64;
                apply_random_crop_and_resize(input, dims, height, width, crop_frac, rng)
            }
            ImageAugmentation::Zoom => {
                let zoom_range = config.get_parameter("zoom_range").unwrap_or(0.2) as f64;
                let zoom: f64 = 1.0 + rng.random_range(0.0..zoom_range);
                apply_scaling(input, dims, height, width, zoom)
            }
            ImageAugmentation::Shear => {
                let max_shear = config.get_parameter("max_shear").unwrap_or(0.2) as f64;
                let shear_x: f64 = rng.random_range(-max_shear..max_shear);
                let shear_y: f64 = rng.random_range(-max_shear..max_shear);
                apply_shear(input, dims, height, width, shear_x, shear_y)
            }
            ImageAugmentation::GaussianNoise => {
                let std_dev = config.get_parameter("std").unwrap_or(0.1) as f64;
                apply_gaussian_noise(input, std_dev, rng)
            }
            ImageAugmentation::SaltPepperNoise => {
                let noise_prob = config.get_parameter("probability").unwrap_or(0.02) as f64;
                apply_salt_pepper_noise(input, noise_prob, rng)
            }
            ImageAugmentation::RandomErasing => {
                let area_ratio = config.get_parameter("area_ratio").unwrap_or(0.15) as f64;
                apply_random_erasing(input, dims, height, width, area_ratio, rng)
            }
            ImageAugmentation::Cutout => {
                let size = config.get_parameter("size").unwrap_or(16.0) as usize;
                apply_cutout(input, dims, height, width, size, rng)
            }
            ImageAugmentation::Mixup => {
                let alpha = config.get_parameter("alpha").unwrap_or(0.2) as f64;
                apply_self_mixup(input, alpha, rng)
            }
            ImageAugmentation::CutMix => {
                let alpha = config.get_parameter("alpha").unwrap_or(1.0) as f64;
                apply_self_cutmix(input, dims, height, width, alpha, rng)
            }
        }
    }

    /// Apply a single sequence augmentation to the tensor.
    fn apply_single_sequence_augmentation<T>(
        &self,
        aug: SequenceAugmentation,
        input: &Tensor<T>,
        config: &AugmentationConfig,
        rng: &mut StdRng,
    ) -> Result<Tensor<T>>
    where
        T: Float + FromPrimitive + Clone + Default + Zero,
    {
        match aug {
            SequenceAugmentation::WordDeletion => {
                let prob = config.get_parameter("deletion_prob").unwrap_or(0.1) as f64;
                apply_token_deletion(input, prob, rng)
            }
            SequenceAugmentation::WordInsertion => {
                let prob = config.get_parameter("insertion_prob").unwrap_or(0.1) as f64;
                apply_token_noise_insertion(input, prob, rng)
            }
            SequenceAugmentation::WordSwap => {
                let n_swaps = config.get_parameter("n_swaps").unwrap_or(2.0) as usize;
                apply_token_swap(input, n_swaps, rng)
            }
            SequenceAugmentation::WordSubstitution => {
                let prob = config.get_parameter("substitution_prob").unwrap_or(0.1) as f64;
                apply_token_substitution(input, prob, rng)
            }
            SequenceAugmentation::BackTranslation | SequenceAugmentation::SynonymReplacement => {
                // These require external dictionaries / models; apply additive noise
                // as a differentiable proxy that perturbs embeddings similarly to how
                // paraphrasing / synonym replacement would shift representations.
                let std_dev = config.get_parameter("noise_std").unwrap_or(0.05) as f64;
                apply_gaussian_noise(input, std_dev, rng)
            }
            SequenceAugmentation::NoiseInsertion => {
                let std_dev = config.get_parameter("std").unwrap_or(0.1) as f64;
                apply_gaussian_noise(input, std_dev, rng)
            }
            SequenceAugmentation::Reversal => apply_sequence_reversal(input),
            SequenceAugmentation::TimeWarping => {
                let warp_factor = config.get_parameter("warp_factor").unwrap_or(0.1) as f64;
                apply_time_warping(input, warp_factor, rng)
            }
            SequenceAugmentation::TimeMasking => {
                let max_mask = config.get_parameter("max_time_mask").unwrap_or(100.0) as usize;
                apply_time_masking(input, max_mask, rng)
            }
            SequenceAugmentation::FrequencyMasking => {
                let max_mask = config.get_parameter("max_freq_mask").unwrap_or(27.0) as usize;
                apply_frequency_masking(input, max_mask, rng)
            }
        }
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

// ---------------------------------------------------------------------------
// Shared helpers used by both image_ops and sequence_ops
// ---------------------------------------------------------------------------

fn make_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(12345);
            StdRng::seed_from_u64(t)
        }
    }
}

fn f64_to_t<T: Float + FromPrimitive>(val: f64) -> Result<T> {
    T::from_f64(val).ok_or_else(|| {
        TensorError::invalid_argument(format!("Cannot convert {val} to target type"))
    })
}

fn box_muller(rng: &mut StdRng) -> (f64, f64) {
    let u1: f64 = rng.random_range(1e-10_f64..1.0_f64);
    let u2: f64 = rng.random_range(0.0_f64..1.0_f64);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

/// Sample from Beta(a, b) using Johnk's algorithm.
fn sample_beta(a: f64, b: f64, rng: &mut StdRng) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        return 0.5;
    }
    for _ in 0..200 {
        let u: f64 = rng.random_range(1e-10_f64..1.0_f64);
        let v: f64 = rng.random_range(1e-10_f64..1.0_f64);
        let x = u.powf(1.0 / a);
        let y = v.powf(1.0 / b);
        let s = x + y;
        if s <= 1.0 && s > 0.0 {
            return x / s;
        }
    }
    0.5
}

/// Fisher-Yates shuffle for a mutable slice.
fn fisher_yates_shuffle(arr: &mut [usize], rng: &mut StdRng) {
    let n = arr.len();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        arr.swap(i, j);
    }
}

// ---------------------------------------------------------------------------
// Pre-configured augmentation pipelines for common use cases
// ---------------------------------------------------------------------------

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
