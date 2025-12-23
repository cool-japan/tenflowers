//! Transform implementations for multimodal datasets

use super::{dataset::MultimodalDataset, sample::MultimodalSample};
use crate::Dataset;
use tenflowers_core::{Result, Tensor};

/// Transform trait for multimodal data
pub trait MultimodalTransform<T> {
    /// Apply transformation to a multimodal sample
    fn apply_multimodal(&self, sample: MultimodalSample<T>) -> Result<MultimodalSample<T>>;
}

/// Wrapper to apply multimodal transforms to datasets
#[derive(Debug, Clone)]
pub struct MultimodalTransformedDataset<T, Tr>
where
    Tr: MultimodalTransform<T>,
{
    dataset: MultimodalDataset<T>,
    transform: Tr,
}

impl<T, Tr> MultimodalTransformedDataset<T, Tr>
where
    Tr: MultimodalTransform<T>,
    T: Clone + Default + Send + Sync + 'static,
{
    /// Create a new transformed multimodal dataset
    pub fn new(dataset: MultimodalDataset<T>, transform: Tr) -> Self {
        Self { dataset, transform }
    }

    /// Get reference to the underlying dataset
    pub fn dataset(&self) -> &MultimodalDataset<T> {
        &self.dataset
    }

    /// Get reference to the transform
    pub fn transform(&self) -> &Tr {
        &self.transform
    }

    /// Unwrap into dataset and transform
    pub fn into_parts(self) -> (MultimodalDataset<T>, Tr) {
        (self.dataset, self.transform)
    }
}

impl<T, Tr> Dataset<T> for MultimodalTransformedDataset<T, Tr>
where
    Tr: MultimodalTransform<T>,
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::Float
        + Send
        + Sync
        + 'static,
{
    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        let sample = self.dataset.get_multimodal(index)?.clone();
        let transformed_sample = self.transform.apply_multimodal(sample)?;

        // Fuse and return
        let fused_features = self.dataset.fuse_modalities(&transformed_sample)?;
        Ok((fused_features, transformed_sample.label))
    }
}

/// Identity transform that does nothing
#[derive(Debug, Clone, Default)]
pub struct Identity;

impl<T> MultimodalTransform<T> for Identity {
    fn apply_multimodal(&self, sample: MultimodalSample<T>) -> Result<MultimodalSample<T>> {
        Ok(sample)
    }
}

/// Transform that applies different transforms to different modalities
pub struct ModalitySpecificTransform<T> {
    text_transform: Option<Box<dyn Fn(Tensor<T>) -> Result<Tensor<T>> + Send + Sync>>,
    image_transform: Option<Box<dyn Fn(Tensor<T>) -> Result<Tensor<T>> + Send + Sync>>,
    audio_transform: Option<Box<dyn Fn(Tensor<T>) -> Result<Tensor<T>> + Send + Sync>>,
    video_transform: Option<Box<dyn Fn(Tensor<T>) -> Result<Tensor<T>> + Send + Sync>>,
}

impl<T> ModalitySpecificTransform<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    /// Create a new modality-specific transform
    pub fn new() -> Self {
        Self {
            text_transform: None,
            image_transform: None,
            audio_transform: None,
            video_transform: None,
        }
    }

    /// Set text transform
    pub fn with_text_transform<F>(mut self, transform: F) -> Self
    where
        F: Fn(Tensor<T>) -> Result<Tensor<T>> + Send + Sync + 'static,
    {
        self.text_transform = Some(Box::new(transform));
        self
    }

    /// Set image transform
    pub fn with_image_transform<F>(mut self, transform: F) -> Self
    where
        F: Fn(Tensor<T>) -> Result<Tensor<T>> + Send + Sync + 'static,
    {
        self.image_transform = Some(Box::new(transform));
        self
    }

    /// Set audio transform
    pub fn with_audio_transform<F>(mut self, transform: F) -> Self
    where
        F: Fn(Tensor<T>) -> Result<Tensor<T>> + Send + Sync + 'static,
    {
        self.audio_transform = Some(Box::new(transform));
        self
    }

    /// Set video transform
    pub fn with_video_transform<F>(mut self, transform: F) -> Self
    where
        F: Fn(Tensor<T>) -> Result<Tensor<T>> + Send + Sync + 'static,
    {
        self.video_transform = Some(Box::new(transform));
        self
    }
}

impl<T> std::fmt::Debug for ModalitySpecificTransform<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModalitySpecificTransform")
            .field("text_transform", &self.text_transform.is_some())
            .field("image_transform", &self.image_transform.is_some())
            .field("audio_transform", &self.audio_transform.is_some())
            .field("video_transform", &self.video_transform.is_some())
            .finish()
    }
}

impl<T> Default for ModalitySpecificTransform<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> MultimodalTransform<T> for ModalitySpecificTransform<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    fn apply_multimodal(&self, mut sample: MultimodalSample<T>) -> Result<MultimodalSample<T>> {
        // Apply text transform
        if let Some(ref transform) = self.text_transform {
            if let Some(text_tensor) = sample.text.take() {
                sample.text = Some(transform(text_tensor)?);
            }
        }

        // Apply image transform
        if let Some(ref transform) = self.image_transform {
            if let Some(image_tensor) = sample.image.take() {
                sample.image = Some(transform(image_tensor)?);
            }
        }

        // Apply audio transform
        if let Some(ref transform) = self.audio_transform {
            if let Some(audio_tensor) = sample.audio.take() {
                sample.audio = Some(transform(audio_tensor)?);
            }
        }

        // Apply video transform
        if let Some(ref transform) = self.video_transform {
            if let Some(video_tensor) = sample.video.take() {
                sample.video = Some(transform(video_tensor)?);
            }
        }

        Ok(sample)
    }
}

/// Compose multiple transforms sequentially
pub struct ComposedTransform<T> {
    transforms: Vec<Box<dyn MultimodalTransform<T> + Send + Sync>>,
}

impl<T> ComposedTransform<T> {
    /// Create a new composed transform
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    /// Add a transform to the composition
    pub fn add_transform<Tr>(mut self, transform: Tr) -> Self
    where
        Tr: MultimodalTransform<T> + Send + Sync + 'static,
    {
        self.transforms.push(Box::new(transform));
        self
    }

    /// Create a composed transform from a vector of transforms
    pub fn from_transforms(transforms: Vec<Box<dyn MultimodalTransform<T> + Send + Sync>>) -> Self {
        Self { transforms }
    }
}

impl<T> std::fmt::Debug for ComposedTransform<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComposedTransform")
            .field("transforms_count", &self.transforms.len())
            .finish()
    }
}

impl<T> Default for ComposedTransform<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> MultimodalTransform<T> for ComposedTransform<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    fn apply_multimodal(&self, mut sample: MultimodalSample<T>) -> Result<MultimodalSample<T>> {
        for transform in &self.transforms {
            sample = transform.apply_multimodal(sample)?;
        }
        Ok(sample)
    }
}

/// Probabilistic transform that applies with given probability
#[derive(Debug, Clone)]
pub struct ProbabilisticTransform<T, Tr>
where
    Tr: MultimodalTransform<T>,
{
    transform: Tr,
    probability: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, Tr> ProbabilisticTransform<T, Tr>
where
    Tr: MultimodalTransform<T>,
{
    /// Create a new probabilistic transform
    pub fn new(transform: Tr, probability: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&probability),
            "Probability must be between 0.0 and 1.0"
        );
        Self {
            transform,
            probability,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the probability
    pub fn probability(&self) -> f64 {
        self.probability
    }

    /// Get reference to the underlying transform
    pub fn transform(&self) -> &Tr {
        &self.transform
    }
}

impl<T, Tr> MultimodalTransform<T> for ProbabilisticTransform<T, Tr>
where
    T: Clone + Default + Send + Sync + 'static,
    Tr: MultimodalTransform<T>,
{
    fn apply_multimodal(&self, sample: MultimodalSample<T>) -> Result<MultimodalSample<T>> {
        use scirs2_core::random::{rng, Random, Rng};

        let mut rng = rng();
        let random_value: f64 = rng.random();

        if random_value < self.probability {
            self.transform.apply_multimodal(sample)
        } else {
            Ok(sample)
        }
    }
}

/// Conditional transform that applies based on sample properties
#[derive(Debug, Clone)]
pub struct ConditionalTransform<T, Tr, F>
where
    Tr: MultimodalTransform<T>,
    F: Fn(&MultimodalSample<T>) -> bool,
{
    transform: Tr,
    condition: F,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, Tr, F> ConditionalTransform<T, Tr, F>
where
    Tr: MultimodalTransform<T>,
    F: Fn(&MultimodalSample<T>) -> bool,
{
    /// Create a new conditional transform
    pub fn new(transform: Tr, condition: F) -> Self {
        Self {
            transform,
            condition,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, Tr, F> MultimodalTransform<T> for ConditionalTransform<T, Tr, F>
where
    T: Clone + Default + Send + Sync + 'static,
    Tr: MultimodalTransform<T>,
    F: Fn(&MultimodalSample<T>) -> bool + Send + Sync,
{
    fn apply_multimodal(&self, sample: MultimodalSample<T>) -> Result<MultimodalSample<T>> {
        if (self.condition)(&sample) {
            self.transform.apply_multimodal(sample)
        } else {
            Ok(sample)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multimodal::types::Modality;
    use tenflowers_core::Tensor;

    #[test]
    fn test_identity_transform() {
        let label = Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap();
        let sample = MultimodalSample::new(label.clone())
            .with_text(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap());

        let transform = Identity;
        let transformed = transform.apply_multimodal(sample.clone()).unwrap();

        assert_eq!(
            transformed.available_modalities(),
            sample.available_modalities()
        );
    }

    #[test]
    fn test_composed_transform() {
        let label = Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap();
        let sample = MultimodalSample::new(label.clone())
            .with_text(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap());

        let transform = ComposedTransform::new()
            .add_transform(Identity)
            .add_transform(Identity);

        let transformed = transform.apply_multimodal(sample.clone()).unwrap();
        assert_eq!(
            transformed.available_modalities(),
            sample.available_modalities()
        );
    }

    #[test]
    fn test_probabilistic_transform() {
        let label = Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap();
        let sample = MultimodalSample::new(label.clone())
            .with_text(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap());

        // Test with probability 0.0 (should never apply)
        let transform = ProbabilisticTransform::new(Identity, 0.0);
        let transformed = transform.apply_multimodal(sample.clone()).unwrap();
        assert_eq!(
            transformed.available_modalities(),
            sample.available_modalities()
        );

        // Test with probability 1.0 (should always apply)
        let transform = ProbabilisticTransform::new(Identity, 1.0);
        let transformed = transform.apply_multimodal(sample.clone()).unwrap();
        assert_eq!(
            transformed.available_modalities(),
            sample.available_modalities()
        );
    }
}
