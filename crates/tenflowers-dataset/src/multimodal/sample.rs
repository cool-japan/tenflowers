//! Multimodal sample implementation

use super::types::Modality;
use std::collections::HashMap;
use tenflowers_core::Tensor;

/// A single multimodal sample containing data from multiple modalities
#[derive(Debug, Clone)]
pub struct MultimodalSample<T> {
    pub text: Option<Tensor<T>>,
    pub image: Option<Tensor<T>>,
    pub audio: Option<Tensor<T>>,
    pub video: Option<Tensor<T>>,
    pub embeddings: Option<Tensor<T>>,
    pub custom: HashMap<String, Tensor<T>>,
    pub metadata: HashMap<String, String>,
    pub label: Tensor<T>,
}

impl<T> MultimodalSample<T>
where
    T: Clone + Default,
{
    /// Create a new empty multimodal sample
    pub fn new(label: Tensor<T>) -> Self {
        Self {
            text: None,
            image: None,
            audio: None,
            video: None,
            embeddings: None,
            custom: HashMap::new(),
            metadata: HashMap::new(),
            label,
        }
    }

    /// Add text modality
    pub fn with_text(mut self, text: Tensor<T>) -> Self {
        self.text = Some(text);
        self
    }

    /// Add image modality
    pub fn with_image(mut self, image: Tensor<T>) -> Self {
        self.image = Some(image);
        self
    }

    /// Add audio modality
    pub fn with_audio(mut self, audio: Tensor<T>) -> Self {
        self.audio = Some(audio);
        self
    }

    /// Add video modality
    pub fn with_video(mut self, video: Tensor<T>) -> Self {
        self.video = Some(video);
        self
    }

    /// Add embeddings
    pub fn with_embeddings(mut self, embeddings: Tensor<T>) -> Self {
        self.embeddings = Some(embeddings);
        self
    }

    /// Add custom modality
    pub fn with_custom(mut self, key: String, data: Tensor<T>) -> Self {
        self.custom.insert(key, data);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get available modalities
    pub fn available_modalities(&self) -> Vec<Modality> {
        let mut modalities = Vec::new();
        if self.text.is_some() {
            modalities.push(Modality::Text);
        }
        if self.image.is_some() {
            modalities.push(Modality::Image);
        }
        if self.audio.is_some() {
            modalities.push(Modality::Audio);
        }
        if self.video.is_some() {
            modalities.push(Modality::Video);
        }
        if self.embeddings.is_some() {
            modalities.push(Modality::Embeddings);
        }
        for key in self.custom.keys() {
            modalities.push(Modality::Custom(key.clone()));
        }
        modalities
    }

    /// Check if sample has a specific modality
    pub fn has_modality(&self, modality: &Modality) -> bool {
        match modality {
            Modality::Text => self.text.is_some(),
            Modality::Image => self.image.is_some(),
            Modality::Audio => self.audio.is_some(),
            Modality::Video => self.video.is_some(),
            Modality::Embeddings => self.embeddings.is_some(),
            Modality::Custom(key) => self.custom.contains_key(key),
        }
    }

    /// Get data for a specific modality
    pub fn get_modality(&self, modality: &Modality) -> Option<&Tensor<T>> {
        match modality {
            Modality::Text => self.text.as_ref(),
            Modality::Image => self.image.as_ref(),
            Modality::Audio => self.audio.as_ref(),
            Modality::Video => self.video.as_ref(),
            Modality::Embeddings => self.embeddings.as_ref(),
            Modality::Custom(key) => self.custom.get(key),
        }
    }

    /// Get mutable data for a specific modality
    pub fn get_modality_mut(&mut self, modality: &Modality) -> Option<&mut Tensor<T>> {
        match modality {
            Modality::Text => self.text.as_mut(),
            Modality::Image => self.image.as_mut(),
            Modality::Audio => self.audio.as_mut(),
            Modality::Video => self.video.as_mut(),
            Modality::Embeddings => self.embeddings.as_mut(),
            Modality::Custom(key) => self.custom.get_mut(key),
        }
    }

    /// Remove a modality from the sample
    pub fn remove_modality(&mut self, modality: &Modality) -> Option<Tensor<T>> {
        match modality {
            Modality::Text => self.text.take(),
            Modality::Image => self.image.take(),
            Modality::Audio => self.audio.take(),
            Modality::Video => self.video.take(),
            Modality::Embeddings => self.embeddings.take(),
            Modality::Custom(key) => self.custom.remove(key),
        }
    }

    /// Get the number of available modalities
    pub fn modality_count(&self) -> usize {
        self.available_modalities().len()
    }

    /// Check if the sample is empty (no modalities)
    pub fn is_empty(&self) -> bool {
        self.modality_count() == 0
    }
}
