//! Core types for multimodal datasets

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Represents different modalities in a multimodal dataset
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
    Embeddings,
    Custom(String),
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text => write!(f, "text"),
            Self::Image => write!(f, "image"),
            Self::Audio => write!(f, "audio"),
            Self::Video => write!(f, "video"),
            Self::Embeddings => write!(f, "embeddings"),
            Self::Custom(name) => write!(f, "custom({})", name),
        }
    }
}

/// Strategies for fusing multiple modalities into a single representation
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum FusionStrategy {
    /// Concatenate features from all modalities
    Concatenation,
    /// Use attention mechanism to fuse modalities
    Attention,
    /// Use early fusion (combine at input level)
    EarlyFusion,
    /// Use late fusion (combine at output level)
    LateFusion,
    /// Keep modalities separate (no fusion)
    Separate,
}

impl Default for FusionStrategy {
    fn default() -> Self {
        Self::Concatenation
    }
}

impl std::fmt::Display for FusionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Concatenation => write!(f, "concatenation"),
            Self::Attention => write!(f, "attention"),
            Self::EarlyFusion => write!(f, "early_fusion"),
            Self::LateFusion => write!(f, "late_fusion"),
            Self::Separate => write!(f, "separate"),
        }
    }
}
