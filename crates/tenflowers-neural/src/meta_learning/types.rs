//! Shared types for meta-learning: Episode, MetaLearningError.

use std::collections::HashMap;
use thiserror::Error;

/// Errors that may arise in meta-learning computations.
#[derive(Debug, Error)]
pub enum MetaLearningError {
    /// Not enough labelled examples to fill the requested episode.
    #[error("insufficient data: needed {needed} examples but found {found}")]
    InsufficientData { needed: usize, found: usize },

    /// Requested N-way exceeds the number of available classes.
    #[error("invalid n_way={n_way}: only {available_classes} classes available")]
    InvalidNWay {
        n_way: usize,
        available_classes: usize,
    },

    /// Two tensors/vectors have incompatible lengths.
    #[error("dimension mismatch: expected {expected} but found {found}")]
    DimensionMismatch { expected: usize, found: usize },

    /// An operation was called on an empty embedding slice.
    #[error("empty embeddings: cannot proceed with zero-length input")]
    EmptyEmbeddings,
}

/// An N-way K-shot episode sampled from a dataset.
#[derive(Debug, Clone)]
pub struct Episode {
    /// Support-set features: `[n_way × k_shot, feat_dim]`.
    pub support_features: Vec<Vec<f32>>,
    /// Original class label for each support example (from the dataset).
    pub support_labels: Vec<usize>,
    /// Relabelled support labels in `0..n_way`.
    pub support_relabeled: Vec<usize>,
    /// Query-set features: `[n_way × n_query, feat_dim]`.
    pub query_features: Vec<Vec<f32>>,
    /// Relabelled query labels in `0..n_way`.
    pub query_labels: Vec<usize>,
    /// Number of classes per episode.
    pub n_way: usize,
    /// Number of support examples per class.
    pub k_shot: usize,
}

/// Numerically stable softmax (shared utility).
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max_v).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum < 1e-30 {
        let n = exps.len() as f32;
        return exps.iter().map(|_| 1.0 / n).collect();
    }
    exps.iter().map(|e| e / sum).collect()
}
