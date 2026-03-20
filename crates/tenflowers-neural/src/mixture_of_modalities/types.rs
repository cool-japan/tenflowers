//! Core types: math helpers, modality types, tokens, sequences, and modality data.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// §0 Internal math helpers
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn mat_vec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

pub(super) fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

pub(super) fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh()))
}

pub(super) fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_v = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|x| (x - max_v).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum < f64::EPSILON {
        vec![1.0 / logits.len() as f64; logits.len()]
    } else {
        exps.iter().map(|x| x / sum).collect()
    }
}

pub(super) fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < f64::EPSILON || norm_b < f64::EPSILON {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

pub(super) fn xavier_init(rng: &mut StdRng, rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let scale = (2.0 / (rows + cols) as f64).sqrt();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                .collect()
        })
        .collect()
}

pub(super) fn ones_vec(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

pub(super) fn zeros_vec(n: usize) -> Vec<f64> {
    vec![0.0; n]
}

pub(super) fn pool_mean(vecs: &[Vec<f64>]) -> Vec<f64> {
    if vecs.is_empty() {
        return Vec::new();
    }
    let dim = vecs[0].len();
    let mut sum = vec![0.0f64; dim];
    for v in vecs {
        for (s, x) in sum.iter_mut().zip(v.iter()) {
            *s += x;
        }
    }
    let n = vecs.len() as f64;
    sum.iter().map(|x| x / n).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// §1 Modality Types & Tokens
// ─────────────────────────────────────────────────────────────────────────────

/// All modality types supported by the unified framework.
///
/// Prefixed `Mom` to avoid collision with [`crate::ModalityType`] from
/// the `cross_modal_retrieval` module (which has Vision/Text/Audio/Video/Tabular).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MomModalityType {
    Text,
    Image,
    Audio,
    Video,
    Tabular,
    Code,
    Math,
}

impl MomModalityType {
    /// Returns the number of distinct modality types.
    pub fn count() -> usize {
        7
    }
}

/// A single token with modality metadata.
#[derive(Debug, Clone)]
pub struct ModalToken {
    /// Which modality this token belongs to.
    pub modality: MomModalityType,
    /// Vocabulary token ID.
    pub token_id: usize,
    /// Position within the overall sequence.
    pub position: usize,
}

/// A sequence of tokens potentially spanning multiple modalities.
#[derive(Debug, Clone)]
pub struct ModalSequence {
    /// All tokens in the sequence.
    pub tokens: Vec<ModalToken>,
    /// Non-overlapping spans: (modality_type, start_inclusive, end_exclusive).
    pub modality_spans: Vec<(MomModalityType, usize, usize)>,
}

impl Default for ModalSequence {
    fn default() -> Self {
        Self::new()
    }
}

impl ModalSequence {
    /// Create an empty sequence.
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            modality_spans: Vec::new(),
        }
    }

    /// Append a batch of tokens (all assumed to belong to the same modality).
    pub fn append(&mut self, tokens: Vec<ModalToken>) {
        if tokens.is_empty() {
            return;
        }
        let start = self.tokens.len();
        let modality = tokens[0].modality.clone();
        self.tokens.extend(tokens);
        let end = self.tokens.len();
        self.modality_spans.push((modality, start, end));
    }

    /// Return the (start, end) span of the first occurrence of a modality.
    pub fn get_span(&self, modality: &MomModalityType) -> Option<(usize, usize)> {
        self.modality_spans
            .iter()
            .find(|(m, _, _)| m == modality)
            .map(|(_, s, e)| (*s, *e))
    }

    /// Total number of tokens in the sequence.
    pub fn total_tokens(&self) -> usize {
        self.tokens.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2 Modality Data & Tokenizer Trait
// ─────────────────────────────────────────────────────────────────────────────

/// Raw data for any modality.
#[derive(Debug, Clone)]
pub enum ModalityData {
    /// Plain text string.
    Text(String),
    /// Image patches: `[n_patches][patch_dim]`.
    ImagePatches(Vec<Vec<f64>>),
    /// Audio frames: `[n_frames][frame_dim]`.
    AudioFrames(Vec<Vec<f64>>),
    /// Video frames: `[n_frames][n_patches][patch_dim]`.
    VideoFrames(Vec<Vec<Vec<f64>>>),
    /// Tabular data: `[n_rows][n_cols]`.
    Tabular(Vec<Vec<f64>>),
    /// Source code string.
    Code(String),
    /// Mathematical expression string.
    Math(String),
}

impl ModalityData {
    /// Returns the corresponding [`MomModalityType`].
    pub fn modality_type(&self) -> MomModalityType {
        match self {
            ModalityData::Text(_) => MomModalityType::Text,
            ModalityData::ImagePatches(_) => MomModalityType::Image,
            ModalityData::AudioFrames(_) => MomModalityType::Audio,
            ModalityData::VideoFrames(_) => MomModalityType::Video,
            ModalityData::Tabular(_) => MomModalityType::Tabular,
            ModalityData::Code(_) => MomModalityType::Code,
            ModalityData::Math(_) => MomModalityType::Math,
        }
    }

    /// Human-readable description of the data.
    pub fn description(&self) -> String {
        match self {
            ModalityData::Text(s) => format!("Text({} chars)", s.len()),
            ModalityData::ImagePatches(p) => format!("Image({} patches)", p.len()),
            ModalityData::AudioFrames(f) => format!("Audio({} frames)", f.len()),
            ModalityData::VideoFrames(f) => format!("Video({} frames)", f.len()),
            ModalityData::Tabular(r) => format!("Tabular({} rows)", r.len()),
            ModalityData::Code(s) => format!("Code({} chars)", s.len()),
            ModalityData::Math(s) => format!("Math({} chars)", s.len()),
        }
    }
}

/// Trait for modality-specific tokenizers.
pub trait ModalityTokenizer: Send + Sync {
    /// The modality this tokenizer handles.
    fn modality(&self) -> MomModalityType;
    /// Encode raw data into token IDs.
    fn encode(&self, data: &ModalityData) -> Vec<usize>;
    /// Decode token IDs back into raw data.
    fn decode(&self, token_ids: &[usize]) -> ModalityData;
    /// Vocabulary size.
    fn vocab_size(&self) -> usize;
}
