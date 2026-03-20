//! Self-supervised learning pretext tasks and objectives — Track AA.
//!
//! This module provides the foundational building blocks for self-supervised
//! pre-training of language and vision models:
//!
//! - **MLM** (Masked Language Modelling, BERT-style): token masking strategy,
//!   per-position cross-entropy loss, and token prediction accuracy.
//! - **Causal LM** (GPT-style): next-token prediction loss over a full sequence.
//! - **NSP** (Next Sentence Prediction, BERT-style): binary sentence-pair
//!   classification loss.
//! - **Rotation prediction**: lightweight image-rotation pretext task applied
//!   to embedding vectors via circular shift.
//! - **Patch agreement**: simplified contrastive objective maximising cosine
//!   similarity between two augmented views.
//!
//! # Design decisions
//!
//! * All functions operate on flat `&[f32]` / `&[u32]` buffers rather than
//!   `Tensor<T>` to keep the module self-contained and to avoid tensor-graph
//!   overhead in pre-training inner loops.
//! * Randomness is sourced from `scirs2_core::random` — never from the `rand`
//!   crate directly.
//! * No `unwrap()` is used anywhere; every fallible path returns a descriptive
//!   `Result<_, TensorError>`.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically stable log-sum-exp over a slice.
#[inline]
fn log_sum_exp(values: &[f32]) -> f32 {
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max.is_infinite() {
        return max;
    }
    let sum: f32 = values.iter().map(|&v| (v - max).exp()).sum();
    max + sum.ln()
}

/// Log-softmax for a single vocabulary row `logits[offset .. offset+vocab_size]`.
///
/// Returns the log-probability for `target_id`.
#[inline]
fn log_prob_at(logits: &[f32], offset: usize, vocab_size: usize, target_id: usize) -> Result<f32> {
    if offset + vocab_size > logits.len() {
        return Err(TensorError::InvalidArgument {
            operation: "log_prob_at".to_string(),
            reason: format!(
                "logits slice too short: need offset({offset})+vocab_size({vocab_size})={}, have {}",
                offset + vocab_size,
                logits.len()
            ),
            context: None,
        });
    }
    if target_id >= vocab_size {
        return Err(TensorError::InvalidArgument {
            operation: "log_prob_at".to_string(),
            reason: format!("target_id {target_id} >= vocab_size {vocab_size}"),
            context: None,
        });
    }
    let row = &logits[offset..offset + vocab_size];
    let lse = log_sum_exp(row);
    Ok(row[target_id] - lse)
}

/// Dot product of two slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm of a slice.
#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// MlmMasker
// ─────────────────────────────────────────────────────────────────────────────

/// Output produced by [`MlmMasker::mask_sequence`].
///
/// The triple `(masked_ids, original_ids, mask_positions)` contains:
/// * `masked_ids` — the corrupted input sequence fed to the model (same length
///   as the original);
/// * `original_ids` — a full copy of the original sequence (labels);
/// * `mask_positions` — sorted indices of the positions that were selected for
///   masking (the subset on which the MLM loss is computed).
#[derive(Debug, Clone)]
pub struct MlmSample {
    /// Input to the model: selected positions replaced with a mask token, a
    /// random token, or kept as-is according to the BERT masking schedule.
    pub masked_ids: Vec<u32>,
    /// Original (unmodified) token IDs — ground-truth labels.
    pub original_ids: Vec<u32>,
    /// Sorted indices of positions selected for masking.
    pub mask_positions: Vec<usize>,
}

/// BERT-style masked language modelling masking strategy.
///
/// For each token position the masker independently samples with probability
/// `mask_prob` whether to select it for masking.  Of the selected positions:
///
/// * `(1 − random_prob − keep_prob)` fraction → replaced with `mask_token_id`
/// * `random_prob` fraction → replaced with a uniformly random vocabulary token
/// * `keep_prob` fraction → kept as the original token (but still counted as a
///   masked position for loss computation)
///
/// Default parameters match the original BERT paper:
/// `mask_prob = 0.15`, `random_prob = 0.10`, `keep_prob = 0.10`.
#[derive(Debug, Clone)]
pub struct MlmMasker {
    /// Probability that any given position is selected for masking.
    pub mask_prob: f32,
    /// Token ID substituted for `[MASK]`-replaced positions.
    pub mask_token_id: u32,
    /// Vocabulary size (used for uniform random token sampling).
    pub vocab_size: u32,
    /// Fraction of selected positions replaced with a random token.
    pub random_prob: f32,
    /// Fraction of selected positions kept as the original token.
    pub keep_prob: f32,
}

impl MlmMasker {
    /// Create a new `MlmMasker` with BERT default probabilities.
    ///
    /// * `mask_prob = 0.15`
    /// * `random_prob = 0.10`
    /// * `keep_prob = 0.10`
    pub fn new(mask_token_id: u32, vocab_size: u32) -> Self {
        Self {
            mask_prob: 0.15,
            mask_token_id,
            vocab_size,
            random_prob: 0.10,
            keep_prob: 0.10,
        }
    }

    /// Override the masking probability (builder pattern).
    pub fn with_mask_prob(mut self, p: f32) -> Self {
        self.mask_prob = p;
        self
    }

    /// Apply masking to a sequence of token IDs.
    ///
    /// Returns an [`MlmSample`] with:
    /// * `masked_ids` — input to the model (some tokens replaced),
    /// * `original_ids` — ground-truth labels (full copy),
    /// * `mask_positions` — which positions were selected for masking.
    ///
    /// # Errors
    ///
    /// * If `vocab_size == 0`.
    /// * If `ids` is empty.
    pub fn mask_sequence(&self, ids: &[u32], seed: u64) -> Result<MlmSample> {
        if self.vocab_size == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "MlmMasker::mask_sequence".to_string(),
                reason: "vocab_size must be > 0".to_string(),
                context: None,
            });
        }
        if ids.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "MlmMasker::mask_sequence".to_string(),
                reason: "input sequence must be non-empty".to_string(),
                context: None,
            });
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let mut masked_ids = ids.to_vec();
        let original_ids = ids.to_vec();
        let mut mask_positions: Vec<usize> = Vec::new();

        // Threshold for random-token replacement vs keep-as-original vs [MASK].
        // Within a selected position:
        //   [0, random_prob)              → random token
        //   [random_prob, random_prob+keep_prob) → keep original
        //   [random_prob+keep_prob, 1)    → [MASK] token
        let random_threshold = self.random_prob;
        let keep_threshold = self.random_prob + self.keep_prob;

        for (i, &original_tok) in ids.iter().enumerate() {
            let select: f32 = rng.random();
            if select >= self.mask_prob {
                continue;
            }
            mask_positions.push(i);

            let action: f32 = rng.random();
            if action < random_threshold {
                // Replace with a uniformly random token from the vocabulary.
                let rand_tok_idx: u32 = (rng.random::<f64>() * self.vocab_size as f64) as u32;
                // Clamp to valid range in case of floating-point edge-cases.
                masked_ids[i] = rand_tok_idx.min(self.vocab_size - 1);
            } else if action < keep_threshold {
                // Keep original token (no change needed).
                masked_ids[i] = original_tok;
            } else {
                // Replace with the [MASK] token.
                masked_ids[i] = self.mask_token_id;
            }
        }

        Ok(MlmSample {
            masked_ids,
            original_ids,
            mask_positions,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MLM loss
// ─────────────────────────────────────────────────────────────────────────────

/// Masked Language Model cross-entropy loss.
///
/// Computes the mean negative log-likelihood over the selected `mask_positions`.
/// Only positions listed in `mask_positions` contribute to the loss — positions
/// that were not masked are ignored.
///
/// # Arguments
///
/// * `logits` — flat buffer of shape `[seq_len, vocab_size]` (row-major).
/// * `targets` — original token IDs at each masked position (`targets[k]` is
///   the ground-truth token for `mask_positions[k]`).
/// * `mask_positions` — which sequence positions were masked.
/// * `vocab_size` — vocabulary size (stride of `logits`).
///
/// # Errors
///
/// * Length mismatches between `targets` and `mask_positions`.
/// * Any `mask_positions` index that is out of bounds for `logits`.
/// * Any `targets[k]` value ≥ `vocab_size`.
/// * Empty `mask_positions`.
pub fn mlm_loss(
    logits: &[f32],
    targets: &[u32],
    mask_positions: &[usize],
    vocab_size: usize,
) -> Result<f32> {
    if vocab_size == 0 {
        return Err(TensorError::InvalidArgument {
            operation: "mlm_loss".to_string(),
            reason: "vocab_size must be > 0".to_string(),
            context: None,
        });
    }
    if targets.len() != mask_positions.len() {
        return Err(TensorError::InvalidArgument {
            operation: "mlm_loss".to_string(),
            reason: format!(
                "targets.len()={} != mask_positions.len()={}",
                targets.len(),
                mask_positions.len()
            ),
            context: None,
        });
    }
    if mask_positions.is_empty() {
        return Err(TensorError::InvalidArgument {
            operation: "mlm_loss".to_string(),
            reason: "mask_positions must be non-empty".to_string(),
            context: None,
        });
    }

    let mut total_loss = 0.0_f32;
    for (k, &pos) in mask_positions.iter().enumerate() {
        let offset = pos * vocab_size;
        let target_id = targets[k] as usize;
        let lp = log_prob_at(logits, offset, vocab_size, target_id)?;
        total_loss += -lp;
    }

    Ok(total_loss / mask_positions.len() as f32)
}

// ─────────────────────────────────────────────────────────────────────────────
// Causal LM loss
// ─────────────────────────────────────────────────────────────────────────────

/// Causal Language Model (GPT-style) next-token prediction loss.
///
/// Given `logits` of shape `[seq_len, vocab_size]` and `targets` of length
/// `seq_len`, computes the mean cross-entropy loss over positions
/// `0 .. seq_len − 1` where `targets[i]` is the token that follows position
/// `i` (i.e. `targets[i] = input[i+1]`).
///
/// Position `seq_len − 1` is excluded because it has no following token.
///
/// # Errors
///
/// * `logits` length is not a multiple of `vocab_size`.
/// * `targets` length does not match the implied `seq_len`.
/// * Any `targets[i]` ≥ `vocab_size`.
pub fn causal_lm_loss(logits: &[f32], targets: &[u32], vocab_size: usize) -> Result<f32> {
    if vocab_size == 0 {
        return Err(TensorError::InvalidArgument {
            operation: "causal_lm_loss".to_string(),
            reason: "vocab_size must be > 0".to_string(),
            context: None,
        });
    }
    if logits.len() % vocab_size != 0 {
        return Err(TensorError::InvalidArgument {
            operation: "causal_lm_loss".to_string(),
            reason: format!(
                "logits.len()={} is not divisible by vocab_size={}",
                logits.len(),
                vocab_size
            ),
            context: None,
        });
    }
    let seq_len = logits.len() / vocab_size;
    if targets.len() != seq_len {
        return Err(TensorError::InvalidArgument {
            operation: "causal_lm_loss".to_string(),
            reason: format!("targets.len()={} != seq_len={}", targets.len(), seq_len),
            context: None,
        });
    }
    if seq_len < 2 {
        return Err(TensorError::InvalidArgument {
            operation: "causal_lm_loss".to_string(),
            reason: "sequence length must be at least 2 for causal LM loss".to_string(),
            context: None,
        });
    }

    // Predict token at position i+1 using logits at position i.
    let predict_len = seq_len - 1;
    let mut total_loss = 0.0_f32;
    for i in 0..predict_len {
        let offset = i * vocab_size;
        // targets[i] is the next token (input[i+1]).
        let target_id = targets[i] as usize;
        let lp = log_prob_at(logits, offset, vocab_size, target_id)?;
        total_loss += -lp;
    }

    Ok(total_loss / predict_len as f32)
}

// ─────────────────────────────────────────────────────────────────────────────
// Next Sentence Prediction
// ─────────────────────────────────────────────────────────────────────────────

/// A sentence-pair sample for Next Sentence Prediction.
#[derive(Debug, Clone)]
pub struct NspSample {
    /// Token IDs for sentence A.
    pub sentence_a: Vec<u32>,
    /// Token IDs for sentence B.
    pub sentence_b: Vec<u32>,
    /// `true` if sentence B immediately follows sentence A in the source text.
    pub is_next: bool,
}

/// Next Sentence Prediction binary cross-entropy loss.
///
/// The model outputs two logits: `logit_true` (score for "is next") and
/// `logit_false` (score for "is not next").  We apply a two-class softmax and
/// return the negative log-likelihood of the correct label.
///
/// Mathematically this is equivalent to sigmoid binary cross-entropy when the
/// two logits are `[score, 0]`, but using two explicit logits (as BERT does)
/// is cleaner and avoids tied parameterisation.
pub fn nsp_loss(logit_true: f32, logit_false: f32, is_next: bool) -> f32 {
    // Two-class softmax then NLL.
    let lse = log_sum_exp(&[logit_true, logit_false]);
    let log_p_true = logit_true - lse;
    let log_p_false = logit_false - lse;

    if is_next {
        -log_p_true
    } else {
        -log_p_false
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotation prediction
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-entropy loss for a 4-class rotation prediction pretext task.
///
/// The model outputs `logits` of length 4 corresponding to 0°, 90°, 180°, and
/// 270° rotation classes.  The loss is the negative log-probability assigned to
/// `rotation_class`.
///
/// # Errors
///
/// * `logits` does not have exactly 4 elements.
/// * `rotation_class` ≥ 4.
pub fn rotation_prediction_loss(logits: &[f32], rotation_class: usize) -> Result<f32> {
    const NUM_ROTATIONS: usize = 4;
    if logits.len() != NUM_ROTATIONS {
        return Err(TensorError::InvalidArgument {
            operation: "rotation_prediction_loss".to_string(),
            reason: format!(
                "logits must have exactly {} elements (got {})",
                NUM_ROTATIONS,
                logits.len()
            ),
            context: None,
        });
    }
    if rotation_class >= NUM_ROTATIONS {
        return Err(TensorError::InvalidArgument {
            operation: "rotation_prediction_loss".to_string(),
            reason: format!("rotation_class must be in 0..4, got {rotation_class}"),
            context: None,
        });
    }

    let lse = log_sum_exp(logits);
    let log_p = logits[rotation_class] - lse;
    Ok(-log_p)
}

/// Apply a synthetic "rotation" to an embedding via circular shift.
///
/// Rotation `k` shifts the embedding by `k * (dim / 4)` positions to the left
/// (i.e. a cyclic permutation).  Rotation 0 is the identity.
///
/// This is a simple embedding-space analogue of 2-D image rotation that
/// preserves the L2 norm and is invertible.
///
/// # Arguments
///
/// * `embedding` — the embedding vector to rotate.
/// * `rotation` — rotation index in `{0, 1, 2, 3}`.
///
/// # Errors
///
/// * `rotation` ≥ 4.
/// * `embedding` is empty.
pub fn rotate_embedding(embedding: &[f32], rotation: usize) -> Result<Vec<f32>> {
    const NUM_ROTATIONS: usize = 4;
    if rotation >= NUM_ROTATIONS {
        return Err(TensorError::InvalidArgument {
            operation: "rotate_embedding".to_string(),
            reason: format!("rotation must be in 0..4, got {rotation}"),
            context: None,
        });
    }
    if embedding.is_empty() {
        return Err(TensorError::InvalidArgument {
            operation: "rotate_embedding".to_string(),
            reason: "embedding must be non-empty".to_string(),
            context: None,
        });
    }

    let dim = embedding.len();
    // Shift amount: k * (dim / 4), using integer division.
    // For dim < 4 the shift collapses to 0, giving the identity for all k.
    let shift = (rotation * (dim / 4)) % dim;

    if shift == 0 {
        return Ok(embedding.to_vec());
    }

    // Circular left-shift by `shift`.
    let mut out = Vec::with_capacity(dim);
    out.extend_from_slice(&embedding[shift..]);
    out.extend_from_slice(&embedding[..shift]);
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Patch agreement (simplified contrastive objective)
// ─────────────────────────────────────────────────────────────────────────────

/// Patch agreement loss between two augmented views.
///
/// Maximises cosine similarity between `z1` and `z2` by returning its negative
/// (so minimising the loss → maximising agreement).
///
/// Formally: `loss = −cosine_sim(z1, z2) / temperature`
///
/// where `temperature` provides a scaling factor (typically 0.07–0.5).
/// A `temperature` close to 0 gives a sharper objective.
///
/// # Errors
///
/// * `z1` and `z2` have different lengths.
/// * Either embedding is empty.
/// * `temperature ≤ 0`.
pub fn patch_agreement_loss(z1: &[f32], z2: &[f32], temperature: f32) -> Result<f32> {
    if z1.len() != z2.len() {
        return Err(TensorError::InvalidArgument {
            operation: "patch_agreement_loss".to_string(),
            reason: format!("z1.len()={} != z2.len()={}", z1.len(), z2.len()),
            context: None,
        });
    }
    if z1.is_empty() {
        return Err(TensorError::InvalidArgument {
            operation: "patch_agreement_loss".to_string(),
            reason: "embeddings must be non-empty".to_string(),
            context: None,
        });
    }
    if temperature <= 0.0 {
        return Err(TensorError::InvalidArgument {
            operation: "patch_agreement_loss".to_string(),
            reason: format!("temperature must be > 0, got {temperature}"),
            context: None,
        });
    }

    let norm1 = l2_norm(z1);
    let norm2 = l2_norm(z2);
    let cos_sim = if norm1 == 0.0 || norm2 == 0.0 {
        0.0_f32
    } else {
        dot(z1, z2) / (norm1 * norm2)
    };

    // Negative cosine similarity: minimising this maximises agreement.
    Ok(-cos_sim / temperature)
}

// ─────────────────────────────────────────────────────────────────────────────
// MLM accuracy
// ─────────────────────────────────────────────────────────────────────────────

/// Token prediction accuracy at masked positions.
///
/// For each masked position, checks whether `argmax(logits[pos * vocab_size ..
/// (pos+1) * vocab_size])` equals the corresponding target token ID.
///
/// Returns the fraction of correct predictions in `[0, 1]`.
///
/// # Errors
///
/// Same preconditions as [`mlm_loss`].
pub fn mlm_accuracy(
    logits: &[f32],
    targets: &[u32],
    mask_positions: &[usize],
    vocab_size: usize,
) -> Result<f32> {
    if vocab_size == 0 {
        return Err(TensorError::InvalidArgument {
            operation: "mlm_accuracy".to_string(),
            reason: "vocab_size must be > 0".to_string(),
            context: None,
        });
    }
    if targets.len() != mask_positions.len() {
        return Err(TensorError::InvalidArgument {
            operation: "mlm_accuracy".to_string(),
            reason: format!(
                "targets.len()={} != mask_positions.len()={}",
                targets.len(),
                mask_positions.len()
            ),
            context: None,
        });
    }
    if mask_positions.is_empty() {
        return Err(TensorError::InvalidArgument {
            operation: "mlm_accuracy".to_string(),
            reason: "mask_positions must be non-empty".to_string(),
            context: None,
        });
    }

    let mut correct = 0usize;
    for (k, &pos) in mask_positions.iter().enumerate() {
        let offset = pos * vocab_size;
        if offset + vocab_size > logits.len() {
            return Err(TensorError::InvalidArgument {
                operation: "mlm_accuracy".to_string(),
                reason: format!(
                    "logits slice too short: need offset({offset})+vocab_size({vocab_size})={}, have {}",
                    offset + vocab_size,
                    logits.len()
                ),
                context: None,
            });
        }
        let row = &logits[offset..offset + vocab_size];
        // argmax
        let pred = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        if pred == targets[k] as usize {
            correct += 1;
        }
    }

    Ok(correct as f32 / mask_positions.len() as f32)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers for building test fixtures ───────────────────────────────────

    /// Build a logits buffer where position `pos` gets a large score for token
    /// `correct_id` and every other position gets a large score for token 0.
    fn perfect_logits_at(
        seq_len: usize,
        vocab_size: usize,
        perfect_positions: &[(usize, usize)], // (pos, correct_id)
    ) -> Vec<f32> {
        let mut logits = vec![0.0_f32; seq_len * vocab_size];
        // Give token 0 a moderate score everywhere as a baseline.
        for pos in 0..seq_len {
            logits[pos * vocab_size] = 1.0;
        }
        // Override the specified positions.
        for &(pos, correct_id) in perfect_positions {
            // Zero out the baseline for token 0 at this position.
            logits[pos * vocab_size] = 0.0;
            logits[pos * vocab_size + correct_id] = 100.0;
        }
        logits
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MlmMasker::mask_sequence tests
    // ─────────────────────────────────────────────────────────────────────────

    /// The number of masked positions should be close to `mask_prob * seq_len`.
    /// We use a long sequence to make the fraction converge via the LLN.
    #[test]
    fn test_mlm_masker_approximately_correct_fraction() {
        let masker = MlmMasker::new(103, 30_000);
        let seq: Vec<u32> = (0u32..1000).collect();
        let sample = masker
            .mask_sequence(&seq, 42)
            .expect("mask_sequence failed");

        let fraction = sample.mask_positions.len() as f32 / seq.len() as f32;
        // Allow ±5 percentage points around the nominal 15 %.
        assert!(
            (fraction - 0.15).abs() < 0.05,
            "masking fraction {fraction:.3} not close to 0.15"
        );
    }

    /// The masked_ids at mask_positions must differ from (or equal to, for the
    /// "keep" action) the originals; at *non*-masked positions they must be
    /// identical to the originals.
    #[test]
    fn test_mlm_masker_non_masked_positions_unchanged() {
        let masker = MlmMasker::new(103, 30_000);
        let seq: Vec<u32> = (0u32..200).collect();
        let sample = masker.mask_sequence(&seq, 7).expect("mask_sequence failed");

        let masked_set: std::collections::HashSet<usize> =
            sample.mask_positions.iter().cloned().collect();

        for (i, (&orig, &masked)) in sample
            .original_ids
            .iter()
            .zip(sample.masked_ids.iter())
            .enumerate()
        {
            if !masked_set.contains(&i) {
                assert_eq!(
                    orig, masked,
                    "non-masked position {i} was altered: {orig} → {masked}"
                );
            }
        }
    }

    /// mask_positions must be non-empty for a reasonable sequence.
    #[test]
    fn test_mlm_masker_positions_non_empty_for_long_seq() {
        let masker = MlmMasker::new(103, 30_000);
        let seq: Vec<u32> = (0u32..200).collect();
        let sample = masker
            .mask_sequence(&seq, 123)
            .expect("mask_sequence failed");
        assert!(
            !sample.mask_positions.is_empty(),
            "no positions masked for a 200-token sequence"
        );
    }

    /// mask_positions count should be approximately mask_prob * seq_len.
    #[test]
    fn test_mlm_masker_positions_count_approximately_mask_prob() {
        let masker = MlmMasker::new(103, 30_000);
        let seq: Vec<u32> = (0u32..500).collect();
        let sample = masker
            .mask_sequence(&seq, 99)
            .expect("mask_sequence failed");

        let expected = (0.15 * 500.0_f32) as usize;
        let actual = sample.mask_positions.len();
        // Allow generous ±30 % relative tolerance.
        let tolerance = (expected as f32 * 0.35) as usize + 5;
        assert!(
            (actual as i64 - expected as i64).unsigned_abs() as usize <= tolerance,
            "mask count {actual} is far from expected {expected}"
        );
    }

    /// with_mask_prob builder method should change the fraction.
    #[test]
    fn test_mlm_masker_with_mask_prob() {
        let masker = MlmMasker::new(103, 30_000).with_mask_prob(0.30);
        let seq: Vec<u32> = (0u32..500).collect();
        let sample = masker.mask_sequence(&seq, 1).expect("mask_sequence failed");
        let fraction = sample.mask_positions.len() as f32 / seq.len() as f32;
        assert!(
            (fraction - 0.30).abs() < 0.08,
            "masking fraction {fraction:.3} not close to 0.30"
        );
    }

    /// Determinism: same seed must produce same result.
    #[test]
    fn test_mlm_masker_deterministic() {
        let masker = MlmMasker::new(103, 30_000);
        let seq: Vec<u32> = (0u32..100).collect();
        let s1 = masker.mask_sequence(&seq, 55).expect("first call failed");
        let s2 = masker.mask_sequence(&seq, 55).expect("second call failed");
        assert_eq!(
            s1.mask_positions, s2.mask_positions,
            "non-deterministic positions"
        );
        assert_eq!(s1.masked_ids, s2.masked_ids, "non-deterministic masked_ids");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // mlm_loss tests
    // ─────────────────────────────────────────────────────────────────────────

    /// mlm_loss must be non-negative.
    #[test]
    fn test_mlm_loss_non_negative() {
        let vocab_size = 10;
        let seq_len = 5;
        let logits: Vec<f32> = (0..(seq_len * vocab_size))
            .map(|i| (i as f32).sin())
            .collect();
        let targets = vec![3u32, 7];
        let positions = vec![1, 4];
        let loss = mlm_loss(&logits, &targets, &positions, vocab_size).expect("mlm_loss failed");
        assert!(loss >= 0.0, "mlm_loss was negative: {loss}");
    }

    /// With perfect logits the loss should be near zero.
    #[test]
    fn test_mlm_loss_perfect_logits_near_zero() {
        let vocab_size = 10;
        let seq_len = 5;
        let targets = vec![3u32, 7];
        let positions = vec![1, 4];
        let logits = perfect_logits_at(seq_len, vocab_size, &[(1, 3), (4, 7)]);
        let loss = mlm_loss(&logits, &targets, &positions, vocab_size).expect("mlm_loss failed");
        assert!(
            loss < 1e-3,
            "loss with perfect logits was {loss}, expected ~0"
        );
    }

    /// mlm_loss length mismatch should return an error.
    #[test]
    fn test_mlm_loss_length_mismatch_error() {
        let vocab_size = 10;
        let logits = vec![0.0f32; 50];
        let result = mlm_loss(&logits, &[1u32], &[0, 1], vocab_size);
        assert!(
            result.is_err(),
            "expected error for targets/positions mismatch"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // causal_lm_loss tests
    // ─────────────────────────────────────────────────────────────────────────

    /// causal_lm_loss must be non-negative.
    #[test]
    fn test_causal_lm_loss_non_negative() {
        let vocab_size = 8;
        let seq_len = 6;
        let logits: Vec<f32> = (0..(seq_len * vocab_size))
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let targets: Vec<u32> = (0u32..seq_len as u32)
            .map(|i| i % vocab_size as u32)
            .collect();
        let loss = causal_lm_loss(&logits, &targets, vocab_size).expect("causal_lm_loss failed");
        assert!(loss >= 0.0, "causal_lm_loss was negative: {loss}");
    }

    /// With perfect predictions the causal LM loss should be near zero.
    #[test]
    fn test_causal_lm_loss_perfect_predictions() {
        let vocab_size = 8;
        let seq_len = 6;
        // targets[i] = i+1 mod vocab_size (the next token).
        let targets: Vec<u32> = (1u32..=seq_len as u32)
            .map(|i| i % vocab_size as u32)
            .collect();
        // Perfect logits: at position i, score for targets[i] should be very high.
        let logits = perfect_logits_at(
            seq_len,
            vocab_size,
            &(0..seq_len - 1)
                .map(|i| (i, targets[i] as usize))
                .collect::<Vec<_>>(),
        );
        let loss = causal_lm_loss(&logits, &targets, vocab_size).expect("causal_lm_loss failed");
        assert!(loss < 1e-3, "causal_lm_loss with perfect logits was {loss}");
    }

    /// causal_lm_loss requires seq_len ≥ 2.
    #[test]
    fn test_causal_lm_loss_too_short_error() {
        let vocab_size = 5;
        let logits = vec![1.0_f32; vocab_size];
        let targets = vec![0u32];
        let result = causal_lm_loss(&logits, &targets, vocab_size);
        assert!(result.is_err(), "expected error for seq_len < 2");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // nsp_loss tests
    // ─────────────────────────────────────────────────────────────────────────

    /// nsp_loss should behave as binary cross-entropy.
    #[test]
    fn test_nsp_loss_is_binary_cross_entropy() {
        // When logit_true >> logit_false and is_next=true, loss ≈ 0.
        let loss_correct = nsp_loss(100.0, -100.0, true);
        assert!(
            loss_correct < 0.01,
            "nsp_loss for correct prediction was {loss_correct}"
        );

        // When logit_true << logit_false and is_next=true, loss should be large.
        let loss_wrong = nsp_loss(-100.0, 100.0, true);
        assert!(
            loss_wrong > 10.0,
            "nsp_loss for wrong prediction was {loss_wrong}"
        );
    }

    /// nsp_loss is symmetric: flipping is_next flips which logit we want.
    #[test]
    fn test_nsp_loss_is_next_false() {
        let loss = nsp_loss(-100.0, 100.0, false);
        assert!(
            loss < 0.01,
            "nsp_loss for is_next=false, logit_false high, was {loss}"
        );
    }

    /// nsp_loss should be non-negative.
    #[test]
    fn test_nsp_loss_non_negative() {
        for &is_next in &[true, false] {
            for &(lt, lf) in &[(1.0_f32, 0.0_f32), (0.0, 1.0), (0.5, 0.5), (-1.0, 2.0)] {
                let l = nsp_loss(lt, lf, is_next);
                assert!(l >= 0.0, "nsp_loss negative ({l}) for logit_true={lt}, logit_false={lf}, is_next={is_next}");
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // rotation_prediction_loss tests
    // ─────────────────────────────────────────────────────────────────────────

    /// Correct class should yield very low loss when its logit is dominant.
    #[test]
    fn test_rotation_prediction_loss_correct_class_low_loss() {
        let mut logits = [-100.0_f32; 4];
        logits[2] = 100.0;
        let loss = rotation_prediction_loss(&logits, 2).expect("rotation_prediction_loss failed");
        assert!(loss < 1e-3, "loss with dominant correct logit was {loss}");
    }

    /// rotation_prediction_loss should return an error for wrong logit count.
    #[test]
    fn test_rotation_prediction_loss_wrong_logit_count_error() {
        let result = rotation_prediction_loss(&[1.0, 2.0, 3.0], 0);
        assert!(result.is_err(), "expected error for 3 logits");
    }

    /// rotation_prediction_loss should return an error for rotation_class ≥ 4.
    #[test]
    fn test_rotation_prediction_loss_out_of_range_class_error() {
        let result = rotation_prediction_loss(&[1.0, 2.0, 3.0, 4.0], 4);
        assert!(result.is_err(), "expected error for rotation_class=4");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // rotate_embedding tests
    // ─────────────────────────────────────────────────────────────────────────

    /// Rotation 0 should be the identity.
    #[test]
    fn test_rotate_embedding_rotation_0_is_identity() {
        let emb = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let rotated = rotate_embedding(&emb, 0).expect("rotate_embedding failed");
        assert_eq!(rotated, emb, "rotation 0 is not identity");
    }

    /// Rotation 4 should return an error.
    #[test]
    fn test_rotate_embedding_rotation_4_error() {
        let emb = vec![1.0_f32; 8];
        let result = rotate_embedding(&emb, 4);
        assert!(result.is_err(), "expected error for rotation=4");
    }

    /// Rotation should be invertible: 4 rotations of 90° bring back the original.
    #[test]
    fn test_rotate_embedding_four_rotations_returns_identity() {
        let emb: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut current = emb.clone();
        for _ in 0..4 {
            current = rotate_embedding(&current, 1).expect("rotate failed");
        }
        assert_eq!(
            current, emb,
            "four 90° rotations did not return to original"
        );
    }

    /// Rotation 2 should be the same as two rotation-1 applications.
    #[test]
    fn test_rotate_embedding_rotation_2_equals_double_rotation_1() {
        let emb: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let r1 = rotate_embedding(&emb, 1).expect("rotate 1 failed");
        let r1r1 = rotate_embedding(&r1, 1).expect("rotate 1 again failed");
        let r2 = rotate_embedding(&emb, 2).expect("rotate 2 failed");
        assert_eq!(r1r1, r2, "rotation-2 != rotation-1 applied twice");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // patch_agreement_loss tests
    // ─────────────────────────────────────────────────────────────────────────

    /// Identical unit-norm embeddings should yield negative loss (agreement
    /// maximised, cosine_sim = 1, loss = −1/temperature).
    #[test]
    fn test_patch_agreement_loss_identical_embeddings_negative() {
        let z = vec![1.0_f32, 0.0, 0.0, 0.0];
        let loss = patch_agreement_loss(&z, &z, 0.5).expect("patch_agreement_loss failed");
        assert!(
            loss < 0.0,
            "patch_agreement_loss for identical embeddings was {loss}, expected negative"
        );
    }

    /// For identical embeddings with temperature 0.5, loss = −cos(1)/0.5 = −2.
    #[test]
    fn test_patch_agreement_loss_identical_value() {
        let z = vec![1.0_f32, 0.0, 0.0];
        let loss = patch_agreement_loss(&z, &z, 0.5).expect("patch_agreement_loss failed");
        let expected = -1.0_f32 / 0.5;
        assert!(
            (loss - expected).abs() < 1e-5,
            "patch_agreement_loss = {loss}, expected {expected}"
        );
    }

    /// Orthogonal embeddings should have zero loss (cos_sim = 0).
    #[test]
    fn test_patch_agreement_loss_orthogonal_zero() {
        let z1 = vec![1.0_f32, 0.0];
        let z2 = vec![0.0_f32, 1.0];
        let loss = patch_agreement_loss(&z1, &z2, 1.0).expect("patch_agreement_loss failed");
        assert!(
            loss.abs() < 1e-6,
            "orthogonal embeddings gave non-zero loss: {loss}"
        );
    }

    /// Length mismatch should return an error.
    #[test]
    fn test_patch_agreement_loss_length_mismatch_error() {
        let result = patch_agreement_loss(&[1.0, 0.0], &[1.0, 0.0, 0.0], 0.5);
        assert!(result.is_err(), "expected error for length mismatch");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // mlm_accuracy tests
    // ─────────────────────────────────────────────────────────────────────────

    /// With perfect logits, accuracy should be 1.0.
    #[test]
    fn test_mlm_accuracy_perfect_logits() {
        let vocab_size = 10;
        let seq_len = 5;
        let targets = vec![3u32, 7];
        let positions = vec![1, 4];
        let logits = perfect_logits_at(seq_len, vocab_size, &[(1, 3), (4, 7)]);
        let acc =
            mlm_accuracy(&logits, &targets, &positions, vocab_size).expect("mlm_accuracy failed");
        assert!((acc - 1.0).abs() < 1e-6, "expected accuracy 1.0, got {acc}");
    }

    /// With uniform logits (effectively wrong predictions for non-0 targets),
    /// accuracy can be 0 if the target is never at index 0.
    #[test]
    fn test_mlm_accuracy_all_wrong_predictions() {
        let vocab_size = 5;
        let seq_len = 3;
        // All logits are 1.0 → argmax = 0 for every row.
        let logits = vec![1.0_f32; seq_len * vocab_size];
        // Targets are non-zero → always predicted wrongly.
        let targets = vec![1u32, 2];
        let positions = vec![0, 1];
        let acc =
            mlm_accuracy(&logits, &targets, &positions, vocab_size).expect("mlm_accuracy failed");
        assert_eq!(acc, 0.0, "expected accuracy 0.0, got {acc}");
    }

    /// mlm_accuracy should return an error for empty mask_positions.
    #[test]
    fn test_mlm_accuracy_empty_positions_error() {
        let logits = vec![1.0_f32; 10];
        let result = mlm_accuracy(&logits, &[], &[], 5);
        assert!(result.is_err(), "expected error for empty mask_positions");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional edge-case tests
    // ─────────────────────────────────────────────────────────────────────────

    /// MlmMasker should error on empty input.
    #[test]
    fn test_mlm_masker_empty_input_error() {
        let masker = MlmMasker::new(103, 30_000);
        let result = masker.mask_sequence(&[], 0);
        assert!(result.is_err(), "expected error for empty sequence");
    }

    /// MlmMasker should error when vocab_size is 0.
    #[test]
    fn test_mlm_masker_zero_vocab_size_error() {
        let masker = MlmMasker::new(0, 0);
        let result = masker.mask_sequence(&[1, 2, 3], 0);
        assert!(result.is_err(), "expected error for vocab_size=0");
    }

    /// rotate_embedding should error for empty embedding.
    #[test]
    fn test_rotate_embedding_empty_error() {
        let result = rotate_embedding(&[], 1);
        assert!(result.is_err(), "expected error for empty embedding");
    }
}
