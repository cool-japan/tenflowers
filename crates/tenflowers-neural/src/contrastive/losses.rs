//! Contrastive learning loss functions.
//!
//! This module provides modern self-supervised and supervised contrastive losses:
//!
//! - **NT-Xent** (Normalized Temperature-scaled Cross-Entropy): Used in SimCLR.
//! - **Supervised Contrastive Loss**: Extends contrastive learning to supervised settings.
//! - **InfoNCE**: Noise-Contrastive Estimation used broadly in CPC, MoCo, etc.
//! - **Utility helpers**: cosine similarity, L2 normalisation.

use tenflowers_core::{error::TensorError, Result};

// ─── helpers ────────────────────────────────────────────────────────────────

/// Compute the dot product of two equal-length slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute the L2 norm of a slice.
#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

// ─── public API ─────────────────────────────────────────────────────────────

/// Cosine similarity between two (possibly un-normalised) vectors.
///
/// Returns a value in `[-1, 1]` for unit-norm inputs.  Returns `0.0` when
/// either vector has zero norm (degenerate case).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(TensorError::InvalidArgument {
            operation: "cosine_similarity".to_string(),
            reason: format!(
                "vector length mismatch: a.len()={} != b.len()={}",
                a.len(),
                b.len()
            ),
            context: None,
        });
    }
    if a.is_empty() {
        return Err(TensorError::InvalidArgument {
            operation: "cosine_similarity".to_string(),
            reason: "vectors must be non-empty".to_string(),
            context: None,
        });
    }

    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }
    Ok(dot(a, b) / (norm_a * norm_b))
}

/// L2-normalise a batch of embeddings **in-place**.
///
/// Each row of `embeddings` (of length `dim`) is scaled to unit L2 norm.
/// Rows with zero norm are left unchanged.
///
/// # Arguments
/// * `embeddings` – flat buffer of shape `[n, dim]`
/// * `n`          – number of rows
/// * `dim`        – embedding dimension
pub fn l2_normalize_batch(embeddings: &mut [f32], n: usize, dim: usize) -> Result<()> {
    if embeddings.len() != n * dim {
        return Err(TensorError::InvalidShape {
            operation: "l2_normalize_batch".to_string(),
            reason: format!(
                "expected buffer of length n*dim={}*{}={}, got {}",
                n,
                dim,
                n * dim,
                embeddings.len()
            ),
            shape: None,
            context: None,
        });
    }
    for i in 0..n {
        let row = &mut embeddings[i * dim..(i + 1) * dim];
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in row.iter_mut() {
                *x /= norm;
            }
        }
    }
    Ok(())
}

/// NT-Xent loss (SimCLR, Chen et al. 2020).
///
/// Given a batch of 2N embeddings where rows `[0..N)` and `[N..2N)` are
/// corresponding augmented views:
///
/// ```text
/// L = -1/(2N) Σ_i [ log( exp(sim(z_i, z_j(i)) / τ) /
///                         Σ_{k≠i} exp(sim(z_i, z_k) / τ) ) ]
/// ```
///
/// The implementation L2-normalises the embeddings before computing
/// similarities, which is standard practice.
///
/// # Arguments
/// * `embeddings`  – flat `[2*batch_size, dim]` buffer (will **not** be
///                   mutated; normalisation happens on a local copy).
/// * `batch_size`  – N (number of original samples)
/// * `dim`         – embedding dimension
/// * `temperature` – τ, typically 0.07 – 0.5
pub fn nt_xent_loss(
    embeddings: &[f32],
    batch_size: usize,
    dim: usize,
    temperature: f32,
) -> Result<f32> {
    let total = 2 * batch_size;
    if embeddings.len() != total * dim {
        return Err(TensorError::InvalidShape {
            operation: "nt_xent_loss".to_string(),
            reason: format!(
                "expected {} values (2*batch_size*dim), got {}",
                total * dim,
                embeddings.len()
            ),
            shape: None,
            context: None,
        });
    }
    if temperature <= 0.0 {
        return Err(TensorError::InvalidArgument {
            operation: "nt_xent_loss".to_string(),
            reason: format!("temperature must be positive, got {}", temperature),
            context: None,
        });
    }
    if batch_size == 0 {
        return Err(TensorError::InvalidArgument {
            operation: "nt_xent_loss".to_string(),
            reason: "batch_size must be > 0".to_string(),
            context: None,
        });
    }

    // Work on a normalised copy.
    let mut emb = embeddings.to_vec();
    l2_normalize_batch(&mut emb, total, dim)?;

    // Pre-compute all pairwise similarities / temperature.
    // logits[i][j] = sim(z_i, z_j) / τ
    let mut logits = vec![0.0_f32; total * total];
    for i in 0..total {
        for j in 0..total {
            let a = &emb[i * dim..(i + 1) * dim];
            let b = &emb[j * dim..(j + 1) * dim];
            logits[i * total + j] = dot(a, b) / temperature;
        }
    }

    // Positive pair for anchor i is:
    //   if i < N  → j = i + N
    //   if i >= N → j = i - N
    let mut loss_sum = 0.0_f32;
    for i in 0..total {
        let pos_j = if i < batch_size {
            i + batch_size
        } else {
            i - batch_size
        };

        let pos_logit = logits[i * total + pos_j];

        // log-sum-exp over all k ≠ i
        // For numerical stability use max subtraction.
        let max_logit = (0..total)
            .filter(|&k| k != i)
            .map(|k| logits[i * total + k])
            .fold(f32::NEG_INFINITY, f32::max);

        let sum_exp: f32 = (0..total)
            .filter(|&k| k != i)
            .map(|k| (logits[i * total + k] - max_logit).exp())
            .sum();

        // log(exp(pos) / sum_exp_all_neg)
        //   = pos - (max + log(sum_exp))
        let log_denom = max_logit + sum_exp.ln();
        loss_sum += log_denom - pos_logit;
    }

    Ok(loss_sum / total as f32)
}

/// Supervised contrastive loss (Khosla et al. 2020).
///
/// For each anchor `i`, positives are all samples with the same label
/// (excluding the anchor itself).  The loss is:
///
/// ```text
/// L = -1/N Σ_i  1/|P(i)| Σ_{p∈P(i)} log [ exp(sim(z_i, z_p)/τ) /
///                                          Σ_{a≠i} exp(sim(z_i, z_a)/τ) ]
/// ```
///
/// # Arguments
/// * `embeddings`  – flat `[batch_size, dim]` buffer
/// * `labels`      – integer class ids of length `batch_size`
/// * `dim`         – embedding dimension
/// * `temperature` – τ
pub fn supervised_contrastive_loss(
    embeddings: &[f32],
    labels: &[usize],
    dim: usize,
    temperature: f32,
) -> Result<f32> {
    let n = labels.len();
    if n == 0 {
        return Err(TensorError::InvalidArgument {
            operation: "supervised_contrastive_loss".to_string(),
            reason: "labels must be non-empty".to_string(),
            context: None,
        });
    }
    if embeddings.len() != n * dim {
        return Err(TensorError::InvalidShape {
            operation: "supervised_contrastive_loss".to_string(),
            reason: format!(
                "expected {}*{}={} values, got {}",
                n,
                dim,
                n * dim,
                embeddings.len()
            ),
            shape: None,
            context: None,
        });
    }
    if temperature <= 0.0 {
        return Err(TensorError::InvalidArgument {
            operation: "supervised_contrastive_loss".to_string(),
            reason: format!("temperature must be positive, got {}", temperature),
            context: None,
        });
    }

    // Normalise embeddings.
    let mut emb = embeddings.to_vec();
    l2_normalize_batch(&mut emb, n, dim)?;

    // Pre-compute pairwise scaled similarities.
    let mut logits = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let a = &emb[i * dim..(i + 1) * dim];
            let b = &emb[j * dim..(j + 1) * dim];
            logits[i * n + j] = dot(a, b) / temperature;
        }
    }

    let mut total_loss = 0.0_f32;
    let mut valid_anchors = 0usize;

    for i in 0..n {
        // Collect positives for anchor i.
        let positives: Vec<usize> = (0..n)
            .filter(|&j| j != i && labels[j] == labels[i])
            .collect();

        if positives.is_empty() {
            // Anchor has no in-batch positive; skip (contributes 0 to loss).
            continue;
        }
        valid_anchors += 1;

        // log-sum-exp denominator over all a ≠ i.
        let max_logit = (0..n)
            .filter(|&k| k != i)
            .map(|k| logits[i * n + k])
            .fold(f32::NEG_INFINITY, f32::max);

        let sum_exp: f32 = (0..n)
            .filter(|&k| k != i)
            .map(|k| (logits[i * n + k] - max_logit).exp())
            .sum();

        let log_denom = max_logit + sum_exp.ln();

        // Mean over positives.
        let pos_sum: f32 = positives
            .iter()
            .map(|&p| log_denom - logits[i * n + p])
            .sum();

        total_loss += pos_sum / positives.len() as f32;
    }

    if valid_anchors == 0 {
        // Degenerate: all samples are unique classes — return 0.
        return Ok(0.0);
    }

    Ok(total_loss / valid_anchors as f32)
}

/// InfoNCE loss (van den Oord et al. 2018).
///
/// ```text
/// L = -log [ exp(sim(q, k+)/τ) / ( exp(sim(q,k+)/τ) + Σ_neg exp(sim(q,k-)/τ) ) ]
/// ```
///
/// # Arguments
/// * `query`         – query embedding of length `dim`
/// * `positive_key`  – positive key embedding of length `dim`
/// * `negative_keys` – flat `[n_neg, dim]` buffer of negative keys
/// * `n_neg`         – number of negative keys
/// * `dim`           – embedding dimension
/// * `temperature`   – τ
pub fn info_nce_loss(
    query: &[f32],
    positive_key: &[f32],
    negative_keys: &[f32],
    n_neg: usize,
    dim: usize,
    temperature: f32,
) -> Result<f32> {
    if query.len() != dim {
        return Err(TensorError::InvalidArgument {
            operation: "info_nce_loss".to_string(),
            reason: format!("query length {} != dim {}", query.len(), dim),
            context: None,
        });
    }
    if positive_key.len() != dim {
        return Err(TensorError::InvalidArgument {
            operation: "info_nce_loss".to_string(),
            reason: format!("positive_key length {} != dim {}", positive_key.len(), dim),
            context: None,
        });
    }
    if negative_keys.len() != n_neg * dim {
        return Err(TensorError::InvalidShape {
            operation: "info_nce_loss".to_string(),
            reason: format!(
                "negative_keys length {} != n_neg*dim={}*{}={}",
                negative_keys.len(),
                n_neg,
                dim,
                n_neg * dim
            ),
            shape: None,
            context: None,
        });
    }
    if temperature <= 0.0 {
        return Err(TensorError::InvalidArgument {
            operation: "info_nce_loss".to_string(),
            reason: format!("temperature must be positive, got {}", temperature),
            context: None,
        });
    }

    // L2-normalise query and keys on local copies.
    let q_norm = l2_norm(query);
    let norm_q: Vec<f32> = if q_norm > 0.0 {
        query.iter().map(|x| x / q_norm).collect()
    } else {
        query.to_vec()
    };

    let pk_norm = l2_norm(positive_key);
    let norm_pk: Vec<f32> = if pk_norm > 0.0 {
        positive_key.iter().map(|x| x / pk_norm).collect()
    } else {
        positive_key.to_vec()
    };

    let pos_logit = dot(&norm_q, &norm_pk) / temperature;

    // Negative logits.
    let neg_logits: Vec<f32> = (0..n_neg)
        .map(|k| {
            let nk = &negative_keys[k * dim..(k + 1) * dim];
            let nk_norm = l2_norm(nk);
            let logit = if nk_norm > 0.0 {
                nk.iter()
                    .zip(norm_q.iter())
                    .map(|(a, b)| a / nk_norm * b)
                    .sum::<f32>()
                    / temperature
            } else {
                0.0
            };
            logit
        })
        .collect();

    // log-sum-exp for numerical stability.
    let max_logit = neg_logits.iter().cloned().fold(pos_logit, f32::max);

    let sum_exp: f32 = std::iter::once(pos_logit)
        .chain(neg_logits.iter().cloned())
        .map(|l| (l - max_logit).exp())
        .sum();

    let log_denom = max_logit + sum_exp.ln();
    Ok(log_denom - pos_logit)
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    // ── cosine_similarity ──────────────────────────────────────────────────

    #[test]
    fn test_cosine_similarity_parallel() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![2.0_f32, 0.0, 0.0]; // same direction, different magnitude
        let sim = cosine_similarity(&a, &b).expect("cosine_similarity");
        assert!((sim - 1.0).abs() < EPS, "parallel vectors: got {}", sim);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b).expect("cosine_similarity");
        assert!(sim.abs() < EPS, "orthogonal vectors: got {}", sim);
    }

    #[test]
    fn test_cosine_similarity_anti_parallel() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![-1.0_f32, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b).expect("cosine_similarity");
        assert!(
            (sim + 1.0).abs() < EPS,
            "anti-parallel vectors: got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_length_mismatch() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![1.0_f32, 0.0, 0.0];
        assert!(cosine_similarity(&a, &b).is_err());
    }

    #[test]
    fn test_cosine_similarity_empty_error() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert!(cosine_similarity(&a, &b).is_err());
    }

    // ── l2_normalize_batch ────────────────────────────────────────────────

    #[test]
    fn test_l2_normalize_batch_unit_norms() {
        let mut emb = vec![3.0_f32, 4.0, 0.0, 1.0, 1.0, 1.0];
        l2_normalize_batch(&mut emb, 2, 3).expect("normalize");
        // Row 0: [3/5, 4/5, 0]
        let n0: f32 = emb[..3].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n0 - 1.0).abs() < EPS, "row 0 norm = {}", n0);
        // Row 1: [1/√3, 1/√3, 1/√3]
        let n1: f32 = emb[3..].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n1 - 1.0).abs() < EPS, "row 1 norm = {}", n1);
    }

    #[test]
    fn test_l2_normalize_batch_zero_vector() {
        let mut emb = vec![0.0_f32, 0.0, 0.0];
        l2_normalize_batch(&mut emb, 1, 3).expect("normalize");
        // Zero vector should remain zero (not NaN).
        assert_eq!(emb, vec![0.0_f32, 0.0, 0.0]);
    }

    #[test]
    fn test_l2_normalize_batch_shape_mismatch() {
        let mut emb = vec![1.0_f32, 2.0, 3.0];
        assert!(l2_normalize_batch(&mut emb, 2, 3).is_err());
    }

    // ── nt_xent_loss ──────────────────────────────────────────────────────

    #[test]
    fn test_nt_xent_loss_non_negative() {
        // Arbitrary embeddings — loss must be ≥ 0.
        let emb: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.1, 0.0, // augmented view of sample 0
            0.0, 0.9, 0.1, // augmented view of sample 1
        ];
        let loss = nt_xent_loss(&emb, 2, 3, 0.5).expect("nt_xent");
        assert!(loss >= 0.0, "loss must be non-negative, got {}", loss);
    }

    #[test]
    fn test_nt_xent_loss_lower_for_similar_pairs() {
        // When positive pairs are nearly identical, loss should be lower than
        // when pairs are orthogonal.
        let batch_size = 2;
        let dim = 4;
        // Similar pairs: row 0≈row 2, row 1≈row 3
        let similar: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.99, 0.1, 0.0, 0.0, 0.0, 0.99, 0.1, 0.0,
        ];
        // Dissimilar pairs: row 0⊥row 2, row 1⊥row 3
        let dissimilar: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let loss_similar = nt_xent_loss(&similar, batch_size, dim, 0.5).expect("nt_xent");
        let loss_dissimilar = nt_xent_loss(&dissimilar, batch_size, dim, 0.5).expect("nt_xent");
        assert!(
            loss_similar < loss_dissimilar,
            "similar pairs loss ({}) should be < dissimilar pairs loss ({})",
            loss_similar,
            loss_dissimilar
        );
    }

    #[test]
    fn test_nt_xent_loss_invalid_temperature() {
        let emb = vec![1.0_f32, 0.0, 0.0, 1.0];
        assert!(nt_xent_loss(&emb, 1, 2, 0.0).is_err());
        assert!(nt_xent_loss(&emb, 1, 2, -1.0).is_err());
    }

    // ── info_nce_loss ─────────────────────────────────────────────────────

    #[test]
    fn test_info_nce_loss_one_negative_known_value() {
        // With 1 negative, the loss is:
        //   -log( exp(s_pos/τ) / (exp(s_pos/τ) + exp(s_neg/τ)) )
        // For unit vectors: q=[1,0], k+=[1,0], k-=[0,1]
        //   s_pos = 1.0, s_neg = 0.0, τ = 1.0
        //   loss = log(1 + exp(-1)) ≈ 0.3133
        let q = vec![1.0_f32, 0.0];
        let pk = vec![1.0_f32, 0.0];
        let nk = vec![0.0_f32, 1.0];
        let loss = info_nce_loss(&q, &pk, &nk, 1, 2, 1.0).expect("info_nce");
        let expected = (1.0_f32 + (-1.0_f32).exp()).ln();
        assert!(
            (loss - expected).abs() < 1e-4,
            "expected {}, got {}",
            expected,
            loss
        );
    }

    #[test]
    fn test_info_nce_loss_non_negative() {
        let q = vec![0.5_f32, 0.5, 0.5, 0.5];
        let pk = vec![0.5_f32, 0.5, 0.5, 0.5];
        let nk = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let loss = info_nce_loss(&q, &pk, &nk, 2, 4, 0.1).expect("info_nce");
        assert!(loss >= 0.0, "loss must be non-negative, got {}", loss);
    }

    #[test]
    fn test_info_nce_loss_zero_negatives_is_zero() {
        // With no negatives, the denominator equals the numerator → loss = 0.
        let q = vec![1.0_f32, 0.0];
        let pk = vec![1.0_f32, 0.0];
        let nk: Vec<f32> = vec![];
        let loss = info_nce_loss(&q, &pk, &nk, 0, 2, 0.5).expect("info_nce");
        assert!(loss.abs() < EPS, "expected ~0.0, got {}", loss);
    }

    // ── supervised_contrastive_loss ───────────────────────────────────────

    #[test]
    fn test_supervised_contrastive_all_same_class() {
        // All four samples belong to class 0.  Every pair is a positive pair.
        // The loss should be well-defined and finite.
        let emb: Vec<f32> = vec![1.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.8, 0.2, 0.0, 0.95, 0.05, 0.0];
        let labels = vec![0usize, 0, 0, 0];
        let loss = supervised_contrastive_loss(&emb, &labels, 3, 0.5).expect("sup_con");
        assert!(loss.is_finite(), "loss should be finite, got {}", loss);
        assert!(loss >= 0.0, "loss should be non-negative, got {}", loss);
    }

    #[test]
    fn test_supervised_contrastive_all_unique_classes_returns_zero() {
        // Every sample is its own class → no positives → loss = 0.
        let emb: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let labels = vec![0usize, 1, 2];
        let loss = supervised_contrastive_loss(&emb, &labels, 2, 0.5).expect("sup_con");
        assert!(loss.abs() < EPS, "expected 0, got {}", loss);
    }
}
