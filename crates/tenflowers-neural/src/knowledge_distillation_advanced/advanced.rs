//! Advanced knowledge distillation methods.
//!
//! # Token-Level Distillation (TinyBERT-style)
//! - [`TokenEmbeddingDistiller`]: align token embeddings via MSE + linear projection.
//! - [`AttentionMapDistiller`]: distill attention maps (MSE on softmax weights).
//! - [`HiddenStateDistiller`]: layer-to-layer hidden state transfer with learnable mapping.
//!
//! # Contrastive Distillation
//! - [`ContrastiveDistillationLoss`]: SimKD-style — pull student toward teacher, push away negatives.
//! - [`SemckdDistiller`]: Semantic Calibration KD (Chen 2021) via cross-covariance alignment.
//!
//! # Structured Distillation
//! - [`RelationalKdLoss`]: RKD (Park 2019) — pair-wise and triplet-wise structural relations.
//! - [`GraphDistillationLayer`]: distill graph structure awareness from GNN teacher to student.

use super::{dot, kl_div, l2_normalize_mut, mse_slice, softmax_temp};
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// §11  Token-Level Distillation (TinyBERT-style)
// ─────────────────────────────────────────────────────────────────────────────

/// TinyBERT-style token embedding distiller.
///
/// Projects student token embeddings into teacher embedding space via a learnable
/// linear projection and minimises MSE between projected student and teacher embeddings.
#[derive(Debug, Clone)]
pub struct TokenEmbeddingDistiller {
    /// Projection matrix (row-major, shape `[teacher_dim × student_dim]`).
    pub projection: Vec<f32>,
    /// Student embedding dimension.
    pub student_dim: usize,
    /// Teacher embedding dimension.
    pub teacher_dim: usize,
}

impl TokenEmbeddingDistiller {
    /// Construct with a random projection matrix.
    pub fn new(student_dim: usize, teacher_dim: usize) -> Result<Self> {
        if student_dim == 0 || teacher_dim == 0 {
            return Err(TensorError::invalid_argument(
                "student_dim and teacher_dim must be > 0".to_string(),
            ));
        }
        // Xavier-ish initialisation for the projection
        let gain = (2.0_f32 / (student_dim + teacher_dim) as f32).sqrt();
        let mut seed: u64 = 0x1234_5678_9ABC_DEF0;
        let projection: Vec<f32> = (0..teacher_dim * student_dim)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (seed >> 33) as u32;
                (bits as f32 / u32::MAX as f32) * 2.0 * gain - gain
            })
            .collect();

        Ok(Self {
            projection,
            student_dim,
            teacher_dim,
        })
    }

    /// Project a single student token embedding into teacher space.
    pub fn project(&self, student_emb: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.teacher_dim];
        for i in 0..self.teacher_dim {
            let row = i * self.student_dim;
            out[i] = student_emb
                .iter()
                .enumerate()
                .take(self.student_dim)
                .map(|(j, &x)| {
                    self.projection
                        .get(row + j)
                        .copied()
                        .unwrap_or(0.0)
                        * x
                })
                .sum();
        }
        out
    }

    /// Compute token embedding distillation loss (MSE after projection).
    ///
    /// `student_tokens` and `teacher_tokens` are flat `[seq_len × dim]` matrices.
    ///
    /// # Errors
    /// Returns error if lengths are inconsistent.
    pub fn token_embedding_loss(
        &self,
        student_tokens: &[Vec<f32>],
        teacher_tokens: &[Vec<f32>],
    ) -> Result<f32> {
        if student_tokens.len() != teacher_tokens.len() {
            return Err(TensorError::invalid_argument(format!(
                "student_tokens length {} != teacher_tokens length {}",
                student_tokens.len(),
                teacher_tokens.len()
            )));
        }
        if student_tokens.is_empty() {
            return Ok(0.0);
        }

        let mut total = 0.0_f32;
        for (s, t) in student_tokens.iter().zip(teacher_tokens.iter()) {
            let projected = self.project(s);
            total += mse_slice(&projected, t);
        }
        Ok(total / student_tokens.len() as f32)
    }
}

/// Attention map distiller (TinyBERT-style).
///
/// Distills attention maps from teacher to student layers by minimising MSE
/// between softmax attention weights.
#[derive(Debug, Clone)]
pub struct AttentionMapDistiller {
    /// Number of attention heads.
    pub n_heads: usize,
    /// Sequence length.
    pub seq_len: usize,
}

impl AttentionMapDistiller {
    /// Construct for a given head/seq configuration.
    pub fn new(n_heads: usize, seq_len: usize) -> Result<Self> {
        if n_heads == 0 || seq_len == 0 {
            return Err(TensorError::invalid_argument(
                "n_heads and seq_len must be > 0".to_string(),
            ));
        }
        Ok(Self { n_heads, seq_len })
    }

    /// Compute attention map distillation loss.
    ///
    /// `student_attn` and `teacher_attn` are flat `[n_heads × seq_len × seq_len]`
    /// attention weight arrays (already softmax-ed).
    ///
    /// # Errors
    /// Returns error on size mismatch.
    pub fn attention_map_loss(
        &self,
        student_attn: &[f32],
        teacher_attn: &[f32],
    ) -> Result<f32> {
        let expected = self.n_heads * self.seq_len * self.seq_len;
        if student_attn.len() != expected || teacher_attn.len() != expected {
            return Err(TensorError::invalid_argument(format!(
                "attention arrays must have length {expected} (n_heads*seq_len^2), \
                 got student={} teacher={}",
                student_attn.len(),
                teacher_attn.len()
            )));
        }

        Ok(mse_slice(student_attn, teacher_attn))
    }

    /// Compute per-head average MSE.
    pub fn per_head_loss(
        &self,
        student_attn: &[f32],
        teacher_attn: &[f32],
    ) -> Result<Vec<f32>> {
        let head_size = self.seq_len * self.seq_len;
        let expected = self.n_heads * head_size;
        if student_attn.len() != expected || teacher_attn.len() != expected {
            return Err(TensorError::invalid_argument(format!(
                "expected {expected} attention values, got student={} teacher={}",
                student_attn.len(),
                teacher_attn.len()
            )));
        }

        let losses: Vec<f32> = (0..self.n_heads)
            .map(|h| {
                let s_start = h * head_size;
                let s = &student_attn[s_start..s_start + head_size];
                let t = &teacher_attn[s_start..s_start + head_size];
                mse_slice(s, t)
            })
            .collect();
        Ok(losses)
    }
}

/// Hidden state distiller with learnable layer mapping.
///
/// For each teacher layer, the student provides a corresponding hidden state.
/// A learnable linear projection maps student → teacher space.
#[derive(Debug, Clone)]
pub struct HiddenStateDistiller {
    /// Per-layer projection: `projection[layer]` has shape `[teacher_dim × student_dim]` flat.
    pub projections: Vec<Vec<f32>>,
    /// Student hidden dimension.
    pub student_dim: usize,
    /// Teacher hidden dimension.
    pub teacher_dim: usize,
}

impl HiddenStateDistiller {
    /// Construct with one projection per teacher layer.
    pub fn new(n_layers: usize, student_dim: usize, teacher_dim: usize) -> Result<Self> {
        if n_layers == 0 || student_dim == 0 || teacher_dim == 0 {
            return Err(TensorError::invalid_argument(
                "n_layers, student_dim, teacher_dim must be > 0".to_string(),
            ));
        }
        let gain = (2.0_f32 / (student_dim + teacher_dim) as f32).sqrt();
        let mut seed: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let projections: Vec<Vec<f32>> = (0..n_layers)
            .map(|_| {
                (0..teacher_dim * student_dim)
                    .map(|_| {
                        seed = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let bits = (seed >> 33) as u32;
                        (bits as f32 / u32::MAX as f32) * 2.0 * gain - gain
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            projections,
            student_dim,
            teacher_dim,
        })
    }

    /// Project a student hidden state at a given layer.
    pub fn project_layer(&self, layer: usize, student_hidden: &[f32]) -> Vec<f32> {
        if layer >= self.projections.len() {
            return vec![0.0; self.teacher_dim];
        }
        let proj = &self.projections[layer];
        let mut out = vec![0.0_f32; self.teacher_dim];
        for i in 0..self.teacher_dim {
            let row = i * self.student_dim;
            out[i] = student_hidden
                .iter()
                .enumerate()
                .take(self.student_dim)
                .map(|(j, &x)| proj.get(row + j).copied().unwrap_or(0.0) * x)
                .sum();
        }
        out
    }

    /// Compute hidden state distillation loss across all layers.
    ///
    /// `student_hiddens` and `teacher_hiddens` are indexed by layer.
    ///
    /// # Errors
    /// Returns error if lengths differ.
    pub fn hidden_state_loss(
        &self,
        student_hiddens: &[Vec<f32>],
        teacher_hiddens: &[Vec<f32>],
    ) -> Result<f32> {
        if student_hiddens.len() != teacher_hiddens.len() {
            return Err(TensorError::invalid_argument(format!(
                "student_hiddens ({}) and teacher_hiddens ({}) must have same length",
                student_hiddens.len(),
                teacher_hiddens.len()
            )));
        }
        if student_hiddens.is_empty() {
            return Ok(0.0);
        }
        let mut total = 0.0_f32;
        for (layer, (s, t)) in student_hiddens.iter().zip(teacher_hiddens.iter()).enumerate() {
            let projected = self.project_layer(layer, s);
            total += mse_slice(&projected, t);
        }
        Ok(total / student_hiddens.len() as f32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §12  Contrastive Distillation
// ─────────────────────────────────────────────────────────────────────────────

/// Contrastive distillation loss (SimKD-style).
///
/// Pulls student embeddings toward teacher embeddings and pushes them away from
/// negative (non-matched) teacher embeddings using an InfoNCE-style loss.
#[derive(Debug, Clone)]
pub struct ContrastiveDistillationLoss {
    /// InfoNCE temperature.
    pub temperature: f32,
}

impl ContrastiveDistillationLoss {
    /// Construct with given temperature.
    pub fn new(temperature: f32) -> Result<Self> {
        if temperature <= 0.0 {
            return Err(TensorError::invalid_argument(format!(
                "temperature must be positive, got {temperature}"
            )));
        }
        Ok(Self { temperature })
    }

    /// Compute contrastive distillation loss.
    ///
    /// `student_embs` and `teacher_embs` are parallel arrays of embeddings.
    /// For each student embedding `i`, the positive is `teacher_embs[i]` and
    /// all others are negatives.
    pub fn loss(&self, student_embs: &[Vec<f32>], teacher_embs: &[Vec<f32>]) -> f32 {
        let n = student_embs.len().min(teacher_embs.len());
        if n == 0 {
            return 0.0;
        }

        let t = self.temperature;
        let mut total = 0.0_f32;

        // L2-normalise all embeddings for cosine similarity
        let mut s_normed: Vec<Vec<f32>> = student_embs[..n].to_vec();
        let mut t_normed: Vec<Vec<f32>> = teacher_embs[..n].to_vec();
        for v in s_normed.iter_mut() {
            l2_normalize_mut(v);
        }
        for v in t_normed.iter_mut() {
            l2_normalize_mut(v);
        }

        for i in 0..n {
            let logits: Vec<f32> = (0..n)
                .map(|j| dot(&s_normed[i], &t_normed[j]) / t)
                .collect();

            let soft = softmax_temp(&logits, 1.0);
            // Positive is index i
            total -= (soft[i] + 1e-30_f32).ln();
        }

        total / n as f32
    }

    /// Symmetric contrastive loss (both student→teacher and teacher→student).
    pub fn symmetric_loss(
        &self,
        student_embs: &[Vec<f32>],
        teacher_embs: &[Vec<f32>],
    ) -> f32 {
        let l_s = self.loss(student_embs, teacher_embs);
        let l_t = self.loss(teacher_embs, student_embs);
        (l_s + l_t) * 0.5
    }
}

/// SemCKD distiller — Semantic Calibration for Knowledge Distillation (Chen 2021).
///
/// Aligns the cross-covariance structure of student features to match that of
/// the teacher features.  The loss is the Frobenius norm of the difference
/// between normalised covariance matrices.
#[derive(Debug, Clone)]
pub struct SemckdDistiller {
    /// Feature dimension (must match student and teacher feature dims).
    pub feature_dim: usize,
}

impl SemckdDistiller {
    /// Construct for a given feature dimension.
    pub fn new(feature_dim: usize) -> Result<Self> {
        if feature_dim == 0 {
            return Err(TensorError::invalid_argument(
                "feature_dim must be > 0".to_string(),
            ));
        }
        Ok(Self { feature_dim })
    }

    /// Compute the normalised covariance matrix for a batch of feature vectors.
    ///
    /// Returns a flat `[d × d]` matrix.
    pub fn compute_covariance(&self, features: &[Vec<f32>]) -> Vec<f32> {
        let n = features.len();
        let d = self.feature_dim;
        if n == 0 {
            return vec![0.0; d * d];
        }

        // Compute mean
        let mut mean = vec![0.0_f32; d];
        for feat in features {
            for (m, &x) in mean.iter_mut().zip(feat.iter()) {
                *m += x;
            }
        }
        for m in mean.iter_mut() {
            *m /= n as f32;
        }

        // Compute covariance
        let mut cov = vec![0.0_f32; d * d];
        for feat in features {
            for i in 0..d.min(feat.len()) {
                for j in 0..d.min(feat.len()) {
                    cov[i * d + j] += (feat[i] - mean[i]) * (feat[j] - mean[j]);
                }
            }
        }
        let scale = (n.max(1) as f32).recip();
        for c in cov.iter_mut() {
            *c *= scale;
        }
        cov
    }

    /// Compute the SemCKD alignment loss: normalised Frobenius distance
    /// between teacher and student covariance matrices.
    ///
    /// # Errors
    /// Returns error if student/teacher feature lengths are inconsistent.
    pub fn semckd_loss(
        &self,
        student_features: &[Vec<f32>],
        teacher_features: &[Vec<f32>],
    ) -> Result<f32> {
        if student_features.len() != teacher_features.len() {
            return Err(TensorError::invalid_argument(format!(
                "student_features ({}) and teacher_features ({}) must have same batch size",
                student_features.len(),
                teacher_features.len()
            )));
        }

        let cov_s = self.compute_covariance(student_features);
        let cov_t = self.compute_covariance(teacher_features);

        let d = self.feature_dim;
        let frob_sq: f32 = (0..d * d)
            .map(|i| (cov_s[i] - cov_t[i]).powi(2))
            .sum();

        Ok(frob_sq.sqrt() / d as f32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §13  Structured Distillation
// ─────────────────────────────────────────────────────────────────────────────

/// RKD (Relational Knowledge Distillation) loss — Park 2019.
///
/// Transfers pair-wise (distance) and triplet-wise (angle) structural relations
/// from teacher to student.
#[derive(Debug, Clone)]
pub struct RelationalKdLoss {
    /// Weight for distance loss.
    pub distance_weight: f32,
    /// Weight for angle loss.
    pub angle_weight: f32,
}

impl RelationalKdLoss {
    /// Construct with given loss weights.
    pub fn new(distance_weight: f32, angle_weight: f32) -> Self {
        Self {
            distance_weight,
            angle_weight,
        }
    }

    fn pairwise_distances(embs: &[Vec<f32>]) -> Vec<f32> {
        let n = embs.len();
        let mut dists = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                let d: f32 = embs[i]
                    .iter()
                    .zip(embs[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                dists.push(d);
            }
        }
        dists
    }

    /// Compute RKD distance loss: Huber loss on pairwise distance ratios.
    pub fn distance_loss(
        &self,
        student_embs: &[Vec<f32>],
        teacher_embs: &[Vec<f32>],
    ) -> f32 {
        let n = student_embs.len().min(teacher_embs.len());
        if n < 2 {
            return 0.0;
        }
        let s_dists = Self::pairwise_distances(student_embs);
        let t_dists = Self::pairwise_distances(teacher_embs);

        // Normalise by mean
        let s_mean = s_dists.iter().sum::<f32>() / s_dists.len().max(1) as f32;
        let t_mean = t_dists.iter().sum::<f32>() / t_dists.len().max(1) as f32;

        let s_norm: Vec<f32> = s_dists
            .iter()
            .map(|&d| d / s_mean.max(1e-8))
            .collect();
        let t_norm: Vec<f32> = t_dists
            .iter()
            .map(|&d| d / t_mean.max(1e-8))
            .collect();

        // Huber loss
        let delta = 1.0_f32;
        let huber_loss: f32 = s_norm
            .iter()
            .zip(t_norm.iter())
            .map(|(s, t)| {
                let diff = (s - t).abs();
                if diff < delta {
                    0.5 * diff * diff
                } else {
                    delta * (diff - 0.5 * delta)
                }
            })
            .sum();
        huber_loss / s_norm.len().max(1) as f32
    }

    /// Compute RKD angle loss: MSE on triplet-wise angle cosines.
    pub fn angle_loss(&self, student_embs: &[Vec<f32>], teacher_embs: &[Vec<f32>]) -> f32 {
        let n = student_embs.len().min(teacher_embs.len());
        if n < 3 {
            return 0.0;
        }

        let mut total = 0.0_f32;
        let mut count = 0usize;

        // For every triplet (i, j, k), compute angle at j
        for j in 0..n.min(8) {
            // Limit for tractability
            for i in 0..n {
                if i == j {
                    continue;
                }
                for k in (i + 1)..n {
                    if k == j {
                        continue;
                    }

                    // Student angle
                    let s_ij: Vec<f32> = student_embs[i]
                        .iter()
                        .zip(student_embs[j].iter())
                        .map(|(a, b)| a - b)
                        .collect();
                    let s_jk: Vec<f32> = student_embs[k]
                        .iter()
                        .zip(student_embs[j].iter())
                        .map(|(a, b)| a - b)
                        .collect();

                    let mut s_ij_n = s_ij.clone();
                    let mut s_jk_n = s_jk.clone();
                    l2_normalize_mut(&mut s_ij_n);
                    l2_normalize_mut(&mut s_jk_n);
                    let s_cos = dot(&s_ij_n, &s_jk_n);

                    // Teacher angle
                    let t_ij: Vec<f32> = teacher_embs[i]
                        .iter()
                        .zip(teacher_embs[j].iter())
                        .map(|(a, b)| a - b)
                        .collect();
                    let t_jk: Vec<f32> = teacher_embs[k]
                        .iter()
                        .zip(teacher_embs[j].iter())
                        .map(|(a, b)| a - b)
                        .collect();

                    let mut t_ij_n = t_ij.clone();
                    let mut t_jk_n = t_jk.clone();
                    l2_normalize_mut(&mut t_ij_n);
                    l2_normalize_mut(&mut t_jk_n);
                    let t_cos = dot(&t_ij_n, &t_jk_n);

                    total += (s_cos - t_cos).powi(2);
                    count += 1;
                }
            }
        }

        if count == 0 {
            0.0
        } else {
            total / count as f32
        }
    }

    /// Combined RKD loss = distance_weight * dist_loss + angle_weight * angle_loss.
    pub fn combined_loss(
        &self,
        student_embs: &[Vec<f32>],
        teacher_embs: &[Vec<f32>],
    ) -> f32 {
        let dl = self.distance_loss(student_embs, teacher_embs);
        let al = self.angle_loss(student_embs, teacher_embs);
        self.distance_weight * dl + self.angle_weight * al
    }
}

/// Graph distillation layer — transfers graph structure awareness from a GNN teacher.
///
/// The teacher GNN produces node embeddings that encode local neighbourhood
/// information.  This distiller minimises the discrepancy in pairwise cosine
/// similarities between teacher and student node embeddings.
#[derive(Debug, Clone)]
pub struct GraphDistillationLayer {
    /// Number of graph nodes in the batch.
    pub n_nodes: usize,
    /// Embedding dimension for graph nodes.
    pub node_dim: usize,
    /// KL-divergence temperature for softening the similarity distribution.
    pub temperature: f32,
}

impl GraphDistillationLayer {
    /// Construct for a given graph size and embedding dimension.
    pub fn new(n_nodes: usize, node_dim: usize, temperature: f32) -> Result<Self> {
        if n_nodes == 0 || node_dim == 0 {
            return Err(TensorError::invalid_argument(
                "n_nodes and node_dim must be > 0".to_string(),
            ));
        }
        if temperature <= 0.0 {
            return Err(TensorError::invalid_argument(format!(
                "temperature must be positive, got {temperature}"
            )));
        }
        Ok(Self {
            n_nodes,
            node_dim,
            temperature,
        })
    }

    /// Compute pairwise cosine similarity matrix for node embeddings.
    pub fn similarity_matrix(&self, node_embs: &[Vec<f32>]) -> Vec<f32> {
        let n = node_embs.len();
        let mut normed: Vec<Vec<f32>> = node_embs.to_vec();
        for v in normed.iter_mut() {
            l2_normalize_mut(v);
        }
        let mut sims = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                sims[i * n + j] = dot(&normed[i], &normed[j]);
            }
        }
        sims
    }

    /// Graph distillation loss: KL divergence between teacher and student
    /// row-wise similarity distributions.
    ///
    /// # Errors
    /// Returns error if node embeddings have different counts.
    pub fn graph_distil_loss(
        &self,
        student_nodes: &[Vec<f32>],
        teacher_nodes: &[Vec<f32>],
    ) -> Result<f32> {
        if student_nodes.len() != teacher_nodes.len() {
            return Err(TensorError::invalid_argument(format!(
                "student_nodes ({}) and teacher_nodes ({}) must have same length",
                student_nodes.len(),
                teacher_nodes.len()
            )));
        }
        let n = student_nodes.len();
        if n == 0 {
            return Ok(0.0);
        }

        let s_sims = self.similarity_matrix(student_nodes);
        let t_sims = self.similarity_matrix(teacher_nodes);

        let mut total_kl = 0.0_f32;
        for i in 0..n {
            let s_row: Vec<f32> = (0..n).map(|j| s_sims[i * n + j]).collect();
            let t_row: Vec<f32> = (0..n).map(|j| t_sims[i * n + j]).collect();

            // Softmax with temperature to get distributions
            let s_dist = softmax_temp(&s_row, self.temperature);
            let t_dist = softmax_temp(&t_row, self.temperature);

            total_kl += kl_div(&t_dist, &s_dist);
        }

        Ok(total_kl / n as f32)
    }
}
