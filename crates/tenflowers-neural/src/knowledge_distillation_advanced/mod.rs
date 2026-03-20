//! # Advanced Knowledge Distillation
//!
//! Production-grade advanced knowledge distillation methods beyond the
//! standard Hinton KD (which lives in `distillation.rs`).
//!
//! ## Methods
//!
//! 1. **OnlineDistillation** — mutual teaching among multiple students (no fixed teacher).
//! 2. **SelfDistillation** — later layers teach earlier layers within the same model.
//! 3. **DataFreeDistillation** — synthesize training data from teacher (DREAMING).
//! 4. **TaskAgnosticDistillation** — distill without labels using unlabeled corpus.
//! 5. **PatchDistillation** — vision-specific patch-level representation distillation.
//! 6. **GraphDistillation** — relational knowledge via RKD angle loss.
//! 7. **ProgressiveDistillation** — shrink model step-by-step across stages.
//! 8. **AttentionTransfer** — transfer attention maps and Gram matrices.
//! 9. **DistillationScheduler** (renamed `KdDistilScheduler`) — adaptive temp/alpha scheduling.
//! 10. **EfficientTransferLearning** — discriminative fine-tuning with gradual unfreeze.

pub mod extensions;
pub use extensions::*;

pub mod advanced;
pub use advanced::*;

#[cfg(test)]
mod tests;

use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// §0  Shared utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Compute softmax over a slice with a given temperature.
pub(crate) fn softmax_temp(logits: &[f32], temp: f32) -> Vec<f32> {
    let t = temp.max(1e-8);
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| ((x - max_l) / t).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum < 1e-30 {
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exps.iter().map(|e| e / sum).collect()
    }
}

/// KL divergence KL(p || q) = sum p * log(p/q).
pub(crate) fn kl_div(p: &[f32], q: &[f32]) -> f32 {
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi < 1e-30 {
                0.0
            } else {
                pi * ((pi / (qi + 1e-30)).ln())
            }
        })
        .sum()
}

/// Mean-squared error between two equal-length slices.
pub(crate) fn mse_slice(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    sum / n as f32
}

/// L2-normalise a vector in-place; returns the norm.
pub(crate) fn l2_normalize_mut(v: &mut [f32]) -> f32 {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    norm
}

/// Dot product of two slices (same length).
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  Online Distillation
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for online (mutual) distillation among `n_students` models.
///
/// No teacher is fixed; the ensemble of all students acts as the teacher
/// signal for each individual student.
#[derive(Debug, Clone)]
pub struct OnlineDistilConfig {
    /// Number of student models sharing knowledge.
    pub n_students: usize,
    /// Softmax temperature applied to both ensemble and student logits.
    pub temp: f32,
    /// Weighting of soft (ensemble-distilled) loss vs. hard-label CE.
    /// Must lie in `[0.0, 1.0]`.
    pub alpha: f32,
}

impl OnlineDistilConfig {
    /// Create a validated config.
    ///
    /// # Errors
    /// Returns error if `n_students == 0`, `temp <= 0`, or `alpha ∉ [0,1]`.
    pub fn new(n_students: usize, temp: f32, alpha: f32) -> Result<Self> {
        if n_students == 0 {
            return Err(TensorError::invalid_argument(
                "n_students must be at least 1".to_string(),
            ));
        }
        if temp <= 0.0 {
            return Err(TensorError::invalid_argument(format!(
                "temp must be positive, got {temp}"
            )));
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(TensorError::invalid_argument(format!(
                "alpha must be in [0,1], got {alpha}"
            )));
        }
        Ok(Self {
            n_students,
            temp,
            alpha,
        })
    }
}

impl Default for OnlineDistilConfig {
    fn default() -> Self {
        Self {
            n_students: 3,
            temp: 3.0,
            alpha: 0.5,
        }
    }
}

/// Online distillation engine.
#[derive(Debug, Clone)]
pub struct OnlineDistillation {
    /// Validated config.
    pub config: OnlineDistilConfig,
}

impl OnlineDistillation {
    /// Construct from a validated config.
    pub fn new(config: OnlineDistilConfig) -> Self {
        Self { config }
    }

    /// Compute the ensemble logits as the element-wise average of all students'
    /// logit vectors.
    ///
    /// # Errors
    /// Returns error if `all_student_logits` is empty or vectors have
    /// inconsistent lengths.
    pub fn compute_ensemble_logits(&self, all_student_logits: &[Vec<f32>]) -> Result<Vec<f32>> {
        if all_student_logits.is_empty() {
            return Err(TensorError::invalid_argument(
                "all_student_logits must not be empty".to_string(),
            ));
        }
        let n_cls = all_student_logits[0].len();
        for (i, v) in all_student_logits.iter().enumerate() {
            if v.len() != n_cls {
                return Err(TensorError::invalid_argument(format!(
                    "student {i} has {len} logits but student 0 has {n_cls}",
                    len = v.len()
                )));
            }
        }
        let n = all_student_logits.len() as f32;
        let mut avg = vec![0.0_f32; n_cls];
        for sv in all_student_logits {
            for (a, &x) in avg.iter_mut().zip(sv.iter()) {
                *a += x;
            }
        }
        avg.iter_mut().for_each(|x| *x /= n);
        Ok(avg)
    }

    /// Compute the mutual-distillation loss for one student.
    pub fn mutual_loss(
        &self,
        student_logits: &[f32],
        ensemble_logits: &[f32],
        hard_labels: &[f32],
        temp: f32,
        alpha: f32,
    ) -> f32 {
        let soft_student = softmax_temp(student_logits, temp);
        let soft_ensemble = softmax_temp(ensemble_logits, temp);

        let kl = kl_div(&soft_ensemble, &soft_student);
        let soft_loss = alpha * temp * temp * kl;

        let log_probs: Vec<f32> = soft_student.iter().map(|p| (p + 1e-30).ln()).collect();
        let ce: f32 = hard_labels
            .iter()
            .zip(log_probs.iter())
            .map(|(y, lp)| -y * lp)
            .sum();

        soft_loss + (1.0 - alpha) * ce
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  Self-Distillation
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for intra-model self-distillation.
#[derive(Debug, Clone)]
pub struct SelfDistilConfig {
    /// Total number of hidden layers in the model.
    pub n_layers: usize,
    /// Default distillation temperature.
    pub temp: f32,
}

impl SelfDistilConfig {
    /// Create config with validation.
    pub fn new(n_layers: usize, temp: f32) -> Result<Self> {
        if n_layers < 2 {
            return Err(TensorError::invalid_argument(
                "n_layers must be at least 2 for self-distillation".to_string(),
            ));
        }
        if temp <= 0.0 {
            return Err(TensorError::invalid_argument(format!(
                "temp must be positive, got {temp}"
            )));
        }
        Ok(Self { n_layers, temp })
    }
}

impl Default for SelfDistilConfig {
    fn default() -> Self {
        Self {
            n_layers: 6,
            temp: 4.0,
        }
    }
}

/// Self-distillation: later (deeper) layers teach earlier (shallower) layers.
#[derive(Debug, Clone)]
pub struct SelfDistillation {
    /// Configuration.
    pub config: SelfDistilConfig,
}

impl SelfDistillation {
    /// Construct a new self-distillation engine.
    pub fn new(config: SelfDistilConfig) -> Self {
        Self { config }
    }

    /// Intra-layer KL from early to late hidden via a projection matrix.
    ///
    /// # Errors
    /// Returns error on dimension mismatch.
    pub fn intra_layer_kl(
        &self,
        early_hidden: &[f32],
        late_hidden: &[f32],
        projection: &[f32],
    ) -> Result<f32> {
        let early_dim = early_hidden.len();
        let late_dim = late_hidden.len();
        if projection.len() != late_dim * early_dim {
            return Err(TensorError::invalid_argument(format!(
                "projection size {} != late_dim({late_dim}) * early_dim({early_dim})",
                projection.len()
            )));
        }
        let mut projected = vec![0.0_f32; late_dim];
        for i in 0..late_dim {
            let row_start = i * early_dim;
            projected[i] = early_hidden
                .iter()
                .enumerate()
                .map(|(j, &x)| projection[row_start + j] * x)
                .sum();
        }
        let p = softmax_temp(&projected, self.config.temp);
        let q = softmax_temp(late_hidden, self.config.temp);
        Ok(kl_div(&p, &q))
    }

    /// Born-Again Networks loss.
    pub fn born_again_loss(
        &self,
        pred: &[f32],
        teacher_pred: &[f32],
        target: &[usize],
        temp: f32,
    ) -> f32 {
        let batch = target.len();
        if batch == 0 || pred.is_empty() {
            return 0.0;
        }
        let n_cls = pred.len() / batch.max(1);
        if n_cls == 0 {
            return 0.0;
        }

        let mut total = 0.0_f32;
        for b in 0..batch {
            let s_logits = &pred[b * n_cls..(b + 1) * n_cls];
            let t_logits = &teacher_pred[b * n_cls..(b + 1) * n_cls];

            let soft_s = softmax_temp(s_logits, temp);
            let soft_t = softmax_temp(t_logits, temp);

            let kl = kl_div(&soft_t, &soft_s);

            let cls = target[b].min(n_cls - 1);
            let ce = -(soft_s[cls] + 1e-30_f32).ln();

            total += temp * temp * kl + ce;
        }
        total / batch as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  Data-Free Distillation
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the data generator used in data-free distillation.
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Dimensionality of noise input (z-space).
    pub noise_dim: usize,
    /// Hidden layer width.
    pub hidden_dim: usize,
    /// Output dimension (must match teacher/student input dimensionality).
    pub output_dim: usize,
}

impl GeneratorConfig {
    /// Create a validated config.
    pub fn new(noise_dim: usize, hidden_dim: usize, output_dim: usize) -> Result<Self> {
        if noise_dim == 0 || hidden_dim == 0 || output_dim == 0 {
            return Err(TensorError::invalid_argument(
                "all generator dimensions must be positive".to_string(),
            ));
        }
        Ok(Self {
            noise_dim,
            hidden_dim,
            output_dim,
        })
    }
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            noise_dim: 100,
            hidden_dim: 256,
            output_dim: 784,
        }
    }
}

/// Lightweight learnable generator: noise_dim → hidden_dim → output_dim (ReLU + linear).
#[derive(Debug, Clone)]
pub struct DataGenerator {
    /// Each tuple is `(weight_matrix, bias)` for one linear layer.
    pub layers: Vec<(Vec<f32>, Vec<f32>)>,
    /// Generator configuration.
    pub config: GeneratorConfig,
}

impl DataGenerator {
    /// Create a new generator with Xavier-style initialised weights.
    pub fn new(config: GeneratorConfig) -> Self {
        let dims = [config.noise_dim, config.hidden_dim, config.output_dim];
        let mut layers = Vec::with_capacity(dims.len() - 1);

        let mut seed: u64 = 42;
        let lcg = |s: &mut u64| -> f32 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = ((*s >> 33) as u32) as f32;
            (bits / u32::MAX as f32) * 2.0 - 1.0
        };

        for win in dims.windows(2) {
            let (inp, out) = (win[0], win[1]);
            let gain = (2.0 / (inp + out) as f32).sqrt();
            let w: Vec<f32> = (0..out * inp).map(|_| lcg(&mut seed) * gain).collect();
            let b: Vec<f32> = vec![0.0; out];
            layers.push((w, b));
        }

        Self { layers, config }
    }

    /// Forward pass through the generator.
    ///
    /// # Errors
    /// Returns error if `z.len() != config.noise_dim`.
    pub fn generate(&self, z: &[f32]) -> Result<Vec<f32>> {
        if z.len() != self.config.noise_dim {
            return Err(TensorError::invalid_argument(format!(
                "z has length {} but noise_dim is {}",
                z.len(),
                self.config.noise_dim
            )));
        }

        let mut current: Vec<f32> = z.to_vec();

        for (layer_idx, (w, b)) in self.layers.iter().enumerate() {
            let inp = current.len();
            let out = b.len();
            let mut next = vec![0.0_f32; out];
            for i in 0..out {
                let row = i * inp;
                let val: f32 = current
                    .iter()
                    .enumerate()
                    .map(|(j, &x)| w[row + j] * x)
                    .sum::<f32>()
                    + b[i];
                next[i] = if layer_idx < self.layers.len() - 1 {
                    val.max(0.0)
                } else {
                    val
                };
            }
            current = next;
        }
        Ok(current)
    }
}

/// Loss for data-free dreaming distillation.
#[derive(Debug, Clone)]
pub struct DreemLoss {
    /// Weight for the diversity / entropy-maximisation term.
    pub diversity_weight: f32,
}

impl DreemLoss {
    /// Create with given diversity weight.
    pub fn new(diversity_weight: f32) -> Self {
        Self { diversity_weight }
    }

    /// Compute the dreaming loss.
    pub fn dreaming_loss(
        &self,
        generated: &[f32],
        teacher_logits: &[f32],
        student_logits: &[f32],
    ) -> f32 {
        let _ = generated;

        let soft_t = softmax_temp(teacher_logits, 1.0);
        let soft_s = softmax_temp(student_logits, 1.0);

        let kl = kl_div(&soft_t, &soft_s);

        let entropy: f32 = soft_t
            .iter()
            .map(|&p| if p < 1e-30 { 0.0 } else { -p * p.ln() })
            .sum();

        kl - self.diversity_weight * entropy
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  Task-Agnostic Distillation
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for task-agnostic distillation on an unlabeled corpus.
#[derive(Debug, Clone)]
pub struct UnlabeledDistilConfig {
    /// Number of unlabeled samples per distillation step.
    pub batch_size: usize,
    /// Temperature for teacher soft-target computation.
    pub temp: f32,
    /// Keep only the top-k tokens/classes in soft targets to reduce noise.
    pub top_k_tokens: usize,
}

impl UnlabeledDistilConfig {
    /// Create a validated config.
    pub fn new(batch_size: usize, temp: f32, top_k_tokens: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(TensorError::invalid_argument(
                "batch_size must be positive".to_string(),
            ));
        }
        if temp <= 0.0 {
            return Err(TensorError::invalid_argument(format!(
                "temp must be positive, got {temp}"
            )));
        }
        if top_k_tokens == 0 {
            return Err(TensorError::invalid_argument(
                "top_k_tokens must be positive".to_string(),
            ));
        }
        Ok(Self {
            batch_size,
            temp,
            top_k_tokens,
        })
    }
}

impl Default for UnlabeledDistilConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            temp: 4.0,
            top_k_tokens: 10,
        }
    }
}

/// Task-agnostic distillation engine.
#[derive(Debug, Clone)]
pub struct TaskAgnosticDistillation {
    /// Validated configuration.
    pub config: UnlabeledDistilConfig,
}

impl TaskAgnosticDistillation {
    /// Construct from validated config.
    pub fn new(config: UnlabeledDistilConfig) -> Self {
        Self { config }
    }

    /// Compute soft targets from teacher logits for a batch of unlabeled samples.
    pub fn soft_targets_from_teacher(
        &self,
        teacher_logits: &[Vec<f32>],
        temp: f32,
    ) -> Vec<Vec<f32>> {
        teacher_logits
            .iter()
            .map(|logits| softmax_temp(logits, temp))
            .collect()
    }

    /// Return the top-k `(class_index, probability)` pairs from a soft-target distribution.
    pub fn top_k_soft_targets(&self, soft: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = soft.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }

    /// Cross-entropy distillation loss against sparse top-k soft targets.
    pub fn kd_loss_unlabeled(&self, student_logits: &[f32], soft_targets: &[(usize, f32)]) -> f32 {
        if student_logits.is_empty() || soft_targets.is_empty() {
            return 0.0;
        }
        let probs = softmax_temp(student_logits, 1.0);
        let n_cls = probs.len();
        let mut loss = 0.0_f32;
        for &(idx, p) in soft_targets {
            if idx < n_cls {
                loss -= p * (probs[idx] + 1e-30).ln();
            }
        }
        loss
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  Patch Distillation
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for vision patch-level distillation.
#[derive(Debug, Clone)]
pub struct PatchDistilConfig {
    /// Number of image patches (e.g. 196 for 14×14 grid on 224-px input).
    pub n_patches: usize,
    /// Dimension of each patch representation (pre-projection).
    pub patch_dim: usize,
    /// Projection dimension used when comparing student and teacher patches.
    pub proj_dim: usize,
}

impl PatchDistilConfig {
    /// Create a validated config.
    pub fn new(n_patches: usize, patch_dim: usize, proj_dim: usize) -> Result<Self> {
        if n_patches == 0 || patch_dim == 0 || proj_dim == 0 {
            return Err(TensorError::invalid_argument(
                "all patch config dimensions must be positive".to_string(),
            ));
        }
        Ok(Self {
            n_patches,
            patch_dim,
            proj_dim,
        })
    }
}

impl Default for PatchDistilConfig {
    fn default() -> Self {
        Self {
            n_patches: 196,
            patch_dim: 768,
            proj_dim: 256,
        }
    }
}

/// Patch-level distillation engine.
#[derive(Debug, Clone)]
pub struct PatchDistillation {
    /// Configuration.
    pub config: PatchDistilConfig,
}

impl PatchDistillation {
    /// Construct from validated config.
    pub fn new(config: PatchDistilConfig) -> Self {
        Self { config }
    }

    /// Compute the patch similarity distillation loss.
    ///
    /// # Errors
    /// Returns error if the patch counts or dimensions are inconsistent.
    pub fn patch_similarity_loss(
        &self,
        student_patches: &[Vec<f32>],
        teacher_patches: &[Vec<f32>],
    ) -> Result<f32> {
        if student_patches.len() != teacher_patches.len() {
            return Err(TensorError::invalid_argument(format!(
                "student has {} patches but teacher has {}",
                student_patches.len(),
                teacher_patches.len()
            )));
        }
        if student_patches.is_empty() {
            return Ok(0.0);
        }

        let n = student_patches.len();
        let anchor_s = &student_patches[0];
        let anchor_t = &teacher_patches[0];

        let mut sims_s = vec![0.0_f32; n];
        let mut sims_t = vec![0.0_f32; n];

        let norm_as = anchor_s
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(1e-10);
        let norm_at = anchor_t
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(1e-10);

        for i in 0..n {
            let ps = &student_patches[i];
            let pt = &teacher_patches[i];
            let norm_ps = ps.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            let norm_pt = pt.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);

            let d_s = dot(anchor_s, ps) / (norm_as * norm_ps);
            let d_t = dot(anchor_t, pt) / (norm_at * norm_pt);
            sims_s[i] = d_s;
            sims_t[i] = d_t;
        }

        Ok(mse_slice(&sims_s, &sims_t))
    }

    /// Masked patch distillation.
    ///
    /// # Errors
    /// Returns error if slice lengths are inconsistent.
    pub fn masked_patch_distil(
        &self,
        student_patches: &[Vec<f32>],
        teacher_patches: &[Vec<f32>],
        mask: &[bool],
    ) -> Result<f32> {
        if student_patches.len() != teacher_patches.len() || student_patches.len() != mask.len() {
            return Err(TensorError::invalid_argument(
                "student_patches, teacher_patches, mask must have the same length".to_string(),
            ));
        }

        let masked_s: Vec<Vec<f32>> = student_patches
            .iter()
            .zip(mask.iter())
            .filter_map(|(p, &m)| if m { Some(p.clone()) } else { None })
            .collect();
        let masked_t: Vec<Vec<f32>> = teacher_patches
            .iter()
            .zip(mask.iter())
            .filter_map(|(p, &m)| if m { Some(p.clone()) } else { None })
            .collect();

        if masked_s.is_empty() {
            return Ok(0.0);
        }

        let mut total = 0.0_f32;
        let mut count = 0usize;
        for (sp, tp) in masked_s.iter().zip(masked_t.iter()) {
            let dim = sp.len().min(tp.len());
            for d in 0..dim {
                total += (sp[d] - tp[d]).powi(2);
                count += 1;
            }
        }
        if count == 0 {
            Ok(0.0)
        } else {
            Ok(total / count as f32)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  Graph (Relational) Distillation
// ─────────────────────────────────────────────────────────────────────────────

/// Symmetric pairwise relation matrix over `n_samples` samples.
#[derive(Debug, Clone)]
pub struct RelationMatrix {
    /// Number of samples.
    pub n_samples: usize,
    /// Flat pairwise similarity values (length `n_samples²`).
    pub similarities: Vec<f32>,
}

impl RelationMatrix {
    /// Access similarity `(i, j)`.  Returns 0 if out-of-bounds.
    pub fn get(&self, i: usize, j: usize) -> f32 {
        let idx = i * self.n_samples + j;
        self.similarities.get(idx).copied().unwrap_or(0.0)
    }
}

/// Compute a pairwise L2-normalised cosine similarity matrix.
///
/// # Errors
/// Returns error if `embeddings` is empty or inner dimension is 0.
pub fn compute_relation_matrix(embeddings: &[Vec<f32>]) -> Result<RelationMatrix> {
    if embeddings.is_empty() {
        return Err(TensorError::invalid_argument(
            "embeddings must not be empty".to_string(),
        ));
    }
    let dim = embeddings[0].len();
    if dim == 0 {
        return Err(TensorError::invalid_argument(
            "embedding dimension must be positive".to_string(),
        ));
    }
    let n = embeddings.len();
    let mut normed: Vec<Vec<f32>> = embeddings.to_vec();
    for v in normed.iter_mut() {
        l2_normalize_mut(v);
    }

    let mut sims = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            sims[i * n + j] = dot(&normed[i], &normed[j]).clamp(-1.0, 1.0);
        }
    }
    Ok(RelationMatrix {
        n_samples: n,
        similarities: sims,
    })
}

/// Graph distillation engine.
#[derive(Debug, Clone, Default)]
pub struct GraphDistillation;

impl GraphDistillation {
    /// Compute the RKD angle loss.
    ///
    /// # Errors
    /// Returns error if the two matrices have different sizes.
    pub fn rku_loss(
        &self,
        student_rel: &RelationMatrix,
        teacher_rel: &RelationMatrix,
    ) -> Result<f32> {
        if student_rel.n_samples != teacher_rel.n_samples {
            return Err(TensorError::invalid_argument(format!(
                "student and teacher relation matrices have different sizes: {} vs {}",
                student_rel.n_samples, teacher_rel.n_samples
            )));
        }
        let n = student_rel.n_samples;
        if n < 3 {
            return Ok(0.0);
        }

        let mut total = 0.0_f32;
        let mut count = 0u64;

        for j in 0..n {
            for i in 0..n {
                if i == j {
                    continue;
                }
                for k in 0..n {
                    if k == j || k == i {
                        continue;
                    }
                    let t_angle =
                        teacher_rel.get(i, k) - teacher_rel.get(i, j) * teacher_rel.get(j, k);
                    let s_angle =
                        student_rel.get(i, k) - student_rel.get(i, j) * student_rel.get(j, k);
                    total += (t_angle - s_angle).powi(2);
                    count += 1;
                }
            }
        }

        if count == 0 {
            Ok(0.0)
        } else {
            Ok(total / count as f32)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  Progressive Distillation
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for step-by-step model shrinkage.
#[derive(Debug, Clone)]
pub struct ProgressiveConfig {
    /// Total number of compression stages.
    pub n_stages: usize,
    /// Fraction by which the hidden dimension shrinks at each stage.
    pub shrink_ratio: f32,
}

impl ProgressiveConfig {
    /// Create a validated config.
    pub fn new(n_stages: usize, shrink_ratio: f32) -> Result<Self> {
        if n_stages == 0 {
            return Err(TensorError::invalid_argument(
                "n_stages must be at least 1".to_string(),
            ));
        }
        if shrink_ratio <= 0.0 || shrink_ratio >= 1.0 {
            return Err(TensorError::invalid_argument(format!(
                "shrink_ratio must be in (0, 1), got {shrink_ratio}"
            )));
        }
        Ok(Self {
            n_stages,
            shrink_ratio,
        })
    }
}

impl Default for ProgressiveConfig {
    fn default() -> Self {
        Self {
            n_stages: 3,
            shrink_ratio: 0.5,
        }
    }
}

/// A model in a particular compression stage.
#[derive(Debug, Clone)]
pub struct StagedModel {
    /// Current stage index (0 = original).
    pub stage: usize,
    /// Hidden dimensions at the current stage.
    pub hidden_dims: Vec<usize>,
}

impl StagedModel {
    /// Create a new staged model with initial hidden dims.
    pub fn new(hidden_dims: Vec<usize>) -> Self {
        Self {
            stage: 0,
            hidden_dims,
        }
    }

    /// Advance to the next stage, shrinking each hidden dim by `shrink_ratio`.
    pub fn advance_stage(&mut self, shrink_ratio: f32) {
        self.stage += 1;
        self.hidden_dims = self
            .hidden_dims
            .iter()
            .map(|&d| ((d as f32 * shrink_ratio) as usize).max(1))
            .collect();
    }
}

/// Progressive distillation engine.
#[derive(Debug, Clone)]
pub struct ProgressiveDistillation {
    /// Configuration.
    pub config: ProgressiveConfig,
}

impl ProgressiveDistillation {
    /// Construct from validated config.
    pub fn new(config: ProgressiveConfig) -> Self {
        Self { config }
    }

    /// MSE loss between a student hidden state and a teacher hidden state.
    pub fn stage_loss(&self, student_hidden: &[f32], teacher_hidden: &[f32]) -> f32 {
        mse_slice(student_hidden, teacher_hidden)
    }

    /// Pool a teacher hidden vector to half its size by averaging adjacent pairs.
    pub fn distill_stage(&self, _stage: usize, teacher_hidden: &[f32]) -> Vec<f32> {
        let n = teacher_hidden.len();
        if n == 0 {
            return Vec::new();
        }
        let out_len = (n + 1) / 2;
        let mut out = vec![0.0_f32; out_len];
        for (i, chunk) in teacher_hidden.chunks(2).enumerate() {
            out[i] = chunk.iter().sum::<f32>() / chunk.len() as f32;
        }
        out
    }
}
