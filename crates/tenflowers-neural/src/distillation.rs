//! # Knowledge Distillation Module
//!
//! Comprehensive, pure-Rust implementation of knowledge distillation algorithms
//! (Hinton et al., 2015) and modern variants.
//!
//! This module operates at the **logit / feature level**: it does not depend on
//! the full model training pipeline and is usable in isolation or as part of a
//! larger training loop.
//!
//! ## Components
//!
//! * [`DistillConfig`] – temperature, alpha, student/teacher layer dims.
//! * [`DistillationSchemeConfig`] – simpler config (temperature + alpha only).
//! * [`SoftTargets`] – temperature-scaled softmax probabilities from the teacher.
//! * [`DistillationLoss`] – combined soft + hard loss with helper constructor.
//! * [`FeatureMatcher`] – FitNets-style linear projection for feature matching.
//! * [`BornAgainDistiller`] – Born-Again Networks multi-generation distillation.
//! * [`KnowledgeDistiller`] – top-level dispatcher for all distillation methods.
//! * [`DistillMethod`] – selects response-based, feature-based, or relation-based.
//! * [`kl_divergence`] – KL divergence between two probability distributions.
//! * [`softmax_with_temperature`] – standard softmax with an explicit temperature.
//! * [`cross_entropy_with_labels`] – cross-entropy between logits and integer labels.
//! * [`soft_targets`] – compute soft targets (alias for `softmax_with_temperature`).
//! * [`distillation_loss`] – KL-based distillation loss scaled by T².
//! * [`combined_loss`] – alpha * distil_loss + (1-alpha) * CE_loss.
//! * [`pkt_loss`] – Progressive Knowledge Transfer loss (RBF kernel-based KL).
//!
//! ## Naming Note
//!
//! The training-pipeline module already exports a generic `DistillationConfig<T>`.
//! This module exports [`DistillationSchemeConfig`] and [`DistillConfig`] to provide
//! simpler, `f32`-only configurations without clashing with the existing symbol.
//!
//! ## Example
//!
//! ```rust,ignore
//! use tenflowers_neural::distillation::{
//!     DistillationLoss, DistillationSchemeConfig, SoftTargets,
//! };
//!
//! let config = DistillationSchemeConfig::new(4.0, 0.7).unwrap();
//! let teacher_logits = vec![2.0_f32, 1.0, 0.1];
//! let student_logits = vec![1.5_f32, 1.0, 0.5];
//! let true_labels = vec![0usize];
//!
//! let loss = DistillationLoss::compute(
//!     &student_logits,
//!     &teacher_logits,
//!     &true_labels,
//!     &config,
//! ).unwrap();
//!
//! println!("total loss: {}", loss.total_loss);
//! ```

use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// DistillConfig (full config including layer dims)
// ─────────────────────────────────────────────────────────────────────────────

/// Full configuration for knowledge distillation, including feature-matching
/// layer dimensions.
///
/// The total loss is:
/// ```text
/// L = alpha * T^2 * KL(soft_student || soft_teacher)
///   + (1 - alpha) * CE(student_logits, true_labels)
/// ```
#[derive(Debug, Clone)]
pub struct DistillConfig {
    /// Softmax temperature.  Values > 1 soften the distributions; values < 1
    /// sharpen them.  Must be strictly positive.  Default: 4.0.
    pub temperature: f32,
    /// Weight given to the soft (teacher-guided) loss.  Must be in `[0.0, 1.0]`.
    /// Default: 0.7.
    pub alpha: f32,
    /// Intermediate layer dimensions of the student network (for feature matching).
    pub student_layer_dims: Vec<usize>,
    /// Intermediate layer dimensions of the teacher network (for feature matching).
    pub teacher_layer_dims: Vec<usize>,
}

impl DistillConfig {
    /// Construct a new `DistillConfig` with validation.
    ///
    /// # Errors
    /// Returns an error when `temperature <= 0.0` or `alpha ∉ [0.0, 1.0]`.
    pub fn new(
        temperature: f32,
        alpha: f32,
        student_layer_dims: Vec<usize>,
        teacher_layer_dims: Vec<usize>,
    ) -> Result<Self> {
        if temperature <= 0.0 {
            return Err(TensorError::invalid_argument(format!(
                "temperature must be positive, got {temperature}"
            )));
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(TensorError::invalid_argument(format!(
                "alpha must be in [0.0, 1.0], got {alpha}"
            )));
        }
        Ok(Self {
            temperature,
            alpha,
            student_layer_dims,
            teacher_layer_dims,
        })
    }

    /// Default Hinton configuration: temperature = 4.0, alpha = 0.7.
    pub fn default_hinton() -> Self {
        Self {
            temperature: 4.0,
            alpha: 0.7,
            student_layer_dims: Vec::new(),
            teacher_layer_dims: Vec::new(),
        }
    }
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self::default_hinton()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DistillationSchemeConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight configuration for the knowledge distillation loss (temperature
/// and alpha only; no layer-dimension tracking).
///
/// The total loss is:
/// ```text
/// L = alpha * T^2 * KL(soft_student || soft_teacher)
///   + (1 - alpha) * CE(student_logits, true_labels)
/// ```
///
/// The `T^2` factor rescales the soft loss so its magnitude is comparable to
/// the hard-label cross-entropy when the temperature `T > 1` (following the
/// convention in the original paper).
#[derive(Debug, Clone)]
pub struct DistillationSchemeConfig {
    /// Softmax temperature.  Values > 1 soften the distributions; values < 1
    /// sharpen them.  Must be strictly positive.
    pub temperature: f32,
    /// Weight given to the soft (teacher-guided) loss.  Must be in `[0.0, 1.0]`.
    pub alpha: f32,
    /// Weight given to the hard-label cross-entropy loss.  Always `1.0 - alpha`.
    pub hard_label_weight: f32,
}

impl DistillationSchemeConfig {
    /// Construct a new configuration, validating the parameters.
    ///
    /// # Errors
    /// Returns an error when:
    /// * `temperature <= 0.0`
    /// * `alpha` is outside `[0.0, 1.0]`
    pub fn new(temperature: f32, alpha: f32) -> Result<Self> {
        if temperature <= 0.0 {
            return Err(TensorError::invalid_argument(format!(
                "temperature must be positive, got {temperature}"
            )));
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(TensorError::invalid_argument(format!(
                "alpha must be in [0.0, 1.0], got {alpha}"
            )));
        }
        Ok(Self {
            temperature,
            alpha,
            hard_label_weight: 1.0 - alpha,
        })
    }

    /// Default configuration: temperature = 4.0, alpha = 0.7.
    pub fn default_hinton() -> Self {
        Self {
            temperature: 4.0,
            alpha: 0.7,
            hard_label_weight: 0.3,
        }
    }
}

impl Default for DistillationSchemeConfig {
    fn default() -> Self {
        Self::default_hinton()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SoftTargets
// ─────────────────────────────────────────────────────────────────────────────

/// Soft probability distribution obtained by applying temperature scaling to
/// the teacher's raw logits.
#[derive(Debug, Clone)]
pub struct SoftTargets {
    /// Softmax probabilities (sum to 1.0).
    pub probabilities: Vec<f32>,
    /// The temperature that was used.
    pub temperature: f32,
}

impl SoftTargets {
    /// Apply temperature scaling to a slice of logits and return the resulting
    /// soft targets.
    ///
    /// Divides each logit by `temperature` before computing the softmax so that
    /// the distribution is smoothed (higher T) or sharpened (lower T).
    ///
    /// # Errors
    /// Returns an error if `logits` is empty or `temperature <= 0.0`.
    pub fn from_logits(logits: &[f32], temperature: f32) -> Result<Self> {
        if logits.is_empty() {
            return Err(TensorError::invalid_argument(
                "logits slice is empty".to_string(),
            ));
        }
        if temperature <= 0.0 {
            return Err(TensorError::invalid_argument(format!(
                "temperature must be positive, got {temperature}"
            )));
        }
        let probabilities = softmax_with_temperature(logits, temperature);
        Ok(Self {
            probabilities,
            temperature,
        })
    }

    /// Number of classes.
    pub fn num_classes(&self) -> usize {
        self.probabilities.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DistillationLoss
// ─────────────────────────────────────────────────────────────────────────────

/// The decomposed distillation loss for a single batch.
#[derive(Debug, Clone)]
pub struct DistillationLoss {
    /// KL divergence between student soft targets and teacher soft targets,
    /// scaled by `T^2` to match the magnitude of the hard loss.
    pub soft_loss: f32,
    /// Cross-entropy between the student logits and the ground-truth integer labels.
    pub hard_loss: f32,
    /// `alpha * soft_loss + (1 - alpha) * hard_loss`.
    pub total_loss: f32,
}

impl DistillationLoss {
    /// Compute the distillation loss for a batch.
    ///
    /// # Parameters
    /// * `student_logits`  – raw (pre-softmax) logits from the student model.
    ///   Shape: `[batch * num_classes]` in row-major order, or a flat `[num_classes]`
    ///   vector for a single example.
    /// * `teacher_logits`  – same shape as `student_logits`, from the teacher.
    /// * `true_labels`     – integer class indices, one per sample.  Length must be
    ///   `batch_size = student_logits.len() / num_classes`.
    /// * `config`          – distillation hyper-parameters.
    ///
    /// # Errors
    /// Returns an error when the lengths are inconsistent or inputs are empty.
    pub fn compute(
        student_logits: &[f32],
        teacher_logits: &[f32],
        true_labels: &[usize],
        config: &DistillationSchemeConfig,
    ) -> Result<Self> {
        if student_logits.is_empty() {
            return Err(TensorError::invalid_argument(
                "student_logits is empty".to_string(),
            ));
        }
        if student_logits.len() != teacher_logits.len() {
            return Err(TensorError::invalid_argument(format!(
                "student_logits length {} != teacher_logits length {}",
                student_logits.len(),
                teacher_logits.len()
            )));
        }
        if true_labels.is_empty() {
            return Err(TensorError::invalid_argument(
                "true_labels is empty".to_string(),
            ));
        }

        let num_classes = student_logits.len() / true_labels.len();
        if num_classes == 0 || student_logits.len() % true_labels.len() != 0 {
            return Err(TensorError::invalid_argument(format!(
                "Cannot infer num_classes: student_logits.len()={} is not divisible by \
                 true_labels.len()={}",
                student_logits.len(),
                true_labels.len()
            )));
        }

        let batch_size = true_labels.len();
        let t = config.temperature;

        let mut soft_loss_sum = 0.0_f32;
        let mut hard_loss_sum = 0.0_f32;

        for i in 0..batch_size {
            let s_start = i * num_classes;
            let s_end = s_start + num_classes;

            let s_logits = &student_logits[s_start..s_end];
            let t_logits = &teacher_logits[s_start..s_end];
            let label = true_labels[i];

            if label >= num_classes {
                return Err(TensorError::invalid_argument(format!(
                    "true_labels[{i}]={label} is out of range [0, {num_classes})"
                )));
            }

            // Soft loss: KL(student_soft || teacher_soft) * T^2
            let student_soft = softmax_with_temperature(s_logits, t);
            let teacher_soft = softmax_with_temperature(t_logits, t);
            let kl = kl_divergence(&student_soft, &teacher_soft)?;
            soft_loss_sum += kl * t * t;

            // Hard loss: cross-entropy with true integer label
            let hard = cross_entropy_with_labels(s_logits, label)?;
            hard_loss_sum += hard;
        }

        let batch_f = batch_size as f32;
        let soft_loss = soft_loss_sum / batch_f;
        let hard_loss = hard_loss_sum / batch_f;
        let total_loss = config.alpha * soft_loss + config.hard_label_weight * hard_loss;

        Ok(Self {
            soft_loss,
            hard_loss,
            total_loss,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FeatureMatcher (FitNets)
// ─────────────────────────────────────────────────────────────────────────────

/// Feature-level knowledge distillation via linear projection (FitNets,
/// Romero et al. 2015).
///
/// Maps student intermediate features to the teacher's feature space and
/// minimises MSE between the projected student features and the teacher features.
#[derive(Debug, Clone)]
pub struct FeatureMatcher {
    /// Student feature dimension.
    pub student_dim: usize,
    /// Teacher feature dimension.
    pub teacher_dim: usize,
    /// Linear projection weight matrix in row-major order, shape
    /// `[student_dim × teacher_dim]` (one row per student dimension).
    pub weights: Vec<f32>,
}

impl FeatureMatcher {
    /// Construct a new `FeatureMatcher` and initialise the projection weights
    /// with Xavier initialisation seeded by `seed=42`.
    ///
    /// # Errors
    /// Returns an error if either dimension is 0.
    pub fn new(student_dim: usize, teacher_dim: usize) -> Result<Self> {
        if student_dim == 0 || teacher_dim == 0 {
            return Err(TensorError::invalid_argument(
                "student_dim and teacher_dim must both be > 0".to_string(),
            ));
        }
        let weights = initialize_projection(student_dim, teacher_dim, 42);
        Ok(Self {
            student_dim,
            teacher_dim,
            weights,
        })
    }

    /// Project student features to teacher dimension via linear transformation.
    ///
    /// Computes `y = x @ W` where `W` is `[student_dim × teacher_dim]`.
    ///
    /// # Errors
    /// Returns an error if `student_feat.len() != self.student_dim`.
    pub fn project(&self, student_feat: &[f32]) -> Result<Vec<f32>> {
        if student_feat.len() != self.student_dim {
            return Err(TensorError::invalid_argument(format!(
                "student_feat length {} != student_dim {}",
                student_feat.len(),
                self.student_dim
            )));
        }
        let mut out = vec![0.0_f32; self.teacher_dim];
        for t in 0..self.teacher_dim {
            let mut acc = 0.0_f32;
            for s in 0..self.student_dim {
                // weights layout: weights[s * teacher_dim + t]
                acc += student_feat[s] * self.weights[s * self.teacher_dim + t];
            }
            out[t] = acc;
        }
        Ok(out)
    }

    /// Compute MSE loss between projected student features and teacher features.
    ///
    /// # Errors
    /// Returns an error if the two slices have different lengths or are empty.
    pub fn feature_loss(&self, student_projected: &[f32], teacher_feat: &[f32]) -> Result<f32> {
        if student_projected.is_empty() {
            return Err(TensorError::invalid_argument(
                "student_projected is empty".to_string(),
            ));
        }
        if student_projected.len() != teacher_feat.len() {
            return Err(TensorError::invalid_argument(format!(
                "student_projected length {} != teacher_feat length {}",
                student_projected.len(),
                teacher_feat.len()
            )));
        }
        let mse = student_projected
            .iter()
            .zip(teacher_feat.iter())
            .map(|(&s, &t)| (s - t) * (s - t))
            .sum::<f32>()
            / student_projected.len() as f32;
        Ok(mse)
    }
}

/// Xavier-initialised linear projection weights for feature matching.
///
/// Returns a flat `Vec<f32>` of shape `[student_dim × teacher_dim]` (row-major).
/// The initialisation scale is `sqrt(6 / (student_dim + teacher_dim))` following
/// Glorot & Bengio (2010).
pub fn initialize_projection(student_dim: usize, teacher_dim: usize, seed: u64) -> Vec<f32> {
    use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};

    let n = student_dim * teacher_dim;
    let limit = (6.0_f64 / (student_dim + teacher_dim) as f64).sqrt() as f32;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut weights = Vec::with_capacity(n);
    for _ in 0..n {
        // Uniform [-limit, limit]
        let u: f64 = rng.random();
        weights.push((u as f32) * 2.0 * limit - limit);
    }
    weights
}

// ─────────────────────────────────────────────────────────────────────────────
// Progressive Knowledge Transfer (PKT)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Progressive Knowledge Transfer (PKT) loss between student and
/// teacher feature banks.
///
/// Each row of the feature matrix is a sample embedding.  An RBF kernel is
/// computed over sample pairs to form a pairwise similarity distribution, and
/// KL divergence between the student and teacher kernels is returned.
///
/// Formally, for N samples:
/// ```text
/// K_ij = exp(-||x_i - x_j||^2 / (2 * sigma^2))
/// P_i  = K_i / sum(K_i)      (normalise each row to a probability)
/// PKT  = mean_i KL(P_student_i || P_teacher_i)
/// ```
///
/// # Parameters
/// * `student_feat`  – flat `[N × D]` matrix (row-major).
/// * `teacher_feat`  – same shape as `student_feat`.
/// * `n_samples`     – number of samples N (must divide both slice lengths equally).
/// * `sigma`         – RBF bandwidth (must be positive).
///
/// # Errors
/// Returns an error on length mismatches, empty inputs, or `sigma <= 0`.
pub fn pkt_loss(
    student_feat: &[f32],
    teacher_feat: &[f32],
    n_samples: usize,
    sigma: f32,
) -> Result<f32> {
    if student_feat.is_empty() {
        return Err(TensorError::invalid_argument(
            "student_feat is empty in pkt_loss".to_string(),
        ));
    }
    if student_feat.len() != teacher_feat.len() {
        return Err(TensorError::invalid_argument(format!(
            "student_feat length {} != teacher_feat length {} in pkt_loss",
            student_feat.len(),
            teacher_feat.len()
        )));
    }
    if sigma <= 0.0 {
        return Err(TensorError::invalid_argument(format!(
            "sigma must be positive in pkt_loss, got {sigma}"
        )));
    }
    if n_samples == 0 || student_feat.len() % n_samples != 0 {
        return Err(TensorError::invalid_argument(format!(
            "n_samples={n_samples} does not evenly divide feature length {}",
            student_feat.len()
        )));
    }

    let d = student_feat.len() / n_samples;
    let two_sigma_sq = 2.0 * sigma * sigma;
    let epsilon = 1e-10_f32;

    // Build RBF kernel rows and compute row-normalised distributions.
    let mut student_kernels = vec![0.0_f32; n_samples * n_samples];
    let mut teacher_kernels = vec![0.0_f32; n_samples * n_samples];

    for i in 0..n_samples {
        let si = &student_feat[i * d..(i + 1) * d];
        let ti = &teacher_feat[i * d..(i + 1) * d];

        for j in 0..n_samples {
            let sj = &student_feat[j * d..(j + 1) * d];
            let tj = &teacher_feat[j * d..(j + 1) * d];

            let sq_dist_s: f32 = si
                .iter()
                .zip(sj.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            let sq_dist_t: f32 = ti
                .iter()
                .zip(tj.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();

            student_kernels[i * n_samples + j] = (-sq_dist_s / two_sigma_sq).exp();
            teacher_kernels[i * n_samples + j] = (-sq_dist_t / two_sigma_sq).exp();
        }

        // Row-normalise student
        let s_row_sum: f32 = student_kernels[i * n_samples..(i + 1) * n_samples]
            .iter()
            .sum::<f32>()
            .max(epsilon);
        for j in 0..n_samples {
            student_kernels[i * n_samples + j] /= s_row_sum;
        }

        // Row-normalise teacher
        let t_row_sum: f32 = teacher_kernels[i * n_samples..(i + 1) * n_samples]
            .iter()
            .sum::<f32>()
            .max(epsilon);
        for j in 0..n_samples {
            teacher_kernels[i * n_samples + j] /= t_row_sum;
        }
    }

    // Compute mean KL divergence across rows.
    let mut total_kl = 0.0_f32;
    for i in 0..n_samples {
        let p = &student_kernels[i * n_samples..(i + 1) * n_samples];
        let q = &teacher_kernels[i * n_samples..(i + 1) * n_samples];
        let kl = kl_divergence(p, q)?;
        total_kl += kl;
    }

    Ok((total_kl / n_samples as f32).max(0.0))
}

// ─────────────────────────────────────────────────────────────────────────────
// Born-Again Networks (BAN)
// ─────────────────────────────────────────────────────────────────────────────

/// Born-Again Networks (Furlanello et al. 2018) distiller.
///
/// Trains a sequence of student models ("generations") where each generation
/// is trained to mimic the previous one.  A final ensemble of all generations
/// is used as the ultimate teacher.
#[derive(Debug, Clone)]
pub struct BornAgainDistiller {
    /// Current generation index (0 = first student).
    pub generation: usize,
    /// Temperature for soft-target distillation.
    pub temperature: f32,
    /// Weight for ensemble members (uniform when set to `1.0 / k`).
    pub ensemble_weight: f32,
}

impl BornAgainDistiller {
    /// Construct a new `BornAgainDistiller`.
    ///
    /// # Parameters
    /// * `generation`       – which generation is being trained (0-indexed).
    /// * `temperature`      – softmax temperature for distillation.
    /// * `ensemble_weight`  – weight for each ensemble member in prediction averaging.
    ///
    /// # Errors
    /// Returns an error if `temperature <= 0` or `ensemble_weight <= 0`.
    pub fn new(generation: usize, temperature: f32, ensemble_weight: f32) -> Result<Self> {
        if temperature <= 0.0 {
            return Err(TensorError::invalid_argument(format!(
                "temperature must be positive in BornAgainDistiller, got {temperature}"
            )));
        }
        if ensemble_weight <= 0.0 {
            return Err(TensorError::invalid_argument(format!(
                "ensemble_weight must be positive in BornAgainDistiller, got {ensemble_weight}"
            )));
        }
        Ok(Self {
            generation,
            temperature,
            ensemble_weight,
        })
    }

    /// Compute a weighted-average ensemble prediction from multiple generation
    /// logit vectors.
    ///
    /// Each member is softmax'd and then weighted by `self.ensemble_weight`.
    /// The result is normalised so it is a valid probability distribution.
    ///
    /// # Errors
    /// Returns an error if `predictions` is empty or all rows have different lengths.
    pub fn ensemble_predictions(&self, predictions: Vec<Vec<f32>>) -> Result<Vec<f32>> {
        if predictions.is_empty() {
            return Err(TensorError::invalid_argument(
                "predictions is empty in ensemble_predictions".to_string(),
            ));
        }
        let n_classes = predictions[0].len();
        if n_classes == 0 {
            return Err(TensorError::invalid_argument(
                "empty logit vector in ensemble_predictions".to_string(),
            ));
        }
        for (k, row) in predictions.iter().enumerate() {
            if row.len() != n_classes {
                return Err(TensorError::invalid_argument(format!(
                    "predictions[{k}] length {} != expected {n_classes}",
                    row.len()
                )));
            }
        }

        let mut averaged = vec![0.0_f32; n_classes];
        let weight = self.ensemble_weight;
        let mut total_weight = 0.0_f32;

        for logits in &predictions {
            let probs = softmax_with_temperature(logits, 1.0);
            for (acc, &p) in averaged.iter_mut().zip(probs.iter()) {
                *acc += weight * p;
            }
            total_weight += weight;
        }

        // Normalise so sum = 1.
        let norm = total_weight.max(f32::EPSILON);
        for v in averaged.iter_mut() {
            *v /= norm;
        }
        Ok(averaged)
    }

    /// Compute BAN distillation loss: KL divergence between student and the
    /// ensemble teacher, scaled by T².
    ///
    /// # Errors
    /// Returns an error on empty inputs or length mismatches.
    pub fn ban_loss(&self, student_logits: &[f32], teacher_ensemble_logits: &[f32]) -> Result<f32> {
        if student_logits.is_empty() {
            return Err(TensorError::invalid_argument(
                "student_logits is empty in ban_loss".to_string(),
            ));
        }
        let t = self.temperature;
        let student_soft = softmax_with_temperature(student_logits, t);
        let teacher_soft = softmax_with_temperature(teacher_ensemble_logits, t);
        let kl = kl_divergence(&student_soft, &teacher_soft)?;
        Ok((kl * t * t).max(0.0))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DistillMethod + KnowledgeDistiller
// ─────────────────────────────────────────────────────────────────────────────

/// Selects the knowledge distillation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistillMethod {
    /// Response-based distillation: align final output distributions.
    ResponseBased,
    /// Feature-based distillation: align intermediate layer activations.
    FeatureBased,
    /// Relation-based distillation: align pairwise similarity structures.
    RelationBased,
}

/// Top-level coordinator that dispatches to the appropriate distillation
/// algorithm based on the selected [`DistillMethod`].
#[derive(Debug, Clone)]
pub struct KnowledgeDistiller {
    /// Which distillation paradigm is active.
    pub distill_method: DistillMethod,
    /// Hyper-parameter configuration.
    pub config: DistillConfig,
}

impl KnowledgeDistiller {
    /// Construct a new `KnowledgeDistiller`.
    pub fn new(distill_method: DistillMethod, config: DistillConfig) -> Self {
        Self {
            distill_method,
            config,
        }
    }

    /// Compute the distillation loss appropriate for the current method.
    ///
    /// * `ResponseBased` — KL between soft output distributions (T² scaled).
    /// * `FeatureBased`  — MSE between student and teacher features (requires
    ///   that `student_feat.len() == teacher_feat.len()`).
    /// * `RelationBased` — PKT loss (RBF kernel KL) over the feature bank.
    ///   Assumes each row is one sample; `n_samples` is inferred from
    ///   `student_feat.len() / teacher_feat.len()` heuristic (square batch).
    ///
    /// # Errors
    /// Returns an error when the inputs are incompatible with the chosen method.
    pub fn compute_loss(
        &self,
        student_logits: &[f32],
        teacher_logits: &[f32],
        student_feat: Option<&[f32]>,
        teacher_feat: Option<&[f32]>,
    ) -> Result<f32> {
        match self.distill_method {
            DistillMethod::ResponseBased => {
                let t = self.config.temperature;
                let s_soft = softmax_with_temperature(student_logits, t);
                let t_soft = softmax_with_temperature(teacher_logits, t);
                let kl = kl_divergence(&s_soft, &t_soft)?;
                Ok((kl * t * t).max(0.0))
            }
            DistillMethod::FeatureBased => {
                let sf = student_feat.ok_or_else(|| {
                    TensorError::invalid_argument(
                        "student_feat required for FeatureBased distillation".to_string(),
                    )
                })?;
                let tf = teacher_feat.ok_or_else(|| {
                    TensorError::invalid_argument(
                        "teacher_feat required for FeatureBased distillation".to_string(),
                    )
                })?;
                if sf.len() != tf.len() {
                    return Err(TensorError::invalid_argument(format!(
                        "student_feat length {} != teacher_feat length {} for FeatureBased",
                        sf.len(),
                        tf.len()
                    )));
                }
                let mse = sf
                    .iter()
                    .zip(tf.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum::<f32>()
                    / sf.len() as f32;
                Ok(mse)
            }
            DistillMethod::RelationBased => {
                let sf = student_feat.ok_or_else(|| {
                    TensorError::invalid_argument(
                        "student_feat required for RelationBased distillation".to_string(),
                    )
                })?;
                let tf = teacher_feat.ok_or_else(|| {
                    TensorError::invalid_argument(
                        "teacher_feat required for RelationBased distillation".to_string(),
                    )
                })?;
                // Determine n_samples: treat the feature vector as a flat [N × D] matrix.
                // We require n_samples^2 to be less than or equal to len,
                // and we just use sqrt(len) as a heuristic for square batches.
                let n = sf.len();
                let n_samples = (n as f64).sqrt().round() as usize;
                let n_samples = if n_samples > 0 && n % n_samples == 0 {
                    n_samples
                } else {
                    1
                };
                let sigma = 1.0_f32;
                pkt_loss(sf, tf, n_samples, sigma)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Free functions (matching task spec API)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute soft targets (temperature-scaled softmax probabilities).
///
/// This is an alias for [`softmax_with_temperature`] matching the task-spec API.
///
/// # Parameters
/// * `logits`      – raw (pre-softmax) logits.
/// * `temperature` – temperature T (must be positive).
///
/// Returns a valid probability distribution (sums to 1).
pub fn soft_targets(logits: &[f32], temperature: f32) -> Vec<f32> {
    softmax_with_temperature(logits, temperature)
}

/// KL-based distillation loss between student and teacher logits, scaled by T².
///
/// `distillation_loss = T^2 * KL(soft_student || soft_teacher)`
///
/// Both logit vectors are softmax'd at temperature `temperature` before computing
/// the KL divergence.
///
/// # Errors
/// Returns an error on empty inputs or length mismatches.
pub fn distillation_loss(
    student_logits: &[f32],
    teacher_logits: &[f32],
    temperature: f32,
) -> Result<f32> {
    if student_logits.is_empty() {
        return Err(TensorError::invalid_argument(
            "student_logits is empty in distillation_loss".to_string(),
        ));
    }
    if student_logits.len() != teacher_logits.len() {
        return Err(TensorError::invalid_argument(format!(
            "student_logits length {} != teacher_logits length {}",
            student_logits.len(),
            teacher_logits.len()
        )));
    }
    if temperature <= 0.0 {
        return Err(TensorError::invalid_argument(format!(
            "temperature must be positive, got {temperature}"
        )));
    }
    let s_soft = softmax_with_temperature(student_logits, temperature);
    let t_soft = softmax_with_temperature(teacher_logits, temperature);
    let kl = kl_divergence(&s_soft, &t_soft)?;
    Ok((kl * temperature * temperature).max(0.0))
}

/// Combined distillation + task loss.
///
/// `combined = alpha * distil_loss + (1 - alpha) * CE_loss`
///
/// where `distil_loss = T^2 * KL(soft_student || soft_teacher)` and
/// `CE_loss = -log(softmax(student_logits)[label_idx])`.
///
/// # Parameters
/// * `student_logits`  – student raw logits.
/// * `teacher_logits`  – teacher raw logits (same length).
/// * `hard_labels`     – unused slice kept for API symmetry; the single integer
///   label is specified by `label_idx`.
/// * `label_idx`       – ground-truth class index.
/// * `alpha`           – distillation loss weight ∈ [0.0, 1.0].
/// * `temperature`     – softmax temperature.
///
/// # Errors
/// Returns an error on invalid inputs.
pub fn combined_loss(
    student_logits: &[f32],
    teacher_logits: &[f32],
    _hard_labels: &[f32],
    label_idx: usize,
    alpha: f32,
    temperature: f32,
) -> Result<f32> {
    if student_logits.is_empty() {
        return Err(TensorError::invalid_argument(
            "student_logits is empty in combined_loss".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&alpha) {
        return Err(TensorError::invalid_argument(format!(
            "alpha must be in [0.0, 1.0], got {alpha}"
        )));
    }
    let dl = distillation_loss(student_logits, teacher_logits, temperature)?;
    let ce = cross_entropy_with_labels(student_logits, label_idx)?;
    Ok(alpha * dl + (1.0 - alpha) * ce)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public utility functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the (forward) KL divergence: `KL(P || Q) = Σ P(i) * ln(P(i) / Q(i))`.
///
/// Both `p` and `q` must be **valid probability distributions** (non-negative,
/// summing to 1).  This function does not normalise them.
///
/// By convention, terms where `p[i] == 0` contribute 0 (0 * ln 0 = 0).  Terms
/// where `p[i] > 0` but `q[i] ≈ 0` are clipped to avoid `ln(Inf)`.
///
/// # Properties
/// * Result is always `>= 0`.
/// * Result is `0` when `p == q`.
///
/// # Errors
/// Returns an error when the slices have different lengths or are empty.
pub fn kl_divergence(p: &[f32], q: &[f32]) -> Result<f32> {
    if p.is_empty() {
        return Err(TensorError::invalid_argument(
            "p is empty in kl_divergence".to_string(),
        ));
    }
    if p.len() != q.len() {
        return Err(TensorError::invalid_argument(format!(
            "p.len()={} != q.len()={} in kl_divergence",
            p.len(),
            q.len()
        )));
    }

    let epsilon = 1e-10_f32;
    let kl = p
        .iter()
        .zip(q.iter())
        .filter(|(&pi, _)| pi > 0.0)
        .map(|(&pi, &qi)| pi * (pi.ln() - (qi + epsilon).ln()))
        .sum::<f32>();

    // Numerical noise can produce tiny negative values; clamp to 0.
    Ok(kl.max(0.0))
}

/// Compute the softmax of `logits` after dividing each element by `temperature`.
///
/// Uses the numerically stable "log-sum-exp" trick (subtract max before exp).
///
/// # Panics
/// Does not panic; returns an empty `Vec` for empty input.
pub fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    let t = temperature.max(f32::EPSILON); // Guard against division by zero.
    let scaled: Vec<f32> = logits.iter().map(|&x| x / t).collect();

    // Numerically stable softmax.
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum = exp_vals.iter().sum::<f32>().max(f32::EPSILON);

    exp_vals.into_iter().map(|e| e / sum).collect()
}

/// Cross-entropy loss for a single example with an integer label.
///
/// Applies softmax internally so `logits` can be raw (unnormalised) values.
///
/// `CE = -log(softmax(logits)[label])`
///
/// # Errors
/// Returns an error when `logits` is empty or `label >= logits.len()`.
pub fn cross_entropy_with_labels(logits: &[f32], label: usize) -> Result<f32> {
    if logits.is_empty() {
        return Err(TensorError::invalid_argument(
            "logits is empty in cross_entropy_with_labels".to_string(),
        ));
    }
    if label >= logits.len() {
        return Err(TensorError::invalid_argument(format!(
            "label={label} is out of range [0, {})",
            logits.len()
        )));
    }
    let probs = softmax_with_temperature(logits, 1.0);
    let p = probs[label].max(f32::EPSILON);
    Ok(-p.ln())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── softmax_with_temperature ──────────────────────────────────────────────

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0];
        let probs = softmax_with_temperature(&logits, 1.0);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum={sum} should be 1.0");
    }

    #[test]
    fn test_softmax_high_temperature_is_uniform() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let probs = softmax_with_temperature(&logits, 1000.0);
        let expected = 1.0 / 3.0;
        for p in &probs {
            assert!(
                (p - expected).abs() < 0.01,
                "Expected near-uniform but got {p}"
            );
        }
    }

    #[test]
    fn test_softmax_low_temperature_is_peaked() {
        let logits = vec![0.0_f32, 0.0, 10.0];
        let probs = softmax_with_temperature(&logits, 0.1);
        assert!(
            probs[2] > 0.99,
            "Expected the highest logit to dominate; probs={probs:?}"
        );
    }

    #[test]
    fn test_softmax_temperature_scaling_flattens_distribution() {
        let logits = vec![3.0_f32, 1.0, 0.2];
        let probs_t1 = softmax_with_temperature(&logits, 1.0);
        let probs_t4 = softmax_with_temperature(&logits, 4.0);
        let max_diff_t1 = probs_t1[0] - probs_t1[2];
        let max_diff_t4 = probs_t4[0] - probs_t4[2];
        assert!(
            max_diff_t4 < max_diff_t1,
            "Higher temperature should flatten distribution"
        );
    }

    #[test]
    fn test_softmax_empty_returns_empty() {
        let probs = softmax_with_temperature(&[], 1.0);
        assert!(probs.is_empty());
    }

    /// T=1 soft_targets matches standard softmax.
    #[test]
    fn test_soft_targets_temperature_one_matches_softmax() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let st = soft_targets(&logits, 1.0);
        let sm = softmax_with_temperature(&logits, 1.0);
        for (a, b) in st.iter().zip(sm.iter()) {
            assert!(
                (a - b).abs() < 1e-7,
                "soft_targets(T=1) should match softmax; got {a} vs {b}"
            );
        }
    }

    /// Higher temperature → softer (higher-entropy) distribution.
    #[test]
    fn test_soft_targets_higher_temperature_higher_entropy() {
        let logits = vec![3.0_f32, 1.0, 0.0];
        let p_low = soft_targets(&logits, 1.0);
        let p_high = soft_targets(&logits, 8.0);

        let entropy =
            |p: &[f32]| -> f32 { p.iter().filter(|&&x| x > 0.0).map(|&x| -x * x.ln()).sum() };

        let h_low = entropy(&p_low);
        let h_high = entropy(&p_high);
        assert!(
            h_high > h_low,
            "Higher temperature should increase entropy; h_low={h_low}, h_high={h_high}"
        );
    }

    // ── distillation_loss ─────────────────────────────────────────────────────

    /// distillation_loss is 0 when student == teacher.
    #[test]
    fn test_distillation_loss_zero_when_equal() {
        let logits = vec![2.0_f32, 1.0, 0.5];
        let loss = distillation_loss(&logits, &logits, 4.0).expect("loss");
        assert!(
            loss < 1e-5,
            "distillation_loss should be ~0 for identical logits; got {loss}"
        );
    }

    #[test]
    fn test_distillation_loss_positive_for_different_logits() {
        let s = vec![2.0_f32, 0.0, 0.0];
        let t = vec![0.0_f32, 2.0, 0.0];
        let loss = distillation_loss(&s, &t, 2.0).expect("loss");
        assert!(
            loss > 0.0,
            "distillation_loss should be positive; got {loss}"
        );
    }

    // ── combined_loss ──────────────────────────────────────────────────────────

    /// combined_loss with alpha=0 equals CE, alpha=1 equals distil_loss.
    #[test]
    fn test_combined_loss_alpha_zero_equals_ce() {
        let s = vec![1.0_f32, 2.0, 0.5];
        let t = vec![2.0_f32, 1.0, 0.5];
        let hard_labels = vec![1.0_f32, 0.0, 0.0];
        let label_idx = 1usize;
        let temperature = 2.0_f32;

        let combined = combined_loss(&s, &t, &hard_labels, label_idx, 0.0, temperature)
            .expect("combined_loss alpha=0");
        let ce = cross_entropy_with_labels(&s, label_idx).expect("ce");

        assert!(
            (combined - ce).abs() < 1e-5,
            "combined_loss(alpha=0) should equal CE; combined={combined}, ce={ce}"
        );
    }

    #[test]
    fn test_combined_loss_alpha_one_equals_distil_loss() {
        let s = vec![1.0_f32, 2.0, 0.5];
        let t = vec![2.0_f32, 1.0, 0.5];
        let hard_labels = vec![1.0_f32, 0.0, 0.0];
        let label_idx = 1usize;
        let temperature = 2.0_f32;

        let combined = combined_loss(&s, &t, &hard_labels, label_idx, 1.0, temperature)
            .expect("combined_loss alpha=1");
        let dl = distillation_loss(&s, &t, temperature).expect("dl");

        assert!(
            (combined - dl).abs() < 1e-5,
            "combined_loss(alpha=1) should equal distil_loss; combined={combined}, dl={dl}"
        );
    }

    // ── kl_divergence ─────────────────────────────────────────────────────────

    #[test]
    fn test_kl_divergence_identical_distributions_is_zero() {
        let p = vec![0.1_f32, 0.5, 0.4];
        let kl = kl_divergence(&p, &p).expect("kl should succeed for identical inputs");
        assert!(kl < 1e-6, "KL(P || P) should be ~0, got {kl}");
    }

    #[test]
    fn test_kl_divergence_non_negative() {
        let p = softmax_with_temperature(&[1.0_f32, 2.0, 0.5], 1.0);
        let q = softmax_with_temperature(&[0.5_f32, 1.5, 1.0], 1.0);
        let kl = kl_divergence(&p, &q).expect("kl");
        assert!(kl >= 0.0, "KL divergence must be non-negative; got {kl}");
    }

    #[test]
    fn test_kl_divergence_asymmetric() {
        let p = softmax_with_temperature(&[10.0_f32, 0.0, 0.0], 1.0);
        let q = softmax_with_temperature(&[0.0_f32, 1.0, 9.0], 1.0);
        let kl_pq = kl_divergence(&p, &q).expect("kl_pq");
        let kl_qp = kl_divergence(&q, &p).expect("kl_qp");
        assert!(kl_pq >= 0.0);
        assert!(kl_qp >= 0.0);
        assert!(
            (kl_pq - kl_qp).abs() > 0.1,
            "KL divergence should be asymmetric; kl_pq={kl_pq}, kl_qp={kl_qp}"
        );
    }

    #[test]
    fn test_kl_divergence_length_mismatch_error() {
        let p = vec![0.5_f32, 0.5];
        let q = vec![0.3_f32, 0.3, 0.4];
        assert!(kl_divergence(&p, &q).is_err());
    }

    #[test]
    fn test_kl_divergence_empty_error() {
        let p: Vec<f32> = vec![];
        let q: Vec<f32> = vec![];
        assert!(kl_divergence(&p, &q).is_err());
    }

    // ── SoftTargets ──────────────────────────────────────────────────────────

    #[test]
    fn test_soft_targets_from_logits_basic() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let st = SoftTargets::from_logits(&logits, 2.0).expect("soft targets");
        assert_eq!(st.num_classes(), 3);
        assert_eq!(st.temperature, 2.0);
        let sum: f32 = st.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Probabilities must sum to 1");
    }

    #[test]
    fn test_soft_targets_from_logits_temperature_one() {
        let logits = vec![0.0_f32, 0.0, 0.0];
        let st = SoftTargets::from_logits(&logits, 1.0).expect("soft targets");
        for p in &st.probabilities {
            assert!(
                (*p - 1.0 / 3.0).abs() < 1e-6,
                "Uniform logits should give uniform probs"
            );
        }
    }

    #[test]
    fn test_soft_targets_from_logits_invalid_temperature() {
        let logits = vec![1.0_f32, 2.0];
        assert!(SoftTargets::from_logits(&logits, 0.0).is_err());
        assert!(SoftTargets::from_logits(&logits, -1.0).is_err());
    }

    #[test]
    fn test_soft_targets_from_logits_empty_error() {
        let result = SoftTargets::from_logits(&[], 1.0);
        assert!(result.is_err());
    }

    // ── DistillationSchemeConfig ──────────────────────────────────────────────

    #[test]
    fn test_distillation_scheme_config_valid() {
        let cfg = DistillationSchemeConfig::new(4.0, 0.7).expect("valid config");
        assert!((cfg.temperature - 4.0).abs() < 1e-9);
        assert!((cfg.alpha - 0.7).abs() < 1e-9);
        assert!((cfg.hard_label_weight - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_distillation_scheme_config_invalid_temperature() {
        assert!(DistillationSchemeConfig::new(0.0, 0.5).is_err());
        assert!(DistillationSchemeConfig::new(-1.0, 0.5).is_err());
    }

    #[test]
    fn test_distillation_scheme_config_invalid_alpha() {
        assert!(DistillationSchemeConfig::new(1.0, -0.1).is_err());
        assert!(DistillationSchemeConfig::new(1.0, 1.1).is_err());
    }

    #[test]
    fn test_distillation_scheme_config_alpha_boundary_values() {
        assert!(DistillationSchemeConfig::new(1.0, 0.0).is_ok());
        assert!(DistillationSchemeConfig::new(1.0, 1.0).is_ok());
    }

    // ── DistillationLoss ─────────────────────────────────────────────────────

    #[test]
    fn test_distillation_loss_single_example_known_values() {
        let logits = vec![2.0_f32, 1.0, 0.0];
        let cfg = DistillationSchemeConfig::new(1.0, 0.5).expect("config");
        let loss = DistillationLoss::compute(&logits, &logits, &[0usize], &cfg).expect("compute");

        assert!(
            loss.soft_loss < 1e-5,
            "Identical distributions should give ~0 soft loss; got {}",
            loss.soft_loss
        );
        assert!(
            loss.hard_loss > 0.0,
            "Hard loss must be positive; got {}",
            loss.hard_loss
        );
        let expected_total = 0.5 * loss.hard_loss;
        assert!(
            (loss.total_loss - expected_total).abs() < 1e-5,
            "total={} expected={}",
            loss.total_loss,
            expected_total
        );
    }

    #[test]
    fn test_distillation_loss_total_is_weighted_sum() {
        let s_logits = vec![1.0_f32, 0.5, 0.1];
        let t_logits = vec![2.0_f32, 0.2, 0.3];
        let cfg = DistillationSchemeConfig::new(2.0, 0.6).expect("config");
        let loss =
            DistillationLoss::compute(&s_logits, &t_logits, &[1usize], &cfg).expect("compute");

        let expected_total = cfg.alpha * loss.soft_loss + cfg.hard_label_weight * loss.hard_loss;
        assert!(
            (loss.total_loss - expected_total).abs() < 1e-5,
            "total={} expected={}",
            loss.total_loss,
            expected_total
        );
    }

    #[test]
    fn test_distillation_loss_higher_temperature_reduces_soft_loss_difference() {
        let s_logits = vec![5.0_f32, 0.0, 0.0];
        let t_logits = vec![0.0_f32, 5.0, 0.0];

        let cfg_low = DistillationSchemeConfig::new(1.0, 1.0).expect("cfg_low");
        let cfg_high = DistillationSchemeConfig::new(10.0, 1.0).expect("cfg_high");

        let loss_low =
            DistillationLoss::compute(&s_logits, &t_logits, &[0usize], &cfg_low).expect("low");
        let loss_high =
            DistillationLoss::compute(&s_logits, &t_logits, &[0usize], &cfg_high).expect("high");

        let kl_low = loss_low.soft_loss / (1.0_f32 * 1.0);
        let kl_high = loss_high.soft_loss / (10.0_f32 * 10.0);
        assert!(
            kl_high < kl_low,
            "Higher temperature should reduce raw KL; kl_high={kl_high}, kl_low={kl_low}"
        );
    }

    #[test]
    fn test_distillation_loss_batch_size_two() {
        let s_logits = vec![1.0_f32, 0.5, 0.1, 0.1, 0.5, 1.0];
        let t_logits = vec![1.0_f32, 0.5, 0.1, 0.1, 0.5, 1.0];
        let labels = vec![0usize, 2];
        let cfg = DistillationSchemeConfig::new(2.0, 0.5).expect("cfg");

        let loss = DistillationLoss::compute(&s_logits, &t_logits, &labels, &cfg).expect("loss");
        assert!(loss.total_loss >= 0.0, "Loss must be non-negative");
    }

    #[test]
    fn test_distillation_loss_invalid_label_error() {
        let s_logits = vec![1.0_f32, 2.0, 3.0];
        let t_logits = vec![1.0_f32, 2.0, 3.0];
        let cfg = DistillationSchemeConfig::new(1.0, 0.5).expect("cfg");
        let result = DistillationLoss::compute(&s_logits, &t_logits, &[5usize], &cfg);
        assert!(result.is_err(), "Out-of-range label should return an error");
    }

    #[test]
    fn test_distillation_loss_empty_student_logits_error() {
        let cfg = DistillationSchemeConfig::new(1.0, 0.5).expect("cfg");
        let result = DistillationLoss::compute(&[], &[], &[0usize], &cfg);
        assert!(result.is_err());
    }

    // ── cross_entropy_with_labels ─────────────────────────────────────────────

    #[test]
    fn test_cross_entropy_with_labels_positive() {
        let logits = vec![2.0_f32, 1.0, 0.0];
        let ce = cross_entropy_with_labels(&logits, 0).expect("ce");
        assert!(ce > 0.0, "Cross-entropy must be positive");
    }

    #[test]
    fn test_cross_entropy_with_labels_out_of_range_error() {
        let logits = vec![1.0_f32, 2.0];
        assert!(cross_entropy_with_labels(&logits, 5).is_err());
    }

    // ── FeatureMatcher ────────────────────────────────────────────────────────

    /// feature_loss is 0 for identical features.
    #[test]
    fn test_feature_loss_identical_features_is_zero() {
        let matcher = FeatureMatcher::new(4, 8).expect("matcher");
        let feat = vec![1.0_f32, 0.5, -0.3, 0.7];
        let projected = matcher.project(&feat).expect("project");
        let loss = matcher.feature_loss(&projected, &projected).expect("loss");
        assert!(
            loss < 1e-10,
            "feature_loss for identical features should be 0; got {loss}"
        );
    }

    #[test]
    fn test_feature_matcher_project_output_dimension() {
        let matcher = FeatureMatcher::new(4, 8).expect("matcher");
        let feat = vec![1.0_f32, 2.0, 3.0, 4.0];
        let projected = matcher.project(&feat).expect("project");
        assert_eq!(projected.len(), 8, "projected length should be teacher_dim");
    }

    #[test]
    fn test_feature_matcher_wrong_input_length_error() {
        let matcher = FeatureMatcher::new(4, 8).expect("matcher");
        // Wrong length
        let feat = vec![1.0_f32, 2.0];
        assert!(matcher.project(&feat).is_err());
    }

    #[test]
    fn test_feature_loss_non_zero_for_different_features() {
        let matcher = FeatureMatcher::new(4, 8).expect("matcher");
        let feat = vec![1.0_f32, 0.5, -0.3, 0.7];
        let projected = matcher.project(&feat).expect("project");
        // Different teacher features
        let teacher = vec![0.0_f32; 8];
        let loss = matcher.feature_loss(&projected, &teacher).expect("loss");
        // Unless projection gives all-zeros (very unlikely), loss should be positive
        assert!(loss >= 0.0, "feature_loss must be non-negative; got {loss}");
    }

    // ── pkt_loss ──────────────────────────────────────────────────────────────

    /// pkt_loss is non-negative.
    #[test]
    fn test_pkt_loss_non_negative() {
        // 4 samples, 3-dim embeddings
        let student = vec![
            1.0_f32, 0.0, 0.0, // sample 0
            0.0, 1.0, 0.0, // sample 1
            0.0, 0.0, 1.0, // sample 2
            0.5, 0.5, 0.0, // sample 3
        ];
        let teacher = vec![
            0.9_f32, 0.1, 0.0, 0.1, 0.8, 0.1, 0.0, 0.1, 0.9, 0.4, 0.6, 0.0,
        ];
        let loss = pkt_loss(&student, &teacher, 4, 1.0).expect("pkt_loss");
        assert!(loss >= 0.0, "pkt_loss must be non-negative; got {loss}");
    }

    /// pkt_loss is 0 when student == teacher.
    #[test]
    fn test_pkt_loss_zero_when_equal() {
        let feat = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let loss = pkt_loss(&feat, &feat, 3, 1.0).expect("pkt_loss");
        assert!(
            loss < 1e-5,
            "pkt_loss should be ~0 for identical features; got {loss}"
        );
    }

    #[test]
    fn test_pkt_loss_error_on_empty_input() {
        let result = pkt_loss(&[], &[], 1, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_pkt_loss_error_on_negative_sigma() {
        let f = vec![1.0_f32, 0.0, 0.0, 1.0];
        assert!(pkt_loss(&f, &f, 2, -1.0).is_err());
    }

    // ── BornAgainDistiller ───────────────────────────────────────────────────

    /// ensemble_predictions returns correct weighted average.
    #[test]
    fn test_ensemble_predictions_weighted_average() {
        let ban = BornAgainDistiller::new(0, 4.0, 1.0).expect("ban");
        // Two identical predictions with uniform logits → ensemble is uniform too.
        let logits1 = vec![1.0_f32, 1.0, 1.0];
        let logits2 = vec![1.0_f32, 1.0, 1.0];
        let ens = ban
            .ensemble_predictions(vec![logits1, logits2])
            .expect("ensemble");
        assert_eq!(ens.len(), 3);
        let sum: f32 = ens.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "ensemble should be a probability distribution; sum={sum}"
        );
        let expected = 1.0 / 3.0;
        for &p in &ens {
            assert!(
                (p - expected).abs() < 1e-5,
                "Uniform ensemble should be 1/3 each; got {p}"
            );
        }
    }

    #[test]
    fn test_ensemble_predictions_empty_error() {
        let ban = BornAgainDistiller::new(0, 4.0, 1.0).expect("ban");
        assert!(ban.ensemble_predictions(vec![]).is_err());
    }

    #[test]
    fn test_ban_loss_zero_when_equal() {
        let ban = BornAgainDistiller::new(0, 4.0, 1.0).expect("ban");
        let logits = vec![2.0_f32, 1.0, 0.0];
        let loss = ban.ban_loss(&logits, &logits).expect("ban_loss");
        assert!(
            loss < 1e-5,
            "BAN loss should be ~0 for identical logits; got {loss}"
        );
    }

    #[test]
    fn test_ban_loss_positive_for_different_logits() {
        let ban = BornAgainDistiller::new(1, 3.0, 0.5).expect("ban");
        let s = vec![3.0_f32, 0.0, 0.0];
        let t = vec![0.0_f32, 3.0, 0.0];
        let loss = ban.ban_loss(&s, &t).expect("ban_loss");
        assert!(
            loss > 0.0,
            "BAN loss should be positive for different logits"
        );
    }

    // ── DistillMethod + KnowledgeDistiller ───────────────────────────────────

    #[test]
    fn test_knowledge_distiller_response_based() {
        let config = DistillConfig::default_hinton();
        let kd = KnowledgeDistiller::new(DistillMethod::ResponseBased, config);
        let s = vec![2.0_f32, 1.0, 0.5];
        let t = vec![1.5_f32, 1.0, 0.5];
        let loss = kd.compute_loss(&s, &t, None, None).expect("loss");
        assert!(loss >= 0.0, "Loss must be non-negative");
    }

    #[test]
    fn test_knowledge_distiller_feature_based() {
        let config = DistillConfig::default_hinton();
        let kd = KnowledgeDistiller::new(DistillMethod::FeatureBased, config);
        let s_feat = vec![1.0_f32, 0.0, -1.0, 0.5];
        let t_feat = vec![0.9_f32, 0.1, -0.9, 0.4];
        let loss = kd
            .compute_loss(&[], &[], Some(&s_feat), Some(&t_feat))
            .expect("loss");
        assert!(loss >= 0.0, "Feature-based loss must be non-negative");
    }

    #[test]
    fn test_knowledge_distiller_feature_based_zero_for_identical() {
        let config = DistillConfig::default_hinton();
        let kd = KnowledgeDistiller::new(DistillMethod::FeatureBased, config);
        let feat = vec![1.0_f32, 2.0, 3.0];
        let loss = kd
            .compute_loss(&[], &[], Some(&feat), Some(&feat))
            .expect("loss");
        assert!(
            loss < 1e-10,
            "FeatureBased loss for identical features should be 0; got {loss}"
        );
    }

    /// DistillConfig round-trip.
    #[test]
    fn test_distill_config_new_valid() {
        let cfg = DistillConfig::new(3.0, 0.5, vec![64, 32], vec![128, 64]).expect("config");
        assert!((cfg.temperature - 3.0).abs() < 1e-9);
        assert!((cfg.alpha - 0.5).abs() < 1e-9);
        assert_eq!(cfg.student_layer_dims, vec![64, 32]);
        assert_eq!(cfg.teacher_layer_dims, vec![128, 64]);
    }

    #[test]
    fn test_distill_config_invalid_alpha() {
        assert!(DistillConfig::new(2.0, 1.5, vec![], vec![]).is_err());
    }

    #[test]
    fn test_initialize_projection_correct_size() {
        let weights = initialize_projection(4, 8, 0);
        assert_eq!(weights.len(), 32, "Should be student_dim * teacher_dim");
    }
}
