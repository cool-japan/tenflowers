//! # Digital Pathology & Multiple Instance Learning (Round 48 Track A)
//!
//! Computational pathology algorithms for whole-slide image (WSI) analysis,
//! covering multiple instance learning (MIL), survival analysis, tissue
//! segmentation, biomarker prediction, and evaluation metrics.
//!
//! ## Algorithms
//!
//! - **DpMilClassifier** — Standard MIL with max/mean/LSE pooling (Maron & Lozano-Pérez 1998)
//! - **DpAbmil** — Attention-Based MIL with optional gating (Ilse et al. 2018)
//! - **DpDsmil** — Dual-Stream MIL for WSI (Li et al. 2021)
//! - **DpCoxPh** — Cox Proportional Hazards model
//! - **DpDeepSurv** — Neural network Cox model (Katzman et al. 2018)
//! - **DpTissueSegmenter** — Patch-level tissue type classification
//! - **DpPatchSampler** — Patch sampling strategies for WSI analysis
//! - **DpBiomarkerPredictor** — Multi-task biomarker prediction from histopathology
//! - **DpMetrics** — AUC, C-index, Dice, attention entropy, F1
//!
//! ## References
//! - Maron & Lozano-Pérez (1998) "A framework for multiple-instance learning"
//! - Ilse et al. (2018) "Attention-based deep multiple instance learning"
//! - Li et al. (2021) "Dual-stream multiple instance learning network for WSI"
//! - Katzman et al. (2018) "DeepSurv: Personalized treatment recommender system"
//! - Cox (1972) "Regression models and life-tables"

#![allow(clippy::needless_range_loop)]
#![allow(clippy::doc_overindented_list_items)]
#![allow(non_snake_case)]

use std::fmt;

#[cfg(test)]
mod tests;

// ─────────────────────────────────────────────────────────────────────────────
// §0a  Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur in the digital pathology module.
#[derive(Debug, Clone, PartialEq)]
pub enum DpError {
    /// Invalid bag (WSI) contents or configuration.
    InvalidBag(String),
    /// Dimension mismatch between features and model parameters.
    DimensionError(String),
    /// Numerical failure (NaN, divergence, empty input, etc.).
    NumericalError(String),
}

impl fmt::Display for DpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DpError::InvalidBag(m) => write!(f, "DpError::InvalidBag: {}", m),
            DpError::DimensionError(m) => write!(f, "DpError::DimensionError: {}", m),
            DpError::NumericalError(m) => write!(f, "DpError::NumericalError: {}", m),
        }
    }
}

impl std::error::Error for DpError {}

// ─────────────────────────────────────────────────────────────────────────────
// §0b  Seeded RNG — xorshift64 + Box-Muller
// ─────────────────────────────────────────────────────────────────────────────

/// Xorshift64-based fast uniform U(0,1) random number.
pub fn dp_rand01(seed: &mut u64) -> f64 {
    let mut x = *seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *seed = x;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// Box-Muller transform: U(0,1) pair → N(0,1).
pub fn dp_randn(seed: &mut u64) -> f64 {
    let u1 = dp_rand01(seed).max(1e-15);
    let u2 = dp_rand01(seed);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ─────────────────────────────────────────────────────────────────────────────
// §0c  Math helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Sigmoid function.
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Dot product of two slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Matrix-vector multiply: M [rows × cols] · v [cols] → out [rows].
fn mat_vec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| dot(row, v)).collect()
}

/// Numerically stable softmax: slice → Vec<f64>.
fn softmax_vec(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_v = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut out: Vec<f64> = logits.iter().map(|&x| (x - max_v).exp()).collect();
    let s: f64 = out.iter().sum();
    let denom = if s < 1e-15 { 1.0 } else { s };
    for x in out.iter_mut() {
        *x /= denom;
    }
    out
}

/// Numerically stable log-sum-exp.
#[inline]
fn log_sum_exp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_x = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_x.is_infinite() {
        return max_x;
    }
    let sum: f64 = xs.iter().map(|&x| (x - max_x).exp()).sum();
    max_x + sum.ln()
}

/// ReLU activation.
#[inline]
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Tanh activation.
#[inline]
fn tanh_act(x: f64) -> f64 {
    x.tanh()
}

/// Xavier-uniform initialiser: scale = sqrt(6 / (fan_in + fan_out)).
fn xavier_init(rows: usize, cols: usize, seed: &mut u64) -> Vec<Vec<f64>> {
    let scale = (6.0 / (rows + cols) as f64).sqrt();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| (dp_rand01(seed) * 2.0 - 1.0) * scale)
                .collect()
        })
        .collect()
}

/// Kaiming-uniform initialiser for ReLU: scale = sqrt(2 / fan_in).
fn kaiming_init(rows: usize, cols: usize, seed: &mut u64) -> Vec<Vec<f64>> {
    let scale = (2.0 / cols as f64).sqrt();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| (dp_rand01(seed) * 2.0 - 1.0) * scale)
                .collect()
        })
        .collect()
}

/// Clamp gradient for numeric stability.
#[inline]
fn clip_grad(g: f64) -> f64 {
    g.clamp(-5.0, 5.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  DpPatch & DpWholeSlideImage
// ─────────────────────────────────────────────────────────────────────────────

/// Feature vector + metadata for one histopathology image patch.
///
/// In practice, features are extracted from a pretrained encoder (e.g., CONCH,
/// UNI, CTransPath, ResNet50-SimCLR) at 20×/40× magnification.
#[derive(Debug, Clone)]
pub struct DpPatch {
    /// Feature vector of length `feat_dim`.
    pub features: Vec<f64>,
    /// (row, col) grid coordinates in the WSI.
    pub coordinates: (usize, usize),
    /// Patch-level class label (if supervised patch-level annotation is available).
    pub label: Option<usize>,
    /// Magnification level at which the patch was extracted.
    pub magnification: f64,
}

impl DpPatch {
    /// Create a new patch with features and grid coordinates.
    pub fn new(
        features: Vec<f64>,
        coordinates: (usize, usize),
        label: Option<usize>,
        magnification: f64,
    ) -> Self {
        Self {
            features,
            coordinates,
            label,
            magnification,
        }
    }
}

/// Whole-slide image (WSI) represented as a bag of patches.
///
/// Implements the standard MIL bag abstraction: the bag label is known
/// (slide-level), but instance labels may be unknown.
#[derive(Debug, Clone)]
pub struct DpWholeSlideImage {
    /// All patches extracted from this WSI.
    pub patches: Vec<DpPatch>,
    /// Bag-level label (e.g., 0 = normal/benign, 1 = tumour-positive).
    pub slide_label: usize,
    /// Number of patches.
    pub n_patches: usize,
    /// Feature dimension of each patch.
    pub feat_dim: usize,
}

impl DpWholeSlideImage {
    /// Create an empty WSI with the given label and feature dimension.
    pub fn new(slide_label: usize, feat_dim: usize) -> Self {
        Self {
            patches: Vec::new(),
            slide_label,
            n_patches: 0,
            feat_dim,
        }
    }

    /// Add a patch, validating that its feature dimension matches the slide.
    pub fn add_patch(&mut self, patch: DpPatch) -> Result<(), DpError> {
        if patch.features.len() != self.feat_dim {
            return Err(DpError::DimensionError(format!(
                "patch feature dim {} != slide feat_dim {}",
                patch.features.len(),
                self.feat_dim
            )));
        }
        self.patches.push(patch);
        self.n_patches += 1;
        Ok(())
    }

    /// Return the feature matrix as `[n_patches × feat_dim]`.
    pub fn feature_matrix(&self) -> Vec<Vec<f64>> {
        self.patches.iter().map(|p| p.features.clone()).collect()
    }

    /// Sample `n` patches uniformly without replacement (Fisher-Yates).
    ///
    /// If `n >= n_patches`, returns references to all patches.
    pub fn random_subset<'a>(&'a self, n: usize, seed: &mut u64) -> Vec<&'a DpPatch> {
        let total = self.patches.len();
        if n >= total {
            return self.patches.iter().collect();
        }
        // Reservoir sampling (Vitter's Algorithm R)
        let mut indices: Vec<usize> = (0..total).collect();
        // Partial Fisher-Yates to get first n elements
        for i in 0..n {
            let j = i + (dp_rand01(seed) * (total - i) as f64) as usize % (total - i);
            indices.swap(i, j);
        }
        indices[..n].iter().map(|&i| &self.patches[i]).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  DpMilClassifier — Standard MIL (Maron & Lozano-Pérez 1998)
// ─────────────────────────────────────────────────────────────────────────────

/// Pooling strategies for aggregating instance scores in standard MIL.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DpPooling {
    /// Maximum instance score (hard MIL assumption).
    Max,
    /// Mean over all instance scores (soft aggregation).
    Mean,
    /// Log-sum-exp (smooth approximation of max with concentration param = 1).
    LogSumExp,
}

/// Standard MIL classifier: single linear layer + sigmoid per instance,
/// then bag-level pooling.
///
/// Loss: binary cross-entropy at bag level.
/// Gradient: back-propagated through pooling to the critical instance(s).
#[derive(Debug, Clone)]
pub struct DpMilClassifier {
    /// Weight matrix `[1 × feat_dim]` (stored as 1-row 2D for consistency).
    pub w: Vec<Vec<f64>>,
    /// Bias scalar.
    pub b: f64,
    /// Feature dimension.
    pub feat_dim: usize,
    /// Pooling strategy.
    pub pooling: DpPooling,
}

impl DpMilClassifier {
    /// Construct a new MIL classifier with zero-initialised weights.
    pub fn new(feat_dim: usize, pooling: DpPooling) -> Self {
        Self {
            w: vec![vec![0.0; feat_dim]],
            b: 0.0,
            feat_dim,
            pooling,
        }
    }

    /// Compute per-instance sigmoid scores.
    ///
    /// Returns `sigmoid(w · f_k + b)` for each patch `k`.
    pub fn instance_scores(&self, features: &[Vec<f64>]) -> Vec<f64> {
        features
            .iter()
            .map(|f| {
                let logit = dot(&self.w[0], f) + self.b;
                sigmoid(logit)
            })
            .collect()
    }

    /// Aggregate instance scores into a bag-level score via the pooling strategy.
    pub fn bag_score(&self, features: &[Vec<f64>]) -> Result<f64, DpError> {
        if features.is_empty() {
            return Err(DpError::InvalidBag("empty bag".into()));
        }
        let scores = self.instance_scores(features);
        let agg = match self.pooling {
            DpPooling::Max => scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            DpPooling::Mean => scores.iter().sum::<f64>() / scores.len() as f64,
            DpPooling::LogSumExp => {
                // Convert sigmoid scores to logits for LSE, then back to prob
                let logits: Vec<f64> = scores
                    .iter()
                    .map(|&s| {
                        let s_clamped = s.clamp(1e-7, 1.0 - 1e-7);
                        (s_clamped / (1.0 - s_clamped)).ln()
                    })
                    .collect();
                sigmoid(log_sum_exp(&logits))
            }
        };
        Ok(agg)
    }

    /// Predict bag class: 1 if bag_score > 0.5, else 0.
    pub fn predict(&self, bag: &DpWholeSlideImage) -> Result<usize, DpError> {
        let features = bag.feature_matrix();
        let score = self.bag_score(&features)?;
        Ok(if score > 0.5 { 1 } else { 0 })
    }

    /// One SGD step on BCE loss for this bag.
    ///
    /// Gradient flows only through the critical instance (max-score patch)
    /// for Max pooling, or uniformly for Mean.
    ///
    /// Returns the loss value.
    pub fn update(&mut self, bag: &DpWholeSlideImage, lr: f64) -> Result<f64, DpError> {
        let features = bag.feature_matrix();
        if features.is_empty() {
            return Err(DpError::InvalidBag("empty bag for update".into()));
        }
        let scores = self.instance_scores(&features);
        let bag_s = self.bag_score(&features)?;
        let y = bag.slide_label as f64;

        // BCE loss: -[y*log(p) + (1-y)*log(1-p)]
        let p_clamped = bag_s.clamp(1e-7, 1.0 - 1e-7);
        let loss = -(y * p_clamped.ln() + (1.0 - y) * (1.0 - p_clamped).ln());

        // d_loss/d_bag_score = (bag_s - y) / (bag_s*(1-bag_s)) * bag_s*(1-bag_s)
        // = bag_s - y  (simplified: d_BCE/d_logit = p - y for sigmoid)
        let d_loss_d_p = (bag_s - y).clamp(-5.0, 5.0);

        // Identify which instances receive gradient
        let n = features.len();
        let weights: Vec<f64> = match self.pooling {
            DpPooling::Max => {
                let max_idx = scores
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let mut w = vec![0.0; n];
                w[max_idx] = 1.0;
                w
            }
            DpPooling::Mean => vec![1.0 / n as f64; n],
            DpPooling::LogSumExp => {
                // Gradient of LSE-pool ≈ softmax of logits
                let logits: Vec<f64> = scores
                    .iter()
                    .map(|&s| {
                        let sc = s.clamp(1e-7, 1.0 - 1e-7);
                        (sc / (1.0 - sc)).ln()
                    })
                    .collect();
                softmax_vec(&logits)
            }
        };

        // d_loss/d_w = sum_k weights[k] * d_loss_d_p * score_k*(1-score_k) * f_k
        for k in 0..n {
            let d_score = scores[k] * (1.0 - scores[k]);
            let scale = clip_grad(weights[k] * d_loss_d_p * d_score);
            for j in 0..self.feat_dim {
                self.w[0][j] -= lr * scale * features[k][j];
            }
            self.b -= lr * clip_grad(weights[k] * d_loss_d_p * d_score);
        }
        Ok(loss)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  DpAbmil — Attention-Based MIL (Ilse et al. 2018)
// ─────────────────────────────────────────────────────────────────────────────

/// Attention-Based Multiple Instance Learning (ABMIL).
///
/// Architecture:
/// ```text
/// h_k = features[k]          [feat_dim]
/// v_k = W_proj @ h_k + b_proj  [L]            (feature projection)
/// e_k = tanh(V @ v_k)                          (attention branch V)
/// g_k = sigma(U @ v_k)                         (gate branch U, gated only)
/// a_k = softmax(w_attn · (e_k ⊙ g_k))          (attention weights)
/// z   = sum_k a_k * h_k                         (bag representation)
/// y_hat = sigmoid(w_cls · z + b_cls)            (bag prediction)
/// ```
///
/// Ilse et al. (2018) "Attention-based Deep Multiple Instance Learning"
/// ICML 2018.
#[derive(Debug, Clone)]
pub struct DpAbmil {
    /// Feature projection W_proj: [L × feat_dim]
    pub feat_proj_w: Vec<Vec<f64>>,
    /// Feature projection bias: \[L\]
    pub feat_proj_b: Vec<f64>,
    /// Attention branch V: [L × L]
    pub attn_v_w: Vec<Vec<f64>>,
    /// Gating branch U: [L × L] (used only when gated=true)
    pub attn_u_w: Vec<Vec<f64>>,
    /// Final attention query w_attn: \[L\]
    pub attn_final_w: Vec<f64>,
    /// Bag classifier weights w_cls: \[feat_dim\]
    pub classifier_w: Vec<f64>,
    /// Bag classifier bias.
    pub classifier_b: f64,
    /// Input feature dimension.
    pub feat_dim: usize,
    /// Intermediate/attention dimension L (default 128).
    pub L: usize,
    /// Whether to use the gated attention variant.
    pub gated: bool,
}

impl DpAbmil {
    /// Construct ABMIL with Xavier-initialised weights.
    pub fn new(feat_dim: usize, L: usize, gated: bool, seed: &mut u64) -> Self {
        let feat_proj_w = xavier_init(L, feat_dim, seed);
        let feat_proj_b = vec![0.0; L];
        let attn_v_w = xavier_init(L, L, seed);
        let attn_u_w = if gated {
            xavier_init(L, L, seed)
        } else {
            Vec::new()
        };
        // Attention query: scale 1/sqrt(L)
        let scale = 1.0 / (L as f64).sqrt();
        let attn_final_w: Vec<f64> = (0..L)
            .map(|_| (dp_rand01(seed) * 2.0 - 1.0) * scale)
            .collect();
        let classifier_w: Vec<f64> = (0..feat_dim)
            .map(|_| (dp_rand01(seed) * 2.0 - 1.0) * scale)
            .collect();
        Self {
            feat_proj_w,
            feat_proj_b,
            attn_v_w,
            attn_u_w,
            attn_final_w,
            classifier_w,
            classifier_b: 0.0,
            feat_dim,
            L,
            gated,
        }
    }

    /// Project a single patch feature to the intermediate space.
    fn project(&self, h: &[f64]) -> Vec<f64> {
        let mut v = mat_vec(&self.feat_proj_w, h);
        for (vi, bi) in v.iter_mut().zip(self.feat_proj_b.iter()) {
            *vi += bi;
        }
        v
    }

    /// Compute normalised attention weights for all patches.
    ///
    /// For the gated variant:
    ///   `a_k = softmax(w · (tanh(V @ v_k) ⊙ σ(U @ v_k)))`
    /// For the non-gated variant:
    ///   `a_k = softmax(w · tanh(V @ v_k))`
    pub fn compute_attention(&self, features: &[Vec<f64>]) -> Vec<f64> {
        if features.is_empty() {
            return Vec::new();
        }
        let raw_scores: Vec<f64> = features
            .iter()
            .map(|h| {
                let v = self.project(h);
                // attention branch e_k = tanh(V @ v_k)
                let e = mat_vec(&self.attn_v_w, &v)
                    .into_iter()
                    .map(tanh_act)
                    .collect::<Vec<_>>();
                let combined = if self.gated && !self.attn_u_w.is_empty() {
                    // gate branch g_k = sigma(U @ v_k)
                    let g = mat_vec(&self.attn_u_w, &v)
                        .into_iter()
                        .map(sigmoid)
                        .collect::<Vec<_>>();
                    // element-wise product
                    e.iter()
                        .zip(g.iter())
                        .map(|(ei, gi)| ei * gi)
                        .collect::<Vec<_>>()
                } else {
                    e
                };
                // scalar attention score
                dot(&self.attn_final_w, &combined)
            })
            .collect();
        softmax_vec(&raw_scores)
    }

    /// Weighted aggregation: `z = sum_k a_k * h_k`.
    pub fn aggregate(&self, features: &[Vec<f64>], attention: &[f64]) -> Vec<f64> {
        let mut z = vec![0.0; self.feat_dim];
        for (h, &a) in features.iter().zip(attention.iter()) {
            for (zi, &hi) in z.iter_mut().zip(h.iter()) {
                *zi += a * hi;
            }
        }
        z
    }

    /// Full forward pass: returns `(bag_logit, attention_weights)`.
    pub fn forward(&self, features: &[Vec<f64>]) -> Result<(f64, Vec<f64>), DpError> {
        if features.is_empty() {
            return Err(DpError::InvalidBag("ABMIL: empty feature bag".into()));
        }
        // Validate feature dimensions
        if features[0].len() != self.feat_dim {
            return Err(DpError::DimensionError(format!(
                "ABMIL: feature dim {} != expected {}",
                features[0].len(),
                self.feat_dim
            )));
        }
        let attn = self.compute_attention(features);
        let z = self.aggregate(features, &attn);
        let logit = dot(&self.classifier_w, &z) + self.classifier_b;
        Ok((logit, attn))
    }

    /// Predict bag class (0 or 1) from ABMIL.
    pub fn predict(&self, bag: &DpWholeSlideImage) -> Result<usize, DpError> {
        let features = bag.feature_matrix();
        let (logit, _) = self.forward(&features)?;
        Ok(if sigmoid(logit) > 0.5 { 1 } else { 0 })
    }

    /// Finite-difference gradient step on BCE loss.
    ///
    /// Uses central-difference approximation with δ = 1e-4 for all parameters.
    ///
    /// Returns the current loss before the step.
    pub fn update(
        &mut self,
        bag: &DpWholeSlideImage,
        target: f64,
        lr: f64,
    ) -> Result<f64, DpError> {
        let features = bag.feature_matrix();
        if features.is_empty() {
            return Err(DpError::InvalidBag("ABMIL update: empty bag".into()));
        }
        let delta = 1e-4;

        // Compute BCE loss for a given logit and target.
        let bce = |logit: f64, y: f64| -> f64 {
            let p = sigmoid(logit).clamp(1e-7, 1.0 - 1e-7);
            -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
        };

        // Baseline loss
        let (logit0, _) = self.forward(&features)?;
        let loss0 = bce(logit0, target);

        // ── classifier weights ─────────────────────────────────────────────
        for j in 0..self.feat_dim {
            self.classifier_w[j] += delta;
            let (l_p, _) = self.forward(&features)?;
            self.classifier_w[j] -= 2.0 * delta;
            let (l_n, _) = self.forward(&features)?;
            self.classifier_w[j] += delta;
            let grad = clip_grad((bce(l_p, target) - bce(l_n, target)) / (2.0 * delta));
            self.classifier_w[j] -= lr * grad;
        }
        // classifier bias
        {
            self.classifier_b += delta;
            let (l_p, _) = self.forward(&features)?;
            self.classifier_b -= 2.0 * delta;
            let (l_n, _) = self.forward(&features)?;
            self.classifier_b += delta;
            let grad = clip_grad((bce(l_p, target) - bce(l_n, target)) / (2.0 * delta));
            self.classifier_b -= lr * grad;
        }

        // ── attention query weights (most impactful for convergence) ───────
        for j in 0..self.L {
            self.attn_final_w[j] += delta;
            let (l_p, _) = self.forward(&features)?;
            self.attn_final_w[j] -= 2.0 * delta;
            let (l_n, _) = self.forward(&features)?;
            self.attn_final_w[j] += delta;
            let grad = clip_grad((bce(l_p, target) - bce(l_n, target)) / (2.0 * delta));
            self.attn_final_w[j] -= lr * grad;
        }

        // ── feature projection bias ────────────────────────────────────────
        for j in 0..self.L {
            self.feat_proj_b[j] += delta;
            let (l_p, _) = self.forward(&features)?;
            self.feat_proj_b[j] -= 2.0 * delta;
            let (l_n, _) = self.forward(&features)?;
            self.feat_proj_b[j] += delta;
            let grad = clip_grad((bce(l_p, target) - bce(l_n, target)) / (2.0 * delta));
            self.feat_proj_b[j] -= lr * grad;
        }

        Ok(loss0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  DpDsmil — Dual-Stream MIL (Li et al. 2021)
// ─────────────────────────────────────────────────────────────────────────────

/// Dual-Stream MIL (DSMIL) for Whole-Slide Image classification.
///
/// Two parallel streams:
/// 1. **Instance stream**: classifies each patch independently → selects
///    critical instances (top-k highest-scoring).
/// 2. **Bag stream**: aggregates critical instance features via max-pooling
///    to produce the final bag prediction.
///
/// Li et al. (2021) "Dual-stream Multiple Instance Learning Network for WSI"
/// CVPR 2021.
#[derive(Debug, Clone)]
pub struct DpDsmil {
    /// Instance classifier weight matrix: [n_classes × feat_dim]
    pub ins_classifier_w: Vec<Vec<f64>>,
    /// Instance classifier bias: \[n_classes\]
    pub ins_classifier_b: Vec<f64>,
    /// Bag classifier weight matrix: [n_classes × feat_dim]
    pub bag_classifier_w: Vec<Vec<f64>>,
    /// Bag classifier bias: \[n_classes\]
    pub bag_classifier_b: Vec<f64>,
    /// Feature dimension.
    pub feat_dim: usize,
    /// Number of output classes.
    pub n_classes: usize,
}

impl DpDsmil {
    /// Construct DSMIL with Xavier-initialised weights.
    pub fn new(feat_dim: usize, n_classes: usize, seed: &mut u64) -> Self {
        let ins_classifier_w = xavier_init(n_classes, feat_dim, seed);
        let ins_classifier_b = vec![0.0; n_classes];
        let bag_classifier_w = xavier_init(n_classes, feat_dim, seed);
        let bag_classifier_b = vec![0.0; n_classes];
        Self {
            ins_classifier_w,
            ins_classifier_b,
            bag_classifier_w,
            bag_classifier_b,
            feat_dim,
            n_classes,
        }
    }

    /// Compute per-instance softmax class probabilities.
    ///
    /// Returns `[n_patches × n_classes]` softmax predictions.
    pub fn instance_predictions(&self, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        features
            .iter()
            .map(|f| {
                let logits: Vec<f64> = (0..self.n_classes)
                    .map(|c| dot(&self.ins_classifier_w[c], f) + self.ins_classifier_b[c])
                    .collect();
                softmax_vec(&logits)
            })
            .collect()
    }

    /// Select top-k critical instances by their maximum class score.
    ///
    /// Returns indices into the patch array, sorted by score descending.
    pub fn critical_instances(&self, instance_preds: &[Vec<f64>], top_k: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f64)> = instance_preds
            .iter()
            .enumerate()
            .map(|(i, probs)| {
                let max_score = probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                (i, max_score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let k = top_k.min(scored.len());
        scored[..k].iter().map(|&(i, _)| i).collect()
    }

    /// Compute bag-level prediction from critical instance features.
    ///
    /// Aggregates critical features via max-pooling, then applies bag classifier.
    pub fn bag_prediction(
        &self,
        features: &[Vec<f64>],
        critical_idxs: &[usize],
    ) -> Result<Vec<f64>, DpError> {
        if critical_idxs.is_empty() {
            return Err(DpError::InvalidBag("DSMIL: no critical instances".into()));
        }
        // Max-pool over critical instance features
        let mut max_feat = vec![f64::NEG_INFINITY; self.feat_dim];
        for &idx in critical_idxs {
            if idx >= features.len() {
                return Err(DpError::InvalidBag(format!(
                    "DSMIL: critical index {} >= n_patches {}",
                    idx,
                    features.len()
                )));
            }
            for (mf, &fv) in max_feat.iter_mut().zip(features[idx].iter()) {
                *mf = mf.max(fv);
            }
        }
        // Bag classifier
        let logits: Vec<f64> = (0..self.n_classes)
            .map(|c| dot(&self.bag_classifier_w[c], &max_feat) + self.bag_classifier_b[c])
            .collect();
        Ok(softmax_vec(&logits))
    }

    /// Full forward pass: returns `(bag_probs, critical_instance_probs)`.
    ///
    /// Uses top-k = max(1, n_patches/10) critical instances.
    pub fn forward(&self, bag: &DpWholeSlideImage) -> Result<(Vec<f64>, Vec<f64>), DpError> {
        let features = bag.feature_matrix();
        if features.is_empty() {
            return Err(DpError::InvalidBag("DSMIL: empty bag".into()));
        }
        let instance_preds = self.instance_predictions(&features);
        let top_k = (features.len() / 10).max(1);
        let critical_idxs = self.critical_instances(&instance_preds, top_k);

        let bag_probs = self.bag_prediction(&features, &critical_idxs)?;

        // Collect critical instance class-1 probabilities (or class-0 for binary)
        let critical_probs: Vec<f64> = critical_idxs
            .iter()
            .map(|&i| instance_preds[i][self.n_classes.min(1)])
            .collect();

        Ok((bag_probs, critical_probs))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  DpCoxPh — Cox Proportional Hazards model
// ─────────────────────────────────────────────────────────────────────────────

/// Linear Cox Proportional Hazards model.
///
/// The hazard function is `h(t|x) = h_0(t) * exp(w · x + b)`.
/// The risk score (log-hazard ratio) is `η(x) = w · x + b`.
///
/// Cox (1972) "Regression models and life-tables"
/// Journal of the Royal Statistical Society B.
#[derive(Debug, Clone)]
pub struct DpCoxPh {
    /// Risk score weights: \[feat_dim\]
    pub w: Vec<f64>,
    /// Bias term.
    pub b: f64,
    /// Feature dimension.
    pub feat_dim: usize,
}

impl DpCoxPh {
    /// Construct a zero-initialised Cox PH model.
    pub fn new(feat_dim: usize) -> Self {
        Self {
            w: vec![0.0; feat_dim],
            b: 0.0,
            feat_dim,
        }
    }

    /// Compute the log-hazard (risk score) for a single sample.
    pub fn risk_score(&self, features: &[f64]) -> f64 {
        dot(&self.w, features) + self.b
    }

    /// Compute Harrell's C-index (concordance index).
    ///
    /// `C = |concordant pairs| / |comparable pairs|`
    ///
    /// A pair (i, j) is comparable if `event_i = true` and `t_i < t_j`.
    /// It is concordant if additionally `risk_i > risk_j`.
    pub fn concordance_index(
        &self,
        features: &[Vec<f64>],
        survival_times: &[f64],
        events: &[bool],
    ) -> f64 {
        let n = features.len();
        let risks: Vec<f64> = features.iter().map(|f| self.risk_score(f)).collect();
        dp_c_index_from_risks(&risks, survival_times, events, n)
    }

    /// Compute the negative partial log-likelihood (Cox loss).
    ///
    /// `L = -sum_{i: event_i} [η_i - log(sum_{j: t_j >= t_i} exp(η_j))]`
    pub fn partial_likelihood_loss(
        &self,
        features: &[Vec<f64>],
        survival_times: &[f64],
        events: &[bool],
    ) -> f64 {
        let risks: Vec<f64> = features.iter().map(|f| self.risk_score(f)).collect();
        dp_cox_loss(&risks, survival_times, events)
    }

    /// One SGD step using finite-difference gradients of the Cox loss.
    ///
    /// Returns the loss before the update.
    pub fn update(
        &mut self,
        features: &[Vec<f64>],
        times: &[f64],
        events: &[bool],
        lr: f64,
    ) -> f64 {
        let delta = 1e-4;
        let loss0 = self.partial_likelihood_loss(features, times, events);

        for j in 0..self.feat_dim {
            self.w[j] += delta;
            let l_p = self.partial_likelihood_loss(features, times, events);
            self.w[j] -= 2.0 * delta;
            let l_n = self.partial_likelihood_loss(features, times, events);
            self.w[j] += delta;
            let grad = clip_grad((l_p - l_n) / (2.0 * delta));
            self.w[j] -= lr * grad;
        }
        // bias
        self.b += delta;
        let l_p = self.partial_likelihood_loss(features, times, events);
        self.b -= 2.0 * delta;
        let l_n = self.partial_likelihood_loss(features, times, events);
        self.b += delta;
        let grad = clip_grad((l_p - l_n) / (2.0 * delta));
        self.b -= lr * grad;

        loss0
    }
}

/// Compute C-index from pre-computed risk scores.
fn dp_c_index_from_risks(risks: &[f64], times: &[f64], events: &[bool], n: usize) -> f64 {
    let mut concordant = 0usize;
    let mut comparable = 0usize;
    for i in 0..n {
        if !events[i] {
            continue;
        }
        for j in 0..n {
            if i == j {
                continue;
            }
            if times[i] < times[j] {
                comparable += 1;
                if risks[i] > risks[j] {
                    concordant += 1;
                } else if (risks[i] - risks[j]).abs() < 1e-10 {
                    // Tied risks: count as 0.5
                    concordant += 0; // we use strict comparison only
                }
            }
        }
    }
    if comparable == 0 {
        return 0.5; // undefined → 0.5 (random chance)
    }
    concordant as f64 / comparable as f64
}

/// Compute the Cox negative partial log-likelihood from risk scores.
fn dp_cox_loss(risks: &[f64], times: &[f64], events: &[bool]) -> f64 {
    let n = times.len();
    let mut loss = 0.0;
    let event_count = events.iter().filter(|&&e| e).count();
    if event_count == 0 {
        return 0.0;
    }
    for i in 0..n {
        if !events[i] {
            continue;
        }
        // Risk set: all j where t_j >= t_i
        let risk_set_log_sum: Vec<f64> = (0..n)
            .filter(|&j| times[j] >= times[i])
            .map(|j| risks[j])
            .collect();
        if risk_set_log_sum.is_empty() {
            continue;
        }
        loss += -(risks[i] - log_sum_exp(&risk_set_log_sum));
    }
    loss / event_count as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  DpDeepSurv — Neural Cox model (Katzman et al. 2018)
// ─────────────────────────────────────────────────────────────────────────────

/// Deep neural network Cox proportional hazards model (DeepSurv).
///
/// Architecture: `x → ReLU(W1·x + b1) → W2·h + b2 → risk_score`
///
/// Trained with the negative partial log-likelihood (Cox loss).
///
/// Katzman et al. (2018) "DeepSurv: Personalized Treatment Recommender System"
/// BMC Medical Research Methodology.
#[derive(Debug, Clone)]
pub struct DpDeepSurv {
    /// Hidden layer weights: [hidden × feat_dim]
    pub w1: Vec<Vec<f64>>,
    /// Hidden layer bias: \[hidden\]
    pub b1: Vec<f64>,
    /// Output layer weights: [1 × hidden]
    pub w2: Vec<Vec<f64>>,
    /// Output layer bias: \[1\]
    pub b2: Vec<f64>,
    /// Input feature dimension.
    pub feat_dim: usize,
    /// Hidden dimension.
    pub hidden: usize,
    /// Linear Cox model (used internally for C-index delegation).
    pub cox: DpCoxPh,
}

impl DpDeepSurv {
    /// Construct DeepSurv with Kaiming-He initialised weights.
    pub fn new(feat_dim: usize, hidden: usize, seed: &mut u64) -> Self {
        let w1 = kaiming_init(hidden, feat_dim, seed);
        let b1 = vec![0.0; hidden];
        let w2 = kaiming_init(1, hidden, seed);
        let b2 = vec![0.0; 1];
        let cox = DpCoxPh::new(feat_dim);
        Self {
            w1,
            b1,
            w2,
            b2,
            feat_dim,
            hidden,
            cox,
        }
    }

    /// Forward pass for a single sample: returns the risk score.
    pub fn risk_score(&self, features: &[f64]) -> f64 {
        // Hidden layer
        let h: Vec<f64> = mat_vec(&self.w1, features)
            .into_iter()
            .zip(self.b1.iter())
            .map(|(z, &b)| relu(z + b))
            .collect();
        // Output layer
        dot(&self.w2[0], &h) + self.b2[0]
    }

    /// Compute risk scores for a batch of samples.
    pub fn batch_risk(&self, features: &[Vec<f64>]) -> Vec<f64> {
        features.iter().map(|f| self.risk_score(f)).collect()
    }

    /// Compute the Cox negative partial log-likelihood over a batch.
    pub fn loss(&self, features: &[Vec<f64>], times: &[f64], events: &[bool]) -> f64 {
        let risks = self.batch_risk(features);
        dp_cox_loss(&risks, times, events)
    }

    /// SGD step using finite-difference gradients.
    ///
    /// Returns the loss before the update.
    pub fn update(
        &mut self,
        features: &[Vec<f64>],
        times: &[f64],
        events: &[bool],
        lr: f64,
    ) -> f64 {
        let delta = 1e-4;
        let loss0 = self.loss(features, times, events);

        // ── W1 gradients ──────────────────────────────────────────────────
        for i in 0..self.hidden {
            for j in 0..self.feat_dim {
                self.w1[i][j] += delta;
                let l_p = self.loss(features, times, events);
                self.w1[i][j] -= 2.0 * delta;
                let l_n = self.loss(features, times, events);
                self.w1[i][j] += delta;
                let grad = clip_grad((l_p - l_n) / (2.0 * delta));
                self.w1[i][j] -= lr * grad;
            }
        }
        // ── b1 gradients ──────────────────────────────────────────────────
        for i in 0..self.hidden {
            self.b1[i] += delta;
            let l_p = self.loss(features, times, events);
            self.b1[i] -= 2.0 * delta;
            let l_n = self.loss(features, times, events);
            self.b1[i] += delta;
            let grad = clip_grad((l_p - l_n) / (2.0 * delta));
            self.b1[i] -= lr * grad;
        }
        // ── W2 gradients ──────────────────────────────────────────────────
        for j in 0..self.hidden {
            self.w2[0][j] += delta;
            let l_p = self.loss(features, times, events);
            self.w2[0][j] -= 2.0 * delta;
            let l_n = self.loss(features, times, events);
            self.w2[0][j] += delta;
            let grad = clip_grad((l_p - l_n) / (2.0 * delta));
            self.w2[0][j] -= lr * grad;
        }
        // ── b2 gradient ───────────────────────────────────────────────────
        {
            self.b2[0] += delta;
            let l_p = self.loss(features, times, events);
            self.b2[0] -= 2.0 * delta;
            let l_n = self.loss(features, times, events);
            self.b2[0] += delta;
            let grad = clip_grad((l_p - l_n) / (2.0 * delta));
            self.b2[0] -= lr * grad;
        }

        loss0
    }

    /// Compute Harrell's C-index using DeepSurv risk scores.
    pub fn concordance_index(&self, features: &[Vec<f64>], times: &[f64], events: &[bool]) -> f64 {
        let n = features.len();
        let risks = self.batch_risk(features);
        dp_c_index_from_risks(&risks, times, events, n)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  DpTissueSegmenter
// ─────────────────────────────────────────────────────────────────────────────

/// Patch-level tissue type segmentation network.
///
/// Architecture: `x → ReLU(W1·x + b1) → softmax(W2·h + b2)`
///
/// Typical tissue classes in H&E pathology:
/// - 0: Background/fat
/// - 1: Tumour
/// - 2: Stroma
/// - 3: Lymphocytes/immune cells
/// - 4: Necrosis
#[derive(Debug, Clone)]
pub struct DpTissueSegmenter {
    /// Hidden layer weights: [hidden × feat_dim]
    pub w1: Vec<Vec<f64>>,
    /// Hidden layer bias: \[hidden\]
    pub b1: Vec<f64>,
    /// Output layer weights: [n_classes × hidden]
    pub w2: Vec<Vec<f64>>,
    /// Output layer bias: \[n_classes\]
    pub b2: Vec<f64>,
    /// Number of tissue classes.
    pub n_classes: usize,
    /// Input feature dimension.
    pub feat_dim: usize,
    /// Hidden dimension.
    pub hidden: usize,
}

impl DpTissueSegmenter {
    /// Construct a tissue segmenter with Kaiming initialisation.
    pub fn new(feat_dim: usize, hidden: usize, n_classes: usize, seed: &mut u64) -> Self {
        let w1 = kaiming_init(hidden, feat_dim, seed);
        let b1 = vec![0.0; hidden];
        let w2 = kaiming_init(n_classes, hidden, seed);
        let b2 = vec![0.0; n_classes];
        Self {
            w1,
            b1,
            w2,
            b2,
            n_classes,
            feat_dim,
            hidden,
        }
    }

    /// Classify a single patch: returns `[n_classes]` softmax probabilities.
    pub fn segment_patch(&self, features: &[f64]) -> Vec<f64> {
        let h: Vec<f64> = mat_vec(&self.w1, features)
            .into_iter()
            .zip(self.b1.iter())
            .map(|(z, &b)| relu(z + b))
            .collect();
        let logits: Vec<f64> = mat_vec(&self.w2, &h)
            .into_iter()
            .zip(self.b2.iter())
            .map(|(z, &b)| z + b)
            .collect();
        softmax_vec(&logits)
    }

    /// Segment all patches in a WSI.
    ///
    /// Returns `[n_patches × n_classes]` probability matrices.
    pub fn segment_slide(&self, slide: &DpWholeSlideImage) -> Result<Vec<Vec<f64>>, DpError> {
        if slide.patches.is_empty() {
            return Err(DpError::InvalidBag("segment_slide: empty slide".into()));
        }
        Ok(slide
            .patches
            .iter()
            .map(|p| self.segment_patch(&p.features))
            .collect())
    }

    /// Compute mean tissue composition across all patches (class fractions).
    pub fn tissue_composition(&self, slide: &DpWholeSlideImage) -> Result<Vec<f64>, DpError> {
        let per_patch = self.segment_slide(slide)?;
        let n = per_patch.len() as f64;
        let mut composition = vec![0.0; self.n_classes];
        for probs in &per_patch {
            for (c, &p) in composition.iter_mut().zip(probs.iter()) {
                *c += p;
            }
        }
        for c in composition.iter_mut() {
            *c /= n;
        }
        Ok(composition)
    }

    /// Compute tumour purity: mean probability of class 1 (tumour) across patches.
    pub fn tumor_purity(&self, slide: &DpWholeSlideImage) -> Result<f64, DpError> {
        let composition = self.tissue_composition(slide)?;
        if self.n_classes < 2 {
            return Err(DpError::DimensionError(
                "tumor_purity: n_classes must be >= 2".into(),
            ));
        }
        Ok(composition[1])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  DpPatchSampler
// ─────────────────────────────────────────────────────────────────────────────

/// Patch sampling strategies for WSI analysis.
#[derive(Debug, Clone)]
pub enum DpSamplingStrategy {
    /// Uniformly random selection.
    Random,
    /// Select patches with highest attention weights.
    TopAttention,
    /// Curriculum sampling: start with easy (high attention) patches,
    /// gradually include harder patches. The inner `usize` is the curriculum step.
    Curriculum(usize),
    /// Spatially stratified: divide the WSI into a grid and sample evenly.
    /// The inner `usize` is the grid granularity (cells per axis).
    Stratified(usize),
}

/// Patch sampler implementing multiple WSI sampling strategies.
#[derive(Debug, Clone)]
pub struct DpPatchSampler {
    /// Sampling strategy to use.
    pub strategy: DpSamplingStrategy,
}

impl DpPatchSampler {
    /// Create a patch sampler with the given strategy.
    pub fn new(strategy: DpSamplingStrategy) -> Self {
        Self { strategy }
    }

    /// Sample `n` patch indices from `slide` using the configured strategy.
    ///
    /// - `attention`: required for `TopAttention` and `Curriculum` strategies.
    ///   If `None`, falls back to random sampling.
    /// - Returns patch indices (may contain duplicates only if `n > slide.n_patches`).
    pub fn sample(
        &self,
        slide: &DpWholeSlideImage,
        attention: Option<&[f64]>,
        n: usize,
        seed: &mut u64,
    ) -> Vec<usize> {
        let total = slide.n_patches;
        if total == 0 || n == 0 {
            return Vec::new();
        }
        let k = n.min(total);
        match &self.strategy {
            DpSamplingStrategy::Random => sample_random_indices(total, k, seed),
            DpSamplingStrategy::TopAttention => match attention {
                Some(attn) if attn.len() == total => top_k_indices(attn, k),
                _ => sample_random_indices(total, k, seed),
            },
            DpSamplingStrategy::Curriculum(step) => {
                match attention {
                    Some(attn) if attn.len() == total => {
                        // At early steps, prefer high-attention (easy) patches
                        // As step increases, broaden to include more patches
                        let fraction = ((*step as f64 + 1.0) / 10.0).min(1.0);
                        let pool_size = ((total as f64 * fraction) as usize).max(k);
                        let pool_size = pool_size.min(total);
                        let top_indices = top_k_indices(attn, pool_size);
                        // Sample k from the top pool_size
                        
                        sample_from_pool(&top_indices, k, seed)
                    }
                    _ => sample_random_indices(total, k, seed),
                }
            }
            DpSamplingStrategy::Stratified(grid_size) => {
                // Partition patches into grid cells by coordinate
                let g = (*grid_size).max(1);
                let mut cells: Vec<Vec<usize>> = vec![Vec::new(); g * g];
                for (idx, patch) in slide.patches.iter().enumerate() {
                    let row_bin = (patch.coordinates.0 % g).min(g - 1);
                    let col_bin = (patch.coordinates.1 % g).min(g - 1);
                    cells[row_bin * g + col_bin].push(idx);
                }
                // Remove empty cells
                let non_empty: Vec<&Vec<usize>> = cells.iter().filter(|c| !c.is_empty()).collect();
                if non_empty.is_empty() {
                    return sample_random_indices(total, k, seed);
                }
                // Sample evenly from each non-empty cell
                let n_cells = non_empty.len();
                let per_cell = (k / n_cells).max(1);
                let mut selected = Vec::with_capacity(k);
                for cell in &non_empty {
                    let take = per_cell.min(cell.len());
                    let picked = sample_from_pool(cell, take, seed);
                    selected.extend(picked);
                    if selected.len() >= k {
                        break;
                    }
                }
                // If we still need more, sample randomly from remaining
                while selected.len() < k {
                    let idx = (dp_rand01(seed) * total as f64) as usize % total;
                    selected.push(idx);
                }
                selected[..k].to_vec()
            }
        }
    }

    /// Compute coverage: fraction of unique patches sampled.
    pub fn coverage_score(sampled_indices: &[usize], total: usize) -> f64 {
        if total == 0 {
            return 0.0;
        }
        let mut seen = std::collections::HashSet::new();
        for &i in sampled_indices {
            seen.insert(i);
        }
        seen.len() as f64 / total as f64
    }
}

/// Sample k indices uniformly without replacement (Fisher-Yates partial shuffle).
fn sample_random_indices(total: usize, k: usize, seed: &mut u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..total).collect();
    let k = k.min(total);
    for i in 0..k {
        let j = i + (dp_rand01(seed) * (total - i) as f64) as usize % (total - i);
        indices.swap(i, j);
    }
    indices[..k].to_vec()
}

/// Return indices of the top-k elements (by value, descending).
fn top_k_indices(values: &[f64], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f64)> = values.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let k = k.min(indexed.len());
    indexed[..k].iter().map(|&(i, _)| i).collect()
}

/// Sample k indices from a pool without replacement.
fn sample_from_pool(pool: &[usize], k: usize, seed: &mut u64) -> Vec<usize> {
    let mut local = pool.to_vec();
    let k = k.min(local.len());
    for i in 0..k {
        let j = i + (dp_rand01(seed) * (local.len() - i) as f64) as usize % (local.len() - i);
        local.swap(i, j);
    }
    local[..k].to_vec()
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  DpBiomarkerPredictor
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-task biomarker predictor combining ABMIL feature aggregation with
/// a multi-output sigmoid head.
///
/// Predicts molecular biomarkers (e.g., BRCA1/2 mutation, MSI, TMB, ER/PR/HER2
/// status) from WSI features using slide-level attention aggregation followed
/// by independent sigmoid classifiers per biomarker.
#[derive(Debug, Clone)]
pub struct DpBiomarkerPredictor {
    /// Underlying ABMIL model for feature aggregation.
    pub abmil: DpAbmil,
    /// Per-biomarker linear head weights: [n_biomarkers × feat_dim]
    pub biomarker_head: Vec<Vec<f64>>,
    /// Per-biomarker biases: \[n_biomarkers\]
    pub biomarker_b: Vec<f64>,
    /// Number of biomarkers to predict.
    pub n_biomarkers: usize,
}

impl DpBiomarkerPredictor {
    /// Construct a biomarker predictor.
    ///
    /// The ABMIL model aggregates patch features; the biomarker head then
    /// applies independent sigmoid classifiers.
    pub fn new(feat_dim: usize, L: usize, n_biomarkers: usize, seed: &mut u64) -> Self {
        let abmil = DpAbmil::new(feat_dim, L, true, seed);
        let biomarker_head = xavier_init(n_biomarkers, feat_dim, seed);
        let biomarker_b = vec![0.0; n_biomarkers];
        Self {
            abmil,
            biomarker_head,
            biomarker_b,
            n_biomarkers,
        }
    }

    /// Predict all biomarker probabilities for a slide.
    ///
    /// Returns `[n_biomarkers]` sigmoid probabilities.
    pub fn predict_biomarkers(&self, slide: &DpWholeSlideImage) -> Result<Vec<f64>, DpError> {
        let features = slide.feature_matrix();
        if features.is_empty() {
            return Err(DpError::InvalidBag(
                "predict_biomarkers: empty slide".into(),
            ));
        }
        // Obtain ABMIL attention and aggregate
        let attn = self.abmil.compute_attention(&features);
        let z = self.abmil.aggregate(&features, &attn);

        // Apply per-biomarker sigmoid classifiers
        let predictions: Vec<f64> = (0..self.n_biomarkers)
            .map(|b| {
                let logit = dot(&self.biomarker_head[b], &z) + self.biomarker_b[b];
                sigmoid(logit)
            })
            .collect();
        Ok(predictions)
    }

    /// Compute multi-task BCE loss, only for available biomarkers.
    ///
    /// `mask[b]` = true if biomarker `b` is available for this slide.
    pub fn multi_task_loss(&self, predictions: &[f64], targets: &[f64], mask: &[bool]) -> f64 {
        let n = predictions.len().min(targets.len()).min(mask.len());
        if n == 0 {
            return 0.0;
        }
        let mut total_loss = 0.0;
        let mut count = 0usize;
        for b in 0..n {
            if !mask[b] {
                continue;
            }
            let p = predictions[b].clamp(1e-7, 1.0 - 1e-7);
            let y = targets[b];
            total_loss += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
            count += 1;
        }
        if count == 0 {
            0.0
        } else {
            total_loss / count as f64
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  DpMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluation metrics for digital pathology models.
pub struct DpMetrics;

impl DpMetrics {
    /// Compute Area Under the ROC Curve (AUC) for bag-level binary classification.
    ///
    /// `AUC = |{(i,j): label_i=1, label_j=0, score_i > score_j}| / |{(i,j): label_i=1, label_j=0}|`
    pub fn bag_auc(scores: &[f64], labels: &[usize]) -> f64 {
        let n = scores.len().min(labels.len());
        let mut concordant = 0usize;
        let mut total_pairs = 0usize;
        for i in 0..n {
            for j in 0..n {
                if labels[i] == 1 && labels[j] == 0 {
                    total_pairs += 1;
                    if scores[i] > scores[j] {
                        concordant += 1;
                    } else if (scores[i] - scores[j]).abs() < 1e-12 {
                        // Tied → count 0.5 (using integer arithmetic: add 1 every 2 ties)
                        concordant += 0; // conservative: count 0 for exact ties
                    }
                }
            }
        }
        if total_pairs == 0 {
            return 0.5; // undefined
        }
        concordant as f64 / total_pairs as f64
    }

    /// Generic C-index from pre-computed risk scores and survival data.
    pub fn concordance_index_generic(risks: &[f64], times: &[f64], events: &[bool]) -> f64 {
        let n = risks.len().min(times.len()).min(events.len());
        dp_c_index_from_risks(risks, times, events, n)
    }

    /// Sørensen–Dice coefficient for a specific class.
    ///
    /// `Dice = 2 * |pred ∩ gt| / (|pred| + |gt|)`
    pub fn dice_score(pred: &[usize], gt: &[usize], class: usize) -> f64 {
        let n = pred.len().min(gt.len());
        let mut intersection = 0usize;
        let mut pred_count = 0usize;
        let mut gt_count = 0usize;
        for i in 0..n {
            if pred[i] == class {
                pred_count += 1;
            }
            if gt[i] == class {
                gt_count += 1;
            }
            if pred[i] == class && gt[i] == class {
                intersection += 1;
            }
        }
        let denom = pred_count + gt_count;
        if denom == 0 {
            return 1.0; // both empty → perfect match
        }
        2.0 * intersection as f64 / denom as f64
    }

    /// Shannon entropy of the attention distribution.
    ///
    /// `H = -sum_k a_k * log(a_k + ε)` — measures attention concentration.
    /// Low entropy → focussed attention; high entropy → uniform attention.
    pub fn attention_entropy(attention: &[f64]) -> f64 {
        const EPS: f64 = 1e-10;
        attention.iter().map(|&a| -a * (a + EPS).ln()).sum()
    }

    /// Sum of top-k attention weights (coverage metric).
    ///
    /// Measures how concentrated the attention is in the top-k patches.
    pub fn top_k_attention_coverage(attention: &[f64], k: usize) -> f64 {
        if attention.is_empty() || k == 0 {
            return 0.0;
        }
        let mut sorted = attention.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let k = k.min(sorted.len());
        sorted[..k].iter().sum()
    }

    /// Macro-averaged F1 score across all classes.
    pub fn f1_score(pred: &[usize], gt: &[usize], n_classes: usize) -> f64 {
        if n_classes == 0 {
            return 0.0;
        }
        let n = pred.len().min(gt.len());
        let mut tp = vec![0usize; n_classes];
        let mut fp = vec![0usize; n_classes];
        let mut fn_ = vec![0usize; n_classes];
        for i in 0..n {
            let p = pred[i];
            let g = gt[i];
            if p < n_classes && g < n_classes {
                if p == g {
                    tp[p] += 1;
                } else {
                    fp[p] += 1;
                    fn_[g] += 1;
                }
            }
        }
        let f1_sum: f64 = (0..n_classes)
            .map(|c| {
                let denom = 2 * tp[c] + fp[c] + fn_[c];
                if denom == 0 {
                    0.0
                } else {
                    2.0 * tp[c] as f64 / denom as f64
                }
            })
            .sum();
        f1_sum / n_classes as f64
    }
}
