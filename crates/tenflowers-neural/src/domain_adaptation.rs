//! Domain Adaptation and Domain Generalization Methods
//!
//! Implements DANN (Ganin et al. 2016), CORAL (Sun & Saenko 2016),
//! MMD (Gretton et al. 2012), IRM (Arjovsky et al. 2019), TENT (Wang et al. 2021),
//! and associated evaluation metrics for domain shift scenarios.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// 1. Domain Data Representations
// ─────────────────────────────────────────────────────────────────────────────

/// A single sample from a domain with optional class label.
#[derive(Clone, Debug)]
pub struct DomainSample {
    pub features: Vec<f32>,
    pub label: Option<usize>,
    pub domain_id: usize,
}

/// Dataset containing samples from multiple domains.
#[derive(Clone, Debug)]
pub struct DomainDataset {
    pub samples: Vec<DomainSample>,
    pub n_domains: usize,
    pub feature_dim: usize,
}

impl DomainDataset {
    /// Create a new domain dataset.
    pub fn new(samples: Vec<DomainSample>, n_domains: usize, feature_dim: usize) -> Self {
        Self {
            samples,
            n_domains,
            feature_dim,
        }
    }

    /// Return references to all source-domain samples (domain_id == 0).
    pub fn source_samples(&self) -> Vec<&DomainSample> {
        self.samples.iter().filter(|s| s.domain_id == 0).collect()
    }

    /// Return references to all target-domain samples (domain_id != 0).
    pub fn target_samples(&self) -> Vec<&DomainSample> {
        self.samples.iter().filter(|s| s.domain_id != 0).collect()
    }
}

/// Summary statistics for a single domain.
#[derive(Clone, Debug)]
pub struct DomainStats {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub n_samples: usize,
}

/// Compute per-feature mean and standard deviation for the given domain_id.
pub fn domain_statistics(dataset: &DomainDataset, domain_id: usize) -> DomainStats {
    let domain_samples: Vec<&DomainSample> = dataset
        .samples
        .iter()
        .filter(|s| s.domain_id == domain_id)
        .collect();
    let n = domain_samples.len();
    let dim = dataset.feature_dim;
    if n == 0 || dim == 0 {
        return DomainStats {
            mean: vec![0.0; dim],
            std: vec![0.0; dim],
            n_samples: 0,
        };
    }
    let mut mean = vec![0.0f32; dim];
    for s in &domain_samples {
        for (i, &v) in s.features.iter().enumerate() {
            if i < dim {
                mean[i] += v;
            }
        }
    }
    for m in &mut mean {
        *m /= n as f32;
    }
    let mut variance = vec![0.0f32; dim];
    for s in &domain_samples {
        for (i, &v) in s.features.iter().enumerate() {
            if i < dim {
                let diff = v - mean[i];
                variance[i] += diff * diff;
            }
        }
    }
    let denom = if n > 1 { (n - 1) as f32 } else { 1.0 };
    let std: Vec<f32> = variance.iter().map(|&v| (v / denom).sqrt()).collect();
    DomainStats {
        mean,
        std,
        n_samples: n,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. DANN — Domain-Adversarial Neural Networks
// ─────────────────────────────────────────────────────────────────────────────

/// Gradient Reversal Layer — identity in the forward pass; during back-prop the
/// gradient is multiplied by -lambda.
#[derive(Clone, Debug)]
pub struct GradientReversalLayer {
    pub lambda: f32,
}

impl GradientReversalLayer {
    pub fn new(lambda: f32) -> Self {
        Self { lambda }
    }

    /// Forward pass: identity transformation.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        x.to_vec()
    }

    /// Returns the gradient scaling factor (negative lambda for reversal).
    pub fn backward_scale(&self) -> f32 {
        -self.lambda
    }
}

/// Xavier uniform initializer helper (in place).
fn xavier_fill(w: &mut [f32], fan_in: usize, fan_out: usize, rng: &mut StdRng) {
    let limit = (6.0f32 / (fan_in + fan_out) as f32).sqrt();
    for v in w.iter_mut() {
        *v = rng.random_range(-limit..limit);
    }
}

/// Simple dense-layer sigmoid activation.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Softmax over a slice.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// ReLU activation.
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Matrix-vector multiply: W * x + b (W shape: [out, in]).
fn linear_forward(weights: &[Vec<f32>], bias: &[f32], x: &[f32]) -> Vec<f32> {
    weights
        .iter()
        .zip(bias.iter())
        .map(|(row, &b)| {
            let dot: f32 = row.iter().zip(x.iter()).map(|(&w, &xi)| w * xi).sum();
            dot + b
        })
        .collect()
}

/// A simple two-hidden-layer feature extractor (input → hidden → output).
#[derive(Clone, Debug)]
pub struct DaFeatureExtractor {
    /// Each tuple is (weight_matrix [out x in], bias \[out\]).
    pub layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
}

impl DaFeatureExtractor {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, rng: &mut StdRng) -> Self {
        let make_layer = |fan_in: usize, fan_out: usize, rng: &mut StdRng| {
            let mut w: Vec<Vec<f32>> = (0..fan_out).map(|_| vec![0.0f32; fan_in]).collect();
            for row in &mut w {
                xavier_fill(row, fan_in, fan_out, rng);
            }
            let b = vec![0.0f32; fan_out];
            (w, b)
        };
        let l1 = make_layer(input_dim, hidden_dim, rng);
        let l2 = make_layer(hidden_dim, output_dim, rng);
        Self {
            layers: vec![l1, l2],
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut h = x.to_vec();
        for (i, (w, b)) in self.layers.iter().enumerate() {
            h = linear_forward(w, b, &h);
            // Apply ReLU to all but the last layer
            if i + 1 < self.layers.len() {
                h = h.iter().map(|&v| relu(v)).collect();
            }
        }
        h
    }
}

/// Binary domain classifier (source vs. target), producing a probability via sigmoid.
#[derive(Clone, Debug)]
pub struct DannDomainClassifier {
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
}

impl DannDomainClassifier {
    pub fn new(feature_dim: usize, rng: &mut StdRng) -> Self {
        let hidden = 64.max(feature_dim / 2);
        let mut w1: Vec<Vec<f32>> = (0..hidden).map(|_| vec![0.0f32; feature_dim]).collect();
        for row in &mut w1 {
            xavier_fill(row, feature_dim, hidden, rng);
        }
        let b1 = vec![0.0f32; hidden];
        let mut w2: Vec<Vec<f32>> = vec![vec![0.0f32; hidden]];
        xavier_fill(&mut w2[0], hidden, 1, rng);
        let b2 = vec![0.0f32; 1];
        // Store as a two-layer network flattened into weights/bias fields
        // For simplicity, store w1+w2 concatenated as "weights", and b1+b2 as "bias"
        // We'll keep it as two separate layers stored in a wrapper struct below.
        // Reuse the struct fields to hold both layers by encoding: first hidden rows are w1,
        // the next 1 row is w2.
        let mut all_weights = w1;
        all_weights.extend(w2);
        let mut all_bias = b1;
        all_bias.extend(b2);
        Self {
            weights: all_weights,
            bias: all_bias,
        }
    }

    pub fn forward(&self, x: &[f32]) -> f32 {
        let hidden = self.bias.len() - 1;
        let feature_dim = if hidden > 0 && !self.weights.is_empty() {
            self.weights[0].len()
        } else {
            x.len()
        };
        let _ = feature_dim;
        // Layer 1: hidden units (first `hidden` rows)
        let h1: Vec<f32> = self.weights[..hidden]
            .iter()
            .zip(self.bias[..hidden].iter())
            .map(|(row, &b)| {
                let dot: f32 = row.iter().zip(x.iter()).map(|(&w, &xi)| w * xi).sum();
                relu(dot + b)
            })
            .collect();
        // Layer 2: single output neuron
        let row2 = &self.weights[hidden];
        let b2 = self.bias[hidden];
        let logit: f32 = row2
            .iter()
            .zip(h1.iter())
            .map(|(&w, &h)| w * h)
            .sum::<f32>()
            + b2;
        sigmoid(logit)
    }
}

/// Multi-class label predictor.
#[derive(Clone, Debug)]
pub struct DaLabelPredictor {
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
}

impl DaLabelPredictor {
    pub fn new(feature_dim: usize, n_classes: usize, rng: &mut StdRng) -> Self {
        let mut w: Vec<Vec<f32>> = (0..n_classes).map(|_| vec![0.0f32; feature_dim]).collect();
        for row in &mut w {
            xavier_fill(row, feature_dim, n_classes, rng);
        }
        let b = vec![0.0f32; n_classes];
        Self {
            weights: w,
            bias: b,
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let logits = linear_forward(&self.weights, &self.bias, x);
        softmax(&logits)
    }
}

/// DANN: Feature extractor + domain classifier + label predictor with GRL.
#[derive(Clone, Debug)]
pub struct DannModel {
    pub feature_extractor: DaFeatureExtractor,
    pub domain_classifier: DannDomainClassifier,
    pub label_predictor: DaLabelPredictor,
    pub grl: GradientReversalLayer,
}

impl DannModel {
    pub fn new(
        input_dim: usize,
        feature_dim: usize,
        n_classes: usize,
        lambda: f32,
        rng: &mut StdRng,
    ) -> Self {
        let feature_extractor =
            DaFeatureExtractor::new(input_dim, feature_dim * 2, feature_dim, rng);
        let domain_classifier = DannDomainClassifier::new(feature_dim, rng);
        let label_predictor = DaLabelPredictor::new(feature_dim, n_classes, rng);
        let grl = GradientReversalLayer::new(lambda);
        Self {
            feature_extractor,
            domain_classifier,
            label_predictor,
            grl,
        }
    }

    /// Return class probability logits (softmax) for an input.
    pub fn classify(&self, x: &[f32]) -> Vec<f32> {
        let features = self.feature_extractor.forward(x);
        self.label_predictor.forward(&features)
    }

    /// Return domain discrimination probability (0=source, 1=target) via GRL path.
    pub fn domain_discriminate(&self, x: &[f32]) -> f32 {
        let features = self.feature_extractor.forward(x);
        let reversed = self.grl.forward(&features);
        self.domain_classifier.forward(&reversed)
    }

    /// DANN combined loss: cross-entropy label loss + binary domain cross-entropy.
    ///
    /// * `src_features` — raw input features for labelled source samples
    /// * `src_labels`   — ground-truth class indices
    /// * `tgt_features` — raw input features for unlabelled target samples
    pub fn dann_loss(
        &self,
        src_features: &[Vec<f32>],
        src_labels: &[usize],
        tgt_features: &[Vec<f32>],
    ) -> f32 {
        // Classification loss on source samples
        let eps = 1e-10f32;
        let n_src = src_features.len();
        let cls_loss: f32 = if n_src == 0 {
            0.0
        } else {
            src_features
                .iter()
                .zip(src_labels.iter())
                .map(|(x, &y)| {
                    let probs = self.classify(x);
                    let p = probs.get(y).copied().unwrap_or(eps).max(eps);
                    -p.ln()
                })
                .sum::<f32>()
                / n_src as f32
        };

        // Domain classification loss (source = 0, target = 1)
        let n_dom = src_features.len() + tgt_features.len();
        let domain_loss: f32 = if n_dom == 0 {
            0.0
        } else {
            let src_dom: f32 = src_features
                .iter()
                .map(|x| {
                    let p = self.domain_discriminate(x).clamp(eps, 1.0 - eps);
                    -(1.0 - p).ln() // label = 0 (source)
                })
                .sum();
            let tgt_dom: f32 = tgt_features
                .iter()
                .map(|x| {
                    let p = self.domain_discriminate(x).clamp(eps, 1.0 - eps);
                    -p.ln() // label = 1 (target)
                })
                .sum();
            (src_dom + tgt_dom) / n_dom as f32
        };

        cls_loss + domain_loss
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. CORAL — Correlation Alignment
// ─────────────────────────────────────────────────────────────────────────────

/// Compute unbiased sample covariance matrix for a set of feature vectors.
/// Returns a d×d matrix.
pub fn compute_covariance(features: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = features.len();
    if n == 0 {
        return vec![];
    }
    let d = features[0].len();
    if d == 0 {
        return vec![];
    }

    // Compute mean
    let mut mean = vec![0.0f32; d];
    for x in features {
        for (i, &v) in x.iter().enumerate() {
            if i < d {
                mean[i] += v;
            }
        }
    }
    for m in &mut mean {
        *m /= n as f32;
    }

    // Accumulate (x - μ)(x - μ)^T
    let mut cov = vec![vec![0.0f32; d]; d];
    for x in features {
        let centered: Vec<f32> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| if i < d { v - mean[i] } else { 0.0 })
            .collect();
        for i in 0..d {
            for j in 0..d {
                cov[i][j] += centered[i] * centered[j];
            }
        }
    }

    let denom = if n > 1 { (n - 1) as f32 } else { 1.0 };
    for row in &mut cov {
        for v in row.iter_mut() {
            *v /= denom;
        }
    }
    cov
}

/// CORAL loss: (1/(4d²)) ‖C_S − C_T‖_F²
pub fn coral_loss(source: &[Vec<f32>], target: &[Vec<f32>]) -> f32 {
    if source.is_empty() || target.is_empty() {
        return 0.0;
    }
    let d = source[0].len();
    if d == 0 {
        return 0.0;
    }
    let cs = compute_covariance(source);
    let ct = compute_covariance(target);
    let frobenius_sq: f32 = cs
        .iter()
        .zip(ct.iter())
        .flat_map(|(rs, rt)| rs.iter().zip(rt.iter()).map(|(&a, &b)| (a - b).powi(2)))
        .sum();
    frobenius_sq / (4.0 * (d * d) as f32)
}

/// ZCA-like whitening: align source features to target covariance.
/// Returns whitened source features.
pub fn align_features(source: &[Vec<f32>], target: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if source.is_empty() {
        return vec![];
    }
    let d = source[0].len();
    if d == 0 {
        return source.to_vec();
    }

    // Compute per-feature std from target
    let target_stats = if target.is_empty() {
        vec![1.0f32; d]
    } else {
        let mut mean_t = vec![0.0f32; d];
        for x in target {
            for (i, &v) in x.iter().enumerate() {
                if i < d {
                    mean_t[i] += v;
                }
            }
        }
        for m in &mut mean_t {
            *m /= target.len() as f32;
        }
        let mut var_t = vec![0.0f32; d];
        for x in target {
            for (i, &v) in x.iter().enumerate() {
                if i < d {
                    var_t[i] += (v - mean_t[i]).powi(2);
                }
            }
        }
        let n = target.len();
        let denom = if n > 1 { (n - 1) as f32 } else { 1.0 };
        var_t.iter().map(|&v| (v / denom + 1e-8).sqrt()).collect()
    };

    // Compute per-feature std from source
    let n_src = source.len();
    let mut mean_s = vec![0.0f32; d];
    for x in source {
        for (i, &v) in x.iter().enumerate() {
            if i < d {
                mean_s[i] += v;
            }
        }
    }
    for m in &mut mean_s {
        *m /= n_src as f32;
    }
    let mut var_s = vec![0.0f32; d];
    for x in source {
        for (i, &v) in x.iter().enumerate() {
            if i < d {
                var_s[i] += (v - mean_s[i]).powi(2);
            }
        }
    }
    let denom_s = if n_src > 1 { (n_src - 1) as f32 } else { 1.0 };
    let std_s: Vec<f32> = var_s.iter().map(|&v| (v / denom_s + 1e-8).sqrt()).collect();

    // ZCA-like: standardize source, then rescale to target scale
    source
        .iter()
        .map(|x| {
            x.iter()
                .enumerate()
                .map(|(i, &v)| {
                    if i < d {
                        let normalized = (v - mean_s[i]) / std_s[i];
                        normalized * target_stats[i]
                    } else {
                        v
                    }
                })
                .collect()
        })
        .collect()
}

/// Persistent CORAL adapter storing fitted covariance matrices.
#[derive(Clone, Debug)]
pub struct CoralAdapter {
    pub source_cov: Vec<Vec<f32>>,
    pub target_cov: Vec<Vec<f32>>,
    pub dim: usize,
}

impl CoralAdapter {
    pub fn fit(source: &[Vec<f32>], target: &[Vec<f32>]) -> Self {
        let dim = source.first().map(|x| x.len()).unwrap_or(0);
        let source_cov = compute_covariance(source);
        let target_cov = compute_covariance(target);
        Self {
            source_cov,
            target_cov,
            dim,
        }
    }

    /// Transform a source feature vector using the fitted covariance statistics.
    pub fn transform(&self, x: &[f32]) -> Vec<f32> {
        // Per-feature ZCA-like: use diagonal of source_cov and target_cov
        if self.dim == 0 {
            return x.to_vec();
        }
        let src_std: Vec<f32> = (0..self.dim)
            .map(|i| {
                self.source_cov
                    .get(i)
                    .and_then(|r| r.get(i))
                    .copied()
                    .unwrap_or(1.0)
                    .max(0.0)
                    .sqrt()
                    + 1e-8
            })
            .collect();
        let tgt_std: Vec<f32> = (0..self.dim)
            .map(|i| {
                self.target_cov
                    .get(i)
                    .and_then(|r| r.get(i))
                    .copied()
                    .unwrap_or(1.0)
                    .max(0.0)
                    .sqrt()
                    + 1e-8
            })
            .collect();
        x.iter()
            .enumerate()
            .map(|(i, &v)| {
                if i < self.dim {
                    v / src_std[i] * tgt_std[i]
                } else {
                    v
                }
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. MMD — Maximum Mean Discrepancy
// ─────────────────────────────────────────────────────────────────────────────

/// Radial Basis Function (Gaussian) kernel for MMD computations.
#[derive(Clone, Debug)]
pub struct DaRbfKernel {
    pub bandwidth: f32,
}

impl DaRbfKernel {
    pub fn new(bandwidth: f32) -> Self {
        Self { bandwidth }
    }

    /// Evaluate k(x, y) = exp(-‖x−y‖² / (2σ²)).
    pub fn compute(&self, x: &[f32], y: &[f32]) -> f32 {
        let sq_dist: f32 = x.iter().zip(y.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
        let denom = 2.0 * self.bandwidth * self.bandwidth;
        if denom == 0.0 {
            return 0.0;
        }
        (-sq_dist / denom).exp()
    }

    /// Estimate bandwidth as the median pairwise squared distance.
    pub fn median_bandwidth(samples: &[Vec<f32>]) -> f32 {
        let n = samples.len();
        if n < 2 {
            return 1.0;
        }
        let mut dists = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let d: f32 = samples[i]
                    .iter()
                    .zip(samples[j].iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum();
                dists.push(d);
            }
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = dists[dists.len() / 2];
        (med / 2.0).sqrt().max(1e-6)
    }
}

/// Compute the biased MMD² estimator:
/// E[k(xs,xs)] − 2·E[k(xs,xt)] + E[k(xt,xt)]
pub fn mmd_loss(source: &[Vec<f32>], target: &[Vec<f32>], kernel: &DaRbfKernel) -> f32 {
    let ns = source.len();
    let nt = target.len();
    if ns == 0 || nt == 0 {
        return 0.0;
    }

    // E[k(xs, xs)]
    let kss: f32 = (0..ns)
        .flat_map(|i| (0..ns).map(move |j| (i, j)))
        .map(|(i, j)| kernel.compute(&source[i], &source[j]))
        .sum::<f32>()
        / (ns * ns) as f32;

    // E[k(xt, xt)]
    let ktt: f32 = (0..nt)
        .flat_map(|i| (0..nt).map(move |j| (i, j)))
        .map(|(i, j)| kernel.compute(&target[i], &target[j]))
        .sum::<f32>()
        / (nt * nt) as f32;

    // E[k(xs, xt)]
    let kst: f32 = (0..ns)
        .flat_map(|i| (0..nt).map(move |j| (i, j)))
        .map(|(i, j)| kernel.compute(&source[i], &target[j]))
        .sum::<f32>()
        / (ns * nt) as f32;

    (kss - 2.0 * kst + ktt).max(0.0)
}

/// Multi-kernel MMD: sum MMD values over multiple RBF kernels with different bandwidths.
pub fn multi_kernel_mmd(source: &[Vec<f32>], target: &[Vec<f32>], bandwidths: &[f32]) -> f32 {
    bandwidths
        .iter()
        .map(|&bw| {
            let kernel = DaRbfKernel::new(bw);
            mmd_loss(source, target, &kernel)
        })
        .sum()
}

/// MMD-based domain adaptation adapter.
#[derive(Clone, Debug)]
pub struct MmdAdapter {
    pub feature_dim: usize,
    pub kernel: DaRbfKernel,
}

impl MmdAdapter {
    pub fn new(feature_dim: usize, bandwidth: f32) -> Self {
        Self {
            feature_dim,
            kernel: DaRbfKernel::new(bandwidth),
        }
    }

    /// Compute the MMD adaptation loss between source and target feature batches.
    pub fn adaptation_loss(&self, source: &[Vec<f32>], target: &[Vec<f32>]) -> f32 {
        mmd_loss(source, target, &self.kernel)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Domain Generalization
// ─────────────────────────────────────────────────────────────────────────────

/// Invariant Risk Minimization (Arjovsky et al. 2019).
pub struct IrmTrainer;

impl IrmTrainer {
    /// Compute the IRM penalty as the variance of per-domain losses.
    ///
    /// Uses a simplified form: Var_d\[mean_loss_d\].
    pub fn irm_penalty(logits: &[Vec<f32>], labels: &[usize], domain_ids: &[usize]) -> f32 {
        if logits.is_empty() {
            return 0.0;
        }
        let eps = 1e-10f32;
        // Collect unique domain IDs
        let mut domains: Vec<usize> = domain_ids.to_vec();
        domains.sort_unstable();
        domains.dedup();
        if domains.len() < 2 {
            return 0.0;
        }

        // Compute per-domain mean cross-entropy
        let per_domain_losses: Vec<f32> = domains
            .iter()
            .map(|&dom| {
                let indices: Vec<usize> = domain_ids
                    .iter()
                    .enumerate()
                    .filter(|(_, &d)| d == dom)
                    .map(|(i, _)| i)
                    .collect();
                if indices.is_empty() {
                    return 0.0;
                }
                let loss: f32 = indices
                    .iter()
                    .map(|&i| {
                        let probs = softmax(&logits[i]);
                        let y = labels[i];
                        let p = probs.get(y).copied().unwrap_or(eps).max(eps);
                        -p.ln()
                    })
                    .sum::<f32>()
                    / indices.len() as f32;
                loss
            })
            .collect();

        // Variance of per-domain losses
        let n = per_domain_losses.len() as f32;
        let mean = per_domain_losses.iter().sum::<f32>() / n;
        per_domain_losses
            .iter()
            .map(|&l| (l - mean).powi(2))
            .sum::<f32>()
            / n
    }

    /// IRM total loss: task_loss + lambda * irm_penalty.
    pub fn irm_loss(task_loss: f32, irm_penalty: f32, lambda: f32) -> f32 {
        task_loss + lambda * irm_penalty
    }
}

/// Domain Mixup augmentation.
pub struct MixupDomain;

impl MixupDomain {
    /// Linearly interpolate two feature vectors:  λ·x1 + (1−λ)·x2.
    /// Returns (mixed_features, lambda) where lambda ∼ Beta(alpha, alpha) ≈ Uniform here.
    pub fn mixup(x1: &[f32], x2: &[f32], alpha: f32, rng: &mut StdRng) -> (Vec<f32>, f32) {
        // Approximate Beta(alpha, alpha) via a ratio of Gamma samples.
        // For simplicity and no-ndarray policy, use a simple uniform when alpha >= 1,
        // or a beta approximation using the ratio-of-uniforms method.
        let lambda = if alpha <= 0.0 {
            0.5
        } else {
            // Simple Beta(alpha, alpha) approximation via Johnk's method
            let u1: f32 = rng.random::<f32>();
            let u2: f32 = rng.random::<f32>();
            // Use the ratio u1^(1/alpha) / (u1^(1/alpha) + u2^(1/alpha))
            let a1 = u1.powf(1.0 / alpha);
            let a2 = u2.powf(1.0 / alpha);
            let s = a1 + a2;
            if s > 0.0 {
                (a1 / s).clamp(0.0, 1.0)
            } else {
                0.5
            }
        };
        let mixed: Vec<f32> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| lambda * a + (1.0 - lambda) * b)
            .collect();
        (mixed, lambda)
    }

    /// Create mixed samples by randomly pairing within and across domains.
    pub fn domain_mixup_batch(
        domain_samples: &[Vec<Vec<f32>>],
        alpha: f32,
        rng: &mut StdRng,
    ) -> Vec<Vec<f32>> {
        let n_domains = domain_samples.len();
        if n_domains == 0 {
            return vec![];
        }
        let mut result = Vec::new();
        for d in 0..n_domains {
            let other_d = rng.random_range(0..n_domains);
            let samples_d = &domain_samples[d];
            let samples_other = &domain_samples[other_d];
            if samples_d.is_empty() || samples_other.is_empty() {
                continue;
            }
            for x1 in samples_d {
                let idx = rng.random_range(0..samples_other.len());
                let x2 = &samples_other[idx];
                let (mixed, _) = Self::mixup(x1, x2, alpha, rng);
                result.push(mixed);
            }
        }
        result
    }
}

/// Feature statistics-based style transfer using AdaIN.
pub struct StyleTransferDa;

impl StyleTransferDa {
    /// Adaptive Instance Normalization: normalize content, rescale with style statistics.
    pub fn adain(content: &[f32], style_mean: &[f32], style_std: &[f32]) -> Vec<f32> {
        let n = content.len();
        if n == 0 {
            return vec![];
        }
        // Compute content statistics
        let mean_c: f32 = content.iter().sum::<f32>() / n as f32;
        let var_c: f32 = content.iter().map(|&v| (v - mean_c).powi(2)).sum::<f32>() / n as f32;
        let std_c = (var_c + 1e-8).sqrt();
        content
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let normalized = (v - mean_c) / std_c;
                let sm = style_mean.get(i).copied().unwrap_or(0.0);
                let ss = style_std.get(i).copied().unwrap_or(1.0).max(1e-8);
                normalized * ss + sm
            })
            .collect()
    }

    /// Transfer style from target_stats to all source feature vectors.
    pub fn domain_style_transfer(source: &[Vec<f32>], target_stats: &DomainStats) -> Vec<Vec<f32>> {
        source
            .iter()
            .map(|x| Self::adain(x, &target_stats.mean, &target_stats.std))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Test-Time Adaptation (TENT)
// ─────────────────────────────────────────────────────────────────────────────

/// Entropy minimization at test time.
pub struct EntropyMinimization;

impl EntropyMinimization {
    /// Compute Shannon entropy of a softmax distribution: −∑ p·log(p).
    pub fn prediction_entropy(logits: &[f32]) -> f32 {
        let probs = softmax(logits);
        let eps = 1e-10f32;
        probs
            .iter()
            .map(|&p| if p > eps { -p * p.ln() } else { 0.0 })
            .sum()
    }

    /// Mean entropy over a batch of logit vectors.
    pub fn batch_entropy(batch_logits: &[Vec<f32>]) -> f32 {
        if batch_logits.is_empty() {
            return 0.0;
        }
        batch_logits
            .iter()
            .map(|l| Self::prediction_entropy(l))
            .sum::<f32>()
            / batch_logits.len() as f32
    }
}

/// Running batch-norm statistics for test-time adaptation.
#[derive(Clone, Debug)]
pub struct BatchNormStats {
    pub running_mean: Vec<f32>,
    pub running_var: Vec<f32>,
}

impl BatchNormStats {
    pub fn new(dim: usize) -> Self {
        Self {
            running_mean: vec![0.0f32; dim],
            running_var: vec![1.0f32; dim],
        }
    }
}

/// Update batch-norm statistics using exponential moving average.
pub fn update_bn_stats(
    new_mean: &[f32],
    new_var: &[f32],
    momentum: f32,
    stats: &mut BatchNormStats,
) {
    for (rm, &nm) in stats.running_mean.iter_mut().zip(new_mean.iter()) {
        *rm = (1.0 - momentum) * *rm + momentum * nm;
    }
    for (rv, &nv) in stats.running_var.iter_mut().zip(new_var.iter()) {
        *rv = (1.0 - momentum) * *rv + momentum * nv;
    }
}

/// TENT adapter combining entropy minimization with BN stat updates.
#[derive(Clone, Debug)]
pub struct TentAdapter {
    pub lr: f32,
    pub momentum: f32,
    pub bn_stats: BatchNormStats,
}

impl TentAdapter {
    pub fn new(feature_dim: usize, lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            bn_stats: BatchNormStats::new(feature_dim),
        }
    }
}

/// Test-time pseudo-labeling utilities.
pub struct TestTimePseudoLabeling;

impl TestTimePseudoLabeling {
    /// Assign pseudo-labels only where max probability exceeds the threshold.
    pub fn generate_pseudo_labels(
        logits: &[Vec<f32>],
        confidence_threshold: f32,
    ) -> Vec<Option<usize>> {
        logits
            .iter()
            .map(|l| {
                let probs = softmax(l);
                let (argmax, max_p) =
                    probs
                        .iter()
                        .enumerate()
                        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                            if v > bv {
                                (i, v)
                            } else {
                                (bi, bv)
                            }
                        });
                if max_p >= confidence_threshold {
                    Some(argmax)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Cross-entropy loss using pseudo-labels (skip None entries).
    pub fn pseudo_label_loss(logits: &[Vec<f32>], pseudo_labels: &[Option<usize>]) -> f32 {
        let eps = 1e-10f32;
        let valid: Vec<(usize, usize)> = logits
            .iter()
            .zip(pseudo_labels.iter())
            .enumerate()
            .filter_map(|(i, (_, label))| label.map(|l| (i, l)))
            .collect();
        if valid.is_empty() {
            return 0.0;
        }
        valid
            .iter()
            .map(|&(i, y)| {
                let probs = softmax(&logits[i]);
                let p = probs.get(y).copied().unwrap_or(eps).max(eps);
                -p.ln()
            })
            .sum::<f32>()
            / valid.len() as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. DA Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Train a simple linear classifier to distinguish source from target domains;
/// proxy-A-distance = 2(1 − 2ε) where ε is the leave-out error rate.
pub fn proxy_a_distance(source: &[Vec<f32>], target: &[Vec<f32>]) -> f32 {
    let n_src = source.len();
    let n_tgt = target.len();
    let n_total = n_src + n_tgt;
    if n_total == 0 {
        return 0.0;
    }
    let dim = source
        .first()
        .or_else(|| target.first())
        .map(|x| x.len())
        .unwrap_or(0);
    if dim == 0 {
        return 0.0;
    }

    // Initialise weight vector (all zeros = trivially 50% error for balanced data)
    let mut w = vec![0.0f32; dim];
    let mut b = 0.0f32;
    let lr = 0.01f32;
    let n_steps = 200;

    // Combine samples: source → label 0, target → label 1
    let all_x: Vec<&Vec<f32>> = source.iter().chain(target.iter()).collect();
    let all_y: Vec<f32> = (0..n_src)
        .map(|_| 0.0f32)
        .chain((0..n_tgt).map(|_| 1.0f32))
        .collect();

    // Logistic regression via gradient descent
    for _ in 0..n_steps {
        let mut dw = vec![0.0f32; dim];
        let mut db = 0.0f32;
        for (x, &y) in all_x.iter().zip(all_y.iter()) {
            let logit: f32 = w
                .iter()
                .zip(x.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>()
                + b;
            let pred = sigmoid(logit);
            let err = pred - y;
            for (dw_i, &xi) in dw.iter_mut().zip(x.iter()) {
                *dw_i += err * xi;
            }
            db += err;
        }
        let scale = lr / n_total as f32;
        for (wi, &dwi) in w.iter_mut().zip(dw.iter()) {
            *wi -= scale * dwi;
        }
        b -= scale * db;
    }

    // Estimate error rate
    let errors: usize = all_x
        .iter()
        .zip(all_y.iter())
        .filter(|(&x, &y)| {
            let logit: f32 = w
                .iter()
                .zip(x.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>()
                + b;
            let pred = if sigmoid(logit) >= 0.5 { 1.0 } else { 0.0 };
            (pred - y).abs() > 0.5
        })
        .count();
    let error_rate = errors as f32 / n_total as f32;
    // Clamp to valid range before computing PAD
    let pad = 2.0 * (1.0 - 2.0 * error_rate);
    pad.clamp(0.0, 2.0)
}

/// Fréchet-like domain gap: Euclidean distance between domain means.
pub fn domain_gap(source: &[Vec<f32>], target: &[Vec<f32>]) -> f32 {
    let n_src = source.len();
    let n_tgt = target.len();
    if n_src == 0 || n_tgt == 0 {
        return 0.0;
    }
    let dim = source[0].len();
    let mut mean_s = vec![0.0f32; dim];
    let mut mean_t = vec![0.0f32; dim];
    for x in source {
        for (i, &v) in x.iter().enumerate() {
            if i < dim {
                mean_s[i] += v;
            }
        }
    }
    for x in target {
        for (i, &v) in x.iter().enumerate() {
            if i < dim {
                mean_t[i] += v;
            }
        }
    }
    for m in &mut mean_s {
        *m /= n_src as f32;
    }
    for m in &mut mean_t {
        *m /= n_tgt as f32;
    }
    mean_s
        .iter()
        .zip(mean_t.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Conditional domain shift for a specific class: mean-feature Euclidean distance.
pub fn class_conditional_shift(
    source: &DomainDataset,
    target: &DomainDataset,
    class_id: usize,
) -> f32 {
    let src_feats: Vec<Vec<f32>> = source
        .samples
        .iter()
        .filter(|s| s.label == Some(class_id) && s.domain_id == 0)
        .map(|s| s.features.clone())
        .collect();
    let tgt_feats: Vec<Vec<f32>> = target
        .samples
        .iter()
        .filter(|s| s.label == Some(class_id) && s.domain_id != 0)
        .map(|s| s.features.clone())
        .collect();
    domain_gap(&src_feats, &tgt_feats)
}

/// Evaluation report for domain adaptation.
#[derive(Clone, Debug)]
pub struct DaEvalReport {
    pub source_accuracy: f32,
    pub target_accuracy: f32,
    pub mmd: f32,
    pub proxy_ad: f32,
    pub domain_gap: f32,
}

/// Evaluate a DANN model on a dataset, computing DA metrics.
pub fn da_evaluate(model: &DannModel, test_data: &DomainDataset) -> DaEvalReport {
    let src_samples = test_data.source_samples();
    let tgt_samples = test_data.target_samples();

    // Source accuracy
    let source_accuracy = if src_samples.is_empty() {
        0.0
    } else {
        let correct = src_samples
            .iter()
            .filter(|s| {
                if let Some(label) = s.label {
                    let probs = model.classify(&s.features);
                    let pred = probs
                        .iter()
                        .enumerate()
                        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                            if v > bv {
                                (i, v)
                            } else {
                                (bi, bv)
                            }
                        })
                        .0;
                    pred == label
                } else {
                    false
                }
            })
            .count();
        correct as f32 / src_samples.len() as f32
    };

    // Target accuracy (if labels available)
    let target_accuracy = if tgt_samples.is_empty() {
        0.0
    } else {
        let labelled: Vec<&&DomainSample> =
            tgt_samples.iter().filter(|s| s.label.is_some()).collect();
        if labelled.is_empty() {
            0.0
        } else {
            let correct = labelled
                .iter()
                .filter(|s| {
                    if let Some(label) = s.label {
                        let probs = model.classify(&s.features);
                        let pred = probs
                            .iter()
                            .enumerate()
                            .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                                if v > bv {
                                    (i, v)
                                } else {
                                    (bi, bv)
                                }
                            })
                            .0;
                        pred == label
                    } else {
                        false
                    }
                })
                .count();
            correct as f32 / labelled.len() as f32
        }
    };

    let src_feats: Vec<Vec<f32>> = src_samples.iter().map(|s| s.features.clone()).collect();
    let tgt_feats: Vec<Vec<f32>> = tgt_samples.iter().map(|s| s.features.clone()).collect();

    let kernel = DaRbfKernel::new(1.0);
    let mmd_val = if src_feats.is_empty() || tgt_feats.is_empty() {
        0.0
    } else {
        mmd_loss(&src_feats, &tgt_feats, &kernel)
    };
    let pad = proxy_a_distance(&src_feats, &tgt_feats);
    let gap = domain_gap(&src_feats, &tgt_feats);

    DaEvalReport {
        source_accuracy,
        target_accuracy,
        mmd: mmd_val,
        proxy_ad: pad,
        domain_gap: gap,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    fn sample_dataset() -> DomainDataset {
        let mut samples = Vec::new();
        for i in 0..10 {
            samples.push(DomainSample {
                features: vec![i as f32, (i * 2) as f32, (i * 3) as f32],
                label: Some(i % 3),
                domain_id: 0,
            });
        }
        for i in 0..5 {
            samples.push(DomainSample {
                features: vec![10.0 + i as f32, 20.0 + i as f32, 30.0 + i as f32],
                label: Some(i % 3),
                domain_id: 1,
            });
        }
        DomainDataset::new(samples, 2, 3)
    }

    // ─── Domain Data ───────────────────────────────────────────────────────────

    #[test]
    fn test_domain_dataset_source_samples() {
        let ds = sample_dataset();
        let src = ds.source_samples();
        assert_eq!(src.len(), 10);
        for s in &src {
            assert_eq!(s.domain_id, 0);
        }
    }

    #[test]
    fn test_domain_dataset_target_samples() {
        let ds = sample_dataset();
        let tgt = ds.target_samples();
        assert_eq!(tgt.len(), 5);
        for t in &tgt {
            assert_ne!(t.domain_id, 0);
        }
    }

    #[test]
    fn test_domain_statistics_mean() {
        let ds = sample_dataset();
        let stats = domain_statistics(&ds, 0);
        assert_eq!(stats.n_samples, 10);
        // Feature 0: mean of 0..9 = 4.5
        let expected_mean0 = (0..10).map(|i| i as f32).sum::<f32>() / 10.0;
        assert!((stats.mean[0] - expected_mean0).abs() < 1e-4);
    }

    #[test]
    fn test_domain_statistics_std() {
        let ds = sample_dataset();
        let stats = domain_statistics(&ds, 0);
        assert!(stats.std[0] > 0.0);
        assert_eq!(stats.std.len(), 3);
    }

    // ─── GRL ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_gradient_reversal_identity_forward() {
        let grl = GradientReversalLayer::new(0.5);
        let x = vec![1.0, -2.0, 3.0];
        let out = grl.forward(&x);
        assert_eq!(out, x);
    }

    #[test]
    fn test_gradient_reversal_scale() {
        let grl = GradientReversalLayer::new(0.3);
        assert!((grl.backward_scale() + 0.3).abs() < 1e-6);
    }

    // ─── Feature Extractor ─────────────────────────────────────────────────────

    #[test]
    fn test_feature_extractor_forward_shape() {
        let mut rng = make_rng(1);
        let fe = DaFeatureExtractor::new(8, 16, 4, &mut rng);
        let x = vec![1.0f32; 8];
        let out = fe.forward(&x);
        assert_eq!(out.len(), 4);
    }

    // ─── Domain Classifier ─────────────────────────────────────────────────────

    #[test]
    fn test_domain_classifier_output_range() {
        let mut rng = make_rng(2);
        let dc = DannDomainClassifier::new(8, &mut rng);
        for seed in 0..20 {
            let x: Vec<f32> = (0..8).map(|i| (i + seed) as f32 * 0.1).collect();
            let p = dc.forward(&x);
            assert!((0.0..=1.0).contains(&p), "p={p} out of range");
        }
    }

    // ─── Label Predictor ───────────────────────────────────────────────────────

    #[test]
    fn test_label_predictor_shape() {
        let mut rng = make_rng(3);
        let lp = DaLabelPredictor::new(8, 5, &mut rng);
        let x = vec![0.5f32; 8];
        let out = lp.forward(&x);
        assert_eq!(out.len(), 5);
        let sum: f32 = out.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "probabilities must sum to 1, got {sum}"
        );
    }

    // ─── DANN Model ────────────────────────────────────────────────────────────

    #[test]
    fn test_dann_model_creation() {
        let mut rng = make_rng(4);
        let _model = DannModel::new(10, 8, 3, 0.5, &mut rng);
    }

    #[test]
    fn test_dann_classify_shape() {
        let mut rng = make_rng(5);
        let model = DannModel::new(10, 8, 3, 0.5, &mut rng);
        let x = vec![0.1f32; 10];
        let probs = model.classify(&x);
        assert_eq!(probs.len(), 3);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_dann_domain_discriminate_range() {
        let mut rng = make_rng(6);
        let model = DannModel::new(10, 8, 3, 0.5, &mut rng);
        for i in 0..10 {
            let x: Vec<f32> = (0..10).map(|j| (i * 10 + j) as f32 * 0.01).collect();
            let p = model.domain_discriminate(&x);
            assert!((0.0..=1.0).contains(&p), "p={p}");
        }
    }

    #[test]
    fn test_dann_loss_finite() {
        let mut rng = make_rng(7);
        let model = DannModel::new(4, 4, 2, 1.0, &mut rng);
        let src_feats: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32 * 0.1; 4]).collect();
        let src_labels: Vec<usize> = vec![0, 1, 0, 1, 0];
        let tgt_feats: Vec<Vec<f32>> = (0..3).map(|i| vec![i as f32 * 0.2 + 1.0; 4]).collect();
        let loss = model.dann_loss(&src_feats, &src_labels, &tgt_feats);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_dann_loss_nonneg() {
        let mut rng = make_rng(8);
        let model = DannModel::new(4, 4, 2, 1.0, &mut rng);
        let src_feats: Vec<Vec<f32>> = vec![vec![0.5; 4], vec![0.2; 4]];
        let src_labels: Vec<usize> = vec![0, 1];
        let tgt_feats: Vec<Vec<f32>> = vec![vec![0.8; 4]];
        let loss = model.dann_loss(&src_feats, &src_labels, &tgt_feats);
        assert!(loss >= 0.0);
    }

    // ─── Covariance ────────────────────────────────────────────────────────────

    #[test]
    fn test_compute_covariance_shape() {
        let samples: Vec<Vec<f32>> = (0..6).map(|i| vec![i as f32, (i * 2) as f32]).collect();
        let cov = compute_covariance(&samples);
        assert_eq!(cov.len(), 2);
        assert_eq!(cov[0].len(), 2);
    }

    #[test]
    fn test_compute_covariance_symmetric() {
        let samples: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![i as f32, i as f32 * 1.5, i as f32 + 2.0])
            .collect();
        let cov = compute_covariance(&samples);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (cov[i][j] - cov[j][i]).abs() < 1e-4,
                    "cov not symmetric at [{i}][{j}]"
                );
            }
        }
    }

    // ─── CORAL ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_coral_loss_zero_same_distribution() {
        let samples: Vec<Vec<f32>> = (0..8)
            .map(|i| vec![i as f32 * 0.5, (8 - i) as f32 * 0.3])
            .collect();
        let loss = coral_loss(&samples, &samples);
        assert!(
            loss < 1e-6,
            "CORAL loss for identical distributions should be ~0, got {loss}"
        );
    }

    #[test]
    fn test_coral_loss_positive_different() {
        let src: Vec<Vec<f32>> = (0..8).map(|i| vec![i as f32, 0.0]).collect();
        let tgt: Vec<Vec<f32>> = (0..8).map(|i| vec![0.0, i as f32]).collect();
        let loss = coral_loss(&src, &tgt);
        assert!(loss > 0.0);
    }

    // ─── CORAL Adapter ─────────────────────────────────────────────────────────

    #[test]
    fn test_coral_adapter_transform_shape() {
        let src: Vec<Vec<f32>> = (0..8).map(|i| vec![i as f32; 4]).collect();
        let tgt: Vec<Vec<f32>> = (0..8).map(|i| vec![(i + 2) as f32; 4]).collect();
        let adapter = CoralAdapter::fit(&src, &tgt);
        let out = adapter.transform(&src[0]);
        assert_eq!(out.len(), 4);
    }

    // ─── RBF Kernel ────────────────────────────────────────────────────────────

    #[test]
    fn test_rbf_kernel_same_point() {
        let k = DaRbfKernel::new(1.0);
        let x = vec![1.0, 2.0, 3.0];
        let v = k.compute(&x, &x);
        assert!((v - 1.0).abs() < 1e-6, "k(x,x) should be 1.0, got {v}");
    }

    #[test]
    fn test_rbf_kernel_decreases_with_distance() {
        let k = DaRbfKernel::new(1.0);
        let x = vec![0.0; 3];
        let y1 = vec![1.0, 0.0, 0.0];
        let y2 = vec![2.0, 0.0, 0.0];
        let k1 = k.compute(&x, &y1);
        let k2 = k.compute(&x, &y2);
        assert!(
            k1 > k2,
            "kernel should decrease with distance: k1={k1} k2={k2}"
        );
    }

    #[test]
    fn test_median_bandwidth_positive() {
        let samples: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32; 3]).collect();
        let bw = DaRbfKernel::median_bandwidth(&samples);
        assert!(bw > 0.0);
    }

    // ─── MMD ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_mmd_loss_zero_same() {
        let samples: Vec<Vec<f32>> = (0..6).map(|i| vec![i as f32 * 0.1; 3]).collect();
        let k = DaRbfKernel::new(1.0);
        let loss = mmd_loss(&samples, &samples, &k);
        assert!(
            loss < 1e-6,
            "MMD of identical distributions should be ~0, got {loss}"
        );
    }

    #[test]
    fn test_mmd_loss_positive_different() {
        let src: Vec<Vec<f32>> = (0..6).map(|_| vec![0.0; 3]).collect();
        let tgt: Vec<Vec<f32>> = (0..6).map(|_| vec![10.0; 3]).collect();
        let k = DaRbfKernel::new(1.0);
        let loss = mmd_loss(&src, &tgt, &k);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_multi_kernel_mmd_nonneg() {
        let src: Vec<Vec<f32>> = (0..4).map(|i| vec![i as f32; 2]).collect();
        let tgt: Vec<Vec<f32>> = (0..4).map(|i| vec![(i + 5) as f32; 2]).collect();
        let bws = [0.5, 1.0, 2.0];
        let val = multi_kernel_mmd(&src, &tgt, &bws);
        assert!(val >= 0.0);
    }

    #[test]
    fn test_mmd_adaptation_loss() {
        let adapter = MmdAdapter::new(3, 1.0);
        let src: Vec<Vec<f32>> = vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]];
        let tgt: Vec<Vec<f32>> = vec![vec![10.0, 11.0, 12.0], vec![13.0, 14.0, 15.0]];
        let loss = adapter.adaptation_loss(&src, &tgt);
        assert!(loss >= 0.0 && loss.is_finite());
    }

    // ─── IRM ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_irm_penalty_finite() {
        let logits = vec![
            vec![2.0, 0.5, 0.1],
            vec![0.1, 2.0, 0.3],
            vec![0.2, 0.1, 2.0],
        ];
        let labels = vec![0, 1, 2];
        let domains = vec![0, 0, 1];
        let penalty = IrmTrainer::irm_penalty(&logits, &labels, &domains);
        assert!(penalty.is_finite());
    }

    #[test]
    fn test_irm_loss_finite() {
        let loss = IrmTrainer::irm_loss(1.5, 0.3, 2.0);
        assert!((loss - (1.5 + 2.0 * 0.3)).abs() < 1e-5);
    }

    // ─── Mixup ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_mixup_interpolation() {
        let mut rng = make_rng(42);
        let x1 = vec![1.0f32; 4];
        let x2 = vec![3.0f32; 4];
        let (_mixed, lambda) = MixupDomain::mixup(&x1, &x2, 1.0, &mut rng);
        assert!((0.0..=1.0).contains(&lambda), "lambda={lambda}");
    }

    #[test]
    fn test_mixup_shape_preserved() {
        let mut rng = make_rng(43);
        let x1 = vec![1.0f32; 6];
        let x2 = vec![2.0f32; 6];
        let (mixed, _) = MixupDomain::mixup(&x1, &x2, 0.4, &mut rng);
        assert_eq!(mixed.len(), 6);
    }

    #[test]
    fn test_domain_mixup_batch_shape() {
        let mut rng = make_rng(44);
        let domain_samples: Vec<Vec<Vec<f32>>> = vec![
            (0..4).map(|i| vec![i as f32; 3]).collect(),
            (0..4).map(|i| vec![(i + 10) as f32; 3]).collect(),
        ];
        let mixed = MixupDomain::domain_mixup_batch(&domain_samples, 0.4, &mut rng);
        assert!(!mixed.is_empty());
        for m in &mixed {
            assert_eq!(m.len(), 3);
        }
    }

    // ─── AdaIN ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_adain_shape() {
        let content = vec![1.0f32, 2.0, 3.0, 4.0];
        let style_mean = vec![0.0f32, 0.0, 0.0, 0.0];
        let style_std = vec![1.0f32, 1.0, 1.0, 1.0];
        let out = StyleTransferDa::adain(&content, &style_mean, &style_std);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_adain_stats_match_target() {
        let content = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let style_mean = vec![5.0f32; 8];
        let style_std = vec![2.0f32; 8];
        let out = StyleTransferDa::adain(&content, &style_mean, &style_std);
        // The output mean should be close to style_mean
        let out_mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!((out_mean - 5.0).abs() < 0.5, "out_mean={out_mean}");
    }

    #[test]
    fn test_domain_style_transfer_shape() {
        let ds = sample_dataset();
        let stats = domain_statistics(&ds, 1);
        let src_feats: Vec<Vec<f32>> = ds
            .source_samples()
            .iter()
            .map(|s| s.features.clone())
            .collect();
        let out = StyleTransferDa::domain_style_transfer(&src_feats, &stats);
        assert_eq!(out.len(), src_feats.len());
        for row in &out {
            assert_eq!(row.len(), 3);
        }
    }

    // ─── Entropy ───────────────────────────────────────────────────────────────

    #[test]
    fn test_entropy_zero_certain() {
        // Near one-hot logits → near zero entropy
        let logits = vec![100.0f32, 0.0, 0.0];
        let h = EntropyMinimization::prediction_entropy(&logits);
        assert!(h < 0.01, "entropy of near one-hot should be ~0, got {h}");
    }

    #[test]
    fn test_entropy_max_uniform() {
        // Uniform logits → maximum entropy
        let logits = vec![0.0f32; 4];
        let h = EntropyMinimization::prediction_entropy(&logits);
        let max_h = (4.0f32).ln();
        assert!(
            (h - max_h).abs() < 1e-4,
            "uniform entropy should be ln(4)={max_h}, got {h}"
        );
    }

    #[test]
    fn test_batch_entropy_average() {
        let batch = vec![vec![0.0f32; 3], vec![0.0f32; 3]];
        let h = EntropyMinimization::batch_entropy(&batch);
        let expected = EntropyMinimization::prediction_entropy(&[0.0f32; 3]);
        assert!((h - expected).abs() < 1e-5);
    }

    // ─── BatchNorm Update ──────────────────────────────────────────────────────

    #[test]
    fn test_update_bn_stats() {
        let mut stats = BatchNormStats::new(3);
        let new_mean = vec![2.0f32, 4.0, 6.0];
        let new_var = vec![0.5f32, 1.5, 2.5];
        update_bn_stats(&new_mean, &new_var, 0.1, &mut stats);
        assert!((stats.running_mean[0] - 0.2).abs() < 1e-5);
        assert!((stats.running_var[0] - (0.9 + 0.05)).abs() < 1e-5);
    }

    // ─── Pseudo Labels ─────────────────────────────────────────────────────────

    #[test]
    fn test_pseudo_labels_confident() {
        let logits = vec![vec![10.0f32, 0.0, 0.0]]; // near-certain class 0
        let labels = TestTimePseudoLabeling::generate_pseudo_labels(&logits, 0.5);
        assert_eq!(labels[0], Some(0));
    }

    #[test]
    fn test_pseudo_labels_uncertain() {
        let logits = vec![vec![0.0f32, 0.0, 0.0]]; // uniform → low confidence
        let labels = TestTimePseudoLabeling::generate_pseudo_labels(&logits, 0.9);
        assert_eq!(labels[0], None);
    }

    #[test]
    fn test_pseudo_label_loss_finite() {
        let logits = vec![vec![2.0f32, 0.5, 0.1], vec![0.1, 2.0, 0.3]];
        let pseudo_labels = vec![Some(0), Some(1)];
        let loss = TestTimePseudoLabeling::pseudo_label_loss(&logits, &pseudo_labels);
        assert!(loss.is_finite() && loss >= 0.0);
    }

    // ─── DA Metrics ────────────────────────────────────────────────────────────

    #[test]
    fn test_proxy_a_distance_range() {
        let src: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32; 4]).collect();
        let tgt: Vec<Vec<f32>> = (0..10).map(|i| vec![(i + 20) as f32; 4]).collect();
        let pad = proxy_a_distance(&src, &tgt);
        assert!((0.0..=2.0).contains(&pad), "PAD={pad} outside [0,2]");
    }

    #[test]
    fn test_domain_gap_zero_same() {
        let samples: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32; 3]).collect();
        let gap = domain_gap(&samples, &samples);
        assert!(
            gap < 1e-5,
            "gap of identical distributions should be ~0, got {gap}"
        );
    }

    #[test]
    fn test_domain_gap_positive_different() {
        let src: Vec<Vec<f32>> = vec![vec![0.0; 3]];
        let tgt: Vec<Vec<f32>> = vec![vec![10.0; 3]];
        let gap = domain_gap(&src, &tgt);
        assert!(gap > 0.0);
    }

    #[test]
    fn test_da_eval_report_fields() {
        let mut rng = make_rng(99);
        let model = DannModel::new(3, 4, 3, 0.5, &mut rng);
        let ds = sample_dataset();
        let report = da_evaluate(&model, &ds);
        assert!(report.source_accuracy >= 0.0 && report.source_accuracy <= 1.0);
        assert!(report.target_accuracy >= 0.0 && report.target_accuracy <= 1.0);
        assert!(report.mmd >= 0.0);
        assert!(report.domain_gap >= 0.0);
        assert!(report.proxy_ad >= 0.0 && report.proxy_ad <= 2.0);
    }

    #[test]
    fn test_evaluate_runs() {
        let mut rng = make_rng(100);
        let model = DannModel::new(3, 4, 3, 1.0, &mut rng);
        let ds = sample_dataset();
        let report = da_evaluate(&model, &ds);
        assert!(report.source_accuracy.is_finite());
        assert!(report.mmd.is_finite());
    }
}
