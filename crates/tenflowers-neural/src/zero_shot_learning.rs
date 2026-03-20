//! Zero-Shot Learning (ZSL) and Generalized Zero-Shot Learning (GZSL).
//!
//! This module implements a comprehensive suite of zero-shot learning methods,
//! covering semantic-space construction, visual-semantic compatibility mappings,
//! generative feature synthesis (CVAE), transductive propagation, and evaluation
//! metrics (ZSL top-k, GZSL harmonic mean, AUSUC).
//!
//! # Modules
//!
//! | Component | Description |
//! |-----------|-------------|
//! | [`SemanticSpace`] | Class attribute / word-vector semantic representations |
//! | [`LinearCompatibility`] | Linear visual–semantic compatibility W |
//! | [`BilinearCompatibility`] | Bilinear compatibility V^T W S |
//! | [`DeVise`] | DeViSE: separate projections for visual and semantic |
//! | [`ZslClassifier`] | ZSL / GZSL classifier with seen-class bias correction |
//! | [`SemanticAutoencoder`] | SAE encoder+decoder (Kodirov et al. 2017) |
//! | [`ZslVae`] | Conditional VAE for generative zero-shot feature synthesis |
//! | \[`TransductiveZsl`\] | Label propagation + domain-shift calibration |
//! | [`ZslMetrics`] | Per-class accuracy, H-mean, AUSUC, full evaluation report |

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Sample a single N(0,1) value via the Box–Muller transform.
#[inline]
fn sample_normal_f32(rng: &mut impl Rng) -> f32 {
    let u1: f32 = rng.random::<f32>().max(1e-10_f32);
    let u2: f32 = rng.random::<f32>();
    (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * std::f32::consts::PI * u2).cos()
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm of a slice.
#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Cosine similarity (returns 0.0 if either vector has zero norm).
#[inline]
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let na = l2_norm(a);
    let nb = l2_norm(b);
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot(a, b) / (na * nb)
}

/// Matrix–vector product  y = W x   (W is rows × cols, x is cols-dim).
fn matvec(w: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
    w.iter().map(|row| dot(row, x)).collect()
}

/// Random weight matrix (He uniform: ±sqrt(6 / fan_in)).
fn random_matrix(rows: usize, cols: usize, rng: &mut impl Rng) -> Vec<Vec<f32>> {
    let scale = (6.0_f32 / cols.max(1) as f32).sqrt();
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.random_range(-scale..scale)).collect())
        .collect()
}

/// Random bias vector (zeros).
fn zero_vec(n: usize) -> Vec<f32> {
    vec![0.0_f32; n]
}

/// ReLU activation.
#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Apply elementwise ReLU to a vector (in-place).
fn relu_vec(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = relu(*x);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  SemanticSpace
// ─────────────────────────────────────────────────────────────────────────────

/// Binary or continuous per-class attribute descriptor.
#[derive(Debug, Clone)]
pub struct ClassAttributes {
    /// Human-readable class name.
    pub class_name: String,
    /// Attribute vector (binary 0/1 or continuous values).
    pub attributes: Vec<f32>,
}

/// Word-vector representation for a single class label.
#[derive(Debug, Clone)]
pub struct WordVector {
    /// The word / class label string.
    pub word: String,
    /// Dense word embedding (word2vec / GloVe-style).
    pub embedding: Vec<f32>,
}

/// Semantic space holding class attribute descriptions and the seen/unseen split.
///
/// * `seen_classes`   – indices into `classes` used during training.
/// * `unseen_classes` – indices reserved for zero-shot evaluation.
#[derive(Debug, Clone)]
pub struct SemanticSpace {
    /// All class attribute descriptors.
    pub classes: Vec<ClassAttributes>,
    /// Indices of *seen* (training) classes.
    pub seen_classes: Vec<usize>,
    /// Indices of *unseen* (test) classes.
    pub unseen_classes: Vec<usize>,
}

impl SemanticSpace {
    /// Total number of classes (seen + unseen).
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

    /// Dimensionality of the attribute vectors.
    pub fn attribute_dim(&self) -> usize {
        self.classes
            .first()
            .map(|c| c.attributes.len())
            .unwrap_or(0)
    }

    /// Number of seen classes.
    pub fn seen_count(&self) -> usize {
        self.seen_classes.len()
    }

    /// Number of unseen classes.
    pub fn unseen_count(&self) -> usize {
        self.unseen_classes.len()
    }

    /// Cosine similarity between the attribute vectors of class `i` and class `j`.
    pub fn class_similarity(&self, i: usize, j: usize) -> f32 {
        let ai = &self.classes[i].attributes;
        let aj = &self.classes[j].attributes;
        cosine_sim(ai, aj)
    }

    /// Return the full attribute matrix of shape `(n_classes, attr_dim)`.
    pub fn attribute_matrix(&self) -> Vec<Vec<f32>> {
        self.classes.iter().map(|c| c.attributes.clone()).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  Visual-Semantic Compatibility Mappings
// ─────────────────────────────────────────────────────────────────────────────

/// Linear compatibility:  `score(v, s) = v^T W s`.
///
/// The projection `forward` maps a visual feature vector to the semantic space
/// via `p = W^T v`, so that the dot product `p · s ≈ score`.
#[derive(Debug, Clone)]
pub struct LinearCompatibility {
    /// Weight matrix of shape `(visual_dim × attr_dim)`.
    pub w: Vec<Vec<f32>>,
}

impl LinearCompatibility {
    /// Initialise with random weights (He uniform).
    pub fn new(visual_dim: usize, attr_dim: usize, rng: &mut impl Rng) -> Self {
        Self {
            w: random_matrix(visual_dim, attr_dim, rng),
        }
    }

    /// Compute the compatibility score  `v^T W s`.
    pub fn score(&self, visual: &[f32], semantic: &[f32]) -> f32 {
        // w has shape (visual_dim, attr_dim); (visual_dim,) * (visual_dim, attr_dim) * (attr_dim,)
        // = sum_i  visual[i] * (sum_j  w[i][j] * semantic[j])
        let proj = matvec(&self.w, semantic); // (visual_dim,)
        dot(visual, &proj)
    }

    /// Project visual features to the semantic space:  `p = W^T v`.
    ///
    /// Result has length `attr_dim`.
    pub fn forward(&self, visual: &[f32]) -> Vec<f32> {
        let attr_dim = self.w.first().map(|r| r.len()).unwrap_or(0);
        let mut out = vec![0.0_f32; attr_dim];
        for (row, &vi) in self.w.iter().zip(visual.iter()) {
            for (o, &wij) in out.iter_mut().zip(row.iter()) {
                *o += vi * wij;
            }
        }
        out
    }
}

/// Bilinear compatibility:  `score(v, s) = v^T W s`  where `W` is square
/// (`dim × dim`) and both `visual` and `semantic` are `dim`-dimensional.
#[derive(Debug, Clone)]
pub struct BilinearCompatibility {
    /// Square weight matrix of shape `(dim × dim)`.
    pub w: Vec<Vec<f32>>,
}

impl BilinearCompatibility {
    /// Initialise with a random square matrix.
    pub fn new(dim: usize, rng: &mut impl Rng) -> Self {
        Self {
            w: random_matrix(dim, dim, rng),
        }
    }

    /// Compute the bilinear score  `v^T W s`.
    pub fn score(&self, visual: &[f32], semantic: &[f32]) -> f32 {
        let proj = matvec(&self.w, semantic); // W s  →  (dim,)
        dot(visual, &proj)
    }
}

/// DeViSE (Deep Visual–Semantic Embedding) compatibility.
///
/// Visual features are projected to `embed_dim` via `visual_proj`,
/// semantic attributes are projected via `semantic_proj`, and the score
/// is the dot product of the two projected vectors.
#[derive(Debug, Clone)]
pub struct DeVise {
    /// Projection for visual features: `(visual_dim × embed_dim)`.
    pub visual_proj: Vec<Vec<f32>>,
    /// Projection for semantic attributes: `(attr_dim × embed_dim)`.
    pub semantic_proj: Vec<Vec<f32>>,
}

impl DeVise {
    /// Initialise with random projection matrices.
    pub fn new(visual_dim: usize, attr_dim: usize, embed_dim: usize, rng: &mut impl Rng) -> Self {
        Self {
            visual_proj: random_matrix(visual_dim, embed_dim, rng),
            semantic_proj: random_matrix(attr_dim, embed_dim, rng),
        }
    }

    /// Projected visual vector: `v_proj = visual_proj^T v` (embed_dim).
    fn project_visual(&self, visual: &[f32]) -> Vec<f32> {
        let embed_dim = self.visual_proj.first().map(|r| r.len()).unwrap_or(0);
        let mut out = vec![0.0_f32; embed_dim];
        for (row, &vi) in self.visual_proj.iter().zip(visual.iter()) {
            for (o, &wij) in out.iter_mut().zip(row.iter()) {
                *o += vi * wij;
            }
        }
        out
    }

    /// Projected semantic vector: `s_proj = semantic_proj^T s` (embed_dim).
    fn project_semantic(&self, semantic: &[f32]) -> Vec<f32> {
        let embed_dim = self.semantic_proj.first().map(|r| r.len()).unwrap_or(0);
        let mut out = vec![0.0_f32; embed_dim];
        for (row, &si) in self.semantic_proj.iter().zip(semantic.iter()) {
            for (o, &wij) in out.iter_mut().zip(row.iter()) {
                *o += si * wij;
            }
        }
        out
    }

    /// Compute the DeViSE compatibility score:  `V(x)^T S(c)`.
    pub fn score(&self, visual: &[f32], semantic: &[f32]) -> f32 {
        let vp = self.project_visual(visual);
        let sp = self.project_semantic(semantic);
        dot(&vp, &sp)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  ZSL Classifier
// ─────────────────────────────────────────────────────────────────────────────

/// A single zero-shot prediction entry.
#[derive(Debug, Clone)]
pub struct ZslPrediction {
    /// Index into the full class list.
    pub class_idx: usize,
    /// Compatibility score assigned by the mapping.
    pub score: f32,
    /// Human-readable class name copied from the semantic space.
    pub class_name: String,
}

/// Wrapper enum that holds one of the supported compatibility functions.
#[derive(Debug, Clone)]
pub enum CompatibilityType {
    /// Linear compatibility `v^T W s`.
    Linear(LinearCompatibility),
    /// Bilinear compatibility `v^T W s` with square W.
    Bilinear(BilinearCompatibility),
    /// DeViSE compatibility with dual projections.
    Devise(DeVise),
}

impl CompatibilityType {
    /// Dispatch to the underlying score function.
    fn score(&self, visual: &[f32], semantic: &[f32]) -> f32 {
        match self {
            CompatibilityType::Linear(m) => m.score(visual, semantic),
            CompatibilityType::Bilinear(m) => m.score(visual, semantic),
            CompatibilityType::Devise(m) => m.score(visual, semantic),
        }
    }
}

/// Zero-shot / generalised zero-shot classifier.
#[derive(Debug, Clone)]
pub struct ZslClassifier {
    /// The compatibility function to use.
    pub compatibility: CompatibilityType,
    /// The semantic space (class attributes + seen/unseen split).
    pub space: SemanticSpace,
}

impl ZslClassifier {
    /// Create a new classifier.
    pub fn new(compatibility: CompatibilityType, space: SemanticSpace) -> Self {
        Self {
            compatibility,
            space,
        }
    }

    /// **ZSL inference**: score the visual feature against all *unseen* classes
    /// and return predictions sorted by score (highest first).
    pub fn classify_zsl(&self, visual: &[f32]) -> Vec<ZslPrediction> {
        let mut preds: Vec<ZslPrediction> = self
            .space
            .unseen_classes
            .iter()
            .map(|&idx| {
                let s = self
                    .compatibility
                    .score(visual, &self.space.classes[idx].attributes);
                ZslPrediction {
                    class_idx: idx,
                    score: s,
                    class_name: self.space.classes[idx].class_name.clone(),
                }
            })
            .collect();
        preds.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        preds
    }

    /// **GZSL inference**: score against *all* classes.
    ///
    /// Seen-class scores are penalised by subtracting `seen_bias` to correct
    /// the model's preference for seen categories.
    pub fn classify_gzsl(&self, visual: &[f32], seen_bias: f32) -> Vec<ZslPrediction> {
        let seen_set: std::collections::HashSet<usize> =
            self.space.seen_classes.iter().copied().collect();

        let mut preds: Vec<ZslPrediction> = (0..self.space.n_classes())
            .map(|idx| {
                let mut s = self
                    .compatibility
                    .score(visual, &self.space.classes[idx].attributes);
                if seen_set.contains(&idx) {
                    s -= seen_bias;
                }
                ZslPrediction {
                    class_idx: idx,
                    score: s,
                    class_name: self.space.classes[idx].class_name.clone(),
                }
            })
            .collect();
        preds.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        preds
    }

    /// Return the top-`k` predictions from an already-sorted slice.
    pub fn top_k<'a>(&self, predictions: &'a [ZslPrediction], k: usize) -> Vec<&'a ZslPrediction> {
        predictions.iter().take(k).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  Semantic Autoencoder (SAE)
// ─────────────────────────────────────────────────────────────────────────────

/// Encoder network: visual features → semantic attributes.
#[derive(Debug, Clone)]
pub struct SaeEncoder {
    /// Weight matrix `(visual_dim × attr_dim)`.
    pub w: Vec<Vec<f32>>,
    /// Bias vector `(attr_dim,)`.
    pub b: Vec<f32>,
}

impl SaeEncoder {
    /// Create with random weights.
    pub fn new(visual_dim: usize, attr_dim: usize, rng: &mut impl Rng) -> Self {
        Self {
            w: random_matrix(visual_dim, attr_dim, rng),
            b: zero_vec(attr_dim),
        }
    }

    /// Forward pass: returns `ReLU(W^T v + b)`, shape `(attr_dim,)`.
    pub fn forward(&self, visual: &[f32]) -> Vec<f32> {
        let attr_dim = self.b.len();
        let mut out = self.b.clone();
        for (row, &vi) in self.w.iter().zip(visual.iter()) {
            for (j, &wij) in row.iter().enumerate() {
                if j < attr_dim {
                    out[j] += vi * wij;
                }
            }
        }
        relu_vec(&mut out);
        out
    }
}

/// Decoder network: semantic attributes → visual features.
#[derive(Debug, Clone)]
pub struct SaeDecoder {
    /// Weight matrix `(attr_dim × visual_dim)`.
    pub w: Vec<Vec<f32>>,
    /// Bias vector `(visual_dim,)`.
    pub b: Vec<f32>,
}

impl SaeDecoder {
    /// Create with random weights.
    pub fn new(attr_dim: usize, visual_dim: usize, rng: &mut impl Rng) -> Self {
        Self {
            w: random_matrix(attr_dim, visual_dim, rng),
            b: zero_vec(visual_dim),
        }
    }

    /// Forward pass: returns `W^T s + b`, shape `(visual_dim,)`.
    pub fn forward(&self, semantic: &[f32]) -> Vec<f32> {
        let visual_dim = self.b.len();
        let mut out = self.b.clone();
        for (row, &si) in self.w.iter().zip(semantic.iter()) {
            for (j, &wij) in row.iter().enumerate() {
                if j < visual_dim {
                    out[j] += si * wij;
                }
            }
        }
        out
    }
}

/// Semantic Autoencoder (Kodirov et al. 2017):
/// encoder maps visual → semantic, decoder reconstructs visual from semantic.
///
/// Loss = reconstruction_loss + λ * semantic_loss.
#[derive(Debug, Clone)]
pub struct SemanticAutoencoder {
    /// Visual-to-semantic encoder.
    pub encoder: SaeEncoder,
    /// Semantic-to-visual decoder.
    pub decoder: SaeDecoder,
    /// Regularisation weight for the semantic loss term.
    pub lambda: f32,
}

impl SemanticAutoencoder {
    /// Initialise a new SAE with random weights.
    pub fn new(visual_dim: usize, attr_dim: usize, rng: &mut impl Rng) -> Self {
        Self {
            encoder: SaeEncoder::new(visual_dim, attr_dim, rng),
            decoder: SaeDecoder::new(attr_dim, visual_dim, rng),
            lambda: 1.0,
        }
    }

    /// Encode a visual feature vector to the semantic space.
    pub fn encode(&self, visual: &[f32]) -> Vec<f32> {
        self.encoder.forward(visual)
    }

    /// Decode a semantic vector back to visual feature space.
    pub fn decode(&self, semantic: &[f32]) -> Vec<f32> {
        self.decoder.forward(semantic)
    }

    /// MSE reconstruction loss: `||decode(encode(visual)) - visual||^2 / n`.
    pub fn reconstruction_loss(&self, visual: &[f32]) -> f32 {
        let recon = self.decode(&self.encode(visual));
        let n = visual.len().max(1) as f32;
        visual
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / n
    }

    /// MSE semantic loss: `||encode(visual) - target_semantic||^2 / n`.
    pub fn semantic_loss(&self, visual: &[f32], target_semantic: &[f32]) -> f32 {
        let encoded = self.encode(visual);
        let n = encoded.len().max(1) as f32;
        encoded
            .iter()
            .zip(target_semantic.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / n
    }

    /// Combined loss: `reconstruction_loss + λ * semantic_loss`.
    pub fn total_loss(&self, visual: &[f32], target_semantic: &[f32]) -> f32 {
        self.reconstruction_loss(visual) + self.lambda * self.semantic_loss(visual, target_semantic)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  Generative ZSL — Conditional VAE
// ─────────────────────────────────────────────────────────────────────────────

/// Conditional VAE for zero-shot visual feature synthesis.
///
/// The model is conditioned on class attribute vectors (appended as extra
/// inputs at the encoder and decoder).
///
/// Architecture:
/// - Encoder input:  `visual ++ attrs`  (visual_dim + attr_dim)
/// - Hidden state:   `encoder_w` → `h`  (latent_dim)
/// - Mean head:      `mu_w`  → `mu`     (latent_dim)
/// - Logvar head:    `logvar_w` → `logvar` (latent_dim)
/// - Decoder input:  `z ++ attrs`  (latent_dim + attr_dim)
/// - Output:         `decoder_w` → `x_recon` (visual_dim)
#[derive(Debug, Clone)]
pub struct ZslVae {
    /// Encoder hidden layer weights `((visual_dim+attr_dim) × latent_dim)`.
    pub encoder_w: Vec<Vec<f32>>,
    /// Mean-head weights `(latent_dim × latent_dim)`.
    pub mu_w: Vec<Vec<f32>>,
    /// Log-variance head weights `(latent_dim × latent_dim)`.
    pub logvar_w: Vec<Vec<f32>>,
    /// Decoder weights `((latent_dim+attr_dim) × visual_dim)`.
    pub decoder_w: Vec<Vec<f32>>,
    /// Dimensionality of the latent space.
    pub latent_dim: usize,
}

impl ZslVae {
    /// Initialise with random weights.
    pub fn new(visual_dim: usize, attr_dim: usize, latent_dim: usize, rng: &mut impl Rng) -> Self {
        // encoder_w: rows=(visual_dim+attr_dim), cols=latent_dim
        //   → matvec(encoder_w, inp) gives (visual_dim+attr_dim,) — wrong; we need (latent_dim,).
        //   So encoder_w must be (latent_dim × (visual_dim+attr_dim)) for W*x → (latent_dim,).
        // decoder_w: rows=visual_dim, cols=(latent_dim+attr_dim)
        //   → matvec(decoder_w, inp) gives (visual_dim,) ✓
        Self {
            encoder_w: random_matrix(latent_dim, visual_dim + attr_dim, rng),
            mu_w: random_matrix(latent_dim, latent_dim, rng),
            logvar_w: random_matrix(latent_dim, latent_dim, rng),
            decoder_w: random_matrix(visual_dim, latent_dim + attr_dim, rng),
            latent_dim,
        }
    }

    /// Encode `(visual ++ attrs)` to obtain `(mu, logvar)` each of size `latent_dim`.
    pub fn encode(&self, visual: &[f32], attrs: &[f32]) -> (Vec<f32>, Vec<f32>) {
        // Concatenate input
        let mut inp: Vec<f32> = Vec::with_capacity(visual.len() + attrs.len());
        inp.extend_from_slice(visual);
        inp.extend_from_slice(attrs);

        // Hidden layer: ReLU(encoder_w^T inp)
        let mut h = matvec(&self.encoder_w, &inp);
        relu_vec(&mut h);

        // Mean and log-variance heads
        let mu = matvec(&self.mu_w, &h);
        let logvar = matvec(&self.logvar_w, &h);
        (mu, logvar)
    }

    /// Reparameterise: `z = mu + eps * exp(0.5 * logvar)`, eps ~ N(0,I).
    pub fn reparameterize(mu: &[f32], logvar: &[f32], rng: &mut impl Rng) -> Vec<f32> {
        mu.iter()
            .zip(logvar.iter())
            .map(|(&m, &lv)| {
                let std = (0.5_f32 * lv).exp();
                let eps = sample_normal_f32(rng);
                m + eps * std
            })
            .collect()
    }

    /// Decode `(z ++ attrs)` to reconstruct visual features.
    pub fn decode(&self, z: &[f32], attrs: &[f32]) -> Vec<f32> {
        let mut inp: Vec<f32> = Vec::with_capacity(z.len() + attrs.len());
        inp.extend_from_slice(z);
        inp.extend_from_slice(attrs);
        matvec(&self.decoder_w, &inp)
    }

    /// Synthesise `n_samples` visual feature vectors for a class described by `attrs`.
    pub fn synthesize_features(
        &self,
        attrs: &[f32],
        n_samples: usize,
        rng: &mut impl Rng,
    ) -> Vec<Vec<f32>> {
        (0..n_samples)
            .map(|_| {
                let mu = vec![0.0_f32; self.latent_dim];
                let logvar = vec![0.0_f32; self.latent_dim];
                let z = Self::reparameterize(&mu, &logvar, rng);
                self.decode(&z, attrs)
            })
            .collect()
    }

    /// ELBO loss for a single data point.
    ///
    /// `ELBO = -MSE(recon, visual) + 0.5 * sum(1 + logvar - mu^2 - exp(logvar))`
    ///
    /// Returns the *negative* ELBO (loss to minimise).
    pub fn elbo_loss(&self, visual: &[f32], attrs: &[f32], rng: &mut impl Rng) -> f32 {
        let (mu, logvar) = self.encode(visual, attrs);
        let z = Self::reparameterize(&mu, &logvar, rng);
        let recon = self.decode(&z, attrs);

        // Reconstruction term (negative MSE)
        let n = visual.len().max(1) as f32;
        let mse: f32 = visual
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / n;

        // KL divergence term
        let kl: f32 = mu
            .iter()
            .zip(logvar.iter())
            .map(|(&m, &lv)| 1.0_f32 + lv - m * m - lv.exp())
            .sum::<f32>()
            * 0.5_f32;

        // Negative ELBO
        mse - kl
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  Transductive ZSL
// ─────────────────────────────────────────────────────────────────────────────

/// Structured prediction via label propagation on the semantic similarity graph.
pub struct StructuredPrediction;

impl StructuredPrediction {
    /// Build a semantic similarity graph.
    ///
    /// `W_ij = exp(-||a_i - a_j||^2 / sigma^2)`, then row-normalise so each
    /// row sums to 1 (set diagonal to zero before normalising).
    pub fn build_semantic_graph(space: &SemanticSpace, sigma: f32) -> Vec<Vec<f32>> {
        let n = space.n_classes();
        let sigma2 = sigma * sigma;

        let mut graph = vec![vec![0.0_f32; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let ai = &space.classes[i].attributes;
                let aj = &space.classes[j].attributes;
                let sq_dist: f32 = ai.iter().zip(aj.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                graph[i][j] = (-sq_dist / sigma2.max(1e-12)).exp();
            }
        }

        // Row-normalise
        for row in graph.iter_mut() {
            let s: f32 = row.iter().sum();
            if s > 1e-12 {
                for v in row.iter_mut() {
                    *v /= s;
                }
            }
        }
        graph
    }

    /// Label spreading:  `F_{t+1} = alpha * W * F_t + (1-alpha) * Y`.
    ///
    /// - `graph`: row-normalised adjacency `(n × n)`.
    /// - `initial_scores`: score matrix `(n × k)` (n instances, k classes).
    /// - `alpha`: propagation strength (0 = no propagation, 1 = full).
    /// - `n_iter`: number of iterations.
    pub fn propagate(
        graph: &[Vec<f32>],
        initial_scores: &[Vec<f32>],
        alpha: f32,
        n_iter: usize,
    ) -> Vec<Vec<f32>> {
        let n = initial_scores.len();
        if n == 0 {
            return Vec::new();
        }
        let k = initial_scores[0].len();
        let mut f = initial_scores.to_vec();

        for _ in 0..n_iter {
            let mut new_f = vec![vec![0.0_f32; k]; n];
            // W * F
            for i in 0..n {
                if i >= graph.len() {
                    break;
                }
                for j in 0..n {
                    if j >= graph[i].len() || j >= f.len() {
                        break;
                    }
                    let wij = graph[i][j];
                    for c in 0..k {
                        if c < f[j].len() {
                            new_f[i][c] += wij * f[j][c];
                        }
                    }
                }
            }
            // alpha * W * F + (1-alpha) * Y
            for i in 0..n {
                for c in 0..k {
                    let y = if c < initial_scores[i].len() {
                        initial_scores[i][c]
                    } else {
                        0.0
                    };
                    new_f[i][c] = alpha * new_f[i][c] + (1.0 - alpha) * y;
                }
            }
            f = new_f;
        }
        f
    }
}

/// Domain-shift calibration utilities for GZSL.
pub struct DomainShiftCorrection;

impl DomainShiftCorrection {
    /// Multiply unseen-class scores by `(1 - seen_prior)` and seen-class scores
    /// by `seen_prior` to account for dataset-level class-frequency bias.
    ///
    /// `visual_scores`: `(n_samples × n_classes)` raw score matrix.
    /// The first `n_seen` columns correspond to seen classes; the remaining to
    /// unseen classes.  Here we use the simple convention that
    /// seen/unseen partition is encoded by `seen_prior` applied uniformly.
    pub fn calibrate_scores(visual_scores: &[Vec<f32>], seen_prior: f32) -> Vec<Vec<f32>> {
        let unseen_prior = 1.0 - seen_prior;
        visual_scores
            .iter()
            .map(|row| {
                // Heuristic: first half of classes are "seen", second half "unseen"
                let half = row.len() / 2;
                row.iter()
                    .enumerate()
                    .map(|(j, &s)| {
                        if j < half {
                            s * seen_prior
                        } else {
                            s * unseen_prior
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Softmax with temperature `T`:  `softmax(scores / T)`.
    ///
    /// High temperature → more uniform; low temperature → sharper peaks.
    pub fn temperature_scaling(scores: &[Vec<f32>], temperature: f32) -> Vec<Vec<f32>> {
        let t = temperature.max(1e-12);
        scores
            .iter()
            .map(|row| {
                let scaled: Vec<f32> = row.iter().map(|&s| s / t).collect();
                let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scaled.iter().map(|&v| (v - max_val).exp()).collect();
                let sum: f32 = exps.iter().sum::<f32>().max(1e-12);
                exps.iter().map(|&e| e / sum).collect()
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  ZSL Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Zero-shot learning evaluation metrics.
pub struct ZslMetrics;

impl ZslMetrics {
    /// Per-class accuracy vector of length `n_classes`.
    ///
    /// `accuracy[c] = correct_c / total_c`; classes with no samples get `0.0`.
    pub fn per_class_accuracy(
        predictions: &[usize],
        targets: &[usize],
        n_classes: usize,
    ) -> Vec<f32> {
        let mut correct = vec![0usize; n_classes];
        let mut total = vec![0usize; n_classes];
        for (&pred, &tgt) in predictions.iter().zip(targets.iter()) {
            if tgt < n_classes {
                total[tgt] += 1;
                if pred == tgt {
                    correct[tgt] += 1;
                }
            }
        }
        correct
            .iter()
            .zip(total.iter())
            .map(|(&c, &t)| if t == 0 { 0.0 } else { c as f32 / t as f32 })
            .collect()
    }

    /// Top-k accuracy: fraction of samples where the correct class is in the
    /// top-k predicted scores.
    ///
    /// `scores`: `(n_samples × n_classes)` score matrix (higher = better).
    pub fn top_k_accuracy(scores: &[Vec<f32>], targets: &[usize], k: usize) -> f32 {
        if scores.is_empty() {
            return 0.0;
        }
        let mut correct = 0usize;
        for (row, &tgt) in scores.iter().zip(targets.iter()) {
            // Find indices of top-k scores
            let mut indexed: Vec<(usize, f32)> =
                row.iter().enumerate().map(|(i, &s)| (i, s)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top_k_indices: std::collections::HashSet<usize> =
                indexed.iter().take(k).map(|(i, _)| *i).collect();
            if top_k_indices.contains(&tgt) {
                correct += 1;
            }
        }
        correct as f32 / scores.len() as f32
    }

    /// GZSL harmonic mean accuracy: `H = 2 * S * U / (S + U)`.
    ///
    /// Returns `0.0` if `S + U == 0`.
    pub fn harmonic_mean_accuracy(seen_acc: f32, unseen_acc: f32) -> f32 {
        let denom = seen_acc + unseen_acc;
        if denom < 1e-12 {
            0.0
        } else {
            2.0 * seen_acc * unseen_acc / denom
        }
    }

    /// Area under the seen-vs-unseen accuracy curve (trapezoidal rule).
    ///
    /// `seen_accs` and `unseen_accs` should have the same length.  Returns `0.0`
    /// if fewer than 2 points are provided.
    pub fn area_under_seen_unseen_curve(seen_accs: &[f32], unseen_accs: &[f32]) -> f32 {
        let n = seen_accs.len().min(unseen_accs.len());
        if n < 2 {
            return 0.0;
        }
        let mut area = 0.0_f32;
        for i in 1..n {
            let dx = seen_accs[i] - seen_accs[i - 1];
            let avg_y = 0.5 * (unseen_accs[i] + unseen_accs[i - 1]);
            area += dx.abs() * avg_y;
        }
        area
    }
}

/// Summary report produced by [`ZslMetrics::evaluate`].
#[derive(Debug, Clone)]
pub struct ZslEvalReport {
    /// ZSL top-1 accuracy (over unseen classes only).
    pub zsl_top1: f32,
    /// ZSL top-5 accuracy (over unseen classes only).
    pub zsl_top5: f32,
    /// GZSL accuracy on seen classes (seen split, all-class prediction).
    pub gzsl_seen: f32,
    /// GZSL accuracy on unseen classes (unseen split, all-class prediction).
    pub gzsl_unseen: f32,
    /// GZSL harmonic mean of seen and unseen accuracies.
    pub gzsl_harmonic: f32,
}

impl ZslMetrics {
    /// Full evaluation of a [`ZslClassifier`] on a set of labelled visual features.
    ///
    /// `visuals` – one visual feature vector per sample.
    /// `labels`  – ground-truth class indices (into the full class list).
    ///
    /// The method splits samples by whether their label is in `seen_classes` or
    /// `unseen_classes` and computes all standard metrics.
    pub fn evaluate(
        classifier: &ZslClassifier,
        visuals: &[Vec<f32>],
        labels: &[usize],
    ) -> ZslEvalReport {
        let seen_set: std::collections::HashSet<usize> =
            classifier.space.seen_classes.iter().copied().collect();
        let unseen_set: std::collections::HashSet<usize> =
            classifier.space.unseen_classes.iter().copied().collect();

        // Collect unseen samples for ZSL evaluation
        let mut zsl_scores: Vec<Vec<f32>> = Vec::new();
        let mut zsl_targets: Vec<usize> = Vec::new();

        // Collect all samples for GZSL evaluation (split into seen/unseen)
        let mut gzsl_seen_correct = 0usize;
        let mut gzsl_seen_total = 0usize;
        let mut gzsl_unseen_correct = 0usize;
        let mut gzsl_unseen_total = 0usize;

        for (visual, &label) in visuals.iter().zip(labels.iter()) {
            // ZSL: score against unseen classes only
            if unseen_set.contains(&label) {
                let preds = classifier.classify_zsl(visual);
                let score_row: Vec<f32> = {
                    // Build a score row indexed by unseen class position
                    let n_unseen = classifier.space.unseen_classes.len();
                    let mut row = vec![0.0_f32; n_unseen];
                    for (pos, &cidx) in classifier.space.unseen_classes.iter().enumerate() {
                        if let Some(p) = preds.iter().find(|p| p.class_idx == cidx) {
                            row[pos] = p.score;
                        }
                    }
                    row
                };
                // Convert label to position in unseen_classes
                let pos = classifier
                    .space
                    .unseen_classes
                    .iter()
                    .position(|&x| x == label);
                if let Some(p) = pos {
                    zsl_scores.push(score_row);
                    zsl_targets.push(p);
                }
            }

            // GZSL: score against all classes
            let gzsl_preds = classifier.classify_gzsl(visual, 0.0);
            let top1_idx = gzsl_preds
                .first()
                .map(|p| p.class_idx)
                .unwrap_or(usize::MAX);

            if seen_set.contains(&label) {
                gzsl_seen_total += 1;
                if top1_idx == label {
                    gzsl_seen_correct += 1;
                }
            } else if unseen_set.contains(&label) {
                gzsl_unseen_total += 1;
                if top1_idx == label {
                    gzsl_unseen_correct += 1;
                }
            }
        }

        let zsl_top1 = ZslMetrics::top_k_accuracy(&zsl_scores, &zsl_targets, 1);
        let zsl_top5 = ZslMetrics::top_k_accuracy(&zsl_scores, &zsl_targets, 5);
        let gzsl_seen = if gzsl_seen_total == 0 {
            0.0
        } else {
            gzsl_seen_correct as f32 / gzsl_seen_total as f32
        };
        let gzsl_unseen = if gzsl_unseen_total == 0 {
            0.0
        } else {
            gzsl_unseen_correct as f32 / gzsl_unseen_total as f32
        };
        let gzsl_harmonic = ZslMetrics::harmonic_mean_accuracy(gzsl_seen, gzsl_unseen);

        ZslEvalReport {
            zsl_top1,
            zsl_top5,
            gzsl_seen,
            gzsl_unseen,
            gzsl_harmonic,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    /// Build a small semantic space: 6 classes (4 seen, 2 unseen), 8 attributes.
    fn make_space() -> SemanticSpace {
        let classes = vec![
            ClassAttributes {
                class_name: "cat".into(),
                attributes: vec![1., 0., 1., 0., 1., 0., 1., 0.],
            },
            ClassAttributes {
                class_name: "dog".into(),
                attributes: vec![0., 1., 0., 1., 0., 1., 0., 1.],
            },
            ClassAttributes {
                class_name: "bird".into(),
                attributes: vec![1., 1., 0., 0., 1., 1., 0., 0.],
            },
            ClassAttributes {
                class_name: "fish".into(),
                attributes: vec![0., 0., 1., 1., 0., 0., 1., 1.],
            },
            ClassAttributes {
                class_name: "horse".into(),
                attributes: vec![1., 0., 0., 1., 1., 0., 0., 1.],
            },
            ClassAttributes {
                class_name: "whale".into(),
                attributes: vec![0., 1., 1., 0., 0., 1., 1., 0.],
            },
        ];
        SemanticSpace {
            classes,
            seen_classes: vec![0, 1, 2, 3],
            unseen_classes: vec![4, 5],
        }
    }

    fn make_visual(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = make_rng(seed);
        (0..dim).map(|_| rng.random::<f32>()).collect()
    }

    // ── §1 SemanticSpace ──────────────────────────────────────────────────────

    #[test]
    fn test_semantic_space_creation() {
        let space = make_space();
        assert_eq!(space.classes.len(), 6);
        assert_eq!(space.classes[0].class_name, "cat");
    }

    #[test]
    fn test_semantic_space_counts() {
        let space = make_space();
        assert_eq!(space.n_classes(), 6);
        assert_eq!(space.seen_count(), 4);
        assert_eq!(space.unseen_count(), 2);
    }

    #[test]
    fn test_class_similarity_self() {
        let space = make_space();
        let sim = space.class_similarity(0, 0);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "self-similarity should be 1, got {sim}"
        );
    }

    #[test]
    fn test_class_similarity_range() {
        let space = make_space();
        for i in 0..space.n_classes() {
            for j in 0..space.n_classes() {
                let s = space.class_similarity(i, j);
                assert!((-1.01..=1.01).contains(&s), "cosine sim out of [-1,1]: {s}");
            }
        }
    }

    #[test]
    fn test_attribute_matrix_shape() {
        let space = make_space();
        let mat = space.attribute_matrix();
        assert_eq!(mat.len(), 6);
        for row in &mat {
            assert_eq!(row.len(), 8);
        }
    }

    // ── §2 VisualSemanticMapping ───────────────────────────────────────────────

    #[test]
    fn test_linear_compatibility_forward_shape() {
        let mut rng = make_rng(1);
        let lc = LinearCompatibility::new(16, 8, &mut rng);
        let v = make_visual(16, 2);
        let proj = lc.forward(&v);
        assert_eq!(proj.len(), 8);
    }

    #[test]
    fn test_linear_compatibility_score_finite() {
        let mut rng = make_rng(3);
        let lc = LinearCompatibility::new(16, 8, &mut rng);
        let v = make_visual(16, 4);
        let s = make_visual(8, 5);
        let score = lc.score(&v, &s);
        assert!(score.is_finite(), "score should be finite, got {score}");
    }

    #[test]
    fn test_bilinear_compatibility_score_finite() {
        let mut rng = make_rng(6);
        let bc = BilinearCompatibility::new(8, &mut rng);
        let v = make_visual(8, 7);
        let s = make_visual(8, 8);
        let score = bc.score(&v, &s);
        assert!(
            score.is_finite(),
            "bilinear score should be finite, got {score}"
        );
    }

    #[test]
    fn test_devise_score_finite() {
        let mut rng = make_rng(9);
        let dv = DeVise::new(16, 8, 12, &mut rng);
        let v = make_visual(16, 10);
        let s = make_visual(8, 11);
        let score = dv.score(&v, &s);
        assert!(
            score.is_finite(),
            "devise score should be finite, got {score}"
        );
    }

    #[test]
    fn test_devise_projection_shape() {
        let mut rng = make_rng(12);
        let dv = DeVise::new(16, 8, 12, &mut rng);
        let v = make_visual(16, 13);
        let vp = dv.project_visual(&v);
        assert_eq!(vp.len(), 12);
    }

    // ── §3 ZSL Classifier ─────────────────────────────────────────────────────

    fn make_classifier() -> ZslClassifier {
        let mut rng = make_rng(42);
        let space = make_space();
        let lc = LinearCompatibility::new(16, 8, &mut rng);
        ZslClassifier::new(CompatibilityType::Linear(lc), space)
    }

    #[test]
    fn test_zsl_classifier_zsl_output_count() {
        let clf = make_classifier();
        let v = make_visual(16, 99);
        let preds = clf.classify_zsl(&v);
        assert_eq!(preds.len(), clf.space.unseen_count());
    }

    #[test]
    fn test_zsl_classifier_gzsl_output_count() {
        let clf = make_classifier();
        let v = make_visual(16, 100);
        let preds = clf.classify_gzsl(&v, 0.5);
        assert_eq!(preds.len(), clf.space.n_classes());
    }

    #[test]
    fn test_zsl_predictions_sorted_descending() {
        let clf = make_classifier();
        let v = make_visual(16, 101);
        let preds = clf.classify_zsl(&v);
        for w in preds.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "predictions not sorted: {} < {}",
                w[0].score,
                w[1].score
            );
        }
    }

    #[test]
    fn test_gzsl_seen_bias_effect() {
        let mut rng = make_rng(42);
        let space = make_space();
        let lc = LinearCompatibility::new(16, 8, &mut rng);
        let clf = ZslClassifier::new(CompatibilityType::Linear(lc), space);
        let v = make_visual(16, 102);

        let preds_no_bias = clf.classify_gzsl(&v, 0.0);
        let preds_bias = clf.classify_gzsl(&v, 5.0);

        // With large bias, seen classes should have lower scores
        let seen_set: std::collections::HashSet<usize> =
            clf.space.seen_classes.iter().copied().collect();

        let score_seen_no_bias: f32 = preds_no_bias
            .iter()
            .filter(|p| seen_set.contains(&p.class_idx))
            .map(|p| p.score)
            .sum();
        let score_seen_bias: f32 = preds_bias
            .iter()
            .filter(|p| seen_set.contains(&p.class_idx))
            .map(|p| p.score)
            .sum();
        assert!(
            score_seen_bias < score_seen_no_bias,
            "seen-class scores should be lower with positive bias"
        );
    }

    #[test]
    fn test_top_k_count() {
        let clf = make_classifier();
        let v = make_visual(16, 103);
        let preds = clf.classify_zsl(&v);
        let top2 = clf.top_k(&preds, 2);
        assert_eq!(top2.len(), 2.min(preds.len()));
    }

    // ── §4 Semantic Autoencoder ───────────────────────────────────────────────

    fn make_sae() -> SemanticAutoencoder {
        let mut rng = make_rng(50);
        SemanticAutoencoder::new(16, 8, &mut rng)
    }

    #[test]
    fn test_sae_encoder_shape() {
        let sae = make_sae();
        let v = make_visual(16, 200);
        let enc = sae.encode(&v);
        assert_eq!(enc.len(), 8);
    }

    #[test]
    fn test_sae_decoder_shape() {
        let sae = make_sae();
        let s = make_visual(8, 201);
        let dec = sae.decode(&s);
        assert_eq!(dec.len(), 16);
    }

    #[test]
    fn test_sae_reconstruction_loss_nonneg() {
        let sae = make_sae();
        let v = make_visual(16, 202);
        let loss = sae.reconstruction_loss(&v);
        assert!(
            loss >= 0.0,
            "reconstruction loss must be non-negative, got {loss}"
        );
        assert!(loss.is_finite(), "reconstruction loss must be finite");
    }

    #[test]
    fn test_sae_semantic_loss_nonneg() {
        let sae = make_sae();
        let v = make_visual(16, 203);
        let t = make_visual(8, 204);
        let loss = sae.semantic_loss(&v, &t);
        assert!(
            loss >= 0.0,
            "semantic loss must be non-negative, got {loss}"
        );
        assert!(loss.is_finite());
    }

    #[test]
    fn test_sae_total_loss_decomposition() {
        let sae = make_sae();
        let v = make_visual(16, 205);
        let t = make_visual(8, 206);
        let total = sae.total_loss(&v, &t);
        let expected = sae.reconstruction_loss(&v) + sae.lambda * sae.semantic_loss(&v, &t);
        assert!(
            (total - expected).abs() < 1e-5,
            "total loss decomposition mismatch"
        );
    }

    // ── §5 ZslVae ─────────────────────────────────────────────────────────────

    fn make_zsl_vae() -> ZslVae {
        let mut rng = make_rng(60);
        ZslVae::new(16, 8, 12, &mut rng)
    }

    #[test]
    fn test_zsl_vae_encode_returns_mu_logvar() {
        let vae = make_zsl_vae();
        let v = make_visual(16, 300);
        let a = make_visual(8, 301);
        let (mu, logvar) = vae.encode(&v, &a);
        assert_eq!(mu.len(), 12);
        assert_eq!(logvar.len(), 12);
    }

    #[test]
    fn test_zsl_vae_mu_shape() {
        let vae = make_zsl_vae();
        let v = make_visual(16, 302);
        let a = make_visual(8, 303);
        let (mu, _) = vae.encode(&v, &a);
        assert_eq!(mu.len(), vae.latent_dim);
    }

    #[test]
    fn test_zsl_vae_logvar_shape() {
        let vae = make_zsl_vae();
        let v = make_visual(16, 304);
        let a = make_visual(8, 305);
        let (_, logvar) = vae.encode(&v, &a);
        assert_eq!(logvar.len(), vae.latent_dim);
    }

    #[test]
    fn test_reparameterize_shape() {
        let mut rng = make_rng(70);
        let mu: Vec<f32> = vec![0.0; 12];
        let logvar: Vec<f32> = vec![0.0; 12];
        let z = ZslVae::reparameterize(&mu, &logvar, &mut rng);
        assert_eq!(z.len(), 12);
    }

    #[test]
    fn test_zsl_vae_decode_shape() {
        let vae = make_zsl_vae();
        let mut rng = make_rng(71);
        let z: Vec<f32> = (0..12).map(|_| rng.random::<f32>()).collect();
        let a = make_visual(8, 306);
        let out = vae.decode(&z, &a);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_synthesize_features_count() {
        let vae = make_zsl_vae();
        let mut rng = make_rng(72);
        let attrs = make_visual(8, 307);
        let samples = vae.synthesize_features(&attrs, 5, &mut rng);
        assert_eq!(samples.len(), 5);
    }

    #[test]
    fn test_synthesize_features_shape() {
        let vae = make_zsl_vae();
        let mut rng = make_rng(73);
        let attrs = make_visual(8, 308);
        let samples = vae.synthesize_features(&attrs, 3, &mut rng);
        for s in &samples {
            assert_eq!(s.len(), 16);
        }
    }

    #[test]
    fn test_elbo_loss_finite() {
        let vae = make_zsl_vae();
        let mut rng = make_rng(74);
        let v = make_visual(16, 309);
        let a = make_visual(8, 310);
        let loss = vae.elbo_loss(&v, &a, &mut rng);
        assert!(loss.is_finite(), "ELBO loss must be finite, got {loss}");
    }

    // ── §6 TransductiveZsl ────────────────────────────────────────────────────

    #[test]
    fn test_build_semantic_graph_shape() {
        let space = make_space();
        let graph = StructuredPrediction::build_semantic_graph(&space, 1.0);
        assert_eq!(graph.len(), 6);
        for row in &graph {
            assert_eq!(row.len(), 6);
        }
    }

    #[test]
    fn test_semantic_graph_row_normalized() {
        let space = make_space();
        let graph = StructuredPrediction::build_semantic_graph(&space, 1.0);
        for (i, row) in graph.iter().enumerate() {
            let s: f32 = row.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-5 || s < 1e-12,
                "row {i} does not sum to 1 (sum = {s})"
            );
        }
    }

    #[test]
    fn test_propagate_shape() {
        let space = make_space();
        let graph = StructuredPrediction::build_semantic_graph(&space, 1.0);
        // initial_scores: 4 samples × 6 classes
        let initial: Vec<Vec<f32>> = (0..4).map(|i| vec![i as f32 * 0.1; 6]).collect();
        let result = StructuredPrediction::propagate(&graph, &initial, 0.5, 3);
        assert_eq!(result.len(), 4);
        for row in &result {
            assert_eq!(row.len(), 6);
        }
    }

    #[test]
    fn test_propagate_stays_bounded() {
        let space = make_space();
        let graph = StructuredPrediction::build_semantic_graph(&space, 1.0);
        let initial: Vec<Vec<f32>> = (0..4)
            .map(|_| {
                let mut row = vec![0.0_f32; 6];
                row[0] = 1.0;
                row
            })
            .collect();
        let result = StructuredPrediction::propagate(&graph, &initial, 0.5, 10);
        for row in &result {
            for &v in row {
                assert!(v.is_finite(), "propagated value not finite: {v}");
            }
        }
    }

    #[test]
    fn test_calibrate_scores_shape() {
        let scores: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0, 4.0]; 3];
        let cal = DomainShiftCorrection::calibrate_scores(&scores, 0.6);
        assert_eq!(cal.len(), 3);
        for row in &cal {
            assert_eq!(row.len(), 4);
        }
    }

    #[test]
    fn test_temperature_scaling_shape() {
        let scores: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0]; 4];
        let scaled = DomainShiftCorrection::temperature_scaling(&scores, 2.0);
        assert_eq!(scaled.len(), 4);
        for row in &scaled {
            assert_eq!(row.len(), 3);
        }
    }

    #[test]
    fn test_temperature_scaling_high_temp_uniform() {
        let scores: Vec<Vec<f32>> = vec![vec![1.0, 100.0, 0.0]];
        // At high temperature the distribution should be nearly uniform
        let scaled = DomainShiftCorrection::temperature_scaling(&scores, 1000.0);
        let row = &scaled[0];
        let min = row.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (max - min) < 0.1,
            "high-T softmax should be ~uniform, got min={min} max={max}"
        );
    }

    // ── §7 ZSL Metrics ────────────────────────────────────────────────────────

    #[test]
    fn test_per_class_accuracy_shape() {
        let preds = vec![0, 1, 2, 0];
        let targets = vec![0, 1, 1, 2];
        let acc = ZslMetrics::per_class_accuracy(&preds, &targets, 3);
        assert_eq!(acc.len(), 3);
    }

    #[test]
    fn test_per_class_accuracy_perfect() {
        let labels = vec![0, 1, 2, 0, 1, 2];
        let acc = ZslMetrics::per_class_accuracy(&labels, &labels, 3);
        for &a in &acc {
            assert!((a - 1.0).abs() < 1e-6, "perfect accuracy expected, got {a}");
        }
    }

    #[test]
    fn test_top_k_accuracy_k1_perfect() {
        // Every sample's correct class has the highest score
        let scores = vec![
            vec![5.0, 1.0, 0.0],
            vec![0.0, 4.0, 1.0],
            vec![0.0, 1.0, 3.0],
        ];
        let targets = vec![0, 1, 2];
        let acc = ZslMetrics::top_k_accuracy(&scores, &targets, 1);
        assert!(
            (acc - 1.0).abs() < 1e-6,
            "expected perfect top-1, got {acc}"
        );
    }

    #[test]
    fn test_top_k_accuracy_k5_ge_k1() {
        let mut rng = make_rng(80);
        let scores: Vec<Vec<f32>> = (0..20)
            .map(|_| (0..10).map(|_| rng.random::<f32>()).collect())
            .collect();
        let targets: Vec<usize> = (0..20).map(|i| i % 10).collect();
        let acc1 = ZslMetrics::top_k_accuracy(&scores, &targets, 1);
        let acc5 = ZslMetrics::top_k_accuracy(&scores, &targets, 5);
        assert!(
            acc5 >= acc1 - 1e-6,
            "top-5 accuracy ({acc5}) should be ≥ top-1 ({acc1})"
        );
    }

    #[test]
    fn test_harmonic_mean_accuracy() {
        let h = ZslMetrics::harmonic_mean_accuracy(0.6, 0.4);
        let expected = 2.0 * 0.6 * 0.4 / (0.6 + 0.4);
        assert!(
            (h - expected).abs() < 1e-6,
            "H-mean mismatch: {h} vs {expected}"
        );
    }

    #[test]
    fn test_harmonic_mean_zero_unseen() {
        let h = ZslMetrics::harmonic_mean_accuracy(0.9, 0.0);
        assert!(
            (h - 0.0).abs() < 1e-6,
            "H-mean with zero unseen should be 0, got {h}"
        );
    }

    #[test]
    fn test_area_under_curve_shape() {
        let seen = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let unseen = vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3];
        let area = ZslMetrics::area_under_seen_unseen_curve(&seen, &unseen);
        assert!(area >= 0.0, "area should be non-negative, got {area}");
        assert!(area.is_finite(), "area should be finite");
    }

    #[test]
    fn test_zsl_eval_report_fields() {
        let report = ZslEvalReport {
            zsl_top1: 0.5,
            zsl_top5: 0.8,
            gzsl_seen: 0.7,
            gzsl_unseen: 0.4,
            gzsl_harmonic: ZslMetrics::harmonic_mean_accuracy(0.7, 0.4),
        };
        assert!(report.zsl_top1 >= 0.0);
        assert!(report.zsl_top5 >= report.zsl_top1 - 1e-6);
        assert!(report.gzsl_harmonic.is_finite());
    }

    #[test]
    fn test_evaluate_runs() {
        let clf = make_classifier();
        let n = 20;
        let visuals: Vec<Vec<f32>> = (0..n).map(|i| make_visual(16, i as u64)).collect();
        // Labels: half from seen (0..4), half from unseen (4..6)
        let labels: Vec<usize> = (0..n).map(|i| i % 6).collect();
        let report = ZslMetrics::evaluate(&clf, &visuals, &labels);
        assert!(report.zsl_top1.is_finite());
        assert!(report.gzsl_harmonic.is_finite());
    }

    #[test]
    fn test_word_vector_creation() {
        let wv = WordVector {
            word: "zebra".into(),
            embedding: vec![0.1, 0.2, 0.3, 0.4],
        };
        assert_eq!(wv.word, "zebra");
        assert_eq!(wv.embedding.len(), 4);
    }
}
