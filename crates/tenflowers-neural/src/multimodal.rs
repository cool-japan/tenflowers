//! Multimodal Fusion Module — Round 13 Track C.
//!
//! Implements CLIP-style contrastive learning, cross-modal attention, and
//! flexible fusion strategies for multimodal deep learning.
//!
//! # Implemented Components
//!
//! | Component | Description |
//! |-----------|-------------|
//! | [`LinearEncoder`] | MLP encoder with Xavier-init, ReLU hidden layer |
//! | [`ClipModel`] | CLIP-style symmetric InfoNCE contrastive learning |
//! | [`CrossModalAttention`] | Multi-head cross-modal attention |
//! | [`MultimodalFusion`] | Early/Late/Gated/Bilinear fusion strategies |
//! | [`MoeLayer`] | Mixture-of-Experts with top-k routing and load balancing |
//!
//! # Quick start
//!
//! ```rust,ignore
//! use tenflowers_neural::multimodal::{ClipConfig, ClipModel};
//! use scirs2_core::random::{rngs::StdRng, SeedableRng};
//!
//! let mut rng = StdRng::seed_from_u64(42);
//! let config = ClipConfig {
//!     embedding_dim: 64,
//!     temperature: 2.659,
//!     batch_size: 4,
//!     learning_rate: 1e-3,
//!     n_epochs: 10,
//!     seed: 42,
//! };
//! let model = ClipModel::new(128, 64, 128, config, &mut rng);
//! let img_embed = model.encode_image(&vec![0.0_f64; 128])?;
//! assert_eq!(img_embed.len(), 64);
//! # Ok::<(), tenflowers_core::TensorError>(())
//! ```

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::TensorError;

// ─────────────────────────────────────────────────────────────────────────────
// Utility math helpers
// ─────────────────────────────────────────────────────────────────────────────

/// L2-normalize a vector so its Euclidean norm equals 1.0.
/// If the norm is zero, returns the zero vector unchanged.
pub fn l2_normalize(v: &[f64]) -> Vec<f64> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < f64::EPSILON {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Dot product of two equal-length slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Cosine similarity between two vectors (assumes neither is zero).
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < f64::EPSILON || norm_b < f64::EPSILON {
        return 0.0;
    }
    dot(a, b) / (norm_a * norm_b)
}

/// Numerically-stable softmax over a 1-D slice.
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum < f64::EPSILON {
        let n = exps.len();
        return vec![1.0 / n as f64; n];
    }
    exps.iter().map(|e| e / sum).collect()
}

/// Cross-entropy loss: `- (1/N) Σ log(prob[i][target[i]])` with `1e-12` clamping.
pub fn cross_entropy_loss(logits: &[Vec<f64>], targets: &[usize]) -> Result<f64, TensorError> {
    if logits.len() != targets.len() {
        return Err(TensorError::invalid_argument_op(
            "cross_entropy_loss",
            "logits and targets must have the same length",
        ));
    }
    if logits.is_empty() {
        return Err(TensorError::invalid_argument_op(
            "cross_entropy_loss",
            "logits must be non-empty",
        ));
    }
    let mut total = 0.0_f64;
    for (row, &t) in logits.iter().zip(targets.iter()) {
        if t >= row.len() {
            return Err(TensorError::invalid_argument_op(
                "cross_entropy_loss",
                "target index out of range for logit row",
            ));
        }
        let probs = softmax(row);
        total -= probs[t].max(1e-12_f64).ln();
    }
    Ok(total / logits.len() as f64)
}

/// Xavier uniform initialisation for a dense weight matrix [rows × cols].
/// Scale = sqrt(6 / (fan_in + fan_out)).
fn xavier_uniform_2d(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let limit = (6.0_f64 / (rows + cols) as f64).sqrt();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    let u: f64 = rng.random();
                    u * 2.0 * limit - limit
                })
                .collect()
        })
        .collect()
}

/// Matrix–vector multiply: W [rows x cols] applied to x [cols] → out [rows].
fn matvec(w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    w.iter().map(|row| dot(row, x)).collect()
}

/// Apply bias and add in-place: `out[i] += bias[i]`.
fn add_bias(out: &mut [f64], bias: &[f64]) {
    for (o, b) in out.iter_mut().zip(bias.iter()) {
        *o += b;
    }
}

/// ReLU activation, element-wise.
fn relu_vec(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| x.max(0.0)).collect()
}

/// Sigmoid activation, element-wise.
fn sigmoid_vec(v: &[f64]) -> Vec<f64> {
    v.iter()
        .map(|&x| {
            if x >= 0.0 {
                let e = (-x).exp();
                1.0 / (1.0 + e)
            } else {
                let e = x.exp();
                e / (1.0 + e)
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Modality Encoders
// ─────────────────────────────────────────────────────────────────────────────

/// Identifies the data modality of a tensor stream.
#[derive(Debug, Clone, PartialEq)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
    Tabular,
    Custom(String),
}

/// Trait implemented by all modality-specific encoders.
pub trait ModalityEncoder {
    /// Project raw input into the encoder's embedding space.
    fn encode(&self, input: &[f64], output_dim: usize) -> Result<Vec<f64>, TensorError>;
    /// Returns the output embedding dimensionality.
    fn output_dim(&self) -> usize;
    /// Returns the data modality this encoder targets.
    fn modality(&self) -> Modality;
}

/// Two-layer MLP encoder: `input → hidden (ReLU) → output`.
/// Weights are Xavier-uniform initialised.
pub struct LinearEncoder {
    input_dim: usize,
    hidden_dim: usize,
    enc_output_dim: usize,
    weights1: Vec<Vec<f64>>, // [hidden_dim x input_dim]
    bias1: Vec<f64>,
    weights2: Vec<Vec<f64>>, // [enc_output_dim x hidden_dim]
    bias2: Vec<f64>,
    enc_modality: Modality,
}

impl LinearEncoder {
    /// Create a new `LinearEncoder` with Xavier-uniform weight initialisation.
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        modality: Modality,
        rng: &mut StdRng,
    ) -> Self {
        let weights1 = xavier_uniform_2d(hidden_dim, input_dim, rng);
        let bias1 = vec![0.0_f64; hidden_dim];
        let weights2 = xavier_uniform_2d(output_dim, hidden_dim, rng);
        let bias2 = vec![0.0_f64; output_dim];
        Self {
            input_dim,
            hidden_dim,
            enc_output_dim: output_dim,
            weights1,
            bias1,
            weights2,
            bias2,
            enc_modality: modality,
        }
    }

    /// Forward pass: linear → ReLU → linear.
    pub fn forward(&self, input: &[f64]) -> Result<Vec<f64>, TensorError> {
        if input.len() != self.input_dim {
            return Err(TensorError::invalid_argument_op(
                "LinearEncoder::forward",
                &format!("expected input_dim={}, got {}", self.input_dim, input.len()),
            ));
        }
        let mut h = matvec(&self.weights1, input);
        add_bias(&mut h, &self.bias1);
        let h = relu_vec(&h);
        let mut out = matvec(&self.weights2, &h);
        add_bias(&mut out, &self.bias2);
        Ok(out)
    }
}

impl ModalityEncoder for LinearEncoder {
    fn encode(&self, input: &[f64], output_dim: usize) -> Result<Vec<f64>, TensorError> {
        if output_dim != self.enc_output_dim {
            return Err(TensorError::invalid_argument_op(
                "LinearEncoder::encode",
                &format!(
                    "requested output_dim={} but encoder has output_dim={}",
                    output_dim, self.enc_output_dim
                ),
            ));
        }
        self.forward(input)
    }

    fn output_dim(&self) -> usize {
        self.enc_output_dim
    }

    fn modality(&self) -> Modality {
        self.enc_modality.clone()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. CLIP-Style Contrastive Learning
// ─────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters for a CLIP-style model.
#[derive(Debug, Clone)]
pub struct ClipConfig {
    /// Final projected embedding dimensionality.
    pub embedding_dim: usize,
    /// Initial log-temperature value (`log(1/0.07) ≈ 2.659`).
    pub temperature: f64,
    /// Mini-batch size used during training.
    pub batch_size: usize,
    /// Gradient descent learning rate (informational; training is external).
    pub learning_rate: f64,
    /// Number of training epochs (informational).
    pub n_epochs: usize,
    /// PRNG seed for weight initialisation.
    pub seed: u64,
}

impl Default for ClipConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 256,
            temperature: 2.659, // log(1/0.07)
            batch_size: 32,
            learning_rate: 1e-3,
            n_epochs: 10,
            seed: 42,
        }
    }
}

/// Per-batch output of the CLIP contrastive loss computation.
#[derive(Debug, Clone)]
pub struct ClipLoss {
    /// Cross-entropy from image→text direction.
    pub image_text_loss: f64,
    /// Cross-entropy from text→image direction.
    pub text_image_loss: f64,
    /// Symmetric loss: `(image_text_loss + text_image_loss) / 2`.
    pub total_loss: f64,
}

/// CLIP-style model with two `LinearEncoder` towers plus linear projection heads.
///
/// Embeddings are L2-normalised before computing similarity, matching the
/// original CLIP paper (Radford et al., 2021).
pub struct ClipModel {
    config: ClipConfig,
    image_encoder: LinearEncoder,
    text_encoder: LinearEncoder,
    /// Log-temperature parameter (scalar).
    pub temperature: f64,
    // Projection heads: [embedding_dim x encoder_dim]
    image_proj: Vec<Vec<f64>>,
    image_proj_bias: Vec<f64>,
    text_proj: Vec<Vec<f64>>,
    text_proj_bias: Vec<f64>,
}

impl ClipModel {
    /// Create a new `ClipModel`.
    ///
    /// Both encoders use a two-layer MLP (`input_dim → encoder_dim/2 → encoder_dim`)
    /// and each is followed by a linear projection head into `config.embedding_dim`.
    pub fn new(
        image_input_dim: usize,
        text_input_dim: usize,
        encoder_dim: usize,
        config: ClipConfig,
        rng: &mut StdRng,
    ) -> Self {
        let hidden_dim = (encoder_dim / 2).max(1);
        let image_encoder = LinearEncoder::new(
            image_input_dim,
            hidden_dim,
            encoder_dim,
            Modality::Image,
            rng,
        );
        let text_encoder =
            LinearEncoder::new(text_input_dim, hidden_dim, encoder_dim, Modality::Text, rng);

        let image_proj = xavier_uniform_2d(config.embedding_dim, encoder_dim, rng);
        let image_proj_bias = vec![0.0_f64; config.embedding_dim];
        let text_proj = xavier_uniform_2d(config.embedding_dim, encoder_dim, rng);
        let text_proj_bias = vec![0.0_f64; config.embedding_dim];

        let temperature = config.temperature;
        Self {
            config,
            image_encoder,
            text_encoder,
            temperature,
            image_proj,
            image_proj_bias,
            text_proj,
            text_proj_bias,
        }
    }

    /// Project encoder output through the image projection head, then L2-normalise.
    fn project_image(&self, enc_out: &[f64]) -> Vec<f64> {
        let mut out = matvec(&self.image_proj, enc_out);
        add_bias(&mut out, &self.image_proj_bias);
        l2_normalize(&out)
    }

    /// Project encoder output through the text projection head, then L2-normalise.
    fn project_text(&self, enc_out: &[f64]) -> Vec<f64> {
        let mut out = matvec(&self.text_proj, enc_out);
        add_bias(&mut out, &self.text_proj_bias);
        l2_normalize(&out)
    }

    /// Encode an image input and return a L2-normalised embedding of length `embedding_dim`.
    pub fn encode_image(&self, image: &[f64]) -> Result<Vec<f64>, TensorError> {
        let enc = self.image_encoder.forward(image)?;
        Ok(self.project_image(&enc))
    }

    /// Encode a text input and return a L2-normalised embedding of length `embedding_dim`.
    pub fn encode_text(&self, text: &[f64]) -> Result<Vec<f64>, TensorError> {
        let enc = self.text_encoder.forward(text)?;
        Ok(self.project_text(&enc))
    }

    /// Compute the `N×N` cosine similarity matrix scaled by `exp(temperature)`.
    ///
    /// `similarity[i][j] = image_embed[i] · text_embed[j] * exp(temperature)`
    pub fn similarity_matrix(
        &self,
        image_embeds: &[Vec<f64>],
        text_embeds: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, TensorError> {
        if image_embeds.is_empty() || text_embeds.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "ClipModel::similarity_matrix",
                "embedding sets must be non-empty",
            ));
        }
        let scale = self.temperature.exp();
        let sim = image_embeds
            .iter()
            .map(|img| {
                text_embeds
                    .iter()
                    .map(|txt| dot(img, txt) * scale)
                    .collect::<Vec<_>>()
            })
            .collect();
        Ok(sim)
    }

    /// Symmetric InfoNCE (CLIP) contrastive loss.
    ///
    /// Diagonal entries are treated as positive pairs.  The loss is the average
    /// of the two directional cross-entropies.
    pub fn contrastive_loss(
        &self,
        image_embeds: &[Vec<f64>],
        text_embeds: &[Vec<f64>],
    ) -> Result<ClipLoss, TensorError> {
        let n = image_embeds.len();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "ClipModel::contrastive_loss",
                "batch must be non-empty",
            ));
        }
        if text_embeds.len() != n {
            return Err(TensorError::invalid_argument_op(
                "ClipModel::contrastive_loss",
                "image_embeds and text_embeds must have equal length",
            ));
        }

        let sim = self.similarity_matrix(image_embeds, text_embeds)?;
        let targets: Vec<usize> = (0..n).collect();

        let image_text_loss = cross_entropy_loss(&sim, &targets)?;

        // Transpose sim for text→image direction
        let sim_t: Vec<Vec<f64>> = (0..n)
            .map(|j| (0..n).map(|i| sim[i][j]).collect())
            .collect();
        let text_image_loss = cross_entropy_loss(&sim_t, &targets)?;

        let total_loss = (image_text_loss + text_image_loss) / 2.0;
        Ok(ClipLoss {
            image_text_loss,
            text_image_loss,
            total_loss,
        })
    }

    /// Zero-shot classification: returns the index of the class whose text
    /// embedding has the highest cosine similarity with the image embedding.
    pub fn zero_shot_classify(
        &self,
        image: &[f64],
        class_text_embeds: &[Vec<f64>],
    ) -> Result<usize, TensorError> {
        if class_text_embeds.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "ClipModel::zero_shot_classify",
                "class_text_embeds must be non-empty",
            ));
        }
        let img_embed = self.encode_image(image)?;
        let scale = self.temperature.exp();
        let best = class_text_embeds
            .iter()
            .enumerate()
            .map(|(i, te)| (i, dot(&img_embed, te) * scale))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(best)
    }

    /// Text-image retrieval: return the top-k text indices (by similarity) for the
    /// given image embedding.  Returns at most `k` indices sorted by descending similarity.
    pub fn retrieve_text(
        &self,
        image: &[f64],
        text_embeds: &[Vec<f64>],
        k: usize,
    ) -> Result<Vec<usize>, TensorError> {
        if text_embeds.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "ClipModel::retrieve_text",
                "text_embeds must be non-empty",
            ));
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        let img_embed = self.encode_image(image)?;
        let scale = self.temperature.exp();
        let mut scored: Vec<(usize, f64)> = text_embeds
            .iter()
            .enumerate()
            .map(|(i, te)| (i, dot(&img_embed, te) * scale))
            .collect();
        scored.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().take(k).map(|(i, _)| i).collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Cross-Modal Attention
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for `CrossModalAttention`.
#[derive(Debug, Clone)]
pub struct CrossModalAttentionConfig {
    /// Dimensionality of the query modality.
    pub query_dim: usize,
    /// Dimensionality of the key/value modality.
    pub key_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Dropout rate (stored but not applied during inference).
    pub dropout: f64,
    /// PRNG seed for weight initialisation.
    pub seed: u64,
}

/// Multi-head cross-modal attention.
///
/// Queries come from modality A; keys and values from modality B.
/// Projects queries using `W_Q`, keys using `W_K`, values using `W_V`,
/// computes scaled dot-product attention per head, then projects output
/// with `W_O`.
pub struct CrossModalAttention {
    config: CrossModalAttentionConfig,
    head_dim: usize,
    /// W_Q: [query_dim x (n_heads * head_dim)]
    wq: Vec<Vec<f64>>,
    /// W_K: [key_dim x (n_heads * head_dim)]
    wk: Vec<Vec<f64>>,
    /// W_V: [key_dim x (n_heads * head_dim)]
    wv: Vec<Vec<f64>>,
    /// W_O: [(n_heads * head_dim) x query_dim]
    wo: Vec<Vec<f64>>,
    bq: Vec<f64>,
    bk: Vec<f64>,
    bv: Vec<f64>,
    bo: Vec<f64>,
}

impl CrossModalAttention {
    /// Create a new `CrossModalAttention` with Xavier-uniform weight initialisation.
    ///
    /// Projection dimensions:
    /// - W_Q: `[total_dim x query_dim]`  (query_dim → total_dim)
    /// - W_K: `[total_dim x key_dim]`    (key_dim   → total_dim)
    /// - W_V: `[total_dim x key_dim]`    (key_dim   → total_dim)
    /// - W_O: `[query_dim x total_dim]`  (total_dim → query_dim)
    pub fn new(config: CrossModalAttentionConfig, rng: &mut StdRng) -> Self {
        let n_heads = config.n_heads.max(1);
        // head_dim derived from query_dim, divided by n_heads
        let head_dim = (config.query_dim / n_heads).max(1);
        let total_dim = n_heads * head_dim;

        // Rows = output size, cols = input size for matvec
        let wq = xavier_uniform_2d(total_dim, config.query_dim, rng);
        let wk = xavier_uniform_2d(total_dim, config.key_dim, rng);
        let wv = xavier_uniform_2d(total_dim, config.key_dim, rng);
        let wo = xavier_uniform_2d(config.query_dim, total_dim, rng);

        let bq = vec![0.0; total_dim];
        let bk = vec![0.0; total_dim];
        let bv = vec![0.0; total_dim];
        let bo = vec![0.0; config.query_dim];

        Self {
            config,
            head_dim,
            wq,
            wk,
            wv,
            wo,
            bq,
            bk,
            bv,
            bo,
        }
    }

    /// Project a sequence of vectors: `seq [T x in_dim]` → `seq [T x out_dim]`.
    fn project_seq(w: &[Vec<f64>], b: &[f64], seq: &[Vec<f64>]) -> Vec<Vec<f64>> {
        seq.iter()
            .map(|x| {
                let mut out = matvec(w, x);
                add_bias(&mut out, b);
                out
            })
            .collect()
    }

    /// Scaled dot-product attention for a single head.
    ///
    /// `q`: `[seq_q x head_dim]`, `k`/`v`: `[seq_k x head_dim]`.
    /// Returns `[seq_q x head_dim]`.
    fn attention(
        &self,
        q: &[Vec<f64>],
        k: &[Vec<f64>],
        v: &[Vec<f64>],
        scale: f64,
    ) -> Result<Vec<Vec<f64>>, TensorError> {
        if q.is_empty() || k.is_empty() || v.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "CrossModalAttention::attention",
                "query, key, and value sequences must be non-empty",
            ));
        }
        if k.len() != v.len() {
            return Err(TensorError::invalid_argument_op(
                "CrossModalAttention::attention",
                "key and value sequences must have the same length",
            ));
        }
        // Compute attention scores: [seq_q x seq_k]
        let out = q
            .iter()
            .map(|qi| {
                let scores: Vec<f64> = k.iter().map(|kj| dot(qi, kj) * scale).collect();
                let weights = softmax(&scores);
                // weighted sum of values
                let head_dim = v[0].len();
                let mut result = vec![0.0_f64; head_dim];
                for (w_j, vj) in weights.iter().zip(v.iter()) {
                    for (r, vk) in result.iter_mut().zip(vj.iter()) {
                        *r += w_j * vk;
                    }
                }
                result
            })
            .collect();
        Ok(out)
    }

    /// Full multi-head cross-modal attention forward pass.
    ///
    /// `query`: `[seq_q x query_dim]`, `key`/`value`: `[seq_k x key_dim]`.
    /// Returns `[seq_q x query_dim]`.
    pub fn forward(
        &self,
        query: &[Vec<f64>],
        key: &[Vec<f64>],
        value: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, TensorError> {
        if query.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "CrossModalAttention::forward",
                "query sequence must be non-empty",
            ));
        }
        if key.len() != value.len() {
            return Err(TensorError::invalid_argument_op(
                "CrossModalAttention::forward",
                "key and value sequences must have the same length",
            ));
        }

        let n_heads = self.config.n_heads.max(1);
        let hd = self.head_dim;
        let scale = 1.0 / (hd as f64).sqrt();

        // Project all sequences
        let q_proj = Self::project_seq(&self.wq, &self.bq, query);
        let k_proj = Self::project_seq(&self.wk, &self.bk, key);
        let v_proj = Self::project_seq(&self.wv, &self.bv, value);

        let seq_q = query.len();

        // Accumulate heads
        let mut concat = vec![vec![0.0_f64; n_heads * hd]; seq_q];
        for h in 0..n_heads {
            let start = h * hd;
            let end = start + hd;
            // Slice head h out of projected sequences
            let q_h: Vec<Vec<f64>> = q_proj.iter().map(|v| v[start..end].to_vec()).collect();
            let k_h: Vec<Vec<f64>> = k_proj.iter().map(|v| v[start..end].to_vec()).collect();
            let v_h: Vec<Vec<f64>> = v_proj.iter().map(|v| v[start..end].to_vec()).collect();

            let attended = self.attention(&q_h, &k_h, &v_h, scale)?;
            for (i, row) in attended.iter().enumerate() {
                for (j, &val) in row.iter().enumerate() {
                    concat[i][start + j] += val;
                }
            }
        }

        // Output projection: [n_heads*hd x query_dim]
        let out: Vec<Vec<f64>> = concat
            .iter()
            .map(|row| {
                let mut o = matvec(&self.wo, row);
                add_bias(&mut o, &self.bo);
                o
            })
            .collect();
        Ok(out)
    }

    /// Compute aggregated attention weights averaged over all heads.
    ///
    /// Returns `[seq_q x seq_k]` weight matrix (rows sum to 1).
    pub fn attention_weights(
        &self,
        query: &[Vec<f64>],
        key: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, TensorError> {
        if query.is_empty() || key.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "CrossModalAttention::attention_weights",
                "query and key sequences must be non-empty",
            ));
        }
        let n_heads = self.config.n_heads.max(1);
        let hd = self.head_dim;
        let scale = 1.0 / (hd as f64).sqrt();

        let q_proj = Self::project_seq(&self.wq, &self.bq, query);
        let k_proj = Self::project_seq(&self.wk, &self.bk, key);

        let seq_q = query.len();
        let seq_k = key.len();
        let mut avg_weights = vec![vec![0.0_f64; seq_k]; seq_q];

        for h in 0..n_heads {
            let start = h * hd;
            let end = start + hd;
            let q_h: Vec<Vec<f64>> = q_proj.iter().map(|v| v[start..end].to_vec()).collect();
            let k_h: Vec<Vec<f64>> = k_proj.iter().map(|v| v[start..end].to_vec()).collect();

            for (i, qi) in q_h.iter().enumerate() {
                let scores: Vec<f64> = k_h.iter().map(|kj| dot(qi, kj) * scale).collect();
                let weights = softmax(&scores);
                for (j, &w) in weights.iter().enumerate() {
                    avg_weights[i][j] += w / n_heads as f64;
                }
            }
        }
        Ok(avg_weights)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Fusion Strategies
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy used by [`MultimodalFusion`] to combine modality representations.
#[derive(Debug, Clone)]
pub enum FusionStrategy {
    /// Concatenate raw inputs before any processing.
    EarlyFusion,
    /// Encode each modality independently then concatenate.
    LateFusion,
    /// Hierarchical fusion at multiple encoding levels.
    HybridFusion,
    /// Use cross-modal attention to integrate modalities.
    CrossAttention,
    /// Use sigmoid-gated learned weights per modality.
    GatedFusion,
    /// Approximate bilinear (outer-product) pooling.
    BilinearFusion,
}

/// Configuration for [`MultimodalFusion`].
#[derive(Debug, Clone)]
pub struct ModalFusionConfig {
    /// Strategy to use when fusing modality embeddings.
    pub strategy: FusionStrategy,
    /// Per-modality input dimensionalities.
    pub input_dims: Vec<usize>,
    /// Output embedding dimensionality after fusion.
    pub output_dim: usize,
    /// PRNG seed for weight initialisation.
    pub seed: u64,
}

/// Output produced by [`MultimodalFusion::fuse`].
#[derive(Debug, Clone)]
pub struct FusionResult {
    /// Fused multimodal embedding.
    pub fused_embedding: Vec<f64>,
    /// Per-modality weight/gate value (0.0–1.0).
    pub modality_weights: Vec<f64>,
    /// Fusion strategy that was applied.
    pub strategy: FusionStrategy,
}

/// Multimodal fusion module supporting six strategies.
pub struct MultimodalFusion {
    config: ModalFusionConfig,
    /// Per-modality projection: `(weights [proj_dim x input_dim], bias [proj_dim])`
    projections: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    /// Final linear layer applied after fusion.
    fusion_weights: Vec<Vec<f64>>,
    fusion_bias: Vec<f64>,
    /// Gating network weights for `GatedFusion`.
    gate_weights: Vec<Vec<f64>>,
    gate_bias: Vec<f64>,
}

impl MultimodalFusion {
    /// Create a new `MultimodalFusion` module.
    ///
    /// Each modality is projected to `proj_dim = output_dim` before the chosen
    /// fusion strategy is applied.
    pub fn new(config: ModalFusionConfig, rng: &mut StdRng) -> Self {
        let n_mod = config.input_dims.len().max(1);
        let proj_dim = config.output_dim;

        let projections = config
            .input_dims
            .iter()
            .map(|&in_dim| {
                let w = xavier_uniform_2d(proj_dim, in_dim, rng);
                let b = vec![0.0_f64; proj_dim];
                (w, b)
            })
            .collect::<Vec<_>>();

        // Fusion layer input size depends on strategy
        let fusion_in = match &config.strategy {
            FusionStrategy::EarlyFusion => config.input_dims.iter().sum::<usize>(),
            FusionStrategy::BilinearFusion => {
                // element-wise product of the first two projected modalities
                proj_dim
            }
            _ => proj_dim * n_mod,
        };

        let fusion_weights = xavier_uniform_2d(config.output_dim, fusion_in, rng);
        let fusion_bias = vec![0.0_f64; config.output_dim];

        // Gate: for each position in the concatenated projected space, output n_mod scalars
        let gate_weights = xavier_uniform_2d(n_mod, proj_dim, rng);
        let gate_bias = vec![0.0_f64; n_mod];

        Self {
            config,
            projections,
            fusion_weights,
            fusion_bias,
            gate_weights,
            gate_bias,
        }
    }

    /// Fuse a slice of per-modality input vectors.
    ///
    /// `modality_inputs[i]` must have length `config.input_dims[i]`.
    pub fn fuse(&self, modality_inputs: &[Vec<f64>]) -> Result<FusionResult, TensorError> {
        if modality_inputs.len() != self.config.input_dims.len() {
            return Err(TensorError::invalid_argument_op(
                "MultimodalFusion::fuse",
                &format!(
                    "expected {} modality inputs, got {}",
                    self.config.input_dims.len(),
                    modality_inputs.len()
                ),
            ));
        }

        match &self.config.strategy {
            FusionStrategy::EarlyFusion | FusionStrategy::HybridFusion => {
                self.early_fuse(modality_inputs)
            }
            FusionStrategy::LateFusion | FusionStrategy::CrossAttention => {
                self.late_fuse(modality_inputs)
            }
            FusionStrategy::GatedFusion => self.gated_fuse(modality_inputs),
            FusionStrategy::BilinearFusion => {
                if modality_inputs.len() < 2 {
                    return Err(TensorError::invalid_argument_op(
                        "MultimodalFusion::fuse",
                        "BilinearFusion requires at least 2 modalities",
                    ));
                }
                // Project first two modalities then element-wise product
                let proj_a = self.project_modality(0, &modality_inputs[0])?;
                let proj_b = self.project_modality(1, &modality_inputs[1])?;
                let fused = self.bilinear_fuse(&proj_a, &proj_b)?;
                let n_mod = modality_inputs.len();
                Ok(FusionResult {
                    fused_embedding: fused,
                    modality_weights: vec![1.0 / n_mod as f64; n_mod],
                    strategy: FusionStrategy::BilinearFusion,
                })
            }
        }
    }

    /// Project the i-th modality input through its linear projection.
    fn project_modality(&self, idx: usize, input: &[f64]) -> Result<Vec<f64>, TensorError> {
        let (ref w, ref b) = self.projections[idx];
        if input.len() != self.config.input_dims[idx] {
            return Err(TensorError::invalid_argument_op(
                "MultimodalFusion::project_modality",
                &format!(
                    "modality {} expects input_dim={}, got {}",
                    idx,
                    self.config.input_dims[idx],
                    input.len()
                ),
            ));
        }
        let mut out = matvec(w, input);
        add_bias(&mut out, b);
        Ok(out)
    }

    /// Early fusion: concatenate raw inputs, then apply fusion layer with ReLU.
    fn early_fuse(&self, inputs: &[Vec<f64>]) -> Result<FusionResult, TensorError> {
        let concatenated: Vec<f64> = inputs.iter().flat_map(|v| v.iter().cloned()).collect();
        let mut out = matvec(&self.fusion_weights, &concatenated);
        add_bias(&mut out, &self.fusion_bias);
        let out = relu_vec(&out);
        let n = inputs.len();
        Ok(FusionResult {
            fused_embedding: out,
            modality_weights: vec![1.0 / n as f64; n],
            strategy: FusionStrategy::EarlyFusion,
        })
    }

    /// Late fusion: project each modality, concatenate, apply fusion layer with ReLU.
    fn late_fuse(&self, inputs: &[Vec<f64>]) -> Result<FusionResult, TensorError> {
        let mut projected_cat = Vec::new();
        for (i, inp) in inputs.iter().enumerate() {
            let proj = self.project_modality(i, inp)?;
            projected_cat.extend(proj);
        }
        let mut out = matvec(&self.fusion_weights, &projected_cat);
        add_bias(&mut out, &self.fusion_bias);
        let out = relu_vec(&out);
        let n = inputs.len();
        Ok(FusionResult {
            fused_embedding: out,
            modality_weights: vec![1.0 / n as f64; n],
            strategy: FusionStrategy::LateFusion,
        })
    }

    /// Gated fusion: learn a sigmoid gate per modality, weighted sum of projections.
    fn gated_fuse(&self, inputs: &[Vec<f64>]) -> Result<FusionResult, TensorError> {
        let n = inputs.len();
        // Project each modality
        let projs: Vec<Vec<f64>> = (0..n)
            .map(|i| self.project_modality(i, &inputs[i]))
            .collect::<Result<Vec<_>, _>>()?;

        // Use the mean projected embedding to compute gate logits
        let proj_dim = self.config.output_dim;
        let mut mean_proj = vec![0.0_f64; proj_dim];
        for p in &projs {
            for (m, &v) in mean_proj.iter_mut().zip(p.iter()) {
                *m += v / n as f64;
            }
        }
        let mut gate_logits = matvec(&self.gate_weights, &mean_proj);
        add_bias(&mut gate_logits, &self.gate_bias);
        let gates = sigmoid_vec(&gate_logits); // n values in (0,1)

        // Normalise gates so they sum to 1 (soft attention over modalities)
        let gate_sum: f64 = gates.iter().sum();
        let gate_sum = if gate_sum < f64::EPSILON {
            1.0
        } else {
            gate_sum
        };
        let norm_gates: Vec<f64> = gates.iter().map(|&g| g / gate_sum).collect();

        // Weighted sum of projected modalities
        let mut fused = vec![0.0_f64; proj_dim];
        for (proj, &g) in projs.iter().zip(norm_gates.iter()) {
            for (f, &p) in fused.iter_mut().zip(proj.iter()) {
                *f += g * p;
            }
        }
        // Final projection + ReLU
        let mut out = matvec(&self.fusion_weights, &fused);
        add_bias(&mut out, &self.fusion_bias);
        let out = relu_vec(&out);

        Ok(FusionResult {
            fused_embedding: out,
            modality_weights: norm_gates,
            strategy: FusionStrategy::GatedFusion,
        })
    }

    /// Bilinear fusion (Tucker approximation): element-wise product of projected embeddings.
    ///
    /// Approximates `W ⋅ (a ⊗ b)` as `project(a) ⊙ project(b)`.
    fn bilinear_fuse(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, TensorError> {
        if a.len() != b.len() {
            return Err(TensorError::invalid_argument_op(
                "MultimodalFusion::bilinear_fuse",
                "both projected embeddings must have the same length",
            ));
        }
        // element-wise product ← approximates outer product projection
        let product: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();
        // Apply fusion layer
        let mut out = matvec(&self.fusion_weights, &product);
        add_bias(&mut out, &self.fusion_bias);
        Ok(relu_vec(&out))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Multimodal Training Utilities
// ─────────────────────────────────────────────────────────────────────────────

/// A mini-batch of multi-modal data.
#[derive(Debug, Clone)]
pub struct MultimodalBatch {
    /// `modality_inputs[m][b]` is the input for modality `m`, sample `b`.
    pub modality_inputs: Vec<Vec<Vec<f64>>>,
    /// Optional class labels for each sample.
    pub labels: Option<Vec<usize>>,
    /// Optional positive pair indices `(i, j)` for contrastive learning.
    pub contrastive_pairs: Option<Vec<(usize, usize)>>,
}

/// Retrieval and alignment metrics for paired embeddings.
#[derive(Debug, Clone)]
pub struct MultimodalMetrics {
    /// Image→Text Recall@1 (diagonal element is rank-1 match).
    pub retrieval_recall_at_1: f64,
    /// Image→Text Recall@5.
    pub retrieval_recall_at_5: f64,
    /// Image→Text Recall@10.
    pub retrieval_recall_at_10: f64,
    /// Mean Reciprocal Rank (MRR) over all queries.
    pub mean_reciprocal_rank: f64,
    /// Average cosine similarity between matched image/text pairs.
    pub modal_alignment_score: f64,
}

/// Compute standard retrieval metrics given paired normalised embeddings.
///
/// Assumes `image_embeds[i]` and `text_embeds[i]` form a positive pair,
/// and that both embedding sets are already L2-normalised.
pub fn compute_multimodal_metrics(
    image_embeds: &[Vec<f64>],
    text_embeds: &[Vec<f64>],
) -> Result<MultimodalMetrics, TensorError> {
    let n = image_embeds.len();
    if n == 0 {
        return Err(TensorError::invalid_argument_op(
            "compute_multimodal_metrics",
            "embedding sets must be non-empty",
        ));
    }
    if text_embeds.len() != n {
        return Err(TensorError::invalid_argument_op(
            "compute_multimodal_metrics",
            "image_embeds and text_embeds must have equal length",
        ));
    }

    let mut r1 = 0.0_f64;
    let mut r5 = 0.0_f64;
    let mut r10 = 0.0_f64;
    let mut mrr = 0.0_f64;
    let mut alignment = 0.0_f64;

    for (i, img) in image_embeds.iter().enumerate() {
        // Cosine similarities for query i against all texts
        let mut sims: Vec<(usize, f64)> = text_embeds
            .iter()
            .enumerate()
            .map(|(j, txt)| (j, cosine_similarity(img, txt)))
            .collect();
        sims.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Rank of the correct text (0-indexed)
        let rank = sims.iter().position(|(j, _)| *j == i).unwrap_or(n);
        let rank_1indexed = rank + 1;

        if rank == 0 {
            r1 += 1.0;
        }
        if rank < 5 {
            r5 += 1.0;
        }
        if rank < 10 {
            r10 += 1.0;
        }
        mrr += 1.0 / rank_1indexed as f64;
        alignment += cosine_similarity(img, &text_embeds[i]);
    }

    let n_f = n as f64;
    Ok(MultimodalMetrics {
        retrieval_recall_at_1: r1 / n_f,
        retrieval_recall_at_5: r5 / n_f,
        retrieval_recall_at_10: r10 / n_f,
        mean_reciprocal_rank: mrr / n_f,
        modal_alignment_score: alignment / n_f,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Mixture of Experts (MoE)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a `MoeLayer`.
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Total number of expert networks.
    pub n_experts: usize,
    /// Number of experts activated per forward call (top-k routing).
    pub top_k: usize,
    /// Input dimensionality.
    pub input_dim: usize,
    /// Hidden dimensionality inside each expert.
    pub expert_dim: usize,
    /// Output dimensionality.
    pub output_dim: usize,
    /// Auxiliary load-balancing loss coefficient.
    pub load_balancing_coeff: f64,
    /// PRNG seed.
    pub seed: u64,
}

/// Sparse Mixture-of-Experts layer with top-k routing.
///
/// Each expert is a two-layer MLP (`input_dim → expert_dim → output_dim`).
/// The gating network (`input_dim → n_experts`) produces router probabilities;
/// the top-k experts' outputs are combined with their softmax-normalised gate
/// weights.
pub struct MoeLayer {
    config: MoeConfig,
    /// Gating network weights: `[n_experts x input_dim]`.
    gate_weights: Vec<Vec<f64>>,
    gate_bias: Vec<f64>,
    /// Expert first-layer weights: `[n_experts][expert_dim x input_dim]`.
    expert_w1: Vec<Vec<Vec<f64>>>,
    expert_b1: Vec<Vec<f64>>,
    /// Expert second-layer weights: `[n_experts][output_dim x expert_dim]`.
    expert_w2: Vec<Vec<Vec<f64>>>,
    expert_b2: Vec<Vec<f64>>,
}

impl MoeLayer {
    /// Create a new `MoeLayer` with Xavier-uniform weight initialisation.
    pub fn new(config: MoeConfig, rng: &mut StdRng) -> Self {
        let ne = config.n_experts.max(1);
        let gate_weights = xavier_uniform_2d(ne, config.input_dim, rng);
        let gate_bias = vec![0.0_f64; ne];

        let mut expert_w1 = Vec::with_capacity(ne);
        let mut expert_b1 = Vec::with_capacity(ne);
        let mut expert_w2 = Vec::with_capacity(ne);
        let mut expert_b2 = Vec::with_capacity(ne);

        for _ in 0..ne {
            expert_w1.push(xavier_uniform_2d(config.expert_dim, config.input_dim, rng));
            expert_b1.push(vec![0.0_f64; config.expert_dim]);
            expert_w2.push(xavier_uniform_2d(config.output_dim, config.expert_dim, rng));
            expert_b2.push(vec![0.0_f64; config.output_dim]);
        }

        Self {
            config,
            gate_weights,
            gate_bias,
            expert_w1,
            expert_b1,
            expert_w2,
            expert_b2,
        }
    }

    /// Compute gating probabilities (softmax over all experts).
    fn gate_probs(&self, input: &[f64]) -> Vec<f64> {
        let mut logits = matvec(&self.gate_weights, input);
        add_bias(&mut logits, &self.gate_bias);
        softmax(&logits)
    }

    /// Run the two-layer MLP for expert `idx`.
    fn expert_forward(&self, idx: usize, input: &[f64]) -> Vec<f64> {
        let mut h = matvec(&self.expert_w1[idx], input);
        add_bias(&mut h, &self.expert_b1[idx]);
        let h = relu_vec(&h);
        let mut out = matvec(&self.expert_w2[idx], &h);
        add_bias(&mut out, &self.expert_b2[idx]);
        out
    }

    /// Forward pass with top-k sparse routing.
    ///
    /// Returns `(output [output_dim], selected_expert_indices, gate_weights_for_selected)`.
    pub fn forward(&self, input: &[f64]) -> Result<(Vec<f64>, Vec<usize>, Vec<f64>), TensorError> {
        if input.len() != self.config.input_dim {
            return Err(TensorError::invalid_argument_op(
                "MoeLayer::forward",
                &format!(
                    "expected input_dim={}, got {}",
                    self.config.input_dim,
                    input.len()
                ),
            ));
        }
        let ne = self.config.n_experts.max(1);
        let k = self.config.top_k.min(ne).max(1);

        let probs = self.gate_probs(input);

        // Select top-k expert indices
        let mut indexed: Vec<(usize, f64)> = probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let top_k_indices: Vec<usize> = indexed.iter().take(k).map(|(i, _)| *i).collect();

        // Re-normalise selected gate weights
        let top_k_probs_raw: Vec<f64> = top_k_indices.iter().map(|&i| probs[i]).collect();
        let prob_sum: f64 = top_k_probs_raw.iter().sum();
        let prob_sum = if prob_sum < f64::EPSILON {
            1.0
        } else {
            prob_sum
        };
        let top_k_probs: Vec<f64> = top_k_probs_raw.iter().map(|&p| p / prob_sum).collect();

        // Weighted sum of expert outputs
        let mut output = vec![0.0_f64; self.config.output_dim];
        for (&expert_idx, &weight) in top_k_indices.iter().zip(top_k_probs.iter()) {
            let expert_out = self.expert_forward(expert_idx, input);
            for (o, &e) in output.iter_mut().zip(expert_out.iter()) {
                *o += weight * e;
            }
        }

        Ok((output, top_k_indices, top_k_probs))
    }

    /// Auxiliary load-balancing loss (Switch Transformer style).
    ///
    /// `gate_probs_batch[b]` is the full softmax probability vector for sample `b`.
    /// Loss = n_experts * Σ_e (f_e * p_e) where f_e is the fraction of tokens
    /// routed to expert e and p_e is the average gate probability for expert e.
    pub fn load_balancing_loss(&self, gate_probs_batch: &[Vec<f64>]) -> f64 {
        if gate_probs_batch.is_empty() {
            return 0.0;
        }
        let ne = self.config.n_experts.max(1);
        let k = self.config.top_k.min(ne).max(1);
        let batch = gate_probs_batch.len();

        // f_e: fraction of tokens for which expert e is in the top-k
        let mut counts = vec![0.0_f64; ne];
        let mut avg_probs = vec![0.0_f64; ne];

        for probs in gate_probs_batch {
            let mut indexed: Vec<(usize, f64)> = probs.iter().cloned().enumerate().collect();
            indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            for (i, _) in indexed.iter().take(k) {
                counts[*i] += 1.0;
            }
            for (e, &p) in probs.iter().enumerate().take(ne) {
                avg_probs[e] += p;
            }
        }

        let batch_f = batch as f64;
        let f: Vec<f64> = counts.iter().map(|&c| c / batch_f).collect();
        let p: Vec<f64> = avg_probs.iter().map(|&s| s / batch_f).collect();

        let loss: f64 = f.iter().zip(p.iter()).map(|(&fe, &pe)| fe * pe).sum();
        (ne as f64) * loss
    }

    /// Compute expert utilisation as a fraction of tokens routed to each expert
    /// across the given batch of expert index selections.
    pub fn expert_utilization(&self, expert_indices_batch: &[Vec<usize>]) -> Vec<f64> {
        let ne = self.config.n_experts.max(1);
        if expert_indices_batch.is_empty() {
            return vec![0.0; ne];
        }
        let mut counts = vec![0.0_f64; ne];
        let mut total = 0.0_f64;
        for indices in expert_indices_batch {
            for &idx in indices {
                if idx < ne {
                    counts[idx] += 1.0;
                    total += 1.0;
                }
            }
        }
        if total < f64::EPSILON {
            return vec![0.0; ne];
        }
        counts.iter().map(|&c| c / total).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::{rngs::StdRng, SeedableRng};

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    fn rand_vec(dim: usize, seed: u64) -> Vec<f64> {
        let mut rng = make_rng(seed);
        (0..dim).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect()
    }

    fn make_clip(embed: usize, enc: usize) -> ClipModel {
        let mut rng = make_rng(0);
        let config = ClipConfig {
            embedding_dim: embed,
            temperature: 2.659,
            batch_size: 4,
            learning_rate: 1e-3,
            n_epochs: 1,
            seed: 0,
        };
        ClipModel::new(32, 16, enc, config, &mut rng)
    }

    // ── 1. LinearEncoder ─────────────────────────────────────────────────────

    #[test]
    fn test_linear_encoder_forward_shape() {
        let mut rng = make_rng(1);
        let enc = LinearEncoder::new(8, 16, 4, Modality::Image, &mut rng);
        let out = enc.forward(&[0.5_f64; 8]).expect("should succeed");
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_linear_encoder_output_dim() {
        let mut rng = make_rng(2);
        let enc = LinearEncoder::new(10, 20, 6, Modality::Text, &mut rng);
        assert_eq!(enc.output_dim(), 6);
    }

    #[test]
    fn test_linear_encoder_modality() {
        let mut rng = make_rng(3);
        let enc = LinearEncoder::new(4, 8, 4, Modality::Audio, &mut rng);
        assert_eq!(enc.modality(), Modality::Audio);
    }

    #[test]
    fn test_linear_encoder_wrong_input_dim() {
        let mut rng = make_rng(4);
        let enc = LinearEncoder::new(8, 16, 4, Modality::Image, &mut rng);
        let result = enc.forward(&[0.0_f64; 5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_encoder_trait_encode_correct() {
        let mut rng = make_rng(5);
        let enc = LinearEncoder::new(8, 16, 4, Modality::Tabular, &mut rng);
        let out = enc.encode(&[1.0_f64; 8], 4).expect("should succeed");
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_linear_encoder_trait_encode_wrong_dim() {
        let mut rng = make_rng(6);
        let enc = LinearEncoder::new(8, 16, 4, Modality::Text, &mut rng);
        let result = enc.encode(&[0.0_f64; 8], 7);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_encoder_custom_modality() {
        let mut rng = make_rng(7);
        let enc = LinearEncoder::new(4, 8, 4, Modality::Custom("lidar".to_string()), &mut rng);
        assert_eq!(enc.modality(), Modality::Custom("lidar".to_string()));
    }

    // ── 2. ClipModel ─────────────────────────────────────────────────────────

    #[test]
    fn test_clip_encode_image_unit_norm() {
        let model = make_clip(16, 32);
        let img = rand_vec(32, 10);
        let embed = model.encode_image(&img).expect("should succeed");
        let norm: f64 = embed.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9, "norm={}", norm);
    }

    #[test]
    fn test_clip_encode_text_unit_norm() {
        let model = make_clip(16, 32);
        let txt = rand_vec(16, 11);
        let embed = model.encode_text(&txt).expect("should succeed");
        let norm: f64 = embed.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9, "norm={}", norm);
    }

    #[test]
    fn test_clip_encode_image_output_dim() {
        let model = make_clip(16, 32);
        let img = rand_vec(32, 12);
        let embed = model.encode_image(&img).expect("should succeed");
        assert_eq!(embed.len(), 16);
    }

    #[test]
    fn test_clip_similarity_matrix_shape() {
        let model = make_clip(16, 32);
        let imgs: Vec<Vec<f64>> = (0..4)
            .map(|i| l2_normalize(&rand_vec(16, i as u64)))
            .collect();
        let txts: Vec<Vec<f64>> = (0..4)
            .map(|i| l2_normalize(&rand_vec(16, i as u64 + 100)))
            .collect();
        let sim = model
            .similarity_matrix(&imgs, &txts)
            .expect("should succeed");
        assert_eq!(sim.len(), 4);
        assert_eq!(sim[0].len(), 4);
    }

    #[test]
    fn test_clip_similarity_matrix_diagonal_dominance() {
        // Use the same embedding for image and text → diagonal should be max
        let model = make_clip(16, 32);
        let embeds: Vec<Vec<f64>> = (0..3)
            .map(|i| l2_normalize(&rand_vec(16, i as u64 + 200)))
            .collect();
        let sim = model
            .similarity_matrix(&embeds, &embeds)
            .expect("should succeed");
        for (i, row) in sim.iter().enumerate() {
            let diag = row[i];
            for (j, &s) in row.iter().enumerate() {
                if j != i {
                    assert!(
                        diag >= s - 1e-9,
                        "diagonal {diag} < off-diagonal {s} at row {i} col {j}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_clip_contrastive_loss_positive() {
        let model = make_clip(16, 32);
        let imgs: Vec<Vec<f64>> = (0..3)
            .map(|i| l2_normalize(&rand_vec(16, i as u64)))
            .collect();
        let txts: Vec<Vec<f64>> = (0..3)
            .map(|i| l2_normalize(&rand_vec(16, i as u64 + 50)))
            .collect();
        let loss = model
            .contrastive_loss(&imgs, &txts)
            .expect("should succeed");
        assert!(loss.total_loss > 0.0);
        assert!(
            (loss.total_loss - (loss.image_text_loss + loss.text_image_loss) / 2.0).abs() < 1e-9
        );
    }

    #[test]
    fn test_clip_contrastive_loss_perfect_alignment() {
        // When image and text embeds are identical, loss should be low (min per-class entropy)
        let model = make_clip(16, 32);
        let embeds: Vec<Vec<f64>> = (0..4)
            .map(|i| {
                // Create orthogonal-ish unit vectors
                let mut v = vec![0.0_f64; 16];
                v[i % 16] = 1.0;
                v
            })
            .collect();
        let loss_perfect = model
            .contrastive_loss(&embeds, &embeds)
            .expect("should succeed");
        let loss_random = model
            .contrastive_loss(
                &(0..4)
                    .map(|i| l2_normalize(&rand_vec(16, i as u64)))
                    .collect::<Vec<_>>(),
                &(0..4)
                    .map(|i| l2_normalize(&rand_vec(16, i as u64 + 99)))
                    .collect::<Vec<_>>(),
            )
            .expect("should succeed");
        assert!(loss_perfect.total_loss <= loss_random.total_loss + 1e-6);
    }

    #[test]
    fn test_clip_zero_shot_classify_correct() {
        let model = make_clip(16, 32);
        // Use unit basis vectors as class embeddings; query = first class
        let class_embeds: Vec<Vec<f64>> = (0..4)
            .map(|i| {
                let mut v = vec![0.0_f64; 16];
                v[i] = 1.0;
                v
            })
            .collect();
        // Manually get image embedding of a zero-input image and find which class matches best
        // The point is: classify returns a valid index in [0, n_classes)
        let img = rand_vec(32, 77);
        let cls = model
            .zero_shot_classify(&img, &class_embeds)
            .expect("should succeed");
        assert!(cls < 4);
    }

    #[test]
    fn test_clip_retrieve_text_top_k() {
        let model = make_clip(16, 32);
        let text_embeds: Vec<Vec<f64>> = (0..10)
            .map(|i| l2_normalize(&rand_vec(16, i as u64)))
            .collect();
        let img = rand_vec(32, 42);
        let result = model
            .retrieve_text(&img, &text_embeds, 3)
            .expect("should succeed");
        assert_eq!(result.len(), 3);
        // All indices should be in range
        for &idx in &result {
            assert!(idx < 10);
        }
    }

    #[test]
    fn test_clip_retrieve_text_k_larger_than_n() {
        let model = make_clip(16, 32);
        let text_embeds: Vec<Vec<f64>> = (0..5)
            .map(|i| l2_normalize(&rand_vec(16, i as u64)))
            .collect();
        let img = rand_vec(32, 43);
        let result = model
            .retrieve_text(&img, &text_embeds, 10)
            .expect("should succeed");
        assert_eq!(result.len(), 5); // capped at available items
    }

    #[test]
    fn test_clip_retrieve_text_k_zero() {
        let model = make_clip(16, 32);
        let text_embeds: Vec<Vec<f64>> = (0..5)
            .map(|i| l2_normalize(&rand_vec(16, i as u64)))
            .collect();
        let img = rand_vec(32, 44);
        let result = model
            .retrieve_text(&img, &text_embeds, 0)
            .expect("should succeed");
        assert!(result.is_empty());
    }

    // ── 3. CrossModalAttention ────────────────────────────────────────────────

    fn make_cma(q_dim: usize, k_dim: usize, n_heads: usize) -> CrossModalAttention {
        let config = CrossModalAttentionConfig {
            query_dim: q_dim,
            key_dim: k_dim,
            n_heads,
            dropout: 0.0,
            seed: 0,
        };
        let mut rng = make_rng(0);
        CrossModalAttention::new(config, &mut rng)
    }

    #[test]
    fn test_cma_forward_output_shape() {
        let cma = make_cma(8, 8, 2);
        let query: Vec<Vec<f64>> = (0..3).map(|i| rand_vec(8, i as u64)).collect();
        let key: Vec<Vec<f64>> = (0..5).map(|i| rand_vec(8, i as u64 + 10)).collect();
        let out = cma.forward(&query, &key, &key).expect("should succeed");
        assert_eq!(out.len(), 3, "seq_q dimension mismatch");
        assert_eq!(out[0].len(), 8, "output dim mismatch");
    }

    #[test]
    fn test_cma_forward_different_key_query_dims() {
        let cma = make_cma(12, 8, 2);
        let query: Vec<Vec<f64>> = (0..2).map(|i| rand_vec(12, i as u64)).collect();
        let key: Vec<Vec<f64>> = (0..4).map(|i| rand_vec(8, i as u64 + 20)).collect();
        let out = cma.forward(&query, &key, &key).expect("should succeed");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 12);
    }

    #[test]
    fn test_cma_attention_weights_normalized() {
        let cma = make_cma(8, 8, 2);
        let query: Vec<Vec<f64>> = (0..3).map(|i| rand_vec(8, i as u64)).collect();
        let key: Vec<Vec<f64>> = (0..5).map(|i| rand_vec(8, i as u64 + 30)).collect();
        let weights = cma.attention_weights(&query, &key).expect("should succeed");
        assert_eq!(weights.len(), 3);
        for row in &weights {
            assert_eq!(row.len(), 5);
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "row sum = {}", sum);
        }
    }

    #[test]
    fn test_cma_attention_weights_non_negative() {
        let cma = make_cma(6, 6, 1);
        let query: Vec<Vec<f64>> = (0..2).map(|i| rand_vec(6, i as u64)).collect();
        let key: Vec<Vec<f64>> = (0..4).map(|i| rand_vec(6, i as u64 + 40)).collect();
        let weights = cma.attention_weights(&query, &key).expect("should succeed");
        for row in &weights {
            for &w in row {
                assert!(w >= 0.0 - 1e-12, "negative weight: {}", w);
            }
        }
    }

    #[test]
    fn test_cma_key_value_length_mismatch() {
        let cma = make_cma(8, 8, 2);
        let query: Vec<Vec<f64>> = vec![rand_vec(8, 0)];
        let key: Vec<Vec<f64>> = (0..3).map(|i| rand_vec(8, i as u64)).collect();
        let value: Vec<Vec<f64>> = (0..5).map(|i| rand_vec(8, i as u64 + 100)).collect();
        let result = cma.forward(&query, &key, &value);
        assert!(result.is_err());
    }

    // ── 4. MultimodalFusion ───────────────────────────────────────────────────

    fn make_fusion(
        strategy: FusionStrategy,
        input_dims: Vec<usize>,
        out_dim: usize,
    ) -> MultimodalFusion {
        let config = ModalFusionConfig {
            strategy,
            input_dims,
            output_dim: out_dim,
            seed: 42,
        };
        let mut rng = make_rng(42);
        MultimodalFusion::new(config, &mut rng)
    }

    #[test]
    fn test_early_fusion_output_shape() {
        let fusion = make_fusion(FusionStrategy::EarlyFusion, vec![8, 12], 16);
        let inputs = vec![rand_vec(8, 0), rand_vec(12, 1)];
        let result = fusion.fuse(&inputs).expect("should succeed");
        assert_eq!(result.fused_embedding.len(), 16);
    }

    #[test]
    fn test_late_fusion_output_shape() {
        let fusion = make_fusion(FusionStrategy::LateFusion, vec![8, 10], 12);
        let inputs = vec![rand_vec(8, 0), rand_vec(10, 1)];
        let result = fusion.fuse(&inputs).expect("should succeed");
        assert_eq!(result.fused_embedding.len(), 12);
    }

    #[test]
    fn test_gated_fusion_weights_sum_to_one() {
        let fusion = make_fusion(FusionStrategy::GatedFusion, vec![8, 8, 8], 16);
        let inputs = vec![rand_vec(8, 0), rand_vec(8, 1), rand_vec(8, 2)];
        let result = fusion.fuse(&inputs).expect("should succeed");
        let weight_sum: f64 = result.modality_weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-9, "sum={}", weight_sum);
    }

    #[test]
    fn test_gated_fusion_correct_n_weights() {
        let fusion = make_fusion(FusionStrategy::GatedFusion, vec![6, 6, 6], 12);
        let inputs = vec![rand_vec(6, 0), rand_vec(6, 1), rand_vec(6, 2)];
        let result = fusion.fuse(&inputs).expect("should succeed");
        assert_eq!(result.modality_weights.len(), 3);
    }

    #[test]
    fn test_bilinear_fusion_output_shape() {
        let fusion = make_fusion(FusionStrategy::BilinearFusion, vec![8, 8], 16);
        let inputs = vec![rand_vec(8, 0), rand_vec(8, 1)];
        let result = fusion.fuse(&inputs).expect("should succeed");
        assert_eq!(result.fused_embedding.len(), 16);
    }

    #[test]
    fn test_bilinear_fusion_requires_two_modalities() {
        let fusion = make_fusion(FusionStrategy::BilinearFusion, vec![8], 8);
        let inputs = vec![rand_vec(8, 0)];
        let result = fusion.fuse(&inputs);
        assert!(result.is_err());
    }

    #[test]
    fn test_fusion_wrong_modality_count() {
        let fusion = make_fusion(FusionStrategy::LateFusion, vec![8, 8], 16);
        let inputs = vec![rand_vec(8, 0)]; // only 1 of 2
        let result = fusion.fuse(&inputs);
        assert!(result.is_err());
    }

    #[test]
    fn test_hybrid_fusion_output_shape() {
        let fusion = make_fusion(FusionStrategy::HybridFusion, vec![10, 10], 8);
        let inputs = vec![rand_vec(10, 0), rand_vec(10, 1)];
        let result = fusion.fuse(&inputs).expect("should succeed");
        assert_eq!(result.fused_embedding.len(), 8);
    }

    // ── 5. MoeLayer ───────────────────────────────────────────────────────────

    fn make_moe(ne: usize, k: usize, id: usize, ed: usize, od: usize) -> MoeLayer {
        let config = MoeConfig {
            n_experts: ne,
            top_k: k,
            input_dim: id,
            expert_dim: ed,
            output_dim: od,
            load_balancing_coeff: 0.01,
            seed: 0,
        };
        let mut rng = make_rng(0);
        MoeLayer::new(config, &mut rng)
    }

    #[test]
    fn test_moe_forward_output_shape() {
        let moe = make_moe(4, 2, 8, 16, 6);
        let input = rand_vec(8, 0);
        let (out, _, _) = moe.forward(&input).expect("should succeed");
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn test_moe_top_k_routing_count() {
        let moe = make_moe(8, 3, 16, 32, 8);
        let input = rand_vec(16, 1);
        let (_, experts, weights) = moe.forward(&input).expect("should succeed");
        assert_eq!(experts.len(), 3, "should select exactly top-3 experts");
        assert_eq!(weights.len(), 3);
    }

    #[test]
    fn test_moe_gate_weights_sum_to_one() {
        let moe = make_moe(6, 2, 12, 24, 8);
        let input = rand_vec(12, 2);
        let (_, _, weights) = moe.forward(&input).expect("should succeed");
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "gate weight sum={}", sum);
    }

    #[test]
    fn test_moe_selected_experts_in_range() {
        let moe = make_moe(5, 2, 8, 16, 4);
        let input = rand_vec(8, 3);
        let (_, experts, _) = moe.forward(&input).expect("should succeed");
        for &e in &experts {
            assert!(e < 5, "expert index {} out of range", e);
        }
    }

    #[test]
    fn test_moe_load_balancing_loss_non_negative() {
        let moe = make_moe(4, 2, 8, 16, 4);
        let batch: Vec<Vec<f64>> = (0..8)
            .map(|i| {
                let v = rand_vec(4, i as u64);
                softmax(&v) // simulate gate probs
            })
            .collect();
        let loss = moe.load_balancing_loss(&batch);
        assert!(loss >= 0.0, "load balancing loss is negative: {}", loss);
    }

    #[test]
    fn test_moe_load_balancing_empty_batch() {
        let moe = make_moe(4, 2, 8, 16, 4);
        let loss = moe.load_balancing_loss(&[]);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_moe_expert_utilization_sums_to_one() {
        let moe = make_moe(4, 2, 8, 16, 4);
        let indices: Vec<Vec<usize>> = vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![0, 3]];
        let util = moe.expert_utilization(&indices);
        assert_eq!(util.len(), 4);
        let sum: f64 = util.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "utilization sum={}", sum);
    }

    #[test]
    fn test_moe_wrong_input_dim() {
        let moe = make_moe(4, 2, 8, 16, 4);
        let result = moe.forward(&rand_vec(5, 0));
        assert!(result.is_err());
    }

    // ── 6. compute_multimodal_metrics ─────────────────────────────────────────

    #[test]
    fn test_metrics_perfect_retrieval_r1() {
        // Identical image and text embeddings → perfect retrieval
        let n = 5;
        let embeds: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut v = vec![0.0_f64; n];
                v[i] = 1.0;
                v
            })
            .collect();
        let metrics = compute_multimodal_metrics(&embeds, &embeds).expect("should succeed");
        assert!((metrics.retrieval_recall_at_1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_metrics_perfect_mrr() {
        let n = 4;
        let embeds: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut v = vec![0.0_f64; n];
                v[i] = 1.0;
                v
            })
            .collect();
        let metrics = compute_multimodal_metrics(&embeds, &embeds).expect("should succeed");
        assert!((metrics.mean_reciprocal_rank - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_metrics_perfect_alignment_score() {
        let n = 3;
        let embeds: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut v = vec![0.0_f64; n];
                v[i] = 1.0;
                v
            })
            .collect();
        let metrics = compute_multimodal_metrics(&embeds, &embeds).expect("should succeed");
        assert!((metrics.modal_alignment_score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_metrics_empty_error() {
        let result = compute_multimodal_metrics(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_metrics_length_mismatch_error() {
        let img: Vec<Vec<f64>> = vec![vec![1.0, 0.0]];
        let txt: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = compute_multimodal_metrics(&img, &txt);
        assert!(result.is_err());
    }

    #[test]
    fn test_metrics_recall_at_5_and_10() {
        // With 20 items, perfect retrieval should give R@1=R@5=R@10=1.0
        let n = 20;
        let embeds: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut v = vec![0.0_f64; n];
                v[i] = 1.0;
                v
            })
            .collect();
        let metrics = compute_multimodal_metrics(&embeds, &embeds).expect("should succeed");
        assert!((metrics.retrieval_recall_at_5 - 1.0).abs() < 1e-9);
        assert!((metrics.retrieval_recall_at_10 - 1.0).abs() < 1e-9);
    }

    // ── 7. Utility Functions ──────────────────────────────────────────────────

    #[test]
    fn test_l2_normalize_unit_norm() {
        let v = vec![3.0_f64, 4.0];
        let n = l2_normalize(&v);
        let norm: f64 = n.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let v = vec![0.0_f64, 0.0, 0.0];
        let n = l2_normalize(&v);
        assert_eq!(n, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0_f64, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0_f64, 0.0];
        let b = vec![0.0_f64, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0_f64, 0.0];
        let b = vec![-1.0_f64, 0.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0_f64, 2.0, 3.0, 0.5];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "sum={}", sum);
    }

    #[test]
    fn test_softmax_all_same() {
        let logits = vec![2.0_f64; 5];
        let probs = softmax(&logits);
        for &p in &probs {
            assert!((p - 0.2).abs() < 1e-12);
        }
    }

    #[test]
    fn test_softmax_empty() {
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn test_cross_entropy_correct_class_lower_loss() {
        // Two-class logits: [10, 0] should give near-zero loss for class 0
        let logits = vec![vec![10.0_f64, 0.0]];
        let loss_correct = cross_entropy_loss(&logits, &[0]).expect("should succeed");
        let loss_wrong = cross_entropy_loss(&logits, &[1]).expect("should succeed");
        assert!(loss_correct < loss_wrong);
    }

    #[test]
    fn test_cross_entropy_length_mismatch() {
        let logits = vec![vec![1.0_f64, 2.0]];
        let result = cross_entropy_loss(&logits, &[0, 1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_entropy_target_out_of_range() {
        let logits = vec![vec![1.0_f64, 2.0]];
        let result = cross_entropy_loss(&logits, &[5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_entropy_empty() {
        let result = cross_entropy_loss(&[], &[]);
        assert!(result.is_err());
    }
}
