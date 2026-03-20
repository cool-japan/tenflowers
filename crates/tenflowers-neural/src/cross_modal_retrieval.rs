//! Cross-Modal Retrieval Systems
//!
//! Implements image-text, audio-visual, and general cross-modal retrieval.
//!
//! # Implemented Components
//!
//! | Component | Description |
//! |-----------|-------------|
//! | [`CmrModalityEncoder`] | Generic MLP encoder with Xavier init and L2-normalized output |
//! | [`DualEncoder`] | Two-tower CLIP-style retrieval model |
//! | [`FlatIndex`] | Brute-force embedding index with L2/Cosine/DotProduct metrics |
//! | [`CmrCrossModalAttention`] | Multi-head cross-modal attention |
//! | [`ZeroShotClassifier`] | CLIP-style zero-shot visual classification |
//! | [`MultiModalEmbedder`] | Joint embedding space for multiple modalities |
//! | [`RetrievalMetrics`] | Recall@k, MAP, NDCG, median rank, R-precision |

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers (f32)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the L2 norm of a slice.
fn l2_norm_f32(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// L2-normalise a vector in place.  Returns the zero vector unchanged.
fn l2_normalize_f32(v: &[f32]) -> Vec<f32> {
    let norm = l2_norm_f32(v);
    if norm < f32::EPSILON {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Dot product of two equal-length slices.
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Cosine similarity between two f32 vectors.
pub fn cmr_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let na = l2_norm_f32(a);
    let nb = l2_norm_f32(b);
    if na < f32::EPSILON || nb < f32::EPSILON {
        return 0.0;
    }
    dot_f32(a, b) / (na * nb)
}

/// Squared Euclidean distance between two f32 vectors.
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Numerically-stable softmax over a 1-D f32 slice.
fn softmax_f32(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum < f32::EPSILON {
        let n = exps.len();
        return vec![1.0_f32 / n as f32; n];
    }
    exps.iter().map(|e| e / sum).collect()
}

/// Xavier-uniform weight initialisation: uniform in [-limit, limit]
/// where limit = sqrt(6 / (fan_in + fan_out)).
fn xavier_uniform(fan_in: usize, fan_out: usize, rng: &mut impl Rng) -> f32 {
    let limit = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
    rng.random::<f32>() * 2.0 * limit - limit
}

/// Build a weight matrix [out_dim × in_dim] with Xavier-uniform init.
fn xavier_matrix(in_dim: usize, out_dim: usize, rng: &mut impl Rng) -> Vec<Vec<f32>> {
    (0..out_dim)
        .map(|_| {
            (0..in_dim)
                .map(|_| xavier_uniform(in_dim, out_dim, rng))
                .collect()
        })
        .collect()
}

/// Matrix-vector product: y = W x  (W is [out × in], x is [in]).
fn matvec(w: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
    w.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. CmrModalityEncoder
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`CmrModalityEncoder`].
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub n_layers: usize,
}

/// Generic MLP encoder for a single modality.
///
/// Architecture: `n_layers` hidden layers with ReLU, then a projection to
/// `output_dim` whose output is L2-normalised.
#[derive(Clone, Debug)]
pub struct CmrModalityEncoder {
    pub config: EncoderConfig,
    /// weights[i] is the weight matrix for layer i.
    weights: Vec<Vec<Vec<f32>>>,
    /// biases[i] is the bias vector for layer i.
    biases: Vec<Vec<f32>>,
}

impl CmrModalityEncoder {
    /// Create a new encoder with Xavier-uniform weight initialisation.
    pub fn new(config: EncoderConfig, rng: &mut impl Rng) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        // Hidden layers
        let n_hidden = config.n_layers.max(1);
        for i in 0..n_hidden {
            let in_dim = if i == 0 {
                config.input_dim
            } else {
                config.hidden_dim
            };
            let out_dim = config.hidden_dim;
            weights.push(xavier_matrix(in_dim, out_dim, rng));
            biases.push(vec![0.0_f32; out_dim]);
        }
        // Output projection
        weights.push(xavier_matrix(config.hidden_dim, config.output_dim, rng));
        biases.push(vec![0.0_f32; config.output_dim]);

        Self {
            config,
            weights,
            biases,
        }
    }

    /// Encode a single input vector.  Output is L2-normalised.
    pub fn encode(&self, input: &[f32]) -> Vec<f32> {
        let n_hidden = self.config.n_layers.max(1);
        let mut h: Vec<f32> = input.to_vec();

        for i in 0..n_hidden {
            let z: Vec<f32> = matvec(&self.weights[i], &h)
                .iter()
                .zip(self.biases[i].iter())
                .map(|(z, b)| z + b)
                .collect();
            // ReLU
            h = z.iter().map(|x| x.max(0.0)).collect();
        }

        // Output projection (no activation, then L2 normalise)
        let out_idx = n_hidden;
        let z_out: Vec<f32> = matvec(&self.weights[out_idx], &h)
            .iter()
            .zip(self.biases[out_idx].iter())
            .map(|(z, b)| z + b)
            .collect();

        l2_normalize_f32(&z_out)
    }

    /// Encode a batch of input vectors.
    pub fn encode_batch(&self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        inputs.iter().map(|x| self.encode(x)).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. DualEncoder
// ─────────────────────────────────────────────────────────────────────────────

/// Two-tower retrieval model (CLIP-style).
///
/// Maintains separate encoders for vision and text modalities and computes
/// symmetric InfoNCE / NT-Xent contrastive loss.
#[derive(Clone, Debug)]
pub struct DualEncoder {
    pub vision_encoder: CmrModalityEncoder,
    pub text_encoder: CmrModalityEncoder,
    pub temperature: f32,
}

impl DualEncoder {
    /// Create a new dual encoder.
    pub fn new(
        vision_config: EncoderConfig,
        text_config: EncoderConfig,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            vision_encoder: CmrModalityEncoder::new(vision_config, rng),
            text_encoder: CmrModalityEncoder::new(text_config, rng),
            temperature,
        }
    }

    /// Encode a vision input.
    pub fn encode_vision(&self, x: &[f32]) -> Vec<f32> {
        self.vision_encoder.encode(x)
    }

    /// Encode a text input.
    pub fn encode_text(&self, x: &[f32]) -> Vec<f32> {
        self.text_encoder.encode(x)
    }

    /// Compute pairwise cosine similarity matrix scaled by temperature.
    ///
    /// Returns an `N × M` matrix where `S[i][j] = cos(v_i, t_j) / temperature`.
    pub fn similarity_matrix(
        &self,
        vision_embeds: &[Vec<f32>],
        text_embeds: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        vision_embeds
            .iter()
            .map(|v| {
                text_embeds
                    .iter()
                    .map(|t| cmr_cosine_similarity(v, t) / self.temperature)
                    .collect()
            })
            .collect()
    }

    /// Symmetric InfoNCE / NT-Xent contrastive loss.
    ///
    /// Both vision-to-text and text-to-vision cross-entropy are averaged.
    /// Diagonal entries are positive pairs; off-diagonals are negatives.
    pub fn contrastive_loss(&self, vision_embeds: &[Vec<f32>], text_embeds: &[Vec<f32>]) -> f32 {
        let n = vision_embeds.len().min(text_embeds.len());
        if n == 0 {
            return 0.0;
        }

        let sim = self.similarity_matrix(vision_embeds, text_embeds);

        // Vision-to-text loss
        let mut v2t_loss = 0.0_f32;
        for i in 0..n {
            let row_logits: Vec<f32> = sim[i][..n].to_vec();
            let probs = softmax_f32(&row_logits);
            let p_pos = probs[i].max(f32::EPSILON);
            v2t_loss -= p_pos.ln();
        }
        v2t_loss /= n as f32;

        // Text-to-vision loss (transpose)
        let mut t2v_loss = 0.0_f32;
        for j in 0..n {
            let col_logits: Vec<f32> = (0..n).map(|i| sim[i][j]).collect();
            let probs = softmax_f32(&col_logits);
            let p_pos = probs[j].max(f32::EPSILON);
            t2v_loss -= p_pos.ln();
        }
        t2v_loss /= n as f32;

        (v2t_loss + t2v_loss) / 2.0
    }

    /// Retrieve the top-k text indices most similar to the query vision embedding.
    pub fn retrieve_text(
        &self,
        query_vision: &[f32],
        text_embeddings: &[Vec<f32>],
        top_k: usize,
    ) -> Vec<usize> {
        let mut scores: Vec<(usize, f32)> = text_embeddings
            .iter()
            .enumerate()
            .map(|(i, t)| (i, cmr_cosine_similarity(query_vision, t)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
            .iter()
            .take(top_k.min(scores.len()))
            .map(|(i, _)| *i)
            .collect()
    }

    /// Retrieve the top-k vision indices most similar to the query text embedding.
    pub fn retrieve_vision(
        &self,
        query_text: &[f32],
        vision_embeddings: &[Vec<f32>],
        top_k: usize,
    ) -> Vec<usize> {
        let mut scores: Vec<(usize, f32)> = vision_embeddings
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cmr_cosine_similarity(query_text, v)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
            .iter()
            .take(top_k.min(scores.len()))
            .map(|(i, _)| *i)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. FlatIndex
// ─────────────────────────────────────────────────────────────────────────────

/// Distance / similarity metric for [`FlatIndex`].
#[derive(Clone, Debug, PartialEq)]
pub enum DistanceMetric {
    L2,
    Cosine,
    DotProduct,
}

/// A single retrieval result.
#[derive(Clone, Debug)]
pub struct RetrievalResult {
    pub index: usize,
    pub score: f32,
    pub metadata: String,
}

/// Brute-force flat embedding index.
pub struct FlatIndex {
    embeddings: Vec<Vec<f32>>,
    metadata: Vec<String>,
    metric: DistanceMetric,
}

impl FlatIndex {
    /// Create an empty index with the given distance metric.
    pub fn new(metric: DistanceMetric) -> Self {
        Self {
            embeddings: Vec::new(),
            metadata: Vec::new(),
            metric,
        }
    }

    /// Add a single embedding with associated metadata.
    pub fn add(&mut self, embedding: Vec<f32>, meta: impl Into<String>) {
        self.embeddings.push(embedding);
        self.metadata.push(meta.into());
    }

    /// Add a batch of embeddings with associated metadata.
    pub fn add_batch(&mut self, embeddings: Vec<Vec<f32>>, metas: Vec<String>) {
        for (emb, meta) in embeddings.into_iter().zip(metas) {
            self.embeddings.push(emb);
            self.metadata.push(meta);
        }
    }

    /// Search for the k nearest neighbours to `query`.
    ///
    /// For `L2`, lower scores are better (distance).
    /// For `Cosine` / `DotProduct`, higher scores are better.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<RetrievalResult> {
        if self.embeddings.is_empty() {
            return Vec::new();
        }

        let mut scores: Vec<(usize, f32)> = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| {
                let s = match self.metric {
                    DistanceMetric::L2 => -l2_distance(query, emb), // negate so sort gives nearest first
                    DistanceMetric::Cosine => cmr_cosine_similarity(query, emb),
                    DistanceMetric::DotProduct => dot_f32(query, emb),
                };
                (i, s)
            })
            .collect();

        // Sort descending: highest score first (for L2 we negated, so smallest dist = highest score)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores
            .iter()
            .take(k.min(scores.len()))
            .map(|(i, s)| {
                let actual_score = match self.metric {
                    DistanceMetric::L2 => -s, // restore positive distance
                    _ => *s,
                };
                RetrievalResult {
                    index: *i,
                    score: actual_score,
                    metadata: self.metadata[*i].clone(),
                }
            })
            .collect()
    }

    /// Number of entries in the index.
    pub fn size(&self) -> usize {
        self.embeddings.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. CmrCrossModalAttention
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-head cross-attention between two modalities.
///
/// Named `CmrCrossModalAttention` to avoid collision with
/// `multimodal::CrossModalAttention`.
pub struct CmrCrossModalAttention {
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    /// Query projection: [d_model × d_model]
    wq: Vec<Vec<f32>>,
    /// Key projection: [d_model × d_model]
    wk: Vec<Vec<f32>>,
    /// Value projection: [d_model × d_model]
    wv: Vec<Vec<f32>>,
    /// Output projection: [d_model × d_model]
    wo: Vec<Vec<f32>>,
}

impl CmrCrossModalAttention {
    /// Create a new cross-modal attention layer with Xavier-uniform init.
    pub fn new(d_model: usize, n_heads: usize, rng: &mut impl Rng) -> Self {
        let head_dim = (d_model / n_heads).max(1);
        Self {
            d_model,
            n_heads,
            head_dim,
            wq: xavier_matrix(d_model, d_model, rng),
            wk: xavier_matrix(d_model, d_model, rng),
            wv: xavier_matrix(d_model, d_model, rng),
            wo: xavier_matrix(d_model, d_model, rng),
        }
    }

    /// Scaled dot-product attention for a single head.
    ///
    /// `q_mat`: [q_len × head_dim], `k_mat`: [kv_len × head_dim],
    /// `v_mat`: [kv_len × head_dim].  Returns [q_len × head_dim].
    fn scaled_attention(
        q_mat: &[Vec<f32>],
        k_mat: &[Vec<f32>],
        v_mat: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let scale = (q_mat.first().map(|r| r.len()).unwrap_or(1) as f32).sqrt();
        let q_len = q_mat.len();
        let kv_len = k_mat.len();

        // Attention scores: [q_len × kv_len]
        let attn_scores: Vec<Vec<f32>> = (0..q_len)
            .map(|qi| {
                (0..kv_len)
                    .map(|ki| dot_f32(&q_mat[qi], &k_mat[ki]) / scale)
                    .collect()
            })
            .collect();

        // Softmax over key dimension
        let attn_weights: Vec<Vec<f32>> = attn_scores.iter().map(|row| softmax_f32(row)).collect();

        // Weighted sum of values: [q_len × head_dim]
        (0..q_len)
            .map(|qi| {
                let hd = v_mat.first().map(|r| r.len()).unwrap_or(1);
                let mut out = vec![0.0_f32; hd];
                for ki in 0..kv_len {
                    let w = attn_weights[qi][ki];
                    for d in 0..hd {
                        out[d] += w * v_mat[ki][d];
                    }
                }
                out
            })
            .collect()
    }

    /// Project a sequence through a weight matrix.
    ///
    /// `seq`: [seq_len × in_dim], `w`: [out_dim × in_dim].
    /// Returns [seq_len × out_dim].
    fn project_seq(seq: &[Vec<f32>], w: &[Vec<f32>]) -> Vec<Vec<f32>> {
        seq.iter().map(|x| matvec(w, x)).collect()
    }

    /// Multi-head cross-attention.
    ///
    /// `query_seq`: tokens from modality 1 ([q_len × d_model]).
    /// `kv_seq`:    tokens from modality 2 ([kv_len × d_model]).
    /// Returns attended output [q_len × d_model].
    pub fn attend(&self, query_seq: &[Vec<f32>], kv_seq: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let q_len = query_seq.len();
        let d = self.d_model;
        let h = self.n_heads;
        let hd = self.head_dim;

        if q_len == 0 || kv_seq.is_empty() {
            return query_seq.to_vec();
        }

        // Project full sequences
        let q_proj = Self::project_seq(query_seq, &self.wq); // [q_len × d]
        let k_proj = Self::project_seq(kv_seq, &self.wk); // [kv_len × d]
        let v_proj = Self::project_seq(kv_seq, &self.wv); // [kv_len × d]

        // Concatenate head outputs
        let mut concat_out = vec![vec![0.0_f32; d]; q_len];

        for head in 0..h {
            let start = head * hd;
            let end = (start + hd).min(d);
            if start >= d {
                break;
            }

            // Slice head dimension
            let q_head: Vec<Vec<f32>> = q_proj.iter().map(|row| row[start..end].to_vec()).collect();
            let k_head: Vec<Vec<f32>> = k_proj.iter().map(|row| row[start..end].to_vec()).collect();
            let v_head: Vec<Vec<f32>> = v_proj.iter().map(|row| row[start..end].to_vec()).collect();

            let head_out = Self::scaled_attention(&q_head, &k_head, &v_head);

            // Place into output at head slice position
            for qi in 0..q_len {
                let actual_hd = end - start;
                for di in 0..actual_hd {
                    concat_out[qi][start + di] = head_out[qi][di];
                }
            }
        }

        // Output projection
        Self::project_seq(&concat_out, &self.wo)
    }

    /// Fuse two modal sequences.
    ///
    /// Attends modal1 → modal2, mean-pools the attended output,
    /// concatenates with modal1 mean, then returns a vector of length `2 * d_model`.
    pub fn fuse(&self, modal1: &[Vec<f32>], modal2: &[Vec<f32>]) -> Vec<f32> {
        let d = self.d_model;

        // Mean of modal1
        let modal1_mean = if modal1.is_empty() {
            vec![0.0_f32; d]
        } else {
            let mut m = vec![0.0_f32; d];
            for row in modal1.iter() {
                for (di, &v) in row.iter().enumerate().take(d) {
                    m[di] += v;
                }
            }
            let n = modal1.len() as f32;
            m.iter().map(|x| x / n).collect()
        };

        let attended = self.attend(modal1, modal2);

        // Mean-pool attended output
        let attended_mean = if attended.is_empty() {
            vec![0.0_f32; d]
        } else {
            let mut m = vec![0.0_f32; d];
            for row in attended.iter() {
                for (di, &v) in row.iter().enumerate().take(d) {
                    m[di] += v;
                }
            }
            let n = attended.len() as f32;
            m.iter().map(|x| x / n).collect()
        };

        // Concatenate modal1_mean and attended_mean
        let mut out = modal1_mean;
        out.extend_from_slice(&attended_mean);
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. ZeroShotClassifier
// ─────────────────────────────────────────────────────────────────────────────

/// CLIP-style zero-shot visual classifier.
///
/// Text embeddings for each class are pre-computed and stored.  At inference,
/// a vision embedding is scored against all class embeddings using cosine
/// similarity.
pub struct ZeroShotClassifier {
    pub encoder: DualEncoder,
    pub class_embeddings: Vec<Vec<f32>>,
    pub class_names: Vec<String>,
}

impl ZeroShotClassifier {
    /// Create an empty classifier wrapping the given dual encoder.
    pub fn new(encoder: DualEncoder) -> Self {
        Self {
            encoder,
            class_embeddings: Vec::new(),
            class_names: Vec::new(),
        }
    }

    /// Register a class with its pre-computed text embedding.
    pub fn add_class(&mut self, name: impl Into<String>, text_embedding: Vec<f32>) {
        self.class_names.push(name.into());
        self.class_embeddings.push(text_embedding);
    }

    /// Score all classes for a given vision embedding.
    ///
    /// Returns `(class_name, score)` pairs sorted by score descending.
    pub fn classify(&self, vision_embedding: &[f32]) -> Vec<(String, f32)> {
        let mut scores: Vec<(String, f32)> = self
            .class_names
            .iter()
            .zip(self.class_embeddings.iter())
            .map(|(name, emb)| {
                let s = cmr_cosine_similarity(vision_embedding, emb);
                (name.clone(), s)
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// Return the top-k classes for a given vision embedding.
    pub fn top_k_classes(&self, vision_embedding: &[f32], k: usize) -> Vec<(String, f32)> {
        self.classify(vision_embedding)
            .into_iter()
            .take(k)
            .collect()
    }

    /// Classify a batch of vision embeddings.
    pub fn batch_classify(&self, vision_embeddings: &[Vec<f32>]) -> Vec<Vec<(String, f32)>> {
        vision_embeddings
            .iter()
            .map(|emb| self.classify(emb))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. MultiModalEmbedder
// ─────────────────────────────────────────────────────────────────────────────

/// Modality type identifier.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ModalityType {
    Vision,
    Text,
    Audio,
    Video,
    Tabular,
}

/// Configuration for [`MultiModalEmbedder`].
#[derive(Clone, Debug)]
pub struct EmbedderConfig {
    /// Pairs of (modality, input_dim) for each modality encoder.
    pub modality_dims: Vec<(ModalityType, usize)>,
    /// Target dimension of the joint embedding space.
    pub joint_dim: usize,
}

/// Joint embedding space for multiple modalities.
pub struct MultiModalEmbedder {
    /// One `CmrModalityEncoder` per modality in registration order.
    pub encoders: Vec<(ModalityType, CmrModalityEncoder)>,
    pub joint_dim: usize,
    /// Optional projection from concatenated embeddings to `joint_dim`.
    fusion_proj: Option<Vec<Vec<f32>>>,
    /// Number of input features to the fusion projection.
    fused_input_dim: usize,
}

impl MultiModalEmbedder {
    /// Build a new embedder from config, initialising all encoders with Xavier weights.
    pub fn new(config: EmbedderConfig, rng: &mut impl Rng) -> Self {
        let encoders: Vec<(ModalityType, CmrModalityEncoder)> = config
            .modality_dims
            .iter()
            .map(|(mod_type, in_dim)| {
                let enc_cfg = EncoderConfig {
                    input_dim: *in_dim,
                    hidden_dim: config.joint_dim,
                    output_dim: config.joint_dim,
                    n_layers: 2,
                };
                (mod_type.clone(), CmrModalityEncoder::new(enc_cfg, rng))
            })
            .collect();

        let n_mod = encoders.len().max(1);
        let fused_input_dim = config.joint_dim * n_mod;

        // Projection to joint_dim from concatenated encodings
        let fusion_proj = xavier_matrix(fused_input_dim, config.joint_dim, rng);

        Self {
            encoders,
            joint_dim: config.joint_dim,
            fusion_proj: Some(fusion_proj),
            fused_input_dim,
        }
    }

    /// Encode a single input for the specified modality.
    pub fn embed(&self, modality: ModalityType, input: &[f32]) -> Vec<f32> {
        for (mod_type, enc) in &self.encoders {
            if *mod_type == modality {
                return enc.encode(input);
            }
        }
        // Fallback: return zero vector of joint_dim
        vec![0.0_f32; self.joint_dim]
    }

    /// Fuse a list of (modality, embedding) pairs into a single joint vector.
    ///
    /// If all embeddings have the same length as `joint_dim`, returns their mean.
    /// Otherwise, concatenates and projects to `joint_dim`.
    pub fn fuse_embeddings(&self, embeddings: &[(ModalityType, Vec<f32>)]) -> Vec<f32> {
        if embeddings.is_empty() {
            return vec![0.0_f32; self.joint_dim];
        }

        let all_same_dim = embeddings.iter().all(|(_, e)| e.len() == self.joint_dim);

        if all_same_dim {
            // Mean pooling
            let mut mean = vec![0.0_f32; self.joint_dim];
            for (_, emb) in embeddings.iter() {
                for (d, &v) in emb.iter().enumerate() {
                    mean[d] += v;
                }
            }
            let n = embeddings.len() as f32;
            return mean.iter().map(|x| x / n).collect();
        }

        // Concatenate and project
        let mut concat: Vec<f32> = embeddings.iter().flat_map(|(_, e)| e.clone()).collect();
        // Pad or truncate to fused_input_dim
        concat.resize(self.fused_input_dim, 0.0);

        match &self.fusion_proj {
            Some(proj) => l2_normalize_f32(&matvec(proj, &concat)),
            None => concat[..self.joint_dim.min(concat.len())].to_vec(),
        }
    }

    /// Cosine similarity between two joint embeddings.
    pub fn cross_modal_score(embed1: &[f32], embed2: &[f32]) -> f32 {
        cmr_cosine_similarity(embed1, embed2)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. RetrievalMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Comprehensive evaluation report for a retrieval system.
#[derive(Clone, Debug)]
pub struct RetrievalEvalReport {
    /// Recall at rank 1.
    pub r1: f32,
    /// Recall at rank 5.
    pub r5: f32,
    /// Recall at rank 10.
    pub r10: f32,
    /// Mean average precision.
    pub map: f32,
    /// NDCG at rank 10.
    pub ndcg: f32,
    /// Median rank of the first relevant item.
    pub med_rank: f32,
}

/// Collection of retrieval evaluation metrics.
pub struct RetrievalMetrics;

impl RetrievalMetrics {
    /// Recall@k: fraction of queries for which at least one relevant item
    /// appears in the top-k retrieved results.
    pub fn recall_at_k(retrieved: &[Vec<usize>], relevant: &[Vec<usize>], k: usize) -> f32 {
        if retrieved.is_empty() {
            return 0.0;
        }
        let hits: usize = retrieved
            .iter()
            .zip(relevant.iter())
            .filter(|(ret, rel)| {
                let top_k: std::collections::HashSet<usize> = ret.iter().take(k).copied().collect();
                rel.iter().any(|r| top_k.contains(r))
            })
            .count();
        hits as f32 / retrieved.len() as f32
    }

    /// Mean Average Precision (MAP) over all queries.
    pub fn mean_average_precision(retrieved: &[Vec<usize>], relevant: &[Vec<usize>]) -> f32 {
        if retrieved.is_empty() {
            return 0.0;
        }
        let total: f32 = retrieved
            .iter()
            .zip(relevant.iter())
            .map(|(ret, rel)| {
                let rel_set: std::collections::HashSet<usize> = rel.iter().copied().collect();
                if rel_set.is_empty() {
                    return 0.0;
                }
                let mut num_hits = 0_usize;
                let mut sum_prec = 0.0_f32;
                for (rank, &item) in ret.iter().enumerate() {
                    if rel_set.contains(&item) {
                        num_hits += 1;
                        sum_prec += num_hits as f32 / (rank + 1) as f32;
                    }
                }
                sum_prec / rel_set.len() as f32
            })
            .sum();
        total / retrieved.len() as f32
    }

    /// NDCG@k: normalised discounted cumulative gain at rank k.
    pub fn ndcg_at_k(retrieved: &[Vec<usize>], relevant: &[Vec<usize>], k: usize) -> f32 {
        if retrieved.is_empty() {
            return 0.0;
        }
        let total: f32 = retrieved
            .iter()
            .zip(relevant.iter())
            .map(|(ret, rel)| {
                let rel_set: std::collections::HashSet<usize> = rel.iter().copied().collect();

                // DCG
                let dcg: f32 = ret
                    .iter()
                    .take(k)
                    .enumerate()
                    .map(|(rank, &item)| {
                        let gain = if rel_set.contains(&item) {
                            1.0_f32
                        } else {
                            0.0
                        };
                        gain / (rank as f32 + 2.0).log2()
                    })
                    .sum();

                // IDCG
                let n_rel = rel_set.len().min(k);
                let idcg: f32 = (0..n_rel)
                    .map(|rank| 1.0_f32 / (rank as f32 + 2.0).log2())
                    .sum();

                if idcg < f32::EPSILON {
                    0.0
                } else {
                    dcg / idcg
                }
            })
            .sum();
        total / retrieved.len() as f32
    }

    /// Median rank of the first relevant item across all queries.
    pub fn median_rank(retrieved: &[Vec<usize>], relevant: &[Vec<usize>]) -> f32 {
        if retrieved.is_empty() {
            return 0.0;
        }
        let mut ranks: Vec<usize> = retrieved
            .iter()
            .zip(relevant.iter())
            .map(|(ret, rel)| {
                let rel_set: std::collections::HashSet<usize> = rel.iter().copied().collect();
                ret.iter()
                    .position(|item| rel_set.contains(item))
                    .map(|pos| pos + 1) // 1-indexed
                    .unwrap_or(retrieved[0].len() + 1)
            })
            .collect();
        ranks.sort_unstable();
        let n = ranks.len();
        if n % 2 == 1 {
            ranks[n / 2] as f32
        } else {
            (ranks[n / 2 - 1] + ranks[n / 2]) as f32 / 2.0
        }
    }

    /// R-Precision: precision at rank R where R = |relevant items|.
    pub fn r_precision(retrieved: &[Vec<usize>], relevant: &[Vec<usize>]) -> f32 {
        if retrieved.is_empty() {
            return 0.0;
        }
        let total: f32 = retrieved
            .iter()
            .zip(relevant.iter())
            .map(|(ret, rel)| {
                let r = rel.len();
                if r == 0 {
                    return 0.0;
                }
                let rel_set: std::collections::HashSet<usize> = rel.iter().copied().collect();
                let hits = ret
                    .iter()
                    .take(r)
                    .filter(|item| rel_set.contains(item))
                    .count();
                hits as f32 / r as f32
            })
            .sum();
        total / retrieved.len() as f32
    }

    /// Compute a full `RetrievalEvalReport` for a set of queries.
    pub fn evaluate(retrieved: &[Vec<usize>], relevant: &[Vec<usize>]) -> RetrievalEvalReport {
        RetrievalEvalReport {
            r1: Self::recall_at_k(retrieved, relevant, 1),
            r5: Self::recall_at_k(retrieved, relevant, 5),
            r10: Self::recall_at_k(retrieved, relevant, 10),
            map: Self::mean_average_precision(retrieved, relevant),
            ndcg: Self::ndcg_at_k(retrieved, relevant, 10),
            med_rank: Self::median_rank(retrieved, relevant),
        }
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

    fn enc_config(input_dim: usize, output_dim: usize) -> EncoderConfig {
        EncoderConfig {
            input_dim,
            hidden_dim: 32,
            output_dim,
            n_layers: 2,
        }
    }

    fn random_vec(dim: usize, rng: &mut StdRng) -> Vec<f32> {
        (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect()
    }

    // ── 1. CmrModalityEncoder ────────────────────────────────────────────────

    #[test]
    fn test_modality_encoder_forward_shape() {
        let mut rng = make_rng(1);
        let enc = CmrModalityEncoder::new(enc_config(64, 16), &mut rng);
        let input = random_vec(64, &mut rng);
        let out = enc.encode(&input);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_modality_encoder_l2_normalized() {
        let mut rng = make_rng(2);
        let enc = CmrModalityEncoder::new(enc_config(32, 16), &mut rng);
        let input = random_vec(32, &mut rng);
        let out = enc.encode(&input);
        let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm = {norm}");
    }

    #[test]
    fn test_modality_encoder_batch_shape() {
        let mut rng = make_rng(3);
        let enc = CmrModalityEncoder::new(enc_config(16, 8), &mut rng);
        let inputs: Vec<Vec<f32>> = (0..5).map(|_| random_vec(16, &mut rng)).collect();
        let outs = enc.encode_batch(&inputs);
        assert_eq!(outs.len(), 5);
        for out in &outs {
            assert_eq!(out.len(), 8);
        }
    }

    // ── 2. DualEncoder ───────────────────────────────────────────────────────

    fn make_dual_encoder(rng: &mut StdRng) -> DualEncoder {
        DualEncoder::new(enc_config(64, 16), enc_config(48, 16), 0.07, rng)
    }

    #[test]
    fn test_dual_encoder_vision_embed_shape() {
        let mut rng = make_rng(10);
        let de = make_dual_encoder(&mut rng);
        let x = random_vec(64, &mut rng);
        assert_eq!(de.encode_vision(&x).len(), 16);
    }

    #[test]
    fn test_dual_encoder_text_embed_shape() {
        let mut rng = make_rng(11);
        let de = make_dual_encoder(&mut rng);
        let x = random_vec(48, &mut rng);
        assert_eq!(de.encode_text(&x).len(), 16);
    }

    #[test]
    fn test_dual_encoder_similarity_matrix_shape() {
        let mut rng = make_rng(12);
        let de = make_dual_encoder(&mut rng);
        let v_embs: Vec<Vec<f32>> = (0..4)
            .map(|_| l2_normalize_f32(&random_vec(16, &mut rng)))
            .collect();
        let t_embs: Vec<Vec<f32>> = (0..4)
            .map(|_| l2_normalize_f32(&random_vec(16, &mut rng)))
            .collect();
        let sim = de.similarity_matrix(&v_embs, &t_embs);
        assert_eq!(sim.len(), 4);
        assert_eq!(sim[0].len(), 4);
    }

    #[test]
    fn test_dual_encoder_similarity_matrix_range() {
        let mut rng = make_rng(13);
        let de = DualEncoder::new(enc_config(16, 8), enc_config(16, 8), 1.0, &mut rng);
        let v_embs: Vec<Vec<f32>> = (0..3)
            .map(|_| l2_normalize_f32(&random_vec(8, &mut rng)))
            .collect();
        let t_embs: Vec<Vec<f32>> = (0..3)
            .map(|_| l2_normalize_f32(&random_vec(8, &mut rng)))
            .collect();
        let sim = de.similarity_matrix(&v_embs, &t_embs);
        for row in &sim {
            for &s in row {
                assert!((-1.01..=1.01).contains(&s), "out of range: {s}");
            }
        }
    }

    #[test]
    fn test_dual_encoder_contrastive_loss_positive() {
        let mut rng = make_rng(14);
        let de = make_dual_encoder(&mut rng);
        let v_embs: Vec<Vec<f32>> = (0..4)
            .map(|_| l2_normalize_f32(&random_vec(16, &mut rng)))
            .collect();
        let t_embs: Vec<Vec<f32>> = (0..4)
            .map(|_| l2_normalize_f32(&random_vec(16, &mut rng)))
            .collect();
        let loss = de.contrastive_loss(&v_embs, &t_embs);
        assert!(loss >= 0.0, "loss should be non-negative, got {loss}");
    }

    #[test]
    fn test_dual_encoder_retrieve_text_top_k_length() {
        let mut rng = make_rng(15);
        let de = make_dual_encoder(&mut rng);
        let query = l2_normalize_f32(&random_vec(16, &mut rng));
        let texts: Vec<Vec<f32>> = (0..10)
            .map(|_| l2_normalize_f32(&random_vec(16, &mut rng)))
            .collect();
        let result = de.retrieve_text(&query, &texts, 3);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_dual_encoder_retrieve_vision_top_k_length() {
        let mut rng = make_rng(16);
        let de = make_dual_encoder(&mut rng);
        let query = l2_normalize_f32(&random_vec(16, &mut rng));
        let visions: Vec<Vec<f32>> = (0..10)
            .map(|_| l2_normalize_f32(&random_vec(16, &mut rng)))
            .collect();
        let result = de.retrieve_vision(&query, &visions, 5);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_dual_encoder_perfect_retrieval() {
        let mut rng = make_rng(17);
        let de = DualEncoder::new(enc_config(8, 4), enc_config(8, 4), 1.0, &mut rng);
        // Build a database of 5 unique embeddings and put embedding[2] as query
        let db: Vec<Vec<f32>> = (0..5)
            .map(|i| {
                let mut v = vec![0.0_f32; 4];
                v[i % 4] = 1.0;
                v
            })
            .collect();
        let query = db[2].clone();
        let result = de.retrieve_text(&query, &db, 1);
        assert_eq!(result[0], 2, "Should retrieve index 2");
    }

    // ── 3. FlatIndex ─────────────────────────────────────────────────────────

    #[test]
    fn test_flat_index_add_and_search_l2() {
        let mut idx = FlatIndex::new(DistanceMetric::L2);
        idx.add(vec![1.0, 0.0, 0.0], "a");
        idx.add(vec![0.0, 1.0, 0.0], "b");
        idx.add(vec![0.0, 0.0, 1.0], "c");
        let results = idx.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].metadata, "a");
    }

    #[test]
    fn test_flat_index_add_and_search_cosine() {
        let mut idx = FlatIndex::new(DistanceMetric::Cosine);
        idx.add(vec![1.0, 0.0], "right");
        idx.add(vec![0.0, 1.0], "up");
        let results = idx.search(&[1.0, 0.0], 1);
        assert_eq!(results[0].metadata, "right");
    }

    #[test]
    fn test_flat_index_search_top_k_count() {
        let mut idx = FlatIndex::new(DistanceMetric::Cosine);
        for i in 0..10 {
            idx.add(vec![i as f32, 0.0, 0.0], format!("item_{i}"));
        }
        let results = idx.search(&[1.0, 0.0, 0.0], 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_flat_index_search_top1_is_nearest() {
        let mut idx = FlatIndex::new(DistanceMetric::L2);
        idx.add(vec![10.0, 0.0], "far");
        idx.add(vec![1.0, 0.0], "near");
        idx.add(vec![5.0, 0.0], "mid");
        let results = idx.search(&[1.1, 0.0], 1);
        assert_eq!(results[0].metadata, "near");
    }

    #[test]
    fn test_flat_index_size() {
        let mut idx = FlatIndex::new(DistanceMetric::Cosine);
        assert_eq!(idx.size(), 0);
        idx.add(vec![1.0], "a");
        idx.add(vec![2.0], "b");
        assert_eq!(idx.size(), 2);
    }

    #[test]
    fn test_flat_index_add_batch() {
        let mut idx = FlatIndex::new(DistanceMetric::Cosine);
        let embs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let metas = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        idx.add_batch(embs, metas);
        assert_eq!(idx.size(), 3);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let s = cmr_cosine_similarity(&v, &v);
        assert!(
            (s - 1.0).abs() < 1e-5,
            "identical vectors should give 1.0, got {s}"
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let s = cmr_cosine_similarity(&a, &b);
        assert!(
            s.abs() < 1e-5,
            "orthogonal vectors should give 0.0, got {s}"
        );
    }

    #[test]
    fn test_l2_distance_zero() {
        let v = vec![3.0_f32, 4.0, 5.0];
        let d = l2_distance(&v, &v);
        assert!(d < 1e-5, "l2_distance of same vector should be 0, got {d}");
    }

    #[test]
    fn test_flat_index_dot_product_metric() {
        let mut idx = FlatIndex::new(DistanceMetric::DotProduct);
        idx.add(vec![10.0, 0.0], "big");
        idx.add(vec![1.0, 0.0], "small");
        let results = idx.search(&[1.0, 0.0], 1);
        assert_eq!(results[0].metadata, "big");
    }

    // ── 4. CmrCrossModalAttention ────────────────────────────────────────────

    #[test]
    fn test_cross_modal_attention_shape() {
        let mut rng = make_rng(30);
        let cma = CmrCrossModalAttention::new(16, 4, &mut rng);
        let modal1: Vec<Vec<f32>> = (0..3).map(|_| random_vec(16, &mut rng)).collect();
        let modal2: Vec<Vec<f32>> = (0..5).map(|_| random_vec(16, &mut rng)).collect();
        let out = cma.attend(&modal1, &modal2);
        assert_eq!(out.len(), 3, "query len preserved");
        assert_eq!(out[0].len(), 16, "d_model preserved");
    }

    #[test]
    fn test_cross_modal_attention_fuse_shape() {
        let mut rng = make_rng(31);
        let cma = CmrCrossModalAttention::new(8, 2, &mut rng);
        let m1: Vec<Vec<f32>> = (0..3).map(|_| random_vec(8, &mut rng)).collect();
        let m2: Vec<Vec<f32>> = (0..4).map(|_| random_vec(8, &mut rng)).collect();
        let fused = cma.fuse(&m1, &m2);
        assert_eq!(fused.len(), 16, "fuse should produce 2 * d_model = 16");
    }

    #[test]
    fn test_cross_modal_attention_multi_head() {
        let mut rng = make_rng(32);
        let cma = CmrCrossModalAttention::new(16, 8, &mut rng);
        let m1: Vec<Vec<f32>> = (0..2).map(|_| random_vec(16, &mut rng)).collect();
        let m2: Vec<Vec<f32>> = (0..2).map(|_| random_vec(16, &mut rng)).collect();
        let out = cma.attend(&m1, &m2);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 16);
    }

    // ── 5. ZeroShotClassifier ────────────────────────────────────────────────

    #[test]
    fn test_zero_shot_classifier_no_classes() {
        let mut rng = make_rng(40);
        let de = make_dual_encoder(&mut rng);
        let zsc = ZeroShotClassifier::new(de);
        let v_emb = l2_normalize_f32(&random_vec(16, &mut rng));
        let result = zsc.classify(&v_emb);
        assert!(result.is_empty());
    }

    #[test]
    fn test_zero_shot_classifier_add_classes() {
        let mut rng = make_rng(41);
        let de = make_dual_encoder(&mut rng);
        let mut zsc = ZeroShotClassifier::new(de);
        zsc.add_class("cat", l2_normalize_f32(&random_vec(16, &mut rng)));
        zsc.add_class("dog", l2_normalize_f32(&random_vec(16, &mut rng)));
        assert_eq!(zsc.class_names.len(), 2);
    }

    #[test]
    fn test_zero_shot_classifier_classify_sorted() {
        let mut rng = make_rng(42);
        let de = make_dual_encoder(&mut rng);
        let mut zsc = ZeroShotClassifier::new(de);
        for i in 0..5 {
            zsc.add_class(
                format!("class_{i}"),
                l2_normalize_f32(&random_vec(16, &mut rng)),
            );
        }
        let v_emb = l2_normalize_f32(&random_vec(16, &mut rng));
        let result = zsc.classify(&v_emb);
        assert_eq!(result.len(), 5);
        for i in 0..result.len() - 1 {
            assert!(
                result[i].1 >= result[i + 1].1,
                "scores should be descending"
            );
        }
    }

    #[test]
    fn test_zero_shot_classifier_top_k() {
        let mut rng = make_rng(43);
        let de = make_dual_encoder(&mut rng);
        let mut zsc = ZeroShotClassifier::new(de);
        for i in 0..8 {
            zsc.add_class(
                format!("class_{i}"),
                l2_normalize_f32(&random_vec(16, &mut rng)),
            );
        }
        let v_emb = l2_normalize_f32(&random_vec(16, &mut rng));
        let result = zsc.top_k_classes(&v_emb, 3);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_zero_shot_classifier_batch() {
        let mut rng = make_rng(44);
        let de = make_dual_encoder(&mut rng);
        let mut zsc = ZeroShotClassifier::new(de);
        zsc.add_class("a", l2_normalize_f32(&random_vec(16, &mut rng)));
        zsc.add_class("b", l2_normalize_f32(&random_vec(16, &mut rng)));
        let v_embs: Vec<Vec<f32>> = (0..4)
            .map(|_| l2_normalize_f32(&random_vec(16, &mut rng)))
            .collect();
        let results = zsc.batch_classify(&v_embs);
        assert_eq!(results.len(), 4);
        for r in &results {
            assert_eq!(r.len(), 2);
        }
    }

    // ── 6. MultiModalEmbedder ────────────────────────────────────────────────

    fn make_embedder(rng: &mut StdRng) -> MultiModalEmbedder {
        let config = EmbedderConfig {
            modality_dims: vec![
                (ModalityType::Vision, 64),
                (ModalityType::Text, 32),
                (ModalityType::Audio, 48),
            ],
            joint_dim: 16,
        };
        MultiModalEmbedder::new(config, rng)
    }

    #[test]
    fn test_multi_modal_embedder_vision_embed() {
        let mut rng = make_rng(50);
        let emb = make_embedder(&mut rng);
        let x = random_vec(64, &mut rng);
        let out = emb.embed(ModalityType::Vision, &x);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_multi_modal_embedder_text_embed() {
        let mut rng = make_rng(51);
        let emb = make_embedder(&mut rng);
        let x = random_vec(32, &mut rng);
        let out = emb.embed(ModalityType::Text, &x);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_multi_modal_embedder_fuse() {
        let mut rng = make_rng(52);
        let emb = make_embedder(&mut rng);
        let v = emb.embed(ModalityType::Vision, &random_vec(64, &mut rng));
        let t = emb.embed(ModalityType::Text, &random_vec(32, &mut rng));
        let pairs = vec![(ModalityType::Vision, v), (ModalityType::Text, t)];
        let fused = emb.fuse_embeddings(&pairs);
        // Both embeddings have same dim (joint_dim=16), so mean pooling → len 16
        assert_eq!(fused.len(), 16);
    }

    #[test]
    fn test_multi_modal_cross_modal_score_range() {
        let mut rng = make_rng(53);
        let a = l2_normalize_f32(&random_vec(16, &mut rng));
        let b = l2_normalize_f32(&random_vec(16, &mut rng));
        let s = MultiModalEmbedder::cross_modal_score(&a, &b);
        assert!((-1.01..=1.01).contains(&s), "cosine score out of range: {s}");
    }

    // ── 7. RetrievalMetrics ──────────────────────────────────────────────────

    #[test]
    fn test_retrieval_metrics_recall_at_1_perfect() {
        let retrieved = vec![vec![0_usize, 1, 2]];
        let relevant = vec![vec![0_usize]];
        let r = RetrievalMetrics::recall_at_k(&retrieved, &relevant, 1);
        assert!((r - 1.0).abs() < 1e-5, "perfect recall@1 should be 1.0");
    }

    #[test]
    fn test_retrieval_metrics_recall_at_1_zero() {
        let retrieved = vec![vec![1_usize, 2, 3]];
        let relevant = vec![vec![0_usize]];
        let r = RetrievalMetrics::recall_at_k(&retrieved, &relevant, 1);
        assert!(r.abs() < 1e-5, "miss at rank 1 should give 0.0");
    }

    #[test]
    fn test_retrieval_metrics_recall_at_5() {
        // 2/4 queries have relevant item in top-5
        let retrieved = vec![
            vec![0_usize, 1, 2, 3, 4],
            vec![5_usize, 6, 7, 8, 9],
            vec![0_usize, 1, 2, 3, 4],
            vec![5_usize, 6, 7, 8, 9],
        ];
        let relevant = vec![vec![0_usize], vec![0_usize], vec![0_usize], vec![0_usize]];
        let r = RetrievalMetrics::recall_at_k(&retrieved, &relevant, 5);
        assert!((r - 0.5).abs() < 1e-5, "recall@5 should be 0.5, got {r}");
    }

    #[test]
    fn test_retrieval_metrics_map_perfect() {
        // All retrieved are relevant
        let retrieved = vec![vec![0_usize, 1, 2]];
        let relevant = vec![vec![0_usize, 1, 2]];
        let map = RetrievalMetrics::mean_average_precision(&retrieved, &relevant);
        assert!(
            (map - 1.0).abs() < 1e-5,
            "perfect MAP should be 1.0, got {map}"
        );
    }

    #[test]
    fn test_retrieval_metrics_map_zero() {
        let retrieved = vec![vec![3_usize, 4, 5]];
        let relevant = vec![vec![0_usize, 1, 2]];
        let map = RetrievalMetrics::mean_average_precision(&retrieved, &relevant);
        assert!(map.abs() < 1e-5, "no overlap → MAP=0, got {map}");
    }

    #[test]
    fn test_retrieval_metrics_ndcg_perfect() {
        let retrieved = vec![vec![0_usize, 1, 2]];
        let relevant = vec![vec![0_usize, 1, 2]];
        let ndcg = RetrievalMetrics::ndcg_at_k(&retrieved, &relevant, 3);
        assert!(
            (ndcg - 1.0).abs() < 1e-5,
            "perfect NDCG should be 1.0, got {ndcg}"
        );
    }

    #[test]
    fn test_retrieval_metrics_median_rank() {
        // First relevant item found at position 1 (0-indexed) → rank 2
        let retrieved = vec![vec![5_usize, 0, 2], vec![5_usize, 0, 2]];
        let relevant = vec![vec![0_usize], vec![0_usize]];
        let med = RetrievalMetrics::median_rank(&retrieved, &relevant);
        assert!(
            (med - 2.0).abs() < 1e-5,
            "median rank should be 2, got {med}"
        );
    }

    #[test]
    fn test_retrieval_metrics_r_precision() {
        // R=2 relevant, first 2 retrieved are relevant → R-prec = 1.0
        let retrieved = vec![vec![0_usize, 1, 2]];
        let relevant = vec![vec![0_usize, 1]];
        let rp = RetrievalMetrics::r_precision(&retrieved, &relevant);
        assert!(
            (rp - 1.0).abs() < 1e-5,
            "R-precision should be 1.0, got {rp}"
        );
    }

    #[test]
    fn test_retrieval_eval_report_fields() {
        let report = RetrievalEvalReport {
            r1: 0.5,
            r5: 0.8,
            r10: 0.9,
            map: 0.6,
            ndcg: 0.7,
            med_rank: 3.0,
        };
        assert!((report.r1 - 0.5).abs() < 1e-5);
        assert!((report.r5 - 0.8).abs() < 1e-5);
        assert!((report.r10 - 0.9).abs() < 1e-5);
        assert!((report.map - 0.6).abs() < 1e-5);
        assert!((report.ndcg - 0.7).abs() < 1e-5);
        assert!((report.med_rank - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_evaluate_full_report() {
        let retrieved = vec![
            vec![0_usize, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            vec![1_usize, 0, 2, 3, 4, 5, 6, 7, 8, 9],
        ];
        let relevant = vec![vec![0_usize], vec![1_usize]];
        let report = RetrievalMetrics::evaluate(&retrieved, &relevant);
        assert!(
            (report.r1 - 1.0).abs() < 1e-5,
            "r1 should be 1.0, got {}",
            report.r1
        );
        assert!((report.map - 1.0).abs() < 1e-5);
        assert!((report.ndcg - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_contrastive_loss_lower_for_aligned() {
        let mut rng = make_rng(60);
        let de = DualEncoder::new(enc_config(8, 8), enc_config(8, 8), 0.1, &mut rng);

        // Aligned pairs: encode same vector from both encoders
        let inputs: Vec<Vec<f32>> = (0..4)
            .map(|_| {
                let v = random_vec(8, &mut rng);
                l2_normalize_f32(&v)
            })
            .collect();
        let aligned_v: Vec<Vec<f32>> = inputs.clone();
        let aligned_t: Vec<Vec<f32>> = inputs.clone();
        let aligned_loss = de.contrastive_loss(&aligned_v, &aligned_t);

        // Misaligned pairs: random embeddings
        let rand_v: Vec<Vec<f32>> = (0..4)
            .map(|_| l2_normalize_f32(&random_vec(8, &mut rng)))
            .collect();
        let rand_t: Vec<Vec<f32>> = (0..4)
            .map(|_| l2_normalize_f32(&random_vec(8, &mut rng)))
            .collect();
        let random_loss = de.contrastive_loss(&rand_v, &rand_t);

        assert!(
            aligned_loss <= random_loss,
            "aligned loss ({aligned_loss}) should be <= random loss ({random_loss})"
        );
    }

    #[test]
    fn test_dual_encoder_symmetric_retrieval() {
        let mut rng = make_rng(70);
        let de = DualEncoder::new(enc_config(8, 4), enc_config(8, 4), 1.0, &mut rng);

        let embs: Vec<Vec<f32>> = (0..6)
            .map(|i| {
                let mut v = vec![0.0_f32; 4];
                v[i % 4] = 1.0;
                v
            })
            .collect();

        // Query = embs[3]
        let query = embs[3].clone();
        let t2v = de.retrieve_vision(&query, &embs, 1);
        let v2t = de.retrieve_text(&query, &embs, 1);
        // Both should find the most similar (index 3)
        assert_eq!(t2v[0], 3);
        assert_eq!(v2t[0], 3);
    }
}
