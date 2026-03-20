//! Advanced multimodal foundation model components.
//!
//! Adds unified multimodal architectures (UnifiedIO, PaLI, CogVLM),
//! multimodal alignment (N-way contrastive, modality gap, unified embedding space),
//! phrase grounding, referring expression comprehension, and visual chain-of-thought reasoning.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::TensorError;

use super::{dot, gelu, l2_normalize, layer_norm, matvec, rand_mat, rand_vec, sdp_attention, sinusoidal_pe, softmax};

// ─────────────────────────────────────────────────────────────────────────────
// 1. Unified Multimodal Models
// ─────────────────────────────────────────────────────────────────────────────

/// Modality type for unified sequence encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnifiedModality {
    /// Plain text token sequence.
    Text,
    /// Image patch token sequence.
    ImagePatch,
    /// Audio frame token sequence.
    AudioFrame,
    /// Video frame token sequence.
    VideoFrame,
}

/// Token in a unified sequence with modality tag and embedding.
#[derive(Debug, Clone)]
pub struct UnifiedToken {
    /// Modality this token belongs to.
    pub modality: UnifiedModality,
    /// Embedding vector for this token.
    pub embedding: Vec<f64>,
}

/// UnifiedIO (Lu 2022): encode all modalities as token sequences, unify generation via seq2seq.
///
/// All modalities (text, image patches, audio frames) are projected into a shared token space
/// and processed by a single encoder-decoder transformer.
#[derive(Debug, Clone)]
pub struct UnifiedIOModel {
    /// Shared token dimension.
    pub d_model: usize,
    /// Vocabulary size for output generation.
    pub vocab_size: usize,
    /// Modality-specific projection matrices (one per modality).
    text_proj: Vec<Vec<f64>>,
    image_proj: Vec<Vec<f64>>,
    audio_proj: Vec<Vec<f64>>,
    /// Encoder self-attention.
    enc_wq: Vec<Vec<f64>>,
    enc_wk: Vec<Vec<f64>>,
    enc_wv: Vec<Vec<f64>>,
    enc_wo: Vec<Vec<f64>>,
    enc_ff1: Vec<Vec<f64>>,
    enc_ff2: Vec<Vec<f64>>,
    /// Decoder cross-attention to encoder output.
    dec_xwq: Vec<Vec<f64>>,
    dec_xwk: Vec<Vec<f64>>,
    dec_xwv: Vec<Vec<f64>>,
    dec_xwo: Vec<Vec<f64>>,
    /// Output head: [vocab_size, d_model].
    lm_head: Vec<Vec<f64>>,
}

impl UnifiedIOModel {
    /// Create a new UnifiedIO model.
    ///
    /// `in_dim` is the shared input feature dimension before projection.
    pub fn new(in_dim: usize, d_model: usize, vocab_size: usize, rng: &mut StdRng) -> Self {
        Self {
            d_model,
            vocab_size,
            text_proj: rand_mat(d_model, in_dim, rng),
            image_proj: rand_mat(d_model, in_dim, rng),
            audio_proj: rand_mat(d_model, in_dim, rng),
            enc_wq: rand_mat(d_model, d_model, rng),
            enc_wk: rand_mat(d_model, d_model, rng),
            enc_wv: rand_mat(d_model, d_model, rng),
            enc_wo: rand_mat(d_model, d_model, rng),
            enc_ff1: rand_mat(d_model * 4, d_model, rng),
            enc_ff2: rand_mat(d_model, d_model * 4, rng),
            dec_xwq: rand_mat(d_model, d_model, rng),
            dec_xwk: rand_mat(d_model, d_model, rng),
            dec_xwv: rand_mat(d_model, d_model, rng),
            dec_xwo: rand_mat(d_model, d_model, rng),
            lm_head: rand_mat(vocab_size, d_model, rng),
        }
    }

    /// Project tokens from any modality into the unified token space.
    fn project_tokens(&self, tokens: &[UnifiedToken]) -> Vec<Vec<f64>> {
        tokens
            .iter()
            .enumerate()
            .map(|(i, tok)| {
                let proj = match tok.modality {
                    UnifiedModality::Text => matvec(&self.text_proj, &tok.embedding),
                    UnifiedModality::ImagePatch => matvec(&self.image_proj, &tok.embedding),
                    UnifiedModality::AudioFrame | UnifiedModality::VideoFrame => {
                        matvec(&self.audio_proj, &tok.embedding)
                    }
                };
                // Add sinusoidal positional encoding
                let pe = sinusoidal_pe(i, self.d_model);
                proj.iter().zip(pe.iter()).map(|(p, e)| p + e).collect()
            })
            .collect()
    }

    /// Run the encoder transformer on projected tokens.
    fn encode(&self, projected: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if projected.is_empty() {
            return Vec::new();
        }
        let q: Vec<Vec<f64>> = projected.iter().map(|x| matvec(&self.enc_wq, x)).collect();
        let k: Vec<Vec<f64>> = projected.iter().map(|x| matvec(&self.enc_wk, x)).collect();
        let v: Vec<Vec<f64>> = projected.iter().map(|x| matvec(&self.enc_wv, x)).collect();
        let attn = sdp_attention(&q, &k, &v);
        let after_attn: Vec<Vec<f64>> = projected
            .iter()
            .zip(attn.iter())
            .map(|(x, a)| {
                let proj = matvec(&self.enc_wo, a);
                let res: Vec<f64> = x.iter().zip(proj.iter()).map(|(xi, pi)| xi + pi).collect();
                layer_norm(&res)
            })
            .collect();
        after_attn
            .iter()
            .map(|x| {
                let h: Vec<f64> = matvec(&self.enc_ff1, x).iter().map(|v| gelu(*v)).collect();
                let ff = matvec(&self.enc_ff2, &h);
                let res: Vec<f64> = x.iter().zip(ff.iter()).map(|(a, b)| a + b).collect();
                layer_norm(&res)
            })
            .collect()
    }

    /// Forward pass: given input tokens of any modalities, decode `n_steps` output logits.
    ///
    /// Returns `[n_steps, vocab_size]` logit vectors.
    pub fn forward(&self, input_tokens: &[UnifiedToken], n_steps: usize) -> Vec<Vec<f64>> {
        let projected = self.project_tokens(input_tokens);
        let encoded = self.encode(&projected);
        if encoded.is_empty() || n_steps == 0 {
            return Vec::new();
        }
        // Decoder: for each step, cross-attend to encoder output
        (0..n_steps)
            .map(|step| {
                let query = sinusoidal_pe(step, self.d_model);
                let q = matvec(&self.dec_xwq, &query);
                let k: Vec<Vec<f64>> = encoded.iter().map(|x| matvec(&self.dec_xwk, x)).collect();
                let v: Vec<Vec<f64>> = encoded.iter().map(|x| matvec(&self.dec_xwv, x)).collect();
                // Single-query cross-attention
                let scale = (self.d_model as f64).sqrt();
                let scores: Vec<f64> = k.iter().map(|ki| dot(&q, ki) / scale).collect();
                let weights = softmax(&scores);
                let d = if v.is_empty() { self.d_model } else { v[0].len() };
                let mut ctx = vec![0.0_f64; d];
                for (w, vi) in weights.iter().zip(v.iter()) {
                    for (c, &vi_j) in ctx.iter_mut().zip(vi.iter()) {
                        *c += w * vi_j;
                    }
                }
                let dec_out = matvec(&self.dec_xwo, &ctx);
                matvec(&self.lm_head, &dec_out)
            })
            .collect()
    }
}

/// PaLI (Chen 2022): image encoder + text encoder → combined cross-attention language model.
///
/// PaLI-style architecture combining a large vision encoder with a text encoder-decoder via
/// cross-attention for unified visual language understanding and generation.
#[derive(Debug, Clone)]
pub struct PaLiModel {
    /// Vision encoder output dimension.
    pub vis_dim: usize,
    /// Language model dimension.
    pub lang_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Visual projection to language space.
    vis_proj: Vec<Vec<f64>>,
    /// Text encoder self-attention.
    txt_wq: Vec<Vec<f64>>,
    txt_wk: Vec<Vec<f64>>,
    txt_wv: Vec<Vec<f64>>,
    txt_wo: Vec<Vec<f64>>,
    /// Cross-attention from text to vision.
    xattn_wq: Vec<Vec<f64>>,
    xattn_wk: Vec<Vec<f64>>,
    xattn_wv: Vec<Vec<f64>>,
    xattn_wo: Vec<Vec<f64>>,
    /// Output language model head.
    lm_head: Vec<Vec<f64>>,
}

impl PaLiModel {
    /// Create a new PaLI model.
    pub fn new(vis_dim: usize, lang_dim: usize, vocab_size: usize, rng: &mut StdRng) -> Self {
        Self {
            vis_dim,
            lang_dim,
            vocab_size,
            vis_proj: rand_mat(lang_dim, vis_dim, rng),
            txt_wq: rand_mat(lang_dim, lang_dim, rng),
            txt_wk: rand_mat(lang_dim, lang_dim, rng),
            txt_wv: rand_mat(lang_dim, lang_dim, rng),
            txt_wo: rand_mat(lang_dim, lang_dim, rng),
            xattn_wq: rand_mat(lang_dim, lang_dim, rng),
            xattn_wk: rand_mat(lang_dim, lang_dim, rng),
            xattn_wv: rand_mat(lang_dim, lang_dim, rng),
            xattn_wo: rand_mat(lang_dim, lang_dim, rng),
            lm_head: rand_mat(vocab_size, lang_dim, rng),
        }
    }

    /// Encode image patches into language-space tokens.
    pub fn encode_image(&self, image_patches: &[Vec<f64>]) -> Vec<Vec<f64>> {
        image_patches
            .iter()
            .map(|p| matvec(&self.vis_proj, p))
            .collect()
    }

    /// Forward pass: given image patches and text tokens, return logits.
    ///
    /// `image_patches`: [n_patches, vis_dim], `text_tokens`: [n_text, lang_dim].
    /// Returns [n_text, vocab_size] logits.
    pub fn forward(
        &self,
        image_patches: &[Vec<f64>],
        text_tokens: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let vis_enc = self.encode_image(image_patches);
        if text_tokens.is_empty() {
            return Vec::new();
        }
        // Text self-attention
        let tq: Vec<Vec<f64>> = text_tokens.iter().map(|x| matvec(&self.txt_wq, x)).collect();
        let tk: Vec<Vec<f64>> = text_tokens.iter().map(|x| matvec(&self.txt_wk, x)).collect();
        let tv: Vec<Vec<f64>> = text_tokens.iter().map(|x| matvec(&self.txt_wv, x)).collect();
        let txt_self_attn = sdp_attention(&tq, &tk, &tv);
        let txt_after_self: Vec<Vec<f64>> = text_tokens
            .iter()
            .zip(txt_self_attn.iter())
            .map(|(x, a)| {
                let proj = matvec(&self.txt_wo, a);
                let res: Vec<f64> = x.iter().zip(proj.iter()).map(|(xi, pi)| xi + pi).collect();
                layer_norm(&res)
            })
            .collect();
        // Cross-attention: text queries attend to visual keys/values
        let xq: Vec<Vec<f64>> = txt_after_self.iter().map(|x| matvec(&self.xattn_wq, x)).collect();
        let xk: Vec<Vec<f64>> = if vis_enc.is_empty() {
            txt_after_self.iter().map(|x| matvec(&self.xattn_wk, x)).collect()
        } else {
            vis_enc.iter().map(|x| matvec(&self.xattn_wk, x)).collect()
        };
        let xv: Vec<Vec<f64>> = if vis_enc.is_empty() {
            txt_after_self.iter().map(|x| matvec(&self.xattn_wv, x)).collect()
        } else {
            vis_enc.iter().map(|x| matvec(&self.xattn_wv, x)).collect()
        };
        let cross_attn = sdp_attention(&xq, &xk, &xv);
        let fused: Vec<Vec<f64>> = txt_after_self
            .iter()
            .zip(cross_attn.iter())
            .map(|(x, a)| {
                let proj = matvec(&self.xattn_wo, a);
                let res: Vec<f64> = x.iter().zip(proj.iter()).map(|(xi, pi)| xi + pi).collect();
                layer_norm(&res)
            })
            .collect();
        fused.iter().map(|x| matvec(&self.lm_head, x)).collect()
    }
}

/// CogVLM interleaved vision-language layer (Wang 2023).
///
/// Implements an interleaved transformer layer with a dedicated "visual expert" FFN branch
/// that processes visual tokens separately from text, allowing specialized visual reasoning.
#[derive(Debug, Clone)]
pub struct CogVlmLayer {
    /// Model dimension.
    pub d_model: usize,
    /// Shared attention weights.
    wq: Vec<Vec<f64>>,
    wk: Vec<Vec<f64>>,
    wv: Vec<Vec<f64>>,
    wo: Vec<Vec<f64>>,
    /// Standard text FFN.
    text_ff1: Vec<Vec<f64>>,
    text_ff2: Vec<Vec<f64>>,
    /// Visual expert FFN (separate parameters for visual tokens).
    vis_ff1: Vec<Vec<f64>>,
    vis_ff2: Vec<Vec<f64>>,
}

impl CogVlmLayer {
    /// Create a new CogVLM interleaved layer.
    pub fn new(d_model: usize, rng: &mut StdRng) -> Self {
        Self {
            d_model,
            wq: rand_mat(d_model, d_model, rng),
            wk: rand_mat(d_model, d_model, rng),
            wv: rand_mat(d_model, d_model, rng),
            wo: rand_mat(d_model, d_model, rng),
            text_ff1: rand_mat(d_model * 4, d_model, rng),
            text_ff2: rand_mat(d_model, d_model * 4, rng),
            vis_ff1: rand_mat(d_model * 4, d_model, rng),
            vis_ff2: rand_mat(d_model, d_model * 4, rng),
        }
    }

    /// Forward pass with interleaved visual expert FFN.
    ///
    /// `tokens`: all tokens (visual + text concatenated), `n_visual`: count of leading visual tokens.
    /// Returns updated token representations.
    pub fn forward(&self, tokens: &[Vec<f64>], n_visual: usize) -> Vec<Vec<f64>> {
        if tokens.is_empty() {
            return Vec::new();
        }
        // Unified self-attention across all tokens
        let q: Vec<Vec<f64>> = tokens.iter().map(|x| matvec(&self.wq, x)).collect();
        let k: Vec<Vec<f64>> = tokens.iter().map(|x| matvec(&self.wk, x)).collect();
        let v: Vec<Vec<f64>> = tokens.iter().map(|x| matvec(&self.wv, x)).collect();
        let attn = sdp_attention(&q, &k, &v);
        let after_attn: Vec<Vec<f64>> = tokens
            .iter()
            .zip(attn.iter())
            .map(|(x, a)| {
                let proj = matvec(&self.wo, a);
                let res: Vec<f64> = x.iter().zip(proj.iter()).map(|(xi, pi)| xi + pi).collect();
                layer_norm(&res)
            })
            .collect();
        // Route each token through the appropriate FFN (visual expert vs text FFN)
        after_attn
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let (ff1, ff2) = if i < n_visual {
                    (&self.vis_ff1, &self.vis_ff2)
                } else {
                    (&self.text_ff1, &self.text_ff2)
                };
                let h: Vec<f64> = matvec(ff1, x).iter().map(|v| gelu(*v)).collect();
                let ff = matvec(ff2, &h);
                let res: Vec<f64> = x.iter().zip(ff.iter()).map(|(a, b)| a + b).collect();
                layer_norm(&res)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Multimodal Alignment
// ─────────────────────────────────────────────────────────────────────────────

/// N-way multimodal contrastive aligner generalizing CLIP to N > 2 modalities.
///
/// All pairs of modalities are aligned simultaneously via pairwise InfoNCE losses,
/// encouraging a shared embedding space where matching samples across any two modalities
/// are closer than non-matching samples.
#[derive(Debug, Clone)]
pub struct MultimodalAligner {
    /// Number of modalities.
    pub n_modalities: usize,
    /// Shared embedding dimension.
    pub embed_dim: usize,
    /// Per-modality projection matrices: [n_modalities × (embed_dim × in_dim)].
    projectors: Vec<Vec<Vec<f64>>>,
    /// InfoNCE temperature (log-domain, initialized to 0).
    pub log_temperature: f64,
}

impl MultimodalAligner {
    /// Create a new N-modality aligner.
    ///
    /// `in_dims`: input dimension for each modality.
    pub fn new(in_dims: &[usize], embed_dim: usize, rng: &mut StdRng) -> Self {
        let projectors = in_dims
            .iter()
            .map(|&d| rand_mat(embed_dim, d, rng))
            .collect();
        Self {
            n_modalities: in_dims.len(),
            embed_dim,
            projectors,
            log_temperature: 0.0,
        }
    }

    /// Project features for a given modality index.
    pub fn project(&self, modality_idx: usize, features: &[f64]) -> Vec<f64> {
        if modality_idx >= self.projectors.len() {
            return vec![0.0; self.embed_dim];
        }
        l2_normalize(&matvec(&self.projectors[modality_idx], features))
    }

    /// Compute pairwise InfoNCE loss across all modality pairs for a batch.
    ///
    /// `batch_embeddings`: \[n_modalities\]\[batch_size\]\[embed_dim\] — pre-projected embeddings.
    /// Returns the average loss across all pairs.
    pub fn pairwise_contrastive_loss(
        &self,
        batch_embeddings: &[Vec<Vec<f64>>],
    ) -> Result<f64, TensorError> {
        let m = batch_embeddings.len();
        if m < 2 {
            return Err(TensorError::invalid_argument_op(
                "MultimodalAligner::pairwise_contrastive_loss",
                "need at least 2 modalities",
            ));
        }
        let n = batch_embeddings[0].len();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "MultimodalAligner::pairwise_contrastive_loss",
                "batch must be non-empty",
            ));
        }
        let temp = self.log_temperature.exp().max(1e-8);
        let mut total_loss = 0.0_f64;
        let mut pair_count = 0usize;
        for i in 0..m {
            for j in (i + 1)..m {
                // InfoNCE for modality pair (i, j)
                let emb_i = &batch_embeddings[i];
                let emb_j = &batch_embeddings[j];
                if emb_i.len() != n || emb_j.len() != n {
                    continue;
                }
                let mut loss = 0.0_f64;
                for a in 0..n {
                    let scores: Vec<f64> = (0..n)
                        .map(|b| dot(&emb_i[a], &emb_j[b]) / temp)
                        .collect();
                    let sm = softmax(&scores);
                    loss -= sm[a].max(1e-12).ln();
                }
                total_loss += loss / n as f64;
                pair_count += 1;
            }
        }
        if pair_count == 0 {
            return Ok(0.0);
        }
        Ok(total_loss / pair_count as f64)
    }
}

/// Measure the modality gap between two sets of embeddings.
///
/// The modality gap (Liang et al. 2022) is the mean L2 distance between the
/// centroid of modality A embeddings and the centroid of modality B embeddings
/// in a shared embedding space.
#[derive(Debug, Clone)]
pub struct ModalityGapAnalyzer;

impl ModalityGapAnalyzer {
    /// Create a new gap analyzer.
    pub fn new() -> Self {
        Self
    }

    /// Compute the modality gap between two embedding sets.
    ///
    /// Returns the Euclidean distance between their centroids.
    pub fn gap(
        &self,
        embeddings_a: &[Vec<f64>],
        embeddings_b: &[Vec<f64>],
    ) -> Result<f64, TensorError> {
        if embeddings_a.is_empty() || embeddings_b.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "ModalityGapAnalyzer::gap",
                "embedding sets must be non-empty",
            ));
        }
        let d = embeddings_a[0].len();
        let centroid_a = Self::centroid(embeddings_a, d);
        let centroid_b = Self::centroid(embeddings_b, d);
        let gap = centroid_a
            .iter()
            .zip(centroid_b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        Ok(gap)
    }

    fn centroid(embeddings: &[Vec<f64>], d: usize) -> Vec<f64> {
        let n = embeddings.len() as f64;
        let mut c = vec![0.0_f64; d];
        for emb in embeddings {
            for (ci, &ei) in c.iter_mut().zip(emb.iter()) {
                *ci += ei / n;
            }
        }
        c
    }

    /// Compute the cosine similarity between two modality centroids.
    pub fn centroid_cosine_similarity(
        &self,
        embeddings_a: &[Vec<f64>],
        embeddings_b: &[Vec<f64>],
    ) -> Result<f64, TensorError> {
        if embeddings_a.is_empty() || embeddings_b.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "ModalityGapAnalyzer::centroid_cosine_similarity",
                "embedding sets must be non-empty",
            ));
        }
        let d = embeddings_a[0].len();
        let ca = l2_normalize(&Self::centroid(embeddings_a, d));
        let cb = l2_normalize(&Self::centroid(embeddings_b, d));
        Ok(dot(&ca, &cb))
    }
}

impl Default for ModalityGapAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Project all modalities into a single unified embedding space with orthogonality regularization.
///
/// Maintains per-modality linear projectors and a shared L2-normalized output space.
/// An orthogonality loss encourages modality-specific bases to be linearly independent.
#[derive(Debug, Clone)]
pub struct UnifiedEmbeddingSpace {
    /// Number of modalities.
    pub n_modalities: usize,
    /// Shared embedding dimension.
    pub embed_dim: usize,
    /// Per-modality projection matrices.
    projectors: Vec<Vec<Vec<f64>>>,
    /// Orthogonality regularization weight.
    pub ortho_weight: f64,
}

impl UnifiedEmbeddingSpace {
    /// Create a new unified embedding space.
    pub fn new(in_dims: &[usize], embed_dim: usize, ortho_weight: f64, rng: &mut StdRng) -> Self {
        let projectors = in_dims
            .iter()
            .map(|&d| rand_mat(embed_dim, d, rng))
            .collect();
        Self {
            n_modalities: in_dims.len(),
            embed_dim,
            projectors,
            ortho_weight,
        }
    }

    /// Embed features for a given modality (L2-normalized output).
    pub fn embed(&self, modality_idx: usize, features: &[f64]) -> Vec<f64> {
        if modality_idx >= self.projectors.len() {
            return vec![0.0; self.embed_dim];
        }
        l2_normalize(&matvec(&self.projectors[modality_idx], features))
    }

    /// Compute the orthogonality regularization loss across all modality projection matrices.
    ///
    /// For each pair (i, j) of projectors, penalizes `‖P_i P_j^T‖_F²` to encourage orthogonal bases.
    pub fn orthogonality_loss(&self) -> f64 {
        let mut loss = 0.0_f64;
        let m = self.n_modalities;
        for i in 0..m {
            for j in (i + 1)..m {
                let pi = &self.projectors[i];
                let pj = &self.projectors[j];
                // ‖P_i P_j^T‖_F²: sum of squared dot products between rows
                for ri in pi {
                    for rj in pj {
                        let d = dot(ri, rj);
                        loss += d * d;
                    }
                }
            }
        }
        self.ortho_weight * loss
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Grounding
// ─────────────────────────────────────────────────────────────────────────────

/// PhrasalGrounder: given text phrase features + image region features, localize regions.
///
/// Cross-attention from phrase queries to image region keys/values, followed by a sigmoid
/// output to produce per-region relevance scores.
#[derive(Debug, Clone)]
pub struct PhrasalGrounder {
    /// Model dimension.
    pub d_model: usize,
    phrase_proj: Vec<Vec<f64>>,
    region_proj: Vec<Vec<f64>>,
    /// Cross-attention.
    wq: Vec<Vec<f64>>,
    wk: Vec<Vec<f64>>,
    wv: Vec<Vec<f64>>,
    wo: Vec<Vec<f64>>,
    /// Sigmoid scoring head: [1, d_model].
    score_head: Vec<Vec<f64>>,
}

impl PhrasalGrounder {
    /// Create a new phrasal grounder.
    pub fn new(phrase_dim: usize, region_dim: usize, d_model: usize, rng: &mut StdRng) -> Self {
        Self {
            d_model,
            phrase_proj: rand_mat(d_model, phrase_dim, rng),
            region_proj: rand_mat(d_model, region_dim, rng),
            wq: rand_mat(d_model, d_model, rng),
            wk: rand_mat(d_model, d_model, rng),
            wv: rand_mat(d_model, d_model, rng),
            wo: rand_mat(d_model, d_model, rng),
            score_head: rand_mat(1, d_model, rng),
        }
    }

    /// Ground `phrases` to `regions`, returning relevance scores `[n_phrases, n_regions]`.
    pub fn ground(
        &self,
        phrases: &[Vec<f64>],
        regions: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        if phrases.is_empty() || regions.is_empty() {
            return Vec::new();
        }
        let phrase_emb: Vec<Vec<f64>> = phrases.iter().map(|p| matvec(&self.phrase_proj, p)).collect();
        let region_emb: Vec<Vec<f64>> = regions.iter().map(|r| matvec(&self.region_proj, r)).collect();
        let q: Vec<Vec<f64>> = phrase_emb.iter().map(|x| matvec(&self.wq, x)).collect();
        let k: Vec<Vec<f64>> = region_emb.iter().map(|x| matvec(&self.wk, x)).collect();
        let v: Vec<Vec<f64>> = region_emb.iter().map(|x| matvec(&self.wv, x)).collect();
        let attn_out = sdp_attention(&q, &k, &v);
        // Score each (phrase, region) pair via sigmoid
        phrases
            .iter()
            .enumerate()
            .map(|(pi, _)| {
                let ctx = matvec(&self.wo, &attn_out[pi]);
                regions
                    .iter()
                    .enumerate()
                    .map(|(ri, _)| {
                        let r_emb = &region_emb[ri];
                        let combined: Vec<f64> = ctx.iter().zip(r_emb.iter()).map(|(a, b)| a + b).collect();
                        let score = dot(&self.score_head[0], &combined);
                        1.0 / (1.0 + (-score).exp()) // sigmoid
                    })
                    .collect()
            })
            .collect()
    }
}

/// REC (Referring Expression Comprehension) decoder.
///
/// Given a fused multimodal representation, predicts bounding box coordinates (cx, cy, w, h)
/// normalized to [0, 1] via sigmoid activation.
#[derive(Debug, Clone)]
pub struct RecRefDecoder {
    /// Input feature dimension.
    pub in_dim: usize,
    /// MLP hidden dimension.
    hidden_dim: usize,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
    /// Box prediction head: [4, hidden_dim] → (cx, cy, w, h).
    box_head: Vec<Vec<f64>>,
}

impl RecRefDecoder {
    /// Create a new REC decoder.
    pub fn new(in_dim: usize, hidden_dim: usize, rng: &mut StdRng) -> Self {
        Self {
            in_dim,
            hidden_dim,
            w1: rand_mat(hidden_dim, in_dim, rng),
            b1: vec![0.0; hidden_dim],
            w2: rand_mat(hidden_dim, hidden_dim, rng),
            b2: vec![0.0; hidden_dim],
            box_head: rand_mat(4, hidden_dim, rng),
        }
    }

    /// Predict normalized bounding box (cx, cy, w, h) ∈ \[0,1\]^4 from fused representation.
    pub fn predict_box(&self, fused_repr: &[f64]) -> [f64; 4] {
        let h1: Vec<f64> = matvec(&self.w1, fused_repr)
            .iter()
            .zip(self.b1.iter())
            .map(|(v, b)| gelu(v + b))
            .collect();
        let h2: Vec<f64> = matvec(&self.w2, &h1)
            .iter()
            .zip(self.b2.iter())
            .map(|(v, b)| (v + b).max(0.0)) // ReLU
            .collect();
        let raw = matvec(&self.box_head, &h2);
        [
            1.0 / (1.0 + (-raw[0]).exp()), // cx ∈ [0,1]
            1.0 / (1.0 + (-raw[1]).exp()), // cy ∈ [0,1]
            1.0 / (1.0 + (-raw[2]).exp()), // w ∈ [0,1]
            1.0 / (1.0 + (-raw[3]).exp()), // h ∈ [0,1]
        ]
    }
}

/// Grounding evaluation metrics: IoU@threshold and Recall@K.
#[derive(Debug, Clone)]
pub struct GroundingMetrics {
    /// IoU threshold for a "correct" prediction (e.g., 0.5).
    pub iou_threshold: f64,
}

impl GroundingMetrics {
    /// Create a new `GroundingMetrics` with the given IoU threshold.
    pub fn new(iou_threshold: f64) -> Self {
        Self { iou_threshold }
    }

    /// Compute IoU between two boxes in (cx, cy, w, h) format.
    pub fn iou_cxcywh(pred: &[f64; 4], gt: &[f64; 4]) -> f64 {
        let p_x1 = pred[0] - pred[2] / 2.0;
        let p_y1 = pred[1] - pred[3] / 2.0;
        let p_x2 = pred[0] + pred[2] / 2.0;
        let p_y2 = pred[1] + pred[3] / 2.0;
        let g_x1 = gt[0] - gt[2] / 2.0;
        let g_y1 = gt[1] - gt[3] / 2.0;
        let g_x2 = gt[0] + gt[2] / 2.0;
        let g_y2 = gt[1] + gt[3] / 2.0;
        let inter_x1 = p_x1.max(g_x1);
        let inter_y1 = p_y1.max(g_y1);
        let inter_x2 = p_x2.min(g_x2);
        let inter_y2 = p_y2.min(g_y2);
        let inter_w = (inter_x2 - inter_x1).max(0.0);
        let inter_h = (inter_y2 - inter_y1).max(0.0);
        let inter = inter_w * inter_h;
        let area_p = pred[2] * pred[3];
        let area_g = gt[2] * gt[3];
        let union = area_p + area_g - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }

    /// Compute accuracy@IoU: fraction of predictions with IoU ≥ threshold.
    pub fn accuracy_at_iou(&self, preds: &[[f64; 4]], gts: &[[f64; 4]]) -> f64 {
        if preds.is_empty() || preds.len() != gts.len() {
            return 0.0;
        }
        let correct = preds
            .iter()
            .zip(gts.iter())
            .filter(|(p, g)| Self::iou_cxcywh(p, g) >= self.iou_threshold)
            .count();
        correct as f64 / preds.len() as f64
    }

    /// Compute Recall@K: fraction of ground-truth regions matched among top-K predictions.
    ///
    /// `scores`: [n_phrases × n_regions] relevance scores, `k`: top-K cutoff.
    pub fn recall_at_k(&self, scores: &[Vec<f64>], k: usize) -> f64 {
        if scores.is_empty() {
            return 0.0;
        }
        let n_correct: usize = scores
            .iter()
            .enumerate()
            .map(|(phrase_idx, row)| {
                // The "ground truth" region index for phrase i is i (diagonal match assumption)
                let gt_idx = phrase_idx;
                if gt_idx >= row.len() {
                    return 0;
                }
                let topk: usize = k.min(row.len());
                let mut indexed: Vec<(usize, f64)> = row.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                if indexed[..topk].iter().any(|(i, _)| *i == gt_idx) {
                    1
                } else {
                    0
                }
            })
            .sum();
        n_correct as f64 / scores.len() as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Multimodal Reasoning
// ─────────────────────────────────────────────────────────────────────────────

/// Visual Chain-of-Thought: interleave image descriptions with reasoning steps.
///
/// Generates a sequence of reasoning tokens by alternating between visual context
/// and language reasoning, each step attending to the visual encoder output.
#[derive(Debug, Clone)]
pub struct ChainOfThoughtMultimodal {
    /// Language model dimension.
    pub d_model: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Number of reasoning steps.
    pub n_steps: usize,
    /// Visual context projector.
    vis_proj: Vec<Vec<f64>>,
    /// Per-step reasoning heads: one language head per step.
    step_heads: Vec<Vec<Vec<f64>>>,
    /// Cross-attention to visual context.
    xwq: Vec<Vec<f64>>,
    xwk: Vec<Vec<f64>>,
    xwv: Vec<Vec<f64>>,
    xwo: Vec<Vec<f64>>,
}

impl ChainOfThoughtMultimodal {
    /// Create a new visual chain-of-thought model.
    pub fn new(vis_dim: usize, d_model: usize, vocab_size: usize, n_steps: usize, rng: &mut StdRng) -> Self {
        let step_heads = (0..n_steps).map(|_| rand_mat(vocab_size, d_model, rng)).collect();
        Self {
            d_model,
            vocab_size,
            n_steps,
            vis_proj: rand_mat(d_model, vis_dim, rng),
            step_heads,
            xwq: rand_mat(d_model, d_model, rng),
            xwk: rand_mat(d_model, d_model, rng),
            xwv: rand_mat(d_model, d_model, rng),
            xwo: rand_mat(d_model, d_model, rng),
        }
    }

    /// Generate chain-of-thought logits given visual tokens.
    ///
    /// Returns `[n_steps, vocab_size]` logits, one per reasoning step.
    pub fn generate(&self, visual_tokens: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let vis_enc: Vec<Vec<f64>> = visual_tokens
            .iter()
            .map(|v| matvec(&self.vis_proj, v))
            .collect();
        (0..self.n_steps)
            .map(|step| {
                // Step query: sinusoidal encoding of step index
                let query = sinusoidal_pe(step, self.d_model);
                let q = matvec(&self.xwq, &query);
                let scale = (self.d_model as f64).sqrt();
                if vis_enc.is_empty() {
                    return vec![0.0; self.vocab_size];
                }
                let scores: Vec<f64> = vis_enc.iter().map(|k| {
                    let ki = matvec(&self.xwk, k);
                    dot(&q, &ki) / scale
                }).collect();
                let weights = softmax(&scores);
                let d = vis_enc[0].len();
                let mut ctx = vec![0.0_f64; d];
                for (w, vi) in weights.iter().zip(vis_enc.iter()) {
                    let vi_proj = matvec(&self.xwv, vi);
                    for (c, &vij) in ctx.iter_mut().zip(vi_proj.iter()) {
                        *c += w * vij;
                    }
                }
                let out = matvec(&self.xwo, &ctx);
                matvec(&self.step_heads[step], &out)
            })
            .collect()
    }
}

/// Symbolic scene graph node.
#[derive(Debug, Clone)]
pub struct SceneNode {
    /// Object category index.
    pub category_id: usize,
    /// Feature embedding for this node.
    pub features: Vec<f64>,
}

/// Symbolic scene graph edge representing a relation.
#[derive(Debug, Clone)]
pub struct SceneEdge {
    /// Index of source node.
    pub src: usize,
    /// Index of target node.
    pub dst: usize,
    /// Relation type index.
    pub relation_id: usize,
}

/// SymbolicVisualReasoner: extract symbolic scene graph, apply rules, generate answer.
///
/// Represents scene understanding as a graph of objects and relations, then applies
/// a learned relation network to answer questions.
#[derive(Debug, Clone)]
pub struct SymbolicVisualReasoner {
    /// Feature dimension for scene nodes.
    pub node_dim: usize,
    /// Number of object categories.
    pub n_categories: usize,
    /// Number of relation types.
    pub n_relations: usize,
    /// Number of answer classes.
    pub n_answers: usize,
    /// Node classifier: [n_categories, node_dim].
    node_classifier: Vec<Vec<f64>>,
    /// Edge relation scorer: [n_relations, 2 * node_dim].
    edge_scorer: Vec<Vec<f64>>,
    /// Answer head: [n_answers, node_dim].
    answer_head: Vec<Vec<f64>>,
}

impl SymbolicVisualReasoner {
    /// Create a new symbolic visual reasoner.
    pub fn new(
        node_dim: usize,
        n_categories: usize,
        n_relations: usize,
        n_answers: usize,
        rng: &mut StdRng,
    ) -> Self {
        Self {
            node_dim,
            n_categories,
            n_relations,
            n_answers,
            node_classifier: rand_mat(n_categories, node_dim, rng),
            edge_scorer: rand_mat(n_relations, 2 * node_dim, rng),
            answer_head: rand_mat(n_answers, node_dim, rng),
        }
    }

    /// Classify object nodes into categories.
    pub fn classify_nodes(&self, node_features: &[Vec<f64>]) -> Vec<usize> {
        node_features
            .iter()
            .map(|f| {
                let scores = matvec(&self.node_classifier, f);
                scores
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Score edge relations between node pairs.
    pub fn score_edges(&self, src_feat: &[f64], dst_feat: &[f64]) -> Vec<f64> {
        let combined: Vec<f64> = src_feat.iter().chain(dst_feat.iter()).cloned().collect();
        matvec(&self.edge_scorer, &combined)
    }

    /// Generate answer logits by pooling node features and applying answer head.
    ///
    /// `node_features`: [n_nodes, node_dim] — visual features for each node.
    /// Returns `[n_answers]` answer logits.
    pub fn answer(&self, node_features: &[Vec<f64>]) -> Vec<f64> {
        if node_features.is_empty() {
            return vec![0.0; self.n_answers];
        }
        let n = node_features.len() as f64;
        let mut pooled = vec![0.0_f64; self.node_dim];
        for feat in node_features {
            for (p, &v) in pooled.iter_mut().zip(feat.iter()) {
                *p += v / n;
            }
        }
        matvec(&self.answer_head, &pooled)
    }
}

/// Metrics for multimodal reasoning evaluation.
#[derive(Debug, Clone)]
pub struct MmReasoningMetrics {
    /// Number of exact-match correct predictions.
    pub exact_matches: usize,
    /// Total number of predictions.
    pub total: usize,
    /// Sum of partial credit scores (0.0–1.0 per example).
    pub partial_credit_sum: f64,
    /// Number of program-execution-correct predictions.
    pub program_correct: usize,
}

impl MmReasoningMetrics {
    /// Create empty metrics.
    pub fn new() -> Self {
        Self {
            exact_matches: 0,
            total: 0,
            partial_credit_sum: 0.0,
            program_correct: 0,
        }
    }

    /// Update metrics for one prediction.
    ///
    /// `pred_idx`: predicted class, `gt_idx`: ground-truth class, `partial_credit`: 0.0–1.0.
    pub fn update(&mut self, pred_idx: usize, gt_idx: usize, partial_credit: f64) {
        self.total += 1;
        if pred_idx == gt_idx {
            self.exact_matches += 1;
            self.program_correct += 1;
        }
        self.partial_credit_sum += partial_credit.clamp(0.0, 1.0);
    }

    /// Exact-match accuracy.
    pub fn exact_match_accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.exact_matches as f64 / self.total as f64
        }
    }

    /// Average partial credit score.
    pub fn average_partial_credit(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.partial_credit_sum / self.total as f64
        }
    }

    /// Program execution accuracy.
    pub fn program_execution_accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.program_correct as f64 / self.total as f64
        }
    }
}

impl Default for MmReasoningMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(99)
    }

    fn rand_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
        (0..rows)
            .map(|_| (0..cols).map(|_| rng.random::<f64>() * 0.1 - 0.05).collect())
            .collect()
    }

    // ── UnifiedIOModel ──────────────────────────────────────────────────────

    #[test]
    fn test_unified_io_forward_shape() {
        let mut rng = make_rng();
        let model = UnifiedIOModel::new(8, 16, 32, &mut rng);
        let tokens = vec![
            UnifiedToken { modality: UnifiedModality::Text, embedding: vec![0.1; 8] },
            UnifiedToken { modality: UnifiedModality::ImagePatch, embedding: vec![0.2; 8] },
            UnifiedToken { modality: UnifiedModality::AudioFrame, embedding: vec![0.3; 8] },
        ];
        let out = model.forward(&tokens, 4);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].len(), 32);
        assert!(out.iter().all(|v| v.iter().all(|x| x.is_finite())));
    }

    #[test]
    fn test_unified_io_empty_input() {
        let mut rng = make_rng();
        let model = UnifiedIOModel::new(8, 16, 32, &mut rng);
        let out = model.forward(&[], 3);
        assert!(out.is_empty());
    }

    #[test]
    fn test_unified_io_zero_steps() {
        let mut rng = make_rng();
        let model = UnifiedIOModel::new(8, 16, 32, &mut rng);
        let tokens = vec![UnifiedToken {
            modality: UnifiedModality::Text,
            embedding: vec![0.1; 8],
        }];
        let out = model.forward(&tokens, 0);
        assert!(out.is_empty());
    }

    // ── PaLiModel ──────────────────────────────────────────────────────────

    #[test]
    fn test_pali_forward_shape() {
        let mut rng = make_rng();
        let model = PaLiModel::new(12, 16, 50, &mut rng);
        let patches: Vec<Vec<f64>> = (0..4).map(|_| vec![0.1; 12]).collect();
        let text_tokens: Vec<Vec<f64>> = (0..6).map(|_| vec![0.2; 16]).collect();
        let out = model.forward(&patches, &text_tokens);
        assert_eq!(out.len(), 6);
        assert_eq!(out[0].len(), 50);
        assert!(out.iter().all(|v| v.iter().all(|x| x.is_finite())));
    }

    #[test]
    fn test_pali_encode_image() {
        let mut rng = make_rng();
        let model = PaLiModel::new(8, 16, 50, &mut rng);
        let patches: Vec<Vec<f64>> = (0..3).map(|_| vec![0.1; 8]).collect();
        let enc = model.encode_image(&patches);
        assert_eq!(enc.len(), 3);
        assert_eq!(enc[0].len(), 16);
    }

    #[test]
    fn test_pali_empty_text() {
        let mut rng = make_rng();
        let model = PaLiModel::new(8, 16, 50, &mut rng);
        let patches: Vec<Vec<f64>> = (0..4).map(|_| vec![0.1; 8]).collect();
        let out = model.forward(&patches, &[]);
        assert!(out.is_empty());
    }

    // ── CogVlmLayer ────────────────────────────────────────────────────────

    #[test]
    fn test_cogvlm_forward_shape() {
        let mut rng = make_rng();
        let layer = CogVlmLayer::new(8, &mut rng);
        let tokens: Vec<Vec<f64>> = (0..6).map(|_| vec![0.1; 8]).collect();
        let out = layer.forward(&tokens, 3); // 3 visual, 3 text
        assert_eq!(out.len(), 6);
        assert_eq!(out[0].len(), 8);
        assert!(out.iter().all(|t| t.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn test_cogvlm_all_visual() {
        let mut rng = make_rng();
        let layer = CogVlmLayer::new(8, &mut rng);
        let tokens: Vec<Vec<f64>> = (0..4).map(|_| vec![0.2; 8]).collect();
        let out = layer.forward(&tokens, 4); // all visual
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_cogvlm_empty() {
        let mut rng = make_rng();
        let layer = CogVlmLayer::new(8, &mut rng);
        let out = layer.forward(&[], 0);
        assert!(out.is_empty());
    }

    // ── MultimodalAligner ──────────────────────────────────────────────────

    #[test]
    fn test_multimodal_aligner_project() {
        let mut rng = make_rng();
        let aligner = MultimodalAligner::new(&[8, 12, 16], 32, &mut rng);
        let feat = vec![0.1; 8];
        let emb = aligner.project(0, &feat);
        assert_eq!(emb.len(), 32);
        let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_multimodal_aligner_loss() {
        let mut rng = make_rng();
        let aligner = MultimodalAligner::new(&[8, 8], 16, &mut rng);
        let batch_a: Vec<Vec<f64>> = (0..4).map(|_| {
            l2_normalize(&rand_vec(16, &mut rng))
        }).collect();
        let batch_b: Vec<Vec<f64>> = (0..4).map(|_| {
            l2_normalize(&rand_vec(16, &mut rng))
        }).collect();
        let loss = aligner.pairwise_contrastive_loss(&[batch_a, batch_b]).expect("loss failed");
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn test_multimodal_aligner_single_modality_error() {
        let mut rng = make_rng();
        let aligner = MultimodalAligner::new(&[8], 16, &mut rng);
        let batch: Vec<Vec<f64>> = vec![vec![0.1; 16]];
        let result = aligner.pairwise_contrastive_loss(&[batch]);
        assert!(result.is_err());
    }

    // ── ModalityGapAnalyzer ────────────────────────────────────────────────

    #[test]
    fn test_modality_gap_identical_zero() {
        let analyzer = ModalityGapAnalyzer::new();
        let embs: Vec<Vec<f64>> = (0..5).map(|_| vec![1.0, 0.0, 0.0]).collect();
        let gap = analyzer.gap(&embs, &embs).expect("gap failed");
        assert!(gap.abs() < 1e-10);
    }

    #[test]
    fn test_modality_gap_orthogonal() {
        let analyzer = ModalityGapAnalyzer::new();
        let embs_a: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let embs_b: Vec<Vec<f64>> = vec![vec![0.0, 1.0], vec![0.0, 1.0]];
        let gap = analyzer.gap(&embs_a, &embs_b).expect("gap failed");
        assert!((gap - 2.0_f64.sqrt()).abs() < 1e-8);
    }

    #[test]
    fn test_modality_gap_cosine_similarity() {
        let analyzer = ModalityGapAnalyzer::new();
        let embs: Vec<Vec<f64>> = vec![vec![1.0, 0.0]];
        let sim = analyzer.centroid_cosine_similarity(&embs, &embs).expect("sim failed");
        assert!((sim - 1.0).abs() < 1e-9);
    }

    // ── UnifiedEmbeddingSpace ──────────────────────────────────────────────

    #[test]
    fn test_unified_embedding_embed() {
        let mut rng = make_rng();
        let space = UnifiedEmbeddingSpace::new(&[8, 12], 16, 0.1, &mut rng);
        let feat = vec![0.1; 8];
        let emb = space.embed(0, &feat);
        assert_eq!(emb.len(), 16);
        let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_unified_embedding_ortho_loss_nonnegative() {
        let mut rng = make_rng();
        let space = UnifiedEmbeddingSpace::new(&[8, 12, 16], 16, 0.01, &mut rng);
        let loss = space.orthogonality_loss();
        assert!(loss >= 0.0 && loss.is_finite());
    }

    // ── PhrasalGrounder ────────────────────────────────────────────────────

    #[test]
    fn test_phrasal_grounder_shape() {
        let mut rng = make_rng();
        let grounder = PhrasalGrounder::new(8, 12, 16, &mut rng);
        let phrases: Vec<Vec<f64>> = (0..3).map(|_| vec![0.1; 8]).collect();
        let regions: Vec<Vec<f64>> = (0..5).map(|_| vec![0.2; 12]).collect();
        let scores = grounder.ground(&phrases, &regions);
        assert_eq!(scores.len(), 3);
        assert_eq!(scores[0].len(), 5);
        assert!(scores.iter().all(|row| row.iter().all(|&s| (0.0..=1.0).contains(&s))));
    }

    #[test]
    fn test_phrasal_grounder_empty_input() {
        let mut rng = make_rng();
        let grounder = PhrasalGrounder::new(8, 12, 16, &mut rng);
        let out = grounder.ground(&[], &[vec![0.1; 12]]);
        assert!(out.is_empty());
    }

    // ── RecRefDecoder ──────────────────────────────────────────────────────

    #[test]
    fn test_rec_decoder_box_range() {
        let mut rng = make_rng();
        let decoder = RecRefDecoder::new(16, 32, &mut rng);
        let repr = vec![0.1; 16];
        let bbox = decoder.predict_box(&repr);
        for &coord in &bbox {
            assert!((0.0..=1.0).contains(&coord), "coord out of [0,1]: {coord}");
        }
    }

    #[test]
    fn test_rec_decoder_finite() {
        let mut rng = make_rng();
        let decoder = RecRefDecoder::new(8, 16, &mut rng);
        let repr = vec![1.0, -1.0, 0.5, -0.5, 0.0, 2.0, -2.0, 0.1];
        let bbox = decoder.predict_box(&repr);
        for &coord in &bbox {
            assert!(coord.is_finite());
        }
    }

    // ── GroundingMetrics ───────────────────────────────────────────────────

    #[test]
    fn test_iou_identical_boxes() {
        let box1 = [0.5, 0.5, 0.4, 0.4];
        let iou = GroundingMetrics::iou_cxcywh(&box1, &box1);
        assert!((iou - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_iou_non_overlapping() {
        let box1 = [0.1, 0.1, 0.1, 0.1];
        let box2 = [0.9, 0.9, 0.1, 0.1];
        let iou = GroundingMetrics::iou_cxcywh(&box1, &box2);
        assert_eq!(iou, 0.0);
    }

    #[test]
    fn test_accuracy_at_iou_perfect() {
        let metrics = GroundingMetrics::new(0.5);
        let preds = vec![[0.5_f64, 0.5, 0.4, 0.4]];
        let gts = vec![[0.5_f64, 0.5, 0.4, 0.4]];
        let acc = metrics.accuracy_at_iou(&preds, &gts);
        assert!((acc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_recall_at_k() {
        let metrics = GroundingMetrics::new(0.5);
        // 2 phrases, 3 regions: ground truth is diagonal (phrase i → region i)
        let scores = vec![
            vec![0.9, 0.1, 0.2], // phrase 0 → top-1 is region 0 ✓
            vec![0.1, 0.8, 0.3], // phrase 1 → top-1 is region 1 ✓
        ];
        let recall = metrics.recall_at_k(&scores, 1);
        assert!((recall - 1.0).abs() < 1e-10);
    }

    // ── ChainOfThoughtMultimodal ───────────────────────────────────────────

    #[test]
    fn test_cot_multimodal_shape() {
        let mut rng = make_rng();
        let cot = ChainOfThoughtMultimodal::new(8, 16, 50, 4, &mut rng);
        let vis_tokens: Vec<Vec<f64>> = (0..6).map(|_| vec![0.1; 8]).collect();
        let out = cot.generate(&vis_tokens);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].len(), 50);
        assert!(out.iter().all(|v| v.iter().all(|x| x.is_finite())));
    }

    #[test]
    fn test_cot_multimodal_empty_visual() {
        let mut rng = make_rng();
        let cot = ChainOfThoughtMultimodal::new(8, 16, 50, 3, &mut rng);
        let out = cot.generate(&[]);
        // Each step returns zero logits when no visual tokens
        assert_eq!(out.len(), 3);
        assert!(out[0].iter().all(|&x| x == 0.0));
    }

    // ── SymbolicVisualReasoner ─────────────────────────────────────────────

    #[test]
    fn test_symbolic_reasoner_classify_nodes() {
        let mut rng = make_rng();
        let reasoner = SymbolicVisualReasoner::new(8, 5, 3, 10, &mut rng);
        let feats: Vec<Vec<f64>> = (0..4).map(|_| vec![0.1; 8]).collect();
        let cats = reasoner.classify_nodes(&feats);
        assert_eq!(cats.len(), 4);
        assert!(cats.iter().all(|&c| c < 5));
    }

    #[test]
    fn test_symbolic_reasoner_score_edges() {
        let mut rng = make_rng();
        let reasoner = SymbolicVisualReasoner::new(8, 5, 4, 10, &mut rng);
        let src = vec![0.1; 8];
        let dst = vec![0.2; 8];
        let scores = reasoner.score_edges(&src, &dst);
        assert_eq!(scores.len(), 4);
        assert!(scores.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn test_symbolic_reasoner_answer_shape() {
        let mut rng = make_rng();
        let reasoner = SymbolicVisualReasoner::new(8, 5, 3, 10, &mut rng);
        let feats: Vec<Vec<f64>> = (0..5).map(|_| vec![0.1; 8]).collect();
        let logits = reasoner.answer(&feats);
        assert_eq!(logits.len(), 10);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_symbolic_reasoner_empty_nodes() {
        let mut rng = make_rng();
        let reasoner = SymbolicVisualReasoner::new(8, 5, 3, 10, &mut rng);
        let logits = reasoner.answer(&[]);
        assert_eq!(logits.len(), 10);
        assert!(logits.iter().all(|&v| v == 0.0));
    }

    // ── MmReasoningMetrics ─────────────────────────────────────────────────

    #[test]
    fn test_mm_reasoning_metrics_exact_match() {
        let mut m = MmReasoningMetrics::new();
        m.update(2, 2, 1.0);
        m.update(1, 2, 0.5);
        m.update(3, 3, 1.0);
        assert!((m.exact_match_accuracy() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mm_reasoning_metrics_partial_credit() {
        let mut m = MmReasoningMetrics::new();
        m.update(0, 1, 0.6);
        m.update(1, 1, 1.0);
        let avg = m.average_partial_credit();
        assert!((avg - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_mm_reasoning_metrics_empty() {
        let m = MmReasoningMetrics::new();
        assert_eq!(m.exact_match_accuracy(), 0.0);
        assert_eq!(m.average_partial_credit(), 0.0);
        assert_eq!(m.program_execution_accuracy(), 0.0);
    }

    #[test]
    fn test_mm_reasoning_metrics_all_correct() {
        let mut m = MmReasoningMetrics::new();
        for i in 0..5 {
            m.update(i, i, 1.0);
        }
        assert!((m.exact_match_accuracy() - 1.0).abs() < 1e-10);
        assert!((m.program_execution_accuracy() - 1.0).abs() < 1e-10);
    }
}
