//! Extensions to multimodal foundation models: document table/QA and video understanding.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::TensorError;

use super::{
    dot, gelu, l2_normalize, layer_norm, matvec, rand_mat, sdp_attention, sinusoidal_pe, softmax,
    LayoutAwareAttention,
};

// ── Document Understanding (continued) ──────────────────────────────────────

/// Detect row/column structure from a feature grid.
#[derive(Debug, Clone)]
pub struct TableParser {
    /// Number of rows in the table.
    pub n_rows: usize,
    /// Number of columns in the table.
    pub n_cols: usize,
}

impl TableParser {
    /// Create a new `TableParser` with the given row/column defaults.
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        Self { n_rows, n_cols }
    }

    /// Parse a feature grid (flattened or 2D) into a [n_rows, n_cols] cell representation.
    /// Each cell is the average-pooled feature from the corresponding grid region.
    /// `features`: [n_regions, d] → `[n_rows * n_cols, d]`.
    pub fn parse(&self, features: &[Vec<f64>], n_rows: usize, n_cols: usize) -> Vec<Vec<f64>> {
        let n_rows = if n_rows == 0 { self.n_rows } else { n_rows };
        let n_cols = if n_cols == 0 { self.n_cols } else { n_cols };
        let n_cells = n_rows * n_cols;
        if features.is_empty() {
            return vec![Vec::new(); n_cells];
        }
        let feat_dim = features[0].len();
        let n_src = features.len();
        (0..n_cells)
            .map(|cell| {
                let start = cell * n_src / n_cells;
                let end = ((cell + 1) * n_src / n_cells).max(start + 1).min(n_src);
                let count = (end - start) as f64;
                let mut avg = vec![0.0_f64; feat_dim];
                for feat in &features[start..end] {
                    for (a, &f) in avg.iter_mut().zip(feat.iter()) {
                        *a += f / count;
                    }
                }
                avg
            })
            .collect()
    }
}

/// Predict reading order: score each text region and sort by score.
#[derive(Debug, Clone)]
pub struct ReadingOrderPrediction {
    score_proj: Vec<Vec<f64>>, // [1, d_model] → produces scalar
}

impl ReadingOrderPrediction {
    /// Create a new reading-order predictor.
    pub fn new(d_model: usize, rng: &mut StdRng) -> Self {
        Self {
            score_proj: rand_mat(1, d_model, rng),
        }
    }

    /// Score each region feature and return indices sorted by ascending score.
    pub fn predict_order(&self, region_features: &[Vec<f64>]) -> Vec<usize> {
        let scores: Vec<f64> = region_features
            .iter()
            .map(|f| matvec(&self.score_proj, f)[0])
            .collect();
        let mut indices: Vec<usize> = (0..region_features.len()).collect();
        indices.sort_by(|&a, &b| {
            scores[a]
                .partial_cmp(&scores[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }
}

/// Document Q&A model: LayoutAwareAttention + extraction head.
#[derive(Debug, Clone)]
pub struct DocumentQaModel {
    /// Model dimension.
    pub d_model: usize,
    attn: LayoutAwareAttention,
    /// Start/end span projection: [2, d_model] (row 0 = start, row 1 = end logits).
    span_head: Vec<Vec<f64>>,
}

impl DocumentQaModel {
    /// Create a new document QA model.
    pub fn new(d_model: usize, rng: &mut StdRng) -> Self {
        Self {
            d_model,
            attn: LayoutAwareAttention::new(d_model, rng),
            span_head: rand_mat(2, d_model, rng),
        }
    }

    /// Given document tokens + bboxes, return start/end logits over token positions.
    /// Returns (start_logits: `Vec<f64>`, end_logits: `Vec<f64>`), each len = n_tokens.
    pub fn predict_span(&self, tokens: &[Vec<f64>], bboxes: &[[f64; 4]]) -> (Vec<f64>, Vec<f64>) {
        let attended = self.attn.forward(tokens, bboxes);
        let start_logits: Vec<f64> = attended
            .iter()
            .map(|t| dot(&self.span_head[0], t))
            .collect();
        let end_logits: Vec<f64> = attended
            .iter()
            .map(|t| dot(&self.span_head[1], t))
            .collect();
        (start_logits, end_logits)
    }
}

// ── 5. Video Understanding ─────────────────────────────────────────────────

/// 3D sinusoidal position encoding for video tokens (temporal + spatial).
#[derive(Debug, Clone)]
pub struct TemporalPositionEmbedding {
    /// Model dimension.
    pub d_model: usize,
    /// Maximum temporal index.
    pub max_t: usize,
    /// Maximum height index.
    pub max_h: usize,
    /// Maximum width index.
    pub max_w: usize,
}

impl TemporalPositionEmbedding {
    /// Create a new 3D positional embedding layer.
    pub fn new(d_model: usize, max_t: usize, max_h: usize, max_w: usize) -> Self {
        Self {
            d_model,
            max_t,
            max_h,
            max_w,
        }
    }

    /// Compute 3D positional encoding for token at (t, h, w).
    /// d_model must be divisible by 3 for clean splits; remainder goes to temporal.
    pub fn encode(&self, t: usize, h: usize, w: usize) -> Vec<f64> {
        let dt = (self.d_model / 3).max(1);
        let ds = (self.d_model - dt) / 2;
        let ds2 = self.d_model - dt - ds;
        let pe_t = sinusoidal_pe(t, dt);
        let pe_h = sinusoidal_pe(h, ds);
        let pe_w = sinusoidal_pe(w, ds2);
        pe_t.into_iter().chain(pe_h).chain(pe_w).collect()
    }

    /// Encode a sequence of (t, h, w) positions: [n_tokens, d_model].
    pub fn encode_sequence(&self, positions: &[(usize, usize, usize)]) -> Vec<Vec<f64>> {
        positions
            .iter()
            .map(|&(t, h, w)| self.encode(t, h, w))
            .collect()
    }
}

/// TimeSFormer block: divided space-time attention (temporal then spatial).
#[derive(Debug, Clone)]
pub struct TimesFormerBlock {
    /// Model dimension.
    pub d_model: usize,
    /// Number of video frames.
    pub n_frames: usize,
    /// Number of spatial patches per frame.
    pub n_patches: usize,
    // Temporal attention weights
    t_wq: Vec<Vec<f64>>,
    t_wk: Vec<Vec<f64>>,
    t_wv: Vec<Vec<f64>>,
    t_wo: Vec<Vec<f64>>,
    // Spatial attention weights
    s_wq: Vec<Vec<f64>>,
    s_wk: Vec<Vec<f64>>,
    s_wv: Vec<Vec<f64>>,
    s_wo: Vec<Vec<f64>>,
    ff_w1: Vec<Vec<f64>>,
    ff_w2: Vec<Vec<f64>>,
}

impl TimesFormerBlock {
    /// Create a new TimeSFormer block.
    pub fn new(d_model: usize, n_frames: usize, n_patches: usize, rng: &mut StdRng) -> Self {
        Self {
            d_model,
            n_frames,
            n_patches,
            t_wq: rand_mat(d_model, d_model, rng),
            t_wk: rand_mat(d_model, d_model, rng),
            t_wv: rand_mat(d_model, d_model, rng),
            t_wo: rand_mat(d_model, d_model, rng),
            s_wq: rand_mat(d_model, d_model, rng),
            s_wk: rand_mat(d_model, d_model, rng),
            s_wv: rand_mat(d_model, d_model, rng),
            s_wo: rand_mat(d_model, d_model, rng),
            ff_w1: rand_mat(d_model * 4, d_model, rng),
            ff_w2: rand_mat(d_model, d_model * 4, rng),
        }
    }

    /// Forward: [n_frames * n_patches, d_model] — temporal then spatial attention.
    pub fn forward(&self, tokens: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if tokens.is_empty() {
            return Vec::new();
        }
        let total = tokens.len();
        let nf = self.n_frames.max(1);
        let np = (total / nf).max(1);
        // Temporal: each patch attends across frames
        let mut t_out = tokens.to_vec();
        for p in 0..np {
            let ft: Vec<Vec<f64>> = (0..nf)
                .filter_map(|f| tokens.get(f * np + p))
                .cloned()
                .collect();
            if ft.is_empty() {
                continue;
            }
            let q: Vec<Vec<f64>> = ft.iter().map(|x| matvec(&self.t_wq, x)).collect();
            let k: Vec<Vec<f64>> = ft.iter().map(|x| matvec(&self.t_wk, x)).collect();
            let v: Vec<Vec<f64>> = ft.iter().map(|x| matvec(&self.t_wv, x)).collect();
            for (fi, a) in sdp_attention(&q, &k, &v).iter().enumerate() {
                let idx = fi * np + p;
                if idx < total {
                    let proj = matvec(&self.t_wo, a);
                    let res: Vec<f64> = tokens[idx]
                        .iter()
                        .zip(proj.iter())
                        .map(|(x, y)| x + y)
                        .collect();
                    t_out[idx] = layer_norm(&res);
                }
            }
        }
        // Spatial: each frame attends across patches
        let mut s_out = t_out.clone();
        for f in 0..nf {
            let (st, en) = (f * np, (f * np + np).min(total));
            let ft = &t_out[st..en];
            if ft.is_empty() {
                continue;
            }
            let q: Vec<Vec<f64>> = ft.iter().map(|x| matvec(&self.s_wq, x)).collect();
            let k: Vec<Vec<f64>> = ft.iter().map(|x| matvec(&self.s_wk, x)).collect();
            let v: Vec<Vec<f64>> = ft.iter().map(|x| matvec(&self.s_wv, x)).collect();
            for (pi, a) in sdp_attention(&q, &k, &v).iter().enumerate() {
                let idx = st + pi;
                if idx < total {
                    let proj = matvec(&self.s_wo, a);
                    let res: Vec<f64> = t_out[idx]
                        .iter()
                        .zip(proj.iter())
                        .map(|(x, y)| x + y)
                        .collect();
                    s_out[idx] = layer_norm(&res);
                }
            }
        }
        // FFN
        s_out
            .iter()
            .map(|t| {
                let h: Vec<f64> = matvec(&self.ff_w1, t).iter().map(|v| gelu(*v)).collect();
                let ff = matvec(&self.ff_w2, &h);
                layer_norm(
                    &t.iter()
                        .zip(ff.iter())
                        .map(|(a, b)| a + b)
                        .collect::<Vec<_>>(),
                )
            })
            .collect()
    }
}

/// Slow-Fast dual-pathway video model.
/// Slow path: low fps (full spatial resolution), Fast path: high fps (small spatial).
#[derive(Debug, Clone)]
pub struct VideoSlowFast {
    /// Slow-path model dimension.
    pub slow_d_model: usize,
    /// Fast-path model dimension.
    pub fast_d_model: usize,
    slow_encoder: TimesFormerBlock,
    fast_encoder: TimesFormerBlock,
    /// Lateral connection: project fast features to slow space.
    lateral_proj: Vec<Vec<f64>>,
    /// Final fusion head.
    fusion_head: Vec<Vec<f64>>,
}

impl VideoSlowFast {
    /// Create a new Slow-Fast video model.
    pub fn new(
        slow_d_model: usize,
        fast_d_model: usize,
        slow_frames: usize,
        fast_frames: usize,
        n_patches: usize,
        out_dim: usize,
        rng: &mut StdRng,
    ) -> Self {
        Self {
            slow_d_model,
            fast_d_model,
            slow_encoder: TimesFormerBlock::new(slow_d_model, slow_frames, n_patches, rng),
            fast_encoder: TimesFormerBlock::new(fast_d_model, fast_frames, n_patches, rng),
            lateral_proj: rand_mat(slow_d_model, fast_d_model, rng),
            fusion_head: rand_mat(out_dim, slow_d_model * 2, rng),
        }
    }

    /// Encode slow and fast pathways and fuse via lateral connections.
    pub fn forward(&self, slow_tokens: &[Vec<f64>], fast_tokens: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let slow_out = self.slow_encoder.forward(slow_tokens);
        let fast_out = self.fast_encoder.forward(fast_tokens);
        let ns = slow_out.len().max(1);
        let nf = fast_out.len().max(1);
        slow_out
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let fi = i * nf / ns;
                let fp = if fi < fast_out.len() {
                    matvec(&self.lateral_proj, &fast_out[fi])
                } else {
                    vec![0.0; self.slow_d_model]
                };
                matvec(
                    &self.fusion_head,
                    &s.iter().chain(fp.iter()).cloned().collect::<Vec<_>>(),
                )
            })
            .collect()
    }
}

/// Align video clip representations with text descriptions (contrastive).
#[derive(Debug, Clone)]
pub struct VideoTextAlignment {
    /// Video feature dimension.
    pub d_video: usize,
    /// Text feature dimension.
    pub d_text: usize,
    /// Common embedding dimension.
    pub d_common: usize,
    video_proj: Vec<Vec<f64>>,
    text_proj: Vec<Vec<f64>>,
    /// InfoNCE temperature.
    pub temperature: f64,
}

impl VideoTextAlignment {
    /// Create a new video-text alignment model.
    pub fn new(d_video: usize, d_text: usize, d_common: usize, rng: &mut StdRng) -> Self {
        Self {
            d_video,
            d_text,
            d_common,
            video_proj: rand_mat(d_common, d_video, rng),
            text_proj: rand_mat(d_common, d_text, rng),
            temperature: 0.07,
        }
    }

    /// Encode video features to the shared embedding space (L2-normalized).
    pub fn encode_video(&self, video_feats: &[f64]) -> Vec<f64> {
        let proj = matvec(&self.video_proj, video_feats);
        l2_normalize(&proj)
    }

    /// Encode text features to the shared embedding space (L2-normalized).
    pub fn encode_text(&self, text_feats: &[f64]) -> Vec<f64> {
        let proj = matvec(&self.text_proj, text_feats);
        l2_normalize(&proj)
    }

    /// Symmetric InfoNCE loss for (video, text) pairs.
    pub fn alignment_loss(
        &self,
        video_embeds: &[Vec<f64>],
        text_embeds: &[Vec<f64>],
    ) -> Result<f64, TensorError> {
        let n = video_embeds.len();
        if n == 0 || n != text_embeds.len() {
            return Err(TensorError::invalid_argument_op(
                "VideoTextAlignment",
                "batch sizes must match and be non-zero",
            ));
        }
        let t = self.temperature.max(1e-8);
        let mut sim = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            let ve = l2_normalize(&video_embeds[i]);
            for j in 0..n {
                let te = l2_normalize(&text_embeds[j]);
                sim[i][j] = dot(&ve, &te) / t;
            }
        }
        let mut loss = 0.0_f64;
        for i in 0..n {
            let sm_v = softmax(&sim[i]);
            loss -= sm_v[i].max(1e-12).ln();
            let col: Vec<f64> = (0..n).map(|j| sim[j][i]).collect();
            let sm_t = softmax(&col);
            loss -= sm_t[i].max(1e-12).ln();
        }
        Ok(loss / (2.0 * n as f64))
    }
}

/// Video Q&A model: video encoder + mean-pool + classification head.
#[derive(Debug, Clone)]
pub struct VideoQaModel {
    /// Model dimension.
    pub d_model: usize,
    /// Number of answer classes.
    pub n_answers: usize,
    tsf_block: TimesFormerBlock,
    qa_head: Vec<Vec<f64>>,
}

impl VideoQaModel {
    /// Create a new video Q&A model.
    pub fn new(
        d_model: usize,
        n_frames: usize,
        n_patches: usize,
        n_answers: usize,
        rng: &mut StdRng,
    ) -> Self {
        Self {
            d_model,
            n_answers,
            tsf_block: TimesFormerBlock::new(d_model, n_frames, n_patches, rng),
            qa_head: rand_mat(n_answers, d_model, rng),
        }
    }

    /// Forward pass: encode video tokens and classify into answer space.
    pub fn forward(&self, tokens: &[Vec<f64>]) -> Vec<f64> {
        if tokens.is_empty() {
            return vec![0.0; self.n_answers];
        }
        let encoded = self.tsf_block.forward(tokens);
        let n = encoded.len() as f64;
        let d = encoded[0].len();
        let mut pooled = vec![0.0_f64; d];
        for tok in &encoded {
            for (p, &v) in pooled.iter_mut().zip(tok.iter()) {
                *p += v / n;
            }
        }
        matvec(&self.qa_head, &pooled)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multimodal_foundation::{
        AvContrastiveLoss, AvEncoder, AudioSpectrogram, AudioVisualAttention, DocumentTokenizer,
        GatedCrossAttention, ImageInstructionFormatter, LanguageProjector, LayoutAwareAttention,
        LlavaLoss, LlavaModel, MlpProjector, PerceiverResampler, SpeechVisualGrounding,
        VisualLanguageAligner, VisualTokenizer, VisionEncoder,
    };

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    fn rand_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
        (0..rows)
            .map(|_| {
                (0..cols)
                    .map(|_| rng.random::<f64>() * 0.1 - 0.05)
                    .collect()
            })
            .collect()
    }

    // ─── VL Alignment Tests ───────────────────────────────────────────────

    #[test]
    fn test_visual_tokenizer_count() {
        let vt = VisualTokenizer::new(8);
        let feats: Vec<Vec<f64>> = (0..20).map(|_| vec![1.0; 16]).collect();
        let tokens = vt.tokenize(&feats, 8);
        assert_eq!(tokens.len(), 8);
        assert_eq!(tokens[0].len(), 16);
    }

    #[test]
    fn test_visual_tokenizer_padding() {
        let vt = VisualTokenizer::new(16);
        let feats: Vec<Vec<f64>> = (0..4).map(|_| vec![1.0; 8]).collect();
        let tokens = vt.tokenize(&feats, 16);
        assert_eq!(tokens.len(), 16);
    }

    #[test]
    fn test_visual_tokenizer_identity() {
        let vt = VisualTokenizer::new(5);
        let feats: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64; 4]).collect();
        let tokens = vt.tokenize(&feats, 5);
        assert_eq!(tokens.len(), 5);
        // Identity case: input == output
        for (orig, tok) in feats.iter().zip(tokens.iter()) {
            for (a, b) in orig.iter().zip(tok.iter()) {
                assert!((a - b).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_language_projector() {
        let mut rng = make_rng();
        let proj = LanguageProjector::new(32, 16, &mut rng);
        let input = vec![0.5_f64; 32];
        let out = proj.project(&input);
        assert_eq!(out.len(), 16);
        // GELU output should be finite
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_vl_aligner_construction() {
        let aligner = VisualLanguageAligner::new(64, true);
        assert_eq!(aligner.embed_dim, 64);
        assert!(aligner.use_captioning);
    }

    #[test]
    fn test_vl_aligner_contrastive_loss() {
        let aligner = VisualLanguageAligner::new(8, false);
        let mut rng = make_rng();
        let imgs = rand_matrix(4, 8, &mut rng);
        let txts = rand_matrix(4, 8, &mut rng);
        let loss = aligner.contrastive_loss(&imgs, &txts).expect("contrastive_loss failed");
        assert!(loss.is_finite());
        assert!(loss > 0.0);
    }

    #[test]
    fn test_vl_aligner_captioning_loss() {
        let aligner = VisualLanguageAligner::new(8, true);
        let logits = vec![vec![1.0, 2.0, 0.5], vec![0.1, 0.3, 2.5]];
        let targets = vec![1usize, 2usize];
        let loss = aligner.captioning_loss(&logits, &targets).expect("captioning_loss failed");
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn test_gated_cross_attention_tanh() {
        let mut rng = make_rng();
        let gca = GatedCrossAttention::new(8, &mut rng);
        // alpha = 0 → gate = tanh(0) = 0 → output should equal input
        let x_lang: Vec<Vec<f64>> = (0..3).map(|_| vec![1.0; 8]).collect();
        let x_vis: Vec<Vec<f64>> = (0..5).map(|_| vec![0.5; 8]).collect();
        let out = gca.forward(&x_lang, &x_vis);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].len(), 8);
        // With alpha=0 gate=0, output should equal input exactly
        for (o, x) in out.iter().zip(x_lang.iter()) {
            for (a, b) in o.iter().zip(x.iter()) {
                assert!((a - b).abs() < 1e-12, "gate should be zero at init");
            }
        }
    }

    #[test]
    fn test_gated_cross_attention_nonzero_alpha() {
        let mut rng = make_rng();
        let mut gca = GatedCrossAttention::new(8, &mut rng);
        gca.alpha = 1.0; // gate = tanh(1.0) ≈ 0.76
        let x_lang: Vec<Vec<f64>> = (0..3).map(|_| vec![0.1; 8]).collect();
        let x_vis: Vec<Vec<f64>> = (0..4).map(|_| vec![0.2; 8]).collect();
        let out = gca.forward(&x_lang, &x_vis);
        assert_eq!(out.len(), 3);
        // Output should differ from input when gate is non-zero
        let differs = out[0]
            .iter()
            .zip(x_lang[0].iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(differs);
    }

    #[test]
    fn test_perceiver_resampler_shape() {
        let mut rng = make_rng();
        let pr = PerceiverResampler::new(16, 32, &mut rng);
        let vis_feats: Vec<Vec<f64>> = (0..100).map(|_| vec![0.1; 32]).collect();
        let latents = pr.resample(&vis_feats);
        assert_eq!(latents.len(), 16);
        assert_eq!(latents[0].len(), 32);
    }

    #[test]
    fn test_perceiver_resampler_empty_input() {
        let mut rng = make_rng();
        let pr = PerceiverResampler::new(8, 16, &mut rng);
        let latents = pr.resample(&[]);
        // Should return learned queries unchanged
        assert_eq!(latents.len(), 8);
    }

    // ─── LLaVA Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_vision_encoder_features() {
        let mut rng = make_rng();
        let ve = VisionEncoder::new(4, 16, 2, &mut rng);
        let patch_dim = 4 * 4 * 3;
        let patches: Vec<Vec<f64>> = (0..9).map(|_| vec![0.1; patch_dim]).collect();
        let out = ve.encode(&patches);
        // CLS + 9 patches = 10 tokens
        assert_eq!(out.len(), 10);
        assert_eq!(out[0].len(), 16);
    }

    #[test]
    fn test_vision_encoder_cls_token() {
        let mut rng = make_rng();
        let ve = VisionEncoder::new(2, 8, 1, &mut rng);
        let patch_dim = 2 * 2 * 3;
        let patches: Vec<Vec<f64>> = (0..4).map(|_| vec![0.0; patch_dim]).collect();
        let out = ve.encode(&patches);
        assert_eq!(out.len(), 5); // CLS + 4 patches
    }

    #[test]
    fn test_mlp_projector_shape() {
        let mut rng = make_rng();
        let proj = MlpProjector::new(16, 32, 8, &mut rng);
        let x = vec![0.1_f64; 16];
        let out = proj.project(&x);
        assert_eq!(out.len(), 8);
        assert_eq!(proj.out_dim, 8);
    }

    #[test]
    fn test_mlp_projector_gelu_activation() {
        let mut rng = make_rng();
        let proj = MlpProjector::new(4, 8, 4, &mut rng);
        let x = vec![1.0_f64; 4];
        let out = proj.project(&x);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_llava_construction() {
        let mut rng = make_rng();
        let model = LlavaModel::new(4, 16, 2, 32, 8, 100, &mut rng);
        assert_eq!(model.vocab_size, 100);
    }

    #[test]
    fn test_llava_forward_shape() {
        let mut rng = make_rng();
        let model = LlavaModel::new(4, 16, 2, 32, 8, 50, &mut rng);
        let patch_dim = 4 * 4 * 3;
        let patches: Vec<Vec<f64>> = (0..4).map(|_| vec![0.1; patch_dim]).collect();
        let tokens = vec![1usize, 2, 3, 4];
        let out = model.forward(&patches, &tokens);
        // 1 CLS + 4 patches = 5 positions
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].len(), 50);
    }

    #[test]
    fn test_image_instruction_format() {
        let fmt = ImageInstructionFormatter::new("You are a helpful assistant.");
        let result = fmt.format("What is in this image?");
        assert!(result.contains("<image>"));
        assert!(result.contains("[SYSTEM]"));
        assert!(result.contains("[USER]"));
        assert!(result.contains("What is in this image?"));
    }

    #[test]
    fn test_image_instruction_positions() {
        let fmt = ImageInstructionFormatter::new("sys");
        let (segs, is_image) = fmt.tokenize_with_positions("describe this");
        assert_eq!(segs.len(), is_image.len());
        let has_image_token = is_image.iter().any(|&b| b);
        assert!(has_image_token);
    }

    #[test]
    fn test_llava_loss_text_only() {
        let loss_fn = LlavaLoss::new();
        let logits = vec![
            vec![1.0, 2.0, 0.1],
            vec![0.5, 1.5, 2.0],
            vec![2.0, 0.1, 0.5],
        ];
        let targets = vec![1usize, 2, 0];
        let is_image = vec![true, false, false];
        let loss = loss_fn.compute(&logits, &targets, &is_image).expect("compute failed");
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn test_llava_loss_all_image() {
        let loss_fn = LlavaLoss::new();
        let logits = vec![vec![1.0, 2.0]; 3];
        let targets = vec![0usize, 1, 0];
        let is_image = vec![true, true, true];
        let loss = loss_fn.compute(&logits, &targets, &is_image).expect("compute failed");
        assert_eq!(loss, 0.0);
    }

    // ─── Audio-Visual Tests ───────────────────────────────────────────────

    #[test]
    fn test_audio_spectrogram_shape() {
        let spec = AudioSpectrogram::new(8, 4);
        let waveform: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let frames = spec.extract(&waveform, 8, 4);
        assert!(!frames.is_empty());
        let n_bins = 8 / 2 + 1;
        assert_eq!(frames[0].len(), n_bins);
    }

    #[test]
    fn test_audio_spectrogram_n_frames() {
        let spec = AudioSpectrogram::new(16, 8);
        let waveform: Vec<f64> = vec![0.0; 64];
        let frames = spec.extract(&waveform, 16, 8);
        // (64 - 16) / 8 + 1 = 7 frames
        assert_eq!(frames.len(), 7);
    }

    #[test]
    fn test_av_attention_shape() {
        let mut rng = make_rng();
        let ava = AudioVisualAttention::new(16, &mut rng);
        let audio: Vec<Vec<f64>> = (0..5).map(|_| vec![0.1; 16]).collect();
        let video: Vec<Vec<f64>> = (0..8).map(|_| vec![0.2; 16]).collect();
        let out = ava.attend(&audio, &video);
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].len(), 16);
    }

    #[test]
    fn test_av_encoder() {
        let mut rng = make_rng();
        let enc = AvEncoder::new(8, 12, 16, 5, 7, &mut rng);
        let audio: Vec<Vec<f64>> = (0..5).map(|_| vec![0.1; 8]).collect();
        let video: Vec<Vec<f64>> = (0..7).map(|_| vec![0.2; 12]).collect();
        let out = enc.encode(&audio, &video);
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].len(), 16);
    }

    #[test]
    fn test_av_contrastive_loss_positive() {
        let loss_fn = AvContrastiveLoss::new(0.07);
        let mut rng = make_rng();
        let embeds_a = rand_matrix(4, 16, &mut rng);
        let embeds_v = rand_matrix(4, 16, &mut rng);
        let loss = loss_fn.compute(&embeds_a, &embeds_v).expect("compute failed");
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn test_av_contrastive_loss_identical() {
        let loss_fn = AvContrastiveLoss::new(0.07);
        // When audio == video, diagonal should be maximum → lower loss
        let embeds: Vec<Vec<f64>> = (0..4)
            .map(|i| {
                let mut v = vec![0.0_f64; 8];
                v[i % 8] = 1.0;
                v
            })
            .collect();
        let loss = loss_fn.compute(&embeds, &embeds.clone()).expect("compute failed");
        assert!(loss.is_finite());
    }

    #[test]
    fn test_speech_visual_grounding() {
        let mut rng = make_rng();
        let grounding = SpeechVisualGrounding::new(8, 12, 16, &mut rng);
        let audio_words: Vec<Vec<f64>> = (0..3).map(|_| vec![0.1; 8]).collect();
        let visual_regions: Vec<Vec<f64>> = (0..5).map(|_| vec![0.2; 12]).collect();
        let attn = grounding.ground(&audio_words, &visual_regions);
        assert_eq!(attn.len(), 3);
        assert_eq!(attn[0].len(), 5);
        // Each attention row should sum to ~1.0 (softmax)
        for row in &attn {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    // ─── Document Understanding Tests ────────────────────────────────────

    #[test]
    fn test_layout_aware_attention() {
        let mut rng = make_rng();
        let laa = LayoutAwareAttention::new(16, &mut rng);
        let tokens: Vec<Vec<f64>> = (0..5).map(|_| vec![0.1; 16]).collect();
        let bboxes: Vec<[f64; 4]> = (0..5).map(|i| [i as f64 * 10.0, 0.0, 50.0, 20.0]).collect();
        let out = laa.forward(&tokens, &bboxes);
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].len(), 16);
    }

    #[test]
    fn test_document_tokenizer() {
        let mut rng = make_rng();
        let mut dt = DocumentTokenizer::new(16, &mut rng);
        dt.add_tokens(&["hello", "world", "test"]);
        let pairs: Vec<(String, [f64; 4])> = vec![
            ("hello".to_string(), [0.0, 0.0, 50.0, 20.0]),
            ("world".to_string(), [60.0, 0.0, 50.0, 20.0]),
            ("[UNK]".to_string(), [120.0, 0.0, 30.0, 20.0]),
        ];
        let (ids, embeds) = dt.tokenize(&pairs);
        assert_eq!(ids.len(), 3);
        assert_eq!(embeds.len(), 3);
        assert_eq!(embeds[0].len(), 16);
    }

    #[test]
    fn test_document_tokenizer_unk() {
        let mut rng = make_rng();
        let dt = DocumentTokenizer::new(8, &mut rng);
        let pairs = vec![("unseen_word".to_string(), [0.0, 0.0, 10.0, 10.0])];
        let (ids, _) = dt.tokenize(&pairs);
        // Should map to UNK (id=1)
        assert_eq!(ids[0], 1);
    }

    #[test]
    fn test_table_parser_shape() {
        let tp = TableParser::new(3, 4);
        let features: Vec<Vec<f64>> = (0..24).map(|_| vec![1.0; 8]).collect();
        let cells = tp.parse(&features, 3, 4);
        assert_eq!(cells.len(), 12); // 3*4 cells
        assert_eq!(cells[0].len(), 8);
    }

    #[test]
    fn test_table_parser_values() {
        let tp = TableParser::new(2, 2);
        let features: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
        ];
        let cells = tp.parse(&features, 2, 2);
        assert_eq!(cells.len(), 4);
        // First 2 features average to [1.0, 0.0]
        assert!((cells[0][0] - 1.0).abs() < 1e-10);
        assert!((cells[0][1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_reading_order_sorted() {
        let mut rng = make_rng();
        let rop = ReadingOrderPrediction::new(8, &mut rng);
        let features: Vec<Vec<f64>> = (0..5)
            .map(|_| (0..8).map(|_| rng.random::<f64>()).collect())
            .collect();
        let order = rop.predict_order(&features);
        assert_eq!(order.len(), 5);
        // Check it's a permutation of 0..5
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_reading_order_monotone() {
        let mut rng = make_rng();
        let rop = ReadingOrderPrediction::new(4, &mut rng);
        // Features designed so the score increases with feature value
        let features: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64; 4]).collect();
        let order = rop.predict_order(&features);
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_document_qa_model() {
        let mut rng = make_rng();
        let dqa = DocumentQaModel::new(16, &mut rng);
        let tokens: Vec<Vec<f64>> = (0..6).map(|_| vec![0.1; 16]).collect();
        let bboxes: Vec<[f64; 4]> = (0..6).map(|i| [i as f64 * 20.0, 0.0, 18.0, 12.0]).collect();
        let (start, end) = dqa.predict_span(&tokens, &bboxes);
        assert_eq!(start.len(), 6);
        assert_eq!(end.len(), 6);
        assert!(start.iter().all(|v| v.is_finite()));
        assert!(end.iter().all(|v| v.is_finite()));
    }

    // ─── Video Understanding Tests ────────────────────────────────────────

    #[test]
    fn test_temporal_position_encoding_shape() {
        let tpe = TemporalPositionEmbedding::new(24, 8, 14, 14);
        let enc = tpe.encode(2, 5, 7);
        assert_eq!(enc.len(), 24);
        assert!(enc.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_temporal_position_encoding_distinct() {
        let tpe = TemporalPositionEmbedding::new(24, 8, 14, 14);
        let enc0 = tpe.encode(0, 0, 0);
        let enc1 = tpe.encode(1, 0, 0);
        let enc2 = tpe.encode(0, 1, 0);
        // Different temporal positions should produce different encodings
        assert_ne!(enc0, enc1);
        assert_ne!(enc0, enc2);
    }

    #[test]
    fn test_temporal_position_encoding_sequence() {
        let tpe = TemporalPositionEmbedding::new(12, 4, 4, 4);
        let positions: Vec<(usize, usize, usize)> = (0..8).map(|i| (i / 4, i % 4, 0)).collect();
        let seqs = tpe.encode_sequence(&positions);
        assert_eq!(seqs.len(), 8);
        assert_eq!(seqs[0].len(), 12);
    }

    #[test]
    fn test_timesformer_block() {
        let mut rng = make_rng();
        let tsf = TimesFormerBlock::new(8, 2, 4, &mut rng);
        let tokens: Vec<Vec<f64>> = (0..8).map(|_| vec![0.1; 8]).collect(); // 2 frames * 4 patches
        let out = tsf.forward(&tokens);
        assert_eq!(out.len(), 8);
        assert_eq!(out[0].len(), 8);
        assert!(out.iter().all(|t| t.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn test_timesformer_empty_input() {
        let mut rng = make_rng();
        let tsf = TimesFormerBlock::new(8, 2, 4, &mut rng);
        let out = tsf.forward(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_slow_fast_paths() {
        let mut rng = make_rng();
        // slow: 2 frames * 4 patches = 8 tokens (d=8), fast: 4 frames * 4 patches = 16 tokens (d=4)
        let model = VideoSlowFast::new(8, 4, 2, 4, 4, 6, &mut rng);
        let slow_tokens: Vec<Vec<f64>> = (0..8).map(|_| vec![0.1; 8]).collect();
        let fast_tokens: Vec<Vec<f64>> = (0..16).map(|_| vec![0.05; 4]).collect();
        let out = model.forward(&slow_tokens, &fast_tokens);
        assert_eq!(out.len(), 8);
        assert_eq!(out[0].len(), 6);
        assert!(out.iter().all(|t| t.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn test_slow_fast_output_dim() {
        let mut rng = make_rng();
        let model = VideoSlowFast::new(16, 8, 4, 8, 4, 10, &mut rng);
        let slow_tokens: Vec<Vec<f64>> = (0..16).map(|_| vec![0.1; 16]).collect();
        let fast_tokens: Vec<Vec<f64>> = (0..32).map(|_| vec![0.05; 8]).collect();
        let out = model.forward(&slow_tokens, &fast_tokens);
        assert_eq!(out[0].len(), 10);
    }

    #[test]
    fn test_video_text_alignment() {
        let mut rng = make_rng();
        let vta = VideoTextAlignment::new(16, 12, 8, &mut rng);
        let vfeat = vec![0.1_f64; 16];
        let tfeat = vec![0.2_f64; 12];
        let venc = vta.encode_video(&vfeat);
        let tenc = vta.encode_text(&tfeat);
        assert_eq!(venc.len(), 8);
        assert_eq!(tenc.len(), 8);
        // L2-normalized: norm should be ~1
        let norm: f64 = venc.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_video_text_alignment_loss() {
        let mut rng = make_rng();
        let vta = VideoTextAlignment::new(8, 8, 8, &mut rng);
        let video_embeds = rand_matrix(4, 8, &mut rng);
        let text_embeds = rand_matrix(4, 8, &mut rng);
        let loss = vta.alignment_loss(&video_embeds, &text_embeds).expect("alignment_loss failed");
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn test_video_qa_model() {
        let mut rng = make_rng();
        let vqa = VideoQaModel::new(8, 2, 4, 5, &mut rng);
        let tokens: Vec<Vec<f64>> = (0..8).map(|_| vec![0.1; 8]).collect();
        let logits = vqa.forward(&tokens);
        assert_eq!(logits.len(), 5);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_video_qa_empty_input() {
        let mut rng = make_rng();
        let vqa = VideoQaModel::new(8, 2, 4, 5, &mut rng);
        let logits = vqa.forward(&[]);
        assert_eq!(logits.len(), 5);
    }
}
