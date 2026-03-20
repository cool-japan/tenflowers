//! Mixture of Modalities (MoM) — Unified Multi-Modal Framework.
//!
//! Treats all modalities (text, images, audio, video, tabular, code, math)
//! as token sequences, enabling "any-to-any" cross-modal generation.
//! Inspired by AnyMAL, UnifiedIO, and Gemini's tokenization strategy.
//!
//! # Key Components
//!
//! | Component | Description |
//! |-----------|-------------|
//! | [`MomModalityType`] | Enum of all supported modality types |
//! | [`ModalSequence`] | Unified token sequence spanning multiple modalities |
//! | [`ModalityTokenizer`] | Trait for modality-specific tokenizers |
//! | [`UnifiedTransformer`] | Shared transformer backbone |
//! | [`AnyModalEmbedder`] | Projects any modality into shared embedding space |
//! | [`CrossModalGenerator`] | Auto-regressive cross-modal generation |
//! | [`ModalityRouter`] | MoE-style routing of tokens to modality experts |
//! | [`MultiModalPretrainer`] | Contrastive + generative + masked pretraining |

mod embedder_generator;
mod embedding_transformer;
mod metrics;
mod router_pretrainer;
mod tokenizers;
mod types;

// §1-§2: Core types, tokens, sequences, and tokenizer trait.
pub use types::{ModalSequence, ModalToken, ModalityData, ModalityTokenizer, MomModalityType};

// §3: Concrete tokenizers.
pub use tokenizers::{
    AudioFrameTokenizer, ImagePatchTokenizer, TabularTokenizer, TextModalityTokenizer,
};

// §4-§5: Embeddings and transformer.
pub use embedding_transformer::{
    ModalityEmbedding, UnifiedTransformer, UnifiedTransformerConfig, UnifiedTransformerLayer,
};

// §6-§7: AnyModal embedder and cross-modal generation.
pub use embedder_generator::{
    AnyModalEmbedder, AnyModalEmbedderConfig, CrossModalGenerator, CrossModalGeneratorConfig,
    GenerationMode, ModalProjector,
};

// §8-§9: Router and pretrainer.
pub use router_pretrainer::{
    ModalityExpert, ModalityRouter, ModalityRouterConfig, MultiModalPretrainer, PretrainingConfig,
};

// §10: Metrics.
pub use metrics::{
    compute_mom_metrics, evaluate_cross_modal_alignment, MomMetrics, MomModalityMetrics, MomReport,
};

// ─────────────────────────────────────────────────────────────────────────────
// §11 Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use embedding_transformer::UnifiedTransformer;
    use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
    use scirs2_core::RngExt;
    use types::{xavier_init, ModalToken, ModalityData, MomModalityType};

    fn small_config() -> UnifiedTransformerConfig {
        UnifiedTransformerConfig {
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            d_ff: 32,
            total_vocab: 64,
            max_seq_len: 32,
            dropout: 0.0,
        }
    }

    fn make_text_seq(n_tokens: usize) -> ModalSequence {
        let mut seq = ModalSequence::new();
        let tokens: Vec<ModalToken> = (0..n_tokens)
            .map(|i| ModalToken {
                modality: MomModalityType::Text,
                token_id: i % 64,
                position: i,
            })
            .collect();
        seq.append(tokens);
        seq
    }

    // ── §1 ModalSequence ──────────────────────────────────────────────────

    #[test]
    fn test_modal_sequence_new_empty() {
        let seq = ModalSequence::new();
        assert_eq!(seq.total_tokens(), 0);
        assert!(seq.tokens.is_empty());
        assert!(seq.modality_spans.is_empty());
    }

    #[test]
    fn test_modal_sequence_append_single() {
        let mut seq = ModalSequence::new();
        let toks = vec![ModalToken {
            modality: MomModalityType::Text,
            token_id: 1,
            position: 0,
        }];
        seq.append(toks);
        assert_eq!(seq.total_tokens(), 1);
        assert_eq!(seq.modality_spans.len(), 1);
    }

    #[test]
    fn test_modal_sequence_get_span_present() {
        let seq = make_text_seq(5);
        let span = seq.get_span(&MomModalityType::Text);
        assert_eq!(span, Some((0, 5)));
    }

    #[test]
    fn test_modal_sequence_get_span_absent() {
        let seq = make_text_seq(5);
        let span = seq.get_span(&MomModalityType::Image);
        assert!(span.is_none());
    }

    #[test]
    fn test_modal_sequence_total_tokens() {
        let seq = make_text_seq(10);
        assert_eq!(seq.total_tokens(), 10);
    }

    #[test]
    fn test_modal_sequence_multiple_modalities() {
        let mut seq = ModalSequence::new();
        let text_toks: Vec<ModalToken> = (0..3)
            .map(|i| ModalToken {
                modality: MomModalityType::Text,
                token_id: i,
                position: i,
            })
            .collect();
        seq.append(text_toks);
        let img_toks: Vec<ModalToken> = (0..4)
            .map(|i| ModalToken {
                modality: MomModalityType::Image,
                token_id: i + 3,
                position: i + 3,
            })
            .collect();
        seq.append(img_toks);
        assert_eq!(seq.total_tokens(), 7);
        assert_eq!(seq.modality_spans.len(), 2);
        assert_eq!(seq.get_span(&MomModalityType::Text), Some((0, 3)));
        assert_eq!(seq.get_span(&MomModalityType::Image), Some((3, 7)));
    }

    #[test]
    fn test_modal_sequence_append_empty() {
        let mut seq = ModalSequence::new();
        seq.append(vec![]);
        assert_eq!(seq.total_tokens(), 0);
        assert!(seq.modality_spans.is_empty());
    }

    // ── §2 ModalityData ───────────────────────────────────────────────────

    #[test]
    fn test_modality_data_type_text() {
        let d = ModalityData::Text("hello".into());
        assert_eq!(d.modality_type(), MomModalityType::Text);
    }

    #[test]
    fn test_modality_data_type_image() {
        let d = ModalityData::ImagePatches(vec![vec![0.0; 4]]);
        assert_eq!(d.modality_type(), MomModalityType::Image);
    }

    #[test]
    fn test_modality_data_type_audio() {
        let d = ModalityData::AudioFrames(vec![vec![0.0; 4]]);
        assert_eq!(d.modality_type(), MomModalityType::Audio);
    }

    #[test]
    fn test_modality_data_type_video() {
        let d = ModalityData::VideoFrames(vec![vec![vec![0.0; 4]]]);
        assert_eq!(d.modality_type(), MomModalityType::Video);
    }

    #[test]
    fn test_modality_data_type_tabular() {
        let d = ModalityData::Tabular(vec![vec![1.0, 2.0]]);
        assert_eq!(d.modality_type(), MomModalityType::Tabular);
    }

    #[test]
    fn test_modality_data_type_code() {
        let d = ModalityData::Code("fn main() {}".into());
        assert_eq!(d.modality_type(), MomModalityType::Code);
    }

    #[test]
    fn test_modality_data_type_math() {
        let d = ModalityData::Math("x^2 + y^2".into());
        assert_eq!(d.modality_type(), MomModalityType::Math);
    }

    #[test]
    fn test_modality_data_description() {
        let d = ModalityData::Text("hello world".into());
        let desc = d.description();
        assert!(desc.contains("Text"));
        assert!(desc.contains("11"));
    }

    // ── §3 TextModalityTokenizer ───────────────────────────────────────────

    #[test]
    fn test_text_tokenizer_encode_nonempty() {
        let tok = TextModalityTokenizer::new(256);
        let data = ModalityData::Text("hello".into());
        let ids = tok.encode(&data);
        assert_eq!(ids.len(), 5);
    }

    #[test]
    fn test_text_tokenizer_ids_in_range() {
        let vocab = 128usize;
        let tok = TextModalityTokenizer::new(vocab);
        let data = ModalityData::Text("Hello World 123!".into());
        let ids = tok.encode(&data);
        for &id in &ids {
            assert!(id < vocab);
        }
    }

    #[test]
    fn test_text_tokenizer_decode_produces_text() {
        let tok = TextModalityTokenizer::new(256);
        let ids = vec![65usize, 66, 67];
        let decoded = tok.decode(&ids);
        assert!(matches!(decoded, ModalityData::Text(_)));
    }

    #[test]
    fn test_text_tokenizer_encode_decode_roundtrip_length() {
        let tok = TextModalityTokenizer::new(256);
        let original = "Rust is great!";
        let data = ModalityData::Text(original.into());
        let ids = tok.encode(&data);
        let decoded = tok.decode(&ids);
        if let ModalityData::Text(s) = decoded {
            assert_eq!(s.len(), original.len());
        } else {
            panic!("Expected Text");
        }
    }

    #[test]
    fn test_text_tokenizer_vocab_size() {
        let tok = TextModalityTokenizer::new(512);
        assert_eq!(tok.vocab_size(), 512);
    }

    // ── §3 ImagePatchTokenizer ────────────────────────────────────────────

    #[test]
    fn test_image_tokenizer_nearest_code_in_range() {
        let tok = ImagePatchTokenizer::new(16, 4, 32, 8);
        let patch = vec![0.1f64; 8];
        let idx = tok.nearest_code(&patch);
        assert!(idx < 32);
    }

    #[test]
    fn test_image_tokenizer_encode_length() {
        let n_patches = 6usize;
        let tok = ImagePatchTokenizer::new(16, n_patches, 32, 8);
        let patches: Vec<Vec<f64>> = (0..n_patches).map(|_| vec![0.2; 8]).collect();
        let data = ModalityData::ImagePatches(patches);
        let ids = tok.encode(&data);
        assert_eq!(ids.len(), n_patches);
    }

    #[test]
    fn test_image_tokenizer_ids_in_vocab() {
        let tok = ImagePatchTokenizer::new(16, 4, 32, 8);
        let patches: Vec<Vec<f64>> = (0..4).map(|_| vec![0.5; 8]).collect();
        let data = ModalityData::ImagePatches(patches);
        let ids = tok.encode(&data);
        for &id in &ids {
            assert!(id < 32, "ID {} out of codebook range", id);
        }
    }

    #[test]
    fn test_image_tokenizer_decode_shape() {
        let n_patches = 4usize;
        let patch_dim = 8usize;
        let tok = ImagePatchTokenizer::new(16, n_patches, 32, patch_dim);
        let ids: Vec<usize> = (0..n_patches).map(|i| i % 32).collect();
        let decoded = tok.decode(&ids);
        if let ModalityData::ImagePatches(patches) = decoded {
            assert_eq!(patches.len(), n_patches);
            assert_eq!(patches[0].len(), patch_dim);
        } else {
            panic!("Expected ImagePatches");
        }
    }

    // ── §3 AudioFrameTokenizer ────────────────────────────────────────────

    #[test]
    fn test_audio_tokenizer_encode_length() {
        let n_frames = 5usize;
        let n_cb = 3usize;
        let tok = AudioFrameTokenizer::new(n_cb, 16, 8);
        let frames: Vec<Vec<f64>> = (0..n_frames).map(|_| vec![0.1; 8]).collect();
        let data = ModalityData::AudioFrames(frames);
        let ids = tok.encode(&data);
        assert_eq!(ids.len(), n_frames * n_cb);
    }

    #[test]
    fn test_audio_tokenizer_ids_in_vocab() {
        let tok = AudioFrameTokenizer::new(2, 16, 8);
        let frames = vec![vec![0.5f64; 8]; 3];
        let data = ModalityData::AudioFrames(frames);
        let ids = tok.encode(&data);
        for &id in &ids {
            assert!(id < 16);
        }
    }

    #[test]
    fn test_audio_tokenizer_decode_roundtrip_frames() {
        let n_frames = 4usize;
        let n_cb = 2usize;
        let tok = AudioFrameTokenizer::new(n_cb, 16, 8);
        let ids: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let decoded = tok.decode(&ids);
        if let ModalityData::AudioFrames(frames) = decoded {
            assert_eq!(frames.len(), n_frames);
        } else {
            panic!("Expected AudioFrames");
        }
    }

    // ── §3 TabularTokenizer ────────────────────────────────────────────────

    #[test]
    fn test_tabular_quantize_numeric_min() {
        let tok = TabularTokenizer::new(10, 100);
        assert_eq!(tok.quantize_numeric(0.0, 0.0, 1.0), 0);
    }

    #[test]
    fn test_tabular_quantize_numeric_max() {
        let tok = TabularTokenizer::new(10, 100);
        let bin = tok.quantize_numeric(1.0, 0.0, 1.0);
        assert!(bin < 10);
    }

    #[test]
    fn test_tabular_quantize_numeric_mid() {
        let tok = TabularTokenizer::new(10, 100);
        let bin = tok.quantize_numeric(0.5, 0.0, 1.0);
        assert_eq!(bin, 5);
    }

    #[test]
    fn test_tabular_tokenizer_encode_length() {
        let tok = TabularTokenizer::new(10, 100);
        let rows = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let data = ModalityData::Tabular(rows);
        let ids = tok.encode(&data);
        assert_eq!(ids.len(), 6);
    }

    // ── §4 ModalityEmbedding ──────────────────────────────────────────────

    #[test]
    fn test_modality_embedding_embed_shape() {
        let emb = ModalityEmbedding::new(16, 64, 32);
        let seq = make_text_seq(5);
        let out = emb.embed(&seq);
        assert_eq!(out.len(), 5);
        for row in &out {
            assert_eq!(row.len(), 16);
        }
    }

    #[test]
    fn test_modality_type_index_all_modalities() {
        let modalities = [
            MomModalityType::Text,
            MomModalityType::Image,
            MomModalityType::Audio,
            MomModalityType::Video,
            MomModalityType::Tabular,
            MomModalityType::Code,
            MomModalityType::Math,
        ];
        let indices: Vec<usize> = modalities
            .iter()
            .map(ModalityEmbedding::modality_type_index)
            .collect();
        let mut seen = std::collections::HashSet::new();
        for &idx in &indices {
            assert!(idx < 7);
            assert!(seen.insert(idx), "Duplicate index {}", idx);
        }
    }

    #[test]
    fn test_modality_embedding_embed_empty_seq() {
        let emb = ModalityEmbedding::new(16, 64, 32);
        let seq = ModalSequence::new();
        let out = emb.embed(&seq);
        assert!(out.is_empty());
    }

    // ── §5 UnifiedTransformer ─────────────────────────────────────────────

    #[test]
    fn test_unified_transformer_forward_shape() {
        let cfg = small_config();
        let vocab = cfg.total_vocab;
        let model = UnifiedTransformer::new(cfg);
        let seq = make_text_seq(4);
        let out = model.forward(&seq);
        assert_eq!(out.len(), 4);
        for row in &out {
            assert_eq!(row.len(), vocab);
        }
    }

    #[test]
    fn test_unified_transformer_forward_empty() {
        let model = UnifiedTransformer::new(small_config());
        let seq = ModalSequence::new();
        let out = model.forward(&seq);
        assert!(out.is_empty());
    }

    #[test]
    fn test_layer_norm_mean_approx_zero() {
        let x: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let scale = vec![1.0; 16];
        let bias = vec![0.0; 16];
        let normed = UnifiedTransformer::layer_norm(&x, &scale, &bias);
        let mean: f64 = normed.iter().sum::<f64>() / normed.len() as f64;
        assert!(mean.abs() < 1e-9, "mean = {}", mean);
    }

    #[test]
    fn test_layer_norm_output_length() {
        let x = vec![1.0f64, 2.0, 3.0, 4.0];
        let s = vec![1.0; 4];
        let b = vec![0.0; 4];
        let out = UnifiedTransformer::layer_norm(&x, &s, &b);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_multi_head_attention_output_shape() {
        let d = 16usize;
        let seq_len = 3usize;
        let mut rng = StdRng::seed_from_u64(1);
        let x: Vec<Vec<f64>> = (0..seq_len)
            .map(|_| (0..d).map(|_| rng.random::<f64>()).collect())
            .collect();
        let w = xavier_init(&mut rng, d, d);
        let out = UnifiedTransformer::multi_head_attention(&x, &x, &x, &w, &w, &w, &w, 2, None);
        assert_eq!(out.len(), seq_len);
        for row in &out {
            assert_eq!(row.len(), d);
        }
    }

    // ── §6 AnyModalEmbedder ───────────────────────────────────────────────

    #[test]
    fn test_anymodal_embedder_project_shape() {
        let config = AnyModalEmbedderConfig {
            source_dims: vec![(MomModalityType::Text, 32), (MomModalityType::Image, 64)],
            d_model: 16,
        };
        let embedder = AnyModalEmbedder::new(config);
        let x = vec![0.5f64; 32];
        let out = embedder.project(&x, &MomModalityType::Text);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_anymodal_embedder_project_image_shape() {
        let config = AnyModalEmbedderConfig {
            source_dims: vec![(MomModalityType::Image, 64)],
            d_model: 16,
        };
        let embedder = AnyModalEmbedder::new(config);
        let x = vec![0.1f64; 64];
        let out = embedder.project(&x, &MomModalityType::Image);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_anymodal_embedder_project_sequence_shape() {
        let config = AnyModalEmbedderConfig {
            source_dims: vec![(MomModalityType::Audio, 16)],
            d_model: 8,
        };
        let embedder = AnyModalEmbedder::new(config);
        let seq: Vec<Vec<f64>> = (0..5).map(|_| vec![0.3; 16]).collect();
        let out = embedder.project_sequence(&seq, &MomModalityType::Audio);
        assert_eq!(out.len(), 5);
        for row in &out {
            assert_eq!(row.len(), 8);
        }
    }

    #[test]
    fn test_anymodal_embedder_unknown_modality_returns_zeros() {
        let config = AnyModalEmbedderConfig {
            source_dims: vec![(MomModalityType::Text, 8)],
            d_model: 4,
        };
        let embedder = AnyModalEmbedder::new(config);
        let out = embedder.project(&[0.5; 8], &MomModalityType::Video);
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    // ── §7 CrossModalGenerator ────────────────────────────────────────────

    #[test]
    fn test_crossmodal_generator_generate_nonempty() {
        let cfg = CrossModalGeneratorConfig {
            mode: GenerationMode::TextFromImage,
            max_gen_tokens: 5,
            temperature: 1.0,
            top_k: 10,
        };
        let model = UnifiedTransformer::new(small_config());
        let gen = CrossModalGenerator::new(cfg, model);
        let seq = make_text_seq(3);
        let ids = gen.generate(&seq);
        assert!(!ids.is_empty());
        assert!(ids.len() <= 5);
    }

    #[test]
    fn test_crossmodal_generator_empty_source() {
        let cfg = CrossModalGeneratorConfig {
            mode: GenerationMode::AnyToAny(MomModalityType::Text, MomModalityType::Image),
            max_gen_tokens: 5,
            temperature: 1.0,
            top_k: 5,
        };
        let model = UnifiedTransformer::new(small_config());
        let gen = CrossModalGenerator::new(cfg, model);
        let seq = ModalSequence::new();
        let ids = gen.generate(&seq);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_crossmodal_sample_next_valid_index() {
        let cfg = CrossModalGeneratorConfig {
            mode: GenerationMode::TextFromImage,
            max_gen_tokens: 5,
            temperature: 1.0,
            top_k: 8,
        };
        let model = UnifiedTransformer::new(small_config());
        let gen = CrossModalGenerator::new(cfg, model);
        let logits: Vec<f64> = (0..64).map(|i| i as f64 * 0.01).collect();
        let idx = gen.sample_next(&logits);
        assert!(idx < 64);
    }

    #[test]
    fn test_crossmodal_sample_next_empty_logits() {
        let cfg = CrossModalGeneratorConfig {
            mode: GenerationMode::TextFromImage,
            max_gen_tokens: 5,
            temperature: 1.0,
            top_k: 5,
        };
        let model = UnifiedTransformer::new(small_config());
        let gen = CrossModalGenerator::new(cfg, model);
        let idx = gen.sample_next(&[]);
        assert_eq!(idx, 0);
    }

    // ── §8 ModalityRouter ─────────────────────────────────────────────────

    #[test]
    fn test_modality_router_route_shape() {
        let cfg = ModalityRouterConfig {
            d_model: 16,
            n_modalities: 7,
            expert_dim: 32,
            top_k: 1,
        };
        let router = ModalityRouter::new(cfg);
        let x = vec![0.5f64; 16];
        let out = router.route(&x, &MomModalityType::Text);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_modality_router_route_sequence_shape() {
        let cfg = ModalityRouterConfig {
            d_model: 16,
            n_modalities: 7,
            expert_dim: 32,
            top_k: 1,
        };
        let router = ModalityRouter::new(cfg);
        let xs: Vec<Vec<f64>> = (0..5).map(|_| vec![0.1; 16]).collect();
        let mods = vec![
            MomModalityType::Text,
            MomModalityType::Image,
            MomModalityType::Audio,
            MomModalityType::Video,
            MomModalityType::Tabular,
        ];
        let out = router.route_sequence(&xs, &mods);
        assert_eq!(out.len(), 5);
        for row in &out {
            assert_eq!(row.len(), 16);
        }
    }

    #[test]
    fn test_modality_router_load_balance_nonnegative() {
        let cfg = ModalityRouterConfig {
            d_model: 16,
            n_modalities: 7,
            expert_dim: 32,
            top_k: 1,
        };
        let router = ModalityRouter::new(cfg);
        let xs: Vec<Vec<f64>> = (0..7).map(|_| vec![0.1; 16]).collect();
        let mods: Vec<MomModalityType> = vec![
            MomModalityType::Text,
            MomModalityType::Image,
            MomModalityType::Audio,
            MomModalityType::Video,
            MomModalityType::Tabular,
            MomModalityType::Code,
            MomModalityType::Math,
        ];
        let loss = router.load_balance_loss(&xs, &mods);
        assert!(loss >= 0.0, "load_balance_loss = {}", loss);
    }

    #[test]
    fn test_modality_router_load_balance_empty() {
        let cfg = ModalityRouterConfig {
            d_model: 16,
            n_modalities: 7,
            expert_dim: 32,
            top_k: 1,
        };
        let router = ModalityRouter::new(cfg);
        let loss = router.load_balance_loss(&[], &[]);
        assert_eq!(loss, 0.0);
    }

    // ── §9 MultiModalPretrainer ───────────────────────────────────────────

    #[test]
    fn test_pretrainer_contrastive_loss_positive() {
        let config = PretrainingConfig::default();
        let model = UnifiedTransformer::new(small_config());
        let pretrainer = MultiModalPretrainer::new(config, model);
        let a: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64 * 0.1; 16]).collect();
        let b: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64 * 0.2 + 0.1; 16]).collect();
        let loss = pretrainer.contrastive_loss(&a, &b);
        assert!(loss >= 0.0, "contrastive_loss = {}", loss);
    }

    #[test]
    fn test_pretrainer_contrastive_loss_empty() {
        let config = PretrainingConfig::default();
        let model = UnifiedTransformer::new(small_config());
        let pretrainer = MultiModalPretrainer::new(config, model);
        let loss = pretrainer.contrastive_loss(&[], &[]);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_pretrainer_generative_loss_nonneg() {
        let config = PretrainingConfig::default();
        let model = UnifiedTransformer::new(small_config());
        let pretrainer = MultiModalPretrainer::new(config, model);
        let seq = make_text_seq(4);
        let labels = vec![1usize, 2, 3, 4];
        let loss = pretrainer.generative_loss(&seq, &labels);
        assert!(loss >= 0.0, "generative_loss = {}", loss);
    }

    #[test]
    fn test_pretrainer_masked_loss_nonneg() {
        let config = PretrainingConfig::default();
        let model = UnifiedTransformer::new(small_config());
        let pretrainer = MultiModalPretrainer::new(config, model);
        let seq = make_text_seq(8);
        let loss = pretrainer.masked_loss(&seq);
        assert!(loss >= 0.0, "masked_loss = {}", loss);
    }

    #[test]
    fn test_pretrainer_total_loss_nonneg() {
        let config = PretrainingConfig::default();
        let model = UnifiedTransformer::new(small_config());
        let pretrainer = MultiModalPretrainer::new(config, model);
        let seq = make_text_seq(6);
        let loss = pretrainer.total_loss(&seq, None);
        assert!(loss >= 0.0, "total_loss = {}", loss);
    }

    #[test]
    fn test_pretrainer_total_loss_with_pairs() {
        let config = PretrainingConfig::default();
        let model = UnifiedTransformer::new(small_config());
        let pretrainer = MultiModalPretrainer::new(config, model);
        let seq = make_text_seq(4);
        let sa = make_text_seq(3);
        let sb = make_text_seq(3);
        let loss = pretrainer.total_loss(&seq, Some((&sa, &sb)));
        assert!(loss >= 0.0, "total_loss_with_pairs = {}", loss);
    }

    // ── §10 MoM Metrics ───────────────────────────────────────────────────

    #[test]
    fn test_mom_modality_metrics_reconstruction_error_nonneg() {
        let m = MomModalityMetrics {
            modality: MomModalityType::Text,
            tokenization_rate: 1000.0,
            vocab_coverage: 0.8,
            reconstruction_error: 0.05,
        };
        assert!(m.reconstruction_error >= 0.0);
    }

    #[test]
    fn test_mom_metrics_cross_modal_alignment_range() {
        let cfg = ModalityRouterConfig {
            d_model: 8,
            n_modalities: 7,
            expert_dim: 16,
            top_k: 1,
        };
        let router = ModalityRouter::new(cfg);
        let model = UnifiedTransformer::new(UnifiedTransformerConfig {
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            d_ff: 16,
            total_vocab: 32,
            max_seq_len: 16,
            dropout: 0.0,
        });
        let embedder_cfg = AnyModalEmbedderConfig {
            source_dims: vec![(MomModalityType::Text, 8), (MomModalityType::Image, 8)],
            d_model: 8,
        };
        let embedder = AnyModalEmbedder::new(embedder_cfg);
        let metrics = compute_mom_metrics(&model, &embedder, &router);
        assert!(
            metrics.cross_modal_alignment >= -1.0 && metrics.cross_modal_alignment <= 1.0,
            "cross_modal_alignment = {}",
            metrics.cross_modal_alignment
        );
    }

    #[test]
    fn test_compute_mom_metrics_valid() {
        let cfg = ModalityRouterConfig {
            d_model: 8,
            n_modalities: 7,
            expert_dim: 16,
            top_k: 1,
        };
        let router = ModalityRouter::new(cfg);
        let model = UnifiedTransformer::new(UnifiedTransformerConfig {
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            d_ff: 16,
            total_vocab: 32,
            max_seq_len: 16,
            dropout: 0.0,
        });
        let embedder_cfg = AnyModalEmbedderConfig {
            source_dims: vec![(MomModalityType::Text, 8)],
            d_model: 8,
        };
        let embedder = AnyModalEmbedder::new(embedder_cfg);
        let metrics = compute_mom_metrics(&model, &embedder, &router);
        assert_eq!(metrics.per_modality.len(), 7);
        assert!(metrics.total_parameters > 0);
        assert!(metrics.routing_load_balance >= 0.0);
    }

    #[test]
    fn test_mom_report_modalities_nonempty() {
        let metrics = MomMetrics {
            per_modality: vec![MomModalityMetrics {
                modality: MomModalityType::Text,
                tokenization_rate: 1000.0,
                vocab_coverage: 0.9,
                reconstruction_error: 0.01,
            }],
            cross_modal_alignment: 0.8,
            routing_load_balance: 0.95,
            total_parameters: 1_000_000,
        };
        let report = MomReport {
            metrics,
            modalities_supported: vec![MomModalityType::Text, MomModalityType::Image],
            n_layers: 2,
            d_model: 16,
        };
        assert!(!report.modalities_supported.is_empty());
        assert_eq!(report.n_layers, 2);
    }

    #[test]
    fn test_evaluate_cross_modal_alignment_range() {
        let config = AnyModalEmbedderConfig {
            source_dims: vec![(MomModalityType::Text, 8), (MomModalityType::Image, 8)],
            d_model: 8,
        };
        let embedder = AnyModalEmbedder::new(config);
        let pairs = vec![
            (
                vec![0.1f64; 8],
                MomModalityType::Text,
                vec![0.2f64; 8],
                MomModalityType::Image,
            ),
            (
                vec![0.5f64; 8],
                MomModalityType::Text,
                vec![0.3f64; 8],
                MomModalityType::Image,
            ),
        ];
        let alignment = evaluate_cross_modal_alignment(&embedder, &pairs);
        assert!(
            (-1.0..=1.0).contains(&alignment),
            "alignment = {}",
            alignment
        );
    }

    #[test]
    fn test_evaluate_cross_modal_alignment_empty() {
        let config = AnyModalEmbedderConfig {
            source_dims: vec![(MomModalityType::Text, 8)],
            d_model: 8,
        };
        let embedder = AnyModalEmbedder::new(config);
        let result = evaluate_cross_modal_alignment(&embedder, &[]);
        assert_eq!(result, 0.0);
    }

    // ── Additional coverage tests ─────────────────────────────────────────

    #[test]
    fn test_modal_sequence_default() {
        let seq = ModalSequence::default();
        assert_eq!(seq.total_tokens(), 0);
    }

    #[test]
    fn test_text_tokenizer_code_modality() {
        let tok = TextModalityTokenizer::new(128);
        let data = ModalityData::Code("let x = 1;".into());
        let ids = tok.encode(&data);
        assert_eq!(ids.len(), 10);
    }

    #[test]
    fn test_text_tokenizer_math_modality() {
        let tok = TextModalityTokenizer::new(128);
        let data = ModalityData::Math("e^{i\\pi}+1=0".into());
        let ids = tok.encode(&data);
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_audio_tokenizer_quantize_frame_length() {
        let tok = AudioFrameTokenizer::new(4, 16, 8);
        let frame = vec![0.5f64; 8];
        let ids = tok.quantize_frame(&frame);
        assert_eq!(ids.len(), 4);
    }

    #[test]
    fn test_tabular_tokenizer_empty_data() {
        let tok = TabularTokenizer::new(10, 100);
        let data = ModalityData::Tabular(vec![]);
        let ids = tok.encode(&data);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_tabular_tokenizer_single_value() {
        let tok = TabularTokenizer::new(10, 100);
        let data = ModalityData::Tabular(vec![vec![5.0]]);
        let ids = tok.encode(&data);
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], 0); // single value: min==max -> bin 0
    }

    #[test]
    fn test_image_tokenizer_wrong_modality() {
        let tok = ImagePatchTokenizer::new(16, 4, 32, 8);
        let data = ModalityData::Text("hello".into());
        let ids = tok.encode(&data);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_audio_tokenizer_wrong_modality() {
        let tok = AudioFrameTokenizer::new(2, 16, 8);
        let data = ModalityData::Text("hello".into());
        let ids = tok.encode(&data);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_generation_mode_any_to_any() {
        let cfg = CrossModalGeneratorConfig {
            mode: GenerationMode::AnyToAny(MomModalityType::Image, MomModalityType::Text),
            max_gen_tokens: 3,
            temperature: 0.8,
            top_k: 5,
        };
        let model = UnifiedTransformer::new(small_config());
        let gen = CrossModalGenerator::new(cfg, model);
        let seq = make_text_seq(2);
        let ids = gen.generate(&seq);
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_pretraining_config_default() {
        let cfg = PretrainingConfig::default();
        assert_eq!(cfg.mask_ratio, 0.15);
        assert_eq!(cfg.contrastive_weight, 1.0);
        assert!(cfg.temperature > 0.0);
    }

    #[test]
    fn test_mom_metrics_routing_nonneg() {
        let cfg = ModalityRouterConfig {
            d_model: 8,
            n_modalities: 7,
            expert_dim: 16,
            top_k: 1,
        };
        let router = ModalityRouter::new(cfg);
        let model = UnifiedTransformer::new(UnifiedTransformerConfig {
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            d_ff: 16,
            total_vocab: 32,
            max_seq_len: 16,
            dropout: 0.0,
        });
        let embedder = AnyModalEmbedder::new(AnyModalEmbedderConfig {
            source_dims: vec![(MomModalityType::Text, 8)],
            d_model: 8,
        });
        let metrics = compute_mom_metrics(&model, &embedder, &router);
        assert!(metrics.routing_load_balance >= 0.0);
    }
}
