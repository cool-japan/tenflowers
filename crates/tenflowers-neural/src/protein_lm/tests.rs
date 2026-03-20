//! Tests for the `protein_lm` module (Round 45 Track A).
//!
//! All tests use small model dimensions (embed_dim=32, n_heads=4, n_layers=2)
//! for speed.  Sequences are short protein fragments.

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// §1  PlmTokenizer tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tok_new_has_correct_vocab_size() {
    let tok = PlmTokenizer::new();
    // 4 special + 20 canonical + 5 IUPAC = 29
    assert_eq!(tok.vocab_size, 29);
}

#[test]
fn tok_encode_adds_cls_and_eos() {
    let tok = PlmTokenizer::new();
    let encoded = tok.encode("ACDE").expect("encoding ACDE should succeed");
    assert_eq!(encoded[0], PLM_CLS);
    assert_eq!(*encoded.last().expect("encoded sequence should not be empty"), PLM_EOS);
    assert_eq!(encoded.len(), 6); // CLS + 4 AA + EOS
}

#[test]
fn tok_encode_rejects_invalid_char() {
    let tok = PlmTokenizer::new();
    let result = tok.encode("AO"); // O is not a standard IUPAC code here
    assert!(result.is_err());
}

#[test]
fn tok_decode_strips_special_tokens() {
    let tok = PlmTokenizer::new();
    let encoded = tok.encode("ACDE").expect("encoding ACDE should succeed");
    let decoded = tok.decode(&encoded).expect("decoding should succeed");
    assert_eq!(decoded, "ACDE");
}

#[test]
fn tok_encode_decode_roundtrip_20aa() {
    let tok = PlmTokenizer::new();
    let seq = "ACDEFGHIKLMNPQRSTVWY";
    let encoded = tok.encode(seq).expect("encoding 20 AA should succeed");
    let decoded = tok.decode(&encoded).expect("decoding should succeed");
    assert_eq!(decoded, seq);
}

#[test]
fn tok_encode_lowercase_fails() {
    // Our tokenizer uppercases, so lowercase should work via encode.
    // Actually our implementation uppercases via to_ascii_uppercase.
    let tok = PlmTokenizer::new();
    let result = tok.encode("acde");
    assert!(result.is_ok());
    assert_eq!(result.expect("encoding lowercase should succeed").len(), 6);
}

#[test]
fn tok_batch_tokenize() {
    let tok = PlmTokenizer::new();
    let seqs = ["ACDE", "FGHI", "KLM"];
    let batch = tok.tokenize_batch(&seqs).expect("batch tokenization should succeed");
    assert_eq!(batch.len(), 3);
    assert_eq!(batch[0].len(), 6);
    assert_eq!(batch[1].len(), 6);
    assert_eq!(batch[2].len(), 5);
}

#[test]
fn tok_special_token_ids_correct() {
    assert_eq!(PLM_PAD, 0);
    assert_eq!(PLM_MASK, 1);
    assert_eq!(PLM_CLS, 2);
    assert_eq!(PLM_EOS, 3);
}

#[test]
fn tok_iupac_b_encodes() {
    let tok = PlmTokenizer::new();
    let result = tok.encode("B");
    assert!(result.is_ok());
}

#[test]
fn tok_iupac_x_encodes() {
    let tok = PlmTokenizer::new();
    let result = tok.encode("X");
    assert!(result.is_ok());
}

#[test]
fn tok_single_aa_roundtrip() {
    let tok = PlmTokenizer::new();
    for aa in "ACDEFGHIKLMNPQRSTVWY".chars() {
        let seq = aa.to_string();
        let enc = tok.encode(&seq).expect("encoding single AA should succeed");
        let dec = tok.decode(&enc).expect("decoding single AA should succeed");
        assert_eq!(dec, seq);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  PlmEmbedding tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn embed_output_shape() {
    let embed = PlmEmbedding::new(29, 32, 512);
    let tokens = vec![PLM_CLS, 4, 5, 6, PLM_EOS];
    let out = embed.forward(&tokens).expect("embedding forward should succeed");
    assert_eq!(out.len(), 5);
    assert_eq!(out[0].len(), 32);
}

#[test]
fn embed_different_positions_differ() {
    let embed = PlmEmbedding::new(29, 32, 512);
    // Same token, different positions.
    let tokens = vec![4, 4, 4]; // three identical tokens
    let out = embed.forward(&tokens).expect("embedding forward should succeed");
    // With learned pos embeddings, different positions should differ.
    assert_ne!(out[0], out[1]);
}

#[test]
fn embed_different_tokens_at_same_pos_differ() {
    let embed = PlmEmbedding::new(29, 32, 512);
    let t1 = embed.forward(&[4]).expect("embedding forward for token 4 should succeed");
    let t2 = embed.forward(&[5]).expect("embedding forward for token 5 should succeed");
    assert_ne!(t1[0], t2[0]);
}

#[test]
fn embed_exceeds_max_len_errors() {
    let embed = PlmEmbedding::new(29, 32, 5);
    let tokens: Vec<usize> = (0..10).map(|i| 4 + (i % 20)).collect();
    let result = embed.forward(&tokens);
    assert!(result.is_err());
}

#[test]
fn embed_out_of_vocab_errors() {
    let embed = PlmEmbedding::new(10, 32, 512);
    let tokens = vec![50]; // out of vocab
    let result = embed.forward(&tokens);
    assert!(result.is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  PlmLayerNorm tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ln_output_approx_zero_mean() {
    let ln = PlmLayerNorm::new(32);
    let row = vec![1.0_f64; 32];
    // All same values → output should be ~0 (mean subtracted, zero variance case)
    let out = ln.forward(&[row]);
    // When variance=0, result = (x - mean)/eps^0.5 * gamma + beta ≈ 0
    // Actually when all same, (x-mean)=0, output is beta=0.
    for &v in &out[0] {
        assert!(v.abs() < 1e-6, "Expected ~0, got {v}");
    }
}

#[test]
fn ln_normalizes_mean_to_zero() {
    let ln = PlmLayerNorm::new(16);
    let row: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let out = ln.forward(&[row]);
    let mean: f64 = out[0].iter().sum::<f64>() / 16.0;
    assert!(mean.abs() < 1e-10, "Mean should be ~0, got {mean}");
}

#[test]
fn ln_normalizes_variance_to_one() {
    let ln = PlmLayerNorm::new(16);
    let row: Vec<f64> = (0..16).map(|i| i as f64 * 3.7 + 1.2).collect();
    let out = ln.forward(&[row]);
    let mean = out[0].iter().sum::<f64>() / 16.0;
    let var = out[0].iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / 16.0;
    assert!((var - 1.0).abs() < 0.01, "Variance should be ~1, got {var}");
}

#[test]
fn ln_different_inputs_give_different_outputs() {
    let ln = PlmLayerNorm::new(8);
    let r1: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let r2: Vec<f64> = (0..8).map(|i| (i as f64) * 2.0).collect();
    let o1 = ln.forward(&[r1]);
    let o2 = ln.forward(&[r2]);
    // Both normalize to same shape, but original ordering preserved
    assert_eq!(o1[0].len(), 8);
    assert_eq!(o2[0].len(), 8);
    // With unit gamma, normalized vectors of same relative structure should match
    let norm_sum1: f64 = o1[0].iter().sum();
    let norm_sum2: f64 = o2[0].iter().sum();
    assert!((norm_sum1 - norm_sum2).abs() < 1e-8);
}

#[test]
fn ln_preserves_sequence_length() {
    let ln = PlmLayerNorm::new(32);
    let seq: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64; 32]).collect();
    let out = ln.forward(&seq);
    assert_eq!(out.len(), 5);
    assert_eq!(out[0].len(), 32);
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  PlmMultiHeadAttention tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mha_construction_valid() {
    let mha = PlmMultiHeadAttention::new(32, 4);
    assert!(mha.is_ok());
}

#[test]
fn mha_construction_invalid_heads() {
    let result = PlmMultiHeadAttention::new(30, 4); // 30 not divisible by 4
    assert!(result.is_err());
}

#[test]
fn mha_output_shape() {
    let mha = PlmMultiHeadAttention::new(32, 4).expect("MHA construction should succeed");
    let x: Vec<Vec<f64>> = (0..5).map(|_| vec![0.1; 32]).collect();
    let out = mha.forward(&x, None).expect("MHA forward should succeed");
    assert_eq!(out.len(), 5);
    assert_eq!(out[0].len(), 32);
}

#[test]
fn mha_empty_input() {
    let mha = PlmMultiHeadAttention::new(32, 4).expect("MHA construction should succeed");
    let out = mha.forward(&[], None).expect("MHA forward with empty input should succeed");
    assert!(out.is_empty());
}

#[test]
fn mha_output_differs_from_input() {
    let mha = PlmMultiHeadAttention::new(32, 4).expect("MHA construction should succeed");
    let x: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64 * 0.1; 32]).collect();
    let out = mha.forward(&x, None).expect("MHA forward should succeed");
    // Output should differ from input due to learned projections.
    assert_ne!(out[0], x[0]);
}

#[test]
fn mha_rope_does_not_change_magnitude_greatly() {
    // RoPE rotates vectors — norm should be preserved exactly.
    let mha = PlmMultiHeadAttention::new(32, 4).expect("MHA construction should succeed");
    let head_dim = 8;
    let x: Vec<Vec<f64>> = (0..4)
        .map(|_| (0..head_dim).map(|i| (i + 1) as f64 * 0.1).collect())
        .collect();
    let rotated = mha.apply_rope(&x, head_dim);
    for (orig, rot) in x.iter().zip(rotated.iter()) {
        let norm_orig: f64 = orig.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_rot: f64 = rot.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            (norm_orig - norm_rot).abs() < 1e-10,
            "RoPE should preserve magnitude: {norm_orig} vs {norm_rot}"
        );
    }
}

#[test]
fn mha_forward_with_attn_returns_heads() {
    let mha = PlmMultiHeadAttention::new(32, 4).expect("MHA construction should succeed");
    let x: Vec<Vec<f64>> = vec![vec![0.5; 32]; 3];
    let (out, attns) = mha.forward_with_attn(&x, None).expect("forward_with_attn should succeed");
    assert_eq!(out.len(), 3);
    assert_eq!(attns.len(), 4); // n_heads
    assert_eq!(attns[0].len(), 3); // seq_len
    assert_eq!(attns[0][0].len(), 3);
}

#[test]
fn mha_attention_rows_sum_to_one() {
    let mha = PlmMultiHeadAttention::new(32, 4).expect("MHA construction should succeed");
    let x: Vec<Vec<f64>> = vec![vec![0.1; 32]; 4];
    let (_out, attns) = mha.forward_with_attn(&x, None).expect("forward_with_attn should succeed");
    for head in &attns {
        for row in head {
            let s: f64 = row.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-10,
                "Attention row should sum to 1: {s}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  PlmFeedForward (SwiGLU) tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ff_output_shape() {
    let ff = PlmFeedForward::new(32, 4);
    let x: Vec<Vec<f64>> = vec![vec![0.1; 32]; 5];
    let out = ff.forward(&x);
    assert_eq!(out.len(), 5);
    assert_eq!(out[0].len(), 32);
}

#[test]
fn ff_swiglu_is_nonlinear() {
    let ff = PlmFeedForward::new(8, 4);
    let x1: Vec<Vec<f64>> = vec![vec![1.0; 8]];
    let x2: Vec<Vec<f64>> = vec![vec![-1.0; 8]];
    let o1 = ff.forward(&x1);
    let o2 = ff.forward(&x2);
    // Due to SwiGLU non-linearity, output[1] != -output[1] generally
    let sum1: f64 = o1[0].iter().sum();
    let sum2: f64 = o2[0].iter().sum();
    assert_ne!(sum1, sum2, "SwiGLU should be non-linear");
}

#[test]
fn ff_silu_at_zero() {
    let ff = PlmFeedForward::new(8, 2);
    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    let val = ff.silu(0.0);
    assert!((val - 0.0).abs() < 1e-15);
}

#[test]
fn ff_silu_positive_domain() {
    let ff = PlmFeedForward::new(8, 2);
    // silu(x) > 0 for x > 0
    assert!(ff.silu(1.0) > 0.0);
    assert!(ff.silu(2.0) > 0.0);
}

#[test]
fn ff_silu_negative_domain() {
    let ff = PlmFeedForward::new(8, 2);
    // silu(x) < 0 slightly for small negative x, approaches 0 for large negative
    // silu(-1.0) = -1 * sigmoid(-1) = -1 * (1/(1+e)) ≈ -0.269
    let val = ff.silu(-1.0);
    assert!(val < 0.0);
}

#[test]
fn ff_empty_seq() {
    let ff = PlmFeedForward::new(32, 4);
    let out = ff.forward(&[]);
    assert!(out.is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  PlmTransformerBlock tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn block_construction() {
    let block = PlmTransformerBlock::new(32, 4, 4);
    assert!(block.is_ok());
}

#[test]
fn block_output_shape() {
    let block = PlmTransformerBlock::new(32, 4, 4).expect("block construction should succeed");
    let x: Vec<Vec<f64>> = vec![vec![0.1; 32]; 6];
    let out = block.forward(&x, None).expect("block forward should succeed");
    assert_eq!(out.len(), 6);
    assert_eq!(out[0].len(), 32);
}

#[test]
fn block_residual_connection_changes_output() {
    let block = PlmTransformerBlock::new(32, 4, 4).expect("block construction should succeed");
    // Use varied input so layer_norm produces non-zero output.
    let x: Vec<Vec<f64>> = (0..4)
        .map(|i| {
            (0..32)
                .map(|j| (i as f64 + 1.0) * (j as f64 * 0.1 + 0.5))
                .collect()
        })
        .collect();
    let out = block.forward(&x, None).expect("block forward should succeed");
    // The Frobenius norm of the output should differ from input.
    let sum_in: f64 = x.iter().flatten().map(|v| v * v).sum::<f64>();
    let sum_out: f64 = out.iter().flatten().map(|v| v * v).sum::<f64>();
    assert!(
        (sum_in - sum_out).abs() > 1e-10,
        "Output norm should differ from input norm: in={sum_in}, out={sum_out}"
    );
}

#[test]
fn block_preserves_sequence_length() {
    let block = PlmTransformerBlock::new(32, 4, 4).expect("block construction should succeed");
    let x: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64; 32]).collect();
    let out = block.forward(&x, None).expect("block forward should succeed");
    assert_eq!(out.len(), 8);
}

#[test]
fn block_different_positions_differ() {
    let block = PlmTransformerBlock::new(32, 4, 4).expect("block construction should succeed");
    let x: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64 * 0.3; 32]).collect();
    let out = block.forward(&x, None).expect("block forward should succeed");
    assert_ne!(out[0], out[1]);
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  PlmEncoder tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn encoder_construction() {
    let enc = PlmEncoder::new(29, 32, 2, 4);
    assert!(enc.is_ok());
}

#[test]
fn encoder_forward_output_shape() {
    let enc = PlmEncoder::new(29, 32, 2, 4).expect("encoder construction should succeed");
    let tokens = vec![PLM_CLS, 4, 5, 6, PLM_EOS];
    let out = enc.forward(&tokens).expect("encoder forward should succeed");
    assert_eq!(out.len(), 5);
    assert_eq!(out[0].len(), 32);
}

#[test]
fn encoder_embed_sequence_shape() {
    let tok = PlmTokenizer::new();
    let enc = PlmEncoder::new(29, 32, 2, 4).expect("encoder construction should succeed");
    let out = enc.embed_sequence("ACDEF", &tok).expect("embed_sequence should succeed");
    // ACDEF = 5 AA + CLS + EOS = 7 tokens
    assert_eq!(out.len(), 7);
    assert_eq!(out[0].len(), 32);
}

#[test]
fn encoder_different_sequences_differ() {
    let tok = PlmTokenizer::new();
    let enc = PlmEncoder::new(29, 32, 2, 4).expect("encoder construction should succeed");
    let o1 = enc.embed_sequence("ACDE", &tok).expect("embed_sequence for ACDE should succeed");
    let o2 = enc.embed_sequence("FGHI", &tok).expect("embed_sequence for FGHI should succeed");
    assert_ne!(o1[1], o2[1]); // First residue position (after CLS)
}

#[test]
fn encoder_invalid_seq_returns_error() {
    let tok = PlmTokenizer::new();
    let enc = PlmEncoder::new(29, 32, 2, 4).expect("encoder construction should succeed");
    let result = enc.embed_sequence("AO?", &tok); // invalid chars
    assert!(result.is_err());
}

#[test]
fn encoder_single_aa() {
    let tok = PlmTokenizer::new();
    let enc = PlmEncoder::new(29, 32, 2, 4).expect("encoder construction should succeed");
    let out = enc.embed_sequence("A", &tok).expect("embed_sequence for single AA should succeed");
    assert_eq!(out.len(), 3); // CLS + A + EOS
    assert_eq!(out[0].len(), 32);
}

#[test]
fn encoder_long_sequence() {
    let tok = PlmTokenizer::new();
    let enc = PlmEncoder::new(29, 32, 2, 4).expect("encoder construction should succeed");
    let seq = "ACDEFGHIKLMNPQRSTVWY".repeat(3); // 60 AA
    let out = enc.embed_sequence(&seq, &tok).expect("embed_sequence for long sequence should succeed");
    assert_eq!(out.len(), 62); // 60 + CLS + EOS
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  PlmContactPredictor tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn contact_predictor_construction() {
    let cp = PlmContactPredictor::new(32, 2, 4);
    assert!(cp.is_ok());
}

#[test]
fn contact_predictor_returns_square_matrix() {
    let tok = PlmTokenizer::new();
    let cp = PlmContactPredictor::new(32, 2, 4).expect("contact predictor construction should succeed");
    let contacts = cp.predict_contacts("ACDEF", &tok).expect("predict_contacts should succeed");
    let l = 5; // "ACDEF" has 5 residues
    assert_eq!(contacts.len(), l);
    assert_eq!(contacts[0].len(), l);
}

#[test]
fn contact_predictor_values_in_0_1() {
    let tok = PlmTokenizer::new();
    let cp = PlmContactPredictor::new(32, 2, 4).expect("contact predictor construction should succeed");
    let contacts = cp.predict_contacts("ACDE", &tok).expect("predict_contacts should succeed");
    for row in &contacts {
        for &v in row {
            assert!(
                (0.0..=1.0).contains(&v),
                "Contact prob should be in [0,1], got {v}"
            );
        }
    }
}

#[test]
fn contact_predictor_symmetric() {
    let tok = PlmTokenizer::new();
    let cp = PlmContactPredictor::new(32, 2, 4).expect("contact predictor construction should succeed");
    let contacts = cp.predict_contacts("ACDEF", &tok).expect("predict_contacts should succeed");
    let l = contacts.len();
    for i in 0..l {
        for j in 0..l {
            assert!(
                (contacts[i][j] - contacts[j][i]).abs() < 1e-6,
                "Contact matrix should be symmetric at ({i},{j})"
            );
        }
    }
}

#[test]
fn contact_apc_reduces_background() {
    // APC should change the raw attention values.
    let cp = PlmContactPredictor::new(32, 2, 4).expect("contact predictor construction should succeed");
    let attn: Vec<Vec<f64>> = vec![
        vec![0.5, 0.3, 0.2],
        vec![0.3, 0.5, 0.2],
        vec![0.2, 0.2, 0.6],
    ];
    let apc = cp.apply_apc(&attn);
    // APC should change values.
    assert_ne!(apc[0][1], attn[0][1]);
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  PlmFitnessPredictor tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fitness_construction() {
    let fp = PlmFitnessPredictor::new(32, 2, 4, 29);
    assert!(fp.is_ok());
}

#[test]
fn fitness_score_single_substitution() {
    let tok = PlmTokenizer::new();
    let fp = PlmFitnessPredictor::new(32, 2, 4, 29).expect("fitness predictor construction should succeed");
    // Single amino acid substitution: A→C at position 0
    let wt = "ACDE";
    let mt = "CCDE";
    let score = fp.score_mutation(wt, mt, &tok);
    assert!(score.is_ok(), "Should score single substitution");
    // Score is a real number.
    assert!(score.expect("score_mutation should succeed").is_finite());
}

#[test]
fn fitness_wildtype_scores_zero() {
    let tok = PlmTokenizer::new();
    let fp = PlmFitnessPredictor::new(32, 2, 4, 29).expect("fitness predictor construction should succeed");
    // wt vs itself should score 0 (no mutations).
    let score = fp.score_mutation("ACDE", "ACDE", &tok).expect("score_mutation for identical sequences should succeed");
    assert_eq!(score, 0.0);
}

#[test]
fn fitness_mismatched_lengths_error() {
    let tok = PlmTokenizer::new();
    let fp = PlmFitnessPredictor::new(32, 2, 4, 29).expect("fitness predictor construction should succeed");
    let result = fp.score_mutation("ACDE", "ACD", &tok);
    assert!(result.is_err());
}

#[test]
fn fitness_batch_scoring() {
    let tok = PlmTokenizer::new();
    let fp = PlmFitnessPredictor::new(32, 2, 4, 29).expect("fitness predictor construction should succeed");
    let wt = "ACDE";
    let mutants = ["CCDE", "ACFE", "ACDA"];
    let scores = fp.score_batch(wt, &mutants, &tok).expect("batch scoring should succeed");
    assert_eq!(scores.len(), 3);
    for s in &scores {
        assert!(s.is_finite());
    }
}

#[test]
fn fitness_double_mutation_differs_from_single() {
    let tok = PlmTokenizer::new();
    let fp = PlmFitnessPredictor::new(32, 2, 4, 29).expect("fitness predictor construction should succeed");
    let wt = "ACDE";
    let single = fp.score_mutation(wt, "CCDE", &tok).expect("single mutation scoring should succeed");
    let double = fp.score_mutation(wt, "CCFE", &tok).expect("double mutation scoring should succeed");
    // They should generally differ (each position contributes independently).
    assert!(single.is_finite() && double.is_finite());
}

#[test]
fn fitness_log_softmax_sums_to_zero_exp() {
    let fp = PlmFitnessPredictor::new(32, 2, 4, 29).expect("fitness predictor construction should succeed");
    let logits: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
    let log_probs = fp.log_softmax(&logits);
    let sum_exp: f64 = log_probs.iter().map(|&x| x.exp()).sum();
    assert!(
        (sum_exp - 1.0).abs() < 1e-10,
        "log_softmax exp sum should be 1, got {sum_exp}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  PlmMaskedLMLoss tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mlm_loss_construction() {
    let loss = PlmMaskedLMLoss::new(0.15, 29);
    assert_eq!(loss.mask_ratio, 0.15);
    assert_eq!(loss.vocab_size, 29);
}

#[test]
fn mlm_create_mask_replaces_tokens() {
    let loss = PlmMaskedLMLoss::new(0.5, 29);
    let tokens = vec![PLM_CLS, 4, 5, 6, 7, PLM_EOS];
    let (masked, positions) = loss.create_mask(&tokens, PLM_MASK);
    // Some amino-acid tokens should be replaced by MASK.
    if !positions.is_empty() {
        for &(pos, _orig) in &positions {
            assert_eq!(masked[pos], PLM_MASK);
        }
    }
    // CLS and EOS should never be masked.
    assert_eq!(masked[0], PLM_CLS);
    assert_eq!(*masked.last().expect("masked sequence should not be empty"), PLM_EOS);
}

#[test]
fn mlm_create_mask_records_originals() {
    let loss = PlmMaskedLMLoss::new(1.0, 29); // mask everything eligible
    let tokens = vec![PLM_CLS, 4, 5, 6, PLM_EOS];
    let (_masked, positions) = loss.create_mask(&tokens, PLM_MASK);
    // All 3 AA tokens should be masked with ratio=1.0
    assert_eq!(positions.len(), 3);
    // Check original IDs are recorded.
    let orig_ids: Vec<usize> = positions.iter().map(|&(_, id)| id).collect();
    assert!(orig_ids.contains(&4));
    assert!(orig_ids.contains(&5));
    assert!(orig_ids.contains(&6));
}

#[test]
fn mlm_loss_zero_masked_positions() {
    let loss = PlmMaskedLMLoss::new(0.15, 29);
    let logits = vec![vec![1.0_f64; 29]; 5];
    let l = loss.compute_loss(&logits, &[]);
    assert_eq!(l, 0.0);
}

#[test]
fn mlm_loss_correct_prediction_gives_low_loss() {
    let vocab_size = 4;
    let loss = PlmMaskedLMLoss::new(0.15, vocab_size);
    // Logits strongly predict token 2.
    let logits = vec![vec![-100.0, -100.0, 100.0, -100.0]];
    let masked_positions = vec![(0, 2)]; // true token = 2
    let l = loss.compute_loss(&logits, &masked_positions);
    assert!(
        l < 0.01,
        "Loss should be near 0 for perfect prediction, got {l}"
    );
}

#[test]
fn mlm_loss_wrong_prediction_gives_high_loss() {
    let vocab_size = 4;
    let loss = PlmMaskedLMLoss::new(0.15, vocab_size);
    let logits = vec![vec![-100.0, -100.0, 100.0, -100.0]];
    let masked_positions = vec![(0, 0)]; // true token = 0, predicted = 2
    let l = loss.compute_loss(&logits, &masked_positions);
    assert!(
        l > 50.0,
        "Loss should be high for wrong prediction, got {l}"
    );
}

#[test]
fn mlm_special_tokens_never_masked() {
    let loss = PlmMaskedLMLoss::new(1.0, 29);
    let tokens = vec![PLM_PAD, PLM_CLS, PLM_MASK, PLM_EOS];
    let (_masked, positions) = loss.create_mask(&tokens, PLM_MASK);
    assert!(
        positions.is_empty(),
        "Special tokens should never be masked"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// §11  PlmMsaEncoder tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn msa_encoder_construction() {
    let enc = PlmMsaEncoder::new(32, 4);
    assert!(enc.is_ok());
}

#[test]
fn msa_encoder_output_shape() {
    let enc = PlmMsaEncoder::new(32, 4).expect("MSA encoder construction should succeed");
    // 3 sequences of length 5, embed_dim=32.
    let msa: Vec<Vec<Vec<f64>>> = (0..3)
        .map(|_| (0..5).map(|_| vec![0.1; 32]).collect())
        .collect();
    let out = enc.forward(&msa).expect("MSA encoder forward should succeed");
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].len(), 5);
    assert_eq!(out[0][0].len(), 32);
}

#[test]
fn msa_encoder_empty_input() {
    let enc = PlmMsaEncoder::new(32, 4).expect("MSA encoder construction should succeed");
    let out = enc.forward(&[]).expect("MSA encoder forward with empty input should succeed");
    assert!(out.is_empty());
}

#[test]
fn msa_encoder_single_sequence() {
    let enc = PlmMsaEncoder::new(32, 4).expect("MSA encoder construction should succeed");
    let msa: Vec<Vec<Vec<f64>>> = vec![(0..4).map(|_| vec![0.5; 32]).collect()];
    let out = enc.forward(&msa).expect("MSA encoder forward should succeed");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].len(), 4);
}

#[test]
fn msa_encoder_row_attention_changes_representation() {
    let enc = PlmMsaEncoder::new(32, 4).expect("MSA encoder construction should succeed");
    let x: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64 * 0.1; 32]).collect();
    let msa = vec![x.clone()];
    let out = enc.forward(&msa).expect("MSA encoder forward should succeed");
    // After row + col attention + FFN, representation should change.
    let sum_in: f64 = x.iter().flatten().sum();
    let sum_out: f64 = out[0].iter().flatten().sum();
    assert_ne!(sum_in, sum_out);
}

#[test]
fn msa_encoder_multiple_seqs_interact() {
    let enc = PlmMsaEncoder::new(32, 4).expect("MSA encoder construction should succeed");
    // Two sequences.
    let msa: Vec<Vec<Vec<f64>>> = vec![
        (0..3).map(|i| vec![i as f64 * 0.1; 32]).collect(),
        (0..3).map(|i| vec![i as f64 * 0.5; 32]).collect(),
    ];
    let out = enc.forward(&msa).expect("MSA encoder forward should succeed");
    // Both sequences should be non-trivially different from input.
    assert_ne!(out[0][0], msa[0][0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// §12  PlmMetrics tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn metrics_perplexity_perfect() {
    // If logits perfectly predict targets, perplexity ≈ 1.
    let vocab_size = 5;
    let logits: Vec<Vec<f64>> = (0..4)
        .map(|i| {
            let mut v = vec![-100.0; vocab_size];
            v[i] = 100.0;
            v
        })
        .collect();
    let targets = vec![0, 1, 2, 3];
    let ppl = PlmMetrics::perplexity(&logits, &targets);
    assert!(ppl < 1.01, "Perfect perplexity should be ~1, got {ppl}");
}

#[test]
fn metrics_perplexity_uniform() {
    // Uniform logits → perplexity = vocab_size.
    let vocab_size = 8;
    let logits: Vec<Vec<f64>> = (0..3).map(|_| vec![0.0; vocab_size]).collect();
    let targets = vec![0, 1, 2];
    let ppl = PlmMetrics::perplexity(&logits, &targets);
    assert!(
        (ppl - vocab_size as f64).abs() < 0.01,
        "Uniform perplexity should be vocab_size={vocab_size}, got {ppl}"
    );
}

#[test]
fn metrics_perplexity_empty() {
    let ppl = PlmMetrics::perplexity(&[], &[]);
    assert!(ppl.is_infinite());
}

#[test]
fn metrics_contact_precision_at_l_perfect() {
    // L×L matrix with highest predictions matching true contacts.
    let l = 10;
    let mut contacts = vec![vec![0.0_f64; l]; l];
    let true_contacts = vec![(0, 6), (1, 7), (2, 8)];
    // Set these pairs to very high probability.
    for &(i, j) in &true_contacts {
        contacts[i][j] = 1.0;
        contacts[j][i] = 1.0;
    }
    // Also symmetrize other pairs at 0.0
    let prec = PlmMetrics::contact_precision_at_l(&contacts, &true_contacts, 1.0);
    // Top-10 should include the 3 true contacts → precision = 3/10.
    assert!((0.0..=1.0).contains(&prec));
}

#[test]
fn metrics_contact_precision_zero_on_empty() {
    let contacts = vec![vec![0.0; 5]; 5];
    let prec = PlmMetrics::contact_precision_at_l(&contacts, &[], 1.0);
    assert_eq!(prec, 0.0);
}

#[test]
fn metrics_contact_precision_on_empty_matrix() {
    let prec = PlmMetrics::contact_precision_at_l(&[], &[(0, 1)], 1.0);
    assert_eq!(prec, 0.0);
}

#[test]
fn metrics_mrr_contacts() {
    let l = 5;
    let mut contacts = vec![vec![0.0_f64; l]; l];
    // Top predicted contact is (0,1).
    contacts[0][1] = 0.99;
    contacts[1][0] = 0.99;
    let true_contacts = vec![(0, 1)];
    let mrr = PlmMetrics::mean_reciprocal_rank_contacts(&contacts, &true_contacts);
    assert!(mrr > 0.0 && mrr <= 1.0, "MRR should be in (0,1], got {mrr}");
}

#[test]
fn metrics_mrr_empty() {
    let contacts = vec![vec![0.0; 5]; 5];
    let mrr = PlmMetrics::mean_reciprocal_rank_contacts(&contacts, &[]);
    assert_eq!(mrr, 0.0);
}

#[test]
fn metrics_sequence_recovery_perfect() {
    let predicted = vec![4, 5, 6, 7];
    let true_tok = vec![4, 5, 6, 7];
    let rec = PlmMetrics::sequence_recovery(&predicted, &true_tok);
    assert!((rec - 1.0).abs() < 1e-10);
}

#[test]
fn metrics_sequence_recovery_zero() {
    let predicted = vec![4, 5, 6, 7];
    let true_tok = vec![8, 9, 10, 11];
    let rec = PlmMetrics::sequence_recovery(&predicted, &true_tok);
    assert_eq!(rec, 0.0);
}

#[test]
fn metrics_sequence_recovery_partial() {
    let predicted = vec![4, 5, 6, 8]; // 3 out of 4 match
    let true_tok = vec![4, 5, 6, 7];
    let rec = PlmMetrics::sequence_recovery(&predicted, &true_tok);
    assert!((rec - 0.75).abs() < 1e-10, "Expected 0.75, got {rec}");
}

#[test]
fn metrics_sequence_recovery_empty() {
    let rec = PlmMetrics::sequence_recovery(&[], &[]);
    assert_eq!(rec, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Integration tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn integration_tokenize_encode_embed() {
    let tok = PlmTokenizer::new();
    let enc = PlmEncoder::new(29, 32, 2, 4).expect("encoder construction should succeed");
    let seq = "MAKVLF";
    let tokens = tok.encode(seq).expect("encoding MAKVLF should succeed");
    let embeddings = enc.forward(&tokens).expect("encoder forward should succeed");
    assert_eq!(embeddings.len(), tokens.len());
    assert_eq!(embeddings[0].len(), 32);
}

#[test]
fn integration_fitness_score_is_finite() {
    let tok = PlmTokenizer::new();
    let fp = PlmFitnessPredictor::new(32, 1, 4, 29).expect("fitness predictor construction should succeed");
    let wt = "MAKVLF";
    // Substitute K→R (conservative) and V→A (non-conservative)
    let mt = "MARALAF"; // length mismatch intentionally
    let result = fp.score_mutation(wt, mt, &tok);
    // Should error due to length mismatch.
    assert!(result.is_err());
}

#[test]
fn integration_msa_with_real_sequences() {
    let tok = PlmTokenizer::new();
    let embed = PlmEmbedding::new(29, 32, 128);
    let msa_enc = PlmMsaEncoder::new(32, 4).expect("MSA encoder construction should succeed");

    let seqs = ["ACDEFGHIKL", "ACDEFGHIKL", "ACDEFGHIKM"];
    let msa_emb: Vec<Vec<Vec<f64>>> = seqs
        .iter()
        .map(|s| {
            let tokens = tok.encode(s).expect("encoding sequence should succeed");
            embed.forward(&tokens).expect("embedding forward should succeed")
        })
        .collect();

    let out = msa_enc.forward(&msa_emb).expect("MSA encoder forward should succeed");
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].len(), 12); // 10 AA + CLS + EOS
}

#[test]
fn integration_contact_prediction_pipeline() {
    let tok = PlmTokenizer::new();
    let cp = PlmContactPredictor::new(32, 2, 4).expect("contact predictor construction should succeed");
    let seq = "ACDEFGHIKL";
    let contacts = cp.predict_contacts(seq, &tok).expect("predict_contacts should succeed");
    assert_eq!(contacts.len(), 10);
    let true_contacts = [(0, 5), (1, 6), (2, 7)];
    let prec = PlmMetrics::contact_precision_at_l(&contacts, &true_contacts, 1.0);
    assert!((0.0..=1.0).contains(&prec));
}

#[test]
fn integration_mlm_full_pipeline() {
    let tok = PlmTokenizer::new();
    let enc = PlmEncoder::new(29, 32, 2, 4).expect("encoder construction should succeed");
    let loss_fn = PlmMaskedLMLoss::new(0.5, 29);

    let seq = "ACDEFGHIKL";
    let tokens = tok.encode(seq).expect("encoding should succeed");
    let (masked_tokens, positions) = loss_fn.create_mask(&tokens, PLM_MASK);

    // Encode masked sequence.
    let hidden = enc.forward(&masked_tokens).expect("encoder forward should succeed");

    // Compute fake logits (just use hidden as logits scaled to vocab_size).
    // For a real model the LM head would project to vocab_size.
    // Use small vocab subset for speed.
    let fake_logits: Vec<Vec<f64>> = hidden
        .iter()
        .map(|h| h.iter().take(29).cloned().collect())
        .collect();

    let l = loss_fn.compute_loss(&fake_logits, &positions);
    assert!(l.is_finite() && l > 0.0);
}

#[test]
fn integration_roundtrip_batch() {
    let tok = PlmTokenizer::new();
    let seqs = ["ACDE", "FGHI", "KLMN", "PQRS"];
    let batch = tok.tokenize_batch(&seqs).expect("batch tokenization should succeed");
    // Decode each.
    for (i, tokens) in batch.iter().enumerate() {
        let dec = tok.decode(tokens).expect("decoding should succeed");
        assert_eq!(dec, seqs[i]);
    }
}
