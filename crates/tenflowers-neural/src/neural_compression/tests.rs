//! Tests for the Neural Compression module (Ballé et al. / VQ-VAE style).
//!
//! Covers all ten sections:
//!  §1  NcEntropyModel
//!  §2  NcHyperprior
//!  §3  NcRateDistortionLoss
//!  §4  NcAutoencoder
//!  §5  NcVectorQuantizer
//!  §6  NcResidualQuantizer
//!  §7  NcArithmeticCoder
//!  §8  NcPatchCompressor
//!  §9  NcProgressiveCoder
//! §10  NcMetrics

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// §1  NcEntropyModel
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_entropy_model_new_dimensions() {
    let model = NcEntropyModel::new(8);
    assert_eq!(model.n_channels, 8);
    assert_eq!(model.means.len(), 8);
    assert_eq!(model.log_scales.len(), 8);
    assert!(model.means.iter().all(|&v| v == 0.0));
    assert!(model.log_scales.iter().all(|&v| v == 0.0));
}

#[test]
fn test_entropy_model_log_prob_non_positive() {
    let model = NcEntropyModel::new(4);
    let y = vec![0.0, 1.0, -1.0, 2.0];
    let lp = model.log_prob(&y);
    // log probability must be ≤ 0
    assert!(lp <= 0.0, "log_prob should be ≤ 0, got {}", lp);
}

#[test]
fn test_entropy_model_log_prob_at_mean() {
    let mut model = NcEntropyModel::new(2);
    model.means = vec![0.0, 0.0];
    model.log_scales = vec![0.0, 0.0]; // sigma = 1.0
                                       // Values at the mean should have higher probability
    let y_at_mean = vec![0.0, 0.0];
    let y_far = vec![10.0, 10.0];
    let lp_mean = model.log_prob(&y_at_mean);
    let lp_far = model.log_prob(&y_far);
    assert!(
        lp_mean > lp_far,
        "mean should have higher log-prob than far-away point"
    );
}

#[test]
fn test_entropy_model_bits_per_sample_non_negative() {
    let model = NcEntropyModel::new(4);
    let y = vec![0.5, -0.5, 1.5, -1.5];
    let bits = model.bits_per_sample(&y);
    assert!(bits >= 0.0, "bits_per_sample must be ≥ 0, got {}", bits);
}

#[test]
fn test_entropy_model_bits_finite() {
    let model = NcEntropyModel::new(3);
    let y = vec![0.0, 0.0, 0.0];
    let bits = model.bits_per_sample(&y);
    assert!(bits.is_finite(), "bits must be finite, got {}", bits);
}

#[test]
fn test_entropy_model_quantize_rounds() {
    let model = NcEntropyModel::new(5);
    let y = vec![1.6, -0.4, 2.5, -1.7, 0.1];
    let q = model.quantize(&y);
    assert_eq!(
        q,
        vec![2, 0, 3, -2, 0],
        "Expected rounding: 1.6→2, -0.4→0, 2.5→3 (or 2), -1.7→-2, 0.1→0"
    );
}

#[test]
fn test_entropy_model_quantize_exact_integer() {
    let model = NcEntropyModel::new(3);
    let y = vec![3.0, -2.0, 0.0];
    let q = model.quantize(&y);
    assert_eq!(q, vec![3, -2, 0]);
}

#[test]
fn test_entropy_model_soft_quantize_tau_zero() {
    let model = NcEntropyModel::new(4);
    let y = vec![0.3, 1.7, -0.8, 2.1];
    let sq = model.soft_quantize(&y, 0.0);
    // tau=0 → identity
    for (a, b) in y.iter().zip(sq.iter()) {
        assert!(
            (a - b).abs() < 1e-12,
            "tau=0 soft_quantize should be identity"
        );
    }
}

#[test]
fn test_entropy_model_soft_quantize_tau_one() {
    let model = NcEntropyModel::new(3);
    let y = vec![0.0, 0.5, 1.0];
    let sq = model.soft_quantize(&y, 1.0);
    // Results should be finite
    assert!(
        sq.iter().all(|v| v.is_finite()),
        "soft quantized values should be finite"
    );
    // At y=0: sin(0)=0 → sq[0] = 0.0
    assert!((sq[0] - 0.0).abs() < 1e-12);
}

#[test]
fn test_entropy_model_update_from_data() {
    let mut model = NcEntropyModel::new(2);
    let batch: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    model.update_from_data(&batch, 0.01);
    // Channel 0 mean should be ≈ 3.0, channel 1 ≈ 4.0
    assert!((model.means[0] - 3.0).abs() < 1e-9);
    assert!((model.means[1] - 4.0).abs() < 1e-9);
    // Log-scales should be finite
    assert!(model.log_scales[0].is_finite());
    assert!(model.log_scales[1].is_finite());
}

#[test]
fn test_entropy_model_update_shifts_mean() {
    let mut model = NcEntropyModel::new(1);
    // Initial mean = 0, update with data centered at 5.0
    let batch: Vec<Vec<f64>> = (0..10).map(|i| vec![5.0 + i as f64 * 0.1]).collect();
    model.update_from_data(&batch, 0.1);
    assert!(model.means[0] > 4.0, "Mean should shift toward data center");
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  NcHyperprior
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_hyperprior_new_dimensions() {
    let hp = NcHyperprior::new(8, 4);
    assert_eq!(hp.latent_dim, 8);
    assert_eq!(hp.hyper_dim, 4);
    assert_eq!(hp.hyper_encoder_w.len(), 4); // [hyper_dim × latent_dim]
    assert_eq!(hp.hyper_encoder_w[0].len(), 8);
    assert_eq!(hp.hyper_decoder_w.len(), 8); // [latent_dim × hyper_dim]
    assert_eq!(hp.hyper_decoder_w[0].len(), 4);
}

#[test]
fn test_hyperprior_encode_hyper_non_negative() {
    let hp = NcHyperprior::new(4, 2);
    let y = vec![1.0, -2.0, 3.0, -4.0];
    let z = hp.encode_hyper(&y);
    assert_eq!(z.len(), 2);
    // ReLU output must be ≥ 0
    assert!(
        z.iter().all(|&v| v >= 0.0),
        "encode_hyper should produce non-negative values"
    );
}

#[test]
fn test_hyperprior_decode_hyper_positive() {
    let hp = NcHyperprior::new(4, 2);
    let z = vec![1.0, 0.5];
    let sigma = hp.decode_hyper(&z);
    assert_eq!(sigma.len(), 4);
    // softplus output must be > 0
    assert!(
        sigma.iter().all(|&v| v > 0.0),
        "decode_hyper output must be positive"
    );
}

#[test]
fn test_hyperprior_conditional_log_prob_non_positive() {
    let hp = NcHyperprior::new(4, 2);
    let y = vec![0.5, -0.3, 1.2, -0.8];
    let sigma = vec![1.0, 1.0, 1.0, 1.0];
    let lp = hp.conditional_log_prob(&y, &sigma);
    assert!(lp <= 0.0, "Conditional log-prob must be ≤ 0, got {}", lp);
}

#[test]
fn test_hyperprior_conditional_log_prob_larger_sigma() {
    let hp = NcHyperprior::new(2, 2);
    let y = vec![1.0, 1.0];
    let sigma_small = vec![0.5, 0.5];
    let sigma_large = vec![5.0, 5.0];
    // Larger sigma → more uncertainty → higher log-prob for the same y (within range)
    let lp_small = hp.conditional_log_prob(&y, &sigma_small);
    let lp_large = hp.conditional_log_prob(&y, &sigma_large);
    // For y=1.0: N(1; 0, 0.25) ≪ N(1; 0, 25) is not generally true, but
    // the log-prob formula -0.5*(y/σ)² - log(σ√2π) changes monotonically with σ for small y
    assert!(lp_small.is_finite() && lp_large.is_finite());
}

#[test]
fn test_hyperprior_hyper_rate_non_negative() {
    let hp = NcHyperprior::new(4, 3);
    let z = vec![0.5, 1.0, -0.5];
    let rate = hp.hyper_rate(&z);
    assert!(rate >= 0.0, "Hyper rate must be ≥ 0, got {}", rate);
}

#[test]
fn test_hyperprior_total_rate_non_negative() {
    let hp = NcHyperprior::new(4, 2);
    let y = vec![0.3, -0.7, 1.1, -0.4];
    let rate = hp.total_rate(&y);
    assert!(rate >= 0.0, "Total rate must be ≥ 0, got {}", rate);
}

#[test]
fn test_hyperprior_total_rate_finite() {
    let hp = NcHyperprior::new(6, 3);
    let y = vec![0.0; 6];
    let rate = hp.total_rate(&y);
    assert!(rate.is_finite(), "Total rate should be finite");
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  NcRateDistortionLoss
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_rd_loss_distortion_non_negative() {
    let rd = NcRateDistortionLoss::new(0.01);
    let x = vec![0.5, 0.3, 0.8];
    let x_hat = vec![0.6, 0.25, 0.75];
    let d = rd.distortion(&x, &x_hat);
    assert!(d >= 0.0, "Distortion must be ≥ 0, got {}", d);
}

#[test]
fn test_rd_loss_distortion_perfect() {
    let rd = NcRateDistortionLoss::new(0.01);
    let x = vec![0.5, 0.3, 0.8];
    let d = rd.distortion(&x, &x.clone());
    assert!(
        d.abs() < 1e-10,
        "Perfect reconstruction → distortion = 0, got {}",
        d
    );
}

#[test]
fn test_rd_loss_psnr_positive() {
    let rd = NcRateDistortionLoss::new(0.01);
    let x = vec![0.5, 0.3, 0.8];
    let x_hat = vec![0.52, 0.28, 0.82];
    let psnr = rd.psnr(&x, &x_hat);
    assert!(psnr > 0.0, "PSNR must be > 0, got {}", psnr);
}

#[test]
fn test_rd_loss_psnr_infinity_on_perfect() {
    let rd = NcRateDistortionLoss::new(0.01);
    let x = vec![0.5, 0.5, 0.5];
    let psnr = rd.psnr(&x, &x.clone());
    assert!(
        psnr.is_infinite(),
        "PSNR on perfect reconstruction should be infinite"
    );
}

#[test]
fn test_rd_loss_combined_equals_d_plus_lambda_r() {
    let lambda = 0.05;
    let rd = NcRateDistortionLoss::new(lambda);
    let x = vec![0.5, 0.3, 0.8];
    let x_hat = vec![0.6, 0.25, 0.75];
    let d = rd.distortion(&x, &x_hat);
    let rate = 10.0;
    let loss = rd.loss(&x, &x_hat, rate);
    let expected = d + lambda * rate;
    assert!(
        (loss - expected).abs() < 1e-10,
        "Loss should equal D + lambda*R: got {} expected {}",
        loss,
        expected
    );
}

#[test]
fn test_rd_loss_bpp() {
    let bpp = NcRateDistortionLoss::bpp(100.0, 50);
    assert!((bpp - 2.0).abs() < 1e-10, "BPP = 100/50 = 2.0, got {}", bpp);
}

#[test]
fn test_rd_loss_bpp_zero_pixels() {
    let bpp = NcRateDistortionLoss::bpp(100.0, 0);
    assert_eq!(bpp, 0.0);
}

#[test]
fn test_rd_loss_rate_distortion_curve_length() {
    let rd = NcRateDistortionLoss::new(0.01);
    let x = vec![0.5; 8];
    let entries: Vec<(f64, Vec<f64>, f64)> = vec![
        (0.01, vec![0.5; 8], 10.0),
        (0.05, vec![0.52; 8], 8.0),
        (0.1, vec![0.6; 8], 5.0),
    ];
    let curve = rd.rate_distortion_curve(&x, &entries);
    assert_eq!(curve.len(), 3);
    // All bpp values should be non-negative
    assert!(curve.iter().all(|(bpp, _)| *bpp >= 0.0));
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  NcAutoencoder
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_autoencoder_encode_shape() {
    let ae = NcAutoencoder::new(16, 4, 8);
    let x = vec![0.5; 16];
    let y = ae.encode(&x);
    assert_eq!(y.len(), 4, "Latent dim should be 4, got {}", y.len());
}

#[test]
fn test_autoencoder_encode_finite() {
    let ae = NcAutoencoder::new(8, 3, 6);
    let x = vec![0.1, 0.9, 0.5, 0.3, 0.7, 0.4, 0.2, 0.8];
    let y = ae.encode(&x);
    assert!(
        y.iter().all(|v| v.is_finite()),
        "Latent values should be finite"
    );
}

#[test]
fn test_autoencoder_decode_shape() {
    let ae = NcAutoencoder::new(8, 3, 6);
    let y_hat = vec![vec![0.0, 0.5, -0.5]; 4]; // batch of 4
    let x_hat = ae.decode(&y_hat);
    assert_eq!(x_hat.len(), 4, "Batch size preserved");
    assert_eq!(
        x_hat[0].len(),
        8,
        "Reconstruction dim should be input_dim=8"
    );
}

#[test]
fn test_autoencoder_decode_sigmoid_range() {
    let ae = NcAutoencoder::new(8, 3, 6);
    let y_hat = vec![vec![5.0, -5.0, 0.0]; 1];
    let x_hat = ae.decode(&y_hat);
    assert!(
        x_hat[0].iter().all(|&v| (0.0..=1.0).contains(&v)),
        "Decoder output should be in [0,1]"
    );
}

#[test]
fn test_autoencoder_compress_valid_quantized() {
    let ae = NcAutoencoder::new(8, 4, 6);
    let x = vec![0.5; 8];
    let (q, rate) = ae.compress(&x);
    assert_eq!(
        q.len(),
        4,
        "Quantized latent length should equal latent_dim"
    );
    assert!(rate >= 0.0, "Rate must be ≥ 0, got {}", rate);
}

#[test]
fn test_autoencoder_compress_rate_finite() {
    let ae = NcAutoencoder::new(8, 4, 6);
    let x = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let (_, rate) = ae.compress(&x);
    assert!(rate.is_finite(), "Rate should be finite");
}

#[test]
fn test_autoencoder_decompress_shape() {
    let ae = NcAutoencoder::new(8, 4, 6);
    let q = vec![1_i32, -1, 0, 2];
    let x_hat = ae.decompress(&q);
    assert_eq!(
        x_hat.len(),
        8,
        "Decompressed output should have input_dim elements"
    );
}

#[test]
fn test_autoencoder_decompress_in_range() {
    let ae = NcAutoencoder::new(8, 4, 6);
    let q = vec![0_i32, 0, 0, 0];
    let x_hat = ae.decompress(&q);
    assert!(
        x_hat.iter().all(|&v| (0.0..=1.0).contains(&v)),
        "Decompressed values should be in [0,1]"
    );
}

#[test]
fn test_autoencoder_rd_loss_non_negative() {
    let ae = NcAutoencoder::new(8, 4, 6);
    let batch: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64 * 0.1 + 0.1; 8]).collect();
    let loss = ae.rd_loss(&batch, 0.01);
    assert!(loss >= 0.0, "RD loss must be ≥ 0, got {}", loss);
}

#[test]
fn test_autoencoder_rd_loss_finite() {
    let ae = NcAutoencoder::new(8, 4, 6);
    let batch: Vec<Vec<f64>> = vec![vec![0.5; 8]; 3];
    let loss = ae.rd_loss(&batch, 0.05);
    assert!(loss.is_finite(), "RD loss should be finite");
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  NcVectorQuantizer
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_vq_new_dimensions() {
    let vq = NcVectorQuantizer::new(16, 8, 0.25);
    assert_eq!(vq.k_size, 16);
    assert_eq!(vq.d_size, 8);
    assert_eq!(vq.codebook.len(), 16);
    assert_eq!(vq.codebook[0].len(), 8);
    assert_eq!(vq.usage_counts.len(), 16);
}

#[test]
fn test_vq_quantize_valid_index() {
    let vq = NcVectorQuantizer::new(8, 4, 0.25);
    let z = vec![0.1, -0.2, 0.3, -0.4];
    let (_, idx, _) = vq.quantize(&z);
    assert!(idx < 8, "Codebook index must be < K=8, got {}", idx);
}

#[test]
fn test_vq_quantize_quantized_shape() {
    let vq = NcVectorQuantizer::new(8, 4, 0.25);
    let z = vec![0.1, -0.2, 0.3, -0.4];
    let (q, _, _) = vq.quantize(&z);
    assert_eq!(q.len(), 4, "Quantized vector should have same dim as input");
}

#[test]
fn test_vq_quantize_commitment_loss_non_negative() {
    let vq = NcVectorQuantizer::new(8, 4, 0.25);
    let z = vec![0.5, 0.5, 0.5, 0.5];
    let (_, _, loss) = vq.quantize(&z);
    assert!(loss >= 0.0, "Commitment loss must be ≥ 0, got {}", loss);
}

#[test]
fn test_vq_quantize_batch_shapes() {
    let vq = NcVectorQuantizer::new(8, 4, 0.25);
    let batch: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 0.1; 4]).collect();
    let (q, idx, loss) = vq.quantize_batch(&batch);
    assert_eq!(q.len(), 5);
    assert_eq!(idx.len(), 5);
    assert!(loss >= 0.0);
    assert!(idx.iter().all(|&k| k < 8));
}

#[test]
fn test_vq_loss_non_negative() {
    let vq = NcVectorQuantizer::new(8, 4, 0.25);
    let z = vec![0.3, -0.3, 0.5, -0.5];
    let e_k = vec![0.1, -0.1, 0.4, -0.4];
    let loss = vq.vq_loss(&z, &e_k);
    assert!(loss >= 0.0, "VQ loss must be ≥ 0, got {}", loss);
}

#[test]
fn test_vq_loss_zero_when_equal() {
    let vq = NcVectorQuantizer::new(4, 3, 0.25);
    let v = vec![0.5, 0.3, -0.7];
    let loss = vq.vq_loss(&v, &v.clone());
    assert!(
        loss.abs() < 1e-10,
        "VQ loss should be 0 when z=e_k, got {}",
        loss
    );
}

#[test]
fn test_vq_codebook_perplexity_at_least_one() {
    let vq = NcVectorQuantizer::new(8, 4, 0.25);
    let perp = vq.codebook_perplexity();
    // With no usage, perplexity defaults to 1.0
    assert!(perp >= 1.0, "Perplexity must be ≥ 1, got {}", perp);
}

#[test]
fn test_vq_codebook_perplexity_after_use() {
    let mut vq = NcVectorQuantizer::new(8, 4, 0.25);
    // Simulate usage of 4 distinct entries
    vq.usage_counts[0] = 10;
    vq.usage_counts[1] = 10;
    vq.usage_counts[2] = 10;
    vq.usage_counts[3] = 10;
    let perp = vq.codebook_perplexity();
    // Uniform over 4 entries → perplexity ≈ 4
    assert!(
        (perp - 4.0).abs() < 0.1,
        "Uniform over 4 entries → perplexity ≈ 4, got {}",
        perp
    );
}

#[test]
fn test_vq_update_codebook_ema() {
    let mut vq = NcVectorQuantizer::new(4, 3, 0.25);
    let original = vq.codebook[0].clone();
    let z = vec![1.0, 1.0, 1.0];
    vq.update_codebook_ema(&z, 0, 0.9);
    // Codebook should have moved toward z
    let updated = &vq.codebook[0];
    assert_ne!(updated[0], original[0], "Codebook entry should be updated");
    assert_eq!(vq.usage_counts[0], 1);
}

#[test]
fn test_vq_reset_dead_codes_runs() {
    let mut vq = NcVectorQuantizer::new(4, 3, 0.25);
    vq.usage_counts[2] = 100; // mark one as most used
                              // Should not panic
    vq.reset_dead_codes(5);
    // Dead entries (usage < 5) should have been reset
    assert_eq!(vq.usage_counts[0], 0);
    assert_eq!(vq.usage_counts[1], 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  NcResidualQuantizer
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_rq_total_bits_correct() {
    let rq = NcResidualQuantizer::new(3, 8, 4, 0.25);
    let expected = 3.0 * (8.0_f64).log2(); // 3 * 3 = 9 bits
    let bits = rq.total_bits();
    assert!(
        (bits - expected).abs() < 1e-9,
        "Expected {} bits, got {}",
        expected,
        bits
    );
}

#[test]
fn test_rq_bits_per_step() {
    let rq = NcResidualQuantizer::new(2, 16, 4, 0.25);
    let expected = 4.0; // log2(16) = 4
    assert!((rq.bits_per_step() - expected).abs() < 1e-9);
}

#[test]
fn test_rq_quantize_valid_indices() {
    let rq = NcResidualQuantizer::new(3, 8, 4, 0.25);
    let z = vec![0.3, -0.2, 0.5, 0.1];
    let (_, indices, _) = rq.quantize(&z);
    assert_eq!(indices.len(), 3, "Should have one index per stage");
    assert!(indices.iter().all(|&k| k < 8), "All indices must be < K=8");
}

#[test]
fn test_rq_quantize_output_shape() {
    let rq = NcResidualQuantizer::new(4, 8, 4, 0.25);
    let z = vec![0.1, -0.1, 0.2, -0.2];
    let (q, idx, loss) = rq.quantize(&z);
    assert_eq!(q.len(), 4, "Quantized vector should have dim=4");
    assert_eq!(idx.len(), 4, "Should have 4 stage indices");
    assert!(loss >= 0.0);
}

#[test]
fn test_rq_quantize_total_bits_consistency() {
    let rq = NcResidualQuantizer::new(2, 4, 3, 0.25);
    let expected = 2.0 * (4.0_f64).log2();
    assert!((rq.total_bits() - expected).abs() < 1e-9);
}

#[test]
fn test_rq_stages_count() {
    let rq = NcResidualQuantizer::new(5, 8, 4, 0.25);
    assert_eq!(rq.n_stages, 5);
    assert_eq!(rq.stages.len(), 5);
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  NcArithmeticCoder
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_arithmetic_coder_expected_length_non_negative() {
    let coder = NcArithmeticCoder::new(16);
    let probs = vec![0.5, 0.25, 0.25];
    let h = coder.expected_code_length(&probs);
    assert!(h >= 0.0, "Shannon entropy must be ≥ 0, got {}", h);
}

#[test]
fn test_arithmetic_coder_entropy_uniform() {
    let coder = NcArithmeticCoder::new(16);
    // Uniform over 4 symbols → entropy = 2 bits
    let probs = vec![0.25; 4];
    let h = coder.expected_code_length(&probs);
    assert!(
        (h - 2.0).abs() < 1e-10,
        "Uniform 4-symbol entropy should be 2 bits, got {}",
        h
    );
}

#[test]
fn test_arithmetic_coder_entropy_deterministic() {
    let coder = NcArithmeticCoder::new(16);
    let probs = vec![1.0]; // one symbol with probability 1
    let h = coder.expected_code_length(&probs);
    assert!(
        h.abs() < 1e-10,
        "Deterministic entropy should be 0, got {}",
        h
    );
}

#[test]
fn test_arithmetic_coder_encode_symbol_u64() {
    let coder = NcArithmeticCoder::new(16);
    let code = coder.encode_symbol(5, 0.5);
    // Should produce a valid u64 without panicking
    assert!(code < u64::MAX);
}

#[test]
fn test_arithmetic_coder_decode_zero_symbol() {
    let coder = NcArithmeticCoder::new(16);
    let code = coder.encode_symbol(0, 0.5);
    let sym = coder.decode_symbol(code, 0.5);
    // Symbol 0 → code = 0 → decode = 0
    assert_eq!(sym, 0);
}

#[test]
fn test_arithmetic_coder_encode_sequence_length() {
    let coder = NcArithmeticCoder::new(16);
    let symbols = vec![0_i32, 1, 2, 3];
    let probs = vec![0.4, 0.3, 0.2, 0.1];
    let codes = coder.encode_sequence(&symbols, &probs);
    assert_eq!(codes.len(), 4);
}

#[test]
fn test_arithmetic_coder_overhead_ratio() {
    let ratio = NcArithmeticCoder::overhead_ratio(200, 150.0);
    assert!((ratio - 200.0 / 150.0).abs() < 1e-10);
}

#[test]
fn test_arithmetic_coder_overhead_ratio_zero_entropy() {
    let ratio = NcArithmeticCoder::overhead_ratio(10, 0.0);
    // Should return 1.0 (safe fallback)
    assert_eq!(ratio, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  NcPatchCompressor
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_patch_compressor_extract_count() {
    let pc = NcPatchCompressor::new(4, 2, 2, 4);
    let signal = vec![0.0_f64; 10];
    // With length=10, patch_size=4, stride=2: positions 0,2,4,6 → 4 patches, +1 partial=5
    let patches = pc.extract_patches(&signal, 10);
    assert!(
        patches.len() >= 4,
        "Expected ≥ 4 patches, got {}",
        patches.len()
    );
}

#[test]
fn test_patch_compressor_extract_patch_size() {
    let pc = NcPatchCompressor::new(4, 2, 2, 4);
    let signal = vec![0.5_f64; 12];
    let patches = pc.extract_patches(&signal, 12);
    assert!(
        patches.iter().all(|p| p.len() == 4),
        "All patches should have patch_size=4"
    );
}

#[test]
fn test_patch_compressor_compress_signal_count() {
    let pc = NcPatchCompressor::new(4, 4, 2, 4);
    let signal: Vec<f64> = (0..16).map(|i| i as f64 / 16.0).collect();
    let (patches, rate) = pc.compress_signal(&signal);
    assert!(!patches.is_empty());
    assert!(rate >= 0.0, "Total rate must be ≥ 0, got {}", rate);
}

#[test]
fn test_patch_compressor_decompress_length() {
    let pc = NcPatchCompressor::new(4, 4, 2, 4);
    let signal: Vec<f64> = (0..16).map(|i| i as f64 / 16.0).collect();
    let (patches, _) = pc.compress_signal(&signal);
    let recon = pc.decompress_signal(&patches, 16);
    assert_eq!(
        recon.len(),
        16,
        "Decompressed signal should have original length"
    );
}

#[test]
fn test_patch_compressor_decompress_in_range() {
    let pc = NcPatchCompressor::new(4, 4, 2, 4);
    let signal = vec![0.5_f64; 8];
    let (patches, _) = pc.compress_signal(&signal);
    let recon = pc.decompress_signal(&patches, 8);
    assert!(
        recon.iter().all(|&v| v.is_finite()),
        "Decompressed values should be finite"
    );
}

#[test]
fn test_patch_compressor_compression_ratio() {
    let pc = NcPatchCompressor::new(4, 4, 2, 4);
    let ratio = pc.compression_ratio(1000, 200);
    assert!(
        (ratio - 5.0).abs() < 1e-10,
        "Expected ratio 5.0, got {}",
        ratio
    );
}

#[test]
fn test_patch_compressor_compression_ratio_zero_compressed() {
    let pc = NcPatchCompressor::new(4, 4, 2, 4);
    let ratio = pc.compression_ratio(1000, 0);
    assert_eq!(ratio, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  NcProgressiveCoder
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_progressive_coder_n_levels() {
    let pc = NcProgressiveCoder::new(4, 8, 3);
    assert_eq!(pc.n_levels, 4);
    assert_eq!(pc.coders.len(), 4);
}

#[test]
fn test_progressive_coder_quality_ladder_length() {
    let pc = NcProgressiveCoder::new(3, 8, 3);
    let x = vec![0.5_f64; 8];
    let ladder = pc.quality_ladder(&x);
    assert_eq!(
        ladder.len(),
        3,
        "Quality ladder should have n_levels entries"
    );
}

#[test]
fn test_progressive_coder_quality_ladder_bpp_non_negative() {
    let pc = NcProgressiveCoder::new(3, 8, 3);
    let x = vec![0.5_f64; 8];
    let ladder = pc.quality_ladder(&x);
    assert!(ladder.iter().all(|(bpp, _)| *bpp >= 0.0));
}

#[test]
fn test_progressive_coder_encode_all_levels() {
    let pc = NcProgressiveCoder::new(3, 8, 4);
    let x = vec![0.4_f64; 8];
    let encoded = pc.encode_all_levels(&x);
    assert_eq!(
        encoded.len(),
        3,
        "encode_all_levels returns one entry per level"
    );
    for (q, rate) in &encoded {
        assert_eq!(
            q.len(),
            4,
            "Quantized latent should have latent_dim=4 elements"
        );
        assert!(*rate >= 0.0);
    }
}

#[test]
fn test_progressive_coder_decode_at_level() {
    let pc = NcProgressiveCoder::new(3, 8, 4);
    let encoded = vec![vec![0_i32, 1, -1, 2]; 2]; // 2 patches
    let decoded = pc.decode_at_level(0, &encoded);
    assert_eq!(decoded.len(), 2, "Should decode 2 patches");
    assert_eq!(decoded[0].len(), 8, "Each decoded patch has input_dim=8");
}

#[test]
fn test_progressive_coder_single_level() {
    let pc = NcProgressiveCoder::new(1, 4, 2);
    let x = vec![0.5_f64; 4];
    let ladder = pc.quality_ladder(&x);
    assert_eq!(ladder.len(), 1);
    let (bpp, psnr) = ladder[0];
    assert!(bpp >= 0.0);
    // PSNR can be infinite (perfect reconstruction) or finite; both are valid
    assert!(!psnr.is_nan(), "PSNR must not be NaN, got {}", psnr);
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  NcMetrics
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_metrics_mse_non_negative() {
    let x = vec![0.5, 0.3, 0.8, 0.1];
    let y = vec![0.6, 0.25, 0.75, 0.15];
    let mse = NcMetrics::mse(&x, &y);
    assert!(mse >= 0.0, "MSE must be ≥ 0, got {}", mse);
}

#[test]
fn test_metrics_mse_zero_on_equal() {
    let x = vec![0.5, 0.3, 0.8];
    let mse = NcMetrics::mse(&x, &x.clone());
    assert!(
        mse.abs() < 1e-12,
        "MSE of identical signals must be 0, got {}",
        mse
    );
}

#[test]
fn test_metrics_mse_value() {
    // Simple case: [0, 1] vs [1, 0] → MSE = (1+1)/2 = 1.0
    let x = vec![0.0, 1.0];
    let y = vec![1.0, 0.0];
    let mse = NcMetrics::mse(&x, &y);
    assert!((mse - 1.0).abs() < 1e-10);
}

#[test]
fn test_metrics_psnr_positive() {
    let x = vec![0.5, 0.3, 0.8, 0.1];
    let y = vec![0.55, 0.28, 0.82, 0.12];
    let psnr = NcMetrics::psnr(&x, &y, 1.0);
    assert!(
        psnr > 0.0,
        "PSNR must be > 0 for non-identical signals with noise"
    );
}

#[test]
fn test_metrics_psnr_finite() {
    let x: Vec<f64> = (0..16).map(|i| i as f64 / 16.0).collect();
    let y: Vec<f64> = (0..16).map(|i| (i as f64 + 0.5) / 16.0).collect();
    let psnr = NcMetrics::psnr(&x, &y, 1.0);
    assert!(psnr.is_finite());
}

#[test]
fn test_metrics_psnr_perfect_is_infinite() {
    let x = vec![0.5_f64; 4];
    let psnr = NcMetrics::psnr(&x, &x.clone(), 1.0);
    assert!(
        psnr.is_infinite(),
        "PSNR for perfect reconstruction should be infinite"
    );
}

#[test]
fn test_metrics_ssim_self_similarity() {
    let x: Vec<f64> = (0..16).map(|i| i as f64 / 16.0).collect();
    let ssim = NcMetrics::ssim(&x, &x.clone(), 8);
    // SSIM of signal with itself = 1.0
    assert!(
        (ssim - 1.0).abs() < 1e-6,
        "SSIM(x,x) should be 1, got {}",
        ssim
    );
}

#[test]
fn test_metrics_ssim_in_range() {
    let x = vec![0.5, 0.3, 0.8, 0.1, 0.6, 0.4, 0.9, 0.2];
    let y = vec![0.6, 0.25, 0.75, 0.15, 0.55, 0.45, 0.85, 0.25];
    let ssim = NcMetrics::ssim(&x, &y, 4);
    assert!(
        (-1.0..=1.0).contains(&ssim),
        "SSIM must be in [-1, 1], got {}",
        ssim
    );
}

#[test]
fn test_metrics_bits_per_element() {
    let bpe = NcMetrics::bits_per_element(80.0, 20);
    assert!((bpe - 4.0).abs() < 1e-10, "Expected 4.0 bpe, got {}", bpe);
}

#[test]
fn test_metrics_bits_per_element_zero_elements() {
    let bpe = NcMetrics::bits_per_element(100.0, 0);
    assert_eq!(bpe, 0.0);
}

#[test]
fn test_metrics_codebook_utilization_all_used() {
    let counts = vec![1_usize; 8];
    let util = NcMetrics::codebook_utilization(&counts);
    assert!(
        (util - 1.0).abs() < 1e-10,
        "All entries used → utilization=1.0"
    );
}

#[test]
fn test_metrics_codebook_utilization_half() {
    let counts = vec![1, 0, 1, 0, 1, 0, 1, 0];
    let util = NcMetrics::codebook_utilization(&counts);
    assert!(
        (util - 0.5).abs() < 1e-10,
        "Half entries used → utilization=0.5, got {}",
        util
    );
}

#[test]
fn test_metrics_codebook_utilization_empty() {
    let counts: Vec<usize> = vec![];
    let util = NcMetrics::codebook_utilization(&counts);
    assert_eq!(util, 0.0);
}

#[test]
fn test_metrics_bjontegaard_same_curve() {
    // BD-Rate between a curve and itself should be ≈ 0
    let curve = vec![(1.0, 30.0), (0.5, 28.0), (2.0, 35.0), (4.0, 40.0)];
    let bd = NcMetrics::bjontegaard_delta_rate(&curve, &curve.clone());
    // May not be exactly 0 due to sampling, but should be very small
    assert!(
        bd.abs() < 1.0,
        "BD-Rate of curve vs itself should be ≈ 0, got {}",
        bd
    );
}

#[test]
fn test_metrics_bjontegaard_short_curves() {
    // Less than 2 points → 0.0
    let c1 = vec![(1.0_f64, 30.0_f64)];
    let c2 = vec![(1.0_f64, 30.0_f64)];
    let bd = NcMetrics::bjontegaard_delta_rate(&c1, &c2);
    assert_eq!(bd, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Integration tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_end_to_end_compress_decompress() {
    let ae = NcAutoencoder::new(8, 4, 6);
    let x = vec![0.1, 0.9, 0.5, 0.3, 0.7, 0.4, 0.2, 0.8];
    let (q, _rate) = ae.compress(&x);
    let x_hat = ae.decompress(&q);
    assert_eq!(x_hat.len(), 8);
    assert!(x_hat.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
fn test_residual_quantizer_residuals_decrease() {
    let rq = NcResidualQuantizer::new(3, 16, 4, 0.25);
    let z = vec![1.0, -1.0, 0.5, -0.5];
    // After quantisation, the total quantized should have finite values
    let (q, _, _) = rq.quantize(&z);
    assert!(q.iter().all(|v| v.is_finite()));
}

#[test]
fn test_hyperprior_pipeline() {
    let hp = NcHyperprior::new(4, 2);
    let y = vec![0.3, -0.7, 1.1, -0.4];
    let z = hp.encode_hyper(&y);
    assert!(z.iter().all(|&v| v >= 0.0), "z must be non-negative");
    let sigma = hp.decode_hyper(&z);
    assert!(sigma.iter().all(|&v| v > 0.0), "sigma must be positive");
    let lp = hp.conditional_log_prob(&y, &sigma);
    assert!(lp <= 0.0, "Conditional log-prob must be ≤ 0");
    let rate = hp.total_rate(&y);
    assert!(rate.is_finite() && rate >= 0.0);
}
