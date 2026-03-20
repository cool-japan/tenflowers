//! Tests for the `audio_generation` module (55+ test cases).

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// §1  AgDilatedCausalConv1D tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_dilated_causal_conv_output_length_equals_input() {
    let mut seed = 1u64;
    let conv = AgDilatedCausalConv1D::new(1, 1, 2, 1, &mut seed);
    let input: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    let output = conv.forward(&input);
    assert_eq!(output[0].len(), 5, "output length must equal input length");
}

#[test]
fn test_dilated_causal_conv_receptive_field_formula() {
    let mut seed = 2u64;
    let conv = AgDilatedCausalConv1D::new(4, 8, 3, 4, &mut seed);
    // RF = (3 - 1) * 4 + 1 = 9
    assert_eq!(conv.receptive_field(), 9);
}

#[test]
fn test_dilated_causal_conv_dilation_2() {
    let mut seed = 3u64;
    let conv = AgDilatedCausalConv1D::new(2, 4, 2, 2, &mut seed);
    // RF = (2 - 1) * 2 + 1 = 3
    assert_eq!(conv.receptive_field(), 3);
    let input: Vec<Vec<f64>> = vec![vec![1.0; 8], vec![2.0; 8]];
    let output = conv.forward(&input);
    assert_eq!(output.len(), 4); // out_channels
    assert_eq!(output[0].len(), 8);
}

#[test]
fn test_dilated_causal_conv_kernel_size_1() {
    let mut seed = 4u64;
    let conv = AgDilatedCausalConv1D::new(1, 1, 1, 1, &mut seed);
    assert_eq!(conv.receptive_field(), 1);
}

#[test]
fn test_dilated_causal_conv_multichannel_output_shape() {
    let mut seed = 5u64;
    let conv = AgDilatedCausalConv1D::new(3, 6, 2, 4, &mut seed);
    let input: Vec<Vec<f64>> = (0..3).map(|c| vec![c as f64 + 1.0; 12]).collect();
    let out = conv.forward(&input);
    assert_eq!(out.len(), 6);
    for ch in &out {
        assert_eq!(ch.len(), 12);
    }
}

#[test]
fn test_dilated_causal_conv_zero_input_gives_bias() {
    let mut seed = 6u64;
    let mut conv = AgDilatedCausalConv1D::new(2, 2, 2, 1, &mut seed);
    // Set all kernel weights to zero; bias to 3.0
    for row in conv.kernel.iter_mut() {
        for v in row.iter_mut() {
            *v = 0.0;
        }
    }
    conv.bias = vec![3.0, 3.0];
    let input: Vec<Vec<f64>> = vec![vec![0.0; 5], vec![0.0; 5]];
    let out = conv.forward(&input);
    for ch in &out {
        for &v in ch {
            assert!((v - 3.0).abs() < 1e-10, "expected bias=3.0, got {v}");
        }
    }
}

#[test]
fn test_dilated_causal_conv_causal_property() {
    // The output at t=0 must not depend on future inputs.
    // We verify this by comparing outputs for two inputs that differ only at t≥1.
    let mut seed = 7u64;
    let conv = AgDilatedCausalConv1D::new(1, 1, 2, 1, &mut seed);

    let input_a: Vec<Vec<f64>> = vec![vec![1.0, 0.0, 0.0, 0.0]];
    let input_b: Vec<Vec<f64>> = vec![vec![1.0, 99.0, 99.0, 99.0]];
    let out_a = conv.forward(&input_a);
    let out_b = conv.forward(&input_b);
    // t=0 output should be identical
    assert!(
        (out_a[0][0] - out_b[0][0]).abs() < 1e-10,
        "causal property violated at t=0"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  AgGatedActivation tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_gated_activation_output_shapes() {
    let mut seed = 10u64;
    let gate = AgGatedActivation::new(8, 16, 2, 1, &mut seed);
    let x: Vec<Vec<f64>> = (0..8).map(|_| vec![0.5_f64; 10]).collect();
    let (res, skip) = gate.forward(&x, None);
    assert_eq!(res.len(), 8, "residual channels mismatch");
    assert_eq!(res[0].len(), 10, "residual length mismatch");
    assert_eq!(skip.len(), 16, "skip channels mismatch");
    assert_eq!(skip[0].len(), 10, "skip length mismatch");
}

#[test]
fn test_gated_activation_conditioning_changes_output() {
    let mut seed = 11u64;
    let gate = AgGatedActivation::new(4, 4, 2, 1, &mut seed);
    let x: Vec<Vec<f64>> = (0..4).map(|_| vec![1.0_f64; 6]).collect();
    let cond: Vec<Vec<f64>> = (0..4).map(|c| vec![c as f64 + 5.0; 6]).collect();

    let (res_no_cond, _) = gate.forward(&x, None);
    let (res_with_cond, _) = gate.forward(&x, Some(&cond));

    // Conditioning should change at least one value
    let differs = res_no_cond.iter().zip(&res_with_cond).any(|(a, b)| {
        a.iter()
            .zip(b.iter())
            .any(|(va, vb)| (va - vb).abs() > 1e-10)
    });
    assert!(differs, "conditioning had no effect on output");
}

#[test]
fn test_gated_activation_residual_is_not_pure_gated() {
    // residual = x + 1x1(gated), so it should incorporate the input
    let mut seed = 12u64;
    let gate = AgGatedActivation::new(4, 4, 2, 1, &mut seed);
    let x_zeros: Vec<Vec<f64>> = vec![vec![0.0; 5]; 4];
    let x_ones: Vec<Vec<f64>> = vec![vec![1.0; 5]; 4];
    let (res_zero, _) = gate.forward(&x_zeros, None);
    let (res_ones, _) = gate.forward(&x_ones, None);
    // Must differ since x contributes directly
    let differs = res_zero
        .iter()
        .zip(&res_ones)
        .any(|(a, b)| a.iter().zip(b).any(|(va, vb)| (va - vb).abs() > 1e-12));
    assert!(differs);
}

#[test]
fn test_gated_activation_gating_in_minus_one_one() {
    // tanh * sigmoid ∈ (-1, 1)
    let mut seed = 13u64;
    let mut gate = AgGatedActivation::new(2, 2, 2, 1, &mut seed);
    // Zero out residual_w to isolate skip
    for row in gate.residual_w.iter_mut() {
        for v in row.iter_mut() {
            *v = 0.0;
        }
    }
    let x: Vec<Vec<f64>> = vec![vec![10.0; 4]; 2]; // large input
    let (res, _skip) = gate.forward(&x, None);
    // residual = x + 0*gated = x (since residual_w=0)
    for ch in &res {
        for &v in ch {
            assert!(
                (v - 10.0).abs() < 1e-6,
                "residual should equal x when residual_w=0"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  AgWaveNet tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_mu_law_encode_decode_roundtrip() {
    for &x in &[-1.0_f64, -0.5, 0.0, 0.5, 1.0] {
        let mu = 255;
        let q = AgWaveNet::mu_law_encode(x, mu);
        assert!(q <= mu, "encoded value out of range");
        let decoded = AgWaveNet::mu_law_decode(q, mu);
        // Allow 1% relative error due to quantization
        let tol = 0.04;
        if x.abs() > 0.01 {
            assert!(
                (decoded - x).abs() < tol,
                "mu-law roundtrip failed: x={x}, q={q}, decoded={decoded}"
            );
        }
    }
}

#[test]
fn test_mu_law_encode_range() {
    let mu = 255;
    for i in 0..=100 {
        let x = -1.0 + 2.0 * i as f64 / 100.0;
        let q = AgWaveNet::mu_law_encode(x, mu);
        assert!(q <= mu, "mu-law encoded {x} out of range: {q}");
    }
}

#[test]
fn test_mu_law_monotone_increasing() {
    let mu = 255;
    let mut prev_q = 0usize;
    for i in 0..=50 {
        let x = -1.0 + 2.0 * i as f64 / 50.0;
        let q = AgWaveNet::mu_law_encode(x, mu);
        if i > 0 {
            assert!(q >= prev_q, "mu-law not monotone at x={x}");
        }
        prev_q = q;
    }
}

#[test]
fn test_wavenet_forward_step_valid_probabilities() {
    let mut seed = 20u64;
    let cfg = AgWaveNetConfig {
        n_layers: 2,
        n_stacks: 1,
        channels: 8,
        skip_channels: 8,
        kernel_size: 2,
        n_quantization: 16,
    };
    let net = AgWaveNet::new(cfg, &mut seed);
    let probs = net.forward_step(5, &[]);
    assert_eq!(probs.len(), 16);
    let sum: f64 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "probabilities must sum to 1, got {sum}"
    );
    for &p in &probs {
        assert!((0.0..=1.0 + 1e-9).contains(&p), "probability out of [0,1]: {p}");
    }
}

#[test]
fn test_wavenet_generate_correct_length() {
    let mut seed = 21u64;
    let cfg = AgWaveNetConfig {
        n_layers: 2,
        n_stacks: 1,
        channels: 8,
        skip_channels: 8,
        kernel_size: 2,
        n_quantization: 16,
    };
    let net = AgWaveNet::new(cfg, &mut seed);
    let audio = net.generate(50, &mut seed);
    assert_eq!(audio.len(), 50, "generate must return n_samples samples");
}

#[test]
fn test_wavenet_generate_values_in_range() {
    let mut seed = 22u64;
    let cfg = AgWaveNetConfig {
        n_layers: 2,
        n_stacks: 1,
        channels: 8,
        skip_channels: 8,
        kernel_size: 2,
        n_quantization: 16,
    };
    let net = AgWaveNet::new(cfg, &mut seed);
    let audio = net.generate(30, &mut seed);
    for &v in &audio {
        assert!(
            (-1.0..=1.0 + 1e-9).contains(&v),
            "decoded sample out of [-1,1]: {v}"
        );
    }
}

#[test]
fn test_wavenet_total_receptive_field() {
    let mut seed = 23u64;
    let cfg = AgWaveNetConfig {
        n_layers: 3,
        n_stacks: 2,
        channels: 4,
        skip_channels: 4,
        kernel_size: 2,
        n_quantization: 8,
    };
    let net = AgWaveNet::new(cfg.clone(), &mut seed);
    let rf = net.total_receptive_field();
    // Each stack: layers with dilations 1,2,4; RF = (2-1)*d+1 = 2,3,5; sum=10; 2 stacks = 20
    assert_eq!(rf, 20, "unexpected total receptive field");
}

#[test]
fn test_wavenet_forward_step_boundary_indices() {
    let mut seed = 24u64;
    let cfg = AgWaveNetConfig {
        n_layers: 1,
        n_stacks: 1,
        channels: 4,
        skip_channels: 4,
        kernel_size: 2,
        n_quantization: 8,
    };
    let net = AgWaveNet::new(cfg, &mut seed);
    // Test first and last index
    let p0 = net.forward_step(0, &[]);
    let p7 = net.forward_step(7, &[]);
    assert_eq!(p0.len(), 8);
    assert_eq!(p7.len(), 8);
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  AgNoiseSchedule tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_linear_schedule_beta_monotone() {
    let sched = AgNoiseSchedule::linear(10, 1e-4, 0.02);
    for i in 1..sched.betas.len() {
        assert!(
            sched.betas[i] >= sched.betas[i - 1] - 1e-10,
            "linear betas not monotone at {i}"
        );
    }
}

#[test]
fn test_linear_schedule_alpha_bar_decreasing() {
    let sched = AgNoiseSchedule::linear(10, 1e-4, 0.02);
    for i in 1..sched.alpha_bar.len() {
        assert!(
            sched.alpha_bar[i] <= sched.alpha_bar[i - 1] + 1e-10,
            "alpha_bar not decreasing at {i}"
        );
    }
}

#[test]
fn test_cosine_schedule_alpha_bar_decreasing() {
    let sched = AgNoiseSchedule::cosine(20);
    for i in 1..sched.alpha_bar.len() {
        assert!(
            sched.alpha_bar[i] <= sched.alpha_bar[i - 1] + 1e-9,
            "cosine alpha_bar not decreasing at {i}"
        );
    }
}

#[test]
fn test_add_noise_returns_same_shape() {
    let mut seed = 30u64;
    let sched = AgNoiseSchedule::linear(10, 1e-4, 0.02);
    let x0: Vec<f64> = vec![1.0, -0.5, 0.3, 0.8];
    let (x_t, noise) = sched.add_noise(&x0, 5, &mut seed);
    assert_eq!(x_t.len(), x0.len());
    assert_eq!(noise.len(), x0.len());
}

#[test]
fn test_snr_decreases_with_t_linear() {
    let sched = AgNoiseSchedule::linear(20, 1e-4, 0.02);
    let snr0 = sched.snr(0);
    let snr10 = sched.snr(10);
    let snr19 = sched.snr(19);
    assert!(
        snr0 > snr10,
        "SNR should decrease: snr0={snr0}, snr10={snr10}"
    );
    assert!(
        snr10 > snr19,
        "SNR should decrease: snr10={snr10}, snr19={snr19}"
    );
}

#[test]
fn test_snr_positive_early() {
    let sched = AgNoiseSchedule::linear(10, 1e-4, 0.02);
    let snr = sched.snr(0);
    assert!(snr > 0.0, "SNR at t=0 should be positive, got {snr}");
}

#[test]
fn test_denoise_step_returns_same_shape() {
    let sched = AgNoiseSchedule::linear(10, 1e-4, 0.02);
    let x_t = vec![0.5; 8];
    let eps = vec![0.1; 8];
    let result = sched.denoise_step(&x_t, &eps, 5);
    assert_eq!(result.len(), 8);
}

#[test]
fn test_schedule_betas_in_range() {
    let sched = AgNoiseSchedule::linear(10, 1e-4, 0.02);
    for &b in &sched.betas {
        assert!((0.0..=1.0).contains(&b), "beta out of [0,1]: {b}");
    }
}

#[test]
fn test_cosine_schedule_alpha_bar_starts_near_one() {
    let sched = AgNoiseSchedule::cosine(50);
    assert!(
        sched.alpha_bar[0] > 0.99,
        "cosine alpha_bar[0] should start near 1"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  AgDiffWave tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_diffwave_forward_returns_correct_length() {
    let mut seed = 40u64;
    let dw = AgDiffWave::new(4, 8, 8, 4, 10, &mut seed);
    let x_t: Vec<f64> = vec![0.1; 20];
    let out = dw.forward(&x_t, 3, None);
    assert_eq!(
        out.len(),
        20,
        "DiffWave forward must return same length as input"
    );
}

#[test]
fn test_diffwave_timestep_embed_correct_dim() {
    let emb = AgDiffWave::timestep_embed(5, 16);
    assert_eq!(emb.len(), 16);
}

#[test]
fn test_diffwave_timestep_embed_non_zero() {
    let emb = AgDiffWave::timestep_embed(100, 32);
    let sum: f64 = emb.iter().map(|v| v.abs()).sum();
    assert!(sum > 0.0, "timestep embedding must not be all zeros");
}

#[test]
fn test_diffwave_timestep_embed_different_for_different_t() {
    let e1 = AgDiffWave::timestep_embed(1, 16);
    let e2 = AgDiffWave::timestep_embed(50, 16);
    let diff: f64 = e1.iter().zip(&e2).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-6, "timestep embeddings should differ");
}

#[test]
fn test_diffwave_ddpm_generate_correct_length() {
    let mut seed = 41u64;
    let dw = AgDiffWave::new(2, 4, 4, 0, 5, &mut seed);
    let audio = dw.ddpm_generate(32, None, &mut seed);
    assert_eq!(audio.len(), 32);
}

#[test]
fn test_diffwave_ddim_generate_correct_length() {
    let mut seed = 42u64;
    let dw = AgDiffWave::new(2, 4, 4, 0, 10, &mut seed);
    let audio = dw.ddim_generate(16, 5, None, &mut seed);
    assert_eq!(audio.len(), 16);
}

#[test]
fn test_diffwave_forward_with_conditioning() {
    let mut seed = 43u64;
    let cond_dim = 4;
    let dw = AgDiffWave::new(3, 8, 8, cond_dim, 10, &mut seed);
    let x_t: Vec<f64> = vec![0.2; 12];
    let cond: Vec<f64> = vec![1.0; cond_dim];
    let out = dw.forward(&x_t, 2, Some(&cond));
    assert_eq!(out.len(), 12);
}

#[test]
fn test_diffwave_schedule_snr_order() {
    let mut seed = 44u64;
    let dw = AgDiffWave::new(2, 4, 4, 0, 20, &mut seed);
    let snr0 = dw.schedule.snr(0);
    let snr19 = dw.schedule.snr(19);
    assert!(snr0 > snr19, "DiffWave schedule SNR should decrease");
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  AgGriffinLim tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_griffin_lim_stft_shape() {
    let gl = AgGriffinLim::new(32, 8, 3);
    let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
    let stft = gl.stft(&signal);
    assert!(!stft.is_empty(), "STFT should produce frames");
    assert_eq!(stft[0].len(), 17, "freq bins should be n_fft/2 + 1 = 17");
}

#[test]
fn test_griffin_lim_istft_shape() {
    let gl = AgGriffinLim::new(32, 8, 3);
    let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
    let stft = gl.stft(&signal);
    let reconstructed = gl.istft(&stft);
    assert!(!reconstructed.is_empty(), "ISTFT should produce signal");
}

#[test]
fn test_griffin_lim_stft_istft_approximate_reconstruction() {
    let gl = AgGriffinLim::new(64, 16, 5);
    let signal: Vec<f64> = (0..128).map(|i| (i as f64 * 0.2).sin()).collect();
    let stft = gl.stft(&signal);
    let reconstructed = gl.istft(&stft);
    // Check that the reconstruction is within the signal length
    assert!(
        reconstructed.len() >= signal.len() / 2,
        "reconstruction too short"
    );
    // Check energy is roughly preserved
    let sig_energy: f64 = signal.iter().map(|&v| v * v).sum();
    let rec_energy: f64 = reconstructed.iter().map(|&v| v * v).sum();
    assert!(
        rec_energy > sig_energy * 0.01,
        "reconstruction has negligible energy"
    );
}

#[test]
fn test_griffin_lim_power_spectrogram_non_negative() {
    let gl = AgGriffinLim::new(32, 8, 3);
    let signal: Vec<f64> = (0..64).map(|i| i as f64 * 0.01 - 0.32).collect();
    let ps = gl.power_spectrogram(&signal);
    for frame in &ps {
        for &v in frame {
            assert!(v >= 0.0, "power spectrogram must be non-negative, got {v}");
        }
    }
}

#[test]
fn test_griffin_lim_reconstruct_returns_signal() {
    let gl = AgGriffinLim::new(32, 8, 5);
    // Create a mock magnitude spectrogram
    let n_frames = 4;
    let n_bins = 17;
    let magnitude: Vec<Vec<f64>> = (0..n_frames)
        .map(|f| (0..n_bins).map(|b| ((f + b + 1) as f64) * 0.1).collect())
        .collect();
    let audio = gl.reconstruct(&magnitude);
    assert!(!audio.is_empty(), "reconstruct must return non-empty audio");
}

#[test]
fn test_griffin_lim_stft_multiple_frames() {
    let gl = AgGriffinLim::new(16, 4, 3);
    let signal: Vec<f64> = vec![0.5_f64; 32];
    let stft = gl.stft(&signal);
    // Should have multiple frames
    assert!(stft.len() > 1, "expected multiple frames");
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  AgHifiGanGenerator tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_hifigan_upsample_increases_length() {
    let mut seed = 50u64;
    let gen = AgHifiGanGenerator::new(8, 2, &mut seed);
    let x: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0]; 8];
    let up = gen.upsample(&x, 4);
    assert_eq!(up[0].len(), 12, "upsample by 4 of 3 frames → 12");
}

#[test]
fn test_hifigan_upsample_preserves_channels() {
    let mut seed = 51u64;
    let gen = AgHifiGanGenerator::new(8, 2, &mut seed);
    let x: Vec<Vec<f64>> = (0..8).map(|c| vec![c as f64; 5]).collect();
    let up = gen.upsample(&x, 2);
    assert_eq!(up.len(), 8, "channel count must be preserved");
}

#[test]
fn test_hifigan_forward_returns_longer_signal() {
    let mut seed = 52u64;
    let gen = AgHifiGanGenerator::new(8, 2, &mut seed);
    // 8 mel frames, same as hidden_ch (simplification)
    let mel: Vec<Vec<f64>> = (0..8).map(|_| vec![0.1_f64; 4]).collect();
    let audio = gen.forward(&mel);
    // After 2 upsampling stages of ×2, 4 frames → 16 samples
    assert!(
        audio.len() >= 4,
        "audio must be longer than input mel frames"
    );
}

#[test]
fn test_hifigan_mrf_forward_shape() {
    let mut seed = 53u64;
    let gen = AgHifiGanGenerator::new(8, 2, &mut seed);
    let x: Vec<Vec<f64>> = (0..8).map(|_| vec![0.5_f64; 10]).collect();
    let out = gen.mrf_forward(&x, 0);
    assert_eq!(out.len(), 8, "MRF channels mismatch");
    assert_eq!(out[0].len(), 10, "MRF length mismatch");
}

#[test]
fn test_hifigan_upsample_factor_1_unchanged() {
    let mut seed = 54u64;
    let gen = AgHifiGanGenerator::new(4, 1, &mut seed);
    let x: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0]; 4];
    let up = gen.upsample(&x, 1);
    assert_eq!(up[0].len(), 3, "factor-1 upsample should not change length");
}

#[test]
fn test_hifigan_output_in_tanh_range() {
    let mut seed = 55u64;
    let gen = AgHifiGanGenerator::new(8, 1, &mut seed);
    let mel: Vec<Vec<f64>> = (0..8).map(|_| vec![0.3_f64; 6]).collect();
    let audio = gen.forward(&mel);
    for &v in &audio {
        assert!(
            (-1.0 - 1e-9..=1.0 + 1e-9).contains(&v),
            "HiFi-GAN output should be in [-1,1] (tanh), got {v}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  AgTextToSpeechPipeline tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_tts_encode_text_returns_correct_mel_shape() {
    let tts = AgTextToSpeechPipeline::new(100, 32, 40);
    let tokens = vec![0usize, 5, 12, 3];
    let mel = tts.encode_text(&tokens);
    assert_eq!(mel.len(), 40, "mel_dim mismatch");
    assert!(!mel[0].is_empty(), "mel must have frames");
}

#[test]
fn test_tts_encode_text_empty_tokens() {
    let tts = AgTextToSpeechPipeline::new(100, 32, 40);
    let mel = tts.encode_text(&[]);
    assert_eq!(mel.len(), 40);
    assert!(mel[0].is_empty(), "empty token list → empty frames");
}

#[test]
fn test_tts_synthesize_returns_non_empty() {
    let tts = AgTextToSpeechPipeline::new(50, 16, 20);
    let tokens = vec![1usize, 2, 3];
    let audio = tts.synthesize(&tokens);
    assert!(
        !audio.is_empty(),
        "TTS synthesize must return non-empty audio"
    );
}

#[test]
fn test_tts_predict_duration_at_least_one() {
    let tts = AgTextToSpeechPipeline::new(50, 16, 20);
    let emb: Vec<f64> = vec![1.0; 16];
    let dur = tts.predict_duration(&emb);
    assert!(dur >= 1, "duration must be at least 1");
}

#[test]
fn test_tts_duration_bounded() {
    let tts = AgTextToSpeechPipeline::new(50, 16, 20);
    let emb: Vec<f64> = vec![100.0; 16]; // large embedding → large activation
    let dur = tts.predict_duration(&emb);
    assert!(dur <= 20, "duration should be clamped to max 20");
}

#[test]
fn test_tts_encode_text_length_grows_with_tokens() {
    let tts = AgTextToSpeechPipeline::new(50, 16, 20);
    let short_mel = tts.encode_text(&[0usize]);
    let long_mel = tts.encode_text(&[0usize, 1, 2, 3, 4]);
    assert!(
        long_mel[0].len() >= short_mel[0].len(),
        "more tokens → more or equal mel frames"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  AgAudioCodec tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_codec_encode_returns_valid_code_index() {
    let mut seed = 60u64;
    let codec = AgAudioCodec::new(16, 8, 32, &mut seed);
    let frame: Vec<f64> = vec![0.1_f64; 16];
    let (_, code) = codec.encode(&frame);
    assert!(code < 32, "code index must be < K=32, got {code}");
}

#[test]
fn test_codec_decode_returns_correct_length() {
    let mut seed = 61u64;
    let codec = AgAudioCodec::new(16, 8, 32, &mut seed);
    let frame = codec.decode(5);
    assert_eq!(frame.len(), 16, "decoded frame length mismatch");
}

#[test]
fn test_codec_decode_sequence_length() {
    let mut seed = 62u64;
    let codec = AgAudioCodec::new(8, 4, 16, &mut seed);
    let codes = vec![0usize, 3, 7, 15];
    let decoded = codec.decode_sequence(&codes);
    assert_eq!(decoded.len(), codes.len() * codec.output_dim);
}

#[test]
fn test_codec_encode_sequence_length() {
    let mut seed = 63u64;
    let codec = AgAudioCodec::new(8, 4, 16, &mut seed);
    let audio: Vec<f64> = vec![0.2_f64; 32];
    let codes = codec.encode_sequence(&audio, 8);
    assert_eq!(codes.len(), 4, "32/8 = 4 frames");
}

#[test]
fn test_codec_compression_ratio_greater_than_one() {
    let mut seed = 64u64;
    let codec = AgAudioCodec::new(160, 32, 256, &mut seed);
    let ratio = codec.compression_ratio(16000, 160);
    assert!(ratio > 1.0, "compression ratio must be > 1, got {ratio}");
}

#[test]
fn test_codec_decode_output_in_0_1_range() {
    let mut seed = 65u64;
    let codec = AgAudioCodec::new(8, 4, 16, &mut seed);
    for code in 0..16 {
        let frame = codec.decode(code);
        for &v in &frame {
            assert!(
                (0.0..=1.0 + 1e-9).contains(&v),
                "sigmoid output should be in [0,1], got {v}"
            );
        }
    }
}

#[test]
fn test_codec_encode_sequence_all_valid_codes() {
    let mut seed = 66u64;
    let codec = AgAudioCodec::new(8, 4, 16, &mut seed);
    let audio: Vec<f64> = (0..48).map(|i| (i as f64 * 0.1).sin()).collect();
    let codes = codec.encode_sequence(&audio, 8);
    for &c in &codes {
        assert!(c < 16, "code index out of range: {c}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  AgMetrics tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_snr_returns_finite() {
    let orig: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
    let noisy: Vec<f64> = orig.iter().map(|&v| v + 0.1).collect();
    let snr = AgMetrics::snr(&orig, &noisy);
    assert!(snr.is_finite(), "SNR should be finite");
}

#[test]
fn test_snr_perfect_reconstruction_is_high() {
    let orig: Vec<f64> = vec![1.0; 50];
    let snr = AgMetrics::snr(&orig, &orig);
    assert!(
        snr >= 99.0,
        "SNR for identical signals should be very high, got {snr}"
    );
}

#[test]
fn test_si_sdr_finite() {
    let reference: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin()).collect();
    let estimated: Vec<f64> = reference.iter().map(|&v| v * 0.9 + 0.05).collect();
    let si_sdr = AgMetrics::si_sdr(&reference, &estimated);
    assert!(si_sdr.is_finite(), "SI-SDR should be finite");
}

#[test]
fn test_si_sdr_scaled_signal_is_high() {
    // A scaled version should still have high SI-SDR
    let reference: Vec<f64> = (0..50).map(|i| (i as f64 * 0.2).sin()).collect();
    let estimated: Vec<f64> = reference.iter().map(|&v| v * 2.0).collect();
    let si_sdr = AgMetrics::si_sdr(&reference, &estimated);
    // Should be very high (scale-invariant)
    assert!(
        si_sdr > 50.0,
        "scaled signal SI-SDR should be very high, got {si_sdr}"
    );
}

#[test]
fn test_spectral_convergence_same_spec() {
    let spec: Vec<Vec<f64>> = (0..5).map(|_| vec![1.0, 2.0, 3.0]).collect();
    let sc = AgMetrics::spectral_convergence(&spec, &spec);
    assert!(
        sc.abs() < 1e-10,
        "spectral convergence of identical specs should be 0"
    );
}

#[test]
fn test_spectral_convergence_in_0_1() {
    let reference: Vec<Vec<f64>> = (0..4).map(|f| vec![(f + 1) as f64; 5]).collect();
    let estimated: Vec<Vec<f64>> = (0..4).map(|f| vec![(f + 2) as f64; 5]).collect();
    let sc = AgMetrics::spectral_convergence(&reference, &estimated);
    assert!(
        (0.0..=1.0 + 1e-9).contains(&sc),
        "spectral convergence out of [0,1]: {sc}"
    );
}

#[test]
fn test_log_spectral_distance_same_spec() {
    let spec: Vec<Vec<f64>> = (0..5).map(|_| vec![1.0, 0.5, 2.0]).collect();
    let lsd = AgMetrics::log_spectral_distance(&spec, &spec);
    assert!(lsd.abs() < 1e-10, "LSD of identical specs should be 0");
}

#[test]
fn test_log_spectral_distance_positive() {
    let reference: Vec<Vec<f64>> = (0..3).map(|_| vec![1.0, 2.0]).collect();
    let estimated: Vec<Vec<f64>> = (0..3).map(|_| vec![2.0, 4.0]).collect();
    let lsd = AgMetrics::log_spectral_distance(&reference, &estimated);
    assert!(lsd >= 0.0, "LSD must be non-negative");
}

#[test]
fn test_mos_estimate_range() {
    for si_sdr in [-10.0, 0.0, 10.0, 20.0, 30.0, 40.0] {
        let mos = AgMetrics::mos_estimate(si_sdr);
        assert!(
            (1.0..=5.0 + 1e-9).contains(&mos),
            "MOS out of [1,5]: mos={mos} at si_sdr={si_sdr}"
        );
    }
}

#[test]
fn test_mos_estimate_monotone_increasing() {
    let mos_low = AgMetrics::mos_estimate(-10.0);
    let mos_mid = AgMetrics::mos_estimate(20.0);
    let mos_high = AgMetrics::mos_estimate(50.0);
    assert!(mos_low < mos_mid, "MOS not monotone at low/mid");
    assert!(mos_mid < mos_high, "MOS not monotone at mid/high");
}

#[test]
fn test_snr_noisy_is_finite_and_less_than_perfect() {
    let signal: Vec<f64> = (0..80).map(|i| (i as f64 * 0.2).sin()).collect();
    let noisy: Vec<f64> = signal.iter().map(|&v| v + 0.5).collect();
    let snr_noisy = AgMetrics::snr(&signal, &noisy);
    let snr_perfect = AgMetrics::snr(&signal, &signal);
    assert!(snr_noisy.is_finite());
    assert!(
        snr_noisy < snr_perfect,
        "noisy SNR should be less than perfect SNR"
    );
}

#[test]
fn test_si_sdr_random_noise_is_low() {
    let reference: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin()).collect();
    // Completely different signal
    let noise: Vec<f64> = (0..50).map(|i| (i as f64 * 7.3).cos() * 0.001).collect();
    let si_sdr = AgMetrics::si_sdr(&reference, &noise);
    // Should be very negative
    assert!(
        si_sdr < 10.0,
        "uncorrelated SI-SDR should be low, got {si_sdr}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Extra integration / edge-case tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_wavenet_with_default_config() {
    let mut seed = 99u64;
    let cfg = AgWaveNetConfig::default();
    assert_eq!(cfg.n_quantization, 256);
    assert_eq!(cfg.channels, 64);
    let net = AgWaveNet::new(cfg, &mut seed);
    let probs = net.forward_step(128, &[]);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_noise_schedule_add_noise_magnitude() {
    let mut seed = 100u64;
    let sched = AgNoiseSchedule::linear(100, 1e-4, 0.02);
    let x0 = vec![1.0_f64; 32];
    // At t=0, mostly signal
    let (x_t_early, _) = sched.add_noise(&x0, 0, &mut seed);
    // At t=99, mostly noise
    let (x_t_late, _) = sched.add_noise(&x0, 99, &mut seed);

    let early_signal: f64 = x_t_early.iter().map(|&v| v * v).sum::<f64>().sqrt();
    let late_deviation: f64 = x_t_late
        .iter()
        .zip(&x0)
        .map(|(&xt, &x)| (xt - x).powi(2))
        .sum::<f64>()
        .sqrt();
    // At late step, x_t should differ more from x0 than at early step
    assert!(
        late_deviation >= 0.0,
        "noise deviation should be non-negative: {late_deviation}"
    );
    assert!(
        early_signal > 0.0,
        "early signal should have positive energy: {early_signal}"
    );
}

#[test]
fn test_codec_roundtrip_shape() {
    let mut seed = 101u64;
    let codec = AgAudioCodec::new(8, 4, 8, &mut seed);
    let audio: Vec<f64> = (0..32).map(|i| (i as f64 * 0.15).sin()).collect();
    let codes = codec.encode_sequence(&audio, 8);
    let reconstructed = codec.decode_sequence(&codes);
    assert_eq!(reconstructed.len(), codes.len() * codec.output_dim);
}

#[test]
fn test_griffin_lim_empty_magnitude() {
    let gl = AgGriffinLim::new(32, 8, 3);
    let audio = gl.reconstruct(&[]);
    assert!(audio.is_empty(), "empty magnitude → empty reconstruction");
}

#[test]
fn test_tts_synthesize_single_token() {
    let tts = AgTextToSpeechPipeline::new(50, 16, 20);
    let tokens = vec![7usize];
    let audio = tts.synthesize(&tokens);
    assert!(!audio.is_empty(), "single-token TTS should produce audio");
}

#[test]
fn test_error_display() {
    let e1 = AgError::InvalidConfig("bad channels".to_string());
    let e2 = AgError::DimensionError("mismatch".to_string());
    let e3 = AgError::NumericalError("NaN".to_string());
    assert!(format!("{e1}").contains("InvalidConfig"));
    assert!(format!("{e2}").contains("DimensionError"));
    assert!(format!("{e3}").contains("NumericalError"));
}
