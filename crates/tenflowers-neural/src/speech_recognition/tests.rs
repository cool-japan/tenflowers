use super::*;

// ──────────────────────────────────────────────────────────────
// log_mel_spectrogram
// ──────────────────────────────────────────────────────────────
#[test]
fn test_log_mel_basic_shape() {
    let audio: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin()).collect();
    let mel = log_mel_spectrogram(&audio, 16000, 400, 160, 80);
    assert!(!mel.is_empty(), "mel should not be empty");
    assert_eq!(mel[0].len(), 80, "n_mels should be 80");
}

#[test]
fn test_log_mel_empty_audio() {
    let mel = log_mel_spectrogram(&[], 16000, 400, 160, 80);
    assert!(mel.is_empty());
}

#[test]
fn test_log_mel_values_finite() {
    let audio: Vec<f32> = (0..3200).map(|i| (i as f32 * 0.001).sin() * 0.5).collect();
    let mel = log_mel_spectrogram(&audio, 16000, 400, 160, 80);
    for frame in &mel {
        for &v in frame {
            assert!(v.is_finite(), "mel value should be finite, got {}", v);
        }
    }
}

#[test]
fn test_hz_mel_roundtrip() {
    let hz = 440.0f32;
    let mel = hz_to_mel(hz);
    let hz2 = mel_to_hz(mel);
    assert!(
        (hz - hz2).abs() < 1.0,
        "mel roundtrip error: {} vs {}",
        hz,
        hz2
    );
}

// ──────────────────────────────────────────────────────────────
// AudioConvStem
// ──────────────────────────────────────────────────────────────
#[test]
fn test_audio_conv_stem_shape() {
    let stem = AudioConvStem::new(80, 384, 3, 42);
    let input: Vec<Vec<f32>> = (0..50).map(|_| vec![0.1f32; 80]).collect();
    let out = stem.forward(&input);
    assert_eq!(out.len(), 50);
    assert_eq!(out[0].len(), 384);
}

#[test]
fn test_audio_conv_stem_empty() {
    let stem = AudioConvStem::new(80, 384, 3, 1);
    let out = stem.forward(&[]);
    assert!(out.is_empty());
}

// ──────────────────────────────────────────────────────────────
// WhisperEncoder
// ──────────────────────────────────────────────────────────────
#[test]
fn test_whisper_encoder_output_shape() {
    let cfg = WhisperConfig {
        n_mels: 80,
        n_fft: 400,
        hop_length: 160,
        n_audio_ctx: 50,
        n_audio_state: 64,
        n_audio_heads: 4,
        n_audio_layers: 2,
    };
    let enc = WhisperEncoder::new(cfg, 42);
    let log_mel: Vec<Vec<f32>> = (0..30).map(|_| vec![0.0f32; 80]).collect();
    let out = enc.encode(&log_mel);
    assert_eq!(out.len(), 30);
    assert_eq!(out[0].len(), 64);
}

#[test]
fn test_whisper_encoder_values_finite() {
    let cfg = WhisperConfig {
        n_mels: 80,
        n_fft: 400,
        hop_length: 160,
        n_audio_ctx: 20,
        n_audio_state: 32,
        n_audio_heads: 2,
        n_audio_layers: 1,
    };
    let enc = WhisperEncoder::new(cfg, 99);
    let log_mel: Vec<Vec<f32>> = (0..10)
        .map(|i| (0..80).map(|j| (i * j) as f32 * 0.01).collect())
        .collect();
    let out = enc.encode(&log_mel);
    for frame in &out {
        for &v in frame {
            assert!(v.is_finite(), "encoder output must be finite");
        }
    }
}

// ──────────────────────────────────────────────────────────────
// WhisperDecoder
// ──────────────────────────────────────────────────────────────
#[test]
fn test_whisper_decoder_logit_shape() {
    let dec = WhisperDecoder::new(100, 32, 2, 2, 50, 7);
    let enc_out: Vec<Vec<f32>> = (0..10).map(|_| vec![0.1f32; 32]).collect();
    let logits = dec.decode_step(&[1, 2, 3], &enc_out);
    assert_eq!(logits.len(), 100);
}

#[test]
fn test_whisper_decoder_greedy() {
    let dec = WhisperDecoder::new(20, 32, 2, 2, 50, 7);
    let enc_out: Vec<Vec<f32>> = (0..5).map(|_| vec![0.0f32; 32]).collect();
    let tokens = dec.greedy_decode(&enc_out, 1, 2, 10);
    assert!(
        !tokens.is_empty(),
        "greedy decode should return at least BOS"
    );
}

#[test]
fn test_whisper_decoder_empty_tokens() {
    let dec = WhisperDecoder::new(20, 32, 2, 1, 50, 3);
    let enc_out: Vec<Vec<f32>> = (0..3).map(|_| vec![0.0f32; 32]).collect();
    let logits = dec.decode_step(&[], &enc_out);
    assert_eq!(
        logits.len(),
        20,
        "empty tokens should still return vocab-size logits"
    );
}

// ──────────────────────────────────────────────────────────────
// CtcBeamDecoder
// ──────────────────────────────────────────────────────────────
#[test]
fn test_ctc_greedy_collapse_repeats() {
    let cfg = CtcConfig {
        vocab_size: 5,
        blank_id: 0,
        beam_width: 3,
    };
    let dec = CtcBeamDecoder::new(cfg);
    // blank=0, tokens 1,1,2,0,2  →  [1,2,2]  collapse repeats and blanks
    let log_probs = vec![
        vec![-10.0, 0.0, -10.0, -10.0, -10.0], // 1
        vec![-10.0, 0.0, -10.0, -10.0, -10.0], // 1 (repeat)
        vec![-10.0, -10.0, 0.0, -10.0, -10.0], // 2
        vec![0.0, -10.0, -10.0, -10.0, -10.0], // blank
        vec![-10.0, -10.0, 0.0, -10.0, -10.0], // 2
    ];
    let result = dec.greedy_decode(&log_probs);
    assert_eq!(result, vec![1, 2, 2], "collapsed: {:?}", result);
}

#[test]
fn test_ctc_beam_search_nonempty() {
    let cfg = CtcConfig {
        vocab_size: 5,
        blank_id: 0,
        beam_width: 3,
    };
    let dec = CtcBeamDecoder::new(cfg);
    let log_probs: Vec<Vec<f32>> = (0..5)
        .map(|t| {
            let mut lp = vec![-5.0f32; 5];
            lp[(t % 4) + 1] = 0.0; // always non-blank
            lp
        })
        .collect();
    let result = dec.beam_search(&log_probs, 3);
    assert!(!result.is_empty(), "beam search should produce tokens");
}

#[test]
fn test_ctc_beam_search_empty_input() {
    let cfg = CtcConfig::default();
    let dec = CtcBeamDecoder::new(cfg);
    let result = dec.beam_search(&[], 5);
    assert!(result.is_empty());
}

#[test]
fn test_ctc_greedy_all_blank() {
    let cfg = CtcConfig {
        vocab_size: 3,
        blank_id: 0,
        beam_width: 2,
    };
    let dec = CtcBeamDecoder::new(cfg);
    let log_probs = vec![vec![0.0, -10.0, -10.0], vec![0.0, -10.0, -10.0]];
    let result = dec.greedy_decode(&log_probs);
    assert!(result.is_empty(), "all blanks should give empty output");
}

// ──────────────────────────────────────────────────────────────
// RnntDecoder
// ──────────────────────────────────────────────────────────────
#[test]
fn test_rnnt_prediction_step_shape() {
    let pred = PredictionNetwork::new(10, 8, 16, 5);
    let h = vec![0.0f32; 16];
    let c = vec![0.0f32; 16];
    let (h_new, c_new) = pred.step(3, &h, &c);
    assert_eq!(h_new.len(), 16);
    assert_eq!(c_new.len(), 16);
}

#[test]
fn test_rnnt_joint_shape() {
    let joint = JointNetwork::new(32, 16, 64, 10, 42);
    let enc = vec![0.1f32; 32];
    let pred = vec![0.1f32; 16];
    let logits = joint.joint(&enc, &pred);
    assert_eq!(logits.len(), 10);
}

#[test]
fn test_rnnt_greedy_decode() {
    let decoder = RnntDecoder::new(10, 8, 16, 32, 64, 0, 42);
    let enc_outs: Vec<Vec<f32>> = (0..5).map(|_| vec![0.0f32; 32]).collect();
    let tokens = decoder.rnnt_greedy(&enc_outs);
    // Just verify it runs without panic and returns valid token ids
    for &t in &tokens {
        assert!(t < 10, "token {} out of range", t);
    }
}

// ──────────────────────────────────────────────────────────────
// NgramLm / LmRescorer
// ──────────────────────────────────────────────────────────────
#[test]
fn test_ngram_lm_add_and_prob() {
    let mut lm = NgramLm::new(2, 10);
    lm.add_sentence(&[1, 2, 3, 2, 1]);
    let lp = lm.log_prob(&[1], 2);
    assert!(lp < 0.0, "log-prob should be negative");
    assert!(lp.is_finite(), "log-prob should be finite");
}

#[test]
fn test_ngram_lm_unseen_context() {
    let lm = NgramLm::new(2, 10);
    let lp = lm.log_prob(&[99], 5);
    assert!(lp < 0.0 && lp.is_finite());
}

#[test]
fn test_shallow_fusion() {
    let asr = vec![-1.0f32, -2.0, -3.0];
    let lm = vec![-0.5f32, -1.5, -2.5];
    let fused = LmRescorer::shallow_fusion(&asr, &lm, 0.3);
    assert_eq!(fused.len(), 3);
    assert!((fused[0] - (-1.0 - 0.15)).abs() < 1e-5);
}

#[test]
fn test_ngram_lm_trigram() {
    let mut lm = NgramLm::new(3, 5);
    lm.add_sentence(&[0, 1, 2, 1, 2, 3]);
    let lp = lm.log_prob(&[0, 1], 2);
    assert!(lp.is_finite());
    // After seeing (0,1)->2 twice, prob should be non-trivial
    let lp2 = lm.log_prob(&[1, 2], 1);
    assert!(lp2.is_finite());
}

// ──────────────────────────────────────────────────────────────
// VoiceActivityDetector
// ──────────────────────────────────────────────────────────────
#[test]
fn test_vad_silence() {
    let cfg = VadConfig {
        frame_size: 160,
        hop_size: 80,
        energy_threshold: 0.1,
        min_speech_frames: 2,
        min_silence_frames: 2,
    };
    let vad = VoiceActivityDetector::new(cfg);
    let silence = vec![0.0f32; 1600];
    let frames = vad.detect(&silence);
    assert!(!frames.is_empty());
    assert!(frames.iter().all(|&s| !s), "all frames should be silence");
}

#[test]
fn test_vad_speech_detection() {
    let cfg = VadConfig {
        frame_size: 160,
        hop_size: 80,
        energy_threshold: 0.001,
        min_speech_frames: 1,
        min_silence_frames: 1,
    };
    let vad = VoiceActivityDetector::new(cfg);
    let speech: Vec<f32> = (0..3200).map(|i| (i as f32 * 0.01).sin()).collect();
    let frames = vad.detect(&speech);
    assert!(frames.iter().any(|&s| s), "should detect speech");
}

#[test]
fn test_vad_segment_output() {
    let cfg = VadConfig {
        frame_size: 160,
        hop_size: 80,
        energy_threshold: 0.001,
        min_speech_frames: 1,
        min_silence_frames: 1,
    };
    let vad = VoiceActivityDetector::new(cfg);
    let audio: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.02).sin()).collect();
    let segs = vad.segment(&audio);
    // Segments should have valid bounds
    for &(s, e) in &segs {
        assert!(s < e, "segment start must be before end");
        assert!(e <= audio.len(), "segment end must be within audio");
    }
}

#[test]
fn test_vad_empty() {
    let vad = VoiceActivityDetector::new(VadConfig::default());
    assert!(vad.detect(&[]).is_empty());
    assert!(vad.segment(&[]).is_empty());
}

// ──────────────────────────────────────────────────────────────
// SpeakerDiarizer
// ──────────────────────────────────────────────────────────────
#[test]
fn test_spectral_clustering_basic() {
    let sc = SpectralClustering::new(2);
    let embs: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.9, 0.1, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![0.1, 0.0, 0.9],
    ];
    let labels = sc.cluster_embeddings(&embs);
    assert_eq!(labels.len(), 4);
    // First two should be together, last two together
    assert_eq!(labels[0], labels[1], "similar embs should cluster together");
    assert_eq!(labels[2], labels[3], "similar embs should cluster together");
    assert_ne!(
        labels[0], labels[2],
        "different embs should be different clusters"
    );
}

#[test]
fn test_embed_frames_shape() {
    let sc = SpectralClustering::new(2);
    let frames: Vec<Vec<f32>> = (0..100).map(|_| vec![0.1f32; 80]).collect();
    let embs = sc.embed_frames(&frames);
    assert!(!embs.is_empty());
    assert_eq!(embs[0].len(), 80);
}

#[test]
fn test_diarizer_output_length() {
    let diarizer = SpeakerDiarizer::new(2);
    let frames: Vec<Vec<f32>> = (0..50).map(|_| vec![0.0f32; 40]).collect();
    let labels = diarizer.diarize(&frames);
    assert_eq!(labels.len(), 50);
}

// ──────────────────────────────────────────────────────────────
// SpeechAugmentation
// ──────────────────────────────────────────────────────────────
#[test]
fn test_add_noise_length() {
    let mut rng = StdRng::seed_from_u64(42);
    let audio: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
    let noisy = SpeechAugmentation::add_noise(&audio, 20.0, &mut rng);
    assert_eq!(noisy.len(), audio.len());
}

#[test]
fn test_add_noise_different_from_original() {
    let mut rng = StdRng::seed_from_u64(7);
    let audio = vec![0.5f32; 100];
    let noisy = SpeechAugmentation::add_noise(&audio, 10.0, &mut rng);
    let diff: f32 = audio
        .iter()
        .zip(noisy.iter())
        .map(|(a, n)| (a - n).abs())
        .sum();
    assert!(diff > 0.0, "noise should change signal");
}

#[test]
fn test_time_stretch_speedup() {
    let audio: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
    let stretched = SpeechAugmentation::time_stretch(&audio, 2.0);
    assert_eq!(stretched.len(), 500, "rate=2 → half length");
}

#[test]
fn test_time_stretch_slowdown() {
    let audio: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
    let stretched = SpeechAugmentation::time_stretch(&audio, 0.5);
    assert_eq!(stretched.len(), 2000, "rate=0.5 → double length");
}

#[test]
fn test_pitch_shift_length_preserved() {
    let audio: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.02).sin()).collect();
    let shifted = SpeechAugmentation::pitch_shift(&audio, 2.0, 16000);
    assert_eq!(
        shifted.len(),
        audio.len(),
        "pitch shift should preserve length"
    );
}

#[test]
fn test_room_impulse_convolve() {
    let audio = vec![1.0f32, 0.5, 0.25, 0.0, 0.0];
    let rir = vec![1.0f32, 0.5];
    let out = SpeechAugmentation::room_impulse_convolve(&audio, &rir);
    assert_eq!(
        out.len(),
        audio.len(),
        "RIR convolve should truncate to audio length"
    );
    assert!(
        (out[0] - 1.0).abs() < 1e-5,
        "first sample = audio[0]*rir[0]"
    );
}

#[test]
fn test_rir_empty() {
    let audio = vec![1.0f32, 2.0, 3.0];
    let out = SpeechAugmentation::room_impulse_convolve(&audio, &[]);
    assert_eq!(out, audio, "empty RIR should return unchanged audio");
}

// ──────────────────────────────────────────────────────────────
// WordErrorRate
// ──────────────────────────────────────────────────────────────
#[test]
fn test_edit_distance_equal() {
    let a = vec![1u32, 2, 3];
    let b = vec![1u32, 2, 3];
    assert_eq!(edit_distance(&a, &b), 0);
}

#[test]
fn test_edit_distance_insertion() {
    let a = vec![1u32, 3];
    let b = vec![1u32, 2, 3];
    assert_eq!(edit_distance(&a, &b), 1);
}

#[test]
fn test_edit_distance_substitution() {
    let a = vec!['a', 'b', 'c'];
    let b = vec!['a', 'x', 'c'];
    assert_eq!(edit_distance(&a, &b), 1);
}

#[test]
fn test_wer_perfect() {
    let hyp = vec!["hello", "world"];
    let refer = vec!["hello", "world"];
    assert!((wer(&hyp, &refer) - 0.0).abs() < 1e-6);
}

#[test]
fn test_wer_one_sub() {
    let hyp = vec!["hello", "earth"];
    let refer = vec!["hello", "world"];
    assert!((wer(&hyp, &refer) - 0.5).abs() < 1e-5);
}

#[test]
fn test_cer_perfect() {
    assert!((cer("hello", "hello") - 0.0).abs() < 1e-6);
}

#[test]
fn test_cer_one_sub() {
    assert!((cer("hxllo", "hello") - 0.2).abs() < 1e-5);
}

#[test]
fn test_asr_metrics_compute() {
    let hyp = vec!["the", "cat", "sat"];
    let refer = vec!["the", "dog", "sat", "down"];
    let m = AsrMetrics::compute(&hyp, &refer);
    assert!(m.wer > 0.0, "non-perfect hyp should have wer > 0");
    assert!(m.reference_length == 4);
}

// ──────────────────────────────────────────────────────────────
// SpeechPipeline
// ──────────────────────────────────────────────────────────────
#[test]
fn test_speech_pipeline_transcribe_runs() {
    let vocab: Vec<String> = (0..10).map(|i| format!("tok{}", i)).collect();
    let enc_cfg = WhisperConfig {
        n_mels: 80,
        n_fft: 400,
        hop_length: 160,
        n_audio_ctx: 50,
        n_audio_state: 32,
        n_audio_heads: 2,
        n_audio_layers: 1,
    };
    let pipeline = SpeechPipeline::new(VadConfig::default(), enc_cfg, 10, vocab, 16000, 42)
        .expect("pipeline creation should succeed");
    let audio: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin()).collect();
    let transcript = pipeline.transcribe(&audio, 16000);
    // Just verify it runs and returns a string
    let _ = transcript;
}

#[test]
fn test_speech_pipeline_empty_audio() {
    let vocab: Vec<String> = (0..5).map(|i| format!("t{}", i)).collect();
    let enc_cfg = WhisperConfig {
        n_mels: 80,
        n_fft: 400,
        hop_length: 160,
        n_audio_ctx: 50,
        n_audio_state: 32,
        n_audio_heads: 2,
        n_audio_layers: 1,
    };
    let pipeline = SpeechPipeline::new(VadConfig::default(), enc_cfg, 5, vocab, 16000, 1)
        .expect("pipeline creation should succeed");
    let result = pipeline.transcribe(&[], 16000);
    assert!(
        result.is_empty(),
        "empty audio should give empty transcript"
    );
}

#[test]
fn test_speech_pipeline_vocab_mismatch_error() {
    let vocab: Vec<String> = vec!["a".to_string()];
    let enc_cfg = WhisperConfig::default();
    let result = SpeechPipeline::new(VadConfig::default(), enc_cfg, 10, vocab, 16000, 1);
    assert!(result.is_err(), "vocab mismatch should error");
}

// ──────────────────────────────────────────────────────────────
// Additional edge cases
// ──────────────────────────────────────────────────────────────
#[test]
fn test_log_add_identity() {
    let neg_inf = f32::NEG_INFINITY;
    assert!((log_add(0.0, neg_inf) - 0.0).abs() < 1e-6);
    assert!((log_add(neg_inf, 0.0) - 0.0).abs() < 1e-6);
}

#[test]
fn test_log_add_equal() {
    let result = log_add(-1.0, -1.0);
    let expected = -1.0 + 2.0_f32.ln();
    assert!((result - expected).abs() < 1e-5);
}

#[test]
fn test_layer_norm_zero_mean() {
    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let normed = layer_norm(&x, 1e-5);
    let mean: f32 = normed.iter().sum::<f32>() / normed.len() as f32;
    assert!(
        mean.abs() < 1e-4,
        "layer norm mean should be ~0, got {}",
        mean
    );
}

#[test]
fn test_ngram_add_multiple_sentences() {
    let mut lm = NgramLm::new(2, 20);
    lm.add_sentence(&[1, 2, 3]);
    lm.add_sentence(&[1, 2, 4]);
    lm.add_sentence(&[5, 6]);
    // Both 3 and 4 should have non-trivial prob after [1,2]
    let lp3 = lm.log_prob(&[2], 3);
    let lp4 = lm.log_prob(&[2], 4);
    assert!(lp3.is_finite() && lp4.is_finite());
}

#[test]
fn test_spectral_clustering_single_speaker() {
    let sc = SpectralClustering::new(1);
    let embs: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.9, 0.1], vec![0.8, 0.2]];
    let labels = sc.cluster_embeddings(&embs);
    assert!(
        labels.iter().all(|&l| l == 0),
        "single speaker: all labels should be 0"
    );
}

#[test]
fn test_rnnt_prediction_network_no_nan() {
    let pred = PredictionNetwork::new(5, 4, 8, 123);
    let h0 = vec![0.1f32; 8];
    let c0 = vec![-0.1f32; 8];
    for tok in 0..5 {
        let (h, c) = pred.step(tok, &h0, &c0);
        for &v in h.iter().chain(c.iter()) {
            assert!(v.is_finite(), "LSTM output must be finite");
        }
    }
}

#[test]
fn test_whisper_encoder_layer_single_frame() {
    let layer = WhisperEncoderLayer::new(16, 2, 77);
    let x = vec![vec![0.1f32; 16]];
    let out = layer.forward(&x);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].len(), 16);
}

#[test]
fn test_wer_empty_reference() {
    let hyp = vec!["extra"];
    let refer: Vec<&str> = vec![];
    let score = wer(&hyp, &refer);
    assert_eq!(score, 1.0);
}

#[test]
fn test_wer_empty_hypothesis() {
    let hyp: Vec<&str> = vec![];
    let refer = vec!["hello", "world"];
    let score = wer(&hyp, &refer);
    assert_eq!(score, 1.0);
}

#[test]
fn test_vad_min_frames_filter() {
    // Very strict min_speech_frames — isolated speech frames should be suppressed
    let cfg = VadConfig {
        frame_size: 160,
        hop_size: 80,
        energy_threshold: 0.0001,
        min_speech_frames: 20,
        min_silence_frames: 1,
    };
    let vad = VoiceActivityDetector::new(cfg);
    // Only 3 loud samples surrounded by silence → should be suppressed
    let mut audio = vec![0.0f32; 3200];
    for i in 1600..1760 {
        audio[i] = 1.0; // only one frame of speech (frame_size=160)
    }
    let frames = vad.detect(&audio);
    // With min_speech_frames=20 and only ~2 speech frames, they should be suppressed
    let speech_count = frames.iter().filter(|&&s| s).count();
    assert_eq!(
        speech_count, 0,
        "short burst should be suppressed by min_speech_frames=20"
    );
}
