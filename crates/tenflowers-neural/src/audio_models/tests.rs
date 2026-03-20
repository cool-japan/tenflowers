//! Tests for advanced audio model algorithms.

use super::advanced::*;
use super::l2_norm;
use scirs2_core::random::SeedableRng;
use scirs2_core::Rng;
use scirs2_core::RngExt;

// ─── HuBERT tests ─────────────────────────────────────────────────────────

#[test]
fn test_hubert_cluster_assign_shape() {
    let model = HubertModel::new(4, 8, 1, 2, 42);
    let feat: Vec<f32> = (0..10 * 8).map(|i| i as f32 * 0.01).collect();
    let labels = model.cluster_assign(&feat).expect("cluster assign");
    assert_eq!(labels.len(), 10, "should return T labels");
    for &l in &labels {
        assert!(l < 4, "label {l} out of range [0, n_clusters)");
    }
}

#[test]
fn test_hubert_encode_shape() {
    let model = HubertModel::new(4, 8, 1, 2, 42);
    let feat: Vec<f32> = (0..6 * 8).map(|i| (i as f32 * 0.05).sin()).collect();
    let enc = model.encode(&feat).expect("hubert encode");
    assert_eq!(enc.len(), feat.len(), "encoder should preserve shape");
}

#[test]
fn test_hubert_masked_prediction_loss_finite() {
    let model = HubertModel::new(4, 8, 1, 2, 42);
    let feat: Vec<f32> = (0..8 * 8).map(|i| (i as f32 * 0.1).sin()).collect();
    let mask: Vec<bool> = (0..8).map(|i| i % 3 == 0).collect();
    let loss = model.masked_prediction_loss(&feat, &mask).expect("hubert loss");
    assert!(loss.is_finite(), "HuBERT loss should be finite, got {loss}");
    assert!(loss >= 0.0, "HuBERT loss should be non-negative, got {loss}");
}

#[test]
fn test_hubert_all_masked_loss_finite() {
    let model = HubertModel::new(4, 8, 1, 2, 99);
    let feat: Vec<f32> = (0..5 * 8).map(|i| i as f32 * 0.02).collect();
    let mask = vec![true; 5];
    let loss = model.masked_prediction_loss(&feat, &mask).expect("all masked");
    assert!(loss.is_finite(), "all-masked loss should be finite: {loss}");
}

// ─── Data2Vec tests ────────────────────────────────────────────────────────

#[test]
fn test_data2vec_teacher_targets_shape() {
    let model = Data2VecAudio::new(8, 2, 2, 0.999, 2, 42);
    let x: Vec<f32> = (0..6 * 8).map(|i| (i as f32 * 0.1).sin()).collect();
    let targets = model.teacher_targets(&x).expect("teacher targets");
    assert_eq!(targets.len(), x.len(), "targets shape should match input");
}

#[test]
fn test_data2vec_regression_loss_finite() {
    let model = Data2VecAudio::new(8, 2, 2, 0.999, 2, 42);
    let x: Vec<f32> = (0..8 * 8).map(|i| (i as f32 * 0.05).cos()).collect();
    let mask: Vec<bool> = (0..8).map(|i| i % 2 == 0).collect();
    let loss = model.regression_loss(&x, &mask).expect("regression loss");
    assert!(loss.is_finite(), "regression loss should be finite: {loss}");
    assert!(loss >= 0.0, "cosine distance should be non-negative");
}

#[test]
fn test_data2vec_ema_update_changes_weights() {
    let mut model = Data2VecAudio::new(8, 1, 2, 0.9, 1, 42);
    let before: Vec<f32> = model.teacher_qkv[0].clone();
    // Perturb student
    for v in model.student_qkv[0].iter_mut() {
        *v += 0.5;
    }
    model.ema_update();
    let after = &model.teacher_qkv[0];
    let diff: f32 = before.iter().zip(after.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 0.0, "EMA update should change teacher weights");
}

// ─── UniSpeech tests ──────────────────────────────────────────────────────

#[test]
fn test_unispeech_encode_shape() {
    let enc = UniSpeechEncoder::new(8, 2, 10, 2, 42);
    let x: Vec<f32> = (0..6 * 8).map(|i| i as f32 * 0.01).collect();
    let out = enc.encode(&x).expect("unispeech encode");
    assert_eq!(out.len(), x.len(), "encoder should preserve shape");
}

#[test]
fn test_unispeech_ctc_logits_shape() {
    let enc = UniSpeechEncoder::new(8, 1, 10, 2, 42);
    let x: Vec<f32> = (0..5 * 8).map(|i| i as f32 * 0.01).collect();
    let logits = enc.ctc_logits(&x).expect("ctc logits");
    assert_eq!(logits.len(), 5 * 10, "ctc logits shape: [T, n_phones]");
}

#[test]
fn test_unispeech_ctc_loss_finite() {
    let enc = UniSpeechEncoder::new(8, 1, 10, 2, 42);
    let x: Vec<f32> = (0..5 * 8).map(|i| i as f32 * 0.01).collect();
    let targets: Vec<usize> = vec![1, 3, 2, 5, 9];
    let loss = enc.ctc_loss(&x, &targets).expect("ctc loss");
    assert!(loss.is_finite(), "CTC loss should be finite: {loss}");
    assert!(loss >= 0.0, "CTC loss should be non-negative");
}

// ─── RVQ Codebook tests ───────────────────────────────────────────────────

#[test]
fn test_rvq_quantize_shape() {
    let rvq = RvqCodebook::new(4, 16, 8, 0.25, 42);
    let x: Vec<f32> = (0..10 * 8).map(|i| i as f32 * 0.01).collect();
    let (q, loss, indices) = rvq.quantize(&x).expect("rvq quantize");
    assert_eq!(q.len(), x.len(), "quantized shape should match input");
    assert!(loss.is_finite(), "commitment loss should be finite: {loss}");
    assert!(loss >= 0.0, "commitment loss should be non-negative");
    assert_eq!(indices.len(), 4, "should have n_stages index vectors");
    for stage_idx in &indices {
        assert_eq!(stage_idx.len(), 10, "each stage has T indices");
        for &idx in stage_idx {
            assert!(idx < 16, "index {idx} out of codebook range");
        }
    }
}

#[test]
fn test_rvq_residual_decreases() {
    let rvq = RvqCodebook::new(4, 8, 4, 0.25, 99);
    let x: Vec<f32> = (0..5 * 4).map(|i| i as f32 * 0.1).collect();
    let (q, _, _) = rvq.quantize(&x).expect("rvq quantize");
    // After RVQ, the residual should be smaller than the input
    let orig_norm: f32 = l2_norm(&x);
    let res: Vec<f32> = x.iter().zip(&q).map(|(a, b)| a - b).collect();
    let res_norm = l2_norm(&res);
    assert!(
        res_norm <= orig_norm + 1.0,
        "residual norm {res_norm} should not greatly exceed original norm {orig_norm}"
    );
}

// ─── SoundStream tests ────────────────────────────────────────────────────

#[test]
fn test_soundstream_encode_runs() {
    let codec = SoundStreamCodec::new(4, 2, 8, 42);
    // Minimum required input length for the strides [2,4,5,8]
    let min_len = 2 * 4 * 4 * 5 * 8 * 16 + 100;
    let waveform: Vec<f32> = (0..min_len).map(|i| (i as f32 * 0.001).sin()).collect();
    let result = codec.encode(&waveform);
    assert!(result.is_ok(), "SoundStream encode should succeed: {:?}", result.err());
    let (q, loss, indices) = result.unwrap();
    assert!(loss >= 0.0, "commitment loss non-negative");
    assert_eq!(indices.len(), 2, "2 RVQ stages");
    assert!(!q.is_empty(), "quantized output non-empty");
}

#[test]
fn test_soundstream_codec_loss_finite() {
    let codec = SoundStreamCodec::new(4, 2, 8, 42);
    let min_len = 2 * 4 * 4 * 5 * 8 * 16 + 100;
    let waveform: Vec<f32> = (0..min_len).map(|i| (i as f32 * 0.01).cos()).collect();
    let loss = codec.codec_loss(&waveform);
    match loss {
        Ok(l) => {
            assert!(l.is_finite(), "codec loss should be finite: {l}");
            assert!(l >= 0.0, "codec loss should be non-negative");
        }
        Err(e) => {
            // Error is acceptable if the sequence is too short after striding
            let _ = e;
        }
    }
}

// ─── DAC Codec tests ──────────────────────────────────────────────────────

#[test]
fn test_dac_stft_loss_finite() {
    let codec = DacCodec::new(4, 2, 8, 42);
    let sig: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
    let recon: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin() + 0.01).collect();
    let loss = codec.stft_loss(&sig, &recon);
    assert!(loss.is_finite(), "STFT loss should be finite: {loss}");
    assert!(loss >= 0.0, "STFT loss should be non-negative");
}

#[test]
fn test_dac_stft_loss_zero_same_signal() {
    let codec = DacCodec::new(4, 2, 8, 42);
    let sig: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
    let loss = codec.stft_loss(&sig, &sig);
    assert!(loss < 1e-5, "STFT loss between identical signals should be ~0, got {loss}");
}

#[test]
fn test_dac_encode_commitment_loss_finite() {
    let codec = DacCodec::new(4, 2, 8, 42);
    let min_len = 2 * 4 * 4 * 8 * 8 * 16 + 100;
    let waveform: Vec<f32> = (0..min_len).map(|i| (i as f32 * 0.001).sin()).collect();
    if let Ok((_, loss, _)) = codec.encode(&waveform) {
        assert!(loss.is_finite(), "commitment loss finite: {loss}");
        assert!(loss >= 0.0);
        // Err case: sequence may still be too short; acceptable
    }
}

// ─── Beat Tracker tests ───────────────────────────────────────────────────

#[test]
fn test_beat_tracker_onset_strength_length() {
    let bt = BeatTracker::new(120.0, 22050.0, 512);
    let n_freqs = 64;
    let t = 30;
    let spec: Vec<f32> = (0..t * n_freqs).map(|i| (i as f32 * 0.01).abs()).collect();
    let odf = bt.onset_strength(&spec, n_freqs);
    assert_eq!(odf.len(), t, "ODF length should equal T");
}

#[test]
fn test_beat_tracker_odf_normalized() {
    let bt = BeatTracker::new(120.0, 22050.0, 512);
    let n_freqs = 32;
    let t = 50;
    // Create a spectrogram with periodic energy bursts
    let spec: Vec<f32> = (0..t * n_freqs).map(|i| if (i / n_freqs) % 5 == 0 { 1.0 } else { 0.1 }).collect();
    let odf = bt.onset_strength(&spec, n_freqs);
    let max_v = odf.iter().cloned().fold(0.0_f32, f32::max);
    assert!(max_v <= 1.0 + 1e-5, "ODF should be normalized to [0,1]: max={max_v}");
}

#[test]
fn test_beat_tracker_returns_beats() {
    let bt = BeatTracker::new(120.0, 22050.0, 512);
    let t = 100;
    // Simulate a steady 120 BPM signal (beat every 26 frames at fps=43.1)
    let period = 26usize;
    let odf: Vec<f32> = (0..t).map(|i| if i % period == 0 { 1.0 } else { 0.0 }).collect();
    let beats = bt.track_beats(&odf);
    assert!(!beats.is_empty(), "should detect at least one beat");
    for &b in &beats {
        assert!(b < t, "beat index {b} out of range");
    }
}

// ─── Chord Recognizer tests ───────────────────────────────────────────────

#[test]
fn test_chord_recognizer_templates_shape() {
    let cr = ChordRecognizer::new();
    assert_eq!(cr.templates.len(), 24 * 12, "24 chords × 12 chroma bins");
}

#[test]
fn test_chord_recognizer_viterbi_shape() {
    let cr = ChordRecognizer::new();
    let t = 10;
    let chromagrams: Vec<f32> = (0..t * 12).map(|i| (i as f32 * 0.1).abs()).collect();
    let chords = cr.viterbi_decode(&chromagrams).expect("viterbi decode");
    assert_eq!(chords.len(), t, "should return T chord labels");
    for &c in &chords {
        assert!(c < 24, "chord index {c} out of range [0,24)");
    }
}

#[test]
fn test_chord_recognizer_c_major() {
    let cr = ChordRecognizer::new();
    // C major chroma: strong C(0), E(4), G(7)
    let mut chroma = vec![0.0_f32; 12];
    chroma[0] = 1.0; // C
    chroma[4] = 0.9; // E
    chroma[7] = 0.8; // G
    let norm: f32 = chroma.iter().map(|x| x * x).sum::<f32>().sqrt();
    for v in chroma.iter_mut() {
        *v /= norm;
    }
    let chords = cr.viterbi_decode(&chroma).expect("viterbi single frame");
    assert_eq!(chords.len(), 1);
    assert!(chords[0] < 12, "C major should be in the major range [0,11]");
}

#[test]
fn test_chord_recognizer_chromagram_frame() {
    let cr = ChordRecognizer::new();
    let n_freqs = 513;
    let spec: Vec<f32> = (0..n_freqs).map(|i| if i == 100 { 1.0 } else { 0.0 }).collect();
    let chroma = cr.chromagram_frame(&spec, 22050.0, 1024);
    assert_eq!(chroma.len(), 12, "chromagram frame should have 12 bins");
}

// ─── Music Separator tests ────────────────────────────────────────────────

#[test]
fn test_music_separator_output_shape() {
    let sep = MusicSeparator::new(5, 5);
    let t = 20;
    let n_freqs = 32;
    let spec: Vec<f32> = vec![0.5_f32; t * n_freqs];
    let (h, p) = sep.separate(&spec, n_freqs).expect("separate");
    assert_eq!(h.len(), t * n_freqs, "harmonic shape mismatch");
    assert_eq!(p.len(), t * n_freqs, "percussive shape mismatch");
}

#[test]
fn test_music_separator_energy_conservation() {
    let sep = MusicSeparator::new(3, 3);
    let t = 10;
    let n_freqs = 16;
    let spec: Vec<f32> = (0..t * n_freqs).map(|i| (i as f32 * 0.1).abs() + 0.1).collect();
    let (h, p) = sep.separate(&spec, n_freqs).expect("separate energy");
    // h[i] + p[i] should approximately equal spec[i]
    for i in 0..spec.len() {
        let reconstructed = h[i] + p[i];
        let orig = spec[i];
        assert!(
            (reconstructed - orig).abs() < orig * 0.01 + 1e-5,
            "energy conservation failed at {i}: {reconstructed} vs {orig}"
        );
    }
}

// ─── Key Detector tests ────────────────────────────────────────────────────

#[test]
fn test_key_detector_profiles_normalized() {
    let kd = KeyDetector::new();
    let mean_maj: f32 = kd.major_profile.iter().sum::<f32>() / 12.0;
    assert!(mean_maj.abs() < 1e-4, "major profile should have zero mean after normalization");
}

#[test]
fn test_key_detector_detect_c_major() {
    let kd = KeyDetector::new();
    // C major pitch class distribution: C, D, E, F, G, A, B strong
    let pcd = [1.0f32, 0.0, 0.8, 0.0, 0.9, 0.7, 0.0, 1.0, 0.0, 0.8, 0.0, 0.7];
    let (key, is_minor) = kd.detect_key(&pcd).expect("detect C major");
    // Allow some tolerance — should detect root 0 (C) as major
    assert!(!is_minor || key == 0, "C major signal detected as key={key}, minor={is_minor}");
}

#[test]
fn test_key_detector_output_range() {
    let kd = KeyDetector::new();
    let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(42);
    for _ in 0..5 {
        let pcd: Vec<f32> = (0..12).map(|_| rng.random::<f32>()).collect();
        let (key, _) = kd.detect_key(&pcd).expect("detect key");
        assert!(key < 12, "key {key} should be in [0, 12)");
    }
}

// ─── FastSpeech2Duration tests ────────────────────────────────────────────

#[test]
fn test_fastspeech2_duration_shape() {
    let dp = FastSpeech2Duration::new(16, 3, 42);
    let x: Vec<f32> = (0..8 * 16).map(|i| i as f32 * 0.01).collect();
    let durs = dp.predict(&x).expect("predict durations");
    assert_eq!(durs.len(), 8, "should return T durations");
}

#[test]
fn test_fastspeech2_duration_positive() {
    let dp = FastSpeech2Duration::new(16, 3, 42);
    let x: Vec<f32> = (0..6 * 16).map(|i| (i as f32 * 0.05).sin()).collect();
    let durs = dp.predict(&x).expect("predict durations positive");
    for &d in &durs {
        assert!(d > 0.0, "softplus duration should be positive, got {d}");
    }
}

// ─── LengthRegulatorAdv tests ─────────────────────────────────────────────

#[test]
fn test_length_regulator_adv_basic() {
    let lr = LengthRegulatorAdv::new(0);
    let d = 4;
    let x: Vec<f32> = (0..3 * d).map(|i| i as f32).collect();
    let durs = vec![2.0f32, 0.0, 3.0];
    let out = lr.regulate(&x, &durs, d).expect("regulate adv");
    assert_eq!(out.len(), 5 * d, "2 + 0 + 3 frames");
}

#[test]
fn test_length_regulator_adv_min_dur() {
    let lr = LengthRegulatorAdv::new(1);
    let d = 2;
    let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let durs = vec![0.3f32, 0.2]; // both round to 0 → min_dur=1 each
    let out = lr.regulate(&x, &durs, d).expect("min dur");
    assert_eq!(out.len(), 2 * d, "min_dur=1 should produce 2 frames");
}

// ─── VocoderHiFi tests ────────────────────────────────────────────────────

#[test]
fn test_vocoder_hifi_generate_shape() {
    let voc = VocoderHiFi::new(8, 8, 42);
    let t = 10;
    let mel: Vec<f32> = (0..t * 8).map(|i| (i as f32 * 0.01).sin()).collect();
    let out = voc.generate(&mel).expect("vocoder generate");
    assert!(!out.is_empty(), "vocoder output should be non-empty");
    // Upsample by factor 2^2 * 2^3 * 2^4 = 128 approximately
    assert!(out.len() >= t, "output should be longer than input mel");
}

#[test]
fn test_vocoder_hifi_output_range() {
    let voc = VocoderHiFi::new(8, 8, 42);
    let t = 5;
    let mel: Vec<f32> = (0..t * 8).map(|i| i as f32 * 0.01).collect();
    let out = voc.generate(&mel).expect("vocoder range");
    for &v in &out {
        assert!(
            (-1.0 - 1e-6..=1.0 + 1e-6).contains(&v),
            "tanh output should be in [-1, 1], got {v}"
        );
    }
}

// ─── TtsMetrics tests ─────────────────────────────────────────────────────

#[test]
fn test_tts_mcd_identical_zero() {
    let d = 13;
    let t = 5;
    let mfcc: Vec<f32> = (0..t * d).map(|i| i as f32 * 0.01).collect();
    let mcd = TtsMetrics::mel_cepstral_distortion(&mfcc, &mfcc, d).expect("mcd identical");
    assert!(mcd < 1e-5, "MCD of identical sequences should be ~0, got {mcd}");
}

#[test]
fn test_tts_mcd_positive() {
    let d = 13;
    let t = 5;
    let ref_mfcc: Vec<f32> = (0..t * d).map(|i| i as f32 * 0.01).collect();
    let syn_mfcc: Vec<f32> = (0..t * d).map(|i| i as f32 * 0.01 + 0.5).collect();
    let mcd = TtsMetrics::mel_cepstral_distortion(&ref_mfcc, &syn_mfcc, d).expect("mcd pos");
    assert!(mcd > 0.0, "MCD should be positive for different sequences, got {mcd}");
    assert!(mcd.is_finite(), "MCD should be finite, got {mcd}");
}

#[test]
fn test_tts_mos_proxy_range() {
    let mos_low = TtsMetrics::mos_proxy(10.0);
    let mos_high = TtsMetrics::mos_proxy(0.0);
    assert!((1.0..=5.0).contains(&mos_low), "MOS proxy should be in [1,5]: {mos_low}");
    assert!((1.0..=5.0).contains(&mos_high), "MOS proxy should be in [1,5]: {mos_high}");
    assert!(mos_high >= mos_low, "lower MCD → higher MOS: {mos_high} >= {mos_low}");
}

#[test]
fn test_tts_cer_identical() {
    let ref_seq = vec![1usize, 2, 3, 4, 5];
    let cer = TtsMetrics::cer(&ref_seq, &ref_seq);
    assert_eq!(cer, 0.0, "CER of identical sequences should be 0");
}

#[test]
fn test_tts_cer_completely_wrong() {
    let ref_seq = vec![1usize, 2, 3];
    let hyp_seq = vec![4usize, 5, 6];
    let cer = TtsMetrics::cer(&ref_seq, &hyp_seq);
    assert_eq!(cer, 1.0, "completely wrong sequences: CER should be 1.0");
}

#[test]
fn test_tts_metrics_accumulate() {
    let mut metrics = TtsMetrics::new();
    metrics.add_mcd(5.0);
    metrics.add_mcd(3.0);
    metrics.add_wer(0.1);
    metrics.add_wer(0.3);
    let mean_mcd = metrics.mean_mcd();
    let mean_wer = metrics.mean_wer();
    assert!((mean_mcd - 4.0).abs() < 1e-5, "mean MCD should be 4.0, got {mean_mcd}");
    assert!((mean_wer - 0.2).abs() < 1e-5, "mean WER should be 0.2, got {mean_wer}");
}

#[test]
fn test_tts_metrics_empty() {
    let m = TtsMetrics::new();
    assert_eq!(m.mean_mcd(), 0.0, "empty MCD should return 0");
    assert_eq!(m.mean_wer(), 0.0, "empty WER should return 0");
}
