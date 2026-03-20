//! Speech recognition extensions (§3-§10).
//!
//! CTC beam decoder, RNN-T decoder, N-gram LM rescorer,
//! VAD, speaker diarization, augmentation, WER metrics, pipeline.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;
use std::f32::consts::PI;
use tenflowers_core::{Result, TensorError};

use super::{
    add_vecs, dot, gelu, inval, l2_norm, layer_norm, log_mel_spectrogram, log_softmax_vec, matvec,
    softmax_vec, WhisperConfig, WhisperEncoder,
};

// ─────────────────────────────────────────────────────────────────────────────
// 3. CTC Beam Decoder
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for CTC decoding.
#[derive(Debug, Clone)]
pub struct CtcConfig {
    pub vocab_size: usize,
    /// Token id representing the CTC blank symbol.
    pub blank_id: usize,
    pub beam_width: usize,
}

impl Default for CtcConfig {
    fn default() -> Self {
        Self {
            vocab_size: 29,
            blank_id: 0,
            beam_width: 10,
        }
    }
}

/// Beam entry tracking prefix probabilities split by blank / non-blank.
#[derive(Debug, Clone)]
pub struct BeamEntry {
    pub tokens: Vec<usize>,
    /// Log-probability of last token being blank.
    pub p_b: f32,
    /// Log-probability of last token being non-blank.
    pub p_nb: f32,
}

impl BeamEntry {
    fn total_log_prob(&self) -> f32 {
        log_add(self.p_b, self.p_nb)
    }
}

/// Log-domain addition: log(exp(a) + exp(b)).
#[inline]
pub(super) fn log_add(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let (big, small) = if a >= b { (a, b) } else { (b, a) };
    big + (1.0 + (small - big).exp()).ln()
}

/// CTC beam-search and greedy decoder.
#[derive(Debug, Clone)]
pub struct CtcBeamDecoder {
    pub config: CtcConfig,
}

impl CtcBeamDecoder {
    pub fn new(config: CtcConfig) -> Self {
        Self { config }
    }

    /// CTC greedy decode: argmax at each frame, collapse repeats, remove blanks.
    ///
    /// `log_probs`: `[T][vocab_size]` log-probabilities.
    pub fn greedy_decode(&self, log_probs: &[Vec<f32>]) -> Vec<usize> {
        let blank = self.config.blank_id;
        let mut best_path: Vec<usize> = log_probs
            .iter()
            .map(|frame| {
                frame
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(blank)
            })
            .collect();

        // Collapse repeats and remove blanks
        let mut result = Vec::new();
        let mut prev = usize::MAX;
        for &tok in &best_path {
            if tok != prev && tok != blank {
                result.push(tok);
            }
            prev = tok;
        }
        result
    }

    /// CTC beam search decode.
    ///
    /// `log_probs`: `[T][vocab_size]` log-probabilities.
    pub fn beam_search(&self, log_probs: &[Vec<f32>], beam_width: usize) -> Vec<usize> {
        if log_probs.is_empty() {
            return Vec::new();
        }
        let blank = self.config.blank_id;
        let v = self.config.vocab_size;
        let bw = beam_width.max(1);
        let neg_inf = f32::NEG_INFINITY;

        // Initialise beam with empty prefix
        let mut beam: Vec<BeamEntry> = vec![BeamEntry {
            tokens: Vec::new(),
            p_b: 0.0, // log(1) = 0
            p_nb: neg_inf,
        }];

        for frame_log_probs in log_probs {
            let mut new_beam: HashMap<Vec<usize>, (f32, f32)> = HashMap::new(); // (p_b, p_nb)

            for entry in &beam {
                let total_prev = entry.total_log_prob();

                // Extend with blank
                let p_b_new = log_add(total_prev + frame_log_probs[blank.min(v - 1)], neg_inf);
                let key = entry.tokens.clone();
                let e = new_beam.entry(key.clone()).or_insert((neg_inf, neg_inf));
                e.0 = log_add(e.0, p_b_new);

                // Extend with non-blank tokens
                for tok in 0..v {
                    if tok == blank {
                        continue;
                    }
                    let lp = frame_log_probs[tok];
                    // If last token == tok, only p_b contributes to non-blank
                    let mut new_p_nb = if entry.tokens.last() == Some(&tok) {
                        entry.p_b + lp
                    } else {
                        total_prev + lp
                    };
                    let mut new_key = entry.tokens.clone();
                    new_key.push(tok);
                    let e2 = new_beam.entry(new_key).or_insert((neg_inf, neg_inf));
                    e2.1 = log_add(e2.1, new_p_nb);
                }
            }

            // Convert map to beam, sort by total log-prob, keep top-bw
            let mut next_beam: Vec<BeamEntry> = new_beam
                .into_iter()
                .map(|(tokens, (p_b, p_nb))| BeamEntry { tokens, p_b, p_nb })
                .collect();
            next_beam.sort_by(|a, b| {
                b.total_log_prob()
                    .partial_cmp(&a.total_log_prob())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            next_beam.truncate(bw);
            beam = next_beam;
        }

        beam.into_iter()
            .max_by(|a, b| {
                a.total_log_prob()
                    .partial_cmp(&b.total_log_prob())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|e| e.tokens)
            .unwrap_or_default()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. RNN-T Decoder
// ─────────────────────────────────────────────────────────────────────────────

/// Single-layer LSTM prediction network for RNN-T.
#[derive(Debug, Clone)]
pub struct PredictionNetwork {
    pub embed: Vec<Vec<f32>>,
    /// LSTM input-hidden and hidden-hidden weights flattened: [4 * hidden * (embed_dim + hidden)]
    pub lstm_weights: Vec<f32>,
    hidden_size: usize,
    embed_dim: usize,
    vocab_size: usize,
}

impl PredictionNetwork {
    pub fn new(vocab_size: usize, embed_dim: usize, hidden_size: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (1.0 / hidden_size as f32).sqrt();
        let embed = (0..vocab_size)
            .map(|_| {
                (0..embed_dim)
                    .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
                    .collect()
            })
            .collect();
        let lstm_size = 4 * hidden_size * (embed_dim + hidden_size);
        let lstm_weights: Vec<f32> = (0..lstm_size)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        Self {
            embed,
            lstm_weights,
            hidden_size,
            embed_dim,
            vocab_size,
        }
    }

    /// Forward one LSTM step.
    ///
    /// `token`: previous token index
    /// `h_prev`, `c_prev`: previous LSTM state
    /// Returns `(h_next, c_next)`.
    pub fn step(&self, token: usize, h_prev: &[f32], c_prev: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let tok_idx = token.min(self.vocab_size.saturating_sub(1));
        let e = &self.embed[tok_idx];
        let h = self.hidden_size;
        let ed = self.embed_dim;
        let in_size = ed + h;

        // Concatenate [embed; h_prev]
        let mut inp = Vec::with_capacity(in_size);
        inp.extend_from_slice(e);
        inp.extend_from_slice(h_prev);

        // LSTM gates: i, f, g, o  each of size h
        let mut gates = vec![0.0f32; 4 * h];
        for gate in 0..4 {
            for r in 0..h {
                let mut val = 0.0f32;
                for c in 0..in_size {
                    val += self.lstm_weights[(gate * h + r) * in_size + c] * inp[c];
                }
                gates[gate * h + r] = val;
            }
        }

        // i, f, o → sigmoid; g → tanh
        let mut h_next = vec![0.0f32; h];
        let mut c_next = vec![0.0f32; h];
        for r in 0..h {
            let i_gate = sigmoid_f32(gates[r]);
            let f_gate = sigmoid_f32(gates[h + r]);
            let g_gate = gates[2 * h + r].tanh();
            let o_gate = sigmoid_f32(gates[3 * h + r]);
            c_next[r] = f_gate * c_prev[r] + i_gate * g_gate;
            h_next[r] = o_gate * c_next[r].tanh();
        }
        (h_next, c_next)
    }
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Joint network combining encoder and prediction network outputs.
#[derive(Debug, Clone)]
pub struct JointNetwork {
    pub encoder_proj: Vec<f32>,
    pub pred_proj: Vec<f32>,
    pub output_proj: Vec<f32>,
    enc_dim: usize,
    pred_dim: usize,
    joint_dim: usize,
    vocab_size: usize,
}

impl JointNetwork {
    pub fn new(
        enc_dim: usize,
        pred_dim: usize,
        joint_dim: usize,
        vocab_size: usize,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (1.0 / joint_dim as f32).sqrt();
        let encoder_proj: Vec<f32> = (0..joint_dim * enc_dim)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        let pred_proj: Vec<f32> = (0..joint_dim * pred_dim)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        let output_proj: Vec<f32> = (0..vocab_size * joint_dim)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        Self {
            encoder_proj,
            pred_proj,
            output_proj,
            enc_dim,
            pred_dim,
            joint_dim,
            vocab_size,
        }
    }

    /// Compute joint logits from encoder hidden state and prediction network hidden state.
    ///
    /// Returns `[vocab_size]` logits.
    pub fn joint(&self, enc_h: &[f32], pred_h: &[f32]) -> Vec<f32> {
        let jd = self.joint_dim;
        let enc_proj = matvec(&self.encoder_proj, enc_h, jd, self.enc_dim);
        let pred_proj = matvec(&self.pred_proj, pred_h, jd, self.pred_dim);
        let joint: Vec<f32> = enc_proj
            .iter()
            .zip(pred_proj.iter())
            .map(|(e, p)| (e + p).tanh())
            .collect();
        matvec(&self.output_proj, &joint, self.vocab_size, jd)
    }
}

/// RNN-T decoder combining prediction network and joint network.
#[derive(Debug, Clone)]
pub struct RnntDecoder {
    pub prediction: PredictionNetwork,
    pub joint: JointNetwork,
    blank_id: usize,
}

impl RnntDecoder {
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        hidden_size: usize,
        enc_dim: usize,
        joint_dim: usize,
        blank_id: usize,
        seed: u64,
    ) -> Self {
        let prediction = PredictionNetwork::new(vocab_size, embed_dim, hidden_size, seed);
        let joint = JointNetwork::new(enc_dim, hidden_size, joint_dim, vocab_size, seed + 1);
        Self {
            prediction,
            joint,
            blank_id,
        }
    }

    /// Greedy decoding over encoder outputs.
    ///
    /// `encoder_outputs`: `[T_enc][enc_dim]`
    /// Returns sequence of token ids (excluding blank).
    pub fn rnnt_greedy(&self, encoder_outputs: &[Vec<f32>]) -> Vec<usize> {
        let h = self.prediction.hidden_size;
        let mut h_pred = vec![0.0f32; h];
        let mut c_pred = vec![0.0f32; h];
        let mut tokens = Vec::new();
        let blank = self.blank_id;

        // Maximum tokens emitted per encoder frame to prevent infinite loops
        // with random / untrained weights.
        let max_per_frame = self.prediction.vocab_size + 1;
        for enc_frame in encoder_outputs {
            let mut frame_count = 0;
            loop {
                if frame_count >= max_per_frame {
                    break; // Safety bound: at most vocab_size tokens per frame
                }
                let logits = self.joint.joint(enc_frame, &h_pred);
                let next = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(blank);
                if next == blank {
                    break;
                }
                tokens.push(next);
                let (h_new, c_new) = self.prediction.step(next, &h_pred, &c_pred);
                h_pred = h_new;
                c_pred = c_new;
                frame_count += 1;
            }
        }
        tokens
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Language Model Rescorer
// ─────────────────────────────────────────────────────────────────────────────

/// N-gram language model with Laplace smoothing.
#[derive(Debug, Clone)]
pub struct NgramLm {
    pub n: usize,
    /// N-gram counts: context (n-1 tokens) → {next_token → count}.
    pub counts: HashMap<Vec<usize>, HashMap<usize, usize>>,
    /// Vocabulary size for Laplace smoothing.
    pub vocab_size: usize,
}

impl NgramLm {
    pub fn new(n: usize, vocab_size: usize) -> Self {
        Self {
            n,
            counts: HashMap::new(),
            vocab_size,
        }
    }

    /// Add a sentence (sequence of token ids) to the LM.
    pub fn add_sentence(&mut self, tokens: &[usize]) {
        if tokens.len() < self.n {
            return;
        }
        for i in 0..=tokens.len() - self.n {
            let context: Vec<usize> = tokens[i..i + self.n - 1].to_vec();
            let next = tokens[i + self.n - 1];
            self.counts
                .entry(context)
                .or_default()
                .entry(next)
                .and_modify(|c| *c += 1)
                .or_insert(1);
        }
    }

    /// Log-probability of `next_token` given `context` (Laplace smoothing).
    pub fn log_prob(&self, context: &[usize], next_token: usize) -> f32 {
        let ctx_len = self.n - 1;
        let ctx: Vec<usize> = if context.len() >= ctx_len {
            context[context.len() - ctx_len..].to_vec()
        } else {
            let mut padded = vec![0usize; ctx_len - context.len()];
            padded.extend_from_slice(context);
            padded
        };

        let vocab = self.vocab_size.max(1) as f32;
        if let Some(next_map) = self.counts.get(&ctx) {
            let total: usize = next_map.values().sum();
            let count = next_map.get(&next_token).copied().unwrap_or(0);
            // Laplace (add-1) smoothing
            ((count + 1) as f32 / (total as f32 + vocab)).ln()
        } else {
            // Unseen context: uniform
            (1.0 / vocab).ln()
        }
    }
}

/// Language model rescorer using shallow fusion.
#[derive(Debug, Clone)]
pub struct LmRescorer {
    pub lm: NgramLm,
}

impl LmRescorer {
    pub fn new(lm: NgramLm) -> Self {
        Self { lm }
    }

    /// Shallow fusion: log-linear interpolation of ASR and LM log-probs.
    ///
    /// `asr_logprobs`: `[vocab_size]` log-probs from ASR model.
    /// `lm_logprobs`: `[vocab_size]` log-probs from LM.
    /// `lm_weight`: interpolation weight λ for LM.
    /// Returns `[vocab_size]` combined log-probs.
    pub fn shallow_fusion(asr_logprobs: &[f32], lm_logprobs: &[f32], lm_weight: f32) -> Vec<f32> {
        asr_logprobs
            .iter()
            .zip(lm_logprobs.iter())
            .map(|(&asr, &lm)| asr + lm_weight * lm)
            .collect()
    }

    /// Rescore an n-best list of hypotheses.
    pub fn rescore_nbest<'a>(
        &'a self,
        hypotheses: &'a [Vec<usize>],
        asr_scores: &[f32],
        lm_weight: f32,
    ) -> Vec<(f32, &'a Vec<usize>)> {
        hypotheses
            .iter()
            .zip(asr_scores.iter())
            .map(|(hyp, &asr_score)| {
                let lm_score: f32 = hyp
                    .iter()
                    .enumerate()
                    .map(|(i, &tok)| {
                        let ctx = if i == 0 { &hyp[..0] } else { &hyp[..i] };
                        self.lm.log_prob(ctx, tok)
                    })
                    .sum();
                (asr_score + lm_weight * lm_score, hyp)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Voice Activity Detector
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for energy-based VAD.
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Frame size in samples.
    pub frame_size: usize,
    /// Hop size between frames.
    pub hop_size: usize,
    /// RMS energy threshold for speech detection.
    pub energy_threshold: f32,
    /// Minimum number of consecutive speech frames.
    pub min_speech_frames: usize,
    /// Minimum number of consecutive silence frames.
    pub min_silence_frames: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            frame_size: 400,
            hop_size: 160,
            energy_threshold: 0.01,
            min_speech_frames: 5,
            min_silence_frames: 10,
        }
    }
}

/// Energy-based Voice Activity Detector.
#[derive(Debug, Clone)]
pub struct VoiceActivityDetector {
    pub config: VadConfig,
}

impl VoiceActivityDetector {
    pub fn new(config: VadConfig) -> Self {
        Self { config }
    }

    /// Compute per-frame speech activity.
    ///
    /// Returns `Vec<bool>` of length `n_frames`, `true` = speech.
    pub fn detect(&self, audio: &[f32]) -> Vec<bool> {
        let fs = self.config.frame_size;
        let hs = self.config.hop_size;
        let threshold = self.config.energy_threshold;
        let min_sp = self.config.min_speech_frames;
        let min_sil = self.config.min_silence_frames;

        if audio.is_empty() || fs == 0 || hs == 0 {
            return Vec::new();
        }

        // Raw energy per frame
        let mut raw: Vec<bool> = Vec::new();
        let mut pos = 0;
        while pos + fs <= audio.len() {
            let rms = (audio[pos..pos + fs].iter().map(|&x| x * x).sum::<f32>() / fs as f32).sqrt();
            raw.push(rms > threshold);
            pos += hs;
        }

        // Median filtering / hangover: minimum speech / silence spans
        let n = raw.len();
        let mut result = raw.clone();

        // Forward pass: if speech starts, requires min_speech_frames consecutive
        {
            let mut i = 0;
            while i < n {
                if raw[i] {
                    // Count consecutive speech frames
                    let mut j = i;
                    while j < n && raw[j] {
                        j += 1;
                    }
                    if j - i < min_sp {
                        for k in i..j {
                            result[k] = false;
                        }
                    }
                    i = j;
                } else {
                    i += 1;
                }
            }
        }

        // Backward pass: if silence, requires min_silence_frames consecutive
        {
            let mut i = 0;
            while i < n {
                if !result[i] {
                    let mut j = i;
                    while j < n && !result[j] {
                        j += 1;
                    }
                    if j - i < min_sil {
                        for k in i..j {
                            result[k] = true;
                        }
                    }
                    i = j;
                } else {
                    i += 1;
                }
            }
        }

        result
    }

    /// Return speech segment boundaries as (start_sample, end_sample) pairs.
    pub fn segment(&self, audio: &[f32]) -> Vec<(usize, usize)> {
        let frames = self.detect(audio);
        let hs = self.config.hop_size;
        let fs = self.config.frame_size;

        let mut segments = Vec::new();
        let mut in_speech = false;
        let mut start = 0usize;

        for (i, &is_speech) in frames.iter().enumerate() {
            let frame_start = i * hs;
            if is_speech && !in_speech {
                start = frame_start;
                in_speech = true;
            } else if !is_speech && in_speech {
                let end = (frame_start + fs).min(audio.len());
                segments.push((start, end));
                in_speech = false;
            }
        }
        // Close final segment
        if in_speech {
            segments.push((start, audio.len()));
        }
        segments
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Speaker Diarizer
// ─────────────────────────────────────────────────────────────────────────────

/// Simple k-means spectral clustering for speaker diarization.
#[derive(Debug, Clone)]
pub struct SpectralClustering {
    pub n_speakers: usize,
    /// Maximum iterations for k-means.
    pub max_iter: usize,
}

impl SpectralClustering {
    pub fn new(n_speakers: usize) -> Self {
        Self {
            n_speakers,
            max_iter: 50,
        }
    }

    /// Mean-pool log-mel frames over fixed-size windows to create d-vectors.
    pub fn embed_frames(&self, log_mel_frames: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if log_mel_frames.is_empty() {
            return Vec::new();
        }
        let window = 25.max(log_mel_frames.len() / self.n_speakers.max(1));
        let step = window / 2;
        let n_frames = log_mel_frames.len();
        let dim = log_mel_frames[0].len();
        let mut embeddings = Vec::new();

        let mut pos = 0;
        while pos < n_frames {
            let end = (pos + window).min(n_frames);
            let seg = &log_mel_frames[pos..end];
            let mean: Vec<f32> = (0..dim)
                .map(|d| seg.iter().map(|f| f[d]).sum::<f32>() / seg.len() as f32)
                .collect();
            embeddings.push(mean);
            if end == n_frames {
                break;
            }
            pos += step;
        }
        embeddings
    }

    /// K-means clustering using cosine distance.
    ///
    /// Returns per-embedding speaker assignment (0..n_speakers).
    pub fn cluster_embeddings(&self, embeddings: &[Vec<f32>]) -> Vec<usize> {
        let n = embeddings.len();
        if n == 0 {
            return Vec::new();
        }
        let k = self.n_speakers.min(n);
        if k == 0 {
            return vec![0; n];
        }

        // Initialise centroids as first k embeddings (normalised)
        let mut centroids: Vec<Vec<f32>> = embeddings[..k]
            .iter()
            .map(|e| l2_normalize_vec(e))
            .collect();

        let mut assignments = vec![0usize; n];

        for _ in 0..self.max_iter {
            // Assignment step
            let mut changed = false;
            for (i, emb) in embeddings.iter().enumerate() {
                let norm_e = l2_normalize_vec(emb);
                let best = centroids
                    .iter()
                    .enumerate()
                    .max_by(|(_, ca), (_, cb)| {
                        let da = dot(&norm_e, ca);
                        let db = dot(&norm_e, cb);
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                if assignments[i] != best {
                    assignments[i] = best;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            let dim = embeddings[0].len();
            let mut new_centroids = vec![vec![0.0f32; dim]; k];
            let mut counts = vec![0usize; k];
            for (i, emb) in embeddings.iter().enumerate() {
                let c = assignments[i];
                for (d, &v) in emb.iter().enumerate() {
                    new_centroids[c][d] += v;
                }
                counts[c] += 1;
            }
            for c in 0..k {
                if counts[c] > 0 {
                    for d in 0..dim {
                        new_centroids[c][d] /= counts[c] as f32;
                    }
                    centroids[c] = l2_normalize_vec(&new_centroids[c]);
                }
            }
        }
        assignments
    }

    /// Diarize: return per-frame speaker labels.
    ///
    /// `audio_frames`: `[T][n_mels]` log-mel frames.
    pub fn diarize(&self, audio_frames: &[Vec<f32>]) -> Vec<usize> {
        let embeddings = self.embed_frames(audio_frames);
        let labels = self.cluster_embeddings(&embeddings);

        if audio_frames.is_empty() || labels.is_empty() {
            return vec![0; audio_frames.len()];
        }

        let window = 25.max(audio_frames.len() / self.n_speakers.max(1));
        let step = window / 2;
        let n_frames = audio_frames.len();

        // Map segment labels back to frames
        let mut frame_labels = vec![0usize; n_frames];
        let mut seg_idx = 0;
        let mut pos = 0;
        while pos < n_frames && seg_idx < labels.len() {
            let end = (pos + window).min(n_frames);
            for fi in pos..end {
                frame_labels[fi] = labels[seg_idx];
            }
            if end == n_frames {
                break;
            }
            pos += step;
            seg_idx += 1;
        }
        frame_labels
    }
}

fn l2_normalize_vec(v: &[f32]) -> Vec<f32> {
    let norm = l2_norm(v);
    if norm < 1e-10 {
        return v.to_vec();
    }
    v.iter().map(|&x| x / norm).collect()
}

/// Speaker diarizer wrapping spectral clustering.
#[derive(Debug, Clone)]
pub struct SpeakerDiarizer {
    pub clustering: SpectralClustering,
}

impl SpeakerDiarizer {
    pub fn new(n_speakers: usize) -> Self {
        Self {
            clustering: SpectralClustering::new(n_speakers),
        }
    }

    /// Diarize audio represented as log-mel frames.
    pub fn diarize(&self, audio_frames: &[Vec<f32>]) -> Vec<usize> {
        self.clustering.diarize(audio_frames)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Audio Augmentation
// ─────────────────────────────────────────────────────────────────────────────

/// Speech-specific audio augmentation utilities.
pub struct SpeechAugmentation;

impl SpeechAugmentation {
    /// Add white noise at given SNR (dB).
    pub fn add_noise(audio: &[f32], snr_db: f32, rng: &mut StdRng) -> Vec<f32> {
        if audio.is_empty() {
            return Vec::new();
        }
        let signal_power = audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32;
        let snr_linear = 10.0_f32.powf(snr_db / 10.0);
        let noise_power = (signal_power / snr_linear).max(1e-10);
        let noise_std = noise_power.sqrt();
        audio
            .iter()
            .map(|&x| {
                // Box-Muller transform for normal sample
                let u1: f32 = rng.random::<f32>().max(1e-10);
                let u2: f32 = rng.random::<f32>();
                let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
                x + noise_std * noise
            })
            .collect()
    }

    /// Time-stretch by `rate` using linear interpolation.
    ///
    /// `rate > 1.0` speeds up (shorter output), `rate < 1.0` slows down.
    pub fn time_stretch(audio: &[f32], rate: f32) -> Vec<f32> {
        if audio.is_empty() || rate <= 0.0 {
            return Vec::new();
        }
        let out_len = (audio.len() as f32 / rate).round() as usize;
        if out_len == 0 {
            return Vec::new();
        }
        (0..out_len)
            .map(|i| {
                let src = i as f32 * rate;
                let lo = src.floor() as usize;
                let hi = (lo + 1).min(audio.len() - 1);
                let frac = src - lo as f32;
                audio[lo] * (1.0 - frac) + audio[hi] * frac
            })
            .collect()
    }

    /// Approximate pitch shift via resampling.
    ///
    /// Shifts pitch by `semitones` semitones: resample to new rate, then
    /// stretch back to original length.
    pub fn pitch_shift(audio: &[f32], semitones: f32, _sample_rate: usize) -> Vec<f32> {
        let ratio = 2.0_f32.powf(semitones / 12.0);
        // Step 1: resample at ratio (time-stretch inverse direction)
        let resampled = Self::time_stretch(audio, ratio);
        // Step 2: resample back to original length
        let orig_len = audio.len();
        if resampled.is_empty() {
            return Vec::new();
        }
        let rs_len = resampled.len();
        (0..orig_len)
            .map(|i| {
                let src = i as f32 * rs_len as f32 / orig_len as f32;
                let lo = (src.floor() as usize).min(rs_len - 1);
                let hi = (lo + 1).min(rs_len - 1);
                let frac = src - lo as f32;
                resampled[lo] * (1.0 - frac) + resampled[hi] * frac
            })
            .collect()
    }

    /// Convolve audio with a Room Impulse Response (RIR) using linear convolution.
    pub fn room_impulse_convolve(audio: &[f32], rir: &[f32]) -> Vec<f32> {
        if audio.is_empty() || rir.is_empty() {
            return audio.to_vec();
        }
        let out_len = audio.len() + rir.len() - 1;
        let mut out = vec![0.0f32; out_len];
        for (i, &a) in audio.iter().enumerate() {
            for (j, &r) in rir.iter().enumerate() {
                out[i + j] += a * r;
            }
        }
        // Truncate to input length
        out.truncate(audio.len());
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Word Error Rate
// ─────────────────────────────────────────────────────────────────────────────

/// Levenshtein edit distance between two sequences.
pub fn edit_distance<T: Eq>(a: &[T], b: &[T]) -> usize {
    let m = a.len();
    let n = b.len();
    // Use a single row DP to save memory
    let mut dp: Vec<usize> = (0..=n).collect();
    let mut prev_row = dp.clone();
    for i in 1..=m {
        prev_row[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            prev_row[j] = (dp[j] + 1).min(prev_row[j - 1] + 1).min(dp[j - 1] + cost);
        }
        dp = prev_row.clone();
    }
    dp[n]
}

/// Compute Word Error Rate (WER) between hypothesis and reference word sequences.
///
/// `WER = (S + D + I) / N` where N = reference length.
pub fn wer(hypothesis: &[&str], reference: &[&str]) -> f32 {
    if reference.is_empty() {
        return if hypothesis.is_empty() { 0.0 } else { 1.0 };
    }
    let dist = edit_distance(hypothesis, reference);
    dist as f32 / reference.len() as f32
}

/// Compute Character Error Rate (CER).
pub fn cer(hypothesis: &str, reference: &str) -> f32 {
    let hyp_chars: Vec<char> = hypothesis.chars().collect();
    let ref_chars: Vec<char> = reference.chars().collect();
    if ref_chars.is_empty() {
        return if hyp_chars.is_empty() { 0.0 } else { 1.0 };
    }
    let dist = edit_distance(&hyp_chars, &ref_chars);
    dist as f32 / ref_chars.len() as f32
}

/// ASR evaluation metrics bundle.
#[derive(Debug, Clone)]
pub struct AsrMetrics {
    pub wer: f32,
    pub cer: f32,
    pub substitutions: usize,
    pub deletions: usize,
    pub insertions: usize,
    pub reference_length: usize,
}

impl AsrMetrics {
    /// Compute detailed alignment statistics using DP backtrace.
    pub fn compute(hypothesis: &[&str], reference: &[&str]) -> Self {
        let m = hypothesis.len();
        let n = reference.len();
        if n == 0 {
            return Self {
                wer: if m == 0 { 0.0 } else { 1.0 },
                cer: 0.0,
                substitutions: 0,
                deletions: 0,
                insertions: m,
                reference_length: 0,
            };
        }

        // DP with backtrace
        let mut dp = vec![vec![(0usize, 0u8); n + 1]; m + 1]; // (cost, op)
        for i in 0..=m {
            dp[i][0] = (i, 1); // deletion
        }
        for j in 0..=n {
            dp[0][j] = (j, 2); // insertion
        }
        for i in 1..=m {
            for j in 1..=n {
                let sub_cost = dp[i - 1][j - 1].0
                    + if hypothesis[i - 1] == reference[j - 1] {
                        0
                    } else {
                        1
                    };
                let del_cost = dp[i - 1][j].0 + 1;
                let ins_cost = dp[i][j - 1].0 + 1;
                if sub_cost <= del_cost && sub_cost <= ins_cost {
                    dp[i][j] = (
                        sub_cost,
                        if hypothesis[i - 1] == reference[j - 1] {
                            0
                        } else {
                            3
                        },
                    );
                } else if del_cost <= ins_cost {
                    dp[i][j] = (del_cost, 1);
                } else {
                    dp[i][j] = (ins_cost, 2);
                }
            }
        }

        // Backtrace
        let mut subs = 0usize;
        let mut dels = 0usize;
        let mut ins = 0usize;
        let (mut i, mut j) = (m, n);
        while i > 0 || j > 0 {
            let op = dp[i][j].1;
            match op {
                0 => {
                    i -= 1;
                    j -= 1;
                } // match
                3 => {
                    subs += 1;
                    i -= 1;
                    j -= 1;
                } // sub
                1 => {
                    dels += 1;
                    i -= 1;
                } // deletion
                2 => {
                    ins += 1;
                    j -= 1;
                } // insertion
                _ => break,
            }
        }

        let total_err = subs + dels + ins;
        let wer_val = total_err as f32 / n as f32;
        let hyp_str = hypothesis.join(" ");
        let ref_str = reference.join(" ");
        let cer_val = cer(&hyp_str, &ref_str);

        Self {
            wer: wer_val,
            cer: cer_val,
            substitutions: subs,
            deletions: dels,
            insertions: ins,
            reference_length: n,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. SpeechPipeline
// ─────────────────────────────────────────────────────────────────────────────

/// End-to-end speech recognition pipeline.
///
/// Pipeline: VAD → log-mel spectrogram → WhisperEncoder → CTC greedy decode → join tokens.
#[derive(Debug, Clone)]
pub struct SpeechPipeline {
    pub vad: VoiceActivityDetector,
    pub encoder: WhisperEncoder,
    pub ctc_decoder: CtcBeamDecoder,
    /// Vocabulary: token_id → string (e.g. BPE subword pieces).
    pub vocabulary: Vec<String>,
    pub sample_rate: usize,
}

impl SpeechPipeline {
    /// Create a new pipeline with default configuration and random weights.
    pub fn new(
        vad_config: VadConfig,
        encoder_config: WhisperConfig,
        vocab_size: usize,
        vocabulary: Vec<String>,
        sample_rate: usize,
        seed: u64,
    ) -> Result<Self> {
        if vocabulary.len() != vocab_size {
            return Err(inval(
                "SpeechPipeline::new",
                format!(
                    "vocabulary.len()={} != vocab_size={}",
                    vocabulary.len(),
                    vocab_size
                ),
            ));
        }
        let ctc_config = CtcConfig {
            vocab_size,
            blank_id: 0,
            beam_width: 5,
        };
        Ok(Self {
            vad: VoiceActivityDetector::new(vad_config),
            encoder: WhisperEncoder::new(encoder_config, seed),
            ctc_decoder: CtcBeamDecoder::new(ctc_config),
            vocabulary,
            sample_rate,
        })
    }

    /// Transcribe raw audio samples to a string.
    ///
    /// Steps:
    /// 1. VAD to extract speech segments.
    /// 2. Compute log-mel spectrogram for each segment.
    /// 3. Encode with WhisperEncoder.
    /// 4. CTC greedy decode.
    /// 5. Join token strings.
    pub fn transcribe(&self, audio: &[f32], sample_rate: usize) -> String {
        if audio.is_empty() {
            return String::new();
        }
        let n_fft = self.encoder.config.n_fft;
        let hop_length = self.encoder.config.hop_length;
        let n_mels = self.encoder.config.n_mels;

        // VAD segmentation
        let segments = self.vad.segment(audio);
        let speech_audio: Vec<f32> = if segments.is_empty() {
            audio.to_vec()
        } else {
            segments
                .iter()
                .flat_map(|&(s, e)| audio[s..e].iter().copied())
                .collect()
        };

        // Log-mel spectrogram
        let log_mel = log_mel_spectrogram(&speech_audio, sample_rate, n_fft, hop_length, n_mels);
        if log_mel.is_empty() {
            return String::new();
        }

        // Encode
        let encoder_out = self.encoder.encode(&log_mel);
        if encoder_out.is_empty() {
            return String::new();
        }

        // CTC greedy decode using encoder outputs as frame-level log-probs
        // (in a real system these would be projected to vocab; here we use mock log-probs)
        let vocab = self.vocabulary.len();
        let log_probs: Vec<Vec<f32>> = encoder_out
            .iter()
            .map(|frame| {
                // Project encoder output to vocab logits via first `vocab` dims (or repeat)
                let raw: Vec<f32> = (0..vocab)
                    .map(|i| {
                        let idx = i % frame.len().max(1);
                        frame[idx]
                    })
                    .collect();
                log_softmax_vec(&raw)
            })
            .collect();

        let token_ids = self.ctc_decoder.greedy_decode(&log_probs);

        // Map token ids to strings
        token_ids
            .iter()
            .filter_map(|&id| self.vocabulary.get(id).map(|s| s.as_str()))
            .collect::<Vec<&str>>()
            .join(" ")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
