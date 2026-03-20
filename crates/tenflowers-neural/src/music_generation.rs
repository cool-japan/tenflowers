//! # Music Generation Module
//!
//! Comprehensive music generation, analysis, and synthesis for TenfloweRS.
//! Includes MIDI representation, piano roll encoding, transformer-based generation,
//! chord detection, rhythm models, audio synthesis, and music theory analysis.
//!
//! # Modules at a Glance
//!
//! | Section | Key Types |
//! |---------|-----------|
//! | MIDI Representation | [`NoteEvent`], [`MidiSequence`] |
//! | Piano-Roll Encoding | [`PianoRollEncoder`] |
//! | Transformer (Muse)  | [`MuseTransformer`] |
//! | Chord Detection     | [`ChordDetector`], [`Chord`], [`ChordDetection`] |
//! | Melody Generation   | [`MelodyGenerator`], [`GenerationConfig`], [`ScaleType`] |
//! | Rhythm Modelling    | [`BjorklundPattern`], [`RhythmGrid`] |
//! | Audio Synthesis     | [`AudioSynthesizer`], [`WaveType`], [`AdsrEnvelope`] |
//! | Variation Generator | [`MusicVariationGenerator`] |
//! | Theory Analyser     | [`MusicTheoryAnalyzer`] |
//! | Evaluation Metrics  | [`MusicMetrics`] |

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// §1  MIDI Representation
// ─────────────────────────────────────────────────────────────────────────────

/// Type of MIDI-like event.
#[derive(Debug, Clone, PartialEq)]
pub enum EventType {
    NoteOn,
    NoteOff,
    TempoChange,
    ProgramChange,
}

/// A single note event in a MIDI-like sequence.
#[derive(Debug, Clone)]
pub struct NoteEvent {
    /// MIDI pitch (0–127).
    pub pitch: u8,
    /// Velocity (0–127).
    pub velocity: u8,
    /// Onset time in ticks.
    pub onset_tick: u32,
    /// Duration in ticks.
    pub duration_tick: u32,
}

/// A MIDI-like sequence of note events.
#[derive(Debug, Clone)]
pub struct MidiSequence {
    pub events: Vec<NoteEvent>,
    pub tempo_bpm: f32,
    pub ticks_per_beat: u32,
}

impl MidiSequence {
    /// Create an empty sequence.
    pub fn new(tempo_bpm: f32, ticks_per_beat: u32) -> Self {
        Self {
            events: Vec::new(),
            tempo_bpm,
            ticks_per_beat,
        }
    }

    /// Total duration in ticks (end tick of the last-ending note).
    pub fn duration_ticks(&self) -> u32 {
        self.events
            .iter()
            .map(|e| e.onset_tick + e.duration_tick)
            .max()
            .unwrap_or(0)
    }

    /// Number of note events.
    pub fn note_count(&self) -> usize {
        self.events.len()
    }

    /// Convert to a piano roll matrix of shape `[time_steps][pitch_bins]`.
    /// Each cell contains the normalised velocity if a note is active, else 0.
    pub fn to_piano_roll(&self, time_steps: usize, pitch_bins: usize) -> Vec<Vec<f32>> {
        let total_ticks = self.duration_ticks().max(1);
        let mut roll = vec![vec![0.0f32; pitch_bins]; time_steps];
        for ev in &self.events {
            let p = ev.pitch as usize;
            if p >= pitch_bins {
                continue;
            }
            let start = ((ev.onset_tick as f64 / total_ticks as f64) * time_steps as f64) as usize;
            let end_tick = ev.onset_tick + ev.duration_tick;
            let end = ((end_tick as f64 / total_ticks as f64) * time_steps as f64) as usize;
            let end = end.min(time_steps);
            let vel = ev.velocity as f32 / 127.0;
            for t in start..end {
                roll[t][p] = vel;
            }
        }
        roll
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  Piano Roll Encoder
// ─────────────────────────────────────────────────────────────────────────────

/// Piano roll matrix encoder / decoder.
pub struct PianoRollEncoder {
    pub time_steps: usize,
    /// (low, high) inclusive MIDI pitch range.
    pub pitch_range: (u8, u8),
    pub velocity_normalize: bool,
}

impl PianoRollEncoder {
    pub fn new(time_steps: usize, pitch_range: (u8, u8), velocity_normalize: bool) -> Self {
        Self {
            time_steps,
            pitch_range,
            velocity_normalize,
        }
    }

    /// Number of pitch bins.
    fn pitch_bins(&self) -> usize {
        (self.pitch_range.1 as usize).saturating_sub(self.pitch_range.0 as usize) + 1
    }

    /// Encode a [`MidiSequence`] into a `[time_steps][pitch_bins]` matrix.
    pub fn encode(&self, seq: &MidiSequence) -> Vec<Vec<f32>> {
        let bins = self.pitch_bins();
        let total_ticks = seq.duration_ticks().max(1);
        let mut roll = vec![vec![0.0f32; bins]; self.time_steps];
        for ev in &seq.events {
            let p = ev.pitch as usize;
            let low = self.pitch_range.0 as usize;
            let high = self.pitch_range.1 as usize;
            if p < low || p > high {
                continue;
            }
            let bin = p - low;
            let start =
                ((ev.onset_tick as f64 / total_ticks as f64) * self.time_steps as f64) as usize;
            let end_tick = ev.onset_tick + ev.duration_tick;
            let end = ((end_tick as f64 / total_ticks as f64) * self.time_steps as f64) as usize;
            let end = end.min(self.time_steps);
            let vel = if self.velocity_normalize {
                ev.velocity as f32 / 127.0
            } else {
                ev.velocity as f32
            };
            for t in start..end {
                roll[t][bin] = vel;
            }
        }
        roll
    }

    /// Decode a piano roll back into a [`MidiSequence`].
    pub fn decode(&self, roll: &[Vec<f32>], threshold: f32) -> MidiSequence {
        let bins = self.pitch_bins();
        let low = self.pitch_range.0 as usize;
        let ticks_per_step: u32 = 48;
        let mut events = Vec::new();
        let mut active: Vec<Option<u32>> = vec![None; bins];
        for t in 0..roll.len() {
            let row = &roll[t];
            for b in 0..bins.min(row.len()) {
                let active_now = row[b] >= threshold;
                match active[b] {
                    None => {
                        if active_now {
                            active[b] = Some(t as u32);
                        }
                    }
                    Some(onset) => {
                        if !active_now {
                            events.push(NoteEvent {
                                pitch: (low + b) as u8,
                                velocity: 80,
                                onset_tick: onset * ticks_per_step,
                                duration_tick: (t as u32 - onset) * ticks_per_step,
                            });
                            active[b] = None;
                        }
                    }
                }
            }
        }
        let t_end = roll.len() as u32;
        for (b, onset_opt) in active.iter().enumerate() {
            if let Some(onset) = onset_opt {
                events.push(NoteEvent {
                    pitch: (low + b) as u8,
                    velocity: 80,
                    onset_tick: onset * ticks_per_step,
                    duration_tick: (t_end - onset) * ticks_per_step,
                });
            }
        }
        MidiSequence {
            events,
            tempo_bpm: 120.0,
            ticks_per_beat: 480,
        }
    }

    /// Flatten a piano roll to a 1-D tensor.
    pub fn roll_to_tensor(&self, roll: &[Vec<f32>]) -> Vec<f32> {
        roll.iter().flat_map(|row| row.iter().copied()).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  MuseTransformer — symbolic music generation
// ─────────────────────────────────────────────────────────────────────────────

/// Scaled-dot-product attention with causal mask (manual, no ndarray).
fn muse_attention(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    head_dim: usize,
) -> Vec<Vec<f32>> {
    let seq_len = q.len();
    let scale = (head_dim as f32).sqrt().max(1e-4);
    let mut out = vec![vec![0.0f32; head_dim]; seq_len];
    for i in 0..seq_len {
        let mut exp_sum = 0.0f32;
        let mut attn = vec![0.0f32; i + 1];
        let mut max_score = f32::NEG_INFINITY;
        for j in 0..=i {
            let dot: f32 = q[i].iter().zip(k[j].iter()).map(|(a, b)| a * b).sum();
            let s = dot / scale;
            if s > max_score {
                max_score = s;
            }
            attn[j] = s;
        }
        for j in 0..=i {
            attn[j] = (attn[j] - max_score).exp();
            exp_sum += attn[j];
        }
        for j in 0..=i {
            let a = attn[j] / (exp_sum + 1e-9);
            for d in 0..head_dim {
                out[i][d] += a * v[j][d];
            }
        }
    }
    out
}

/// Transformer-based event-level music generation model.
pub struct MuseTransformer {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub max_len: usize,
    /// Token embeddings: [vocab_size][d_model]
    embeddings: Vec<Vec<f32>>,
    /// Sinusoidal positional encodings: [max_len][d_model]
    pos_enc: Vec<Vec<f32>>,
    /// Output projection: [d_model][vocab_size]
    out_proj: Vec<Vec<f32>>,
}

impl MuseTransformer {
    /// Build with Xavier-initialised weights.
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        max_len: usize,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let scale_emb = (2.0 / (vocab_size + d_model) as f32).sqrt();
        let scale_out = (2.0 / (d_model + vocab_size) as f32).sqrt();

        let embeddings: Vec<Vec<f32>> = (0..vocab_size)
            .map(|_| {
                (0..d_model)
                    .map(|_| rng.random_range(-scale_emb..scale_emb))
                    .collect()
            })
            .collect();

        // Sinusoidal positional encoding
        let pos_enc: Vec<Vec<f32>> = (0..max_len)
            .map(|pos| {
                (0..d_model)
                    .map(|i| {
                        let denom = 10000f32.powf(2.0 * (i / 2) as f32 / d_model as f32);
                        if i % 2 == 0 {
                            (pos as f32 / denom).sin()
                        } else {
                            (pos as f32 / denom).cos()
                        }
                    })
                    .collect()
            })
            .collect();

        let out_proj: Vec<Vec<f32>> = (0..d_model)
            .map(|_| {
                (0..vocab_size)
                    .map(|_| rng.random_range(-scale_out..scale_out))
                    .collect()
            })
            .collect();

        Self {
            vocab_size,
            d_model,
            n_heads,
            n_layers,
            max_len,
            embeddings,
            pos_enc,
            out_proj,
        }
    }

    fn embed(&self, tokens: &[u32]) -> Vec<Vec<f32>> {
        tokens
            .iter()
            .enumerate()
            .map(|(pos, &tok)| {
                let idx = (tok as usize).min(self.vocab_size.saturating_sub(1));
                let pos = pos.min(self.max_len.saturating_sub(1));
                (0..self.d_model)
                    .map(|d| self.embeddings[idx][d] + self.pos_enc[pos][d])
                    .collect()
            })
            .collect()
    }

    fn layer_norm(x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let d = if x.is_empty() { 0 } else { x[0].len() };
        x.iter()
            .map(|v| {
                let mean = v.iter().sum::<f32>() / d.max(1) as f32;
                let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / d.max(1) as f32;
                let std = (var + 1e-6).sqrt();
                v.iter().map(|x| (x - mean) / std).collect()
            })
            .collect()
    }

    fn transformer_layer(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let head_dim = (self.d_model / self.n_heads.max(1)).max(1);
        let seq_len = x.len();
        let mut out = x.to_vec();
        // Multi-head self-attention with residual
        for _h in 0..self.n_heads {
            let q: Vec<Vec<f32>> = x.iter().map(|v| v[..head_dim].to_vec()).collect();
            let k: Vec<Vec<f32>> = x.iter().map(|v| v[..head_dim].to_vec()).collect();
            let v: Vec<Vec<f32>> = x.iter().map(|v| v[..head_dim].to_vec()).collect();
            let attn_out = muse_attention(&q, &k, &v, head_dim);
            for i in 0..seq_len {
                for d in 0..head_dim {
                    out[i][d] += attn_out[i][d];
                }
            }
        }
        Self::layer_norm(&out)
    }

    /// Forward pass: returns logits of shape `[seq_len][vocab_size]`.
    pub fn forward(&self, tokens: &[u32]) -> Vec<Vec<f32>> {
        let mut h = self.embed(tokens);
        for _ in 0..self.n_layers {
            h = self.transformer_layer(&h);
        }
        h.iter()
            .map(|hv| {
                (0..self.vocab_size)
                    .map(|v| {
                        hv.iter()
                            .enumerate()
                            .map(|(d, &x)| x * self.out_proj[d][v])
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }

    /// Autoregressive generation with temperature sampling.
    pub fn generate(
        &self,
        prompt: &[u32],
        max_new: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> Vec<u32> {
        let mut tokens: Vec<u32> = prompt.to_vec();
        let temp = temperature.max(1e-3);
        for _ in 0..max_new {
            let logits = self.forward(&tokens);
            let last = &logits[logits.len().saturating_sub(1)];
            let max_l = last.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = last.iter().map(|l| ((l - max_l) / temp).exp()).collect();
            let sum_e: f32 = exp.iter().sum::<f32>().max(1e-9);
            let probs: Vec<f32> = exp.iter().map(|e| e / sum_e).collect();
            let u: f32 = rng.random_range(0.0f32..1.0f32);
            let mut cum = 0.0f32;
            let mut chosen = self.vocab_size.saturating_sub(1);
            for (i, &p) in probs.iter().enumerate() {
                cum += p;
                if u < cum {
                    chosen = i;
                    break;
                }
            }
            tokens.push(chosen as u32);
        }
        tokens
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  Chord Detector
// ─────────────────────────────────────────────────────────────────────────────

/// Chord quality (triad/seventh chord type).
#[derive(Debug, Clone, PartialEq)]
pub enum Chord {
    Major,
    Minor,
    Diminished,
    Augmented,
    Dom7,
    Maj7,
    Min7,
}

/// Chord detection result.
#[derive(Debug, Clone)]
pub struct ChordDetection {
    /// Root pitch class (0 = C … 11 = B).
    pub root: u8,
    pub quality: Chord,
    pub confidence: f32,
}

// (quality, interval pattern in semitones from root)
const CHORD_TEMPLATES: &[(Chord, &[u8])] = &[
    (Chord::Major, &[0, 4, 7]),
    (Chord::Minor, &[0, 3, 7]),
    (Chord::Diminished, &[0, 3, 6]),
    (Chord::Augmented, &[0, 4, 8]),
    (Chord::Dom7, &[0, 4, 7, 10]),
    (Chord::Maj7, &[0, 4, 7, 11]),
    (Chord::Min7, &[0, 3, 7, 10]),
];

/// Chord recognition from pitch sets and piano rolls.
pub struct ChordDetector;

impl ChordDetector {
    /// Detect chord quality and root from a slice of MIDI pitches.
    pub fn detect_chord(pitches: &[u8]) -> ChordDetection {
        if pitches.is_empty() {
            return ChordDetection {
                root: 0,
                quality: Chord::Major,
                confidence: 0.0,
            };
        }
        let mut chroma = [0.0f32; 12];
        for &p in pitches {
            chroma[(p % 12) as usize] += 1.0;
        }
        let total: f32 = chroma.iter().sum::<f32>().max(1.0);
        for v in &mut chroma {
            *v /= total;
        }

        let mut best_score = -1.0f32;
        let mut best_root = 0u8;
        let mut best_quality = Chord::Major;

        for root in 0u8..12 {
            for (quality, intervals) in CHORD_TEMPLATES {
                let score: f32 = intervals
                    .iter()
                    .map(|&i| chroma[((root + i) % 12) as usize])
                    .sum();
                let norm_score = score / intervals.len() as f32;
                if norm_score > best_score {
                    best_score = norm_score;
                    best_root = root;
                    best_quality = quality.clone();
                }
            }
        }
        ChordDetection {
            root: best_root,
            quality: best_quality,
            confidence: best_score,
        }
    }

    /// Detect chords for each time step of a piano roll.
    pub fn detect_sequence(roll: &[Vec<f32>], threshold: f32) -> Vec<Option<ChordDetection>> {
        roll.iter()
            .map(|row| {
                let active: Vec<u8> = row
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| if v >= threshold { Some(i as u8) } else { None })
                    .collect();
                if active.is_empty() {
                    None
                } else {
                    Some(Self::detect_chord(&active))
                }
            })
            .collect()
    }

    /// Krumhansl-Kessler major and minor key profiles.
    fn kk_profiles() -> ([f32; 12], [f32; 12]) {
        let major = [
            6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
        ];
        let minor = [
            6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
        ];
        (major, minor)
    }

    /// Detect the key of a sequence using Krumhansl-Kessler profiles.
    /// Returns `(root, is_major)`.
    pub fn detect_key(sequence: &MidiSequence) -> (u8, bool) {
        let mut chroma = [0.0f32; 12];
        for ev in &sequence.events {
            chroma[(ev.pitch % 12) as usize] += ev.duration_tick as f32;
        }
        let total: f32 = chroma.iter().sum::<f32>().max(1.0);
        for v in &mut chroma {
            *v /= total;
        }
        let (major_profile, minor_profile) = Self::kk_profiles();
        let mut best_score = f32::NEG_INFINITY;
        let mut best_root = 0u8;
        let mut best_is_major = true;
        for root in 0u8..12 {
            let maj: f32 = (0..12)
                .map(|i| chroma[i] * major_profile[(i + 12 - root as usize) % 12])
                .sum();
            if maj > best_score {
                best_score = maj;
                best_root = root;
                best_is_major = true;
            }
            let min: f32 = (0..12)
                .map(|i| chroma[i] * minor_profile[(i + 12 - root as usize) % 12])
                .sum();
            if min > best_score {
                best_score = min;
                best_root = root;
                best_is_major = false;
            }
        }
        (best_root, best_is_major)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  Melody Generator
// ─────────────────────────────────────────────────────────────────────────────

/// Musical scale type.
#[derive(Debug, Clone, PartialEq)]
pub enum ScaleType {
    Major,
    Minor,
    Pentatonic,
    Blues,
    Chromatic,
}

impl ScaleType {
    /// Semitone intervals relative to the root note.
    pub fn intervals(&self) -> Vec<u8> {
        match self {
            ScaleType::Major => vec![0, 2, 4, 5, 7, 9, 11],
            ScaleType::Minor => vec![0, 2, 3, 5, 7, 8, 10],
            ScaleType::Pentatonic => vec![0, 2, 4, 7, 9],
            ScaleType::Blues => vec![0, 3, 5, 6, 7, 10],
            ScaleType::Chromatic => (0u8..12).collect(),
        }
    }
}

/// Configuration for rule-based melody generation.
pub struct GenerationConfig {
    /// Root MIDI pitch (0–127).
    pub root: u8,
    pub scale: ScaleType,
    pub tempo_bpm: f32,
    pub num_bars: usize,
    /// Melodic contour in [−1, 1]: positive = up, negative = down.
    pub contour: Vec<f32>,
}

/// Rule-based, contour-guided melody generator.
pub struct MelodyGenerator;

impl MelodyGenerator {
    /// Build all scale pitches in the playable range [21, 108].
    fn build_scale(root: u8, scale: &ScaleType) -> Vec<u8> {
        let intervals = scale.intervals();
        let root_pc = root % 12;
        let mut pitches = Vec::new();
        for octave in 0u8..=9 {
            for &interval in &intervals {
                let p = octave * 12 + root_pc + interval;
                if (21..=108).contains(&p) {
                    pitches.push(p);
                }
            }
        }
        pitches.sort_unstable();
        pitches.dedup();
        pitches
    }

    /// Generate a melody sequence.
    pub fn generate(config: &GenerationConfig, rng: &mut impl Rng) -> MidiSequence {
        let scale_pitches = Self::build_scale(config.root, &config.scale);
        if scale_pitches.is_empty() {
            return MidiSequence::new(config.tempo_bpm, 480);
        }
        let ticks_per_beat: u32 = 480;
        let ticks_per_eighth = ticks_per_beat / 2;
        let total_steps = config.num_bars * 8;
        let mut events = Vec::new();
        let mut current_idx = scale_pitches.len() / 2;
        let contour_len = config.contour.len();

        for step in 0..total_steps {
            let contour_val = if contour_len > 0 {
                config.contour[step % contour_len]
            } else {
                0.0
            };
            let move_up_prob = (0.5 + contour_val * 0.4).clamp(0.05, 0.95);
            let rv: f32 = rng.random_range(0.0f32..1.0f32);
            let step_size: i32 = if rv < 0.15 {
                0
            } else if rv < move_up_prob {
                rng.random_range(1i32..3i32)
            } else {
                -rng.random_range(1i32..3i32)
            };
            current_idx =
                (current_idx as i32 + step_size).clamp(0, scale_pitches.len() as i32 - 1) as usize;
            let pitch = scale_pitches[current_idx];
            let onset = step as u32 * ticks_per_eighth;
            let rest_rv: f32 = rng.random_range(0.0f32..1.0f32);
            if rest_rv > 0.20 {
                let vel_extra: u8 = rng.random_range(0i32..32i32) as u8;
                events.push(NoteEvent {
                    pitch,
                    velocity: 64 + vel_extra,
                    onset_tick: onset,
                    duration_tick: ticks_per_eighth,
                });
            }
        }
        MidiSequence {
            events,
            tempo_bpm: config.tempo_bpm,
            ticks_per_beat,
        }
    }

    /// Apply swing: push off-beat eighth notes forward by `swing_ratio` amount.
    /// Classic triplet swing uses `swing_ratio = 0.67`.
    pub fn apply_swing(seq: &mut MidiSequence, swing_ratio: f32) {
        let ticks_per_eighth = seq.ticks_per_beat / 2;
        if ticks_per_eighth == 0 {
            return;
        }
        for ev in &mut seq.events {
            let eighth_pos = ev.onset_tick / ticks_per_eighth;
            if eighth_pos % 2 == 1 {
                let nominal = eighth_pos * ticks_per_eighth;
                let shift = ticks_per_eighth as f32 * (swing_ratio - 0.5) * 2.0;
                ev.onset_tick = (nominal as f32 + shift).max(0.0) as u32;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  Rhythm Model
// ─────────────────────────────────────────────────────────────────────────────

/// Euclidean rhythm generator using the Bjorklund algorithm.
pub struct BjorklundPattern {
    pulses: usize,
    steps: usize,
}

impl BjorklundPattern {
    pub fn new(pulses: usize, steps: usize) -> Self {
        Self { pulses, steps }
    }

    /// Generate the Euclidean rhythm pattern.
    pub fn pattern(&self) -> Vec<bool> {
        if self.steps == 0 {
            return Vec::new();
        }
        let pulses = self.pulses.min(self.steps);
        if pulses == 0 {
            return vec![false; self.steps];
        }
        // Bjorklund algorithm via iterative remainder grouping
        let mut ones = pulses;
        let mut zeros = self.steps - pulses;
        // Represent as groups of [1] and [0]
        let mut groups: Vec<Vec<bool>> = (0..ones).map(|_| vec![true]).collect();
        let mut rem: Vec<Vec<bool>> = (0..zeros).map(|_| vec![false]).collect();
        loop {
            if rem.len() <= 1 {
                break;
            }
            let take = ones.min(rem.len());
            let new_groups: Vec<Vec<bool>> = (0..take)
                .map(|i| {
                    let mut g = groups[i].clone();
                    g.extend_from_slice(&rem[i]);
                    g
                })
                .collect();
            let new_rem: Vec<Vec<bool>> = if ones >= rem.len() {
                groups[take..].to_vec()
            } else {
                rem[take..].to_vec()
            };
            ones = new_groups.len();
            zeros = new_rem.len();
            groups = new_groups;
            rem = new_rem;
        }
        let mut result: Vec<bool> = groups.into_iter().flatten().collect();
        result.extend(rem.into_iter().flatten());
        result
    }
}

/// Drum style for pattern generation.
#[derive(Debug, Clone, PartialEq)]
pub enum DrumStyle {
    Rock,
    Jazz,
    Latin,
    Electronic,
}

/// A grid of boolean rhythm patterns (tracks × steps).
pub struct RhythmGrid {
    pub beats_per_bar: usize,
    pub subdivisions: usize,
    /// Rows = instrument tracks, columns = time steps.
    pub patterns: Vec<Vec<bool>>,
}

impl RhythmGrid {
    pub fn steps(&self) -> usize {
        self.beats_per_bar * self.subdivisions
    }
}

/// Generate a drum pattern for the given style over `bars` bars.
pub fn generate_drum_pattern(style: DrumStyle, bars: usize, rng: &mut impl Rng) -> RhythmGrid {
    let beats = 4usize;
    let subdiv = 4usize; // 16th-note grid
    let steps = beats * subdiv; // 16 steps/bar
    let total = steps * bars;

    let (kick_base, snare_base, hihat_pulses) = match style {
        DrumStyle::Rock => {
            let mut kick = vec![false; steps];
            kick[0] = true;
            kick[8] = true;
            let mut snare = vec![false; steps];
            snare[4] = true;
            snare[12] = true;
            (kick, snare, 12usize)
        }
        DrumStyle::Jazz => (
            BjorklundPattern::new(2, steps).pattern(),
            BjorklundPattern::new(3, steps).pattern(),
            8usize,
        ),
        DrumStyle::Latin => (
            BjorklundPattern::new(3, steps).pattern(),
            BjorklundPattern::new(5, steps).pattern(),
            11usize,
        ),
        DrumStyle::Electronic => {
            let mut kick = vec![false; steps];
            kick[0] = true;
            kick[8] = true;
            let mut snare = vec![false; steps];
            snare[4] = true;
            snare[12] = true;
            (kick, snare, 16usize)
        }
    };
    let hihat_base = BjorklundPattern::new(hihat_pulses.min(steps), steps).pattern();

    let mut tile_with_noise = |base: &[bool]| -> Vec<bool> {
        let mut out = Vec::with_capacity(total);
        for _ in 0..bars {
            out.extend_from_slice(base);
        }
        for v in &mut out {
            let r: f32 = rng.random_range(0.0f32..1.0f32);
            if r < 0.05 {
                *v = !*v;
            }
        }
        out
    };

    RhythmGrid {
        beats_per_bar: beats,
        subdivisions: subdiv,
        patterns: vec![
            tile_with_noise(&kick_base),
            tile_with_noise(&snare_base),
            tile_with_noise(&hihat_base),
        ],
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  Audio Synthesizer
// ─────────────────────────────────────────────────────────────────────────────

/// Basic waveform type.
#[derive(Debug, Clone, PartialEq)]
pub enum WaveType {
    Sine,
    Square,
    Sawtooth,
    Triangle,
    Noise,
}

/// ADSR envelope parameters.
#[derive(Debug, Clone)]
pub struct AdsrEnvelope {
    pub attack_ms: f32,
    pub decay_ms: f32,
    /// Sustain level in [0, 1].
    pub sustain: f32,
    pub release_ms: f32,
}

/// Synthesizer configuration.
#[derive(Debug, Clone)]
pub struct SynthConfig {
    pub wave: WaveType,
    pub adsr: AdsrEnvelope,
    pub sample_rate: u32,
}

/// Simple wavetable-style audio synthesizer.
pub struct AudioSynthesizer;

impl AudioSynthesizer {
    /// MIDI pitch → frequency in Hz (A4 = 440 Hz = MIDI 69).
    pub fn pitch_to_hz(midi_pitch: u8) -> f32 {
        440.0 * 2.0f32.powf((midi_pitch as f32 - 69.0) / 12.0)
    }

    /// Synthesize a single note as PCM samples.
    pub fn synthesize_note(pitch_hz: f32, duration_s: f32, config: &SynthConfig) -> Vec<f32> {
        let sr = config.sample_rate as f32;
        let total = (duration_s * sr) as usize;
        let attack_n = (config.adsr.attack_ms / 1000.0 * sr) as usize;
        let decay_n = (config.adsr.decay_ms / 1000.0 * sr) as usize;
        let release_n = (config.adsr.release_ms / 1000.0 * sr) as usize;
        let decay_end = attack_n + decay_n;
        let release_start = if total > release_n {
            total - release_n
        } else {
            total
        };
        let sustain = config.adsr.sustain.clamp(0.0, 1.0);
        let mut noise_rng = StdRng::seed_from_u64(77777);
        let mut samples = Vec::with_capacity(total);
        for n in 0..total {
            let t = n as f32 / sr;
            let phase = 2.0 * std::f32::consts::PI * pitch_hz * t;
            let wave = match config.wave {
                WaveType::Sine => phase.sin(),
                WaveType::Square => {
                    if phase.rem_euclid(2.0 * std::f32::consts::PI) < std::f32::consts::PI {
                        1.0f32
                    } else {
                        -1.0f32
                    }
                }
                WaveType::Sawtooth => 2.0 * (pitch_hz * t).fract() - 1.0,
                WaveType::Triangle => {
                    let frac = (pitch_hz * t).fract();
                    if frac < 0.5 {
                        4.0 * frac - 1.0
                    } else {
                        3.0 - 4.0 * frac
                    }
                }
                WaveType::Noise => noise_rng.random_range(-1.0f32..1.0f32),
            };
            let env = if n < attack_n {
                if attack_n > 0 {
                    n as f32 / attack_n as f32
                } else {
                    1.0
                }
            } else if n < decay_end {
                let t_d = (n - attack_n) as f32 / decay_n.max(1) as f32;
                1.0 - t_d * (1.0 - sustain)
            } else if n < release_start {
                sustain
            } else {
                let t_r = (n - release_start) as f32 / release_n.max(1) as f32;
                sustain * (1.0 - t_r)
            };
            samples.push(wave * env);
        }
        samples
    }

    /// Synthesize a full MIDI sequence into a mixed PCM buffer.
    pub fn synthesize_sequence(seq: &MidiSequence, config: &SynthConfig) -> Vec<f32> {
        if seq.events.is_empty() {
            return Vec::new();
        }
        let sr = config.sample_rate as f32;
        let beats_per_sec = seq.tempo_bpm / 60.0;
        let ticks_per_sec = beats_per_sec * seq.ticks_per_beat as f32;
        let total_ticks = seq.duration_ticks();
        let total_sec = total_ticks as f32 / ticks_per_sec.max(1e-3);
        let total_samples = (total_sec * sr) as usize + config.sample_rate as usize;
        let mut buffer = vec![0.0f32; total_samples];
        for ev in &seq.events {
            let onset_sec = ev.onset_tick as f32 / ticks_per_sec.max(1e-3);
            let dur_sec = ev.duration_tick as f32 / ticks_per_sec.max(1e-3);
            let onset_sample = (onset_sec * sr) as usize;
            let freq = Self::pitch_to_hz(ev.pitch);
            let note_samples = Self::synthesize_note(freq, dur_sec, config);
            let amp = ev.velocity as f32 / 127.0;
            for (i, &s) in note_samples.iter().enumerate() {
                let idx = onset_sample + i;
                if idx < buffer.len() {
                    buffer[idx] += s * amp;
                }
            }
        }
        buffer
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  Music Variation Generator
// ─────────────────────────────────────────────────────────────────────────────

/// Classical music transformation operations.
pub struct MusicVariationGenerator;

impl MusicVariationGenerator {
    /// Transpose all notes by `semitones` (clamped to 0–127).
    pub fn transpose(seq: &MidiSequence, semitones: i32) -> MidiSequence {
        let events = seq
            .events
            .iter()
            .map(|ev| NoteEvent {
                pitch: (ev.pitch as i32 + semitones).clamp(0, 127) as u8,
                ..*ev
            })
            .collect();
        MidiSequence {
            events,
            ..seq.clone()
        }
    }

    /// Invert the melody around `pivot` pitch (melodic inversion).
    pub fn invert(seq: &MidiSequence, pivot: u8) -> MidiSequence {
        let events = seq
            .events
            .iter()
            .map(|ev| {
                let interval = ev.pitch as i32 - pivot as i32;
                let new_pitch = (pivot as i32 - interval).clamp(0, 127) as u8;
                NoteEvent {
                    pitch: new_pitch,
                    ..*ev
                }
            })
            .collect();
        MidiSequence {
            events,
            ..seq.clone()
        }
    }

    /// Retrograde: reverse the time ordering of notes.
    pub fn retrograde(seq: &MidiSequence) -> MidiSequence {
        let total = seq.duration_ticks();
        let mut events: Vec<NoteEvent> = seq
            .events
            .iter()
            .map(|ev| {
                let new_onset = total.saturating_sub(ev.onset_tick + ev.duration_tick);
                NoteEvent {
                    onset_tick: new_onset,
                    ..*ev
                }
            })
            .collect();
        events.sort_by_key(|e| e.onset_tick);
        MidiSequence {
            events,
            ..seq.clone()
        }
    }

    /// Augment: stretch time by `factor`.
    pub fn augment(seq: &MidiSequence, factor: f32) -> MidiSequence {
        let events = seq
            .events
            .iter()
            .map(|ev| NoteEvent {
                onset_tick: (ev.onset_tick as f32 * factor) as u32,
                duration_tick: (ev.duration_tick as f32 * factor) as u32,
                ..*ev
            })
            .collect();
        MidiSequence {
            events,
            ..seq.clone()
        }
    }

    /// Diminish: compress time by `factor`.
    pub fn diminish(seq: &MidiSequence, factor: f32) -> MidiSequence {
        let div = factor.max(1e-6);
        let events = seq
            .events
            .iter()
            .map(|ev| NoteEvent {
                onset_tick: (ev.onset_tick as f32 / div) as u32,
                duration_tick: (ev.duration_tick as f32 / div) as u32,
                ..*ev
            })
            .collect();
        MidiSequence {
            events,
            ..seq.clone()
        }
    }

    /// Add a parallel harmony voice `interval_semitones` above/below.
    pub fn harmonize(seq: &MidiSequence, interval_semitones: i32) -> MidiSequence {
        let mut events = seq.events.clone();
        let harmony: Vec<NoteEvent> = seq
            .events
            .iter()
            .map(|ev| NoteEvent {
                pitch: (ev.pitch as i32 + interval_semitones).clamp(0, 127) as u8,
                velocity: (ev.velocity as u16 * 7 / 8) as u8,
                ..*ev
            })
            .collect();
        events.extend(harmony);
        events.sort_by_key(|e| e.onset_tick);
        MidiSequence {
            events,
            ..seq.clone()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  Music Theory Analyzer
// ─────────────────────────────────────────────────────────────────────────────

/// Voice-leading analysis between two simultaneous sequences.
#[derive(Debug, Clone)]
pub struct VoiceLeadingAnalysis {
    pub parallel_fifths: usize,
    pub parallel_octaves: usize,
    pub contrary_motion_ratio: f32,
}

/// Music theory analysis utilities.
pub struct MusicTheoryAnalyzer;

impl MusicTheoryAnalyzer {
    /// Return note name for a MIDI pitch (pitch-class, not including octave).
    pub fn note_name(midi: u8) -> &'static str {
        const NAMES: &[&str] = &[
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        NAMES[(midi % 12) as usize]
    }

    /// Human-readable interval name for a semitone count.
    pub fn interval_name(semitones: u8) -> &'static str {
        const NAMES: &[&str] = &[
            "P1", "m2", "M2", "m3", "M3", "P4", "TT", "P5", "m6", "M6", "m7", "M7",
        ];
        NAMES[(semitones % 12) as usize]
    }

    /// Analyse voice-leading between two sequences (treated as voice 1 and 2).
    pub fn analyze_voice_leading(seq1: &MidiSequence, seq2: &MidiSequence) -> VoiceLeadingAnalysis {
        let len = seq1.events.len().min(seq2.events.len());
        if len < 2 {
            return VoiceLeadingAnalysis {
                parallel_fifths: 0,
                parallel_octaves: 0,
                contrary_motion_ratio: 0.0,
            };
        }
        let mut parallel_fifths = 0usize;
        let mut parallel_octaves = 0usize;
        let mut contrary = 0usize;
        for i in 0..(len - 1) {
            let p1a = seq1.events[i].pitch as i32;
            let p1b = seq1.events[i + 1].pitch as i32;
            let p2a = seq2.events[i].pitch as i32;
            let p2b = seq2.events[i + 1].pitch as i32;
            let int_a = (p2a - p1a).abs() % 12;
            let int_b = (p2b - p1b).abs() % 12;
            let m1 = p1b - p1a;
            let m2 = p2b - p2a;
            if int_a == 7 && int_b == 7 && m1 != 0 && m2 != 0 && m1.signum() == m2.signum() {
                parallel_fifths += 1;
            }
            if int_a == 0 && int_b == 0 && m1 != 0 && m2 != 0 && m1.signum() == m2.signum() {
                parallel_octaves += 1;
            }
            if (m1 > 0 && m2 < 0) || (m1 < 0 && m2 > 0) {
                contrary += 1;
            }
        }
        VoiceLeadingAnalysis {
            parallel_fifths,
            parallel_octaves,
            contrary_motion_ratio: contrary as f32 / (len - 1) as f32,
        }
    }

    /// Compute NxN cosine-similarity self-similarity matrix from piano-roll rows.
    pub fn compute_self_similarity(roll: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = roll.len();
        let mut sim = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            for j in 0..n {
                let dot: f32 = roll[i].iter().zip(roll[j].iter()).map(|(a, b)| a * b).sum();
                let ni: f32 = roll[i].iter().map(|v| v * v).sum::<f32>().sqrt();
                let nj: f32 = roll[j].iter().map(|v| v * v).sum::<f32>().sqrt();
                let denom = ni * nj;
                sim[i][j] = if denom > 1e-9 { dot / denom } else { 0.0 };
            }
        }
        sim
    }

    /// Detect recurring motifs of `motif_len` notes (interval-based, transpose-invariant).
    /// Returns `(start_index, similarity)` pairs.
    pub fn detect_motif(seq: &MidiSequence, motif_len: usize) -> Vec<(usize, f32)> {
        if seq.events.len() < motif_len * 2 || motif_len == 0 {
            return Vec::new();
        }
        let template: Vec<i32> = seq.events[..motif_len]
            .iter()
            .map(|e| e.pitch as i32)
            .collect();
        let template_iv: Vec<i32> = template.windows(2).map(|w| w[1] - w[0]).collect();
        let mut results = Vec::new();
        for start in 1..=(seq.events.len() - motif_len) {
            let window: Vec<i32> = seq.events[start..start + motif_len]
                .iter()
                .map(|e| e.pitch as i32)
                .collect();
            let window_iv: Vec<i32> = window.windows(2).map(|w| w[1] - w[0]).collect();
            let sim = if motif_len > 1 {
                let matches = template_iv
                    .iter()
                    .zip(window_iv.iter())
                    .filter(|(a, b)| a == b)
                    .count();
                matches as f32 / (motif_len - 1) as f32
            } else {
                if template[0] == window[0] {
                    1.0
                } else {
                    0.0
                }
            };
            results.push((start, sim));
        }
        results
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  Music Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Full evaluation report for a music sequence.
#[derive(Debug, Clone)]
pub struct MusicEvalReport {
    pub density: f32,
    pub pitch_range: u8,
    pub polyphony: f32,
    pub regularity: f32,
}

/// Music quality evaluation metrics.
pub struct MusicMetrics;

impl MusicMetrics {
    /// Notes per beat.
    pub fn note_density(seq: &MidiSequence) -> f32 {
        if seq.events.is_empty() {
            return 0.0;
        }
        let total_ticks = seq.duration_ticks() as f32;
        let beats = total_ticks / seq.ticks_per_beat as f32;
        if beats < 1e-9 {
            return 0.0;
        }
        seq.events.len() as f32 / beats
    }

    /// Semitone range from lowest to highest pitch.
    pub fn pitch_range(seq: &MidiSequence) -> u8 {
        if seq.events.is_empty() {
            return 0;
        }
        let max_p = seq.events.iter().map(|e| e.pitch).max().unwrap_or(0);
        let min_p = seq.events.iter().map(|e| e.pitch).min().unwrap_or(0);
        max_p.saturating_sub(min_p)
    }

    /// Fraction of time steps with more than one simultaneous note active.
    pub fn polyphony_ratio(roll: &[Vec<f32>], threshold: f32) -> f32 {
        if roll.is_empty() {
            return 0.0;
        }
        let poly = roll
            .iter()
            .filter(|row| row.iter().filter(|&&v| v >= threshold).count() > 1)
            .count();
        poly as f32 / roll.len() as f32
    }

    /// Duration-weighted pitch-class histogram (12 bins, normalised to sum 1).
    pub fn pitch_class_histogram(seq: &MidiSequence) -> [f32; 12] {
        let mut hist = [0.0f32; 12];
        for ev in &seq.events {
            hist[(ev.pitch % 12) as usize] += ev.duration_tick as f32;
        }
        let total: f32 = hist.iter().sum();
        if total > 1e-9 {
            for v in &mut hist {
                *v /= total;
            }
        }
        hist
    }

    /// Rhythmic regularity in [0, 1] based on the coefficient of variation of
    /// inter-onset intervals.
    pub fn rhythmic_regularity(seq: &MidiSequence) -> f32 {
        if seq.events.len() < 2 {
            return 0.0;
        }
        let mut onsets: Vec<u32> = seq.events.iter().map(|e| e.onset_tick).collect();
        onsets.sort_unstable();
        let ioi: Vec<f32> = onsets
            .windows(2)
            .map(|w| (w[1] - w[0]) as f32)
            .filter(|&v| v > 0.0)
            .collect();
        if ioi.is_empty() {
            return 1.0;
        }
        let mean = ioi.iter().sum::<f32>() / ioi.len() as f32;
        if mean < 1e-9 {
            return 0.0;
        }
        let var = ioi.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / ioi.len() as f32;
        let cv = var.sqrt() / mean;
        (1.0 / (1.0 + cv)).clamp(0.0, 1.0)
    }

    /// Produce a full evaluation report.
    pub fn evaluate(seq: &MidiSequence, roll: &[Vec<f32>]) -> MusicEvalReport {
        MusicEvalReport {
            density: Self::note_density(seq),
            pitch_range: Self::pitch_range(seq),
            polyphony: Self::polyphony_ratio(roll, 0.1),
            regularity: Self::rhythmic_regularity(seq),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: C major scale C4–C5 (8 notes, each 480 ticks)
    fn make_c_major_seq() -> MidiSequence {
        let notes = [60u8, 62, 64, 65, 67, 69, 71, 72];
        let mut seq = MidiSequence::new(120.0, 480);
        for (i, &p) in notes.iter().enumerate() {
            seq.events.push(NoteEvent {
                pitch: p,
                velocity: 80,
                onset_tick: i as u32 * 480,
                duration_tick: 480,
            });
        }
        seq
    }

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    // ── §1 MIDI ───────────────────────────────────────────────────────────────

    #[test]
    fn test_midi_sequence_basic() {
        let seq = make_c_major_seq();
        assert_eq!(seq.note_count(), 8);
        assert!(seq.duration_ticks() > 0);
        assert_eq!(seq.tempo_bpm, 120.0);
    }

    #[test]
    fn test_midi_sequence_duration() {
        let seq = make_c_major_seq();
        // Last note onset=7*480=3360, dur=480 → end=3840
        assert_eq!(seq.duration_ticks(), 3840);
    }

    // ── §2 Piano Roll ─────────────────────────────────────────────────────────

    #[test]
    fn test_piano_roll_dimensions() {
        let seq = make_c_major_seq();
        let roll = seq.to_piano_roll(32, 128);
        assert_eq!(roll.len(), 32);
        assert_eq!(roll[0].len(), 128);
    }

    #[test]
    fn test_piano_roll_encoder_roundtrip() {
        let seq = make_c_major_seq();
        let enc = PianoRollEncoder::new(64, (48, 84), true);
        let roll = enc.encode(&seq);
        let decoded = enc.decode(&roll, 0.1);
        // Should recover at least 4 notes
        assert!(decoded.note_count() >= 4);
    }

    #[test]
    fn test_piano_roll_to_tensor() {
        let enc = PianoRollEncoder::new(8, (60, 72), true);
        let roll = vec![vec![0.5f32; 13]; 8];
        let tensor = enc.roll_to_tensor(&roll);
        assert_eq!(tensor.len(), 8 * 13);
        assert!(tensor.iter().all(|&v| (v - 0.5).abs() < 1e-6));
    }

    // ── §3 MuseTransformer ────────────────────────────────────────────────────

    #[test]
    fn test_muse_transformer_forward_shape() {
        let mt = MuseTransformer::new(128, 32, 4, 2, 64);
        let tokens = vec![0u32, 1, 2, 3, 4];
        let logits = mt.forward(&tokens);
        assert_eq!(logits.len(), 5);
        assert_eq!(logits[0].len(), 128);
    }

    #[test]
    fn test_muse_transformer_generate_length() {
        let mt = MuseTransformer::new(64, 16, 2, 1, 32);
        let mut rng = make_rng();
        let prompt = vec![0u32, 1, 2];
        let generated = mt.generate(&prompt, 5, 1.0, &mut rng);
        assert_eq!(generated.len(), 8); // prompt(3) + new(5)
    }

    // ── §4 Chord Detector ─────────────────────────────────────────────────────

    #[test]
    fn test_chord_detector_major() {
        // C major triad: C(0), E(4), G(7)
        let pitches = [60u8, 64, 67];
        let det = ChordDetector::detect_chord(&pitches);
        assert_eq!(det.root, 0);
        assert_eq!(det.quality, Chord::Major);
        assert!(det.confidence > 0.0);
    }

    #[test]
    fn test_chord_detector_minor() {
        // A minor triad: A(9), C(0), E(4)
        let pitches = [57u8, 60, 64];
        let det = ChordDetector::detect_chord(&pitches);
        assert!(det.confidence > 0.0);
    }

    #[test]
    fn test_chord_sequence_detection() {
        let roll = vec![
            {
                let mut v = vec![0.0f32; 12];
                v[0] = 0.8;
                v[4] = 0.8;
                v[7] = 0.8;
                v
            },
            vec![0.0f32; 12],
        ];
        let seq_det = ChordDetector::detect_sequence(&roll, 0.5);
        assert_eq!(seq_det.len(), 2);
        assert!(seq_det[0].is_some());
        assert!(seq_det[1].is_none());
    }

    #[test]
    fn test_key_detection() {
        let seq = make_c_major_seq();
        let (root, is_major) = ChordDetector::detect_key(&seq);
        assert_eq!(root, 0); // C
        assert!(is_major);
    }

    // ── §5 Melody Generator ───────────────────────────────────────────────────

    #[test]
    fn test_melody_generator_scale_constraint() {
        let mut rng = make_rng();
        let intervals = ScaleType::Major.intervals();
        let config = GenerationConfig {
            root: 0,
            scale: ScaleType::Major,
            tempo_bpm: 120.0,
            num_bars: 2,
            contour: vec![0.0, 0.2, -0.1],
        };
        let seq = MelodyGenerator::generate(&config, &mut rng);
        for ev in &seq.events {
            let pc = ev.pitch % 12;
            assert!(
                intervals.contains(&pc),
                "pitch {} (pc {}) not in C major scale",
                ev.pitch,
                pc
            );
        }
    }

    #[test]
    fn test_melody_generator_contour() {
        let mut rng = make_rng();
        let config = GenerationConfig {
            root: 0,
            scale: ScaleType::Pentatonic,
            tempo_bpm: 100.0,
            num_bars: 4,
            contour: vec![1.0, 1.0, -1.0, -1.0],
        };
        let seq = MelodyGenerator::generate(&config, &mut rng);
        assert!(!seq.events.is_empty());
    }

    #[test]
    fn test_scale_chromatic_all_pitches() {
        let mut rng = StdRng::seed_from_u64(7);
        let config = GenerationConfig {
            root: 0,
            scale: ScaleType::Chromatic,
            tempo_bpm: 120.0,
            num_bars: 1,
            contour: vec![],
        };
        let seq = MelodyGenerator::generate(&config, &mut rng);
        for ev in &seq.events {
            assert!(ev.pitch <= 127);
        }
    }

    #[test]
    fn test_melody_swing() {
        let mut seq = MidiSequence::new(120.0, 480);
        seq.events.push(NoteEvent {
            pitch: 60,
            velocity: 80,
            onset_tick: 0,
            duration_tick: 240,
        });
        seq.events.push(NoteEvent {
            pitch: 62,
            velocity: 80,
            onset_tick: 240,
            duration_tick: 240,
        });
        MelodyGenerator::apply_swing(&mut seq, 0.67);
        // Off-beat note should be shifted
        assert_ne!(seq.events[1].onset_tick, 240);
    }

    // ── §6 Rhythm ─────────────────────────────────────────────────────────────

    #[test]
    fn test_bjorklund_3_8() {
        let pat = BjorklundPattern::new(3, 8).pattern();
        assert_eq!(pat.len(), 8);
        assert_eq!(pat.iter().filter(|&&v| v).count(), 3);
    }

    #[test]
    fn test_bjorklund_5_8() {
        let pat = BjorklundPattern::new(5, 8).pattern();
        assert_eq!(pat.len(), 8);
        assert_eq!(pat.iter().filter(|&&v| v).count(), 5);
    }

    #[test]
    fn test_rhythm_euclidean_total_pulses() {
        for &(p, s) in &[(3, 8), (5, 16), (7, 12), (1, 4), (4, 4)] {
            let pat = BjorklundPattern::new(p, s).pattern();
            assert_eq!(pat.len(), s, "len for ({p},{s})");
            assert_eq!(
                pat.iter().filter(|&&v| v).count(),
                p,
                "pulses for ({p},{s})"
            );
        }
    }

    #[test]
    fn test_rhythm_drum_pattern_shape() {
        let mut rng = make_rng();
        let grid = generate_drum_pattern(DrumStyle::Rock, 2, &mut rng);
        assert_eq!(grid.patterns.len(), 3); // kick, snare, hihat
        assert_eq!(grid.patterns[0].len(), 32); // 16 × 2 bars
    }

    #[test]
    fn test_generate_drum_rock() {
        let mut rng = make_rng();
        let grid = generate_drum_pattern(DrumStyle::Rock, 1, &mut rng);
        assert!(!grid.patterns.is_empty());
        assert_eq!(grid.patterns[0].len(), grid.steps());
    }

    // ── §7 Audio Synthesizer ──────────────────────────────────────────────────

    #[test]
    fn test_pitch_to_hz() {
        let a4 = AudioSynthesizer::pitch_to_hz(69);
        assert!((a4 - 440.0).abs() < 0.01, "A4={a4}");
        let c4 = AudioSynthesizer::pitch_to_hz(60);
        assert!((c4 - 261.626).abs() < 0.1, "C4={c4}");
    }

    #[test]
    fn test_audio_synthesizer_length() {
        let config = SynthConfig {
            wave: WaveType::Sine,
            adsr: AdsrEnvelope {
                attack_ms: 10.0,
                decay_ms: 10.0,
                sustain: 0.8,
                release_ms: 20.0,
            },
            sample_rate: 44100,
        };
        let samples = AudioSynthesizer::synthesize_note(440.0, 1.0, &config);
        assert_eq!(samples.len(), 44100);
    }

    #[test]
    fn test_adsr_envelope_shape() {
        let config = SynthConfig {
            wave: WaveType::Sine,
            adsr: AdsrEnvelope {
                attack_ms: 100.0,
                decay_ms: 100.0,
                sustain: 0.5,
                release_ms: 100.0,
            },
            sample_rate: 44100,
        };
        let samples = AudioSynthesizer::synthesize_note(440.0, 1.0, &config);
        let attack_end = (0.1 * 44100.0) as usize;
        let decay_end = attack_end + (0.1 * 44100.0) as usize;
        let release_start = 44100 - (0.1 * 44100.0) as usize;
        // Attack rises
        let early = samples[attack_end / 4].abs();
        let late_atk = samples[attack_end * 3 / 4].abs();
        assert!(late_atk >= early * 0.5, "attack should rise");
        // Sustain: roughly flat
        let s1 = samples[decay_end + (release_start - decay_end) / 3].abs();
        let s2 = samples[decay_end + (release_start - decay_end) * 2 / 3].abs();
        assert!((s1 - s2).abs() < 0.35, "sustain should be flat");
    }

    #[test]
    fn test_synthesize_sine() {
        let config = SynthConfig {
            wave: WaveType::Sine,
            adsr: AdsrEnvelope {
                attack_ms: 1.0,
                decay_ms: 1.0,
                sustain: 1.0,
                release_ms: 1.0,
            },
            sample_rate: 44100,
        };
        let s = AudioSynthesizer::synthesize_note(440.0, 0.1, &config);
        assert!(s.iter().all(|&v| (-1.01..=1.01).contains(&v)));
    }

    #[test]
    fn test_synthesize_sequence_nonempty() {
        let seq = make_c_major_seq();
        let config = SynthConfig {
            wave: WaveType::Sine,
            adsr: AdsrEnvelope {
                attack_ms: 5.0,
                decay_ms: 5.0,
                sustain: 0.8,
                release_ms: 5.0,
            },
            sample_rate: 22050,
        };
        let buf = AudioSynthesizer::synthesize_sequence(&seq, &config);
        assert!(!buf.is_empty());
    }

    // ── §8 Variations ─────────────────────────────────────────────────────────

    #[test]
    fn test_variation_transpose() {
        let seq = make_c_major_seq();
        let t = MusicVariationGenerator::transpose(&seq, 12);
        for (o, tr) in seq.events.iter().zip(t.events.iter()) {
            assert_eq!(tr.pitch, o.pitch + 12);
        }
    }

    #[test]
    fn test_variation_retrograde() {
        let seq = make_c_major_seq();
        let retro = MusicVariationGenerator::retrograde(&seq);
        // Retrograde should preserve note count.
        assert_eq!(retro.events.len(), seq.events.len());
        // Retrograde reverses time: the pitch that appeared last should now appear first.
        // In our 8-note ascending scale, the last onset note has pitch 72 (C5).
        // After retrograde, sorting by onset_tick, that note gets onset=0.
        assert_eq!(retro.events[0].pitch, 72, "first retro note should be C5");
        // And the originally first note (C4=60, onset=0) should now have a later onset.
        let retro_last = retro.events.iter().map(|e| e.onset_tick).max().unwrap_or(0);
        assert!(
            retro_last > 0,
            "retrograde should have non-zero max onset, got {retro_last}"
        );
    }

    #[test]
    fn test_variation_invert() {
        let seq = make_c_major_seq();
        let inv = MusicVariationGenerator::invert(&seq, 64);
        // C4(60) around E4(64): 64 - (60-64) = 68
        assert_eq!(inv.events[0].pitch, 68);
    }

    #[test]
    fn test_variation_augment() {
        let seq = make_c_major_seq();
        let aug = MusicVariationGenerator::augment(&seq, 2.0);
        assert_eq!(aug.events[0].duration_tick, 960);
    }

    #[test]
    fn test_music_variation_diminish() {
        let seq = make_c_major_seq();
        let dim = MusicVariationGenerator::diminish(&seq, 2.0);
        assert_eq!(dim.events[0].duration_tick, 240);
    }

    #[test]
    fn test_variation_harmonize() {
        let seq = make_c_major_seq();
        let h = MusicVariationGenerator::harmonize(&seq, 4);
        assert_eq!(h.events.len(), seq.events.len() * 2);
    }

    #[test]
    fn test_harmonize_chord_notes_count() {
        let mut seq = MidiSequence::new(120.0, 480);
        seq.events.push(NoteEvent {
            pitch: 60,
            velocity: 80,
            onset_tick: 0,
            duration_tick: 480,
        });
        let h = MusicVariationGenerator::harmonize(&seq, 7);
        assert_eq!(h.events.len(), 2);
        let pitches: Vec<u8> = h.events.iter().map(|e| e.pitch).collect();
        assert!(pitches.contains(&60));
        assert!(pitches.contains(&67));
    }

    // ── §9 Theory Analyzer ────────────────────────────────────────────────────

    #[test]
    fn test_note_name() {
        assert_eq!(MusicTheoryAnalyzer::note_name(0), "C");
        assert_eq!(MusicTheoryAnalyzer::note_name(2), "D");
        assert_eq!(MusicTheoryAnalyzer::note_name(11), "B");
        assert_eq!(MusicTheoryAnalyzer::note_name(60), "C");
    }

    #[test]
    fn test_voice_leading_parallel_fifths() {
        // Voices moving in parallel fifths: C→D (voice 1), G→A (voice 2)
        let mut s1 = MidiSequence::new(120.0, 480);
        s1.events.push(NoteEvent {
            pitch: 60,
            velocity: 80,
            onset_tick: 0,
            duration_tick: 480,
        });
        s1.events.push(NoteEvent {
            pitch: 62,
            velocity: 80,
            onset_tick: 480,
            duration_tick: 480,
        });
        let mut s2 = MidiSequence::new(120.0, 480);
        s2.events.push(NoteEvent {
            pitch: 67,
            velocity: 80,
            onset_tick: 0,
            duration_tick: 480,
        });
        s2.events.push(NoteEvent {
            pitch: 69,
            velocity: 80,
            onset_tick: 480,
            duration_tick: 480,
        });
        let vla = MusicTheoryAnalyzer::analyze_voice_leading(&s1, &s2);
        assert_eq!(vla.parallel_fifths, 1);
    }

    #[test]
    fn test_voice_leading_analysis() {
        let s1 = make_c_major_seq();
        let s2 = MusicVariationGenerator::transpose(&s1, 7);
        let vla = MusicTheoryAnalyzer::analyze_voice_leading(&s1, &s2);
        assert!(vla.contrary_motion_ratio >= 0.0 && vla.contrary_motion_ratio <= 1.0);
    }

    #[test]
    fn test_self_similarity_diagonal_is_one() {
        let roll = vec![
            vec![1.0f32, 0.0, 0.5],
            vec![0.0f32, 1.0, 0.3],
            vec![0.5f32, 0.3, 1.0],
        ];
        let sim = MusicTheoryAnalyzer::compute_self_similarity(&roll);
        for i in 0..3 {
            assert!((sim[i][i] - 1.0).abs() < 1e-5, "diagonal [{i}][{i}] != 1");
        }
    }

    #[test]
    fn test_motif_detection() {
        let mut seq = MidiSequence::new(120.0, 480);
        // C-E-G C-E-G
        for _ in 0..2 {
            for &p in &[60u8, 64, 67] {
                let onset = seq.events.len() as u32 * 240;
                seq.events.push(NoteEvent {
                    pitch: p,
                    velocity: 80,
                    onset_tick: onset,
                    duration_tick: 240,
                });
            }
        }
        let motifs = MusicTheoryAnalyzer::detect_motif(&seq, 3);
        assert!(!motifs.is_empty());
        let best = motifs
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        assert!(best.map(|m| m.1).unwrap_or(0.0) > 0.5);
    }

    // ── §10 Metrics ───────────────────────────────────────────────────────────

    #[test]
    fn test_music_metrics_density() {
        let seq = make_c_major_seq();
        let d = MusicMetrics::note_density(&seq);
        assert!(d > 0.5, "density={d}");
    }

    #[test]
    fn test_note_density_calculation() {
        let mut seq = MidiSequence::new(120.0, 480);
        seq.events.push(NoteEvent {
            pitch: 60,
            velocity: 80,
            onset_tick: 0,
            duration_tick: 480,
        });
        seq.events.push(NoteEvent {
            pitch: 62,
            velocity: 80,
            onset_tick: 480,
            duration_tick: 480,
        });
        let d = MusicMetrics::note_density(&seq);
        assert!((d - 1.0).abs() < 0.1, "expected ~1.0, got {d}");
    }

    #[test]
    fn test_music_metrics_pitch_range() {
        let seq = make_c_major_seq();
        assert_eq!(MusicMetrics::pitch_range(&seq), 12); // C4–C5
    }

    #[test]
    fn test_music_metrics_polyphony() {
        let mono_roll = vec![
            {
                let mut v = vec![0.0f32; 12];
                v[0] = 0.8;
                v
            },
            {
                let mut v = vec![0.0f32; 12];
                v[2] = 0.7;
                v
            },
        ];
        assert_eq!(MusicMetrics::polyphony_ratio(&mono_roll, 0.5), 0.0);
        let poly_roll = vec![{
            let mut v = vec![0.0f32; 12];
            v[0] = 0.8;
            v[4] = 0.6;
            v
        }];
        assert_eq!(MusicMetrics::polyphony_ratio(&poly_roll, 0.5), 1.0);
    }

    #[test]
    fn test_pitch_class_histogram_sums_to_one() {
        let seq = make_c_major_seq();
        let hist = MusicMetrics::pitch_class_histogram(&seq);
        let sum: f32 = hist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
    }

    #[test]
    fn test_pitch_class_histogram_twelve_bins() {
        let seq = make_c_major_seq();
        assert_eq!(MusicMetrics::pitch_class_histogram(&seq).len(), 12);
    }

    #[test]
    fn test_rhythmic_regularity_range() {
        let seq = make_c_major_seq();
        let r = MusicMetrics::rhythmic_regularity(&seq);
        assert!((0.0..=1.0).contains(&r), "regularity={r}");
    }

    #[test]
    fn test_evaluate_report() {
        let seq = make_c_major_seq();
        let roll = seq.to_piano_roll(32, 128);
        let report = MusicMetrics::evaluate(&seq, &roll);
        assert!(report.density >= 0.0);
        assert!(report.pitch_range <= 127);
    }

    #[test]
    fn test_music_eval_report_fields() {
        let seq = make_c_major_seq();
        let roll = seq.to_piano_roll(16, 128);
        let report = MusicMetrics::evaluate(&seq, &roll);
        assert!(report.regularity >= 0.0 && report.regularity <= 1.0);
        assert!(report.polyphony >= 0.0 && report.polyphony <= 1.0);
    }
}
