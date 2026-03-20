//! Text Generation Pipeline Infrastructure.
//!
//! Provides end-to-end text generation pipelines with multiple sampling strategies,
//! KV-cache management, streaming generation, and throughput monitoring.
//!
//! ## Components
//!
//! - [`TextPipeline`] — end-to-end tokenize → encode → decode → detokenize
//! - [`GreedyDecoder`] — argmax selection at each step
//! - [`TemperatureScaledDecoder`] — temperature-divided logits with sampling
//! - [`MinPSampler`] — min-p sampling (Nguyen et al.)
//! - [`RepetitionPenaltyProcessor`] — penalise already-generated tokens
//! - [`TypicalSamplerTgp`] — locally typical sampling
//! - [`EtaSampler`] — eta-based adaptive threshold sampling
//! - [`CFGDecoder`] — classifier-free guidance merging
//! - [`StreamingGenerator`] — iterator-style token-by-token generation
//! - [`GenerationCache`] — full KV-cache management for incremental decoding
//! - [`TgpPipelineMetrics`] — throughput and latency tracking

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically-stable softmax (f32).
fn softmax_f32(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

/// Compute Shannon entropy of a probability distribution.
fn entropy(probs: &[f32]) -> f32 {
    probs
        .iter()
        .filter(|&&p| p > 1e-12)
        .map(|&p| -p * p.ln())
        .sum()
}

/// Sample a token index from a probability distribution using the given RNG.
fn sample_from_probs(probs: &[f32], rng: &mut StdRng) -> usize {
    let threshold: f32 = rng.random::<f32>();
    let mut cumsum = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= threshold {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

/// Compute argmax of a slice, returning index of maximum value.
fn argmax_f32(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Pipeline Configuration and Results
// ─────────────────────────────────────────────────────────────────────────────

/// Sampling strategy for text generation.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum SamplingStrategy {
    /// Always pick the token with the highest probability.
    #[default]
    Greedy,
    /// Temperature-based sampling.
    Temperature(f32),
    /// Nucleus (top-p) + top-k combined sampling.
    NucleusTopK { top_p: f32, top_k: usize },
    /// Min-p sampling (Nguyen et al.).
    MinP(f32),
    /// Locally typical sampling.
    Typical { mass: f32 },
    /// Eta sampling.
    Eta(f32),
}

/// Configuration for the text generation pipeline.
#[derive(Clone, Debug)]
pub struct TgpConfig {
    /// Sampling strategy to use.
    pub sampling: SamplingStrategy,
    /// Temperature applied before sampling (for Temperature strategy).
    pub temperature: f32,
    /// Top-k filter: keep only this many highest-probability tokens.
    pub top_k: usize,
    /// Nucleus (top-p) filter: keep tokens until cumulative probability reaches this value.
    pub top_p: f32,
    /// Repetition penalty factor applied to already-generated tokens.
    pub repetition_penalty: f32,
    /// Maximum total sequence length (prompt + generated tokens).
    pub max_length: usize,
    /// Stop-token IDs that terminate generation.
    pub stop_tokens: Vec<usize>,
    /// Optional random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for TgpConfig {
    fn default() -> Self {
        Self {
            sampling: SamplingStrategy::Greedy,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            max_length: 512,
            stop_tokens: Vec::new(),
            seed: None,
        }
    }
}

impl TgpConfig {
    /// Create a new configuration with the given maximum sequence length.
    pub fn new(max_length: usize) -> Self {
        Self {
            max_length,
            ..Default::default()
        }
    }

    /// Set the sampling strategy.
    pub fn with_sampling(mut self, strategy: SamplingStrategy) -> Self {
        self.sampling = strategy;
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Set repetition penalty.
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    /// Set stop tokens.
    pub fn with_stop_tokens(mut self, tokens: Vec<usize>) -> Self {
        self.stop_tokens = tokens;
        self
    }

    /// Set random seed for reproducible generation.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Result produced by the text generation pipeline.
#[derive(Clone, Debug)]
pub struct PipelineResult {
    /// Generated token IDs (not including the prompt).
    pub token_ids: Vec<usize>,
    /// Per-token log-probabilities (if available).
    pub log_probs: Vec<f32>,
    /// Number of prompt tokens consumed.
    pub prompt_length: usize,
    /// Total tokens generated.
    pub tokens_generated: usize,
    /// Whether generation was stopped by a stop-token.
    pub stopped_by_stop_token: bool,
    /// Whether generation was stopped by reaching max_length.
    pub stopped_by_max_length: bool,
}

impl PipelineResult {
    fn new(prompt_length: usize) -> Self {
        Self {
            token_ids: Vec::new(),
            log_probs: Vec::new(),
            prompt_length,
            tokens_generated: 0,
            stopped_by_stop_token: false,
            stopped_by_max_length: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. GreedyDecoder
// ─────────────────────────────────────────────────────────────────────────────

/// Always selects the token with the highest logit (argmax).
///
/// This is the deterministic baseline decoding strategy.
#[derive(Clone, Debug, Default)]
pub struct GreedyDecoder;

impl GreedyDecoder {
    /// Create a new greedy decoder.
    pub fn new() -> Self {
        Self
    }

    /// Select the token index with the highest logit value.
    ///
    /// # Arguments
    /// * `logits` — raw model output logits over the vocabulary
    ///
    /// # Returns
    /// Index of the token with the maximum logit.
    pub fn decode_step(&self, logits: &[f32]) -> usize {
        argmax_f32(logits)
    }

    /// Decode an entire sequence greedily given a logit matrix (tokens × vocab).
    pub fn decode_sequence(&self, logit_matrix: &[Vec<f32>]) -> Vec<usize> {
        logit_matrix.iter().map(|l| self.decode_step(l)).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. TemperatureScaledDecoder
// ─────────────────────────────────────────────────────────────────────────────

/// Divides logits by a temperature value before computing softmax and sampling.
///
/// Temperature < 1.0 makes the distribution sharper (more greedy-like).
/// Temperature > 1.0 makes the distribution flatter (more random).
/// Temperature = 1.0 is equivalent to regular softmax sampling.
#[derive(Clone, Debug)]
pub struct TemperatureScaledDecoder {
    /// Default temperature used when none is supplied to `decode_step`.
    pub default_temperature: f32,
}

impl TemperatureScaledDecoder {
    /// Create a new temperature-scaled decoder.
    pub fn new(temperature: f32) -> Self {
        Self {
            default_temperature: temperature.max(1e-8),
        }
    }

    /// Sample a token after dividing logits by `temp`.
    ///
    /// # Arguments
    /// * `logits` — raw model logits
    /// * `temp` — temperature value (clamped to ≥ 1e-8)
    /// * `rng` — random number generator
    pub fn decode_step(&self, logits: &[f32], temp: f32, rng: &mut StdRng) -> usize {
        let effective_temp = temp.max(1e-8_f32);
        let scaled: Vec<f32> = logits.iter().map(|&l| l / effective_temp).collect();
        let probs = softmax_f32(&scaled);
        sample_from_probs(&probs, rng)
    }

    /// Use the stored default temperature.
    pub fn decode_step_default(&self, logits: &[f32], rng: &mut StdRng) -> usize {
        self.decode_step(logits, self.default_temperature, rng)
    }

    /// Return the top-n tokens with their probabilities after applying temperature.
    pub fn top_n_probs(&self, logits: &[f32], n: usize) -> Vec<(usize, f32)> {
        let effective_temp = self.default_temperature.max(1e-8_f32);
        let scaled: Vec<f32> = logits.iter().map(|&l| l / effective_temp).collect();
        let probs = softmax_f32(&scaled);
        let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(n);
        indexed
    }
}

impl Default for TemperatureScaledDecoder {
    fn default() -> Self {
        Self::new(1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. MinPSampler
// ─────────────────────────────────────────────────────────────────────────────

/// Min-p sampling as introduced by Nguyen et al.
///
/// Filters out tokens whose probability is below `min_p * max_p`,
/// where `max_p` is the maximum token probability in the distribution.
/// Then re-normalises and samples from the remaining tokens.
#[derive(Clone, Debug)]
pub struct MinPSampler {
    /// Threshold multiplier relative to the maximum probability.
    pub min_p: f32,
}

impl MinPSampler {
    /// Create a new min-p sampler.
    ///
    /// # Arguments
    /// * `min_p` — tokens below `min_p * max_p` are filtered (typically 0.05–0.2)
    pub fn new(min_p: f32) -> Self {
        Self {
            min_p: min_p.clamp(0.0, 1.0),
        }
    }

    /// Sample a token using min-p filtering.
    ///
    /// # Arguments
    /// * `logits` — raw logits
    /// * `min_p` — per-call override for the min-p threshold
    /// * `rng` — random number generator
    pub fn sample(&self, logits: &[f32], min_p: f32, rng: &mut StdRng) -> usize {
        let probs = softmax_f32(logits);
        if probs.is_empty() {
            return 0;
        }
        let max_p = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let threshold = min_p.clamp(0.0, 1.0) * max_p;

        // Collect indices that pass the filter.
        let filtered: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p >= threshold)
            .map(|(i, &p)| (i, p))
            .collect();

        if filtered.is_empty() {
            // Fall back to argmax if nothing passes the filter.
            return argmax_f32(logits);
        }

        // Re-normalise.
        let total: f32 = filtered.iter().map(|(_, p)| p).sum();
        let renormed: Vec<f32> = filtered.iter().map(|(_, p)| p / total).collect();

        // Sample.
        let drawn: f32 = rng.random::<f32>();
        let mut cumsum = 0.0_f32;
        for (idx, (&(orig_idx, _), &p)) in filtered.iter().zip(renormed.iter()).enumerate() {
            cumsum += p;
            if cumsum >= drawn || idx == filtered.len() - 1 {
                return orig_idx;
            }
        }
        filtered[filtered.len() - 1].0
    }

    /// Use the stored `min_p` field.
    pub fn sample_default(&self, logits: &[f32], rng: &mut StdRng) -> usize {
        self.sample(logits, self.min_p, rng)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. RepetitionPenaltyProcessor
// ─────────────────────────────────────────────────────────────────────────────

/// Applies a repetition penalty to logits based on previously generated tokens.
///
/// For each token that has already been generated:
/// - If the logit is positive, divide it by `penalty` (reduces score).
/// - If the logit is negative, multiply it by `penalty` (further reduces score).
///
/// A penalty of 1.0 is a no-op. Values < 1.0 *increase* repetition (unusual).
/// Values > 1.0 *reduce* repetition (typical use case, e.g. 1.3).
#[derive(Clone, Debug, Default)]
pub struct RepetitionPenaltyProcessor;

impl RepetitionPenaltyProcessor {
    /// Create a new processor.
    pub fn new() -> Self {
        Self
    }

    /// Apply repetition penalty in-place.
    ///
    /// # Arguments
    /// * `logits` — mutable logit vector (len = vocab_size)
    /// * `generated` — slice of already-generated token IDs
    /// * `penalty` — penalty factor; values > 1.0 discourage repetition
    pub fn apply(logits: &mut [f32], generated: &[usize], penalty: f32) {
        if penalty == 1.0 {
            return;
        }
        for &token_id in generated {
            if token_id < logits.len() {
                let logit = logits[token_id];
                if logit >= 0.0 {
                    logits[token_id] = logit / penalty;
                } else {
                    logits[token_id] = logit * penalty;
                }
            }
        }
    }

    /// Apply with a frequency-weighted penalty (tokens penalised proportional to count).
    pub fn apply_frequency_weighted(logits: &mut [f32], generated: &[usize], base_penalty: f32) {
        let mut freq: HashMap<usize, usize> = HashMap::new();
        for &t in generated {
            *freq.entry(t).or_insert(0) += 1;
        }
        for (&token_id, &count) in &freq {
            if token_id < logits.len() {
                let effective_penalty = 1.0 + (base_penalty - 1.0) * count as f32;
                let logit = logits[token_id];
                if logit >= 0.0 {
                    logits[token_id] = logit / effective_penalty;
                } else {
                    logits[token_id] = logit * effective_penalty;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. TypicalSamplerTgp (locally typical sampling)
// ─────────────────────────────────────────────────────────────────────────────

/// Locally typical sampling as described by Meister et al. (2023).
///
/// Filters tokens whose surprisal `|log p(x) - H|` exceeds a delta,
/// where H is the entropy of the distribution. Then samples from the
/// filtered set using a mass parameter.
///
/// Named `TypicalSamplerTgp` to avoid collision with `generation::TypicalSampler`.
#[derive(Clone, Debug)]
pub struct TypicalSamplerTgp {
    /// Mass parameter controlling how much probability mass to retain (0 < mass ≤ 1).
    pub mass: f32,
    /// Maximum allowed deviation from entropy.
    pub delta: f32,
}

impl TypicalSamplerTgp {
    /// Create a new locally typical sampler.
    ///
    /// # Arguments
    /// * `mass` — retain tokens until this fraction of probability mass is covered (e.g. 0.95)
    /// * `delta` — entropy-deviation cutoff; larger values allow more tokens
    pub fn new(mass: f32, delta: f32) -> Self {
        Self {
            mass: mass.clamp(0.0, 1.0),
            delta: delta.max(0.0),
        }
    }

    /// Sample a token using locally typical filtering.
    ///
    /// # Arguments
    /// * `logits` — raw model logits
    /// * `mass` — per-call mass override
    /// * `rng` — random number generator
    pub fn sample(&self, logits: &[f32], mass: f32, rng: &mut StdRng) -> usize {
        let probs = softmax_f32(logits);
        if probs.is_empty() {
            return 0;
        }

        // Compute distribution entropy H.
        let h = entropy(&probs);

        // Compute |log p - H| for each token.
        let mut scored: Vec<(usize, f32, f32)> = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 1e-12)
            .map(|(i, &p)| {
                let surprisal = -p.ln();
                let deviation = (surprisal - h).abs();
                (i, p, deviation)
            })
            .collect();

        // Sort by deviation (most typical first).
        scored.sort_by(|(_, _, d1), (_, _, d2)| {
            d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Collect until `mass` probability is accumulated.
        let effective_mass = mass.clamp(0.0, 1.0);
        let mut cumulative = 0.0_f32;
        let mut kept: Vec<(usize, f32)> = Vec::new();
        for (idx, p, _) in &scored {
            kept.push((*idx, *p));
            cumulative += p;
            if cumulative >= effective_mass {
                break;
            }
        }

        if kept.is_empty() {
            return argmax_f32(logits);
        }

        // Re-normalise and sample.
        let total: f32 = kept.iter().map(|(_, p)| p).sum();
        let renormed: Vec<f32> = kept.iter().map(|(_, p)| p / total).collect();
        let draw: f32 = rng.random::<f32>();
        let mut cumsum = 0.0_f32;
        for (i, (&(orig_idx, _), &p)) in kept.iter().zip(renormed.iter()).enumerate() {
            cumsum += p;
            if cumsum >= draw || i == kept.len() - 1 {
                return orig_idx;
            }
        }
        kept[kept.len() - 1].0
    }

    /// Use the stored `mass` parameter.
    pub fn sample_default(&self, logits: &[f32], rng: &mut StdRng) -> usize {
        self.sample(logits, self.mass, rng)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. EtaSampler
// ─────────────────────────────────────────────────────────────────────────────

/// Eta sampling: an adaptive threshold sampler that scales the minimum
/// probability based on the distribution entropy.
///
/// Tokens are kept if p ≥ min(eta, sqrt(eta) * exp(-H)).
/// This adapts the filter to the predictability of the distribution.
#[derive(Clone, Debug)]
pub struct EtaSampler {
    /// Eta hyperparameter controlling the minimum probability threshold.
    pub eta: f32,
}

impl EtaSampler {
    /// Create a new eta sampler.
    ///
    /// # Arguments
    /// * `eta` — base threshold value (typically 0.0003–0.2)
    pub fn new(eta: f32) -> Self {
        Self {
            eta: eta.clamp(0.0, 1.0),
        }
    }

    /// Sample a token using eta-based adaptive filtering.
    ///
    /// # Arguments
    /// * `logits` — raw logits
    /// * `eta` — per-call eta override
    /// * `rng` — random number generator
    pub fn sample(&self, logits: &[f32], eta: f32, rng: &mut StdRng) -> usize {
        let probs = softmax_f32(logits);
        if probs.is_empty() {
            return 0;
        }

        let h = entropy(&probs);
        let effective_eta = eta.clamp(0.0, 1.0);

        // Adaptive threshold: min(eta, sqrt(eta) * exp(-H))
        let adaptive_thresh = effective_eta.min(effective_eta.sqrt() * (-h).exp());

        // Keep tokens above the adaptive threshold.
        let kept: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p >= adaptive_thresh)
            .map(|(i, &p)| (i, p))
            .collect();

        if kept.is_empty() {
            return argmax_f32(logits);
        }

        // Re-normalise and sample.
        let total: f32 = kept.iter().map(|(_, p)| p).sum();
        let draw: f32 = rng.random::<f32>();
        let mut cumsum = 0.0_f32;
        for (i, &(orig_idx, p)) in kept.iter().enumerate() {
            cumsum += p / total;
            if cumsum >= draw || i == kept.len() - 1 {
                return orig_idx;
            }
        }
        kept[kept.len() - 1].0
    }

    /// Use the stored `eta` field.
    pub fn sample_default(&self, logits: &[f32], rng: &mut StdRng) -> usize {
        self.sample(logits, self.eta, rng)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. CFGDecoder (Classifier-Free Guidance)
// ─────────────────────────────────────────────────────────────────────────────

/// Classifier-free guidance (CFG) logit merging.
///
/// Combines conditional and unconditional model outputs to steer generation
/// towards the condition:
///
/// `cfg_logits = uncond + scale * (cond - uncond)`
///
/// A `guidance_scale` of 1.0 yields exactly the conditional logits.
/// Values > 1.0 amplify the conditional signal at the cost of diversity.
#[derive(Clone, Debug)]
pub struct CFGDecoder {
    /// Guidance scale applied to the conditional–unconditional difference.
    pub guidance_scale: f32,
}

impl CFGDecoder {
    /// Create a new CFG decoder.
    ///
    /// # Arguments
    /// * `guidance_scale` — typical values are 1.5–7.5 (1.0 = no guidance)
    pub fn new(guidance_scale: f32) -> Self {
        Self { guidance_scale }
    }

    /// Merge conditional and unconditional logits using classifier-free guidance.
    ///
    /// # Arguments
    /// * `cond_logits` — logits from the conditional forward pass
    /// * `uncond_logits` — logits from the unconditional forward pass
    /// * `guidance_scale` — per-call scale override
    ///
    /// # Returns
    /// Merged logit vector of the same length as inputs.
    ///
    /// # Errors
    /// Returns an error if the two logit slices have different lengths.
    pub fn cfg_logits(
        &self,
        cond_logits: &[f32],
        uncond_logits: &[f32],
        guidance_scale: f32,
    ) -> Result<Vec<f32>> {
        if cond_logits.len() != uncond_logits.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "cfg_logits".to_string(),
                expected: format!("{}", cond_logits.len()),
                got: format!("{}", uncond_logits.len()),
                context: None,
            });
        }
        let merged: Vec<f32> = uncond_logits
            .iter()
            .zip(cond_logits.iter())
            .map(|(&u, &c)| u + guidance_scale * (c - u))
            .collect();
        Ok(merged)
    }

    /// Convenience wrapper using the stored `guidance_scale`.
    pub fn cfg_logits_default(
        &self,
        cond_logits: &[f32],
        uncond_logits: &[f32],
    ) -> Result<Vec<f32>> {
        self.cfg_logits(cond_logits, uncond_logits, self.guidance_scale)
    }

    /// Apply CFG and then greedily decode.
    pub fn decode_greedy(&self, cond_logits: &[f32], uncond_logits: &[f32]) -> Result<usize> {
        let merged = self.cfg_logits_default(cond_logits, uncond_logits)?;
        Ok(argmax_f32(&merged))
    }

    /// Apply CFG and sample with temperature.
    pub fn decode_sampled(
        &self,
        cond_logits: &[f32],
        uncond_logits: &[f32],
        temperature: f32,
        rng: &mut StdRng,
    ) -> Result<usize> {
        let merged = self.cfg_logits_default(cond_logits, uncond_logits)?;
        let decoder = TemperatureScaledDecoder::new(temperature);
        Ok(decoder.decode_step(&merged, temperature, rng))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. StreamingGenerator
// ─────────────────────────────────────────────────────────────────────────────

/// Internal state for streaming generation.
struct StreamingState {
    generated: Vec<usize>,
    current_logits: Vec<f32>,
    rng: StdRng,
    done: bool,
}

/// Iterator-style token-by-token text generator.
///
/// Each call to `next()` produces one token ID. The generator maintains
/// its own state including generated tokens and an internal RNG.
///
/// The logit provider is a closure called each step: it receives the
/// current sequence of token IDs and returns a logit vector over the vocabulary.
pub struct StreamingGenerator {
    config: TgpConfig,
    state: StreamingState,
    logit_fn: Box<dyn FnMut(&[usize]) -> Vec<f32>>,
}

impl StreamingGenerator {
    /// Create a new streaming generator.
    ///
    /// # Arguments
    /// * `config` — generation configuration
    /// * `prompt_tokens` — initial token IDs (prompt)
    /// * `logit_fn` — closure that produces logits given current token sequence
    pub fn new(
        config: TgpConfig,
        prompt_tokens: Vec<usize>,
        logit_fn: Box<dyn FnMut(&[usize]) -> Vec<f32>>,
    ) -> Self {
        let seed = config.seed.unwrap_or(42);
        let rng = StdRng::seed_from_u64(seed);
        Self {
            config,
            state: StreamingState {
                generated: prompt_tokens,
                current_logits: Vec::new(),
                rng,
                done: false,
            },
            logit_fn,
        }
    }

    /// Return whether generation is complete.
    pub fn is_done(&self) -> bool {
        self.state.done
    }

    /// Return all tokens generated so far (including prompt).
    pub fn generated_tokens(&self) -> &[usize] {
        &self.state.generated
    }

    /// Return the top-n tokens with their probabilities from the last step.
    pub fn peek_top_tokens(&self, n: usize) -> Vec<(usize, f32)> {
        if self.state.current_logits.is_empty() {
            return Vec::new();
        }
        let probs = softmax_f32(&self.state.current_logits);
        let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(n);
        indexed
    }

    /// Apply the configured sampling strategy to current logits.
    fn sample_next_token(&mut self, logits: Vec<f32>) -> usize {
        match &self.config.sampling {
            SamplingStrategy::Greedy => argmax_f32(&logits),
            SamplingStrategy::Temperature(temp) => {
                let t = *temp;
                let decoder = TemperatureScaledDecoder::new(t);
                decoder.decode_step(&logits, t, &mut self.state.rng)
            }
            SamplingStrategy::NucleusTopK { top_p, top_k } => {
                let top_p = *top_p;
                let top_k = *top_k;
                self.nucleus_topk_sample(&logits, top_p, top_k)
            }
            SamplingStrategy::MinP(min_p) => {
                let mp = *min_p;
                let sampler = MinPSampler::new(mp);
                sampler.sample(&logits, mp, &mut self.state.rng)
            }
            SamplingStrategy::Typical { mass } => {
                let m = *mass;
                let sampler = TypicalSamplerTgp::new(m, 1.0);
                sampler.sample(&logits, m, &mut self.state.rng)
            }
            SamplingStrategy::Eta(eta) => {
                let e = *eta;
                let sampler = EtaSampler::new(e);
                sampler.sample(&logits, e, &mut self.state.rng)
            }
        }
    }

    /// Nucleus + top-k sampling.
    fn nucleus_topk_sample(&mut self, logits: &[f32], top_p: f32, top_k: usize) -> usize {
        let probs = softmax_f32(logits);
        let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Apply top-k first.
        if top_k > 0 {
            indexed.truncate(top_k);
        }

        // Apply nucleus (top-p).
        let mut cumsum = 0.0_f32;
        let mut kept = Vec::new();
        for pair in &indexed {
            kept.push(*pair);
            cumsum += pair.1;
            if cumsum >= top_p {
                break;
            }
        }
        if kept.is_empty() {
            kept.push(indexed[0]);
        }

        let total: f32 = kept.iter().map(|(_, p)| p).sum();
        let draw: f32 = self.state.rng.random::<f32>();
        let mut cs = 0.0_f32;
        for (i, &(idx, p)) in kept.iter().enumerate() {
            cs += p / total;
            if cs >= draw || i == kept.len() - 1 {
                return idx;
            }
        }
        kept[kept.len() - 1].0
    }
}

impl Iterator for StreamingGenerator {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.state.done {
            return None;
        }

        // Check length limit (generated = prompt + new tokens).
        if self.state.generated.len() >= self.config.max_length {
            self.state.done = true;
            return None;
        }

        // Obtain logits from the provider.
        let mut logits = (self.logit_fn)(&self.state.generated);

        // Apply repetition penalty.
        if self.config.repetition_penalty != 1.0 {
            RepetitionPenaltyProcessor::apply(
                &mut logits,
                &self.state.generated,
                self.config.repetition_penalty,
            );
        }

        // Store logits for peek_top_tokens.
        self.state.current_logits = logits.clone();

        // Sample next token.
        let token = self.sample_next_token(logits);

        // Check stop tokens.
        if self.config.stop_tokens.contains(&token) {
            self.state.done = true;
            return None;
        }

        self.state.generated.push(token);
        Some(token)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. GenerationCache (KV-cache)
// ─────────────────────────────────────────────────────────────────────────────

/// Per-layer KV-cache entry.
#[derive(Clone, Debug)]
struct KvCacheEntry {
    keys: Vec<Vec<f32>>,   // shape: [seq_len, key_dim]
    values: Vec<Vec<f32>>, // shape: [seq_len, val_dim]
}

/// Full KV-cache manager for incremental decoding.
///
/// Maintains separate key and value caches for each transformer layer,
/// supporting append, truncation, and retrieval operations.
#[derive(Clone, Debug, Default)]
pub struct GenerationCache {
    /// Per-layer caches indexed by layer number.
    layers: HashMap<usize, KvCacheEntry>,
    /// Sequence length tracked across all layers.
    seq_len: usize,
}

impl GenerationCache {
    /// Create a new empty generation cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a new key-value pair for the given layer.
    ///
    /// # Arguments
    /// * `layer` — transformer layer index (0-based)
    /// * `k` — key vector for the current position
    /// * `v` — value vector for the current position
    pub fn push_kv(&mut self, layer: usize, k: Vec<f32>, v: Vec<f32>) {
        let entry = self.layers.entry(layer).or_insert_with(|| KvCacheEntry {
            keys: Vec::new(),
            values: Vec::new(),
        });
        entry.keys.push(k);
        entry.values.push(v);

        // Update global seq_len from any single layer.
        if let Some(e) = self.layers.get(&layer) {
            self.seq_len = self.seq_len.max(e.keys.len());
        }
    }

    /// Retrieve the full key and value history for a layer.
    ///
    /// Returns `None` if the layer has no cache entries.
    pub fn get_kv(&self, layer: usize) -> Option<(&[Vec<f32>], &[Vec<f32>])> {
        self.layers
            .get(&layer)
            .map(|e| (e.keys.as_slice(), e.values.as_slice()))
    }

    /// Retrieve the key and value for a specific position within a layer.
    pub fn get_kv_at(&self, layer: usize, pos: usize) -> Option<(&[f32], &[f32])> {
        self.layers.get(&layer).and_then(|e| {
            if pos < e.keys.len() {
                Some((e.keys[pos].as_slice(), e.values[pos].as_slice()))
            } else {
                None
            }
        })
    }

    /// Truncate all layer caches to `max_len` most-recent entries.
    ///
    /// This is used for sliding-window attention or context-length management.
    pub fn truncate(&mut self, max_len: usize) {
        for entry in self.layers.values_mut() {
            if entry.keys.len() > max_len {
                let drain_len = entry.keys.len() - max_len;
                entry.keys.drain(0..drain_len);
                entry.values.drain(0..drain_len);
            }
        }
        // Recalculate seq_len.
        self.seq_len = self
            .layers
            .values()
            .map(|e| e.keys.len())
            .max()
            .unwrap_or(0);
    }

    /// Return the current sequence length (number of cached positions).
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Return the number of layers that have cache entries.
    pub fn num_cached_layers(&self) -> usize {
        self.layers.len()
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.layers.clear();
        self.seq_len = 0;
    }

    /// Estimate the memory usage of the cache in bytes (f32 = 4 bytes each).
    pub fn memory_bytes(&self) -> usize {
        let mut total_floats = 0usize;
        for entry in self.layers.values() {
            for k in &entry.keys {
                total_floats += k.len();
            }
            for v in &entry.values {
                total_floats += v.len();
            }
        }
        total_floats * 4
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. TgpPipelineMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Throughput and latency tracking for text generation pipelines.
///
/// Records per-token elapsed times in milliseconds and derives:
/// - Tokens per second
/// - P50 (median) latency
/// - P95 latency
///
/// Named `TgpPipelineMetrics` to avoid collision with `pipeline::PipelineMetrics`.
#[derive(Clone, Debug, Default)]
pub struct TgpPipelineMetrics {
    /// Sorted list of per-token latencies in milliseconds.
    latencies_ms: Vec<u64>,
    /// Total tokens recorded.
    total_tokens: usize,
    /// Total elapsed time in milliseconds.
    total_elapsed_ms: u64,
}

impl TgpPipelineMetrics {
    /// Create a new metrics tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the generation of one token with its elapsed time.
    ///
    /// # Arguments
    /// * `elapsed_ms` — wall-clock time in milliseconds for this token
    pub fn record_token(&mut self, elapsed_ms: u64) {
        self.latencies_ms.push(elapsed_ms);
        self.total_tokens += 1;
        self.total_elapsed_ms += elapsed_ms;
    }

    /// Compute throughput as tokens per second.
    ///
    /// Returns 0.0 if no time has elapsed.
    pub fn tokens_per_second(&self) -> f32 {
        if self.total_elapsed_ms == 0 {
            return 0.0;
        }
        (self.total_tokens as f64 / (self.total_elapsed_ms as f64 / 1000.0)) as f32
    }

    /// Compute the median (P50) per-token latency in milliseconds.
    ///
    /// Returns 0.0 if no tokens have been recorded.
    pub fn latency_p50_ms(&self) -> f64 {
        self.percentile(50)
    }

    /// Compute the 95th-percentile per-token latency in milliseconds.
    ///
    /// Returns 0.0 if no tokens have been recorded.
    pub fn latency_p95_ms(&self) -> f64 {
        self.percentile(95)
    }

    /// Compute the given percentile (0–100) of recorded latencies.
    ///
    /// Uses linear interpolation between adjacent sorted values.
    pub fn percentile(&self, p: u8) -> f64 {
        if self.latencies_ms.is_empty() {
            return 0.0;
        }
        let mut sorted = self.latencies_ms.clone();
        sorted.sort_unstable();
        let n = sorted.len();
        if n == 1 {
            return sorted[0] as f64;
        }
        let p_clamped = p.min(100) as f64;
        let idx_f = p_clamped / 100.0 * (n - 1) as f64;
        let lo = idx_f.floor() as usize;
        let hi = idx_f.ceil() as usize;
        let frac = idx_f - lo as f64;
        sorted[lo] as f64 * (1.0 - frac) + sorted[hi] as f64 * frac
    }

    /// Return the total number of tokens recorded.
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Return the total elapsed time in milliseconds.
    pub fn total_elapsed_ms(&self) -> u64 {
        self.total_elapsed_ms
    }

    /// Reset all recorded data.
    pub fn reset(&mut self) {
        self.latencies_ms.clear();
        self.total_tokens = 0;
        self.total_elapsed_ms = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. TextPipeline
// ─────────────────────────────────────────────────────────────────────────────

/// A simple vocabulary for testing without a real tokenizer.
pub type SimpleVocab = HashMap<String, usize>;

/// Minimal tokenizer trait for the TextPipeline.
pub trait Tokenizer: Send + Sync {
    /// Tokenize a text string into a sequence of token IDs.
    fn encode(&self, text: &str) -> Vec<usize>;
    /// Convert a sequence of token IDs back to a string.
    fn decode(&self, tokens: &[usize]) -> String;
    /// Vocabulary size.
    fn vocab_size(&self) -> usize;
}

/// A trivial character-level tokenizer for tests.
pub struct CharTokenizer {
    vocab_size: usize,
}

impl CharTokenizer {
    /// Create a character tokenizer with the given vocabulary size (character range).
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| (c as usize) % self.vocab_size)
            .collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .filter_map(|&t| char::from_u32(t as u32))
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// End-to-end text generation pipeline.
///
/// The pipeline orchestrates:
/// 1. Tokenization of the input prompt
/// 2. Encoding (via a user-provided logit function)
/// 3. Decoding (token selection with the configured strategy)
/// 4. Detokenization back to text
pub struct TextPipeline {
    /// Pipeline configuration.
    pub config: TgpConfig,
    /// Tokenizer for prompt encoding and output decoding.
    tokenizer: Box<dyn Tokenizer>,
    /// Function providing logits given current token sequence.
    logit_fn: Box<dyn FnMut(&[usize]) -> Vec<f32>>,
    /// Optional metrics tracker.
    metrics: Option<TgpPipelineMetrics>,
}

impl TextPipeline {
    /// Create a new text pipeline.
    ///
    /// # Arguments
    /// * `config` — generation configuration
    /// * `tokenizer` — tokenizer implementation
    /// * `logit_fn` — closure returning logits given current token IDs
    pub fn new(
        config: TgpConfig,
        tokenizer: Box<dyn Tokenizer>,
        logit_fn: Box<dyn FnMut(&[usize]) -> Vec<f32>>,
    ) -> Self {
        Self {
            config,
            tokenizer,
            logit_fn,
            metrics: None,
        }
    }

    /// Enable metrics collection.
    pub fn with_metrics(mut self) -> Self {
        self.metrics = Some(TgpPipelineMetrics::new());
        self
    }

    /// Return a reference to the metrics tracker, if enabled.
    pub fn metrics(&self) -> Option<&TgpPipelineMetrics> {
        self.metrics.as_ref()
    }

    /// Run the pipeline for the given prompt.
    ///
    /// # Arguments
    /// * `prompt` — input text to condition generation on
    /// * `max_tokens` — maximum number of new tokens to generate
    ///
    /// # Returns
    /// A `PipelineResult` containing generated token IDs, log-probs, and metadata.
    pub fn run(&mut self, prompt: &str, max_tokens: usize) -> PipelineResult {
        let prompt_tokens = self.tokenizer.encode(prompt);
        let prompt_length = prompt_tokens.len();
        let mut result = PipelineResult::new(prompt_length);

        let seed = self.config.seed.unwrap_or(42);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut current_tokens = prompt_tokens;

        let effective_max = self.config.max_length.min(prompt_length + max_tokens);

        for _ in 0..max_tokens {
            if current_tokens.len() >= effective_max {
                result.stopped_by_max_length = true;
                break;
            }

            let mut logits = (self.logit_fn)(&current_tokens);

            // Apply repetition penalty.
            if self.config.repetition_penalty != 1.0 {
                RepetitionPenaltyProcessor::apply(
                    &mut logits,
                    &result.token_ids,
                    self.config.repetition_penalty,
                );
            }

            // Compute log-prob of the chosen token for diagnostics.
            let probs = softmax_f32(&logits);

            let token = match &self.config.sampling {
                SamplingStrategy::Greedy => argmax_f32(&logits),
                SamplingStrategy::Temperature(temp) => {
                    let t = *temp;
                    let decoder = TemperatureScaledDecoder::new(t);
                    decoder.decode_step(&logits, t, &mut rng)
                }
                SamplingStrategy::NucleusTopK { top_p, top_k } => {
                    let tp = *top_p;
                    let tk = *top_k;
                    Self::nucleus_topk_static(&logits, tp, tk, &mut rng)
                }
                SamplingStrategy::MinP(mp) => {
                    let m = *mp;
                    let sampler = MinPSampler::new(m);
                    sampler.sample(&logits, m, &mut rng)
                }
                SamplingStrategy::Typical { mass } => {
                    let ms = *mass;
                    let sampler = TypicalSamplerTgp::new(ms, 1.0);
                    sampler.sample(&logits, ms, &mut rng)
                }
                SamplingStrategy::Eta(eta) => {
                    let e = *eta;
                    let sampler = EtaSampler::new(e);
                    sampler.sample(&logits, e, &mut rng)
                }
            };

            let log_prob = if token < probs.len() && probs[token] > 0.0 {
                probs[token].ln()
            } else {
                f32::NEG_INFINITY
            };

            // Check stop tokens.
            if self.config.stop_tokens.contains(&token) {
                result.stopped_by_stop_token = true;
                break;
            }

            result.token_ids.push(token);
            result.log_probs.push(log_prob);
            result.tokens_generated += 1;
            current_tokens.push(token);
        }

        result
    }

    /// Decode generated tokens back to text.
    pub fn decode_result(&self, result: &PipelineResult) -> String {
        self.tokenizer.decode(&result.token_ids)
    }

    /// Nucleus + top-k static helper (no self reference).
    fn nucleus_topk_static(logits: &[f32], top_p: f32, top_k: usize, rng: &mut StdRng) -> usize {
        let probs = softmax_f32(logits);
        let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        if top_k > 0 {
            indexed.truncate(top_k);
        }
        let mut cumsum = 0.0_f32;
        let mut kept = Vec::new();
        for &pair in &indexed {
            kept.push(pair);
            cumsum += pair.1;
            if cumsum >= top_p {
                break;
            }
        }
        if kept.is_empty() && !indexed.is_empty() {
            kept.push(indexed[0]);
        }
        let total: f32 = kept.iter().map(|(_, p)| p).sum();
        let draw: f32 = rng.random::<f32>();
        let mut cs = 0.0_f32;
        for (i, &(idx, p)) in kept.iter().enumerate() {
            cs += p / total;
            if cs >= draw || i == kept.len() - 1 {
                return idx;
            }
        }
        kept[kept.len() - 1].0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::SeedableRng;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(12345)
    }

    fn uniform_logits(n: usize) -> Vec<f32> {
        vec![1.0_f32; n]
    }

    fn peaked_logits(n: usize, peak: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n];
        if peak < n {
            v[peak] = 10.0;
        }
        v
    }

    // ─── softmax_f32 ───────────────────────────────────────────────────────

    #[test]
    fn test_softmax_empty() {
        let result = softmax_f32(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_softmax_uniform() {
        let v = softmax_f32(&uniform_logits(4));
        let expected = 0.25_f32;
        for p in &v {
            assert!((p - expected).abs() < 1e-5, "expected {expected}, got {p}");
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let probs = softmax_f32(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
    }

    // ─── entropy ───────────────────────────────────────────────────────────

    #[test]
    fn test_entropy_uniform() {
        let probs = vec![0.25_f32; 4];
        let h = entropy(&probs);
        let expected = (4.0_f32).ln(); // ln(4) ≈ 1.386
        assert!((h - expected).abs() < 1e-4, "h={h} expected={expected}");
    }

    #[test]
    fn test_entropy_deterministic() {
        let probs = vec![1.0_f32, 0.0, 0.0];
        let h = entropy(&probs);
        assert!(h.abs() < 1e-5, "h={h}");
    }

    // ─── GreedyDecoder ─────────────────────────────────────────────────────

    #[test]
    fn test_greedy_decoder_argmax() {
        let decoder = GreedyDecoder::new();
        let logits = vec![0.1_f32, 0.9, 0.5, 0.2];
        assert_eq!(decoder.decode_step(&logits), 1);
    }

    #[test]
    fn test_greedy_decoder_sequence() {
        let decoder = GreedyDecoder::new();
        let matrix = vec![vec![0.1_f32, 0.9], vec![0.8_f32, 0.2]];
        let tokens = decoder.decode_sequence(&matrix);
        assert_eq!(tokens, vec![1, 0]);
    }

    #[test]
    fn test_greedy_deterministic() {
        let d = GreedyDecoder::new();
        let logits = peaked_logits(10, 7);
        assert_eq!(d.decode_step(&logits), 7);
    }

    // ─── TemperatureScaledDecoder ──────────────────────────────────────────

    #[test]
    fn test_temperature_low_concentrates() {
        let decoder = TemperatureScaledDecoder::new(0.01);
        let logits = peaked_logits(10, 5);
        let mut rng = make_rng();
        // With very low temperature, should almost always pick 5.
        let token = decoder.decode_step_default(&logits, &mut rng);
        assert_eq!(token, 5);
    }

    #[test]
    fn test_temperature_high_more_random() {
        let decoder = TemperatureScaledDecoder::new(100.0);
        let logits = uniform_logits(10);
        let mut rng = make_rng();
        // Should not crash and produce valid index.
        let token = decoder.decode_step_default(&logits, &mut rng);
        assert!(token < 10);
    }

    #[test]
    fn test_temperature_top_n_probs() {
        let decoder = TemperatureScaledDecoder::new(1.0);
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let top3 = decoder.top_n_probs(&logits, 3);
        assert_eq!(top3.len(), 3);
        // Should be sorted descending.
        assert!(top3[0].1 >= top3[1].1);
        assert!(top3[1].1 >= top3[2].1);
    }

    #[test]
    fn test_temperature_default_decode() {
        let decoder = TemperatureScaledDecoder::new(1.0);
        let logits = peaked_logits(5, 2);
        let mut rng = make_rng();
        let token = decoder.decode_step(&logits, 1.0, &mut rng);
        assert!(token < 5);
    }

    // ─── MinPSampler ───────────────────────────────────────────────────────

    #[test]
    fn test_minp_sampler_basic() {
        let sampler = MinPSampler::new(0.05);
        let logits = peaked_logits(10, 3);
        let mut rng = make_rng();
        let token = sampler.sample_default(&logits, &mut rng);
        assert!(token < 10);
    }

    #[test]
    fn test_minp_sampler_high_threshold_fallback() {
        // With min_p = 1.0, only max-prob token survives.
        let sampler = MinPSampler::new(1.0);
        let logits = peaked_logits(10, 4);
        let mut rng = make_rng();
        let token = sampler.sample_default(&logits, &mut rng);
        assert_eq!(token, 4);
    }

    #[test]
    fn test_minp_sampler_empty() {
        let sampler = MinPSampler::new(0.1);
        let mut rng = make_rng();
        let token = sampler.sample(&[], 0.1, &mut rng);
        assert_eq!(token, 0);
    }

    #[test]
    fn test_minp_per_call_override() {
        let sampler = MinPSampler::new(0.5);
        let logits = peaked_logits(8, 2);
        let mut rng = make_rng();
        // Override with 0.0 — all tokens survive.
        let token = sampler.sample(&logits, 0.0, &mut rng);
        assert!(token < 8);
    }

    // ─── RepetitionPenaltyProcessor ────────────────────────────────────────

    #[test]
    fn test_repetition_penalty_noop_at_one() {
        let mut logits = vec![1.0_f32, 2.0, 3.0];
        let orig = logits.clone();
        RepetitionPenaltyProcessor::apply(&mut logits, &[0, 1], 1.0);
        assert_eq!(logits, orig);
    }

    #[test]
    fn test_repetition_penalty_reduces_positive() {
        let mut logits = vec![2.0_f32, 0.0, 0.0];
        RepetitionPenaltyProcessor::apply(&mut logits, &[0], 2.0);
        assert!((logits[0] - 1.0).abs() < 1e-5, "logit={}", logits[0]);
    }

    #[test]
    fn test_repetition_penalty_amplifies_negative() {
        let mut logits = vec![-1.0_f32, 0.0, 0.0];
        RepetitionPenaltyProcessor::apply(&mut logits, &[0], 2.0);
        assert!((logits[0] - (-2.0)).abs() < 1e-5, "logit={}", logits[0]);
    }

    #[test]
    fn test_repetition_penalty_out_of_bounds_token() {
        // Token ID outside logits range — should not panic.
        let mut logits = vec![1.0_f32, 1.0];
        RepetitionPenaltyProcessor::apply(&mut logits, &[100], 2.0);
        assert_eq!(logits, vec![1.0_f32, 1.0]);
    }

    #[test]
    fn test_repetition_penalty_frequency_weighted() {
        let mut logits = vec![4.0_f32, 1.0, 1.0];
        // Token 0 generated twice, penalty 2.0 → effective = 1 + (2-1)*2 = 3
        RepetitionPenaltyProcessor::apply_frequency_weighted(&mut logits, &[0, 0], 2.0);
        assert!(logits[0] < 4.0_f32);
    }

    // ─── TypicalSamplerTgp ─────────────────────────────────────────────────

    #[test]
    fn test_typical_sampler_valid_token() {
        let sampler = TypicalSamplerTgp::new(0.95, 1.0);
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut rng = make_rng();
        let token = sampler.sample_default(&logits, &mut rng);
        assert!(token < 4);
    }

    #[test]
    fn test_typical_sampler_empty() {
        let sampler = TypicalSamplerTgp::new(0.95, 1.0);
        let mut rng = make_rng();
        let token = sampler.sample(&[], 0.95, &mut rng);
        assert_eq!(token, 0);
    }

    #[test]
    fn test_typical_sampler_mass_one() {
        // With mass=1.0, should include all tokens.
        let sampler = TypicalSamplerTgp::new(1.0, 2.0);
        let logits = uniform_logits(6);
        let mut rng = make_rng();
        let token = sampler.sample_default(&logits, &mut rng);
        assert!(token < 6);
    }

    // ─── EtaSampler ────────────────────────────────────────────────────────

    #[test]
    fn test_eta_sampler_basic() {
        let sampler = EtaSampler::new(0.1);
        let logits = peaked_logits(10, 6);
        let mut rng = make_rng();
        let token = sampler.sample_default(&logits, &mut rng);
        assert!(token < 10);
    }

    #[test]
    fn test_eta_sampler_empty() {
        let sampler = EtaSampler::new(0.1);
        let mut rng = make_rng();
        let token = sampler.sample(&[], 0.1, &mut rng);
        assert_eq!(token, 0);
    }

    #[test]
    fn test_eta_sampler_high_eta_fallback() {
        // eta=1.0, very strict threshold — should fall back to argmax.
        let sampler = EtaSampler::new(1.0);
        let logits = peaked_logits(10, 3);
        let mut rng = make_rng();
        let token = sampler.sample_default(&logits, &mut rng);
        // 3 should survive since it has the highest probability.
        assert_eq!(token, 3);
    }

    // ─── CFGDecoder ────────────────────────────────────────────────────────

    #[test]
    fn test_cfg_logits_scale_one_equals_cond() {
        let cfg = CFGDecoder::new(1.0);
        let cond = vec![1.0_f32, 2.0, 3.0];
        let uncond = vec![0.5_f32, 1.0, 1.5];
        let result = cfg
            .cfg_logits_default(&cond, &uncond)
            .expect("cfg_logits_default failed");
        for (r, c) in result.iter().zip(cond.iter()) {
            assert!((r - c).abs() < 1e-5, "r={r} c={c}");
        }
    }

    #[test]
    fn test_cfg_logits_scale_zero_equals_uncond() {
        let cfg = CFGDecoder::new(0.0);
        let cond = vec![5.0_f32, 6.0];
        let uncond = vec![1.0_f32, 2.0];
        let result = cfg
            .cfg_logits_default(&cond, &uncond)
            .expect("cfg_logits_default failed");
        for (r, u) in result.iter().zip(uncond.iter()) {
            assert!((r - u).abs() < 1e-5, "r={r} u={u}");
        }
    }

    #[test]
    fn test_cfg_logits_length_mismatch_error() {
        let cfg = CFGDecoder::new(2.0);
        let cond = vec![1.0_f32, 2.0];
        let uncond = vec![1.0_f32];
        let result = cfg.cfg_logits_default(&cond, &uncond);
        assert!(result.is_err());
    }

    #[test]
    fn test_cfg_decode_greedy() {
        let cfg = CFGDecoder::new(2.0);
        let cond = peaked_logits(5, 2);
        let uncond = uniform_logits(5);
        let token = cfg
            .decode_greedy(&cond, &uncond)
            .expect("decode_greedy failed");
        assert!(token < 5);
    }

    #[test]
    fn test_cfg_decode_sampled() {
        let cfg = CFGDecoder::new(1.5);
        let cond = peaked_logits(5, 1);
        let uncond = uniform_logits(5);
        let mut rng = make_rng();
        let token = cfg
            .decode_sampled(&cond, &uncond, 1.0, &mut rng)
            .expect("decode_sampled failed");
        assert!(token < 5);
    }

    // ─── StreamingGenerator ────────────────────────────────────────────────

    #[test]
    fn test_streaming_generator_produces_tokens() {
        let config = TgpConfig::new(20).with_seed(99);
        let vocab_size = 10;
        let logit_fn: Box<dyn FnMut(&[usize]) -> Vec<f32>> =
            Box::new(move |_| peaked_logits(vocab_size, 3));
        let mut gen = StreamingGenerator::new(config, vec![0], logit_fn);
        let token = gen.next();
        assert_eq!(token, Some(3));
    }

    #[test]
    fn test_streaming_generator_stop_token() {
        let config = TgpConfig::new(100).with_stop_tokens(vec![3]).with_seed(0);
        let logit_fn: Box<dyn FnMut(&[usize]) -> Vec<f32>> = Box::new(|_| peaked_logits(10, 3));
        let mut gen = StreamingGenerator::new(config, vec![0], logit_fn);
        let token = gen.next();
        assert!(token.is_none(), "expected None due to stop token");
        assert!(gen.is_done());
    }

    #[test]
    fn test_streaming_generator_max_length() {
        let config = TgpConfig::new(3).with_seed(0);
        let logit_fn: Box<dyn FnMut(&[usize]) -> Vec<f32>> = Box::new(|_| peaked_logits(10, 1));
        // Prompt is already length 3 = max_length.
        let mut gen = StreamingGenerator::new(config, vec![0, 1, 2], logit_fn);
        let token = gen.next();
        assert!(token.is_none());
    }

    #[test]
    fn test_streaming_generator_peek_top_tokens() {
        let config = TgpConfig::new(20).with_seed(7);
        let logit_fn: Box<dyn FnMut(&[usize]) -> Vec<f32>> =
            Box::new(|_| vec![0.0_f32, 1.0, 2.0, 3.0, 4.0]);
        let mut gen = StreamingGenerator::new(config, vec![0], logit_fn);
        let _ = gen.next();
        let top = gen.peek_top_tokens(3);
        assert_eq!(top.len(), 3);
        assert!(top[0].1 >= top[1].1);
    }

    #[test]
    fn test_streaming_generator_collects_sequence() {
        let config = TgpConfig::new(10).with_seed(1);
        let logit_fn: Box<dyn FnMut(&[usize]) -> Vec<f32>> = Box::new(|_| peaked_logits(5, 2));
        let mut gen = StreamingGenerator::new(config, vec![0], logit_fn);
        let tokens: Vec<usize> = gen.by_ref().take(5).collect();
        assert_eq!(tokens.len(), 5);
        for &t in &tokens {
            assert_eq!(t, 2);
        }
    }

    // ─── GenerationCache ───────────────────────────────────────────────────

    #[test]
    fn test_cache_push_and_retrieve() {
        let mut cache = GenerationCache::new();
        cache.push_kv(0, vec![1.0_f32, 2.0], vec![3.0_f32, 4.0]);
        let (k, v) = cache.get_kv(0).expect("get_kv failed");
        assert_eq!(k.len(), 1);
        assert_eq!(v.len(), 1);
        assert_eq!(k[0], &[1.0_f32, 2.0]);
    }

    #[test]
    fn test_cache_seq_len() {
        let mut cache = GenerationCache::new();
        assert_eq!(cache.seq_len(), 0);
        cache.push_kv(0, vec![1.0], vec![1.0]);
        cache.push_kv(0, vec![2.0], vec![2.0]);
        assert_eq!(cache.seq_len(), 2);
    }

    #[test]
    fn test_cache_truncate() {
        let mut cache = GenerationCache::new();
        for i in 0..5 {
            cache.push_kv(0, vec![i as f32], vec![i as f32]);
        }
        cache.truncate(3);
        assert_eq!(cache.seq_len(), 3);
    }

    #[test]
    fn test_cache_miss() {
        let cache = GenerationCache::new();
        assert!(cache.get_kv(99).is_none());
    }

    #[test]
    fn test_cache_multi_layer() {
        let mut cache = GenerationCache::new();
        cache.push_kv(0, vec![1.0], vec![1.0]);
        cache.push_kv(1, vec![2.0], vec![2.0]);
        assert_eq!(cache.num_cached_layers(), 2);
    }

    #[test]
    fn test_cache_memory_bytes() {
        let mut cache = GenerationCache::new();
        // k=[1.0, 2.0] (2 f32), v=[3.0, 4.0] (2 f32) → 4 * 4 = 16 bytes
        cache.push_kv(0, vec![1.0_f32, 2.0], vec![3.0_f32, 4.0]);
        assert_eq!(cache.memory_bytes(), 16);
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = GenerationCache::new();
        cache.push_kv(0, vec![1.0], vec![1.0]);
        cache.clear();
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.num_cached_layers(), 0);
    }

    #[test]
    fn test_cache_get_kv_at() {
        let mut cache = GenerationCache::new();
        cache.push_kv(0, vec![10.0_f32], vec![20.0_f32]);
        cache.push_kv(0, vec![30.0_f32], vec![40.0_f32]);
        let (k, v) = cache.get_kv_at(0, 1).expect("get_kv_at failed");
        assert_eq!(k, &[30.0_f32]);
        assert_eq!(v, &[40.0_f32]);
    }

    // ─── TgpPipelineMetrics ─────────────────────────────────────────────────

    #[test]
    fn test_metrics_no_data() {
        let m = TgpPipelineMetrics::new();
        assert_eq!(m.tokens_per_second(), 0.0);
        assert_eq!(m.latency_p50_ms(), 0.0);
        assert_eq!(m.latency_p95_ms(), 0.0);
    }

    #[test]
    fn test_metrics_throughput() {
        let mut m = TgpPipelineMetrics::new();
        // 10 tokens in 1000 ms = 10 TPS
        for _ in 0..10 {
            m.record_token(100);
        }
        assert!(
            (m.tokens_per_second() - 10.0).abs() < 0.1,
            "tps={}",
            m.tokens_per_second()
        );
    }

    #[test]
    fn test_metrics_p50() {
        let mut m = TgpPipelineMetrics::new();
        for i in 1..=10u64 {
            m.record_token(i * 10);
        }
        let p50 = m.latency_p50_ms();
        // Sorted: 10,20,30,40,50,60,70,80,90,100 → P50 = 55 (linear interp)
        assert!((p50 - 55.0).abs() < 1.0, "p50={p50}");
    }

    #[test]
    fn test_metrics_p95() {
        let mut m = TgpPipelineMetrics::new();
        for i in 1..=20u64 {
            m.record_token(i);
        }
        let p95 = m.latency_p95_ms();
        assert!(p95 > 0.0);
    }

    #[test]
    fn test_metrics_reset() {
        let mut m = TgpPipelineMetrics::new();
        m.record_token(50);
        m.reset();
        assert_eq!(m.total_tokens(), 0);
        assert_eq!(m.total_elapsed_ms(), 0);
    }

    // ─── TgpConfig ─────────────────────────────────────────────────────────

    #[test]
    fn test_tgp_config_builder() {
        let config = TgpConfig::new(256)
            .with_temperature(0.8)
            .with_repetition_penalty(1.2)
            .with_stop_tokens(vec![1, 2])
            .with_seed(42);
        assert_eq!(config.max_length, 256);
        assert!((config.temperature - 0.8).abs() < 1e-5);
        assert!((config.repetition_penalty - 1.2).abs() < 1e-5);
        assert_eq!(config.stop_tokens, vec![1, 2]);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_tgp_config_default() {
        let config = TgpConfig::default();
        assert_eq!(config.sampling, SamplingStrategy::Greedy);
        assert!((config.temperature - 1.0).abs() < 1e-5);
        assert!((config.repetition_penalty - 1.0).abs() < 1e-5);
    }

    // ─── TextPipeline ──────────────────────────────────────────────────────

    #[test]
    fn test_pipeline_greedy_run() {
        let vocab_size = 10;
        let config = TgpConfig::new(20).with_seed(0);
        let tokenizer = Box::new(CharTokenizer::new(vocab_size));
        let logit_fn: Box<dyn FnMut(&[usize]) -> Vec<f32>> =
            Box::new(move |_| peaked_logits(vocab_size, 5));
        let mut pipeline = TextPipeline::new(config, tokenizer, logit_fn);
        let result = pipeline.run("ab", 5);
        assert_eq!(result.prompt_length, 2);
        assert_eq!(result.tokens_generated, 5);
        for &t in &result.token_ids {
            assert_eq!(t, 5);
        }
    }

    #[test]
    fn test_pipeline_stop_token() {
        let vocab_size = 10;
        let config = TgpConfig::new(50).with_stop_tokens(vec![5]).with_seed(0);
        let tokenizer = Box::new(CharTokenizer::new(vocab_size));
        let logit_fn: Box<dyn FnMut(&[usize]) -> Vec<f32>> =
            Box::new(move |_| peaked_logits(vocab_size, 5));
        let mut pipeline = TextPipeline::new(config, tokenizer, logit_fn);
        let result = pipeline.run("a", 20);
        assert!(result.stopped_by_stop_token);
        assert_eq!(result.tokens_generated, 0);
    }

    #[test]
    fn test_pipeline_max_length_cap() {
        let vocab_size = 10;
        // max_length=5, prompt="abc" (3 chars), max_tokens=10 → only 2 more can be generated.
        let config = TgpConfig::new(5).with_seed(0);
        let tokenizer = Box::new(CharTokenizer::new(vocab_size));
        let logit_fn: Box<dyn FnMut(&[usize]) -> Vec<f32>> =
            Box::new(move |_| peaked_logits(vocab_size, 1));
        let mut pipeline = TextPipeline::new(config, tokenizer, logit_fn);
        let result = pipeline.run("abc", 10);
        assert!(result.tokens_generated <= 2);
    }

    #[test]
    fn test_pipeline_decode_result() {
        let config = TgpConfig::new(30).with_seed(0);
        let tokenizer = Box::new(CharTokenizer::new(128));
        let logit_fn: Box<dyn FnMut(&[usize]) -> Vec<f32>> =
            Box::new(move |_| peaked_logits(128, 65)); // 65 = 'A'
        let mut pipeline = TextPipeline::new(config, tokenizer, logit_fn);
        let result = pipeline.run("x", 3);
        let text = pipeline.decode_result(&result);
        // All generated tokens are 65 = 'A'.
        assert_eq!(text, "AAA");
    }
}
