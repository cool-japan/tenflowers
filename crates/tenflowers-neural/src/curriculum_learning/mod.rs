//! Curriculum Learning, Self-Paced Learning & Data Valuation
//!
//! This module provides a comprehensive suite of curriculum learning strategies,
//! self-paced learning frameworks, and data valuation methods:
//!
//! - [`ClDifficultyScorer`]: Sample difficulty estimation (loss-based, margin, forgetting events)
//! - [`ClCurriculumScheduler`]: Difficulty-based training schedule with competence functions
//! - [`SelfPacedLearning`]: SPL framework (Jiang 2015) with hard/soft/mixture weighting
//! - [`CompetenceLearning`]: Competence-based curriculum (Platanios 2019)
//! - [`DataShapley`]: Monte Carlo Data Shapley values (Ghorbani & Zou 2019)
//! - [`KnnShapley`]: Exact KNN-Shapley (Jia et al. 2019)
//! - [`LavaValuation`]: OT-based data valuation (Just et al. 2023)
//! - [`AutoCurriculum`]: Bandit-based automatic curriculum with UCB1
//! - [`CurriculumTrainer`]: Full curriculum training loop
//! - [`CurriculumMetrics`]: Curriculum evaluation metrics and reporting
//!
//! # Design Principles
//!
//! * No `unwrap()` — all fallible paths surface a `TensorError`.
//! * No `unsafe` code.
//! * All public types prefixed with `Cl` to avoid name collisions.
//! * Randomness via `scirs2_core::random`.
//! * Each file stays below 2000 lines per the refactoring policy.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::TensorError;

mod helpers;
pub use helpers::*;

#[cfg(test)]
mod tests;

// Helpers

/// Simple Box-Muller normal sample (f64).
#[inline]
fn cl_sample_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random_range(1e-10..1.0);
    let u2: f64 = rng.random_range(0.0..std::f64::consts::TAU);
    (-2.0 * u1.ln()).sqrt() * u2.cos()
}

/// Partial argsort returning indices sorted by ascending value.
fn cl_argsort_asc(values: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
}

/// Partial argsort returning indices sorted by descending value.
fn cl_argsort_desc(values: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
}

/// Euclidean distance between two vectors.
fn cl_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Normalize values to [0, 1] using min-max normalization.
fn cl_normalize_min_max(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;
    if range < 1e-15 {
        return vec![0.5; values.len()];
    }
    values.iter().map(|&v| (v - min_val) / range).collect()
}

/// Fisher-Yates shuffle for a mutable slice.
fn cl_shuffle<T>(slice: &mut [T], rng: &mut StdRng) {
    let n = slice.len();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        slice.swap(i, j);
    }
}

/// Softmax for a slice of f64.
fn cl_softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum < 1e-30 {
        return vec![1.0 / logits.len() as f64; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

// Section 1 — ClDifficultyScorer

/// Method for estimating sample difficulty.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClDifficultyMethod {
    /// Higher loss = harder sample.
    LossBased,
    /// Distance between top-2 class probabilities (smaller margin = harder).
    PredictionMargin,
    /// Count of forgetting events (correct -> incorrect transitions).
    ForgettingEvents,
    /// Ensemble disagreement (higher = harder).
    EnsembleDisagreement,
    /// Combined: weighted average of loss + margin + forgetting.
    Combined,
}

/// Sample difficulty estimation for curriculum learning.
///
/// Tracks per-sample statistics across training to estimate difficulty.
#[derive(Debug, Clone)]
pub struct ClDifficultyScorer {
    /// Which method to use for difficulty estimation.
    pub method: ClDifficultyMethod,
    /// Number of samples being tracked.
    pub n_samples: usize,
    /// Accumulated loss per sample (running sum).
    pub accumulated_losses: Vec<f64>,
    /// Number of loss observations per sample.
    pub loss_counts: Vec<usize>,
    /// Forgetting event counts per sample.
    pub forgetting_counts: Vec<usize>,
    /// Previous correctness state (true = was correct last time).
    pub prev_correct: Vec<bool>,
    /// Prediction margin history (running sum of margins).
    pub accumulated_margins: Vec<f64>,
    /// Number of margin observations per sample.
    pub margin_counts: Vec<usize>,
    /// Weights for combined method: [loss_weight, margin_weight, forgetting_weight].
    pub combined_weights: [f64; 3],
}

impl ClDifficultyScorer {
    /// Create a new difficulty scorer.
    ///
    /// # Arguments
    /// * `n_samples` - Number of samples in the dataset.
    /// * `method` - Difficulty estimation method.
    pub fn new(n_samples: usize, method: ClDifficultyMethod) -> Result<Self, TensorError> {
        if n_samples == 0 {
            return Err(TensorError::compute_error_simple(
                "ClDifficultyScorer: n_samples must be > 0".to_string(),
            ));
        }
        Ok(Self {
            method,
            n_samples,
            accumulated_losses: vec![0.0; n_samples],
            loss_counts: vec![0; n_samples],
            forgetting_counts: vec![0; n_samples],
            prev_correct: vec![false; n_samples],
            accumulated_margins: vec![0.0; n_samples],
            margin_counts: vec![0; n_samples],
            combined_weights: [0.4, 0.3, 0.3],
        })
    }

    /// Set custom weights for the combined method.
    pub fn set_combined_weights(&mut self, loss_w: f64, margin_w: f64, forgetting_w: f64) {
        self.combined_weights = [loss_w, margin_w, forgetting_w];
    }

    /// Update statistics for a batch of samples.
    ///
    /// # Arguments
    /// * `indices` - Sample indices in this batch.
    /// * `losses` - Per-sample losses for this batch.
    /// * `predictions` - Per-sample predictions (class probabilities), shape [batch, n_classes].
    /// * `labels` - Per-sample true label indices.
    pub fn update_batch(
        &mut self,
        indices: &[usize],
        losses: &[f64],
        predictions: &[Vec<f64>],
        labels: &[usize],
    ) -> Result<(), TensorError> {
        if indices.len() != losses.len()
            || indices.len() != predictions.len()
            || indices.len() != labels.len()
        {
            return Err(TensorError::compute_error_simple(
                "ClDifficultyScorer::update_batch: mismatched batch sizes".to_string(),
            ));
        }

        for (batch_idx, &sample_idx) in indices.iter().enumerate() {
            if sample_idx >= self.n_samples {
                return Err(TensorError::compute_error_simple(format!(
                    "ClDifficultyScorer: sample index {} >= n_samples {}",
                    sample_idx, self.n_samples
                )));
            }

            // Update loss statistics
            self.accumulated_losses[sample_idx] += losses[batch_idx];
            self.loss_counts[sample_idx] += 1;

            // Update margin statistics
            let preds = &predictions[batch_idx];
            if preds.len() >= 2 {
                let mut sorted_preds = preds.clone();
                sorted_preds.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let margin = sorted_preds[0] - sorted_preds[1];
                self.accumulated_margins[sample_idx] += margin;
                self.margin_counts[sample_idx] += 1;
            }

            // Update forgetting events
            let label = labels[batch_idx];
            let is_correct = if label < preds.len() {
                let predicted_class = preds
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                predicted_class == label
            } else {
                false
            };

            if self.prev_correct[sample_idx] && !is_correct {
                self.forgetting_counts[sample_idx] += 1;
            }
            self.prev_correct[sample_idx] = is_correct;
        }

        Ok(())
    }

    /// Compute difficulty scores for a batch of samples.
    ///
    /// # Arguments
    /// * `losses` - Per-sample losses.
    /// * `predictions` - Per-sample class probabilities [batch, n_classes].
    /// * `labels` - Per-sample true label indices.
    ///
    /// Returns normalized difficulty scores in [0, 1].
    pub fn score_batch(
        &self,
        losses: &[f64],
        predictions: &[Vec<f64>],
        labels: &[usize],
    ) -> Result<Vec<f64>, TensorError> {
        if losses.len() != predictions.len() || losses.len() != labels.len() {
            return Err(TensorError::compute_error_simple(
                "ClDifficultyScorer::score_batch: mismatched sizes".to_string(),
            ));
        }
        if losses.is_empty() {
            return Err(TensorError::compute_error_simple(
                "ClDifficultyScorer::score_batch: empty batch".to_string(),
            ));
        }

        let raw_scores = match self.method {
            ClDifficultyMethod::LossBased => losses.to_vec(),
            ClDifficultyMethod::PredictionMargin => {
                let mut margins = Vec::with_capacity(predictions.len());
                for preds in predictions {
                    if preds.len() < 2 {
                        margins.push(1.0); // Hardest if can't compute margin
                    } else {
                        let mut sorted = preds.clone();
                        sorted
                            .sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                        // Smaller margin = harder, so invert
                        margins.push(1.0 - (sorted[0] - sorted[1]).clamp(0.0, 1.0));
                    }
                }
                margins
            }
            ClDifficultyMethod::ForgettingEvents => {
                // Use current accumulated forgetting counts normalized
                let mut scores = Vec::with_capacity(losses.len());
                for i in 0..losses.len() {
                    if i < self.forgetting_counts.len() {
                        scores.push(self.forgetting_counts[i] as f64);
                    } else {
                        scores.push(0.0);
                    }
                }
                scores
            }
            ClDifficultyMethod::EnsembleDisagreement => {
                // Use entropy of predictions as proxy for disagreement
                let mut entropies = Vec::with_capacity(predictions.len());
                for preds in predictions {
                    let probs = cl_softmax(preds);
                    let entropy: f64 = probs
                        .iter()
                        .filter(|&&p| p > 1e-15)
                        .map(|&p| -p * p.ln())
                        .sum();
                    entropies.push(entropy);
                }
                entropies
            }
            ClDifficultyMethod::Combined => {
                let loss_scores = cl_normalize_min_max(losses);
                let mut margin_scores = Vec::with_capacity(predictions.len());
                for preds in predictions {
                    if preds.len() < 2 {
                        margin_scores.push(1.0);
                    } else {
                        let mut sorted = preds.clone();
                        sorted
                            .sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                        margin_scores.push(1.0 - (sorted[0] - sorted[1]).clamp(0.0, 1.0));
                    }
                }
                let margin_norm = cl_normalize_min_max(&margin_scores);

                let mut forget_raw = Vec::with_capacity(losses.len());
                for i in 0..losses.len() {
                    if i < self.forgetting_counts.len() {
                        forget_raw.push(self.forgetting_counts[i] as f64);
                    } else {
                        forget_raw.push(0.0);
                    }
                }
                let forget_norm = cl_normalize_min_max(&forget_raw);

                let [w_l, w_m, w_f] = self.combined_weights;
                let total_w = w_l + w_m + w_f;
                let total_w = if total_w < 1e-15 { 1.0 } else { total_w };

                loss_scores
                    .iter()
                    .zip(margin_norm.iter())
                    .zip(forget_norm.iter())
                    .map(|((&l, &m), &f)| (w_l * l + w_m * m + w_f * f) / total_w)
                    .collect()
            }
        };

        Ok(cl_normalize_min_max(&raw_scores))
    }

    /// Get average difficulty score per sample from accumulated statistics.
    pub fn get_accumulated_difficulties(&self) -> Vec<f64> {
        let mut difficulties = Vec::with_capacity(self.n_samples);
        for i in 0..self.n_samples {
            let avg_loss = if self.loss_counts[i] > 0 {
                self.accumulated_losses[i] / self.loss_counts[i] as f64
            } else {
                0.0
            };
            difficulties.push(avg_loss);
        }
        cl_normalize_min_max(&difficulties)
    }

    /// Reset all accumulated statistics.
    pub fn reset(&mut self) {
        self.accumulated_losses.fill(0.0);
        self.loss_counts.fill(0);
        self.forgetting_counts.fill(0);
        self.prev_correct.fill(false);
        self.accumulated_margins.fill(0.0);
        self.margin_counts.fill(0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 2 — ClCurriculumScheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Competence function strategy: determines what fraction of data is accessible
/// at a given training step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClCompetenceStrategy {
    /// c(t) = t / T (linear ramp).
    Linear,
    /// c(t) = sqrt(t / T) (concave root curve, faster start).
    Root,
    /// c(t) = 1 - exp(-5 * t / T) (exponential approach to 1).
    Exponential,
    /// Piecewise step: jumps at [0.25, 0.5, 0.75, 1.0] of T.
    Step,
    /// Logarithmic: c(t) = log(1 + t) / log(1 + T).
    Logarithmic,
}

/// Baby-step curriculum configuration for discrete difficulty bins.
#[derive(Debug, Clone)]
pub struct ClBabyStepConfig {
    /// Number of difficulty bins.
    pub n_bins: usize,
    /// Number of steps per bin before advancing.
    pub steps_per_bin: usize,
    /// Current bin index.
    pub current_bin: usize,
    /// Steps elapsed in the current bin.
    pub steps_in_bin: usize,
}

impl ClBabyStepConfig {
    /// Create a new baby-step configuration.
    pub fn new(n_bins: usize, steps_per_bin: usize) -> Result<Self, TensorError> {
        if n_bins == 0 {
            return Err(TensorError::compute_error_simple(
                "ClBabyStepConfig: n_bins must be > 0".to_string(),
            ));
        }
        if steps_per_bin == 0 {
            return Err(TensorError::compute_error_simple(
                "ClBabyStepConfig: steps_per_bin must be > 0".to_string(),
            ));
        }
        Ok(Self {
            n_bins,
            steps_per_bin,
            current_bin: 0,
            steps_in_bin: 0,
        })
    }

    /// Advance one step and return current accessible bin count.
    pub fn step(&mut self) -> usize {
        self.steps_in_bin += 1;
        if self.steps_in_bin >= self.steps_per_bin && self.current_bin + 1 < self.n_bins {
            self.current_bin += 1;
            self.steps_in_bin = 0;
        }
        self.current_bin + 1
    }

    /// Reset the baby-step state.
    pub fn reset(&mut self) {
        self.current_bin = 0;
        self.steps_in_bin = 0;
    }
}

/// Difficulty-based training schedule that controls which samples are
/// accessible at each training step.
#[derive(Debug)]
pub struct ClCurriculumScheduler {
    /// Competence function strategy.
    pub strategy: ClCompetenceStrategy,
    /// Total number of training steps (T).
    pub total_steps: usize,
    /// Current step counter.
    pub current_step: usize,
    /// Minimum competence (starting value).
    pub min_competence: f64,
    /// Optional baby-step configuration.
    pub baby_step: Option<ClBabyStepConfig>,
    /// Whether to use anti-curriculum (hard-first).
    pub anti_curriculum: bool,
    /// RNG for stochastic selection.
    rng: StdRng,
}

impl Clone for ClCurriculumScheduler {
    fn clone(&self) -> Self {
        Self {
            strategy: self.strategy.clone(),
            total_steps: self.total_steps,
            current_step: self.current_step,
            min_competence: self.min_competence,
            baby_step: self.baby_step.clone(),
            anti_curriculum: self.anti_curriculum,
            rng: StdRng::seed_from_u64(0xCE55_5CED),
        }
    }
}

impl ClCurriculumScheduler {
    /// Create a new curriculum scheduler.
    ///
    /// # Arguments
    /// * `strategy` - Competence function type.
    /// * `total_steps` - Total training steps (T).
    /// * `seed` - Random seed.
    pub fn new(
        strategy: ClCompetenceStrategy,
        total_steps: usize,
        seed: u64,
    ) -> Result<Self, TensorError> {
        if total_steps == 0 {
            return Err(TensorError::compute_error_simple(
                "ClCurriculumScheduler: total_steps must be > 0".to_string(),
            ));
        }
        Ok(Self {
            strategy,
            total_steps,
            current_step: 0,
            min_competence: 0.01,
            baby_step: None,
            anti_curriculum: false,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// Set baby-step configuration.
    pub fn set_baby_step(&mut self, config: ClBabyStepConfig) {
        self.baby_step = Some(config);
    }

    /// Set anti-curriculum mode (hard-first).
    pub fn set_anti_curriculum(&mut self, enabled: bool) {
        self.anti_curriculum = enabled;
    }

    /// Set minimum competence value.
    pub fn set_min_competence(&mut self, min_c: f64) {
        self.min_competence = min_c.clamp(0.0, 1.0);
    }

    /// Compute the competence value c(t) at the current step.
    pub fn competence(&self) -> f64 {
        let t = self.current_step as f64;
        let total = self.total_steps as f64;

        let raw = match self.strategy {
            ClCompetenceStrategy::Linear => t / total,
            ClCompetenceStrategy::Root => (t / total).sqrt(),
            ClCompetenceStrategy::Exponential => 1.0 - (-5.0 * t / total).exp(),
            ClCompetenceStrategy::Step => {
                if t < 0.25 * total {
                    0.25
                } else if t < 0.5 * total {
                    0.5
                } else if t < 0.75 * total {
                    0.75
                } else {
                    1.0
                }
            }
            ClCompetenceStrategy::Logarithmic => (1.0 + t).ln() / (1.0 + total).ln(),
        };

        raw.clamp(self.min_competence, 1.0)
    }

    /// Select a batch of sample indices based on difficulty and current competence.
    ///
    /// Samples with difficulty <= c(t) are eligible. From those, `batch_size` are
    /// randomly selected (without replacement).
    ///
    /// # Arguments
    /// * `difficulties` - Normalized difficulty scores [0, 1] per sample.
    /// * `batch_size` - Desired batch size.
    ///
    /// Returns selected sample indices.
    pub fn select_batch(
        &mut self,
        difficulties: &[f64],
        batch_size: usize,
    ) -> Result<Vec<usize>, TensorError> {
        if difficulties.is_empty() {
            return Err(TensorError::compute_error_simple(
                "ClCurriculumScheduler::select_batch: empty difficulties".to_string(),
            ));
        }
        if batch_size == 0 {
            return Err(TensorError::compute_error_simple(
                "ClCurriculumScheduler::select_batch: batch_size must be > 0".to_string(),
            ));
        }

        // Handle baby-step mode
        if let Some(ref mut baby) = self.baby_step {
            let accessible_bins = baby.step();
            let bin_threshold = accessible_bins as f64 / baby.n_bins as f64;
            let eligible: Vec<usize> = difficulties
                .iter()
                .enumerate()
                .filter(|(_, &d)| d <= bin_threshold)
                .map(|(i, _)| i)
                .collect();

            if eligible.is_empty() {
                // Fall back to easiest samples
                let sorted = cl_argsort_asc(difficulties);
                let take = batch_size.min(sorted.len());
                return Ok(sorted[..take].to_vec());
            }

            let mut pool = eligible;
            cl_shuffle(&mut pool, &mut self.rng);
            let take = batch_size.min(pool.len());
            return Ok(pool[..take].to_vec());
        }

        let c = self.competence();

        let eligible: Vec<usize> = if self.anti_curriculum {
            // Hard-first: select samples with difficulty >= (1 - c)
            let threshold = 1.0 - c;
            difficulties
                .iter()
                .enumerate()
                .filter(|(_, &d)| d >= threshold)
                .map(|(i, _)| i)
                .collect()
        } else {
            // Standard curriculum: select samples with difficulty <= c
            difficulties
                .iter()
                .enumerate()
                .filter(|(_, &d)| d <= c)
                .map(|(i, _)| i)
                .collect()
        };

        if eligible.is_empty() {
            // Fall back: take easiest/hardest samples
            let sorted = if self.anti_curriculum {
                cl_argsort_desc(difficulties)
            } else {
                cl_argsort_asc(difficulties)
            };
            let take = batch_size.min(sorted.len());
            return Ok(sorted[..take].to_vec());
        }

        let mut pool = eligible;
        cl_shuffle(&mut pool, &mut self.rng);
        let take = batch_size.min(pool.len());
        Ok(pool[..take].to_vec())
    }

    /// Advance the scheduler by one step.
    pub fn step(&mut self) {
        self.current_step = (self.current_step + 1).min(self.total_steps);
    }

    /// Reset the scheduler to step 0.
    pub fn reset(&mut self) {
        self.current_step = 0;
        if let Some(ref mut baby) = self.baby_step {
            baby.reset();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3 — SelfPacedLearning
// ─────────────────────────────────────────────────────────────────────────────

/// Self-paced weighting strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClSplWeightingStrategy {
    /// Hard SPL: v = 1 if loss < lambda, else 0.
    Hard,
    /// Soft SPL: v = max(0, 1 - loss/lambda).
    Soft,
    /// Mixture SPL: linear interpolation between hard and soft.
    Mixture,
    /// Logarithmic SPL: v = max(0, lambda / (lambda + loss)).
    Logarithmic,
}

/// Lambda schedule for self-paced learning threshold annealing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClLambdaSchedule {
    /// Linear increase: lambda(t) = lambda_init + (lambda_max - lambda_init) * t / T.
    Linear,
    /// Geometric increase: lambda(t) = lambda_init * (lambda_max/lambda_init)^(t/T).
    Geometric,
    /// Step increase: double lambda every `step_interval` epochs.
    StepDouble,
}

/// Self-Paced Learning framework (Jiang 2015).
///
/// Implements sample weighting based on loss thresholds, with progressive
/// anneal from easy to hard.
#[derive(Debug, Clone)]
pub struct SelfPacedLearning {
    /// Weighting strategy.
    pub strategy: ClSplWeightingStrategy,
    /// Current lambda threshold.
    pub lambda: f64,
    /// Initial lambda value.
    pub lambda_init: f64,
    /// Maximum lambda value.
    pub lambda_max: f64,
    /// Lambda schedule type.
    pub schedule: ClLambdaSchedule,
    /// Mixture coefficient (for Mixture strategy): interpolation between hard and soft.
    pub mixture_alpha: f64,
    /// Current epoch.
    pub current_epoch: usize,
    /// Total epochs (T).
    pub total_epochs: usize,
    /// Step interval for StepDouble schedule.
    pub step_interval: usize,
}

impl SelfPacedLearning {
    /// Create a new SPL instance.
    ///
    /// # Arguments
    /// * `strategy` - Weighting strategy.
    /// * `lambda_init` - Initial lambda threshold (start with easy samples).
    /// * `lambda_max` - Maximum lambda threshold (eventually include all).
    /// * `total_epochs` - Total training epochs.
    pub fn new(
        strategy: ClSplWeightingStrategy,
        lambda_init: f64,
        lambda_max: f64,
        total_epochs: usize,
    ) -> Result<Self, TensorError> {
        if lambda_init <= 0.0 {
            return Err(TensorError::compute_error_simple(
                "SelfPacedLearning: lambda_init must be > 0".to_string(),
            ));
        }
        if lambda_max < lambda_init {
            return Err(TensorError::compute_error_simple(
                "SelfPacedLearning: lambda_max must be >= lambda_init".to_string(),
            ));
        }
        if total_epochs == 0 {
            return Err(TensorError::compute_error_simple(
                "SelfPacedLearning: total_epochs must be > 0".to_string(),
            ));
        }
        Ok(Self {
            strategy,
            lambda: lambda_init,
            lambda_init,
            lambda_max,
            schedule: ClLambdaSchedule::Linear,
            mixture_alpha: 0.5,
            current_epoch: 0,
            total_epochs,
            step_interval: 5,
        })
    }

    /// Set the lambda schedule type.
    pub fn set_schedule(&mut self, schedule: ClLambdaSchedule) {
        self.schedule = schedule;
    }

    /// Set the mixture coefficient for Mixture strategy.
    pub fn set_mixture_alpha(&mut self, alpha: f64) {
        self.mixture_alpha = alpha.clamp(0.0, 1.0);
    }

    /// Set the step interval for StepDouble schedule.
    pub fn set_step_interval(&mut self, interval: usize) {
        self.step_interval = interval.max(1);
    }

    /// Compute sample weights based on their losses and the current lambda.
    ///
    /// # Arguments
    /// * `losses` - Per-sample losses.
    ///
    /// Returns sample weights in [0, 1].
    pub fn compute_weights(&self, losses: &[f64]) -> Result<Vec<f64>, TensorError> {
        if losses.is_empty() {
            return Err(TensorError::compute_error_simple(
                "SelfPacedLearning::compute_weights: empty losses".to_string(),
            ));
        }

        let weights: Vec<f64> = losses
            .iter()
            .map(|&loss| self.weight_single(loss))
            .collect();
        Ok(weights)
    }

    /// Compute weight for a single sample.
    fn weight_single(&self, loss: f64) -> f64 {
        match self.strategy {
            ClSplWeightingStrategy::Hard => {
                if loss < self.lambda {
                    1.0
                } else {
                    0.0
                }
            }
            ClSplWeightingStrategy::Soft => (1.0 - loss / self.lambda).max(0.0),
            ClSplWeightingStrategy::Mixture => {
                let hard = if loss < self.lambda { 1.0 } else { 0.0 };
                let soft = (1.0 - loss / self.lambda).max(0.0);
                self.mixture_alpha * hard + (1.0 - self.mixture_alpha) * soft
            }
            ClSplWeightingStrategy::Logarithmic => (self.lambda / (self.lambda + loss)).max(0.0),
        }
    }

    /// Advance one epoch and update lambda according to the schedule.
    pub fn step_epoch(&mut self) {
        self.current_epoch += 1;
        let t = self.current_epoch as f64;
        let total = self.total_epochs as f64;

        self.lambda = match self.schedule {
            ClLambdaSchedule::Linear => {
                self.lambda_init + (self.lambda_max - self.lambda_init) * (t / total)
            }
            ClLambdaSchedule::Geometric => {
                if self.lambda_init > 0.0 {
                    let ratio = self.lambda_max / self.lambda_init;
                    self.lambda_init * ratio.powf(t / total)
                } else {
                    self.lambda_max * (t / total)
                }
            }
            ClLambdaSchedule::StepDouble => {
                let doublings = (self.current_epoch / self.step_interval) as f64;
                (self.lambda_init * 2.0_f64.powf(doublings)).min(self.lambda_max)
            }
        };
        self.lambda = self.lambda.min(self.lambda_max);
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.lambda = self.lambda_init;
        self.current_epoch = 0;
    }

    /// Compute the weighted loss: sum(w_i * loss_i) / sum(w_i).
    pub fn weighted_loss(&self, losses: &[f64]) -> Result<f64, TensorError> {
        let weights = self.compute_weights(losses)?;
        let weighted_sum: f64 = weights
            .iter()
            .zip(losses.iter())
            .map(|(&w, &l)| w * l)
            .sum();
        let weight_sum: f64 = weights.iter().sum();
        if weight_sum < 1e-15 {
            return Ok(0.0);
        }
        Ok(weighted_sum / weight_sum)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4 — CompetenceLearning
// ─────────────────────────────────────────────────────────────────────────────

/// Competence-based curriculum learning (Platanios 2019).
///
/// Measures model competence from validation performance, and selects
/// training samples whose difficulty does not exceed the current competence.
#[derive(Debug, Clone)]
pub struct CompetenceLearning {
    /// Current model competence c in [0, 1].
    pub competence: f64,
    /// Minimum competence floor.
    pub min_competence: f64,
    /// Momentum for smoothing competence updates.
    pub momentum: f64,
    /// Competence history.
    pub competence_history: Vec<f64>,
    /// Number of validation updates performed.
    pub n_updates: usize,
    /// Confidence method: accuracy-based or entropy-based.
    pub confidence_method: ClConfidenceMethod,
}

/// Method for computing model confidence from validation data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClConfidenceMethod {
    /// Accuracy on validation set (fraction of correct predictions).
    Accuracy,
    /// Mean confidence of correct predictions.
    MeanConfidence,
    /// 1 - mean entropy of predictions (lower entropy = higher competence).
    InverseEntropy,
}

impl CompetenceLearning {
    /// Create a new competence learner.
    ///
    /// # Arguments
    /// * `min_competence` - Minimum competence floor.
    /// * `momentum` - Exponential moving average momentum for smoothing.
    pub fn new(min_competence: f64, momentum: f64) -> Result<Self, TensorError> {
        if !(0.0..=1.0).contains(&momentum) {
            return Err(TensorError::compute_error_simple(
                "CompetenceLearning: momentum must be in [0, 1]".to_string(),
            ));
        }
        Ok(Self {
            competence: min_competence.clamp(0.0, 1.0),
            min_competence: min_competence.clamp(0.0, 1.0),
            momentum,
            competence_history: Vec::new(),
            n_updates: 0,
            confidence_method: ClConfidenceMethod::Accuracy,
        })
    }

    /// Set the confidence computation method.
    pub fn set_confidence_method(&mut self, method: ClConfidenceMethod) {
        self.confidence_method = method;
    }

    /// Update competence based on validation predictions and labels.
    ///
    /// # Arguments
    /// * `val_predictions` - Validation predictions [n_samples, n_classes].
    /// * `val_labels` - True label indices.
    ///
    /// Returns the updated competence score.
    pub fn update_competence(
        &mut self,
        val_predictions: &[Vec<f64>],
        val_labels: &[usize],
    ) -> Result<f64, TensorError> {
        if val_predictions.len() != val_labels.len() {
            return Err(TensorError::compute_error_simple(
                "CompetenceLearning: prediction/label count mismatch".to_string(),
            ));
        }
        if val_predictions.is_empty() {
            return Err(TensorError::compute_error_simple(
                "CompetenceLearning: empty validation set".to_string(),
            ));
        }

        let raw_competence = match self.confidence_method {
            ClConfidenceMethod::Accuracy => {
                let correct_count = val_predictions
                    .iter()
                    .zip(val_labels.iter())
                    .filter(|(preds, &label)| {
                        preds
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(idx, _)| idx == label)
                            .unwrap_or(false)
                    })
                    .count();
                correct_count as f64 / val_predictions.len() as f64
            }
            ClConfidenceMethod::MeanConfidence => {
                let sum: f64 = val_predictions
                    .iter()
                    .zip(val_labels.iter())
                    .map(|(preds, &label)| {
                        if label < preds.len() {
                            preds[label].max(0.0)
                        } else {
                            0.0
                        }
                    })
                    .sum();
                sum / val_predictions.len() as f64
            }
            ClConfidenceMethod::InverseEntropy => {
                let mean_entropy: f64 = val_predictions
                    .iter()
                    .map(|preds| {
                        // Use predictions directly as probabilities (consistent
                        // with the Accuracy and MeanConfidence branches).
                        // Clamp to [0, 1] and re-normalize to ensure a valid
                        // distribution, without applying softmax which would
                        // squash confident predictions toward uniform.
                        let raw: Vec<f64> = preds.iter().map(|&p| p.max(0.0)).collect();
                        let sum: f64 = raw.iter().sum();
                        let probs: Vec<f64> = if sum > 1e-15 {
                            raw.iter().map(|&p| p / sum).collect()
                        } else {
                            vec![1.0 / preds.len().max(1) as f64; preds.len()]
                        };
                        let n_classes = probs.len().max(2) as f64;
                        let max_entropy = n_classes.ln();
                        let entropy: f64 = probs
                            .iter()
                            .filter(|&&p| p > 1e-15)
                            .map(|&p| -p * p.ln())
                            .sum();
                        if max_entropy > 1e-15 {
                            entropy / max_entropy
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>()
                    / val_predictions.len() as f64;
                1.0 - mean_entropy
            }
        };

        // Smoothed update with EMA
        if self.n_updates == 0 {
            self.competence = raw_competence;
        } else {
            self.competence =
                self.momentum * self.competence + (1.0 - self.momentum) * raw_competence;
        }
        self.competence = self.competence.clamp(self.min_competence, 1.0);
        self.competence_history.push(self.competence);
        self.n_updates += 1;

        Ok(self.competence)
    }

    /// Filter training samples by competence: only samples with difficulty <= competence
    /// are eligible for training.
    ///
    /// # Arguments
    /// * `difficulties` - Per-sample normalized difficulty scores in [0, 1].
    ///
    /// Returns indices of eligible samples.
    pub fn filter_by_competence(&self, difficulties: &[f64]) -> Result<Vec<usize>, TensorError> {
        if difficulties.is_empty() {
            return Err(TensorError::compute_error_simple(
                "CompetenceLearning::filter_by_competence: empty difficulties".to_string(),
            ));
        }

        let eligible: Vec<usize> = difficulties
            .iter()
            .enumerate()
            .filter(|(_, &d)| d <= self.competence)
            .map(|(i, _)| i)
            .collect();

        // If no samples eligible, return easiest one
        if eligible.is_empty() {
            let sorted = cl_argsort_asc(difficulties);
            return Ok(vec![sorted[0]]);
        }

        Ok(eligible)
    }

    /// Estimate sample difficulty from cross-model agreement.
    ///
    /// # Arguments
    /// * `model_predictions` - Predictions from M models, each [n_samples, n_classes].
    ///
    /// Returns per-sample difficulty in [0, 1].
    pub fn estimate_difficulty_from_agreement(
        &self,
        model_predictions: &[Vec<Vec<f64>>],
    ) -> Result<Vec<f64>, TensorError> {
        if model_predictions.is_empty() {
            return Err(TensorError::compute_error_simple(
                "CompetenceLearning: empty model predictions".to_string(),
            ));
        }

        let n_samples = model_predictions[0].len();
        let n_models = model_predictions.len();

        if n_models < 2 {
            // With only one model, use entropy as proxy
            let mut difficulties = Vec::with_capacity(n_samples);
            for preds in &model_predictions[0] {
                let probs = cl_softmax(preds);
                let n_c = probs.len().max(2) as f64;
                let max_ent = n_c.ln();
                let ent: f64 = probs
                    .iter()
                    .filter(|&&p| p > 1e-15)
                    .map(|&p| -p * p.ln())
                    .sum();
                let norm_ent = if max_ent > 1e-15 { ent / max_ent } else { 0.0 };
                difficulties.push(norm_ent);
            }
            return Ok(cl_normalize_min_max(&difficulties));
        }

        // Disagreement: fraction of model pairs that predict different classes
        let mut disagreements = vec![0.0_f64; n_samples];
        let n_pairs = n_models * (n_models - 1) / 2;

        for i in 0..n_samples {
            let mut disagree_count = 0usize;
            for m1 in 0..n_models {
                for m2 in (m1 + 1)..n_models {
                    if m1 < model_predictions.len()
                        && i < model_predictions[m1].len()
                        && m2 < model_predictions.len()
                        && i < model_predictions[m2].len()
                    {
                        let pred1 = model_predictions[m1][i]
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(idx, _)| idx)
                            .unwrap_or(0);
                        let pred2 = model_predictions[m2][i]
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(idx, _)| idx)
                            .unwrap_or(0);
                        if pred1 != pred2 {
                            disagree_count += 1;
                        }
                    }
                }
            }
            disagreements[i] = if n_pairs > 0 {
                disagree_count as f64 / n_pairs as f64
            } else {
                0.0
            };
        }

        Ok(cl_normalize_min_max(&disagreements))
    }

    /// Reset competence to the minimum floor.
    pub fn reset(&mut self) {
        self.competence = self.min_competence;
        self.competence_history.clear();
        self.n_updates = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5 — DataShapley
// ─────────────────────────────────────────────────────────────────────────────

/// Data Shapley value computation (Ghorbani & Zou 2019).
///
/// Uses Monte Carlo permutation sampling to estimate the Shapley value
/// of each training data point.
#[derive(Debug, Clone)]
pub struct DataShapley {
    /// Number of Monte Carlo permutations.
    pub n_permutations: usize,
    /// Truncation threshold: stop if marginal contribution < epsilon.
    pub truncation_epsilon: f64,
    /// Convergence tolerance for early stopping.
    pub convergence_tol: f64,
    /// Random seed.
    pub seed: u64,
}

impl DataShapley {
    /// Create a new DataShapley estimator.
    ///
    /// # Arguments
    /// * `n_permutations` - Number of MC permutation samples.
    /// * `truncation_epsilon` - Truncated MC threshold.
    pub fn new(n_permutations: usize, truncation_epsilon: f64) -> Result<Self, TensorError> {
        if n_permutations == 0 {
            return Err(TensorError::compute_error_simple(
                "DataShapley: n_permutations must be > 0".to_string(),
            ));
        }
        Ok(Self {
            n_permutations,
            truncation_epsilon: truncation_epsilon.max(0.0),
            convergence_tol: 1e-4,
            seed: 42,
        })
    }

    /// Set the random seed.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    /// Set convergence tolerance.
    pub fn set_convergence_tol(&mut self, tol: f64) {
        self.convergence_tol = tol.max(0.0);
    }

    /// Compute Data Shapley values using Monte Carlo permutation sampling.
    ///
    /// # Arguments
    /// * `n_train` - Number of training samples.
    /// * `model_fn` - Function that trains on a subset (given as sorted indices) and
    ///   returns validation performance (higher = better).
    ///
    /// Returns Shapley value for each training sample.
    pub fn compute<F>(&self, n_train: usize, model_fn: &F) -> Result<Vec<f64>, TensorError>
    where
        F: Fn(&[usize]) -> Result<f64, TensorError>,
    {
        if n_train == 0 {
            return Err(TensorError::compute_error_simple(
                "DataShapley::compute: n_train must be > 0".to_string(),
            ));
        }

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut shapley_values = vec![0.0_f64; n_train];
        let mut counts = vec![0usize; n_train];

        for _perm in 0..self.n_permutations {
            // Generate random permutation
            let mut perm: Vec<usize> = (0..n_train).collect();
            cl_shuffle(&mut perm, &mut rng);

            let mut prev_perf = model_fn(&[])?;
            let mut subset: Vec<usize> = Vec::with_capacity(n_train);

            for &idx in &perm {
                subset.push(idx);
                subset.sort();
                let current_perf = model_fn(&subset)?;
                let marginal = current_perf - prev_perf;

                shapley_values[idx] += marginal;
                counts[idx] += 1;

                prev_perf = current_perf;

                // Truncated MC: stop if marginal contribution is negligible
                if marginal.abs() < self.truncation_epsilon && subset.len() > 1 {
                    // Still assign 0 marginal contribution for remaining
                    break;
                }
            }
        }

        // Average over permutations
        for i in 0..n_train {
            if counts[i] > 0 {
                shapley_values[i] /= counts[i] as f64;
            }
        }

        Ok(shapley_values)
    }

    /// Identify high-value and low-value data points.
    ///
    /// Returns (high_value_indices, low_value_indices).
    pub fn identify_value_groups(
        &self,
        shapley_values: &[f64],
        high_percentile: f64,
        low_percentile: f64,
    ) -> Result<(Vec<usize>, Vec<usize>), TensorError> {
        if shapley_values.is_empty() {
            return Err(TensorError::compute_error_simple(
                "DataShapley: empty shapley values".to_string(),
            ));
        }

        let sorted = cl_argsort_asc(shapley_values);
        let n = sorted.len();

        let low_count = ((low_percentile / 100.0) * n as f64).ceil() as usize;
        let high_start = ((1.0 - high_percentile / 100.0) * n as f64).floor() as usize;

        let low_indices = sorted[..low_count.min(n)].to_vec();
        let high_indices = sorted[high_start.min(n)..].to_vec();

        Ok((high_indices, low_indices))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6 — KnnShapley
// ─────────────────────────────────────────────────────────────────────────────

/// Exact KNN-Shapley computation (Jia et al. 2019).
///
/// Computes exact Shapley values for KNN classifiers in O(n * k * log(n)) time.
#[derive(Debug, Clone)]
pub struct KnnShapley {
    /// Number of neighbors (k).
    pub k: usize,
}

impl KnnShapley {
    /// Create a new KNN-Shapley estimator.
    pub fn new(k: usize) -> Result<Self, TensorError> {
        if k == 0 {
            return Err(TensorError::compute_error_simple(
                "KnnShapley: k must be > 0".to_string(),
            ));
        }
        Ok(Self { k })
    }

    /// Compute KNN-Shapley values.
    ///
    /// For each test point, compute the exact Shapley value of each training point
    /// using the recursive formula:
    ///   phi(n) = s(z_n, y*) / n
    ///   phi(j) = phi(j+1) + [s(z_j, y*) - s(z_{j+1}, y*)] / max(j, k)
    ///
    /// where s(z, y*) = 1 if z's label matches y*, else 0.
    ///
    /// # Arguments
    /// * `train_features` - Training feature vectors [n_train, dim].
    /// * `train_labels` - Training labels.
    /// * `test_features` - Test feature vectors [n_test, dim].
    /// * `test_labels` - Test labels (for evaluation).
    ///
    /// Returns per-training-sample Shapley values, averaged over test points.
    pub fn compute(
        &self,
        train_features: &[Vec<f64>],
        train_labels: &[usize],
        test_features: &[Vec<f64>],
        test_labels: &[usize],
    ) -> Result<Vec<f64>, TensorError> {
        if train_features.is_empty() || test_features.is_empty() {
            return Err(TensorError::compute_error_simple(
                "KnnShapley: empty features".to_string(),
            ));
        }
        if train_features.len() != train_labels.len() {
            return Err(TensorError::compute_error_simple(
                "KnnShapley: train feature/label count mismatch".to_string(),
            ));
        }
        if test_features.len() != test_labels.len() {
            return Err(TensorError::compute_error_simple(
                "KnnShapley: test feature/label count mismatch".to_string(),
            ));
        }

        let n_train = train_features.len();
        let n_test = test_features.len();
        let k = self.k.min(n_train);

        let mut total_shapley = vec![0.0_f64; n_train];

        for t in 0..n_test {
            let test_label = test_labels[t];

            // Compute distances from test point to all training points
            let mut dists: Vec<(usize, f64)> = train_features
                .iter()
                .enumerate()
                .map(|(i, feat)| (i, cl_euclidean_distance(feat, &test_features[t])))
                .collect();

            // Sort by distance (ascending)
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // s(z, y*) = 1 if label matches, else 0
            let s = |train_idx: usize| -> f64 {
                if train_labels[train_idx] == test_label {
                    1.0
                } else {
                    0.0
                }
            };

            // Compute Shapley values using the recursive formula
            // phi_pi(n) = s(z_n, y*) / n
            // phi_pi(j) = phi_pi(j+1) + [s(z_j, y*) - s(z_{j+1}, y*)] / max(j, k)
            let mut phi = vec![0.0_f64; n_train];

            // Last element (furthest)
            let last_idx = dists[n_train - 1].0;
            phi[n_train - 1] = s(last_idx) / n_train as f64;

            // Work backwards
            for j in (0..n_train - 1).rev() {
                let j_idx = dists[j].0;
                let j1_idx = dists[j + 1].0;
                let denom = (j + 1).max(k) as f64;
                phi[j] = phi[j + 1] + (s(j_idx) - s(j1_idx)) / denom;
            }

            // Accumulate: map back from sorted order to original indices
            for (sorted_pos, &(orig_idx, _)) in dists.iter().enumerate() {
                total_shapley[orig_idx] += phi[sorted_pos];
            }
        }

        // Average over test points
        for val in &mut total_shapley {
            *val /= n_test as f64;
        }

        Ok(total_shapley)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7 — LavaValuation
// ─────────────────────────────────────────────────────────────────────────────

/// LAVA data valuation (Just et al. 2023).
///
/// Uses optimal transport (Wasserstein distance) to estimate the value
/// of each training data point relative to the validation distribution.
#[derive(Debug, Clone)]
pub struct LavaValuation {
    /// Regularization parameter for Sinkhorn divergence.
    pub sinkhorn_reg: f64,
    /// Number of Sinkhorn iterations.
    pub sinkhorn_iters: usize,
    /// Whether to use dual potentials for valuation.
    pub use_dual: bool,
}

impl LavaValuation {
    /// Create a new LAVA valuation estimator.
    ///
    /// # Arguments
    /// * `sinkhorn_reg` - Entropic regularization epsilon.
    /// * `sinkhorn_iters` - Maximum Sinkhorn iterations.
    pub fn new(sinkhorn_reg: f64, sinkhorn_iters: usize) -> Result<Self, TensorError> {
        if sinkhorn_reg <= 0.0 {
            return Err(TensorError::compute_error_simple(
                "LavaValuation: sinkhorn_reg must be > 0".to_string(),
            ));
        }
        if sinkhorn_iters == 0 {
            return Err(TensorError::compute_error_simple(
                "LavaValuation: sinkhorn_iters must be > 0".to_string(),
            ));
        }
        Ok(Self {
            sinkhorn_reg,
            sinkhorn_iters,
            use_dual: true,
        })
    }

    /// Compute the pairwise cost matrix (squared Euclidean distances).
    fn cost_matrix(&self, train_features: &[Vec<f64>], val_features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = train_features.len();
        let m = val_features.len();
        let mut cost = vec![vec![0.0_f64; m]; n];
        for i in 0..n {
            for j in 0..m {
                let dist = cl_euclidean_distance(&train_features[i], &val_features[j]);
                cost[i][j] = dist * dist; // Squared Euclidean
            }
        }
        cost
    }

    /// Run log-domain Sinkhorn iterations.
    ///
    /// Returns (f, g) dual potentials.
    fn sinkhorn_log_domain(&self, cost: &[Vec<f64>], n: usize, m: usize) -> (Vec<f64>, Vec<f64>) {
        let eps = self.sinkhorn_reg;
        let mut f = vec![0.0_f64; n];
        let mut g = vec![0.0_f64; m];

        for _iter in 0..self.sinkhorn_iters {
            // Update f: f_i = -eps * log( sum_j exp((g_j - C_{ij}) / eps) / m )
            for i in 0..n {
                let max_val = (0..m)
                    .map(|j| (g[j] - cost[i][j]) / eps)
                    .fold(f64::NEG_INFINITY, f64::max);
                let log_sum: f64 = (0..m)
                    .map(|j| ((g[j] - cost[i][j]) / eps - max_val).exp())
                    .sum::<f64>();
                f[i] = -eps * (max_val + log_sum.ln() - (m as f64).ln());
            }

            // Update g: g_j = -eps * log( sum_i exp((f_i - C_{ij}) / eps) / n )
            for j in 0..m {
                let max_val = (0..n)
                    .map(|i| (f[i] - cost[i][j]) / eps)
                    .fold(f64::NEG_INFINITY, f64::max);
                let log_sum: f64 = (0..n)
                    .map(|i| ((f[i] - cost[i][j]) / eps - max_val).exp())
                    .sum::<f64>();
                g[j] = -eps * (max_val + log_sum.ln() - (n as f64).ln());
            }
        }

        (f, g)
    }

    /// Compute LAVA valuation scores for each training sample.
    ///
    /// Higher scores indicate training samples closer to the validation distribution,
    /// hence more valuable for generalization.
    ///
    /// # Arguments
    /// * `train_features` - Training feature vectors [n_train, dim].
    /// * `val_features` - Validation feature vectors [n_val, dim].
    ///
    /// Returns per-training-sample valuation scores.
    pub fn compute(
        &self,
        train_features: &[Vec<f64>],
        val_features: &[Vec<f64>],
    ) -> Result<Vec<f64>, TensorError> {
        if train_features.is_empty() || val_features.is_empty() {
            return Err(TensorError::compute_error_simple(
                "LavaValuation::compute: empty features".to_string(),
            ));
        }

        let n = train_features.len();
        let m = val_features.len();
        let cost = self.cost_matrix(train_features, val_features);

        if self.use_dual {
            // Use dual potentials: f_i measures how "well-matched" training point i is
            let (f, _g) = self.sinkhorn_log_domain(&cost, n, m);
            // Higher f_i = lower transport cost from point i = more valuable
            // Negate and normalize so that higher value = more useful
            let neg_f: Vec<f64> = f.iter().map(|&fi| -fi).collect();
            Ok(cl_normalize_min_max(&neg_f))
        } else {
            // Simple nearest-neighbor approach: closer to val = higher value
            let mut scores = Vec::with_capacity(n);
            for i in 0..n {
                let min_dist = cost[i].iter().copied().fold(f64::INFINITY, f64::min);
                scores.push(-min_dist); // Negate: smaller distance = higher value
            }
            Ok(cl_normalize_min_max(&scores))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 8 — AutoCurriculum
// ─────────────────────────────────────────────────────────────────────────────

/// Single arm (difficulty bin) statistics for UCB1.
#[derive(Debug, Clone)]
pub struct ClBanditArm {
    /// Total reward accumulated.
    pub total_reward: f64,
    /// Number of times this arm was pulled.
    pub pull_count: usize,
}

/// Bandit-based automatic curriculum selection using UCB1.
///
/// Partitions the data into difficulty bins and uses UCB1 to decide
/// which bin to sample from at each step.
#[derive(Debug, Clone)]
pub struct AutoCurriculum {
    /// Number of difficulty bins.
    pub n_bins: usize,
    /// Per-bin bandit statistics.
    pub arms: Vec<ClBanditArm>,
    /// Total number of pulls across all arms.
    pub total_pulls: usize,
    /// UCB exploration parameter (c in UCB1 formula).
    pub exploration_c: f64,
    /// Bin boundaries: sample i goes to bin b if difficulties\[i\] in [boundary\[b\], boundary[b+1]).
    pub bin_boundaries: Vec<f64>,
    /// History of selected bins.
    pub selection_history: Vec<usize>,
    /// History of rewards.
    pub reward_history: Vec<f64>,
}

impl AutoCurriculum {
    /// Create a new auto-curriculum with equal-width bins.
    ///
    /// # Arguments
    /// * `n_bins` - Number of difficulty bins.
    /// * `exploration_c` - UCB1 exploration parameter.
    pub fn new(n_bins: usize, exploration_c: f64) -> Result<Self, TensorError> {
        if n_bins == 0 {
            return Err(TensorError::compute_error_simple(
                "AutoCurriculum: n_bins must be > 0".to_string(),
            ));
        }

        let arms: Vec<ClBanditArm> = (0..n_bins)
            .map(|_| ClBanditArm {
                total_reward: 0.0,
                pull_count: 0,
            })
            .collect();

        // Equal-width bins in [0, 1]
        let step = 1.0 / n_bins as f64;
        let mut bin_boundaries = Vec::with_capacity(n_bins + 1);
        for i in 0..=n_bins {
            bin_boundaries.push(i as f64 * step);
        }

        Ok(Self {
            n_bins,
            arms,
            total_pulls: 0,
            exploration_c: exploration_c.max(0.0),
            bin_boundaries,
            selection_history: Vec::new(),
            reward_history: Vec::new(),
        })
    }

    /// Select which difficulty bin to sample from using UCB1.
    ///
    /// UCB1: argmax_a [ mean_reward(a) + c * sqrt(ln(total) / n_a) ]
    ///
    /// Returns the selected bin index.
    pub fn select_bin(&self) -> Result<usize, TensorError> {
        if self.n_bins == 0 {
            return Err(TensorError::compute_error_simple(
                "AutoCurriculum: no bins".to_string(),
            ));
        }

        // First, pull each arm at least once
        for (i, arm) in self.arms.iter().enumerate() {
            if arm.pull_count == 0 {
                return Ok(i);
            }
        }

        // UCB1 selection
        let ln_total = (self.total_pulls as f64).ln();
        let mut best_bin = 0;
        let mut best_ucb = f64::NEG_INFINITY;

        for (i, arm) in self.arms.iter().enumerate() {
            let mean_reward = arm.total_reward / arm.pull_count as f64;
            let exploration = self.exploration_c * (ln_total / arm.pull_count as f64).sqrt();
            let ucb = mean_reward + exploration;
            if ucb > best_ucb {
                best_ucb = ucb;
                best_bin = i;
            }
        }

        Ok(best_bin)
    }

    /// Update the reward for a bin.
    ///
    /// # Arguments
    /// * `bin_idx` - Which bin was selected.
    /// * `reward` - Observed reward (e.g., validation improvement).
    pub fn update_reward(&mut self, bin_idx: usize, reward: f64) -> Result<(), TensorError> {
        if bin_idx >= self.n_bins {
            return Err(TensorError::compute_error_simple(format!(
                "AutoCurriculum: bin_idx {} >= n_bins {}",
                bin_idx, self.n_bins
            )));
        }

        self.arms[bin_idx].total_reward += reward;
        self.arms[bin_idx].pull_count += 1;
        self.total_pulls += 1;
        self.selection_history.push(bin_idx);
        self.reward_history.push(reward);

        Ok(())
    }

    /// Get sample indices belonging to a specific difficulty bin.
    ///
    /// # Arguments
    /// * `difficulties` - Normalized difficulty scores in [0, 1].
    /// * `bin_idx` - Which bin to query.
    ///
    /// Returns indices of samples in the specified bin.
    pub fn get_bin_samples(
        &self,
        difficulties: &[f64],
        bin_idx: usize,
    ) -> Result<Vec<usize>, TensorError> {
        if bin_idx >= self.n_bins {
            return Err(TensorError::compute_error_simple(format!(
                "AutoCurriculum: bin_idx {} >= n_bins {}",
                bin_idx, self.n_bins
            )));
        }

        let lo = self.bin_boundaries[bin_idx];
        let hi = self.bin_boundaries[bin_idx + 1];

        let indices: Vec<usize> = difficulties
            .iter()
            .enumerate()
            .filter(|(_, &d)| d >= lo && (d < hi || (bin_idx + 1 == self.n_bins && d <= hi)))
            .map(|(i, _)| i)
            .collect();

        Ok(indices)
    }

    /// Get the average reward per bin.
    pub fn mean_rewards(&self) -> Vec<f64> {
        self.arms
            .iter()
            .map(|arm| {
                if arm.pull_count > 0 {
                    arm.total_reward / arm.pull_count as f64
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Reset all bandit statistics.
    pub fn reset(&mut self) {
        for arm in &mut self.arms {
            arm.total_reward = 0.0;
            arm.pull_count = 0;
        }
        self.total_pulls = 0;
        self.selection_history.clear();
        self.reward_history.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 9 — CurriculumTrainer
// ─────────────────────────────────────────────────────────────────────────────

/// Training mode for curriculum learning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClTrainingMode {
    /// Standard curriculum (easy-first).
    Curriculum,
    /// Anti-curriculum (hard-first).
    AntiCurriculum,
    /// Self-paced learning.
    SelfPaced,
    /// Competence-based.
    CompetenceBased,
    /// Random baseline (no curriculum).
    Random,
    /// Bandit-based automatic curriculum.
    AutoBandit,
}

/// Result of a single training epoch.
#[derive(Debug, Clone)]
pub struct ClEpochResult {
    /// Average loss for this epoch.
    pub avg_loss: f64,
    /// Number of samples used in this epoch.
    pub n_samples_used: usize,
    /// Total number of available samples.
    pub n_samples_total: usize,
    /// Fraction of dataset used.
    pub utilization_rate: f64,
    /// Current competence level (for scheduler-based modes).
    pub competence: f64,
    /// Per-sample selection counts.
    pub selection_counts: Vec<usize>,
}

/// Full curriculum training loop integrating difficulty scoring,
/// scheduling, and training steps.
#[derive(Debug)]
pub struct CurriculumTrainer {
    /// Training mode.
    pub mode: ClTrainingMode,
    /// Batch size.
    pub batch_size: usize,
    /// Total epochs.
    pub total_epochs: usize,
    /// Current epoch.
    pub current_epoch: usize,
    /// Per-sample selection counts across all epochs.
    pub selection_counts: Vec<usize>,
    /// Loss history per epoch.
    pub loss_history: Vec<f64>,
    /// Utilization history per epoch.
    pub utilization_history: Vec<f64>,
    /// RNG for random mode.
    rng: StdRng,
}

impl Clone for CurriculumTrainer {
    fn clone(&self) -> Self {
        Self {
            mode: self.mode.clone(),
            batch_size: self.batch_size,
            total_epochs: self.total_epochs,
            current_epoch: self.current_epoch,
            selection_counts: self.selection_counts.clone(),
            loss_history: self.loss_history.clone(),
            utilization_history: self.utilization_history.clone(),
            rng: StdRng::seed_from_u64(0xC0EE_7EA1),
        }
    }
}

impl CurriculumTrainer {
    /// Create a new curriculum trainer.
    ///
    /// # Arguments
    /// * `mode` - Training mode.
    /// * `batch_size` - Samples per batch.
    /// * `total_epochs` - Total training epochs.
    /// * `n_samples` - Total number of training samples.
    /// * `seed` - Random seed.
    pub fn new(
        mode: ClTrainingMode,
        batch_size: usize,
        total_epochs: usize,
        n_samples: usize,
        seed: u64,
    ) -> Result<Self, TensorError> {
        if batch_size == 0 {
            return Err(TensorError::compute_error_simple(
                "CurriculumTrainer: batch_size must be > 0".to_string(),
            ));
        }
        if total_epochs == 0 {
            return Err(TensorError::compute_error_simple(
                "CurriculumTrainer: total_epochs must be > 0".to_string(),
            ));
        }
        if n_samples == 0 {
            return Err(TensorError::compute_error_simple(
                "CurriculumTrainer: n_samples must be > 0".to_string(),
            ));
        }
        Ok(Self {
            mode,
            batch_size,
            total_epochs,
            current_epoch: 0,
            selection_counts: vec![0; n_samples],
            loss_history: Vec::new(),
            utilization_history: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// Train one epoch with a curriculum-aware sample selection.
    ///
    /// # Arguments
    /// * `difficulties` - Per-sample normalized difficulty scores.
    /// * `train_fn` - Function that trains on selected indices and returns loss.
    /// * `scheduler` - Optional curriculum scheduler (for Curriculum/AntiCurriculum modes).
    /// * `spl` - Optional self-paced learner (for SelfPaced mode).
    /// * `competence_learner` - Optional competence learner (for CompetenceBased mode).
    /// * `auto_curriculum` - Optional auto-curriculum (for AutoBandit mode).
    ///
    /// Returns epoch result.
    pub fn train_epoch<F>(
        &mut self,
        difficulties: &[f64],
        losses_per_sample: &[f64],
        train_fn: &F,
        mut scheduler: Option<&mut ClCurriculumScheduler>,
        spl: Option<&SelfPacedLearning>,
        competence_learner: Option<&CompetenceLearning>,
        auto_curriculum: Option<&mut AutoCurriculum>,
    ) -> Result<ClEpochResult, TensorError>
    where
        F: Fn(&[usize]) -> Result<f64, TensorError>,
    {
        let n_samples = difficulties.len();
        if n_samples == 0 {
            return Err(TensorError::compute_error_simple(
                "CurriculumTrainer::train_epoch: empty dataset".to_string(),
            ));
        }

        // Select samples based on mode
        let selected_indices = match self.mode {
            ClTrainingMode::Curriculum | ClTrainingMode::AntiCurriculum => {
                if let Some(ref mut sched) = scheduler {
                    let indices = sched.select_batch(difficulties, self.batch_size)?;
                    sched.step();
                    indices
                } else {
                    // Fallback: sort by difficulty
                    let sorted = if self.mode == ClTrainingMode::Curriculum {
                        cl_argsort_asc(difficulties)
                    } else {
                        cl_argsort_desc(difficulties)
                    };
                    let take = self.batch_size.min(sorted.len());
                    sorted[..take].to_vec()
                }
            }
            ClTrainingMode::SelfPaced => {
                if let Some(spl_ref) = spl {
                    let weights = spl_ref.compute_weights(losses_per_sample)?;
                    let mut weighted_indices: Vec<(usize, f64)> = weights
                        .iter()
                        .enumerate()
                        .filter(|(_, &w)| w > 1e-10)
                        .map(|(i, &w)| (i, w))
                        .collect();
                    weighted_indices
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let take = self.batch_size.min(weighted_indices.len());
                    weighted_indices[..take].iter().map(|(i, _)| *i).collect()
                } else {
                    // Fallback: random
                    let mut indices: Vec<usize> = (0..n_samples).collect();
                    cl_shuffle(&mut indices, &mut self.rng);
                    indices[..self.batch_size.min(n_samples)].to_vec()
                }
            }
            ClTrainingMode::CompetenceBased => {
                if let Some(cl) = competence_learner {
                    let eligible = cl.filter_by_competence(difficulties)?;
                    let mut pool = eligible;
                    cl_shuffle(&mut pool, &mut self.rng);
                    let take = self.batch_size.min(pool.len());
                    pool[..take].to_vec()
                } else {
                    let mut indices: Vec<usize> = (0..n_samples).collect();
                    cl_shuffle(&mut indices, &mut self.rng);
                    indices[..self.batch_size.min(n_samples)].to_vec()
                }
            }
            ClTrainingMode::Random => {
                let mut indices: Vec<usize> = (0..n_samples).collect();
                cl_shuffle(&mut indices, &mut self.rng);
                indices[..self.batch_size.min(n_samples)].to_vec()
            }
            ClTrainingMode::AutoBandit => {
                if let Some(ac) = auto_curriculum {
                    let bin_idx = ac.select_bin()?;
                    let bin_samples = ac.get_bin_samples(difficulties, bin_idx)?;
                    if bin_samples.is_empty() {
                        // Fallback: random
                        let mut indices: Vec<usize> = (0..n_samples).collect();
                        cl_shuffle(&mut indices, &mut self.rng);
                        indices[..self.batch_size.min(n_samples)].to_vec()
                    } else {
                        let mut pool = bin_samples;
                        cl_shuffle(&mut pool, &mut self.rng);
                        let take = self.batch_size.min(pool.len());
                        pool[..take].to_vec()
                    }
                } else {
                    let mut indices: Vec<usize> = (0..n_samples).collect();
                    cl_shuffle(&mut indices, &mut self.rng);
                    indices[..self.batch_size.min(n_samples)].to_vec()
                }
            }
        };

        // Update selection counts
        for &idx in &selected_indices {
            if idx < self.selection_counts.len() {
                self.selection_counts[idx] += 1;
            }
        }

        // Train on selected samples
        let loss = train_fn(&selected_indices)?;

        let utilization = selected_indices.len() as f64 / n_samples as f64;
        self.loss_history.push(loss);
        self.utilization_history.push(utilization);
        self.current_epoch += 1;

        let competence = if let Some(ref sched) = scheduler {
            sched.competence()
        } else {
            1.0
        };

        Ok(ClEpochResult {
            avg_loss: loss,
            n_samples_used: selected_indices.len(),
            n_samples_total: n_samples,
            utilization_rate: utilization,
            competence,
            selection_counts: self.selection_counts.clone(),
        })
    }

    /// Reset the trainer state.
    pub fn reset(&mut self) {
        self.current_epoch = 0;
        self.selection_counts.fill(0);
        self.loss_history.clear();
        self.utilization_history.clear();
    }
}
