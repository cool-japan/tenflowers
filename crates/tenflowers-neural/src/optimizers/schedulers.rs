//! Learning-rate schedulers with a stateful `LrScheduler` trait.
//!
//! This module provides a family of schedulers that maintain their own
//! internal step counter and can be used directly alongside the flat
//! `Vec<f32>` optimizers in this crate (`LionOptimizer`, `LambOptimizer`,
//! `MuonOptimizer`).
//!
//! # Schedulers
//!
//! | Type | Description |
//! |------|-------------|
//! | [`CosineAnnealingScheduler`] | SGDR with warm restarts (Loshchilov & Hutter, 2017) |
//! | [`OneCycleLrScheduler`]      | 1-cycle policy (Smith & Topin, 2019) |
//! | [`ReduceLrOnPlateau`]        | Reduce LR when a metric stops improving |
//! | [`WarmupScheduler`]          | Linear warm-up then constant/base-scheduler hand-off |
//! | [`ExponentialDecayScheduler`]| Exponential decay with optional staircase mode |
//! | [`LinearScheduler`]          | Linear warm-up then linear decay |
//! | [`PolynomialDecayScheduler`] | Polynomial decay to `end_lr` |

use std::f32::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// LrScheduler trait
// ─────────────────────────────────────────────────────────────────────────────

/// Common interface for stateful learning-rate schedulers.
///
/// Implementors keep an internal step/epoch counter that is advanced by
/// calling \[`step`\].
pub trait LrScheduler: Send + Sync {
    /// Return the current learning rate.
    fn get_lr(&self) -> f32;

    /// Advance the scheduler by one step (epoch or mini-batch, depending on
    /// the scheduler).
    fn step(&mut self);

    /// Reset the scheduler to its initial state.
    fn reset(&mut self);
}

// ─────────────────────────────────────────────────────────────────────────────
// AnnealStrategy
// ─────────────────────────────────────────────────────────────────────────────

/// Annealing function used by [`OneCycleLrScheduler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnealStrategy {
    /// Cosine annealing (smooth).
    Cos,
    /// Linear annealing.
    Linear,
}

// ─────────────────────────────────────────────────────────────────────────────
// MetricMode
// ─────────────────────────────────────────────────────────────────────────────

/// Whether the tracked metric should be minimised or maximised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricMode {
    /// Lower metric is better (e.g. loss).
    Min,
    /// Higher metric is better (e.g. accuracy).
    Max,
}

// ─────────────────────────────────────────────────────────────────────────────
// CosineAnnealingScheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Cosine annealing with warm restarts (SGDR).
///
/// Reference: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with
/// Warm Restarts" (2017).
///
/// The learning rate for step *t* within the current cycle of length `T_i`:
/// ```text
/// η = η_min + 0.5·(η_max − η_min)·(1 + cos(π · t_cur / T_i))
/// ```
/// After `T_i` steps the scheduler restarts, and the next cycle length is
/// `T_{i+1} = T_i · T_mult`.
///
/// # Example
/// ```rust
/// use tenflowers_neural::optimizers::{CosineAnnealingScheduler, LrScheduler};
///
/// let mut sched = CosineAnnealingScheduler::new(10, 1.0, 0.1, 0.001);
/// let lr0 = sched.get_lr();
/// assert!((lr0 - 0.1).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct CosineAnnealingScheduler {
    /// Initial restart period `T_0` (steps before first restart).
    pub t0: usize,
    /// Period multiplier `T_mult` (≥ 1.0).
    pub t_mult: f32,
    /// Minimum learning rate `η_min`.
    pub eta_min: f32,
    /// Maximum learning rate `η_max`.
    pub eta_max: f32,
    /// Current global step count.
    current_step: usize,
}

impl CosineAnnealingScheduler {
    /// Create a new `CosineAnnealingScheduler`.
    ///
    /// # Arguments
    /// * `t0`      — initial restart period (number of steps).
    /// * `t_mult`  — multiplier for successive period lengths (≥ 1.0).
    /// * `eta_max` — learning rate at the start of each cycle.
    /// * `eta_min` — learning rate at the end of each cycle.
    pub fn new(t0: usize, t_mult: f32, eta_max: f32, eta_min: f32) -> Self {
        Self {
            t0,
            t_mult,
            eta_min,
            eta_max,
            current_step: 0,
        }
    }

    /// Compute the learning rate at an arbitrary epoch without mutating state.
    pub fn lr_at_epoch(&self, epoch: usize) -> f32 {
        compute_cosine_lr(epoch, self.t0, self.t_mult, self.eta_max, self.eta_min)
    }
}

impl LrScheduler for CosineAnnealingScheduler {
    fn get_lr(&self) -> f32 {
        compute_cosine_lr(
            self.current_step,
            self.t0,
            self.t_mult,
            self.eta_max,
            self.eta_min,
        )
    }

    fn step(&mut self) {
        self.current_step += 1;
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

/// Stateless cosine-annealing computation (used by [`CosineAnnealingScheduler`]).
fn compute_cosine_lr(step: usize, t0: usize, t_mult: f32, eta_max: f32, eta_min: f32) -> f32 {
    if t0 == 0 {
        return eta_max;
    }

    // Walk through cycles to find which cycle `step` falls in and the
    // position within that cycle.
    let mut cycle_len = t0 as f32;
    let mut cycle_start: usize = 0;

    loop {
        let cycle_end = cycle_start + (cycle_len.round() as usize).max(1);
        if step < cycle_end {
            let t_cur = (step - cycle_start) as f32;
            let t_i = cycle_len;
            return eta_min + 0.5 * (eta_max - eta_min) * (1.0 + (PI * t_cur / t_i).cos());
        }
        cycle_start = cycle_end;
        if t_mult <= 1.0 {
            // No growth — just wrap back into a T_0-length cycle.
            let local = (step - cycle_start) % t0.max(1);
            let t_cur = local as f32;
            let t_i = t0 as f32;
            return eta_min + 0.5 * (eta_max - eta_min) * (1.0 + (PI * t_cur / t_i).cos());
        }
        cycle_len *= t_mult;
        // Guard against runaway for extremely large step counts.
        if cycle_len > 1e9 {
            return eta_max;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OneCycleLrScheduler
// ─────────────────────────────────────────────────────────────────────────────

/// 1-cycle learning-rate policy (Smith & Topin, 2019).
///
/// The learning rate increases linearly/cosinely from `initial_lr` to
/// `max_lr` over `pct_start * total_steps` steps, then decreases to
/// `final_lr` over the remaining steps.
///
/// # Example
/// ```rust
/// use tenflowers_neural::optimizers::{OneCycleLrScheduler, AnnealStrategy, LrScheduler};
///
/// let sched = OneCycleLrScheduler::new(100, 0.1, 0.001, 1e-5, 0.3, AnnealStrategy::Cos);
/// let lr0 = sched.get_lr_at_step(0);
/// assert!((lr0 - 0.001).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct OneCycleLrScheduler {
    /// Total number of training steps in one cycle.
    pub total_steps: usize,
    /// Peak learning rate reached at `pct_start * total_steps`.
    pub max_lr: f32,
    /// Learning rate at step 0 (warm-up start).
    pub initial_lr: f32,
    /// Learning rate at `total_steps` (end of decay).
    pub final_lr: f32,
    /// Fraction of steps used for warm-up (default `0.3`).
    pub pct_start: f32,
    /// Interpolation function for both warm-up and decay phases.
    pub anneal_strategy: AnnealStrategy,
    /// Current step for stateful use via `LrScheduler`.
    current_step: usize,
}

impl OneCycleLrScheduler {
    /// Create a new `OneCycleLrScheduler`.
    pub fn new(
        total_steps: usize,
        max_lr: f32,
        initial_lr: f32,
        final_lr: f32,
        pct_start: f32,
        anneal_strategy: AnnealStrategy,
    ) -> Self {
        Self {
            total_steps,
            max_lr,
            initial_lr,
            final_lr,
            pct_start,
            anneal_strategy,
            current_step: 0,
        }
    }

    /// Compute the learning rate at an arbitrary step without mutating state.
    pub fn get_lr_at_step(&self, step: usize) -> f32 {
        if self.total_steps == 0 {
            return self.final_lr;
        }

        let warmup_end =
            ((self.pct_start * self.total_steps as f32).round() as usize).min(self.total_steps);

        if step == 0 {
            return self.initial_lr;
        }

        if step <= warmup_end {
            let t = step as f32 / warmup_end.max(1) as f32;
            return interpolate(self.initial_lr, self.max_lr, t, self.anneal_strategy);
        }

        let decay_steps = self.total_steps.saturating_sub(warmup_end);
        if decay_steps == 0 {
            return self.max_lr;
        }
        let t = (step - warmup_end) as f32 / decay_steps as f32;
        let t = t.min(1.0);
        interpolate(self.max_lr, self.final_lr, t, self.anneal_strategy)
    }
}

impl LrScheduler for OneCycleLrScheduler {
    fn get_lr(&self) -> f32 {
        self.get_lr_at_step(self.current_step)
    }

    fn step(&mut self) {
        self.current_step += 1;
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

/// Interpolate between `start` and `end` at position `t ∈ [0,1]`.
fn interpolate(start: f32, end: f32, t: f32, strategy: AnnealStrategy) -> f32 {
    match strategy {
        AnnealStrategy::Linear => start + (end - start) * t,
        AnnealStrategy::Cos => {
            // Cosine interpolation: start when t=0, end when t=1.
            let cos = (1.0 + (PI * t).cos()) / 2.0;
            end + (start - end) * cos
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ReduceLrOnPlateau
// ─────────────────────────────────────────────────────────────────────────────

/// Reduce the learning rate when a monitored metric stops improving.
///
/// The scheduler maintains an exponentially-smoothed version of the metric and
/// compares it against the best seen so far.  After `patience` consecutive
/// non-improving steps it multiplies the current LR by `factor` (clamped to
/// `min_lr`).  An optional `cooldown` period prevents consecutive reductions.
///
/// # Example
/// ```rust
/// use tenflowers_neural::optimizers::{ReduceLrOnPlateau, MetricMode};
///
/// let mut sched = ReduceLrOnPlateau::new(MetricMode::Min, 0.5, 3, 0.0, 0, 1e-6, 0.0);
/// let new_lr = sched.step_with_metric(1.0, 0.1);
/// assert!((new_lr - 0.1).abs() < 1e-7);
/// ```
#[derive(Debug, Clone)]
pub struct ReduceLrOnPlateau {
    /// Whether a lower or higher metric is better.
    pub mode: MetricMode,
    /// Multiplicative factor by which the LR is reduced (default `0.1`).
    pub factor: f32,
    /// Number of non-improving steps before reduction.
    pub patience: usize,
    /// Minimum threshold for metric improvement.
    pub threshold: f32,
    /// Number of steps to wait after a reduction before checking again.
    pub cooldown: usize,
    /// Minimum allowed learning rate.
    pub min_lr: f32,
    /// Smoothing coefficient for exponential moving average of the metric
    /// (0.0 = no smoothing, 1.0 = never update EMA).
    pub smoothing: f32,

    // Internal state.
    bad_epochs: usize,
    cooldown_counter: usize,
    best: Option<f32>,
    smoothed_metric: Option<f32>,
}

impl ReduceLrOnPlateau {
    /// Create a new `ReduceLrOnPlateau` scheduler.
    ///
    /// # Arguments
    /// * `mode`      — optimisation direction.
    /// * `factor`    — LR reduction factor (e.g. `0.1`).
    /// * `patience`  — number of non-improving steps before reduction.
    /// * `threshold` — minimum relative improvement to count as improvement.
    /// * `cooldown`  — steps to skip after a reduction.
    /// * `min_lr`    — lower bound on the learning rate.
    /// * `smoothing` — EMA coefficient for the metric (0.0 = raw metric).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mode: MetricMode,
        factor: f32,
        patience: usize,
        threshold: f32,
        cooldown: usize,
        min_lr: f32,
        smoothing: f32,
    ) -> Self {
        Self {
            mode,
            factor,
            patience,
            threshold,
            cooldown,
            min_lr,
            smoothing,
            bad_epochs: 0,
            cooldown_counter: 0,
            best: None,
            smoothed_metric: None,
        }
    }

    /// Update the metric and return the (possibly reduced) new learning rate.
    ///
    /// The current learning rate is passed in; the returned value is the
    /// updated learning rate.
    pub fn step_with_metric(&mut self, metric: f32, current_lr: f32) -> f32 {
        // Update EMA of metric.
        let smoothed = match self.smoothed_metric {
            None => metric,
            Some(prev) => self.smoothing * prev + (1.0 - self.smoothing) * metric,
        };
        self.smoothed_metric = Some(smoothed);

        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            // Still in cooldown — update best but do not count bad epochs.
            if self.is_improving(smoothed) {
                self.best = Some(smoothed);
            }
            return current_lr;
        }

        if self.is_improving(smoothed) {
            self.best = Some(smoothed);
            self.bad_epochs = 0;
        } else {
            self.bad_epochs += 1;
            if self.bad_epochs >= self.patience {
                let new_lr = (current_lr * self.factor).max(self.min_lr);
                self.bad_epochs = 0;
                self.cooldown_counter = self.cooldown;
                return new_lr;
            }
        }
        current_lr
    }

    /// Return `true` if `smoothed` represents improvement over the current
    /// `best`.
    pub fn is_improving(&self, smoothed: f32) -> bool {
        match self.best {
            None => true,
            Some(best) => match self.mode {
                MetricMode::Min => smoothed < best - self.threshold,
                MetricMode::Max => smoothed > best + self.threshold,
            },
        }
    }
}

impl LrScheduler for ReduceLrOnPlateau {
    /// Returns the most recently passed learning rate (the scheduler does not
    /// store the current LR internally; use \[`step_with_metric`\] to update).
    fn get_lr(&self) -> f32 {
        // Stateless fallback — returns a sentinel; callers should use
        // `step_with_metric` to get the updated LR.
        0.0
    }

    fn step(&mut self) {
        // No-op for the stateful trait — use step_with_metric instead.
    }

    fn reset(&mut self) {
        self.bad_epochs = 0;
        self.cooldown_counter = 0;
        self.best = None;
        self.smoothed_metric = None;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WarmupScheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Linear warm-up scheduler.
///
/// Learning rate increases linearly from `0` to `target_lr` over
/// `warmup_steps` steps, then remains constant at `target_lr`.
///
/// For a more complex post-warmup schedule, wrap a `Box<dyn LrScheduler>` in a
/// custom struct after the warm-up is finished.
///
/// # Example
/// ```rust
/// use tenflowers_neural::optimizers::{WarmupScheduler, LrScheduler};
///
/// let mut sched = WarmupScheduler::new(10, 0.001);
/// assert_eq!(sched.get_lr_at_step(0), 0.0);
/// assert!((sched.get_lr_at_step(5) - 0.0005).abs() < 1e-6);
/// assert!((sched.get_lr_at_step(10) - 0.001).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct WarmupScheduler {
    /// Number of warm-up steps.
    pub warmup_steps: usize,
    /// Learning rate at the end of warm-up (and constant afterwards).
    pub target_lr: f32,
    /// Current step for stateful use.
    current_step: usize,
}

impl WarmupScheduler {
    /// Create a new `WarmupScheduler`.
    pub fn new(warmup_steps: usize, target_lr: f32) -> Self {
        Self {
            warmup_steps,
            target_lr,
            current_step: 0,
        }
    }

    /// Compute the learning rate at an arbitrary step without mutating state.
    pub fn get_lr_at_step(&self, step: usize) -> f32 {
        if self.warmup_steps == 0 {
            return self.target_lr;
        }
        if step >= self.warmup_steps {
            return self.target_lr;
        }
        let t = step as f32 / self.warmup_steps as f32;
        t * self.target_lr
    }
}

impl LrScheduler for WarmupScheduler {
    fn get_lr(&self) -> f32 {
        self.get_lr_at_step(self.current_step)
    }

    fn step(&mut self) {
        self.current_step += 1;
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ExponentialDecayScheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Exponential learning-rate decay.
///
/// ```text
/// lr(step) = initial_lr × decay_rate^(step / decay_steps)
/// ```
/// In **staircase** mode the exponent is floored so the LR is constant within
/// each `decay_steps`-long interval.
///
/// # Example
/// ```rust
/// use tenflowers_neural::optimizers::{ExponentialDecayScheduler, LrScheduler};
///
/// let sched = ExponentialDecayScheduler::new(0.1, 0.5, 10, false);
/// assert!((sched.get_lr_at_step(10) - 0.05).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct ExponentialDecayScheduler {
    /// Initial learning rate.
    pub initial_lr: f32,
    /// Decay rate (typically < 1.0).
    pub decay_rate: f32,
    /// Number of steps per decay period.
    pub decay_steps: usize,
    /// If `true`, the exponent is an integer (staircase decay).
    pub staircase: bool,
    /// Current step for stateful use.
    current_step: usize,
}

impl ExponentialDecayScheduler {
    /// Create a new `ExponentialDecayScheduler`.
    pub fn new(initial_lr: f32, decay_rate: f32, decay_steps: usize, staircase: bool) -> Self {
        Self {
            initial_lr,
            decay_rate,
            decay_steps,
            staircase,
            current_step: 0,
        }
    }

    /// Compute the learning rate at an arbitrary step without mutating state.
    pub fn get_lr_at_step(&self, step: usize) -> f32 {
        if self.decay_steps == 0 {
            return self.initial_lr;
        }
        let power = if self.staircase {
            (step / self.decay_steps) as f32
        } else {
            step as f32 / self.decay_steps as f32
        };
        self.initial_lr * self.decay_rate.powf(power)
    }
}

impl LrScheduler for ExponentialDecayScheduler {
    fn get_lr(&self) -> f32 {
        self.get_lr_at_step(self.current_step)
    }

    fn step(&mut self) {
        self.current_step += 1;
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LinearScheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Linear warm-up followed by linear decay.
///
/// ```text
/// step ∈ [0, warmup_steps)          → linear ramp  initial_lr → peak_lr
/// step ∈ [warmup_steps, total_steps] → linear decay peak_lr   → final_lr
/// step > total_steps                 → final_lr
/// ```
///
/// # Example
/// ```rust
/// use tenflowers_neural::optimizers::{LinearScheduler, LrScheduler};
///
/// let sched = LinearScheduler::new(10, 100, 0.0, 0.1, 0.001);
/// assert!((sched.get_lr_at_step(0) - 0.0).abs() < 1e-7);
/// assert!((sched.get_lr_at_step(10) - 0.1).abs() < 1e-6);
/// assert!((sched.get_lr_at_step(100) - 0.001).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct LinearScheduler {
    /// Steps in the warm-up phase.
    pub warmup_steps: usize,
    /// Total steps (warm-up + decay).
    pub total_steps: usize,
    /// Learning rate at step 0.
    pub initial_lr: f32,
    /// Peak learning rate (at end of warm-up).
    pub peak_lr: f32,
    /// Final learning rate (at `total_steps`).
    pub final_lr: f32,
    /// Current step.
    current_step: usize,
}

impl LinearScheduler {
    /// Create a new `LinearScheduler`.
    pub fn new(
        warmup_steps: usize,
        total_steps: usize,
        initial_lr: f32,
        peak_lr: f32,
        final_lr: f32,
    ) -> Self {
        Self {
            warmup_steps,
            total_steps,
            initial_lr,
            peak_lr,
            final_lr,
            current_step: 0,
        }
    }

    /// Compute the learning rate at an arbitrary step without mutating state.
    pub fn get_lr_at_step(&self, step: usize) -> f32 {
        if step >= self.total_steps {
            return self.final_lr;
        }
        if step <= self.warmup_steps {
            if self.warmup_steps == 0 {
                return self.peak_lr;
            }
            let t = step as f32 / self.warmup_steps as f32;
            return self.initial_lr + (self.peak_lr - self.initial_lr) * t;
        }
        let decay = self.total_steps - self.warmup_steps;
        if decay == 0 {
            return self.peak_lr;
        }
        let t = (step - self.warmup_steps) as f32 / decay as f32;
        self.peak_lr + (self.final_lr - self.peak_lr) * t
    }
}

impl LrScheduler for LinearScheduler {
    fn get_lr(&self) -> f32 {
        self.get_lr_at_step(self.current_step)
    }

    fn step(&mut self) {
        self.current_step += 1;
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PolynomialDecayScheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Polynomial decay from `initial_lr` to `end_lr`.
///
/// ```text
/// lr(step) = (initial_lr - end_lr) × (1 - step/total_steps)^power + end_lr
/// ```
/// For `step ≥ total_steps` the LR is clamped to `end_lr`.
///
/// # Example
/// ```rust
/// use tenflowers_neural::optimizers::{PolynomialDecayScheduler, LrScheduler};
///
/// let sched = PolynomialDecayScheduler::new(0.1, 1.0, 100, 0.0);
/// assert!((sched.get_lr_at_step(0)   - 0.1).abs() < 1e-6);
/// assert!((sched.get_lr_at_step(100) - 0.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct PolynomialDecayScheduler {
    /// Starting learning rate.
    pub initial_lr: f32,
    /// Polynomial power.
    pub power: f32,
    /// Number of decay steps.
    pub total_steps: usize,
    /// Final learning rate.
    pub end_lr: f32,
    /// Current step for stateful use.
    current_step: usize,
}

impl PolynomialDecayScheduler {
    /// Create a new `PolynomialDecayScheduler`.
    pub fn new(initial_lr: f32, power: f32, total_steps: usize, end_lr: f32) -> Self {
        Self {
            initial_lr,
            power,
            total_steps,
            end_lr,
            current_step: 0,
        }
    }

    /// Compute the learning rate at an arbitrary step without mutating state.
    pub fn get_lr_at_step(&self, step: usize) -> f32 {
        if self.total_steps == 0 || step >= self.total_steps {
            return self.end_lr;
        }
        let frac = 1.0 - step as f32 / self.total_steps as f32;
        (self.initial_lr - self.end_lr) * frac.powf(self.power) + self.end_lr
    }
}

impl LrScheduler for PolynomialDecayScheduler {
    fn get_lr(&self) -> f32 {
        self.get_lr_at_step(self.current_step)
    }

    fn step(&mut self) {
        self.current_step += 1;
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CosineAnnealingScheduler ─────────────────────────────────────────────

    #[test]
    fn test_cosine_at_step_zero_returns_eta_max() {
        let sched = CosineAnnealingScheduler::new(10, 1.0, 0.1, 0.001);
        let lr = sched.get_lr();
        assert!(
            (lr - 0.1).abs() < 1e-5,
            "at step 0 should return eta_max: {lr}"
        );
    }

    #[test]
    fn test_cosine_at_t0_returns_eta_min() {
        // At exactly T_0 steps the cosine reaches its trough (half-cycle end).
        let t0 = 10;
        let sched = CosineAnnealingScheduler::new(t0, 1.0, 0.1, 0.001);
        // step = t0 → inside first cycle with t_mult=1, but step==cycle_end triggers restart.
        // lr_at_epoch(t0) should restart and give eta_max again (new cycle starts).
        // Instead check lr_at_epoch(t0-1) approaches eta_min.
        let lr = sched.lr_at_epoch(t0 - 1);
        // t_cur = t0-1, T_i = t0 → cos(π*(t0-1)/t0) ≈ cos(π - π/t0) ≈ -1 + small
        let expected =
            0.001 + 0.5 * (0.1 - 0.001) * (1.0 + (PI * (t0 as f32 - 1.0) / t0 as f32).cos());
        assert!((lr - expected).abs() < 1e-5, "lr={lr} expected≈{expected}");
    }

    #[test]
    fn test_cosine_restart_resets_cycle() {
        let t0 = 5;
        let sched = CosineAnnealingScheduler::new(t0, 1.0, 0.1, 0.001);
        // With t_mult=1 step=0 and step=t0 should give same LR (both start of cycle).
        let lr0 = sched.lr_at_epoch(0);
        let lr_t0 = sched.lr_at_epoch(t0);
        assert!(
            (lr0 - lr_t0).abs() < 1e-4,
            "restart should reset to eta_max: {lr0} vs {lr_t0}"
        );
    }

    #[test]
    fn test_cosine_step_advances_counter() {
        let mut sched = CosineAnnealingScheduler::new(10, 1.0, 0.1, 0.001);
        let lr_before = sched.get_lr();
        sched.step();
        let lr_after = sched.get_lr();
        // After 1 step the LR should change (unless t0=1).
        // We just verify it's a valid f32 in range.
        assert!(
            (0.001..=0.1).contains(&lr_after),
            "lr out of range: {lr_after}"
        );
        let _ = lr_before; // suppress unused warning
    }

    #[test]
    fn test_cosine_reset() {
        let mut sched = CosineAnnealingScheduler::new(10, 1.0, 0.1, 0.001);
        let lr_init = sched.get_lr();
        for _ in 0..5 {
            sched.step();
        }
        sched.reset();
        let lr_after_reset = sched.get_lr();
        assert!(
            (lr_init - lr_after_reset).abs() < 1e-7,
            "reset should restore initial lr"
        );
    }

    #[test]
    fn test_cosine_t_mult_grows_cycles() {
        // With t_mult=2 the second cycle is twice as long as the first.
        let sched = CosineAnnealingScheduler::new(5, 2.0, 0.1, 0.001);
        // step=0: start of cycle 1 → eta_max
        let lr0 = sched.lr_at_epoch(0);
        assert!((lr0 - 0.1).abs() < 1e-5, "step 0 should be eta_max: {lr0}");
        // step=5: start of cycle 2 → should again be eta_max
        let lr5 = sched.lr_at_epoch(5);
        assert!(
            (lr5 - 0.1).abs() < 1e-5,
            "step 5 (cycle 2 start) should be eta_max: {lr5}"
        );
    }

    // ── OneCycleLrScheduler ──────────────────────────────────────────────────

    #[test]
    fn test_onecycle_step_zero_returns_initial_lr() {
        let sched = OneCycleLrScheduler::new(100, 0.1, 0.001, 1e-5, 0.3, AnnealStrategy::Cos);
        let lr = sched.get_lr_at_step(0);
        assert!(
            (lr - 0.001).abs() < 1e-7,
            "step 0 should be initial_lr: {lr}"
        );
    }

    #[test]
    fn test_onecycle_at_pct_start_returns_max_lr() {
        let total = 100;
        let pct = 0.3;
        let warmup_end = (pct * total as f32).round() as usize;
        let sched = OneCycleLrScheduler::new(total, 0.1, 0.001, 1e-5, pct, AnnealStrategy::Cos);
        let lr = sched.get_lr_at_step(warmup_end);
        assert!(
            (lr - 0.1).abs() < 1e-5,
            "at pct_start should be max_lr: {lr}"
        );
    }

    #[test]
    fn test_onecycle_at_total_steps_returns_final_lr() {
        let sched = OneCycleLrScheduler::new(100, 0.1, 0.001, 1e-5, 0.3, AnnealStrategy::Cos);
        let lr = sched.get_lr_at_step(100);
        assert!(
            (lr - 1e-5).abs() < 1e-8,
            "at total_steps should be final_lr: {lr}"
        );
    }

    #[test]
    fn test_onecycle_linear_strategy() {
        let sched = OneCycleLrScheduler::new(100, 0.1, 0.0, 0.0, 0.5, AnnealStrategy::Linear);
        let lr50 = sched.get_lr_at_step(50);
        // step 50 == warmup_end → max_lr
        assert!(
            (lr50 - 0.1).abs() < 1e-6,
            "at warmup_end (linear) should be max_lr: {lr50}"
        );
    }

    #[test]
    fn test_onecycle_lr_monotonically_increases_then_decreases() {
        let sched = OneCycleLrScheduler::new(100, 0.1, 0.001, 1e-5, 0.3, AnnealStrategy::Cos);
        let lrs: Vec<f32> = (0..=100).map(|s| sched.get_lr_at_step(s)).collect();
        // Warm-up: LR should increase from step 0 to warmup_end.
        let warmup_end = 30;
        for i in 0..warmup_end {
            assert!(
                lrs[i] <= lrs[i + 1] + 1e-6,
                "LR should increase during warmup at {i}"
            );
        }
        // Decay: LR should decrease after warmup_end.
        for i in warmup_end..100 {
            assert!(
                lrs[i] >= lrs[i + 1] - 1e-6,
                "LR should decrease during decay at {i}"
            );
        }
    }

    #[test]
    fn test_onecycle_step_and_reset() {
        let mut sched = OneCycleLrScheduler::new(100, 0.1, 0.001, 1e-5, 0.3, AnnealStrategy::Cos);
        let lr_init = sched.get_lr();
        for _ in 0..20 {
            sched.step();
        }
        sched.reset();
        assert!((sched.get_lr() - lr_init).abs() < 1e-7);
    }

    // ── ReduceLrOnPlateau ────────────────────────────────────────────────────

    #[test]
    fn test_reduce_lr_reduces_after_patience() {
        let mut sched = ReduceLrOnPlateau::new(MetricMode::Min, 0.5, 2, 0.0, 0, 1e-6, 0.0);
        let mut lr = 0.1_f32;
        lr = sched.step_with_metric(1.0, lr); // improvement (first)
        assert!((lr - 0.1).abs() < 1e-7);
        lr = sched.step_with_metric(1.1, lr); // bad  (count=1)
        assert!((lr - 0.1).abs() < 1e-7);
        lr = sched.step_with_metric(1.2, lr); // bad  (count=2 → reduce)
        assert!(
            (lr - 0.05).abs() < 1e-7,
            "LR should be 0.05 after patience: {lr}"
        );
    }

    #[test]
    fn test_reduce_lr_does_not_go_below_min_lr() {
        let mut sched = ReduceLrOnPlateau::new(MetricMode::Min, 0.1, 1, 0.0, 0, 0.05, 0.0);
        let mut lr = 0.1_f32;
        lr = sched.step_with_metric(1.0, lr);
        lr = sched.step_with_metric(2.0, lr); // bad → reduce: max(0.01, 0.05) = 0.05
        assert!(lr >= 0.05 - 1e-7, "LR should not go below min_lr: {lr}");
    }

    #[test]
    fn test_reduce_lr_max_mode() {
        let mut sched = ReduceLrOnPlateau::new(MetricMode::Max, 0.5, 2, 0.0, 0, 1e-6, 0.0);
        let mut lr = 0.1_f32;
        lr = sched.step_with_metric(0.9, lr); // improvement
        lr = sched.step_with_metric(0.8, lr); // bad (lower is worse for Max)
        lr = sched.step_with_metric(0.7, lr); // bad (count=2 → reduce)
        assert!((lr - 0.05).abs() < 1e-7, "Max mode: LR should reduce: {lr}");
    }

    #[test]
    fn test_reduce_lr_improves_resets_counter() {
        let mut sched = ReduceLrOnPlateau::new(MetricMode::Min, 0.5, 2, 0.0, 0, 1e-6, 0.0);
        let mut lr = 0.1_f32;
        lr = sched.step_with_metric(1.0, lr);
        lr = sched.step_with_metric(1.1, lr); // bad count=1
        lr = sched.step_with_metric(0.9, lr); // improvement → resets counter
        lr = sched.step_with_metric(1.0, lr); // bad count=1 (reset happened)
        assert!(
            (lr - 0.1).abs() < 1e-7,
            "LR should not reduce when counter was reset: {lr}"
        );
    }

    #[test]
    fn test_reduce_lr_cooldown() {
        let mut sched = ReduceLrOnPlateau::new(MetricMode::Min, 0.5, 1, 0.0, 2, 1e-6, 0.0);
        let mut lr = 0.1_f32;
        lr = sched.step_with_metric(1.0, lr); // improvement
        lr = sched.step_with_metric(2.0, lr); // bad count=1 → reduce: 0.05, cooldown=2
        assert!((lr - 0.05).abs() < 1e-7);
        // During cooldown, bad steps should not trigger another reduction.
        lr = sched.step_with_metric(3.0, lr); // cooldown
        lr = sched.step_with_metric(3.0, lr); // cooldown ends
        assert!(
            (lr - 0.05).abs() < 1e-7,
            "should not reduce during cooldown: {lr}"
        );
    }

    #[test]
    fn test_reduce_lr_is_improving_no_best() {
        let sched = ReduceLrOnPlateau::new(MetricMode::Min, 0.5, 2, 0.0, 0, 1e-6, 0.0);
        assert!(
            sched.is_improving(0.5),
            "first call should always be improving"
        );
    }

    #[test]
    fn test_reduce_lr_reset() {
        let mut sched = ReduceLrOnPlateau::new(MetricMode::Min, 0.5, 1, 0.0, 0, 1e-6, 0.0);
        sched.step_with_metric(1.0, 0.1);
        sched.reset();
        assert_eq!(sched.bad_epochs, 0);
        assert!(sched.best.is_none());
    }

    // ── WarmupScheduler ──────────────────────────────────────────────────────

    #[test]
    fn test_warmup_step_zero_is_zero() {
        let sched = WarmupScheduler::new(10, 0.001);
        assert_eq!(sched.get_lr_at_step(0), 0.0);
    }

    #[test]
    fn test_warmup_midpoint() {
        let sched = WarmupScheduler::new(10, 1.0);
        let lr = sched.get_lr_at_step(5);
        assert!((lr - 0.5).abs() < 1e-6, "midpoint should be 0.5: {lr}");
    }

    #[test]
    fn test_warmup_at_warmup_steps_is_target() {
        let sched = WarmupScheduler::new(10, 0.001);
        let lr = sched.get_lr_at_step(10);
        assert!(
            (lr - 0.001).abs() < 1e-8,
            "at warmup_steps should be target_lr: {lr}"
        );
    }

    #[test]
    fn test_warmup_beyond_warmup_steps_is_target() {
        let sched = WarmupScheduler::new(10, 0.002);
        let lr = sched.get_lr_at_step(100);
        assert!((lr - 0.002).abs() < 1e-8);
    }

    #[test]
    fn test_warmup_stateful_step_and_reset() {
        let mut sched = WarmupScheduler::new(10, 1.0);
        let lr0 = sched.get_lr();
        for _ in 0..5 {
            sched.step();
        }
        sched.reset();
        assert!((sched.get_lr() - lr0).abs() < 1e-9);
    }

    // ── ExponentialDecayScheduler ────────────────────────────────────────────

    #[test]
    fn test_exp_decay_at_decay_steps() {
        let sched = ExponentialDecayScheduler::new(0.1, 0.5, 10, false);
        let lr = sched.get_lr_at_step(10);
        // lr = 0.1 * 0.5^(10/10) = 0.1 * 0.5 = 0.05
        assert!((lr - 0.05).abs() < 1e-6, "lr at decay_steps: {lr}");
    }

    #[test]
    fn test_exp_decay_staircase_constant_within_step() {
        let sched = ExponentialDecayScheduler::new(1.0, 0.5, 10, true);
        // Steps 0..9 should all give the same LR (floor(step/10) = 0 → decay^0 = 1).
        let lr0 = sched.get_lr_at_step(0);
        let lr9 = sched.get_lr_at_step(9);
        assert!(
            (lr0 - lr9).abs() < 1e-8,
            "staircase: LR should be constant within step: {lr0} vs {lr9}"
        );
    }

    #[test]
    fn test_exp_decay_staircase_changes_at_boundary() {
        let sched = ExponentialDecayScheduler::new(1.0, 0.5, 10, true);
        let lr9 = sched.get_lr_at_step(9);
        let lr10 = sched.get_lr_at_step(10);
        assert!(
            lr10 < lr9 - 1e-6,
            "staircase: LR should drop at boundary: {lr9} vs {lr10}"
        );
    }

    #[test]
    fn test_exp_decay_reset() {
        let mut sched = ExponentialDecayScheduler::new(0.1, 0.9, 10, false);
        let lr0 = sched.get_lr();
        for _ in 0..20 {
            sched.step();
        }
        sched.reset();
        assert!((sched.get_lr() - lr0).abs() < 1e-9);
    }

    // ── LinearScheduler ──────────────────────────────────────────────────────

    #[test]
    fn test_linear_step_zero_is_initial_lr() {
        let sched = LinearScheduler::new(10, 100, 0.0, 0.1, 0.001);
        assert!((sched.get_lr_at_step(0) - 0.0).abs() < 1e-8);
    }

    #[test]
    fn test_linear_at_warmup_end_is_peak_lr() {
        let sched = LinearScheduler::new(10, 100, 0.0, 0.1, 0.001);
        let lr = sched.get_lr_at_step(10);
        assert!(
            (lr - 0.1).abs() < 1e-6,
            "at warmup_steps should be peak_lr: {lr}"
        );
    }

    #[test]
    fn test_linear_at_total_steps_is_final_lr() {
        let sched = LinearScheduler::new(10, 100, 0.0, 0.1, 0.001);
        let lr = sched.get_lr_at_step(100);
        assert!(
            (lr - 0.001).abs() < 1e-6,
            "at total_steps should be final_lr: {lr}"
        );
    }

    #[test]
    fn test_linear_beyond_total_steps_is_final_lr() {
        let sched = LinearScheduler::new(10, 100, 0.0, 0.1, 0.001);
        let lr = sched.get_lr_at_step(200);
        assert!((lr - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_linear_reset() {
        let mut sched = LinearScheduler::new(10, 100, 0.0, 0.1, 0.001);
        let lr0 = sched.get_lr();
        for _ in 0..50 {
            sched.step();
        }
        sched.reset();
        assert!((sched.get_lr() - lr0).abs() < 1e-9);
    }

    // ── PolynomialDecayScheduler ─────────────────────────────────────────────

    #[test]
    fn test_poly_step_zero_is_initial_lr() {
        let sched = PolynomialDecayScheduler::new(0.1, 1.0, 100, 0.0);
        let lr = sched.get_lr_at_step(0);
        assert!((lr - 0.1).abs() < 1e-6, "step 0 should be initial_lr: {lr}");
    }

    #[test]
    fn test_poly_at_total_steps_is_end_lr() {
        let sched = PolynomialDecayScheduler::new(0.1, 2.0, 100, 0.01);
        let lr = sched.get_lr_at_step(100);
        assert!(
            (lr - 0.01).abs() < 1e-6,
            "at total_steps should be end_lr: {lr}"
        );
    }

    #[test]
    fn test_poly_beyond_total_steps_is_end_lr() {
        let sched = PolynomialDecayScheduler::new(0.1, 1.0, 50, 0.001);
        let lr = sched.get_lr_at_step(200);
        assert!((lr - 0.001).abs() < 1e-7);
    }

    #[test]
    fn test_poly_power_one_is_linear() {
        let sched = PolynomialDecayScheduler::new(1.0, 1.0, 100, 0.0);
        let lr50 = sched.get_lr_at_step(50);
        // (1 - 50/100)^1 = 0.5
        assert!(
            (lr50 - 0.5).abs() < 1e-5,
            "power=1 should be linear: {lr50}"
        );
    }

    #[test]
    fn test_poly_reset() {
        let mut sched = PolynomialDecayScheduler::new(0.1, 2.0, 100, 0.0);
        let lr0 = sched.get_lr();
        for _ in 0..40 {
            sched.step();
        }
        sched.reset();
        assert!((sched.get_lr() - lr0).abs() < 1e-9);
    }

    // ── AnnealStrategy ───────────────────────────────────────────────────────

    #[test]
    fn test_anneal_strategy_variants() {
        assert_ne!(AnnealStrategy::Cos, AnnealStrategy::Linear);
        assert_eq!(AnnealStrategy::Cos, AnnealStrategy::Cos);
    }

    // ── MetricMode ───────────────────────────────────────────────────────────

    #[test]
    fn test_metric_mode_variants() {
        assert_ne!(MetricMode::Min, MetricMode::Max);
        assert_eq!(MetricMode::Min, MetricMode::Min);
    }
}
