//! Learning-rate range test and cyclic LR schedule (Smith 2017).
//!
//! # Overview
//!
//! The **LR range test** ([Smith 2017](https://arxiv.org/abs/1506.01186)) helps
//! you pick a good learning rate by training for a short period while
//! exponentially increasing the LR.  Loss initially decreases; once it starts
//! to diverge the test is stopped.  You then choose the LR just before the
//! divergence point.
//!
//! This module provides:
//!
//! - [`LrFinderConfig`]: parameters for the range test
//! - [`LrFinder`]:       stateful accumulator for `(lr, loss)` pairs
//! - [`CyclicLr`]:       cyclic LR schedule with triangular / exp-range modes
//!
//! # Example — LR range test
//!
//! ```rust
//! use tenflowers_neural::lr_finder::{LrFinder, LrFinderConfig};
//!
//! let config = LrFinderConfig::new();
//! let mut finder = LrFinder::new(config);
//!
//! // Simulated training loop
//! for step in 0..100 {
//!     if let Some(lr) = finder.next_lr(step) {
//!         // set your optimizer's lr here, run one mini-batch, get loss
//!         let loss = 2.0 / (1.0 + lr); // toy loss function
//!         if !finder.record_loss(lr, loss) {
//!             break; // diverged
//!         }
//!     }
//! }
//!
//! if let Some(suggested) = finder.suggest_lr() {
//!     println!("Suggested LR: {:.2e}", suggested);
//! }
//! ```

use tenflowers_core::TensorError;

/// Error type for LR-finder operations
#[derive(Debug, thiserror::Error)]
#[allow(clippy::large_enum_variant)]
pub enum LrFinderError {
    /// Configuration values are invalid (e.g. `start_lr >= end_lr`)
    #[error("invalid lr finder configuration: {0}")]
    InvalidConfig(String),
    /// Propagated from tensor operations
    #[error("tensor error: {0}")]
    TensorError(#[from] TensorError),
}

/// Result alias for LR-finder operations
pub type Result<T> = std::result::Result<T, LrFinderError>;

// ---------------------------------------------------------------------------
// LrFinderConfig
// ---------------------------------------------------------------------------

/// Configuration for the LR range test (Smith 2017).
#[derive(Debug, Clone)]
pub struct LrFinderConfig {
    /// Initial (lowest) learning rate to test
    pub start_lr: f32,
    /// Final (highest) learning rate to test
    pub end_lr: f32,
    /// Total number of LR steps to explore
    pub num_iterations: usize,
    /// Exponential moving average factor for smoothing the loss curve.
    /// Typical value: 0.98.
    pub smooth_factor: f32,
    /// Early-stop multiplier: test stops when
    /// `smoothed_loss > diverge_threshold * best_loss`.
    /// Typical value: 4.0.
    pub diverge_threshold: f32,
}

impl LrFinderConfig {
    /// Create a config with sensible defaults:
    /// - `start_lr`          = 1e-7
    /// - `end_lr`            = 10.0
    /// - `num_iterations`    = 100
    /// - `smooth_factor`     = 0.98
    /// - `diverge_threshold` = 4.0
    pub fn new() -> Self {
        Self {
            start_lr: 1e-7,
            end_lr: 10.0,
            num_iterations: 100,
            smooth_factor: 0.98,
            diverge_threshold: 4.0,
        }
    }

    /// Create a config with custom LR range but default other settings.
    pub fn with_range(start_lr: f32, end_lr: f32) -> Self {
        Self {
            start_lr,
            end_lr,
            ..Self::new()
        }
    }
}

impl Default for LrFinderConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// LrStepResult
// ---------------------------------------------------------------------------

/// Recorded result for a single step of the LR range test.
#[derive(Debug, Clone)]
pub struct LrStepResult {
    /// Learning rate used at this step
    pub lr: f32,
    /// Raw (unsmoothed) loss
    pub loss: f32,
    /// EMA-smoothed loss (bias-corrected)
    pub smoothed_loss: f32,
}

// ---------------------------------------------------------------------------
// LrFinder
// ---------------------------------------------------------------------------

/// Stateful LR range-test accumulator.
///
/// Usage:
/// 1. Call [`next_lr`](LrFinder::next_lr) each step to get the next LR to try.
/// 2. Run one mini-batch with that LR and record the loss with
///    [`record_loss`](LrFinder::record_loss).
/// 3. After the test, call [`suggest_lr`](LrFinder::suggest_lr) or
///    [`lr_at_min_loss`](LrFinder::lr_at_min_loss).
pub struct LrFinder {
    /// Configuration for this test
    pub config: LrFinderConfig,
    /// All recorded steps, in order
    pub history: Vec<LrStepResult>,
    /// Best (lowest) smoothed loss seen so far
    best_loss: f32,
    /// Current EMA smoothed loss accumulator (raw, not bias-corrected yet)
    smoothed_loss: f32,
    /// Whether the test was stopped because of divergence
    pub stopped_early: bool,
    /// Step counter (used internally for bias correction)
    step_count: usize,
}

impl LrFinder {
    /// Create a new finder from the given configuration.
    pub fn new(config: LrFinderConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            best_loss: f32::MAX,
            smoothed_loss: 0.0,
            stopped_early: false,
            step_count: 0,
        }
    }

    /// Compute the next LR to try at training `step` using an exponential schedule.
    ///
    /// The schedule interpolates from `start_lr` to `end_lr` over
    /// `num_iterations` steps:
    ///
    ///   `lr = start_lr * (end_lr / start_lr) ^ (step / (num_iterations - 1))`
    ///
    /// Returns `None` when:
    /// - `step >= num_iterations`, or
    /// - early stopping was already triggered.
    pub fn next_lr(&mut self, step: usize) -> Option<f32> {
        if self.stopped_early || step >= self.config.num_iterations {
            return None;
        }

        let num_iter = self.config.num_iterations;
        let start = self.config.start_lr;
        let end = self.config.end_lr;

        let lr = if num_iter <= 1 {
            start
        } else {
            let progress = step as f32 / (num_iter - 1) as f32;
            start * (end / start).powf(progress)
        };

        Some(lr)
    }

    /// Record the raw loss observed after running with `lr`.
    ///
    /// Applies EMA smoothing with bias correction, updates `best_loss`, and
    /// checks for divergence.
    ///
    /// Returns `true` if training should continue, `false` if divergence was
    /// detected and the test should be stopped.
    pub fn record_loss(&mut self, lr: f32, loss: f32) -> bool {
        self.step_count += 1;
        let beta = self.config.smooth_factor;

        // EMA accumulator (not bias-corrected yet)
        self.smoothed_loss = beta * self.smoothed_loss + (1.0 - beta) * loss;

        // Bias correction
        let bias_correction = 1.0 - beta.powi(self.step_count as i32);
        let corrected = if bias_correction.abs() < f32::EPSILON {
            self.smoothed_loss
        } else {
            self.smoothed_loss / bias_correction
        };

        if corrected < self.best_loss {
            self.best_loss = corrected;
        }

        self.history.push(LrStepResult {
            lr,
            loss,
            smoothed_loss: corrected,
        });

        // Divergence check
        if corrected > self.config.diverge_threshold * self.best_loss {
            self.stopped_early = true;
            return false;
        }

        true
    }

    /// Suggest the optimal LR as the point of steepest (most negative) loss
    /// gradient in the smoothed history.
    ///
    /// Returns `None` if fewer than two steps have been recorded.
    pub fn suggest_lr(&self) -> Option<f32> {
        if self.history.len() < 2 {
            return None;
        }

        let mut steepest_idx = 1usize;
        let mut steepest_grad = f32::MAX; // we want most-negative

        for i in 1..self.history.len() {
            let grad = self.history[i].smoothed_loss - self.history[i - 1].smoothed_loss;
            if grad < steepest_grad {
                steepest_grad = grad;
                steepest_idx = i;
            }
        }

        Some(self.history[steepest_idx].lr)
    }

    /// Return the LR at which the smoothed loss reached its minimum.
    ///
    /// Returns `None` if no steps have been recorded.
    pub fn lr_at_min_loss(&self) -> Option<f32> {
        self.history
            .iter()
            .min_by(|a, b| {
                a.smoothed_loss
                    .partial_cmp(&b.smoothed_loss)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.lr)
    }

    /// Number of steps recorded so far.
    pub fn num_steps(&self) -> usize {
        self.history.len()
    }
}

// ---------------------------------------------------------------------------
// CyclicMode
// ---------------------------------------------------------------------------

/// Mode for the [`CyclicLr`] schedule.
#[derive(Debug, Clone)]
pub enum CyclicMode {
    /// Constant amplitude throughout training (triangular wave)
    Triangular,
    /// Amplitude halves each full cycle
    Triangular2,
    /// Amplitude decays exponentially: `gamma ^ cycle_number`
    ExpRange(f32),
}

// ---------------------------------------------------------------------------
// CyclicLr
// ---------------------------------------------------------------------------

/// Cyclic Learning Rate schedule (Smith 2017).
///
/// The LR oscillates between `base_lr` and `max_lr`.  Each cycle consists of
/// a ramp-up phase (`step_size_up` steps) and a ramp-down phase
/// (`step_size_down` steps).
///
/// # Usage
/// ```
/// use tenflowers_neural::lr_finder::{CyclicLr, CyclicMode};
///
/// let mut scheduler = CyclicLr::new(0.001, 0.01, 5).expect("valid config");
/// for _ in 0..20 {
///     let lr = scheduler.get_lr();
///     scheduler.step();
///     // use lr for this training step
///     let _ = lr;
/// }
/// ```
pub struct CyclicLr {
    /// Minimum (base) learning rate
    pub base_lr: f32,
    /// Maximum learning rate
    pub max_lr: f32,
    /// Steps to ramp from `base_lr` to `max_lr`
    pub step_size_up: usize,
    /// Steps to ramp from `max_lr` back to `base_lr`
    pub step_size_down: usize,
    /// Oscillation mode (see [`CyclicMode`])
    pub mode: CyclicMode,
    /// Internal step counter
    step_count: usize,
}

impl CyclicLr {
    /// Create a new [`CyclicLr`] with equal up/down phases.
    ///
    /// `step_size` is the number of steps per phase (up *and* down phases are
    /// both `step_size` steps, so a full cycle is `2 * step_size` steps).
    ///
    /// # Errors
    /// Returns an error if `step_size == 0` or `base_lr >= max_lr`.
    pub fn new(base_lr: f32, max_lr: f32, step_size: usize) -> Result<Self> {
        if step_size == 0 {
            return Err(LrFinderError::InvalidConfig(
                "step_size must be greater than 0".to_string(),
            ));
        }
        if base_lr >= max_lr {
            return Err(LrFinderError::InvalidConfig(format!(
                "base_lr ({}) must be less than max_lr ({})",
                base_lr, max_lr
            )));
        }
        Ok(Self {
            base_lr,
            max_lr,
            step_size_up: step_size,
            step_size_down: step_size,
            mode: CyclicMode::Triangular,
            step_count: 0,
        })
    }

    /// Compute the current LR without advancing the step counter.
    ///
    /// The computation:
    /// 1. Determine which cycle we are in and the position within the cycle.
    /// 2. Compute a triangular amplitude multiplier in `[0, 1]`.
    /// 3. Scale by the mode-specific decay factor.
    /// 4. Interpolate between `base_lr` and `max_lr`.
    pub fn get_lr(&mut self) -> f32 {
        let cycle_length = self.step_size_up + self.step_size_down;

        let step_in_cycle = self.step_count % cycle_length;

        // Amplitude in [0, 1]
        let x = if step_in_cycle < self.step_size_up {
            step_in_cycle as f32 / self.step_size_up as f32
        } else {
            let down_step = step_in_cycle - self.step_size_up;
            1.0 - (down_step as f32 / self.step_size_down as f32)
        };

        // Which full cycle are we in?
        let cycle = self.step_count / cycle_length;

        // Mode-specific scale
        let scale = match &self.mode {
            CyclicMode::Triangular => 1.0_f32,
            CyclicMode::Triangular2 => 0.5_f32.powi(cycle as i32),
            CyclicMode::ExpRange(gamma) => gamma.powi(self.step_count as i32),
        };

        self.base_lr + (self.max_lr - self.base_lr) * x * scale
    }

    /// Advance the internal step counter by one.
    pub fn step(&mut self) {
        self.step_count += 1;
    }

    /// Reset the schedule back to step 0.
    pub fn reset(&mut self) {
        self.step_count = 0;
    }

    /// Return the current (internal) step count.
    pub fn current_step(&self) -> usize {
        self.step_count
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- LrFinderConfig ------------------------------------------------------

    #[test]
    fn test_lr_finder_config_defaults() {
        let cfg = LrFinderConfig::new();
        assert!((cfg.start_lr - 1e-7).abs() < 1e-10, "start_lr default");
        assert!((cfg.end_lr - 10.0).abs() < 1e-6, "end_lr default");
        assert_eq!(cfg.num_iterations, 100);
        assert!((cfg.smooth_factor - 0.98).abs() < 1e-6, "smooth_factor");
        assert!(
            (cfg.diverge_threshold - 4.0).abs() < 1e-6,
            "diverge_threshold"
        );
    }

    #[test]
    fn test_lr_finder_config_with_range() {
        let cfg = LrFinderConfig::with_range(1e-4, 1.0);
        assert!((cfg.start_lr - 1e-4).abs() < 1e-9);
        assert!((cfg.end_lr - 1.0).abs() < 1e-6);
        // other defaults preserved
        assert_eq!(cfg.num_iterations, 100);
    }

    // --- LrFinder::next_lr ---------------------------------------------------

    #[test]
    fn test_lr_finder_next_lr_first_step_is_start() {
        let mut finder = LrFinder::new(LrFinderConfig::with_range(1e-4, 1.0));
        let lr = finder.next_lr(0).expect("step 0 should return Some");
        assert!((lr - 1e-4).abs() < 1e-8, "step 0 LR should be start_lr");
    }

    #[test]
    fn test_lr_finder_next_lr_last_step_is_end() {
        let cfg = LrFinderConfig {
            num_iterations: 10,
            ..LrFinderConfig::with_range(1e-4, 1.0)
        };
        let mut finder = LrFinder::new(cfg);
        let lr = finder.next_lr(9).expect("last step should return Some");
        assert!(
            (lr - 1.0).abs() < 1e-5,
            "last step LR should be end_lr, got {}",
            lr
        );
    }

    #[test]
    fn test_lr_finder_next_lr_exponential_schedule() {
        let cfg = LrFinderConfig {
            num_iterations: 5,
            ..LrFinderConfig::with_range(1.0, 100.0)
        };
        let mut finder = LrFinder::new(cfg);
        // step 0 => 1.0, step 4 => 100.0; step 2 => 10.0 (sqrt(100))
        let lr2 = finder.next_lr(2).expect("step 2");
        assert!(
            (lr2 - 10.0).abs() < 0.1,
            "expected ~10.0 at step 2, got {}",
            lr2
        );
    }

    #[test]
    fn test_lr_finder_next_lr_beyond_iterations_returns_none() {
        let cfg = LrFinderConfig {
            num_iterations: 5,
            ..LrFinderConfig::new()
        };
        let mut finder = LrFinder::new(cfg);
        assert!(finder.next_lr(5).is_none());
        assert!(finder.next_lr(10).is_none());
    }

    // --- LrFinder::record_loss -----------------------------------------------

    #[test]
    fn test_lr_finder_record_loss_continues_normally() {
        let mut finder = LrFinder::new(LrFinderConfig::new());
        // Monotonically decreasing loss — should not diverge
        let should_continue = finder.record_loss(1e-5, 1.0);
        assert!(should_continue);
        let should_continue2 = finder.record_loss(1e-4, 0.9);
        assert!(should_continue2);
    }

    #[test]
    fn test_lr_finder_record_loss_detects_divergence() {
        let mut finder = LrFinder::new(LrFinderConfig::new());
        // First step — establishes a best_loss close to initial value
        finder.record_loss(1e-7, 1.0);
        // Feed a loss that is much higher than best
        // Need several bad losses to overcome EMA smoothing
        for _ in 0..30 {
            finder.record_loss(1.0, 1000.0);
        }
        assert!(
            finder.stopped_early,
            "expected early stopping on large loss"
        );
    }

    #[test]
    fn test_lr_finder_record_loss_accumulates_history() {
        let mut finder = LrFinder::new(LrFinderConfig::new());
        finder.record_loss(1e-5, 2.0);
        finder.record_loss(1e-4, 1.5);
        finder.record_loss(1e-3, 1.2);
        assert_eq!(finder.num_steps(), 3);
    }

    // --- LrFinder::suggest_lr ------------------------------------------------

    #[test]
    fn test_lr_finder_suggest_lr_returns_none_when_empty() {
        let finder = LrFinder::new(LrFinderConfig::new());
        assert!(finder.suggest_lr().is_none());
    }

    #[test]
    fn test_lr_finder_suggest_lr_single_step_returns_none() {
        let mut finder = LrFinder::new(LrFinderConfig::new());
        finder.record_loss(1e-5, 1.0);
        assert!(finder.suggest_lr().is_none());
    }

    #[test]
    fn test_lr_finder_suggest_lr_reasonable_value() {
        let mut finder = LrFinder::new(LrFinderConfig::new());
        // Simulate a typical loss curve: decreasing then exploding
        let lrs = [1e-5_f32, 1e-4, 1e-3, 1e-2, 1e-1];
        let losses = [2.0_f32, 1.5, 1.0, 0.5, 5.0]; // min at lr=1e-2
        for (&lr, &loss) in lrs.iter().zip(losses.iter()) {
            finder.record_loss(lr, loss);
        }
        let suggested = finder.suggest_lr().expect("suggest_lr should return Some");
        // The steepest descent is somewhere before the blowup
        assert!(suggested > 0.0);
        assert!(suggested < 1.0);
    }

    // --- LrFinder::lr_at_min_loss -------------------------------------------

    #[test]
    fn test_lr_at_min_loss_returns_none_when_empty() {
        let finder = LrFinder::new(LrFinderConfig::new());
        assert!(finder.lr_at_min_loss().is_none());
    }

    #[test]
    fn test_lr_at_min_loss_finds_minimum() {
        let mut finder = LrFinder::new(LrFinderConfig::new());
        finder.record_loss(1e-4, 2.0);
        finder.record_loss(1e-3, 0.5); // this is the minimum region
        finder.record_loss(1e-2, 3.0);
        let min_lr = finder.lr_at_min_loss().expect("should find a minimum");
        // The smoothed loss minimum should correspond to the 1e-3 region
        assert!(min_lr > 0.0);
    }

    // --- CyclicLr ------------------------------------------------------------

    #[test]
    fn test_cyclic_lr_new_valid() {
        let sched = CyclicLr::new(0.001, 0.01, 5);
        assert!(sched.is_ok());
    }

    #[test]
    fn test_cyclic_lr_new_invalid_zero_step() {
        let result = CyclicLr::new(0.001, 0.01, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_cyclic_lr_new_invalid_base_ge_max() {
        let result = CyclicLr::new(0.01, 0.001, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_cyclic_lr_step_0_equals_base_lr() {
        let mut sched = CyclicLr::new(0.001, 0.01, 5).expect("valid config");
        let lr = sched.get_lr();
        assert!(
            (lr - 0.001).abs() < 1e-7,
            "expected base_lr at step 0, got {}",
            lr
        );
    }

    #[test]
    fn test_cyclic_lr_at_step_size_up_equals_max_lr() {
        let mut sched = CyclicLr::new(0.001, 0.01, 5).expect("valid config");
        // Advance to step 5 (= step_size_up)
        for _ in 0..5 {
            sched.step();
        }
        let lr = sched.get_lr();
        assert!(
            (lr - 0.01).abs() < 1e-6,
            "expected max_lr at step step_size_up, got {}",
            lr
        );
    }

    #[test]
    fn test_cyclic_lr_triangular2_halves_amplitude() {
        let mut sched = CyclicLr::new(0.0, 1.0, 4).expect("valid config");
        sched.mode = CyclicMode::Triangular2;

        // At the peak of cycle 0 (step 4), amplitude factor = 0.5^0 = 1.0
        for _ in 0..4 {
            sched.step();
        }
        let peak0 = sched.get_lr(); // should be close to 1.0

        // At the peak of cycle 1 (step 12, i.e. 4+4+4), amplitude factor = 0.5^1 = 0.5
        for _ in 0..8 {
            sched.step();
        }
        let peak1 = sched.get_lr(); // should be close to 0.5

        assert!(
            (peak0 - 1.0).abs() < 1e-5,
            "peak0 should be ~1.0, got {}",
            peak0
        );
        assert!(
            (peak1 - 0.5).abs() < 1e-5,
            "peak1 should be ~0.5, got {}",
            peak1
        );
    }

    #[test]
    fn test_cyclic_lr_reset() {
        let mut sched = CyclicLr::new(0.001, 0.01, 5).expect("valid config");
        for _ in 0..10 {
            sched.step();
        }
        assert_eq!(sched.current_step(), 10);
        sched.reset();
        assert_eq!(sched.current_step(), 0);
        let lr = sched.get_lr();
        assert!(
            (lr - 0.001).abs() < 1e-7,
            "after reset, LR should be base_lr, got {}",
            lr
        );
    }

    #[test]
    fn test_cyclic_lr_completes_full_cycle() {
        // After one full cycle (2 * step_size steps), LR should return to base_lr
        let step_size = 6usize;
        let mut sched = CyclicLr::new(0.001, 0.01, step_size).expect("valid config");
        for _ in 0..(2 * step_size) {
            sched.step();
        }
        let lr = sched.get_lr();
        assert!(
            (lr - 0.001).abs() < 1e-6,
            "after full cycle, LR should be base_lr, got {}",
            lr
        );
    }
}
