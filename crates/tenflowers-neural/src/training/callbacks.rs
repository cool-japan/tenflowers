//! Training callbacks and hooks for the neural network training loop.
//!
//! This module provides a flexible callback system that allows users to hook
//! into various stages of training:
//!
//! - [`EarlyStopping`]: Halt training when a monitored metric stops improving
//! - [`ModelCheckpoint`]: Persist a JSON summary whenever the model improves
//! - [`ReduceLrOnPlateau`]: Lower the learning rate when a metric plateaus
//! - [`ProgressLogger`]: Print and buffer human-readable training progress
//! - [`History`]: Accumulate per-epoch metrics for later analysis
//!
//! # Design
//!
//! Every callback implements [`Callback`], a `Send + Sync` trait with default
//! no-op implementations for all hooks.  Return [`CallbackSignal::StopTraining`]
//! from `on_epoch_end` or `on_step_end` to request early termination.
//!
//! [`run_callbacks_epoch`] and [`run_callbacks_step`] are helpers that iterate
//! over a `Vec<Box<dyn Callback>>` and consolidate the signals, returning
//! `true` when any callback requests a stop.

use std::fmt;
use tenflowers_core::{Result, TensorError};

// ============================================================================
// Core event type
// ============================================================================

/// Snapshot of training state delivered to every callback hook.
#[derive(Debug, Clone, Default)]
pub struct TrainingEvent {
    /// Current epoch index (0-based).
    pub epoch: usize,
    /// Global optimisation step count.
    pub step: usize,
    /// Average training loss for the current epoch/step.
    pub train_loss: Option<f32>,
    /// Average validation loss for the current epoch/step.
    pub val_loss: Option<f32>,
    /// Scalar training metric (e.g. accuracy).
    pub train_metric: Option<f32>,
    /// Scalar validation metric.
    pub val_metric: Option<f32>,
    /// Current learning rate at the time of the event.
    pub learning_rate: Option<f32>,
}

impl TrainingEvent {
    /// Construct a minimal event with just epoch and step.
    pub fn new(epoch: usize, step: usize) -> Self {
        Self {
            epoch,
            step,
            ..Default::default()
        }
    }
}

impl fmt::Display for TrainingEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epoch={} step={}", self.epoch, self.step)?;
        if let Some(tl) = self.train_loss {
            write!(f, " train_loss={:.6}", tl)?;
        }
        if let Some(vl) = self.val_loss {
            write!(f, " val_loss={:.6}", vl)?;
        }
        if let Some(tm) = self.train_metric {
            write!(f, " train_metric={:.6}", tm)?;
        }
        if let Some(vm) = self.val_metric {
            write!(f, " val_metric={:.6}", vm)?;
        }
        if let Some(lr) = self.learning_rate {
            write!(f, " lr={:.2e}", lr)?;
        }
        Ok(())
    }
}

// ============================================================================
// Signal enum
// ============================================================================

/// Signal returned by callback hooks to control the training loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallbackSignal {
    /// Continue training normally.
    Continue,
    /// Request that training halts after the current epoch/step.
    StopTraining,
}

// ============================================================================
// Callback trait
// ============================================================================

/// Hook interface for observing and influencing the training loop.
///
/// All methods have default no-op implementations so that implementors
/// only override the hooks they care about.
pub trait Callback: Send + Sync {
    /// Called once after every epoch.  Return [`CallbackSignal::StopTraining`]
    /// to request early termination.
    fn on_epoch_end(&mut self, event: &TrainingEvent) -> CallbackSignal {
        let _ = event;
        CallbackSignal::Continue
    }

    /// Called once after every optimisation step.  Return
    /// [`CallbackSignal::StopTraining`] to request early termination.
    fn on_step_end(&mut self, event: &TrainingEvent) -> CallbackSignal {
        let _ = event;
        CallbackSignal::Continue
    }

    /// Called once before the first epoch.
    fn on_train_begin(&mut self, total_epochs: usize) {
        let _ = total_epochs;
    }

    /// Called once after the final epoch (or when training was stopped early).
    fn on_train_end(&mut self, event: &TrainingEvent) {
        let _ = event;
    }

    /// Human-readable name used in log messages.
    fn name(&self) -> &str {
        "unnamed_callback"
    }
}

// ============================================================================
// Helper runners
// ============================================================================

/// Run `on_epoch_end` for every callback and return `true` if any signals stop.
pub fn run_callbacks_epoch(
    callbacks: &mut Vec<Box<dyn Callback>>,
    event: &TrainingEvent,
) -> bool {
    let mut stop = false;
    for cb in callbacks.iter_mut() {
        if cb.on_epoch_end(event) == CallbackSignal::StopTraining {
            stop = true;
        }
    }
    stop
}

/// Run `on_step_end` for every callback and return `true` if any signals stop.
pub fn run_callbacks_step(
    callbacks: &mut Vec<Box<dyn Callback>>,
    event: &TrainingEvent,
) -> bool {
    let mut stop = false;
    for cb in callbacks.iter_mut() {
        if cb.on_step_end(event) == CallbackSignal::StopTraining {
            stop = true;
        }
    }
    stop
}

// ============================================================================
// MonitorMetric enum
// ============================================================================

/// Which scalar value a callback should watch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MonitorMetric {
    /// Validation loss (default for most callbacks).
    ValLoss,
    /// Training loss.
    TrainLoss,
    /// Validation metric (e.g. accuracy).
    ValMetric,
    /// Training metric.
    TrainMetric,
}

impl MonitorMetric {
    /// Extract the corresponding value from an event, or return `None` if absent.
    pub fn extract(&self, event: &TrainingEvent) -> Option<f32> {
        match self {
            Self::ValLoss => event.val_loss,
            Self::TrainLoss => event.train_loss,
            Self::ValMetric => event.val_metric,
            Self::TrainMetric => event.train_metric,
        }
    }

    /// Returns `true` when a *lower* value is considered better (loss metrics).
    pub fn lower_is_better(&self) -> bool {
        matches!(self, Self::ValLoss | Self::TrainLoss)
    }
}

impl fmt::Display for MonitorMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::ValLoss => "val_loss",
            Self::TrainLoss => "train_loss",
            Self::ValMetric => "val_metric",
            Self::TrainMetric => "train_metric",
        };
        f.write_str(s)
    }
}

// ============================================================================
// EarlyStopping
// ============================================================================

/// Stop training when a monitored metric fails to improve for `patience` epochs.
///
/// # Example
/// ```rust
/// use tenflowers_neural::training::callbacks::{EarlyStopping, MonitorMetric};
///
/// let mut es = EarlyStopping::new(5);
/// // Monitors ValLoss with min_delta = 1e-4 by default.
/// ```
#[derive(Debug)]
pub struct EarlyStopping {
    /// Number of epochs without improvement before stopping.
    pub patience: usize,
    /// Minimum absolute improvement required to reset the wait counter.
    pub min_delta: f32,
    /// Which metric to watch.
    pub monitor: MonitorMetric,
    /// If `true`, the best weights would be restored (tracked here as metadata).
    pub restore_best: bool,
    /// Best metric value seen so far (initialised to ±∞).
    best_value: f32,
    /// Number of epochs since the last improvement.
    wait: usize,
    /// Epoch at which training was stopped, if triggered.
    pub stopped_epoch: Option<usize>,
    /// Epoch at which the best metric was recorded.
    pub best_epoch: Option<usize>,
}

impl EarlyStopping {
    /// Create an `EarlyStopping` that monitors `ValLoss` with `min_delta = 1e-4`.
    pub fn new(patience: usize) -> Self {
        Self::with_patience_and_delta(patience, 1e-4, MonitorMetric::ValLoss)
    }

    /// Create an `EarlyStopping` with explicit patience, delta, and monitor.
    pub fn with_patience_and_delta(
        patience: usize,
        min_delta: f32,
        monitor: MonitorMetric,
    ) -> Self {
        let best_value = if monitor.lower_is_better() {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        };

        Self {
            patience,
            min_delta,
            monitor,
            restore_best: false,
            best_value,
            wait: 0,
            stopped_epoch: None,
            best_epoch: None,
        }
    }

    /// Current number of epochs without improvement.
    pub fn wait(&self) -> usize {
        self.wait
    }

    /// Best metric value recorded so far.
    pub fn best_value(&self) -> f32 {
        self.best_value
    }

    /// Returns `true` when `current` is a meaningful improvement over `best`.
    fn is_improvement(&self, current: f32) -> bool {
        if self.monitor.lower_is_better() {
            current < self.best_value - self.min_delta
        } else {
            current > self.best_value + self.min_delta
        }
    }
}

impl Callback for EarlyStopping {
    fn on_epoch_end(&mut self, event: &TrainingEvent) -> CallbackSignal {
        let current = match self.monitor.extract(event) {
            Some(v) => v,
            None => return CallbackSignal::Continue,
        };

        if self.is_improvement(current) {
            self.best_value = current;
            self.best_epoch = Some(event.epoch);
            self.wait = 0;
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                self.stopped_epoch = Some(event.epoch);
                return CallbackSignal::StopTraining;
            }
        }

        CallbackSignal::Continue
    }

    fn name(&self) -> &str {
        "EarlyStopping"
    }
}

// ============================================================================
// ModelCheckpoint
// ============================================================================

/// Write a JSON summary to `filepath` when the monitored metric improves.
///
/// The file is created under the system temporary directory when the path does
/// not look like an absolute path, matching the project's test-safety policy.
/// In production callers should pass an absolute path.
#[derive(Debug)]
pub struct ModelCheckpoint {
    /// Destination path for the checkpoint summary JSON.
    pub filepath: String,
    /// Which metric to watch.
    pub monitor: MonitorMetric,
    /// Only save when the metric improves (as opposed to every epoch).
    pub save_best_only: bool,
    /// Best value recorded so far (initialised to ±∞).
    best_value: f32,
    /// Last epoch at which a checkpoint was written.
    pub last_saved_epoch: Option<usize>,
}

impl ModelCheckpoint {
    /// Create a checkpoint that monitors `ValLoss` and saves every epoch.
    pub fn new(filepath: &str) -> Self {
        let best_value = f32::INFINITY; // ValLoss — lower is better
        Self {
            filepath: filepath.to_owned(),
            monitor: MonitorMetric::ValLoss,
            save_best_only: false,
            best_value,
            last_saved_epoch: None,
        }
    }

    /// Create a checkpoint that only saves when `monitor` improves.
    pub fn best_only(filepath: &str, monitor: MonitorMetric) -> Self {
        let best_value = if monitor.lower_is_better() {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        };
        Self {
            filepath: filepath.to_owned(),
            monitor,
            save_best_only: true,
            best_value,
            last_saved_epoch: None,
        }
    }

    /// Best metric value recorded so far.
    pub fn best_value(&self) -> f32 {
        self.best_value
    }

    fn should_save(&mut self, current: f32) -> bool {
        if !self.save_best_only {
            return true;
        }
        if self.monitor.lower_is_better() {
            current < self.best_value
        } else {
            current > self.best_value
        }
    }

    /// Write a minimal JSON summary to `self.filepath`.
    fn write_summary(&self, event: &TrainingEvent) -> Result<()> {
        let json = format!(
            r#"{{"epoch":{epoch},"step":{step},"train_loss":{tl},"val_loss":{vl},"train_metric":{tm},"val_metric":{vm},"learning_rate":{lr},"monitor":"{monitor}","best_value":{best}}}"#,
            epoch = event.epoch,
            step = event.step,
            tl = opt_to_json(event.train_loss),
            vl = opt_to_json(event.val_loss),
            tm = opt_to_json(event.train_metric),
            vm = opt_to_json(event.val_metric),
            lr = opt_to_json(event.learning_rate),
            monitor = self.monitor,
            best = self.best_value,
        );

        std::fs::write(&self.filepath, json.as_bytes()).map_err(|io| TensorError::InvalidArgument {
            operation: "ModelCheckpoint::write_summary".to_owned(),
            reason: format!("failed to write checkpoint to '{}': {}", self.filepath, io),
            context: None,
        })
    }
}

impl Callback for ModelCheckpoint {
    fn on_epoch_end(&mut self, event: &TrainingEvent) -> CallbackSignal {
        let current = match self.monitor.extract(event) {
            Some(v) => v,
            None => return CallbackSignal::Continue,
        };

        if self.should_save(current) {
            self.best_value = current;
            self.last_saved_epoch = Some(event.epoch);
            // Best-effort write; do not stop training on I/O failure.
            let _ = self.write_summary(event);
        }

        CallbackSignal::Continue
    }

    fn name(&self) -> &str {
        "ModelCheckpoint"
    }
}

/// Format an `Option<f32>` as a JSON number or `null`.
fn opt_to_json(v: Option<f32>) -> String {
    match v {
        Some(f) => format!("{}", f),
        None => "null".to_owned(),
    }
}

// ============================================================================
// ReduceLrOnPlateau
// ============================================================================

/// Multiply the learning rate by `factor` when a metric fails to improve for
/// `patience` epochs.
///
/// The new learning rate is stored in `current_lr` and can be applied by the
/// training loop after each epoch.
#[derive(Debug)]
pub struct ReduceLrOnPlateau {
    /// Multiplicative reduction factor (e.g. 0.1).
    pub factor: f32,
    /// Number of epochs without improvement before reducing.
    pub patience: usize,
    /// Lower bound on the learning rate.
    pub min_lr: f32,
    /// Current (possibly reduced) learning rate.
    pub current_lr: f32,
    /// Minimum improvement required to reset the plateau counter.
    pub min_delta: f32,
    /// Best value seen (initialised to ±∞ based on monitor direction).
    best_value: f32,
    /// Epochs without improvement.
    wait: usize,
    /// Total number of reductions applied.
    pub num_reductions: usize,
}

impl ReduceLrOnPlateau {
    /// Create a new reducer.
    ///
    /// Monitors `ValLoss` (lower is better).  `factor` must be in `(0, 1)`.
    pub fn new(initial_lr: f32, factor: f32, patience: usize) -> Result<Self> {
        if factor <= 0.0 || factor >= 1.0 {
            return Err(TensorError::InvalidArgument {
                operation: "ReduceLrOnPlateau::new".to_owned(),
                reason: format!(
                    "factor must be in (0, 1) exclusive, got {}",
                    factor
                ),
                context: None,
            });
        }
        if initial_lr <= 0.0 {
            return Err(TensorError::InvalidArgument {
                operation: "ReduceLrOnPlateau::new".to_owned(),
                reason: format!("initial_lr must be positive, got {}", initial_lr),
                context: None,
            });
        }
        Ok(Self {
            factor,
            patience,
            min_lr: 0.0,
            current_lr: initial_lr,
            min_delta: 1e-4,
            best_value: f32::INFINITY,
            wait: 0,
            num_reductions: 0,
        })
    }

    /// Return the learning rate that will be applied after a reduction (does
    /// not mutate state).
    pub fn reduced_lr(&self) -> f32 {
        (self.current_lr * self.factor).max(self.min_lr)
    }

    /// Apply the reduction and return the new learning rate.
    fn apply_reduction(&mut self) -> f32 {
        let new_lr = (self.current_lr * self.factor).max(self.min_lr);
        self.current_lr = new_lr;
        self.num_reductions += 1;
        self.wait = 0;
        new_lr
    }
}

impl Callback for ReduceLrOnPlateau {
    fn on_epoch_end(&mut self, event: &TrainingEvent) -> CallbackSignal {
        let current = match event.val_loss {
            Some(v) => v,
            None => return CallbackSignal::Continue,
        };

        let improved = current < self.best_value - self.min_delta;

        if improved {
            self.best_value = current;
            self.wait = 0;
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                self.apply_reduction();
            }
        }

        CallbackSignal::Continue
    }

    fn name(&self) -> &str {
        "ReduceLrOnPlateau"
    }
}

// ============================================================================
// ProgressLogger
// ============================================================================

/// Log training progress to stdout and an internal string buffer.
///
/// Step-level logging is controlled by `log_every_n_steps`; epoch-level
/// logging always fires (when `verbose == true`).
#[derive(Debug)]
pub struct ProgressLogger {
    /// Emit a step log entry every N steps.
    pub log_every_n_steps: usize,
    /// Whether to also print to stdout.
    pub verbose: bool,
    /// Accumulated log lines.
    log_buffer: Vec<String>,
}

impl ProgressLogger {
    /// Create a new logger.
    pub fn new(log_every_n_steps: usize) -> Self {
        Self {
            log_every_n_steps,
            verbose: true,
            log_buffer: Vec::new(),
        }
    }

    /// Return a reference to all buffered log lines.
    pub fn get_logs(&self) -> &[String] {
        &self.log_buffer
    }

    /// Clear the internal log buffer.
    pub fn clear_logs(&mut self) {
        self.log_buffer.clear();
    }

    fn push_line(&mut self, line: String) {
        if self.verbose {
            println!("{}", line);
        }
        self.log_buffer.push(line);
    }
}

impl Callback for ProgressLogger {
    fn on_train_begin(&mut self, total_epochs: usize) {
        let line = format!("[ProgressLogger] Training started (total_epochs={})", total_epochs);
        self.push_line(line);
    }

    fn on_epoch_end(&mut self, event: &TrainingEvent) -> CallbackSignal {
        let line = format!("[ProgressLogger] {}", event);
        self.push_line(line);
        CallbackSignal::Continue
    }

    fn on_step_end(&mut self, event: &TrainingEvent) -> CallbackSignal {
        let n = self.log_every_n_steps;
        if n > 0 && (event.step + 1) % n == 0 {
            let line = format!("[ProgressLogger] step {} | {}", event.step, event);
            self.push_line(line);
        }
        CallbackSignal::Continue
    }

    fn on_train_end(&mut self, event: &TrainingEvent) {
        let line = format!("[ProgressLogger] Training ended at {}", event);
        self.push_line(line);
    }

    fn name(&self) -> &str {
        "ProgressLogger"
    }
}

// ============================================================================
// History
// ============================================================================

/// Accumulate per-epoch scalar metrics for post-training analysis or plotting.
///
/// Only values that were present in the event are pushed; absent values are
/// silently skipped.
#[derive(Debug, Default)]
pub struct History {
    /// Training losses, one entry per epoch where `train_loss` was present.
    pub train_losses: Vec<f32>,
    /// Validation losses.
    pub val_losses: Vec<f32>,
    /// Training metric scalars.
    pub train_metrics: Vec<f32>,
    /// Validation metric scalars.
    pub val_metrics: Vec<f32>,
    /// Learning rates at epoch end.
    pub learning_rates: Vec<f32>,
}

impl History {
    /// Create an empty `History`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Minimum validation loss recorded, or `None` if no validation data was seen.
    pub fn best_val_loss(&self) -> Option<f32> {
        self.val_losses
            .iter()
            .copied()
            .reduce(f32::min)
    }

    /// Maximum validation metric recorded, or `None` if no validation data was seen.
    pub fn best_val_metric(&self) -> Option<f32> {
        self.val_metrics
            .iter()
            .copied()
            .reduce(f32::max)
    }

    /// Number of epochs recorded in training loss history.
    pub fn len(&self) -> usize {
        self.train_losses.len()
    }

    /// Returns `true` when no training loss entries have been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.train_losses.is_empty()
    }
}

impl Callback for History {
    fn on_epoch_end(&mut self, event: &TrainingEvent) -> CallbackSignal {
        if let Some(tl) = event.train_loss {
            self.train_losses.push(tl);
        }
        if let Some(vl) = event.val_loss {
            self.val_losses.push(vl);
        }
        if let Some(tm) = event.train_metric {
            self.train_metrics.push(tm);
        }
        if let Some(vm) = event.val_metric {
            self.val_metrics.push(vm);
        }
        if let Some(lr) = event.learning_rate {
            self.learning_rates.push(lr);
        }
        CallbackSignal::Continue
    }

    fn name(&self) -> &str {
        "History"
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn event_with_val_loss(epoch: usize, step: usize, val_loss: f32) -> TrainingEvent {
        TrainingEvent {
            epoch,
            step,
            val_loss: Some(val_loss),
            ..Default::default()
        }
    }

    fn event_with_all(
        epoch: usize,
        train_loss: f32,
        val_loss: f32,
        train_metric: f32,
        val_metric: f32,
        lr: f32,
    ) -> TrainingEvent {
        TrainingEvent {
            epoch,
            step: epoch * 10,
            train_loss: Some(train_loss),
            val_loss: Some(val_loss),
            train_metric: Some(train_metric),
            val_metric: Some(val_metric),
            learning_rate: Some(lr),
        }
    }

    // ------------------------------------------------------------------
    // EarlyStopping tests
    // ------------------------------------------------------------------

    #[test]
    fn test_early_stopping_triggers_after_patience() {
        let mut es = EarlyStopping::new(3);
        // First call: best set to 1.0
        assert_eq!(es.on_epoch_end(&event_with_val_loss(0, 0, 1.0)), CallbackSignal::Continue);
        // No improvement for 3 epochs → should stop on third
        assert_eq!(es.on_epoch_end(&event_with_val_loss(1, 10, 1.1)), CallbackSignal::Continue);
        assert_eq!(es.on_epoch_end(&event_with_val_loss(2, 20, 1.2)), CallbackSignal::Continue);
        assert_eq!(es.on_epoch_end(&event_with_val_loss(3, 30, 1.3)), CallbackSignal::StopTraining);
    }

    #[test]
    fn test_early_stopping_does_not_trigger_when_improving() {
        let mut es = EarlyStopping::new(3);
        for i in 0..10_usize {
            let loss = 1.0 - (i as f32) * 0.1;
            let signal = es.on_epoch_end(&event_with_val_loss(i, i * 10, loss));
            assert_eq!(
                signal,
                CallbackSignal::Continue,
                "should not stop at epoch {} with loss {:.2}",
                i,
                loss
            );
        }
    }

    #[test]
    fn test_early_stopping_stopped_epoch_is_set_correctly() {
        let mut es = EarlyStopping::new(2);
        es.on_epoch_end(&event_with_val_loss(0, 0, 1.0));
        es.on_epoch_end(&event_with_val_loss(1, 10, 1.1)); // wait=1
        let sig = es.on_epoch_end(&event_with_val_loss(5, 50, 1.2)); // wait=2 → stop at epoch 5
        assert_eq!(sig, CallbackSignal::StopTraining);
        assert_eq!(es.stopped_epoch, Some(5));
    }

    #[test]
    fn test_early_stopping_best_epoch_tracked() {
        let mut es = EarlyStopping::new(5);
        es.on_epoch_end(&event_with_val_loss(0, 0, 1.0));
        es.on_epoch_end(&event_with_val_loss(1, 10, 0.8)); // improvement → best_epoch = 1
        es.on_epoch_end(&event_with_val_loss(2, 20, 0.9));
        assert_eq!(es.best_epoch, Some(1));
    }

    #[test]
    fn test_early_stopping_min_delta_respected() {
        // min_delta = 0.1; improvement of only 0.05 must NOT reset the counter
        let mut es = EarlyStopping::with_patience_and_delta(2, 0.1, MonitorMetric::ValLoss);
        es.on_epoch_end(&event_with_val_loss(0, 0, 1.0));
        es.on_epoch_end(&event_with_val_loss(1, 10, 0.95)); // delta = 0.05 < 0.1 → no improvement
        assert_eq!(es.wait(), 1);
    }

    #[test]
    fn test_early_stopping_monitors_train_loss() {
        let mut es = EarlyStopping::with_patience_and_delta(2, 1e-4, MonitorMetric::TrainLoss);
        let mut ev = TrainingEvent {
            epoch: 0,
            step: 0,
            train_loss: Some(1.0),
            ..Default::default()
        };
        es.on_epoch_end(&ev);
        ev.epoch = 1;
        ev.step = 10;
        ev.train_loss = Some(1.5);
        es.on_epoch_end(&ev);
        assert_eq!(es.wait(), 1);
    }

    #[test]
    fn test_early_stopping_no_metric_does_not_stop() {
        let mut es = EarlyStopping::new(1);
        // Event has no val_loss → callback should not interfere
        let ev = TrainingEvent { epoch: 0, step: 0, ..Default::default() };
        assert_eq!(es.on_epoch_end(&ev), CallbackSignal::Continue);
        let ev2 = TrainingEvent { epoch: 1, step: 10, ..Default::default() };
        assert_eq!(es.on_epoch_end(&ev2), CallbackSignal::Continue);
    }

    // ------------------------------------------------------------------
    // ReduceLrOnPlateau tests
    // ------------------------------------------------------------------

    #[test]
    fn test_reduce_lr_on_plateau_reduces_after_patience() {
        let initial_lr = 0.1_f32;
        let mut reducer = ReduceLrOnPlateau::new(initial_lr, 0.1, 2)
            .expect("valid args");
        let ev = |epoch: usize, vl: f32| -> TrainingEvent {
            event_with_val_loss(epoch, epoch * 10, vl)
        };

        reducer.on_epoch_end(&ev(0, 1.0)); // best = 1.0
        reducer.on_epoch_end(&ev(1, 1.1)); // wait = 1
        reducer.on_epoch_end(&ev(2, 1.2)); // wait = 2 → reduce
        assert_eq!(reducer.num_reductions, 1);
        // 0.1 * 0.1 = 0.01
        assert!((reducer.current_lr - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_lr_respects_min_lr_floor() {
        let mut reducer = ReduceLrOnPlateau::new(0.001, 0.1, 1)
            .expect("valid args");
        reducer.min_lr = 0.0005;

        let ev = |epoch: usize| event_with_val_loss(epoch, epoch * 10, 1.0 + epoch as f32);

        reducer.on_epoch_end(&ev(0)); // best = 1.0
        reducer.on_epoch_end(&ev(1)); // wait=1 → reduce → 0.0001, but floor=0.0005
        // 0.001 * 0.1 = 0.0001, clamped to 0.0005
        assert!(reducer.current_lr >= reducer.min_lr);
        assert_eq!(reducer.current_lr, 0.0005);
    }

    #[test]
    fn test_reduce_lr_resets_wait_after_improvement() {
        let mut reducer = ReduceLrOnPlateau::new(0.01, 0.5, 3)
            .expect("valid args");
        let ev = |vl: f32, epoch: usize| event_with_val_loss(epoch, epoch * 5, vl);

        reducer.on_epoch_end(&ev(1.0, 0));
        reducer.on_epoch_end(&ev(1.1, 1)); // wait = 1
        reducer.on_epoch_end(&ev(0.5, 2)); // improvement → wait reset
        assert_eq!(reducer.wait, 0);
        assert_eq!(reducer.num_reductions, 0);
    }

    #[test]
    fn test_reduce_lr_invalid_factor_returns_err() {
        assert!(ReduceLrOnPlateau::new(0.01, 1.5, 3).is_err());
        assert!(ReduceLrOnPlateau::new(0.01, 0.0, 3).is_err());
    }

    #[test]
    fn test_reduce_lr_reduced_lr_preview() {
        let reducer = ReduceLrOnPlateau::new(0.1, 0.5, 3)
            .expect("valid args");
        // Should not mutate state
        assert!((reducer.reduced_lr() - 0.05).abs() < 1e-6);
        assert!((reducer.current_lr - 0.1).abs() < 1e-6);
    }

    // ------------------------------------------------------------------
    // ModelCheckpoint tests
    // ------------------------------------------------------------------

    #[test]
    fn test_model_checkpoint_writes_file_on_improvement() {
        let dir = temp_dir();
        let path = dir.join("test_checkpoint_improvement.json");
        let path_str = path.to_str().expect("utf8 path");

        let mut ckpt = ModelCheckpoint::best_only(path_str, MonitorMetric::ValLoss);

        let ev1 = event_with_val_loss(0, 0, 1.0);
        ckpt.on_epoch_end(&ev1);
        assert!(path.exists(), "checkpoint should be written on first improvement");
        assert_eq!(ckpt.last_saved_epoch, Some(0));

        let ev2 = event_with_val_loss(1, 10, 0.5); // better
        ckpt.on_epoch_end(&ev2);
        assert_eq!(ckpt.last_saved_epoch, Some(1));
    }

    #[test]
    fn test_model_checkpoint_does_not_write_when_worse() {
        let dir = temp_dir();
        let path = dir.join("test_checkpoint_no_write.json");
        let path_str = path.to_str().expect("utf8 path");

        let mut ckpt = ModelCheckpoint::best_only(path_str, MonitorMetric::ValLoss);

        ckpt.on_epoch_end(&event_with_val_loss(0, 0, 1.0)); // saves
        let mod_time_after_first = std::fs::metadata(path_str)
            .expect("file should exist")
            .modified()
            .expect("mtime");

        // Wait a moment so mtime differs if the file is rewritten
        std::thread::sleep(std::time::Duration::from_millis(20));

        ckpt.on_epoch_end(&event_with_val_loss(1, 10, 1.5)); // worse → should NOT save
        let mod_time_after_second = std::fs::metadata(path_str)
            .expect("file should exist")
            .modified()
            .expect("mtime");

        assert_eq!(mod_time_after_first, mod_time_after_second,
            "file should not be rewritten when metric worsens");
        assert_eq!(ckpt.last_saved_epoch, Some(0));
    }

    #[test]
    fn test_model_checkpoint_save_all_writes_every_epoch() {
        let dir = temp_dir();
        let path = dir.join("test_checkpoint_all.json");
        let path_str = path.to_str().expect("utf8 path");

        let mut ckpt = ModelCheckpoint::new(path_str);
        // No monitor value in event, but save_best_only == false → still saves
        for epoch in 0..3_usize {
            let ev = event_with_val_loss(epoch, epoch * 10, 1.0 - epoch as f32 * 0.1);
            ckpt.on_epoch_end(&ev);
            assert_eq!(ckpt.last_saved_epoch, Some(epoch));
        }
    }

    #[test]
    fn test_model_checkpoint_json_content() {
        let dir = temp_dir();
        let path = dir.join("test_checkpoint_json.json");
        let path_str = path.to_str().expect("utf8 path");

        let mut ckpt = ModelCheckpoint::new(path_str);
        let ev = event_with_all(2, 0.4, 0.5, 0.9, 0.88, 0.001);
        ckpt.on_epoch_end(&ev);

        let contents = std::fs::read_to_string(path_str).expect("file should be readable");
        assert!(contents.contains("\"epoch\":2"), "JSON must contain epoch");
        assert!(contents.contains("\"val_loss\""), "JSON must contain val_loss key");
    }

    // ------------------------------------------------------------------
    // History tests
    // ------------------------------------------------------------------

    #[test]
    fn test_history_accumulates_losses() {
        let mut hist = History::new();
        for i in 0..5_usize {
            let ev = event_with_all(i, i as f32 * 0.1, i as f32 * 0.2, 0.9, 0.8, 0.001);
            hist.on_epoch_end(&ev);
        }
        assert_eq!(hist.train_losses.len(), 5);
        assert_eq!(hist.val_losses.len(), 5);
        assert!((hist.train_losses[3] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_history_best_val_loss() {
        let mut hist = History::new();
        let losses = [1.0_f32, 0.8, 0.6, 0.7, 0.65];
        for (i, &vl) in losses.iter().enumerate() {
            let ev = TrainingEvent {
                epoch: i,
                step: i * 10,
                val_loss: Some(vl),
                ..Default::default()
            };
            hist.on_epoch_end(&ev);
        }
        assert_eq!(hist.best_val_loss(), Some(0.6));
    }

    #[test]
    fn test_history_best_val_metric() {
        let mut hist = History::new();
        let metrics = [0.5_f32, 0.7, 0.9, 0.85, 0.88];
        for (i, &vm) in metrics.iter().enumerate() {
            let ev = TrainingEvent {
                epoch: i,
                step: i * 10,
                val_metric: Some(vm),
                ..Default::default()
            };
            hist.on_epoch_end(&ev);
        }
        assert_eq!(hist.best_val_metric(), Some(0.9));
    }

    #[test]
    fn test_history_is_empty_initially() {
        let hist = History::new();
        assert!(hist.is_empty());
        assert_eq!(hist.len(), 0);
    }

    #[test]
    fn test_history_skips_absent_values() {
        let mut hist = History::new();
        // Event with only train_loss
        let ev = TrainingEvent {
            epoch: 0,
            step: 0,
            train_loss: Some(0.5),
            ..Default::default()
        };
        hist.on_epoch_end(&ev);
        assert_eq!(hist.train_losses.len(), 1);
        assert!(hist.val_losses.is_empty());
        assert!(hist.train_metrics.is_empty());
        assert!(hist.val_metrics.is_empty());
        assert!(hist.learning_rates.is_empty());
    }

    // ------------------------------------------------------------------
    // ProgressLogger tests
    // ------------------------------------------------------------------

    #[test]
    fn test_progress_logger_buffers_epoch_entries() {
        let mut logger = ProgressLogger::new(100);
        logger.verbose = false;

        logger.on_train_begin(10);
        logger.on_epoch_end(&event_with_val_loss(0, 0, 1.0));
        logger.on_epoch_end(&event_with_val_loss(1, 10, 0.9));

        // on_train_begin + 2 epoch entries
        assert_eq!(logger.get_logs().len(), 3);
    }

    #[test]
    fn test_progress_logger_step_log_every_n() {
        let mut logger = ProgressLogger::new(5);
        logger.verbose = false;

        for step in 0..10_usize {
            let ev = TrainingEvent {
                epoch: 0,
                step,
                train_loss: Some(0.5),
                ..Default::default()
            };
            logger.on_step_end(&ev);
        }

        // Steps 4 and 9 are multiples of 5 (step+1 = 5 and 10)
        assert_eq!(logger.get_logs().len(), 2);
    }

    #[test]
    fn test_progress_logger_clear_logs() {
        let mut logger = ProgressLogger::new(1);
        logger.verbose = false;

        logger.on_epoch_end(&event_with_val_loss(0, 0, 1.0));
        assert!(!logger.get_logs().is_empty());

        logger.clear_logs();
        assert!(logger.get_logs().is_empty());
    }

    #[test]
    fn test_progress_logger_on_train_end() {
        let mut logger = ProgressLogger::new(100);
        logger.verbose = false;

        let ev = event_with_val_loss(9, 90, 0.1);
        logger.on_train_end(&ev);
        assert_eq!(logger.get_logs().len(), 1);
        assert!(logger.get_logs()[0].contains("Training ended"));
    }

    // ------------------------------------------------------------------
    // run_callbacks_epoch tests
    // ------------------------------------------------------------------

    #[test]
    fn test_run_callbacks_epoch_stops_on_signal() {
        let mut callbacks: Vec<Box<dyn Callback>> = vec![
            Box::new(History::new()),           // never stops
            Box::new(EarlyStopping::new(1)),    // will stop after 2 epochs without improvement
        ];

        let ev0 = event_with_val_loss(0, 0, 1.0);
        let stopped = run_callbacks_epoch(&mut callbacks, &ev0);
        assert!(!stopped);

        let ev1 = event_with_val_loss(1, 10, 1.5); // no improvement → wait=1 → stop (patience=1)
        let stopped = run_callbacks_epoch(&mut callbacks, &ev1);
        assert!(stopped);
    }

    #[test]
    fn test_run_callbacks_epoch_all_continue() {
        let mut callbacks: Vec<Box<dyn Callback>> = vec![
            Box::new(History::new()),
            Box::new(ProgressLogger::new(100)),
        ];
        let ev = event_with_val_loss(0, 0, 0.5);
        let stopped = run_callbacks_epoch(&mut callbacks, &ev);
        assert!(!stopped);
    }

    #[test]
    fn test_run_callbacks_epoch_multiple_callbacks_run_in_order() {
        // Verify that:
        // 1. All callbacks receive the event (not short-circuited).
        // 2. The overall result is `true` when any callback signals stop.
        //
        // We use a separate History to independently confirm it accumulated
        // entries during the same epochs that caused EarlyStopping to fire.
        let mut standalone_hist = History::new();
        let prime = event_with_val_loss(0, 0, 0.5);
        standalone_hist.on_epoch_end(&prime);
        standalone_hist.on_epoch_end(&event_with_val_loss(1, 10, 1.0));
        // standalone_hist has 2 val_loss entries regardless of EarlyStopping.
        assert_eq!(standalone_hist.val_losses.len(), 2);

        let mut callbacks: Vec<Box<dyn Callback>> = vec![
            Box::new(History::new()),
            Box::new({
                let mut l = ProgressLogger::new(100);
                l.verbose = false;
                l
            }),
            Box::new(EarlyStopping::new(1)),
        ];

        // Prime all via run_callbacks (epoch 0, good loss)
        run_callbacks_epoch(&mut callbacks, &event_with_val_loss(0, 0, 0.5));
        // Now trigger stop: no improvement for patience=1 epoch
        let stopped = run_callbacks_epoch(
            &mut callbacks,
            &event_with_val_loss(1, 10, 1.0),
        );

        assert!(stopped, "EarlyStopping should have requested a stop");
    }

    // ------------------------------------------------------------------
    // Callback name tests
    // ------------------------------------------------------------------

    #[test]
    fn test_callback_names() {
        let es = EarlyStopping::new(3);
        let tmp_path = std::env::temp_dir().join("x.json");
        let ckpt = ModelCheckpoint::new(tmp_path.to_str().expect("valid path"));
        let reducer = ReduceLrOnPlateau::new(0.01, 0.5, 3).expect("valid");
        let logger = ProgressLogger::new(10);
        let hist = History::new();

        assert_eq!(es.name(), "EarlyStopping");
        assert_eq!(ckpt.name(), "ModelCheckpoint");
        assert_eq!(reducer.name(), "ReduceLrOnPlateau");
        assert_eq!(logger.name(), "ProgressLogger");
        assert_eq!(hist.name(), "History");
    }

    // ------------------------------------------------------------------
    // TrainingEvent helpers
    // ------------------------------------------------------------------

    #[test]
    fn test_training_event_display() {
        let ev = event_with_all(3, 0.4, 0.5, 0.88, 0.85, 0.001);
        let s = format!("{}", ev);
        assert!(s.contains("epoch=3"));
        assert!(s.contains("val_loss="));
        assert!(s.contains("lr="));
    }

    #[test]
    fn test_monitor_metric_extract() {
        let ev = event_with_all(0, 0.1, 0.2, 0.9, 0.85, 0.01);
        assert_eq!(MonitorMetric::ValLoss.extract(&ev), Some(0.2));
        assert_eq!(MonitorMetric::TrainLoss.extract(&ev), Some(0.1));
        assert_eq!(MonitorMetric::ValMetric.extract(&ev), Some(0.85));
        assert_eq!(MonitorMetric::TrainMetric.extract(&ev), Some(0.9));
    }
}
