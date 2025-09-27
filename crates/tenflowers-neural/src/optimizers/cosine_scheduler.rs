//! Cosine Annealing Learning Rate Scheduler
//!
//! Implements Cosine Annealing with Warm Restarts (SGDR), a popular learning rate
//! scheduling technique that uses a cosine function to gradually decrease the learning
//! rate and includes periodic "warm restarts" to escape local minima.
//!
//! Key benefits:
//! - Smooth learning rate decay using cosine function
//! - Periodic restarts help escape local minima
//! - Often leads to better final performance
//! - Well-suited for modern deep learning architectures
//! - Reduces need for manual learning rate tuning
//!
//! References:
//! - "SGDR: Stochastic Gradient Descent with Warm Restarts" (Loshchilov & Hutter, 2016)
//! - "Snapshot Ensembles: Train one, get M for free" (Huang et al., 2017)

use std::f32::consts::PI;

/// Configuration for Cosine Annealing with Warm Restarts
#[derive(Debug, Clone)]
pub struct CosineSchedulerConfig {
    /// Initial learning rate
    pub initial_lr: f32,
    /// Minimum learning rate (never goes below this)
    pub min_lr: f32,
    /// Number of iterations for the first restart cycle
    pub t_0: u32,
    /// Factor by which to multiply the cycle length after each restart (default: 1)
    pub t_mult: u32,
    /// Factor by which to multiply the learning rate after each restart (default: 1.0)
    pub eta_mult: f32,
    /// Whether to restart at the end of each cycle (default: true)
    pub restart: bool,
}

impl Default for CosineSchedulerConfig {
    fn default() -> Self {
        Self {
            initial_lr: 0.001,
            min_lr: 0.0,
            t_0: 10,
            t_mult: 2,
            eta_mult: 1.0,
            restart: true,
        }
    }
}

/// Cosine Annealing Learning Rate Scheduler with Warm Restarts
///
/// This scheduler implements the SGDR (Stochastic Gradient Descent with Warm Restarts)
/// algorithm, which uses a cosine function to schedule the learning rate with periodic
/// restarts to escape local minima.
///
/// # Example
/// ```rust,ignore
/// use tenflowers_neural::optimizers::{CosineScheduler, CosineSchedulerConfig};
///
/// let config = CosineSchedulerConfig {
///     initial_lr: 0.1,
///     min_lr: 0.001,
///     t_0: 50,      // First cycle lasts 50 epochs
///     t_mult: 2,    // Each cycle is twice as long as the previous
///     eta_mult: 0.5, // Reduce max LR by half after each restart
///     restart: true,
/// };
///
/// let mut scheduler = CosineScheduler::new(config);
///
/// // Use in training loop
/// for epoch in 0..200 {
///     let current_lr = scheduler.get_lr();
///     optimizer.set_learning_rate(current_lr);
///     
///     // ... training code ...
///     
///     scheduler.step();
/// }
/// ```
pub struct CosineScheduler {
    /// Scheduler configuration
    config: CosineSchedulerConfig,
    /// Current step/iteration
    current_step: u32,
    /// Current cycle length
    current_t: u32,
    /// Steps completed in current cycle
    steps_in_cycle: u32,
    /// Current maximum learning rate (may decrease after restarts)
    current_max_lr: f32,
    /// Number of restarts completed
    restart_count: u32,
}

impl CosineScheduler {
    /// Create a new cosine scheduler with default configuration
    pub fn new(config: CosineSchedulerConfig) -> Self {
        Self {
            current_max_lr: config.initial_lr,
            current_t: config.t_0,
            config,
            current_step: 0,
            steps_in_cycle: 0,
            restart_count: 0,
        }
    }

    /// Create scheduler with simple parameters (most common use case)
    pub fn simple(initial_lr: f32, min_lr: f32, cycle_length: u32) -> Self {
        let config = CosineSchedulerConfig {
            initial_lr,
            min_lr,
            t_0: cycle_length,
            t_mult: 1,
            eta_mult: 1.0,
            restart: true,
        };
        Self::new(config)
    }

    /// Create scheduler without restarts (standard cosine annealing)
    pub fn without_restarts(initial_lr: f32, min_lr: f32, total_steps: u32) -> Self {
        let config = CosineSchedulerConfig {
            initial_lr,
            min_lr,
            t_0: total_steps,
            t_mult: 1,
            eta_mult: 1.0,
            restart: false,
        };
        Self::new(config)
    }

    /// Get the current learning rate
    pub fn get_lr(&self) -> f32 {
        if !self.config.restart && self.current_step >= self.config.t_0 {
            // If no restarts and we've completed the schedule, return min_lr
            return self.config.min_lr;
        }

        let progress = self.steps_in_cycle as f32 / self.current_t as f32;
        let cosine_factor = 0.5 * (1.0 + (PI * progress).cos());

        self.config.min_lr + (self.current_max_lr - self.config.min_lr) * cosine_factor
    }

    /// Advance to the next step
    pub fn step(&mut self) {
        self.current_step += 1;
        self.steps_in_cycle += 1;

        // Check if we need to restart
        if self.config.restart && self.steps_in_cycle >= self.current_t {
            self.restart();
        }
    }

    /// Perform a restart
    fn restart(&mut self) {
        self.restart_count += 1;
        self.steps_in_cycle = 0;

        // Update cycle length for next restart
        self.current_t *= self.config.t_mult;

        // Update maximum learning rate for next cycle
        self.current_max_lr *= self.config.eta_mult;
    }

    /// Get the current step number
    pub fn current_step(&self) -> u32 {
        self.current_step
    }

    /// Get the current cycle length
    pub fn current_cycle_length(&self) -> u32 {
        self.current_t
    }

    /// Get steps completed in current cycle
    pub fn steps_in_current_cycle(&self) -> u32 {
        self.steps_in_cycle
    }

    /// Get the number of restarts completed
    pub fn restart_count(&self) -> u32 {
        self.restart_count
    }

    /// Get current maximum learning rate
    pub fn current_max_lr(&self) -> f32 {
        self.current_max_lr
    }

    /// Check if we're at the beginning of a new cycle
    pub fn is_restart_step(&self) -> bool {
        self.steps_in_cycle == 0 && self.current_step > 0
    }

    /// Manually trigger a restart (useful for snapshot ensembles)
    pub fn manual_restart(&mut self) {
        if self.config.restart {
            self.restart();
        }
    }

    /// Reset scheduler to initial state
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.steps_in_cycle = 0;
        self.current_t = self.config.t_0;
        self.current_max_lr = self.config.initial_lr;
        self.restart_count = 0;
    }

    /// Get progress through current cycle (0.0 to 1.0)
    pub fn cycle_progress(&self) -> f32 {
        if self.current_t == 0 {
            return 1.0;
        }
        self.steps_in_cycle as f32 / self.current_t as f32
    }

    /// Update configuration (useful for dynamic scheduling)
    pub fn update_config(&mut self, config: CosineSchedulerConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn config(&self) -> &CosineSchedulerConfig {
        &self.config
    }
}

/// Utility function to create a learning rate schedule for a given number of epochs
///
/// This is a convenience function for common training scenarios where you know
/// the total number of training epochs and want to set up reasonable defaults.
pub fn create_cosine_schedule_for_epochs(
    initial_lr: f32,
    min_lr: f32,
    total_epochs: u32,
    steps_per_epoch: u32,
    num_cycles: u32,
) -> CosineScheduler {
    let total_steps = total_epochs * steps_per_epoch;
    let cycle_length = total_steps / num_cycles;

    let config = CosineSchedulerConfig {
        initial_lr,
        min_lr,
        t_0: cycle_length,
        t_mult: 1,
        eta_mult: 0.8, // Slightly reduce max LR after each restart
        restart: num_cycles > 1,
    };

    CosineScheduler::new(config)
}

/// Snapshot ensemble helper
///
/// When using cosine annealing with warm restarts, model snapshots taken at the
/// end of each cycle (when LR is minimum) often perform well individually and
/// can be combined into an ensemble.
pub struct SnapshotEnsemble<T> {
    /// Model snapshots collected at restart points
    snapshots: Vec<T>,
    /// Maximum number of snapshots to keep
    max_snapshots: usize,
}

impl<T: Clone> SnapshotEnsemble<T> {
    /// Create a new snapshot ensemble
    pub fn new(max_snapshots: usize) -> Self {
        Self {
            snapshots: Vec::new(),
            max_snapshots,
        }
    }

    /// Add a snapshot to the ensemble
    pub fn add_snapshot(&mut self, model: T) {
        self.snapshots.push(model);

        // Keep only the most recent snapshots
        if self.snapshots.len() > self.max_snapshots {
            self.snapshots.remove(0);
        }
    }

    /// Get all snapshots
    pub fn snapshots(&self) -> &[T] {
        &self.snapshots
    }

    /// Get the number of snapshots
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Check if ensemble is empty
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Clear all snapshots
    pub fn clear(&mut self) {
        self.snapshots.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_scheduler_creation() {
        let config = CosineSchedulerConfig::default();
        let scheduler = CosineScheduler::new(config);

        assert_eq!(scheduler.current_step(), 0);
        assert_eq!(scheduler.restart_count(), 0);
        assert_eq!(scheduler.get_lr(), 0.001); // Should return initial LR
    }

    #[test]
    fn test_simple_cosine_scheduler() {
        let scheduler = CosineScheduler::simple(0.1, 0.001, 10);

        assert_eq!(scheduler.current_max_lr(), 0.1);
        assert_eq!(scheduler.current_cycle_length(), 10);
        assert!(scheduler.config().restart);
    }

    #[test]
    fn test_cosine_scheduler_without_restarts() {
        let scheduler = CosineScheduler::without_restarts(0.1, 0.001, 100);

        assert!(!scheduler.config().restart);
        assert_eq!(scheduler.current_cycle_length(), 100);
    }

    #[test]
    fn test_learning_rate_progression() {
        let mut scheduler = CosineScheduler::simple(1.0, 0.0, 4);

        // At step 0, should be at maximum
        assert!((scheduler.get_lr() - 1.0).abs() < 1e-6);

        scheduler.step();
        // At step 1 (progress = 0.25), LR should be between max and min
        let lr1 = scheduler.get_lr();
        assert!(lr1 > 0.0 && lr1 < 1.0);

        scheduler.step();
        scheduler.step();
        // At step 3 (progress = 0.75), LR should be lower
        let lr3 = scheduler.get_lr();
        assert!(lr3 < lr1);

        scheduler.step();
        // After step 4, should restart to maximum (if restart enabled)
        if scheduler.config().restart {
            assert!((scheduler.get_lr() - 1.0).abs() < 1e-6);
            assert_eq!(scheduler.restart_count(), 1);
        }
    }

    #[test]
    fn test_cycle_progress() {
        let mut scheduler = CosineScheduler::simple(1.0, 0.0, 4);

        assert_eq!(scheduler.cycle_progress(), 0.0);

        scheduler.step();
        assert!((scheduler.cycle_progress() - 0.25).abs() < 1e-6);

        scheduler.step();
        assert!((scheduler.cycle_progress() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_restart_detection() {
        let mut scheduler = CosineScheduler::simple(1.0, 0.0, 2);

        assert!(!scheduler.is_restart_step());

        scheduler.step();
        assert!(!scheduler.is_restart_step());

        scheduler.step();
        // After completing cycle, we should be at the start of a new cycle
        assert!(scheduler.is_restart_step());

        scheduler.step();
        assert!(!scheduler.is_restart_step());
    }

    #[test]
    fn test_t_mult_progression() {
        let config = CosineSchedulerConfig {
            initial_lr: 1.0,
            min_lr: 0.0,
            t_0: 2,
            t_mult: 2,
            eta_mult: 1.0,
            restart: true,
        };
        let mut scheduler = CosineScheduler::new(config);

        assert_eq!(scheduler.current_cycle_length(), 2);

        // Complete first cycle
        scheduler.step();
        scheduler.step();

        // Should double the cycle length
        assert_eq!(scheduler.current_cycle_length(), 4);
    }

    #[test]
    fn test_eta_mult_progression() {
        let config = CosineSchedulerConfig {
            initial_lr: 1.0,
            min_lr: 0.0,
            t_0: 2,
            t_mult: 1,
            eta_mult: 0.5,
            restart: true,
        };
        let mut scheduler = CosineScheduler::new(config);

        assert_eq!(scheduler.current_max_lr(), 1.0);

        // Complete first cycle
        scheduler.step();
        scheduler.step();

        // Max LR should be reduced by eta_mult
        assert!((scheduler.current_max_lr() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_manual_restart() {
        let mut scheduler = CosineScheduler::simple(1.0, 0.0, 10);

        scheduler.step();
        scheduler.step();
        assert_eq!(scheduler.steps_in_current_cycle(), 2);

        scheduler.manual_restart();
        assert_eq!(scheduler.steps_in_current_cycle(), 0);
        assert_eq!(scheduler.restart_count(), 1);
    }

    #[test]
    fn test_reset_functionality() {
        let mut scheduler = CosineScheduler::simple(1.0, 0.0, 5);

        // Make some progress
        scheduler.step();
        scheduler.step();
        scheduler.manual_restart();

        assert!(scheduler.current_step() > 0);
        assert!(scheduler.restart_count() > 0);

        scheduler.reset();

        assert_eq!(scheduler.current_step(), 0);
        assert_eq!(scheduler.restart_count(), 0);
        assert_eq!(scheduler.steps_in_current_cycle(), 0);
        assert_eq!(scheduler.current_cycle_length(), 5);
        assert_eq!(scheduler.current_max_lr(), 1.0);
    }

    #[test]
    fn test_create_cosine_schedule_for_epochs() {
        let scheduler = create_cosine_schedule_for_epochs(0.1, 0.001, 100, 1000, 5);

        assert_eq!(scheduler.config().initial_lr, 0.1);
        assert_eq!(scheduler.config().min_lr, 0.001);
        assert_eq!(scheduler.current_cycle_length(), 20000); // 100 * 1000 / 5
        assert!(scheduler.config().restart);
    }

    #[test]
    fn test_snapshot_ensemble() {
        let mut ensemble = SnapshotEnsemble::new(3);

        assert!(ensemble.is_empty());
        assert_eq!(ensemble.len(), 0);

        ensemble.add_snapshot("model1");
        ensemble.add_snapshot("model2");
        ensemble.add_snapshot("model3");

        assert_eq!(ensemble.len(), 3);
        assert_eq!(ensemble.snapshots(), &["model1", "model2", "model3"]);

        // Adding beyond capacity should remove oldest
        ensemble.add_snapshot("model4");
        assert_eq!(ensemble.len(), 3);
        assert_eq!(ensemble.snapshots(), &["model2", "model3", "model4"]);

        ensemble.clear();
        assert!(ensemble.is_empty());
    }

    #[test]
    fn test_no_restart_completion() {
        let mut scheduler = CosineScheduler::without_restarts(1.0, 0.1, 3);

        scheduler.step();
        scheduler.step();
        scheduler.step();
        // Should have completed the schedule

        let lr = scheduler.get_lr();
        assert!((lr - 0.1).abs() < 1e-6); // Should be at min_lr
    }

    #[test]
    fn test_config_update() {
        let mut scheduler = CosineScheduler::simple(1.0, 0.0, 10);

        let new_config = CosineSchedulerConfig {
            initial_lr: 0.5,
            min_lr: 0.05,
            t_0: 20,
            t_mult: 3,
            eta_mult: 0.8,
            restart: false,
        };

        scheduler.update_config(new_config);

        assert_eq!(scheduler.config().initial_lr, 0.5);
        assert_eq!(scheduler.config().t_0, 20);
        assert!(!scheduler.config().restart);
    }
}
