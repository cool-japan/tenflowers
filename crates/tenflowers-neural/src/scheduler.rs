//! Learning Rate Scheduling for optimizers
//!
//! This module provides various learning rate scheduling strategies commonly used
//! in deep learning training, such as step decay, exponential decay, cosine annealing, etc.

use std::f32::consts::PI;

/// Trait for learning rate schedulers
pub trait LearningRateScheduler {
    /// Get the learning rate for the given step/epoch
    fn get_lr(&self, step: usize) -> f32;

    /// Update scheduler state (for schedulers that need to track state)
    fn step(&mut self) {}
}

/// Constant learning rate (no scheduling)
#[derive(Debug, Clone)]
pub struct ConstantLR {
    lr: f32,
}

impl ConstantLR {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl LearningRateScheduler for ConstantLR {
    fn get_lr(&self, _step: usize) -> f32 {
        self.lr
    }
}

/// Step decay: multiply lr by gamma every step_size steps
#[derive(Debug, Clone)]
pub struct StepLR {
    initial_lr: f32,
    step_size: usize,
    gamma: f32,
}

impl StepLR {
    pub fn new(initial_lr: f32, step_size: usize, gamma: f32) -> Self {
        Self {
            initial_lr,
            step_size,
            gamma,
        }
    }
}

impl LearningRateScheduler for StepLR {
    fn get_lr(&self, step: usize) -> f32 {
        let epochs = step / self.step_size;
        self.initial_lr * self.gamma.powi(epochs as i32)
    }
}

/// Exponential decay: lr = initial_lr * gamma^step
#[derive(Debug, Clone)]
pub struct ExponentialLR {
    initial_lr: f32,
    gamma: f32,
}

impl ExponentialLR {
    pub fn new(initial_lr: f32, gamma: f32) -> Self {
        Self { initial_lr, gamma }
    }
}

impl LearningRateScheduler for ExponentialLR {
    fn get_lr(&self, step: usize) -> f32 {
        self.initial_lr * self.gamma.powi(step as i32)
    }
}

/// Cosine annealing: lr oscillates following cosine function
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    initial_lr: f32,
    min_lr: f32,
    t_max: usize, // maximum number of iterations
}

impl CosineAnnealingLR {
    pub fn new(initial_lr: f32, t_max: usize) -> Self {
        Self {
            initial_lr,
            min_lr: 0.0,
            t_max,
        }
    }

    pub fn with_min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }
}

impl LearningRateScheduler for CosineAnnealingLR {
    fn get_lr(&self, step: usize) -> f32 {
        let step = step % self.t_max;
        self.min_lr
            + (self.initial_lr - self.min_lr) * (1.0 + (PI * step as f32 / self.t_max as f32).cos())
                / 2.0
    }
}

/// Polynomial decay: lr decays following polynomial function
#[derive(Debug, Clone)]
pub struct PolynomialLR {
    initial_lr: f32,
    final_lr: f32,
    max_steps: usize,
    power: f32,
}

impl PolynomialLR {
    pub fn new(initial_lr: f32, max_steps: usize) -> Self {
        Self {
            initial_lr,
            final_lr: 0.0,
            max_steps,
            power: 1.0,
        }
    }

    pub fn with_final_lr(mut self, final_lr: f32) -> Self {
        self.final_lr = final_lr;
        self
    }

    pub fn with_power(mut self, power: f32) -> Self {
        self.power = power;
        self
    }
}

impl LearningRateScheduler for PolynomialLR {
    fn get_lr(&self, step: usize) -> f32 {
        if step >= self.max_steps {
            return self.final_lr;
        }

        let decay_rate = (1.0 - step as f32 / self.max_steps as f32).powf(self.power);
        (self.initial_lr - self.final_lr) * decay_rate + self.final_lr
    }
}

/// Linear warmup followed by cosine decay
#[derive(Debug, Clone)]
pub struct WarmupCosineDecayLR {
    initial_lr: f32,
    peak_lr: f32,
    min_lr: f32,
    warmup_steps: usize,
    total_steps: usize,
}

impl WarmupCosineDecayLR {
    pub fn new(peak_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            initial_lr: 0.0,
            peak_lr,
            min_lr: 0.0,
            warmup_steps,
            total_steps,
        }
    }

    pub fn with_initial_lr(mut self, initial_lr: f32) -> Self {
        self.initial_lr = initial_lr;
        self
    }

    pub fn with_min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }
}

impl LearningRateScheduler for WarmupCosineDecayLR {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            let progress = step as f32 / self.warmup_steps as f32;
            self.initial_lr + (self.peak_lr - self.initial_lr) * progress
        } else if step >= self.total_steps {
            self.min_lr
        } else {
            // Cosine decay
            let decay_steps = self.total_steps - self.warmup_steps;
            let current_step = step - self.warmup_steps;
            let progress = current_step as f32 / decay_steps as f32;

            self.min_lr + (self.peak_lr - self.min_lr) * (1.0 + (PI * progress).cos()) / 2.0
        }
    }
}

/// Reduce learning rate on plateau (requires manual triggering)
#[derive(Debug, Clone)]
pub struct ReduceLROnPlateau {
    initial_lr: f32,
    current_lr: f32,
    factor: f32,
    patience: usize,
    min_lr: f32,
    no_improvement_count: usize,
    best_metric: Option<f32>,
    mode: PlateauMode,
}

#[derive(Debug, Clone, Copy)]
pub enum PlateauMode {
    Min, // Reduce when metric stops decreasing (e.g., loss)
    Max, // Reduce when metric stops increasing (e.g., accuracy)
}

impl ReduceLROnPlateau {
    pub fn new(initial_lr: f32, factor: f32, patience: usize) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            factor,
            patience,
            min_lr: 0.0,
            no_improvement_count: 0,
            best_metric: None,
            mode: PlateauMode::Min,
        }
    }

    pub fn with_min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }

    pub fn with_mode(mut self, mode: PlateauMode) -> Self {
        self.mode = mode;
        self
    }

    /// Call this method with the current metric value to update the scheduler
    pub fn step_with_metric(&mut self, metric: f32) {
        let is_better = match self.mode {
            PlateauMode::Min => self.best_metric.map_or(true, |best| metric < best),
            PlateauMode::Max => self.best_metric.map_or(true, |best| metric > best),
        };

        if is_better {
            self.best_metric = Some(metric);
            self.no_improvement_count = 0;
        } else {
            self.no_improvement_count += 1;

            if self.no_improvement_count >= self.patience {
                self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
                self.no_improvement_count = 0;
            }
        }
    }

    pub fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.no_improvement_count = 0;
        self.best_metric = None;
    }
}

impl LearningRateScheduler for ReduceLROnPlateau {
    fn get_lr(&self, _step: usize) -> f32 {
        self.current_lr
    }
}

/// One Cycle Learning Rate Schedule
///
/// Implements the "1cycle" learning rate policy described in:
/// "A disciplined approach to neural network hyper-parameters" by Leslie Smith
/// https://arxiv.org/abs/1803.09820
#[derive(Debug, Clone)]
pub struct OneCycleLR {
    max_lr: f32,
    min_lr: f32,
    total_steps: usize,
    pct_start: f32,          // Percentage of cycle spent in increasing phase
    anneal_strategy: String, // 'cos' or 'linear'
    div_factor: f32,         // Initial LR = max_lr / div_factor
    final_div_factor: f32,   // Final LR = max_lr / (div_factor * final_div_factor)
}

impl OneCycleLR {
    /// Create a new OneCycleLR scheduler
    ///
    /// # Arguments
    /// * `max_lr` - Maximum learning rate
    /// * `total_steps` - Total number of training steps
    pub fn new(max_lr: f32, total_steps: usize) -> Self {
        Self {
            max_lr,
            min_lr: max_lr / 25.0, // Default div_factor = 25
            total_steps,
            pct_start: 0.3, // 30% of cycle spent increasing
            anneal_strategy: "cos".to_string(),
            div_factor: 25.0,
            final_div_factor: 1e4,
        }
    }

    /// Set the percentage of cycle spent in increasing phase
    pub fn with_pct_start(mut self, pct_start: f32) -> Self {
        self.pct_start = pct_start;
        self
    }

    /// Set annealing strategy ('cos' or 'linear')
    pub fn with_anneal_strategy(mut self, strategy: &str) -> Self {
        self.anneal_strategy = strategy.to_string();
        self
    }

    /// Set division factors for initial and final learning rates
    pub fn with_div_factors(mut self, div_factor: f32, final_div_factor: f32) -> Self {
        self.div_factor = div_factor;
        self.final_div_factor = final_div_factor;
        self.min_lr = self.max_lr / div_factor;
        self
    }

    fn cosine_annealing(&self, start_lr: f32, end_lr: f32, pct: f32) -> f32 {
        let cos_out = (1.0 + (PI * pct).cos()) / 2.0;
        end_lr + (start_lr - end_lr) * cos_out
    }

    fn linear_annealing(&self, start_lr: f32, end_lr: f32, pct: f32) -> f32 {
        start_lr + (end_lr - start_lr) * pct
    }
}

impl LearningRateScheduler for OneCycleLR {
    fn get_lr(&self, step: usize) -> f32 {
        if step >= self.total_steps {
            return self.max_lr / (self.div_factor * self.final_div_factor);
        }

        let step_ratio = step as f32 / self.total_steps as f32;
        let increase_phase_end = self.pct_start;

        if step_ratio <= increase_phase_end {
            // Increasing phase: from min_lr to max_lr
            let pct = step_ratio / increase_phase_end;
            if self.anneal_strategy == "cos" {
                // For cosine: when pct=0, we want min_lr; when pct=1, we want max_lr
                // cosine_annealing(start, end, pct) returns start when pct=0, end when pct=1
                self.cosine_annealing(self.min_lr, self.max_lr, pct)
            } else {
                self.linear_annealing(self.min_lr, self.max_lr, pct)
            }
        } else {
            // Decreasing phase: from max_lr to final_lr
            let final_lr = self.max_lr / (self.div_factor * self.final_div_factor);
            let pct = (step_ratio - increase_phase_end) / (1.0 - increase_phase_end);

            if self.anneal_strategy == "cos" {
                self.cosine_annealing(self.max_lr, final_lr, pct)
            } else {
                self.linear_annealing(self.max_lr, final_lr, pct)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_lr() {
        let scheduler = ConstantLR::new(0.001);
        assert_eq!(scheduler.get_lr(0), 0.001);
        assert_eq!(scheduler.get_lr(100), 0.001);
        assert_eq!(scheduler.get_lr(1000), 0.001);
    }

    #[test]
    fn test_step_lr() {
        let scheduler = StepLR::new(0.1, 10, 0.1);
        assert_eq!(scheduler.get_lr(0), 0.1);
        assert_eq!(scheduler.get_lr(5), 0.1);
        assert!((scheduler.get_lr(10) - 0.01).abs() < 1e-6); // 0.1 * 0.1^1
        assert!((scheduler.get_lr(20) - 0.001).abs() < 1e-6); // 0.1 * 0.1^2
    }

    #[test]
    fn test_exponential_lr() {
        let scheduler = ExponentialLR::new(0.1, 0.9);
        assert_eq!(scheduler.get_lr(0), 0.1);
        assert!((scheduler.get_lr(1) - 0.09).abs() < 1e-6); // 0.1 * 0.9^1
        assert!((scheduler.get_lr(2) - 0.081).abs() < 1e-6); // 0.1 * 0.9^2
    }

    #[test]
    fn test_cosine_annealing_lr() {
        let scheduler = CosineAnnealingLR::new(0.1, 100);
        assert_eq!(scheduler.get_lr(0), 0.1);
        let mid_lr = scheduler.get_lr(50);
        assert!((mid_lr - 0.05).abs() < 1e-6); // Should be halfway between max and min
        assert!((scheduler.get_lr(100) - 0.1).abs() < 1e-5); // Should return to max at end (cycle restarts)
    }

    #[test]
    fn test_polynomial_lr() {
        let scheduler = PolynomialLR::new(0.1, 100).with_final_lr(0.01);
        assert_eq!(scheduler.get_lr(0), 0.1);
        assert_eq!(scheduler.get_lr(100), 0.01);
        assert_eq!(scheduler.get_lr(200), 0.01); // Beyond max_steps

        // Check intermediate value
        let mid_lr = scheduler.get_lr(50);
        assert!(mid_lr > 0.01 && mid_lr < 0.1);
    }

    #[test]
    fn test_warmup_cosine_decay_lr() {
        let scheduler = WarmupCosineDecayLR::new(0.1, 10, 100);

        // During warmup
        assert_eq!(scheduler.get_lr(0), 0.0);
        assert!((scheduler.get_lr(5) - 0.05).abs() < 1e-6); // Half of warmup
        assert_eq!(scheduler.get_lr(10), 0.1); // End of warmup

        // During decay
        let mid_decay_lr = scheduler.get_lr(55); // Halfway through decay
        assert!(mid_decay_lr < 0.1 && mid_decay_lr > 0.0);

        // After total steps
        assert_eq!(scheduler.get_lr(200), 0.0);
    }

    #[test]
    fn test_reduce_lr_on_plateau() {
        let mut scheduler = ReduceLROnPlateau::new(0.1, 0.5, 2);

        // Initial LR
        assert_eq!(scheduler.get_lr(0), 0.1);

        // Improving metrics
        scheduler.step_with_metric(1.0);
        assert_eq!(scheduler.get_lr(0), 0.1);

        scheduler.step_with_metric(0.8);
        assert_eq!(scheduler.get_lr(0), 0.1);

        // Plateau starts
        scheduler.step_with_metric(0.9); // Worse than 0.8
        assert_eq!(scheduler.get_lr(0), 0.1); // Still patience

        scheduler.step_with_metric(0.85); // Still worse
        assert_eq!(scheduler.get_lr(0), 0.05); // Should reduce now

        // Reset test
        scheduler.reset();
        assert_eq!(scheduler.get_lr(0), 0.1);
    }

    #[test]
    fn test_one_cycle_lr() {
        let scheduler = OneCycleLR::new(0.1, 100).with_pct_start(0.3);

        // Start should be min_lr
        let start_lr = scheduler.get_lr(0);
        assert!((start_lr - 0.004).abs() < 1e-6); // 0.1 / 25

        // At 30% should be max_lr
        let peak_lr = scheduler.get_lr(30);
        assert!((peak_lr - 0.1).abs() < 1e-5);

        // At end should be very low
        let final_lr = scheduler.get_lr(99);
        assert!(final_lr < 0.001);

        // Beyond total steps should be final_lr
        let beyond_lr = scheduler.get_lr(200);
        assert!(beyond_lr < 0.001);
    }

    #[test]
    fn test_one_cycle_lr_linear() {
        let scheduler = OneCycleLR::new(0.1, 100)
            .with_pct_start(0.5)
            .with_anneal_strategy("linear");

        // At 25% should be halfway between min and max in increasing phase
        let quarter_lr = scheduler.get_lr(25);
        let expected = 0.004 + (0.1 - 0.004) * 0.5; // Linear interpolation
        assert!((quarter_lr - expected).abs() < 1e-5);

        // At 75% should be halfway between max and final in decreasing phase
        let three_quarter_lr = scheduler.get_lr(75);
        let final_lr = 0.1 / (25.0 * 1e4);
        let expected = 0.1 + (final_lr - 0.1) * 0.5;
        assert!((three_quarter_lr - expected).abs() < 1e-5);
    }
}
