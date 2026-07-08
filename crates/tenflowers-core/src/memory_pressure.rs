/// GPU Memory Pressure Monitor — threshold-based alerting with callbacks.
///
/// This module tracks whether GPU memory consumption has crossed configurable
/// warning / critical / OOM thresholds and fires registered callbacks when
/// the pressure level changes.

use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// PressureLevel
// ---------------------------------------------------------------------------

/// Classification of the current GPU memory pressure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PressureLevel {
    /// Memory usage is below the warning threshold.
    Normal,
    /// Memory usage has exceeded the warning threshold.
    Warning,
    /// Memory usage has exceeded the critical threshold.
    Critical,
    /// Memory usage has exceeded the total budget (OOM imminent).
    OutOfMemory,
}

// ---------------------------------------------------------------------------
// PressureCallback type alias
// ---------------------------------------------------------------------------

/// A callback invoked when a pressure threshold is crossed.
///
/// Arguments: `(pressure_fraction: f32, current_usage_bytes: usize)`.
pub type PressureCallback = Box<dyn Fn(f32, usize) + Send + Sync>;

// ---------------------------------------------------------------------------
// MemoryPressureMonitor
// ---------------------------------------------------------------------------

/// Threshold-based GPU memory pressure monitor with configurable callbacks.
///
/// Call [`check_pressure`] after every significant allocation to receive the
/// current [`PressureLevel`] and trigger any registered callbacks.
pub struct MemoryPressureMonitor {
    /// Hard budget in bytes (maps to 100% usage).
    total_budget_bytes: usize,
    /// Fraction of budget that triggers a warning (e.g. `0.80`).
    warning_threshold: f32,
    /// Fraction of budget that triggers a critical alert (e.g. `0.95`).
    critical_threshold: f32,
    warning_callback: Option<Arc<PressureCallback>>,
    critical_callback: Option<Arc<PressureCallback>>,
    oom_callback: Option<Arc<PressureCallback>>,
    /// Last observed pressure level, used to detect transitions.
    last_pressure_level: Mutex<PressureLevel>,
}

impl MemoryPressureMonitor {
    /// Create a monitor with a given memory budget.
    ///
    /// Default thresholds: warning = 80%, critical = 95%.
    pub fn new(total_budget_bytes: usize) -> Self {
        Self {
            total_budget_bytes,
            warning_threshold: 0.80,
            critical_threshold: 0.95,
            warning_callback: None,
            critical_callback: None,
            oom_callback: None,
            last_pressure_level: Mutex::new(PressureLevel::Normal),
        }
    }

    /// Set the warning threshold as a fraction of the total budget (0.0 – 1.0).
    pub fn set_warning_threshold(&mut self, fraction: f32) {
        self.warning_threshold = fraction.clamp(0.0, 1.0);
    }

    /// Set the critical threshold as a fraction of the total budget (0.0 – 1.0).
    pub fn set_critical_threshold(&mut self, fraction: f32) {
        self.critical_threshold = fraction.clamp(0.0, 1.0);
    }

    /// Register a callback that fires when usage crosses the warning threshold.
    pub fn set_warning_callback(
        &mut self,
        cb: impl Fn(f32, usize) + Send + Sync + 'static,
    ) {
        self.warning_callback = Some(Arc::new(Box::new(cb)));
    }

    /// Register a callback that fires when usage crosses the critical threshold.
    pub fn set_critical_callback(
        &mut self,
        cb: impl Fn(f32, usize) + Send + Sync + 'static,
    ) {
        self.critical_callback = Some(Arc::new(Box::new(cb)));
    }

    /// Register a callback that fires when usage exceeds the total budget (OOM).
    pub fn set_oom_callback(
        &mut self,
        cb: impl Fn(f32, usize) + Send + Sync + 'static,
    ) {
        self.oom_callback = Some(Arc::new(Box::new(cb)));
    }

    /// Evaluate the current pressure level for `current_usage_bytes`.
    ///
    /// If the level has changed since the last call the appropriate callback is
    /// fired.  Returns the new [`PressureLevel`].
    pub fn check_pressure(&self, current_usage_bytes: usize) -> PressureLevel {
        let budget = self.total_budget_bytes;
        let fraction = if budget > 0 {
            (current_usage_bytes as f32) / (budget as f32)
        } else {
            // Zero budget — any usage is OOM.
            if current_usage_bytes > 0 { 2.0_f32 } else { 0.0_f32 }
        };

        let level = if current_usage_bytes > budget {
            PressureLevel::OutOfMemory
        } else if fraction >= self.critical_threshold {
            PressureLevel::Critical
        } else if fraction >= self.warning_threshold {
            PressureLevel::Warning
        } else {
            PressureLevel::Normal
        };

        // Fire callback only when the level has changed.
        let mut last = match self.last_pressure_level.lock() {
            Ok(guard) => guard,
            Err(_) => return PressureLevel::Normal, // poisoned lock, return safe default
        };

        if *last != level {
            match &level {
                PressureLevel::Warning => {
                    if let Some(cb) = &self.warning_callback {
                        cb(fraction, current_usage_bytes);
                    }
                }
                PressureLevel::Critical => {
                    if let Some(cb) = &self.critical_callback {
                        cb(fraction, current_usage_bytes);
                    }
                }
                PressureLevel::OutOfMemory => {
                    if let Some(cb) = &self.oom_callback {
                        cb(fraction, current_usage_bytes);
                    }
                }
                PressureLevel::Normal => {}
            }
            *last = level.clone();
        }

        level
    }

    /// Return the last observed [`PressureLevel`] without triggering callbacks.
    pub fn current_level(&self) -> PressureLevel {
        match self.last_pressure_level.lock() {
            Ok(guard) => guard.clone(),
            Err(_) => PressureLevel::Normal, // poisoned lock, return safe default
        }
    }

    /// Return the total memory budget in bytes.
    pub fn total_budget_bytes(&self) -> usize {
        self.total_budget_bytes
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn test_pressure_monitor_normal() {
        let monitor = MemoryPressureMonitor::new(1000);
        assert_eq!(monitor.check_pressure(0), PressureLevel::Normal);
        assert_eq!(monitor.check_pressure(700), PressureLevel::Normal);
    }

    #[test]
    fn test_pressure_monitor_warning() {
        let monitor = MemoryPressureMonitor::new(1000);
        assert_eq!(monitor.check_pressure(850), PressureLevel::Warning);
    }

    #[test]
    fn test_pressure_monitor_critical() {
        let monitor = MemoryPressureMonitor::new(1000);
        assert_eq!(monitor.check_pressure(960), PressureLevel::Critical);
    }

    #[test]
    fn test_pressure_monitor_oom() {
        let monitor = MemoryPressureMonitor::new(1000);
        assert_eq!(monitor.check_pressure(1001), PressureLevel::OutOfMemory);
    }

    #[test]
    fn test_pressure_callback_fires() {
        let fired = Arc::new(AtomicBool::new(false));
        let fired_clone = Arc::clone(&fired);

        let mut monitor = MemoryPressureMonitor::new(1000);
        monitor.set_warning_callback(move |_frac, _usage| {
            fired_clone.store(true, Ordering::SeqCst);
        });

        // First call — transitions from Normal to Warning, callback should fire.
        let level = monitor.check_pressure(850);
        assert_eq!(level, PressureLevel::Warning);
        assert!(fired.load(Ordering::SeqCst), "warning callback must fire");
    }

    #[test]
    fn test_pressure_callback_not_refired_for_same_level() {
        let count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let count_clone = Arc::clone(&count);

        let mut monitor = MemoryPressureMonitor::new(1000);
        monitor.set_warning_callback(move |_, _| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        });

        monitor.check_pressure(850); // fires
        monitor.check_pressure(860); // same level — should not re-fire
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_pressure_custom_thresholds() {
        let mut monitor = MemoryPressureMonitor::new(1000);
        monitor.set_warning_threshold(0.5);
        monitor.set_critical_threshold(0.75);
        assert_eq!(monitor.check_pressure(600), PressureLevel::Warning);
        assert_eq!(monitor.check_pressure(800), PressureLevel::Critical);
    }

    #[test]
    fn test_pressure_current_level() {
        let monitor = MemoryPressureMonitor::new(1000);
        assert_eq!(monitor.current_level(), PressureLevel::Normal);
        // 960 / 1000 = 0.96 ≥ critical_threshold(0.95) → Critical
        monitor.check_pressure(960);
        assert_eq!(monitor.current_level(), PressureLevel::Critical);
    }
}
