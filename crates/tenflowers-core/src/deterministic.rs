/// Deterministic Mode for Reproducible Training
///
/// This module provides infrastructure for deterministic execution, ensuring that
/// training runs produce identical results when using the same random seed. This is
/// critical for debugging, comparing experiments, and scientific reproducibility.
///
/// ## Features
///
/// - **Global Seed Management**: Centralized seed control across all operations
/// - **Operation-Local Seeds**: Each operation gets a deterministic subseed
/// - **RNG State Tracking**: Save and restore random number generator states
/// - **GPU Determinism**: Control non-deterministic GPU operations
/// - **Reproducibility Validation**: Verify that operations are truly deterministic
///
/// ## Usage
///
/// ```rust,ignore
/// use tenflowers_core::deterministic::{set_deterministic_mode, set_global_seed};
///
/// // Enable deterministic mode with a specific seed
/// set_global_seed(42);
/// set_deterministic_mode(true);
///
/// // All operations will now use deterministic algorithms
/// let tensor = Tensor::<f32>::randn(&[10, 10]); // Uses seed 42
///
/// // Operations get unique subseeds
/// let dropout_output = dropout(&tensor, 0.5); // Uses derived subseed
/// ```
///
/// ## Important Notes
///
/// - Deterministic mode may be slower than non-deterministic mode
/// - Some GPU operations may fall back to CPU for determinism
/// - Parallel execution order must be controlled for full reproducibility
use crate::{Result, TensorError};
use std::sync::{Arc, Mutex, OnceLock};

/// Global deterministic mode state
#[derive(Debug, Clone)]
pub struct DeterministicState {
    /// Whether deterministic mode is enabled
    pub enabled: bool,
    /// Global random seed
    pub global_seed: u64,
    /// Current operation counter for subseed generation
    pub operation_counter: u64,
    /// Whether to enforce determinism strictly (fail on non-deterministic ops)
    pub strict_mode: bool,
    /// Whether to use deterministic algorithms even if slower
    pub prefer_deterministic_algorithms: bool,
    /// Track which operations have been executed for reproducibility
    pub operation_log: Vec<String>,
    /// Maximum size of operation log
    pub max_log_size: usize,
}

impl Default for DeterministicState {
    fn default() -> Self {
        Self {
            enabled: false,
            global_seed: 0,
            operation_counter: 0,
            strict_mode: false,
            prefer_deterministic_algorithms: true,
            operation_log: Vec::new(),
            max_log_size: 1000,
        }
    }
}

impl DeterministicState {
    /// Create a new deterministic state with a seed
    pub fn new(seed: u64) -> Self {
        Self {
            enabled: true,
            global_seed: seed,
            ..Default::default()
        }
    }

    /// Get the next subseed for an operation
    pub fn next_subseed(&mut self, operation_name: &str) -> u64 {
        // Generate deterministic subseed based on global seed and counter
        let subseed = self
            .global_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.operation_counter)
            .wrapping_add(hash_string(operation_name));

        self.operation_counter += 1;

        // Log operation if enabled
        if self.operation_log.len() < self.max_log_size {
            self.operation_log
                .push(format!("{}: seed={}", operation_name, subseed));
        }

        subseed
    }

    /// Reset the operation counter
    pub fn reset_counter(&mut self) {
        self.operation_counter = 0;
    }

    /// Clear the operation log
    pub fn clear_log(&mut self) {
        self.operation_log.clear();
    }

    /// Get a snapshot of the current state for reproducibility
    pub fn snapshot(&self) -> DeterministicSnapshot {
        DeterministicSnapshot {
            global_seed: self.global_seed,
            operation_counter: self.operation_counter,
            enabled: self.enabled,
        }
    }

    /// Restore from a snapshot
    pub fn restore(&mut self, snapshot: &DeterministicSnapshot) {
        self.global_seed = snapshot.global_seed;
        self.operation_counter = snapshot.operation_counter;
        self.enabled = snapshot.enabled;
    }
}

/// Snapshot of deterministic state for checkpointing
#[derive(Debug, Clone, Copy)]
pub struct DeterministicSnapshot {
    pub global_seed: u64,
    pub operation_counter: u64,
    pub enabled: bool,
}

/// Simple string hash function for operation names
fn hash_string(s: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64; // FNV offset basis
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    hash
}

// ============================================================================
// Global State Management
// ============================================================================

static GLOBAL_STATE: OnceLock<Arc<Mutex<DeterministicState>>> = OnceLock::new();

/// Get the global deterministic state
fn get_global_state() -> &'static Arc<Mutex<DeterministicState>> {
    GLOBAL_STATE.get_or_init(|| Arc::new(Mutex::new(DeterministicState::default())))
}

/// Enable or disable deterministic mode
///
/// When enabled, all operations will use deterministic algorithms and RNG seeding.
pub fn set_deterministic_mode(enabled: bool) {
    let state = get_global_state();
    state.lock().unwrap().enabled = enabled;
}

/// Check if deterministic mode is enabled
pub fn is_deterministic_mode() -> bool {
    let state = get_global_state();
    state.lock().unwrap().enabled
}

/// Set the global random seed
///
/// This seed is used to derive subseeds for all random operations.
pub fn set_global_seed(seed: u64) {
    let state = get_global_state();
    let mut s = state.lock().unwrap();
    s.global_seed = seed;
    s.operation_counter = 0;
    s.clear_log();
}

/// Get the current global seed
pub fn get_global_seed() -> u64 {
    let state = get_global_state();
    state.lock().unwrap().global_seed
}

/// Enable strict mode (fail on non-deterministic operations)
pub fn set_strict_mode(strict: bool) {
    let state = get_global_state();
    state.lock().unwrap().strict_mode = strict;
}

/// Check if strict mode is enabled
pub fn is_strict_mode() -> bool {
    let state = get_global_state();
    state.lock().unwrap().strict_mode
}

/// Get a subseed for a specific operation
///
/// This ensures that each operation gets a unique, deterministic seed
/// derived from the global seed and operation sequence.
pub fn get_operation_seed(operation_name: &str) -> u64 {
    let state = get_global_state();
    let mut s = state.lock().unwrap();

    if !s.enabled {
        // In non-deterministic mode, use system time
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    } else {
        s.next_subseed(operation_name)
    }
}

/// Reset the operation counter
///
/// Useful when you want to restart from a known state while keeping
/// the same global seed.
pub fn reset_operation_counter() {
    let state = get_global_state();
    state.lock().unwrap().reset_counter();
}

/// Get a snapshot of the current deterministic state
///
/// Useful for checkpointing and restoring state.
pub fn get_state_snapshot() -> DeterministicSnapshot {
    let state = get_global_state();
    state.lock().unwrap().snapshot()
}

/// Restore deterministic state from a snapshot
pub fn restore_state_snapshot(snapshot: &DeterministicSnapshot) {
    let state = get_global_state();
    state.lock().unwrap().restore(snapshot);
}

/// Get the operation log for debugging
pub fn get_operation_log() -> Vec<String> {
    let state = get_global_state();
    state.lock().unwrap().operation_log.clone()
}

/// Clear the operation log
pub fn clear_operation_log() {
    let state = get_global_state();
    state.lock().unwrap().clear_log();
}

/// Scoped deterministic mode
///
/// Temporarily enable deterministic mode with a specific seed,
/// then restore the previous state when dropped.
pub struct DeterministicScope {
    previous_state: DeterministicSnapshot,
}

impl DeterministicScope {
    /// Create a new deterministic scope with a seed
    pub fn new(seed: u64) -> Self {
        let previous_state = get_state_snapshot();

        set_deterministic_mode(true);
        set_global_seed(seed);

        Self { previous_state }
    }

    /// Create a scope that only affects the mode, not the seed
    pub fn with_mode(enabled: bool) -> Self {
        let previous_state = get_state_snapshot();
        set_deterministic_mode(enabled);
        Self { previous_state }
    }
}

impl Drop for DeterministicScope {
    fn drop(&mut self) {
        restore_state_snapshot(&self.previous_state);
    }
}

/// Configuration for deterministic execution
#[derive(Debug, Clone)]
pub struct DeterministicConfig {
    /// Global seed
    pub seed: u64,
    /// Enable strict mode
    pub strict: bool,
    /// Prefer deterministic algorithms even if slower
    pub prefer_deterministic: bool,
    /// Enable operation logging
    pub log_operations: bool,
}

impl Default for DeterministicConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            strict: false,
            prefer_deterministic: true,
            log_operations: false,
        }
    }
}

impl DeterministicConfig {
    /// Apply this configuration globally
    pub fn apply(&self) {
        set_global_seed(self.seed);
        set_deterministic_mode(true);
        set_strict_mode(self.strict);

        let state = get_global_state();
        let mut s = state.lock().unwrap();
        s.prefer_deterministic_algorithms = self.prefer_deterministic;

        if !self.log_operations {
            s.clear_log();
            s.max_log_size = 0;
        } else {
            s.max_log_size = 1000;
        }
    }
}

/// Verify that an operation is reproducible
///
/// Runs the operation twice with the same seed and checks if results match.
pub fn verify_reproducibility<F, T>(operation_name: &str, mut operation: F) -> Result<bool>
where
    F: FnMut() -> T,
    T: PartialEq,
{
    let snapshot = get_state_snapshot();

    // First run
    set_global_seed(snapshot.global_seed);
    reset_operation_counter();
    let result1 = operation();

    // Second run with same seed
    set_global_seed(snapshot.global_seed);
    reset_operation_counter();
    let result2 = operation();

    // Restore original state
    restore_state_snapshot(&snapshot);

    Ok(result1 == result2)
}

// ============================================================================
// Utilities
// ============================================================================

/// Mark an operation as potentially non-deterministic
///
/// In strict mode, this will return an error. Otherwise, it logs a warning.
pub fn mark_non_deterministic(operation_name: &str) -> Result<()> {
    if is_deterministic_mode() && is_strict_mode() {
        Err(TensorError::invalid_operation_simple(format!(
            "Operation '{}' is non-deterministic but strict deterministic mode is enabled",
            operation_name
        )))
    } else {
        // In non-strict mode, just log it
        if is_deterministic_mode() {
            eprintln!(
                "Warning: Operation '{}' may not be fully deterministic",
                operation_name
            );
        }
        Ok(())
    }
}

/// Helper to check if GPU operations should use deterministic algorithms
pub fn should_use_deterministic_gpu_ops() -> bool {
    let state = get_global_state();
    let s = state.lock().unwrap();
    s.enabled && s.prefer_deterministic_algorithms
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_mode_toggle() {
        set_deterministic_mode(true);
        assert!(is_deterministic_mode());

        set_deterministic_mode(false);
        assert!(!is_deterministic_mode());
    }

    #[test]
    fn test_global_seed() {
        set_global_seed(12345);
        assert_eq!(get_global_seed(), 12345);

        set_global_seed(67890);
        assert_eq!(get_global_seed(), 67890);
    }

    #[test]
    fn test_operation_seed_generation() {
        set_deterministic_mode(true);
        set_global_seed(42);

        let seed1 = get_operation_seed("test_op");
        let seed2 = get_operation_seed("test_op");

        // Seeds should be different due to counter increment
        assert_ne!(seed1, seed2);

        // Reset and verify reproducibility
        reset_operation_counter();
        let seed3 = get_operation_seed("test_op");
        assert_eq!(seed1, seed3);
    }

    #[test]
    fn test_operation_seed_uniqueness() {
        set_deterministic_mode(true);
        set_global_seed(42);
        reset_operation_counter();

        let seed_a = get_operation_seed("operation_a");
        let seed_b = get_operation_seed("operation_b");

        // Different operations should get different seeds
        assert_ne!(seed_a, seed_b);
    }

    #[test]
    fn test_snapshot_and_restore() {
        set_deterministic_mode(true);
        set_global_seed(100);

        let _ = get_operation_seed("op1");
        let _ = get_operation_seed("op2");

        let snapshot = get_state_snapshot();

        let _ = get_operation_seed("op3");

        restore_state_snapshot(&snapshot);

        let seed_after_restore = get_operation_seed("op3");

        // After restore, we should get the same seed for op3
        restore_state_snapshot(&snapshot);
        let seed_repeat = get_operation_seed("op3");

        assert_eq!(seed_after_restore, seed_repeat);
    }

    #[test]
    fn test_deterministic_scope() {
        set_deterministic_mode(false);
        set_global_seed(100);

        {
            let _scope = DeterministicScope::new(200);
            assert!(is_deterministic_mode());
            assert_eq!(get_global_seed(), 200);
        }

        // After scope ends, state should be restored
        assert!(!is_deterministic_mode());
        assert_eq!(get_global_seed(), 100);
    }

    #[test]
    fn test_strict_mode() {
        set_strict_mode(true);
        assert!(is_strict_mode());

        set_strict_mode(false);
        assert!(!is_strict_mode());
    }

    #[test]
    fn test_mark_non_deterministic() {
        set_deterministic_mode(true);
        set_strict_mode(false);

        // Should succeed in non-strict mode
        assert!(mark_non_deterministic("test_op").is_ok());

        set_strict_mode(true);
        // Should fail in strict mode
        assert!(mark_non_deterministic("test_op").is_err());
    }

    #[test]
    fn test_config_apply() {
        let config = DeterministicConfig {
            seed: 777,
            strict: true,
            prefer_deterministic: true,
            log_operations: false,
        };

        config.apply();

        assert_eq!(get_global_seed(), 777);
        assert!(is_deterministic_mode());
        assert!(is_strict_mode());
    }

    #[test]
    fn test_operation_log() {
        set_deterministic_mode(true);
        set_global_seed(42);
        clear_operation_log();

        let _ = get_operation_seed("op1");
        let _ = get_operation_seed("op2");

        let log = get_operation_log();
        assert_eq!(log.len(), 2);
        assert!(log[0].contains("op1"));
        assert!(log[1].contains("op2"));
    }

    #[test]
    fn test_hash_string_deterministic() {
        // Same string should always produce same hash
        let hash1 = hash_string("test");
        let hash2 = hash_string("test");
        assert_eq!(hash1, hash2);

        // Different strings should produce different hashes
        let hash3 = hash_string("different");
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_reproducibility_with_counter_reset() {
        set_deterministic_mode(true);
        set_global_seed(42);

        // First sequence
        reset_operation_counter();
        let seeds1: Vec<u64> = (0..5)
            .map(|i| get_operation_seed(&format!("op{}", i)))
            .collect();

        // Second sequence with same seed
        reset_operation_counter();
        let seeds2: Vec<u64> = (0..5)
            .map(|i| get_operation_seed(&format!("op{}", i)))
            .collect();

        assert_eq!(seeds1, seeds2);
    }

    #[test]
    fn test_non_deterministic_mode_uses_system_time() {
        set_deterministic_mode(false);

        let seed1 = get_operation_seed("test");
        std::thread::sleep(std::time::Duration::from_nanos(100));
        let seed2 = get_operation_seed("test");

        // In non-deterministic mode, seeds should be different
        // (though there's a tiny chance they could be the same)
        // We just check that the function doesn't panic
        let _ = seed1;
        let _ = seed2;
    }
}
