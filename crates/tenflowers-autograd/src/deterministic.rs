//! # Deterministic Execution Mode
//!
//! This module provides deterministic seed management for reproducible training
//! across forward and backward passes in the autograd system.
//!
//! ## Features
//!
//! - **Global Seed Management**: Set and manage global random seeds
//! - **Per-Operation Seeds**: Fine-grained seed control for individual operations
//! - **Reproducibility Guarantees**: Ensure identical results across runs
//! - **Thread-Safe**: Safe to use in multi-threaded environments
//! - **Tape Integration**: Seamless integration with GradientTape
//!
//! ## Usage
//!
//! ```rust,no_run
//! use tenflowers_autograd::deterministic::{set_deterministic, is_deterministic, DeterministicContext};
//!
//! // Enable deterministic mode globally
//! set_deterministic(true, Some(42));
//!
//! // All operations will now use deterministic seeds
//! assert!(is_deterministic());
//!
//! // Use scoped deterministic context
//! {
//!     let _ctx = DeterministicContext::new(123);
//!     // Operations in this scope use seed 123
//! }
//! // Deterministic mode restored to previous state
//! ```
//!
//! ## Reproducibility
//!
//! To ensure full reproducibility:
//! 1. Set deterministic mode before any operations: `set_deterministic(true, Some(seed))`
//! 2. Use the same hardware (CPU vs GPU can produce different results)
//! 3. Use the same number of threads (for parallel operations)
//! 4. Use the same operation ordering in your computation graph

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

/// Global deterministic configuration state
static DETERMINISTIC_STATE: Lazy<RwLock<DeterministicState>> = Lazy::new(|| {
    RwLock::new(DeterministicState {
        enabled: false,
        global_seed: None,
        operation_seeds: HashMap::new(),
        seed_counter: 0,
    })
});

/// Thread-local seed state for per-thread reproducibility
thread_local! {
    static THREAD_SEED: std::cell::RefCell<Option<u64>> = const { std::cell::RefCell::new(None) };
}

/// Internal state for deterministic execution
#[derive(Debug, Clone)]
struct DeterministicState {
    /// Whether deterministic mode is enabled
    enabled: bool,
    /// Global seed for all operations
    global_seed: Option<u64>,
    /// Per-operation seeds (operation_id -> seed)
    operation_seeds: HashMap<String, u64>,
    /// Counter for generating unique seeds
    seed_counter: u64,
}

impl DeterministicState {
    /// Get the next seed value
    fn next_seed(&mut self) -> u64 {
        if let Some(global_seed) = self.global_seed {
            // Derive new seed from global seed and counter
            let seed = global_seed.wrapping_add(self.seed_counter);
            self.seed_counter = self.seed_counter.wrapping_add(1);
            seed
        } else {
            // Use counter as seed
            let seed = self.seed_counter;
            self.seed_counter = self.seed_counter.wrapping_add(1);
            seed
        }
    }

    /// Get or create seed for an operation
    fn get_or_create_operation_seed(&mut self, operation_id: &str) -> u64 {
        if let Some(&seed) = self.operation_seeds.get(operation_id) {
            seed
        } else {
            let seed = self.next_seed();
            self.operation_seeds.insert(operation_id.to_string(), seed);
            seed
        }
    }

    /// Reset all state
    fn reset(&mut self) {
        self.operation_seeds.clear();
        self.seed_counter = 0;
    }
}

/// Configuration for deterministic execution
#[derive(Debug, Clone)]
pub struct DeterministicConfig {
    /// Global seed for all random operations
    pub global_seed: Option<u64>,
    /// Whether to enforce strict determinism (may be slower)
    pub strict_mode: bool,
    /// Whether to warn on non-deterministic operations
    pub warn_non_deterministic: bool,
    /// Maximum number of operation seeds to cache
    pub max_cached_seeds: usize,
}

impl Default for DeterministicConfig {
    fn default() -> Self {
        Self {
            global_seed: None,
            strict_mode: false,
            warn_non_deterministic: true,
            max_cached_seeds: 10000,
        }
    }
}

impl DeterministicConfig {
    /// Create a new configuration with a specific seed
    pub fn with_seed(seed: u64) -> Self {
        Self {
            global_seed: Some(seed),
            ..Default::default()
        }
    }

    /// Enable strict deterministic mode
    pub fn strict(mut self) -> Self {
        self.strict_mode = true;
        self
    }

    /// Disable warnings for non-deterministic operations
    pub fn no_warnings(mut self) -> Self {
        self.warn_non_deterministic = false;
        self
    }
}

/// RAII guard for scoped deterministic execution
pub struct DeterministicContext {
    previous_enabled: bool,
    previous_seed: Option<u64>,
}

impl DeterministicContext {
    /// Create a new deterministic context with the given seed
    pub fn new(seed: u64) -> Self {
        let mut state = DETERMINISTIC_STATE.write().unwrap();
        let previous_enabled = state.enabled;
        let previous_seed = state.global_seed;

        state.enabled = true;
        state.global_seed = Some(seed);
        state.reset();

        DeterministicContext {
            previous_enabled,
            previous_seed,
        }
    }

    /// Create a new deterministic context with the given configuration
    pub fn with_config(config: DeterministicConfig) -> Self {
        let mut state = DETERMINISTIC_STATE.write().unwrap();
        let previous_enabled = state.enabled;
        let previous_seed = state.global_seed;

        state.enabled = true;
        state.global_seed = config.global_seed;
        state.reset();

        DeterministicContext {
            previous_enabled,
            previous_seed,
        }
    }
}

impl Drop for DeterministicContext {
    fn drop(&mut self) {
        let mut state = DETERMINISTIC_STATE.write().unwrap();
        state.enabled = self.previous_enabled;
        state.global_seed = self.previous_seed;
        if !self.previous_enabled {
            state.reset();
        }
    }
}

/// Enable or disable deterministic mode globally
pub fn set_deterministic(enabled: bool, seed: Option<u64>) {
    let mut state = DETERMINISTIC_STATE.write().unwrap();
    state.enabled = enabled;
    state.global_seed = seed;
    if enabled {
        state.reset();
    }
}

/// Check if deterministic mode is currently enabled
pub fn is_deterministic() -> bool {
    DETERMINISTIC_STATE.read().unwrap().enabled
}

/// Get the current global seed (if any)
pub fn get_global_seed() -> Option<u64> {
    DETERMINISTIC_STATE.read().unwrap().global_seed
}

/// Set the global seed without changing deterministic mode
pub fn set_global_seed(seed: u64) {
    let mut state = DETERMINISTIC_STATE.write().unwrap();
    state.global_seed = Some(seed);
    state.reset();
}

/// Get or create a deterministic seed for a specific operation
pub fn get_operation_seed(operation_id: &str) -> Option<u64> {
    let mut state = DETERMINISTIC_STATE.write().unwrap();
    if state.enabled {
        Some(state.get_or_create_operation_seed(operation_id))
    } else {
        None
    }
}

/// Set a specific seed for an operation
pub fn set_operation_seed(operation_id: &str, seed: u64) {
    let mut state = DETERMINISTIC_STATE.write().unwrap();
    state.operation_seeds.insert(operation_id.to_string(), seed);
}

/// Clear all operation-specific seeds
pub fn clear_operation_seeds() {
    let mut state = DETERMINISTIC_STATE.write().unwrap();
    state.operation_seeds.clear();
}

/// Reset the deterministic state (clears all operation seeds and counter)
pub fn reset_deterministic_state() {
    let mut state = DETERMINISTIC_STATE.write().unwrap();
    state.reset();
}

/// Get the number of unique operations that have been seeded
pub fn get_seeded_operation_count() -> usize {
    DETERMINISTIC_STATE.read().unwrap().operation_seeds.len()
}

/// Seed manager for gradient tape operations
#[derive(Debug, Clone)]
pub struct SeedManager {
    /// Base seed for this manager
    base_seed: u64,
    /// Counter for generating operation seeds
    counter: Arc<Mutex<u64>>,
    /// Cached operation seeds
    seeds: Arc<Mutex<HashMap<String, u64>>>,
}

impl SeedManager {
    /// Create a new seed manager with the given base seed
    pub fn new(base_seed: u64) -> Self {
        Self {
            base_seed,
            counter: Arc::new(Mutex::new(0)),
            seeds: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a seed manager from global deterministic state
    pub fn from_global() -> Option<Self> {
        get_global_seed().map(Self::new)
    }

    /// Get or create a seed for an operation
    pub fn get_seed(&self, operation_id: &str) -> u64 {
        let mut seeds = self.seeds.lock().unwrap();
        if let Some(&seed) = seeds.get(operation_id) {
            seed
        } else {
            let mut counter = self.counter.lock().unwrap();
            let seed = self.base_seed.wrapping_add(*counter);
            *counter = counter.wrapping_add(1);
            seeds.insert(operation_id.to_string(), seed);
            seed
        }
    }

    /// Get the next sequential seed
    pub fn next_seed(&self) -> u64 {
        let mut counter = self.counter.lock().unwrap();
        let seed = self.base_seed.wrapping_add(*counter);
        *counter = counter.wrapping_add(1);
        seed
    }

    /// Reset the seed counter
    pub fn reset(&self) {
        let mut counter = self.counter.lock().unwrap();
        *counter = 0;
        let mut seeds = self.seeds.lock().unwrap();
        seeds.clear();
    }

    /// Get the number of seeds generated
    pub fn seed_count(&self) -> u64 {
        *self.counter.lock().unwrap()
    }
}

/// Reproducibility checker - validates that operations can be reproduced
#[derive(Debug)]
pub struct ReproducibilityChecker {
    /// Reference execution results
    reference_hashes: HashMap<String, u64>,
    /// Whether checking is enabled
    enabled: bool,
}

impl ReproducibilityChecker {
    /// Create a new reproducibility checker
    pub fn new() -> Self {
        Self {
            reference_hashes: HashMap::new(),
            enabled: false,
        }
    }

    /// Enable reproducibility checking
    pub fn enable(&mut self) {
        self.enabled = true;
        self.reference_hashes.clear();
    }

    /// Disable reproducibility checking
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Record a reference hash for an operation
    pub fn record_hash(&mut self, operation_id: &str, hash: u64) {
        if self.enabled {
            self.reference_hashes.insert(operation_id.to_string(), hash);
        }
    }

    /// Check if an operation's hash matches the reference
    pub fn check_hash(&self, operation_id: &str, hash: u64) -> bool {
        if !self.enabled {
            return true;
        }

        if let Some(&reference_hash) = self.reference_hashes.get(operation_id) {
            reference_hash == hash
        } else {
            // No reference hash - this is okay on first run
            true
        }
    }

    /// Get reproducibility statistics
    pub fn stats(&self) -> ReproducibilityStats {
        ReproducibilityStats {
            operations_tracked: self.reference_hashes.len(),
            enabled: self.enabled,
        }
    }
}

impl Default for ReproducibilityChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about reproducibility checking
#[derive(Debug, Clone)]
pub struct ReproducibilityStats {
    /// Number of operations being tracked
    pub operations_tracked: usize,
    /// Whether checking is enabled
    pub enabled: bool,
}

/// Helper function to compute a simple hash of tensor data
/// This is used for reproducibility checking
pub fn hash_tensor_data(data: &[f32]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();

    // Hash the length
    data.len().hash(&mut hasher);

    // Hash a sample of values (for performance)
    let sample_size = data.len().min(1000);
    let step = if data.len() > sample_size {
        data.len() / sample_size
    } else {
        1
    };

    for i in (0..data.len()).step_by(step) {
        // Convert to bits for exact comparison
        data[i].to_bits().hash(&mut hasher);
    }

    hasher.finish()
}

/// Deterministic operation trait for operations that support reproducibility
pub trait DeterministicOperation {
    /// Get the operation's unique identifier
    fn operation_id(&self) -> String;

    /// Check if this operation is deterministic
    fn is_deterministic(&self) -> bool {
        true
    }

    /// Get the seed for this operation (if deterministic mode is enabled)
    fn get_seed(&self) -> Option<u64> {
        if is_deterministic() {
            Some(get_operation_seed(&self.operation_id()).unwrap_or_else(|| {
                let mut state = DETERMINISTIC_STATE.write().unwrap();
                state.next_seed()
            }))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_mode() {
        set_deterministic(false, None);
        assert!(!is_deterministic());

        set_deterministic(true, Some(42));
        assert!(is_deterministic());
        assert_eq!(get_global_seed(), Some(42));

        set_deterministic(false, None);
        assert!(!is_deterministic());
    }

    #[test]
    fn test_deterministic_context() {
        set_deterministic(false, None);

        {
            let _ctx = DeterministicContext::new(123);
            assert!(is_deterministic());
            assert_eq!(get_global_seed(), Some(123));
        }

        assert!(!is_deterministic());
    }

    #[test]
    fn test_operation_seeds() {
        set_deterministic(true, Some(42));
        clear_operation_seeds();

        let seed1 = get_operation_seed("op1");
        let seed2 = get_operation_seed("op2");
        let seed1_again = get_operation_seed("op1");

        assert!(seed1.is_some());
        assert!(seed2.is_some());
        assert_eq!(seed1, seed1_again); // Same operation gets same seed
        assert_ne!(seed1, seed2); // Different operations get different seeds
    }

    #[test]
    fn test_seed_manager() {
        let manager = SeedManager::new(100);

        let seed1 = manager.get_seed("op1");
        let seed2 = manager.get_seed("op2");
        let seed1_again = manager.get_seed("op1");

        assert_eq!(seed1, seed1_again);
        assert_ne!(seed1, seed2);

        manager.reset();
        assert_eq!(manager.seed_count(), 0);
    }

    #[test]
    fn test_reproducibility_checker() {
        let mut checker = ReproducibilityChecker::new();
        checker.enable();

        checker.record_hash("op1", 12345);
        assert!(checker.check_hash("op1", 12345));
        assert!(!checker.check_hash("op1", 67890));

        let stats = checker.stats();
        assert_eq!(stats.operations_tracked, 1);
        assert!(stats.enabled);
    }

    #[test]
    fn test_hash_tensor_data() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0];
        let data3 = vec![1.0, 2.0, 3.0, 5.0];

        let hash1 = hash_tensor_data(&data1);
        let hash2 = hash_tensor_data(&data2);
        let hash3 = hash_tensor_data(&data3);

        assert_eq!(hash1, hash2); // Same data should have same hash
        assert_ne!(hash1, hash3); // Different data should have different hash
    }

    #[test]
    fn test_deterministic_config() {
        let config = DeterministicConfig::with_seed(42).strict().no_warnings();

        assert_eq!(config.global_seed, Some(42));
        assert!(config.strict_mode);
        assert!(!config.warn_non_deterministic);
    }

    #[test]
    fn test_seed_manager_sequential() {
        let manager = SeedManager::new(1000);

        let seed1 = manager.next_seed();
        let seed2 = manager.next_seed();
        let seed3 = manager.next_seed();

        assert_eq!(seed1, 1000);
        assert_eq!(seed2, 1001);
        assert_eq!(seed3, 1002);
    }

    #[test]
    fn test_operation_seed_persistence() {
        set_deterministic(true, Some(42));
        clear_operation_seeds();

        // Set a specific seed for an operation
        set_operation_seed("my_op", 999);

        // Verify it persists
        let seed = get_operation_seed("my_op");
        assert_eq!(seed, Some(999));

        clear_operation_seeds();
        assert_eq!(get_seeded_operation_count(), 0);
    }
}
