//! Experience replay buffers for reinforcement learning.
//!
//! Provides both uniform and prioritized experience replay (PER) buffers
//! for DQN-style off-policy learning.

use scirs2_core::random::Random;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Transition
// ─────────────────────────────────────────────────────────────────────────────

/// A single (s, a, r, s', done) transition tuple.
#[derive(Debug, Clone)]
pub struct Transition {
    /// Current state observation.
    pub state: Vec<f32>,
    /// Index of the action taken.
    pub action: usize,
    /// Scalar reward received.
    pub reward: f32,
    /// Next state observation.
    pub next_state: Vec<f32>,
    /// Whether the episode ended after this transition.
    pub done: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// ReplayBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// Circular (ring) replay buffer for DQN-style experience replay.
///
/// When the buffer is full, the oldest transitions are overwritten.
#[derive(Debug)]
pub struct ReplayBuffer {
    capacity: usize,
    buffer: Vec<Transition>,
    /// Index at which the *next* transition will be written.
    head: usize,
    /// Number of transitions currently stored (≤ capacity).
    size: usize,
}

impl ReplayBuffer {
    /// Create a new `ReplayBuffer` with the given maximum capacity.
    ///
    /// # Errors
    /// Returns [`TensorError::InvalidArgument`] if `capacity == 0`.
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(TensorError::invalid_argument_op(
                "ReplayBuffer::new",
                "capacity must be greater than zero",
            ));
        }
        Ok(Self {
            capacity,
            buffer: Vec::with_capacity(capacity),
            head: 0,
            size: 0,
        })
    }

    /// Push a transition into the buffer.
    ///
    /// If the buffer has reached capacity, the oldest entry is overwritten.
    pub fn push(&mut self, transition: Transition) {
        if self.size < self.capacity {
            // Buffer not yet full — append.
            self.buffer.push(transition);
            self.size += 1;
        } else {
            // Buffer full — overwrite at head position.
            self.buffer[self.head] = transition;
        }
        self.head = (self.head + 1) % self.capacity;
    }

    /// Sample `batch_size` transitions uniformly at random (with replacement).
    ///
    /// # Errors
    /// Returns an error if the buffer has fewer entries than `batch_size`.
    pub fn sample(&self, batch_size: usize, seed: u64) -> Result<Vec<&Transition>> {
        if batch_size > self.size {
            return Err(TensorError::invalid_argument_op(
                "ReplayBuffer::sample",
                &format!(
                    "requested batch_size {} exceeds buffer size {}",
                    batch_size, self.size
                ),
            ));
        }
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let mut rng = Random::seed(seed);
        let mut samples = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let idx = rng.gen_range(0.0..(self.size as f64)) as usize;
            // Clamp in case of floating-point boundary.
            let idx = idx.min(self.size - 1);
            samples.push(&self.buffer[idx]);
        }

        Ok(samples)
    }

    /// Current number of transitions stored.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns `true` if no transitions are stored.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Returns `true` if at least `min_size` transitions are stored.
    pub fn is_ready(&self, min_size: usize) -> bool {
        self.size >= min_size
    }

    /// Maximum number of transitions the buffer can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Remove all stored transitions and reset the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.head = 0;
        self.size = 0;
    }

    // Internal: access by raw logical index (0 = oldest).
    pub(crate) fn get_by_index(&self, index: usize) -> Option<&Transition> {
        if index >= self.size {
            return None;
        }
        // When buffer is full the oldest entry is at `self.head`.
        let physical = if self.size < self.capacity {
            index
        } else {
            (self.head + index) % self.capacity
        };
        self.buffer.get(physical)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PrioritizedReplayBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// Prioritized experience replay (PER) with proportional priorities.
///
/// Transitions with higher TD-error are sampled more frequently.  Importance-
/// sampling weights correct for the resulting bias during gradient updates.
///
/// Reference: Schaul et al., 2015 — "Prioritized Experience Replay".
#[derive(Debug)]
pub struct PrioritizedReplayBuffer {
    buffer: ReplayBuffer,
    /// Per-transition priorities (parallel to `buffer.buffer`).
    priorities: Vec<f32>,
    /// Priority exponent α ∈ [0, 1]. 0 = uniform; 1 = full priority.
    alpha: f32,
    /// IS-weight exponent β ∈ [0, 1].  Anneal towards 1 during training.
    beta: f32,
    /// Small constant ε to ensure every transition has non-zero priority.
    eps: f32,
    /// Running maximum priority (used when `push` is called without explicit priority).
    max_priority: f32,
}

impl PrioritizedReplayBuffer {
    /// Create a new `PrioritizedReplayBuffer`.
    ///
    /// # Arguments
    /// * `capacity` — maximum number of transitions stored.
    /// * `alpha` — priority exponent (0 → uniform, 1 → fully prioritized).
    /// * `beta` — importance-sampling exponent (typically annealed to 1).
    ///
    /// # Errors
    /// Returns an error if `capacity == 0` or hyperparameters are out of range.
    pub fn new(capacity: usize, alpha: f32, beta: f32) -> Result<Self> {
        if capacity == 0 {
            return Err(TensorError::invalid_argument_op(
                "PrioritizedReplayBuffer::new",
                "capacity must be greater than zero",
            ));
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(TensorError::invalid_argument_op(
                "PrioritizedReplayBuffer::new",
                "alpha must be in [0, 1]",
            ));
        }
        if !(0.0..=1.0).contains(&beta) {
            return Err(TensorError::invalid_argument_op(
                "PrioritizedReplayBuffer::new",
                "beta must be in [0, 1]",
            ));
        }

        let buffer = ReplayBuffer::new(capacity)?;
        let priorities = Vec::with_capacity(capacity);

        Ok(Self {
            buffer,
            priorities,
            alpha,
            beta,
            eps: 1e-6,
            max_priority: 1.0,
        })
    }

    /// Push a transition with an explicit priority value.
    pub fn push_with_priority(&mut self, transition: Transition, priority: f32) {
        let p = (priority.abs() + self.eps).max(self.eps);
        if p > self.max_priority {
            self.max_priority = p;
        }

        let capacity = self.buffer.capacity();
        if self.buffer.size < capacity {
            // Buffer still growing.
            self.priorities.push(p);
        } else {
            // Overwrite at head position.
            let head = self.buffer.head;
            if head < self.priorities.len() {
                self.priorities[head] = p;
            } else {
                self.priorities.push(p);
            }
        }

        self.buffer.push(transition);
    }

    /// Push a transition using the current maximum priority (default for new samples).
    pub fn push(&mut self, transition: Transition) {
        let p = self.max_priority;
        self.push_with_priority(transition, p);
    }

    /// Sample `batch_size` transitions according to priorities.
    ///
    /// Returns `(transitions, indices, importance_weights)`.
    ///
    /// * `indices` can be passed to \[`update_priorities`\] after computing TD errors.
    /// * `importance_weights` should multiply the loss to correct for sampling bias.
    ///
    /// # Errors
    /// Returns an error if the buffer has fewer entries than `batch_size`.
    pub fn sample(
        &self,
        batch_size: usize,
        seed: u64,
    ) -> Result<(Vec<&Transition>, Vec<usize>, Vec<f32>)> {
        let n = self.buffer.size;
        if batch_size > n {
            return Err(TensorError::invalid_argument_op(
                "PrioritizedReplayBuffer::sample",
                &format!(
                    "requested batch_size {} exceeds buffer size {}",
                    batch_size, n
                ),
            ));
        }
        if batch_size == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        // Build probability distribution P(i) ∝ p_i^α.
        let prios: Vec<f32> = (0..n)
            .map(|i| {
                self.priorities
                    .get(i)
                    .copied()
                    .unwrap_or(self.eps)
                    .powf(self.alpha)
            })
            .collect();

        let total: f32 = prios.iter().sum();

        let mut rng = Random::seed(seed);
        let mut transitions = Vec::with_capacity(batch_size);
        let mut indices = Vec::with_capacity(batch_size);
        let mut weights = Vec::with_capacity(batch_size);

        // Minimum probability for IS weight normalisation.
        let min_p = prios.iter().copied().fold(f32::INFINITY, f32::min) / total;
        let max_weight = ((n as f32) * min_p).powf(-self.beta);

        for _ in 0..batch_size {
            // Stochastic sampling proportional to p_i^α.
            let mut threshold = rng.gen_range(0.0..total as f64) as f32;
            let mut chosen = n - 1;
            for (i, &p) in prios.iter().enumerate() {
                threshold -= p;
                if threshold <= 0.0 {
                    chosen = i;
                    break;
                }
            }

            let w = ((n as f32) * (prios[chosen] / total)).powf(-self.beta) / max_weight;

            let transition = self.buffer.get_by_index(chosen).ok_or_else(|| {
                TensorError::other_op(
                    "PrioritizedReplayBuffer::sample",
                    "internal index out of bounds",
                )
            })?;

            transitions.push(transition);
            indices.push(chosen);
            weights.push(w);
        }

        Ok((transitions, indices, weights))
    }

    /// Update priorities for a batch of transitions identified by their logical indices.
    ///
    /// Call this after computing TD errors for the sampled batch.
    ///
    /// # Errors
    /// Returns an error if any index is out of range.
    pub fn update_priorities(&mut self, indices: &[usize], priorities: &[f32]) -> Result<()> {
        if indices.len() != priorities.len() {
            return Err(TensorError::invalid_argument_op(
                "PrioritizedReplayBuffer::update_priorities",
                "indices and priorities must have the same length",
            ));
        }
        for (&idx, &p) in indices.iter().zip(priorities.iter()) {
            if idx >= self.buffer.size {
                return Err(TensorError::invalid_argument_op(
                    "PrioritizedReplayBuffer::update_priorities",
                    &format!(
                        "index {} is out of range (buffer size {})",
                        idx, self.buffer.size
                    ),
                ));
            }
            let new_p = (p.abs() + self.eps).max(self.eps);
            if new_p > self.max_priority {
                self.max_priority = new_p;
            }
            // Map logical index to physical storage index.
            let physical = if self.buffer.size < self.buffer.capacity {
                idx
            } else {
                (self.buffer.head + idx) % self.buffer.capacity
            };
            if physical < self.priorities.len() {
                self.priorities[physical] = new_p;
            }
        }
        Ok(())
    }

    /// Current number of transitions stored.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns `true` if no transitions are stored.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_transition(reward: f32) -> Transition {
        Transition {
            state: vec![0.0, 1.0],
            action: 0,
            reward,
            next_state: vec![1.0, 0.0],
            done: false,
        }
    }

    // ── ReplayBuffer ──────────────────────────────────────────────────────────

    #[test]
    fn test_replay_buffer_new_zero_capacity() {
        assert!(ReplayBuffer::new(0).is_err());
    }

    #[test]
    fn test_replay_buffer_push_and_len() {
        let mut rb = ReplayBuffer::new(10).expect("valid capacity");
        assert_eq!(rb.len(), 0);
        assert!(rb.is_empty());

        rb.push(make_transition(1.0));
        rb.push(make_transition(2.0));
        assert_eq!(rb.len(), 2);
        assert!(!rb.is_empty());
    }

    #[test]
    fn test_replay_buffer_capacity_reported_correctly() {
        let rb = ReplayBuffer::new(42).expect("valid capacity");
        assert_eq!(rb.capacity(), 42);
    }

    #[test]
    fn test_replay_buffer_circular_overwrite() {
        let mut rb = ReplayBuffer::new(3).expect("valid capacity");
        for i in 0..5u32 {
            rb.push(make_transition(i as f32));
        }
        // Only the 3 most recent transitions should be present.
        assert_eq!(rb.len(), 3);
    }

    #[test]
    fn test_replay_buffer_sample_correct_batch_size() {
        let mut rb = ReplayBuffer::new(100).expect("valid capacity");
        for i in 0..50 {
            rb.push(make_transition(i as f32));
        }
        let batch = rb.sample(16, 42).expect("sample should succeed");
        assert_eq!(batch.len(), 16);
    }

    #[test]
    fn test_replay_buffer_sample_too_large_fails() {
        let mut rb = ReplayBuffer::new(10).expect("valid capacity");
        rb.push(make_transition(0.0));
        assert!(rb.sample(2, 0).is_err());
    }

    #[test]
    fn test_replay_buffer_is_ready() {
        let mut rb = ReplayBuffer::new(100).expect("valid capacity");
        assert!(!rb.is_ready(1));
        rb.push(make_transition(0.0));
        assert!(rb.is_ready(1));
        assert!(!rb.is_ready(2));
    }

    #[test]
    fn test_replay_buffer_clear() {
        let mut rb = ReplayBuffer::new(10).expect("valid capacity");
        rb.push(make_transition(1.0));
        rb.push(make_transition(2.0));
        rb.clear();
        assert_eq!(rb.len(), 0);
        assert!(rb.is_empty());
    }

    // ── PrioritizedReplayBuffer ───────────────────────────────────────────────

    #[test]
    fn test_per_new_invalid_args() {
        assert!(PrioritizedReplayBuffer::new(0, 0.6, 0.4).is_err());
        assert!(PrioritizedReplayBuffer::new(100, -0.1, 0.4).is_err());
        assert!(PrioritizedReplayBuffer::new(100, 0.6, 1.5).is_err());
    }

    #[test]
    fn test_per_push_and_len() {
        let mut per = PrioritizedReplayBuffer::new(10, 0.6, 0.4).expect("valid args");
        per.push(make_transition(1.0));
        per.push(make_transition(2.0));
        assert_eq!(per.len(), 2);
    }

    #[test]
    fn test_per_sample_returns_correct_size() {
        let mut per = PrioritizedReplayBuffer::new(50, 0.6, 0.4).expect("valid args");
        for i in 0..30 {
            per.push_with_priority(make_transition(i as f32), (i + 1) as f32);
        }
        let (transitions, indices, weights) = per.sample(8, 7).expect("sample should succeed");
        assert_eq!(transitions.len(), 8);
        assert_eq!(indices.len(), 8);
        assert_eq!(weights.len(), 8);
    }

    #[test]
    fn test_per_update_priorities() {
        let mut per = PrioritizedReplayBuffer::new(20, 0.6, 0.4).expect("valid args");
        for i in 0..10 {
            per.push_with_priority(make_transition(i as f32), 1.0);
        }
        let indices: Vec<usize> = (0..5).collect();
        let new_prios: Vec<f32> = vec![2.0, 3.0, 1.0, 0.5, 4.0];
        assert!(per.update_priorities(&indices, &new_prios).is_ok());
    }

    #[test]
    fn test_per_update_priorities_length_mismatch() {
        let mut per = PrioritizedReplayBuffer::new(20, 0.6, 0.4).expect("valid args");
        per.push(make_transition(0.0));
        assert!(per.update_priorities(&[0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_per_update_priorities_out_of_range() {
        let mut per = PrioritizedReplayBuffer::new(20, 0.6, 0.4).expect("valid args");
        per.push(make_transition(0.0));
        assert!(per.update_priorities(&[5], &[1.0]).is_err());
    }
}
