//! Shared type definitions and data structures for reinforcement learning.
//!
//! Provides fundamental RL types including transition tuples, episode buffers,
//! rollout buffers (on-policy), and the [`SumTree`] used by
//! [`PrioritizedReplayBuffer`](super::replay_buffer::PrioritizedReplayBuffer).

use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Primitive type aliases
// ─────────────────────────────────────────────────────────────────────────────

/// Scalar reward signal.
pub type Reward = f32;

/// Whether an episode ended at a given transition.
pub type Done = bool;

/// Flat continuous state vector.
pub type StateVec = Vec<f32>;

/// Discrete action index.
pub type ActionIdx = usize;

// ─────────────────────────────────────────────────────────────────────────────
// Transition
// ─────────────────────────────────────────────────────────────────────────────

/// A single `(s, a, r, s', done)` transition tuple.
///
/// Generic over the state type `S` and the action type `A`, enabling
/// reuse with both flat vectors and structured state representations.
#[derive(Debug, Clone)]
pub struct TypedTransition<S, A> {
    /// Observation at the start of the transition.
    pub state: S,
    /// Action taken.
    pub action: A,
    /// Scalar reward received.
    pub reward: Reward,
    /// Observation after applying the action.
    pub next_state: S,
    /// `true` if the episode terminated at this step.
    pub done: Done,
}

// ─────────────────────────────────────────────────────────────────────────────
// Episode
// ─────────────────────────────────────────────────────────────────────────────

/// A complete (or partial) episode as a sequence of typed transitions.
#[derive(Debug, Clone)]
pub struct TypedEpisode<S, A> {
    /// Ordered transitions making up the episode.
    pub transitions: Vec<TypedTransition<S, A>>,
    /// Sum of all rewards; populated by [`finish`](TypedEpisode::finish).
    pub total_reward: Reward,
    /// Number of steps; populated by [`finish`](TypedEpisode::finish).
    pub length: usize,
}

impl<S: Clone, A: Clone> TypedEpisode<S, A> {
    /// Create an empty episode.
    pub fn new() -> Self {
        Self {
            transitions: Vec::new(),
            total_reward: 0.0,
            length: 0,
        }
    }

    /// Append a transition.
    pub fn push(&mut self, transition: TypedTransition<S, A>) {
        self.transitions.push(transition);
    }

    /// Finalise the episode by computing `total_reward` and `length`.
    pub fn finish(&mut self) {
        self.length = self.transitions.len();
        self.total_reward = self.transitions.iter().map(|t| t.reward).sum();
    }
}

impl<S: Clone, A: Clone> Default for TypedEpisode<S, A> {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RolloutBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// On-policy rollout buffer used by algorithms such as REINFORCE and PPO.
///
/// Collects transitions for a single rollout, then computes discounted
/// returns and advantages in batch.
#[derive(Debug)]
pub struct RolloutBuffer<S, A> {
    /// Collected transitions (cleared after each policy update).
    pub transitions: Vec<TypedTransition<S, A>>,
    /// Discounted returns G_t computed by [`compute_returns`](RolloutBuffer::compute_returns).
    pub returns: Vec<f32>,
    /// Advantage estimates computed by [`advantages`](RolloutBuffer::advantages).
    pub advantage_estimates: Vec<f32>,
}

impl<S: Clone, A: Clone> RolloutBuffer<S, A> {
    /// Create an empty rollout buffer.
    pub fn new() -> Self {
        Self {
            transitions: Vec::new(),
            returns: Vec::new(),
            advantage_estimates: Vec::new(),
        }
    }

    /// Append a transition to the buffer.
    pub fn push(&mut self, transition: TypedTransition<S, A>) {
        self.transitions.push(transition);
    }

    /// Compute discounted Monte-Carlo returns for every stored transition.
    ///
    /// G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …
    ///
    /// The `done` flag zeroes out the bootstrap from the *next* step (i.e. when
    /// an episode terminates at step `t`, `G_t = r_t`).
    pub fn compute_returns(&mut self, gamma: f32) {
        let n = self.transitions.len();
        self.returns = vec![0.0_f32; n];
        let mut cumulative = 0.0_f32;
        for i in (0..n).rev() {
            cumulative = self.transitions[i].reward + gamma * cumulative;
            // Terminal step: no bootstrap from future.
            if self.transitions[i].done {
                cumulative = self.transitions[i].reward;
            }
            self.returns[i] = cumulative;
        }
    }

    /// Return a reference to the pre-computed returns slice.
    ///
    /// Call [`compute_returns`](RolloutBuffer::compute_returns) first.
    pub fn returns_slice(&self) -> &[f32] {
        &self.returns
    }

    /// Return advantage estimates.
    ///
    /// Computes `G_t - baseline` using the stored returns as G_t and a
    /// per-step baseline passed in as `values` (e.g. critic predictions).
    ///
    /// If `values` is empty a zero baseline is used (plain returns).
    pub fn advantages(&mut self, values: &[f32]) -> &[f32] {
        let n = self.returns.len();
        self.advantage_estimates = (0..n)
            .map(|i| {
                let v = if i < values.len() { values[i] } else { 0.0 };
                self.returns[i] - v
            })
            .collect();
        &self.advantage_estimates
    }

    /// Clear all stored data, ready for the next rollout.
    pub fn clear(&mut self) {
        self.transitions.clear();
        self.returns.clear();
        self.advantage_estimates.clear();
    }

    /// Number of transitions currently stored.
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// Returns `true` when the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }
}

impl<S: Clone, A: Clone> Default for RolloutBuffer<S, A> {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SumTree
// ─────────────────────────────────────────────────────────────────────────────

/// Binary segment-tree that stores per-leaf priorities and supports O(log n)
/// prefix-sum queries and O(log n) updates.
///
/// Used internally by
/// [`PrioritizedReplayBuffer`](super::replay_buffer::PrioritizedReplayBuffer)
/// to enable O(log n) stratified sampling.
///
/// ## Layout (1-indexed)
///
/// Array index 0 is unused.  The root is at index 1.  Node `i` has children
/// `2·i` (left) and `2·i+1` (right).  Leaf `k` (0-indexed) is stored at
/// tree index `capacity + k`.  Array size is `2 * capacity + 1`.
#[derive(Debug, Clone)]
pub struct SumTree {
    /// Flat storage (1-indexed): `tree[1]` is the root (total sum), index 0 unused.
    tree: Vec<f32>,
    /// Number of leaves (== buffer capacity).
    capacity: usize,
    /// 0-based index of the leaf that will be written next (circular).
    write_idx: usize,
    /// Current number of valid entries (≤ capacity).
    size: usize,
}

impl SumTree {
    /// Create a new `SumTree` with the given leaf capacity.
    ///
    /// # Errors
    /// Returns an error when `capacity == 0`.
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(TensorError::invalid_argument_op(
                "SumTree::new",
                "capacity must be greater than zero",
            ));
        }
        Ok(Self {
            // Index 0 unused; leaves at [capacity .. 2*capacity); 2*capacity slots needed.
            tree: vec![0.0_f32; 2 * capacity + 1],
            capacity,
            write_idx: 0,
            size: 0,
        })
    }

    /// Return the total sum of all leaf priorities (root node at index 1).
    pub fn total(&self) -> f32 {
        self.tree[1]
    }

    /// Return the number of valid entries currently stored.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns `true` if no priorities have been inserted.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Add (or overwrite) the next leaf with `priority` and return its leaf index.
    pub fn push(&mut self, priority: f32) -> usize {
        let leaf_idx = self.write_idx;
        self.update_leaf(leaf_idx, priority);
        self.write_idx = (self.write_idx + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }
        leaf_idx
    }

    /// Update the priority of leaf `leaf_idx`.
    ///
    /// # Errors
    /// Returns an error if `leaf_idx >= capacity`.
    pub fn update(&mut self, leaf_idx: usize, priority: f32) -> Result<()> {
        if leaf_idx >= self.capacity {
            return Err(TensorError::invalid_argument_op(
                "SumTree::update",
                &format!(
                    "leaf_idx {} out of range (capacity {})",
                    leaf_idx, self.capacity
                ),
            ));
        }
        self.update_leaf(leaf_idx, priority);
        Ok(())
    }

    /// Retrieve the priority of leaf `leaf_idx`.
    ///
    /// Returns `0.0` for out-of-range indices.
    pub fn get(&self, leaf_idx: usize) -> f32 {
        if leaf_idx >= self.capacity {
            return 0.0;
        }
        self.tree[self.capacity + leaf_idx]
    }

    /// Retrieve the leaf index whose prefix-sum segment contains `value`.
    ///
    /// `value` should be in `[0, total())`.  Returns `(leaf_idx, priority)`.
    ///
    /// # Errors
    /// Returns an error if the tree is empty.
    pub fn sample(&self, value: f32) -> Result<(usize, f32)> {
        if self.size == 0 {
            return Err(TensorError::invalid_argument_op(
                "SumTree::sample",
                "cannot sample from an empty SumTree",
            ));
        }
        let clamped = value.max(0.0).min(self.total() * (1.0 - 1e-7));
        let tree_node = self.retrieve(1, clamped);
        // Leaf k is at tree index `capacity + k`.
        let leaf_idx = tree_node - self.capacity;
        let priority = self.tree[tree_node];
        Ok((leaf_idx, priority))
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    /// Descend from `node` following prefix sums until a leaf is reached.
    ///
    /// Leaves are at indices `[capacity, 2*capacity)` in the 1-indexed tree.
    fn retrieve(&self, node: usize, value: f32) -> usize {
        let left = 2 * node;
        // If left child is out of the internal-node range, we are at a leaf.
        if left >= 2 * self.capacity {
            return node;
        }
        let right = left + 1;
        if value <= self.tree[left] {
            self.retrieve(left, value)
        } else {
            self.retrieve(right, value - self.tree[left])
        }
    }

    /// Set `tree[capacity + leaf_idx] = priority` and propagate upward.
    fn update_leaf(&mut self, leaf_idx: usize, priority: f32) {
        let mut idx = self.capacity + leaf_idx;
        self.tree[idx] = priority;
        // Propagate upward to root (index 1).
        while idx > 1 {
            let parent = idx / 2;
            let sibling = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
            self.tree[parent] = self.tree[idx] + self.tree[sibling];
            idx = parent;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── type aliases ──────────────────────────────────────────────────────────

    #[test]
    fn test_typed_transition_roundtrip() {
        let t: TypedTransition<StateVec, ActionIdx> = TypedTransition {
            state: vec![1.0, 2.0],
            action: 3,
            reward: 0.5,
            next_state: vec![1.1, 2.1],
            done: false,
        };
        assert_eq!(t.action, 3_usize);
        assert!((t.reward - 0.5).abs() < 1e-7);
        assert!(!t.done);
    }

    // ── TypedEpisode ──────────────────────────────────────────────────────────

    #[test]
    fn test_typed_episode_finish_computes_totals() {
        let mut ep: TypedEpisode<StateVec, ActionIdx> = TypedEpisode::new();
        for i in 0..3_usize {
            ep.push(TypedTransition {
                state: vec![0.0],
                action: i,
                reward: (i + 1) as f32,
                next_state: vec![0.0],
                done: i == 2,
            });
        }
        ep.finish();
        assert_eq!(ep.length, 3);
        assert!((ep.total_reward - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_typed_episode_default_is_empty() {
        let ep: TypedEpisode<Vec<f32>, usize> = TypedEpisode::default();
        assert_eq!(ep.length, 0);
        assert!(ep.transitions.is_empty());
    }

    // ── RolloutBuffer ─────────────────────────────────────────────────────────

    fn make_rollout(rewards: &[f32], dones: &[bool]) -> RolloutBuffer<StateVec, ActionIdx> {
        let mut buf: RolloutBuffer<StateVec, ActionIdx> = RolloutBuffer::new();
        for (&r, &d) in rewards.iter().zip(dones.iter()) {
            buf.push(TypedTransition {
                state: vec![0.0],
                action: 0,
                reward: r,
                next_state: vec![0.0],
                done: d,
            });
        }
        buf
    }

    #[test]
    fn test_rollout_buffer_compute_returns_no_discount() {
        // γ=1: returns = [1+2+3, 2+3, 3]
        let mut buf = make_rollout(&[1.0, 2.0, 3.0], &[false, false, false]);
        buf.compute_returns(1.0);
        let r = buf.returns_slice();
        assert!((r[0] - 6.0).abs() < 1e-5, "r[0]={}", r[0]);
        assert!((r[1] - 5.0).abs() < 1e-5, "r[1]={}", r[1]);
        assert!((r[2] - 3.0).abs() < 1e-5, "r[2]={}", r[2]);
    }

    #[test]
    fn test_rollout_buffer_compute_returns_with_discount() {
        // γ=0.9, rewards=[1,1,1]
        // G[2]=1, G[1]=1+0.9=1.9, G[0]=1+0.9*1.9=2.71
        let mut buf = make_rollout(&[1.0, 1.0, 1.0], &[false, false, false]);
        buf.compute_returns(0.9);
        let r = buf.returns_slice();
        assert!((r[2] - 1.0).abs() < 1e-5);
        assert!((r[1] - 1.9).abs() < 1e-5, "r[1]={}", r[1]);
        assert!((r[0] - 2.71).abs() < 1e-4, "r[0]={}", r[0]);
    }

    #[test]
    fn test_rollout_buffer_done_resets_cumulative() {
        // rewards=[1,2,3], dones=[false, true, false], gamma=1
        // i=2: cumul=3+0=3, not done → returns[2]=3
        // i=1: cumul=2+1*3=5, done=true → cumul=2, returns[1]=2
        // i=0: cumul=1+1*2=3, not done → returns[0]=3
        let mut buf = make_rollout(&[1.0, 2.0, 3.0], &[false, true, false]);
        buf.compute_returns(1.0);
        let r = buf.returns_slice();
        assert!((r[2] - 3.0).abs() < 1e-5, "r[2]={}", r[2]);
        assert!((r[1] - 2.0).abs() < 1e-5, "r[1]={}", r[1]);
        assert!((r[0] - 3.0).abs() < 1e-5, "r[0]={}", r[0]);
    }

    #[test]
    fn test_rollout_buffer_advantages_vs_values() {
        // rewards=[1,1,1], done=[false,false,false], gamma=1.0
        // compute_returns (reversed): G[2]=1, G[1]=1+1=2, G[0]=1+2=3
        let mut buf = make_rollout(&[1.0, 1.0, 1.0], &[false, false, false]);
        buf.compute_returns(1.0);
        let values = [2.0_f32, 1.5, 1.0];
        let adv = buf.advantages(&values).to_vec();
        // G - V: [3-2, 2-1.5, 1-1] = [1, 0.5, 0]
        assert!((adv[0] - 1.0).abs() < 1e-5, "adv[0]={}", adv[0]);
        assert!((adv[1] - 0.5).abs() < 1e-5, "adv[1]={}", adv[1]);
        assert!((adv[2] - 0.0).abs() < 1e-5, "adv[2]={}", adv[2]);
    }

    #[test]
    fn test_rollout_buffer_advantages_empty_values() {
        let mut buf = make_rollout(&[1.0, 2.0], &[false, false]);
        buf.compute_returns(1.0);
        let adv = buf.advantages(&[]).to_vec();
        // No baseline → advantages == returns
        let r = buf.returns_slice().to_vec();
        for (a, g) in adv.iter().zip(r.iter()) {
            assert!((a - g).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rollout_buffer_clear() {
        let mut buf = make_rollout(&[1.0], &[true]);
        buf.compute_returns(0.99);
        buf.clear();
        assert!(buf.is_empty());
        assert!(buf.returns.is_empty());
    }

    // ── SumTree ───────────────────────────────────────────────────────────────

    #[test]
    fn test_sumtree_new_zero_capacity_fails() {
        assert!(SumTree::new(0).is_err());
    }

    #[test]
    fn test_sumtree_push_and_total() {
        let mut st = SumTree::new(4).expect("valid");
        st.push(1.0);
        st.push(2.0);
        st.push(3.0);
        assert!((st.total() - 6.0).abs() < 1e-6, "total={}", st.total());
        assert_eq!(st.len(), 3);
    }

    #[test]
    fn test_sumtree_overwrite_circular() {
        let mut st = SumTree::new(2).expect("valid");
        st.push(1.0);
        st.push(2.0);
        // Next push overwrites leaf 0.
        st.push(5.0);
        // Total should be 5 + 2 = 7.
        assert!((st.total() - 7.0).abs() < 1e-5, "total={}", st.total());
    }

    #[test]
    fn test_sumtree_update_leaf() {
        let mut st = SumTree::new(4).expect("valid");
        st.push(1.0);
        st.push(1.0);
        st.push(1.0);
        st.push(1.0);
        assert!((st.total() - 4.0).abs() < 1e-6);
        st.update(0, 5.0).expect("valid idx");
        assert!((st.total() - 8.0).abs() < 1e-6, "total={}", st.total());
    }

    #[test]
    fn test_sumtree_update_out_of_range_fails() {
        let mut st = SumTree::new(4).expect("valid");
        assert!(st.update(10, 1.0).is_err());
    }

    #[test]
    fn test_sumtree_sample_returns_valid_leaf() {
        let mut st = SumTree::new(4).expect("valid");
        st.push(1.0);
        st.push(2.0);
        st.push(3.0);
        st.push(4.0);
        // Sample at value=0 should hit the first leaf.
        let (idx, _p) = st.sample(0.0).expect("non-empty");
        assert!(idx < 4, "idx out of range: {}", idx);
    }

    #[test]
    fn test_sumtree_sample_empty_fails() {
        let st = SumTree::new(4).expect("valid");
        assert!(st.sample(0.5).is_err());
    }

    #[test]
    fn test_sumtree_get_priority() {
        let mut st = SumTree::new(4).expect("valid");
        st.push(3.5);
        // Leaf 0 should have priority 3.5.
        let p = st.get(0);
        assert!((p - 3.5).abs() < 1e-6, "priority={}", p);
    }
}
