//! Episode management for on-policy reinforcement learning.
//!
//! Provides [`Episode`] for recording and post-processing single rollouts and
//! [`EpisodeBuffer`] for aggregating multiple episodes (as used in algorithms
//! like PPO or REINFORCE with batched updates).

use crate::rl::replay_buffer::Transition;

// ─────────────────────────────────────────────────────────────────────────────
// Episode
// ─────────────────────────────────────────────────────────────────────────────

/// A complete (or partial) episode consisting of a sequence of transitions.
#[derive(Debug, Clone)]
pub struct Episode {
    /// Ordered list of transitions experienced during the episode.
    pub transitions: Vec<Transition>,
    /// Sum of rewards accumulated over the episode. Updated by \[`finish`\].
    pub total_reward: f32,
    /// Number of steps (transitions) in the episode.
    pub length: usize,
}

impl Episode {
    /// Create a new, empty episode.
    pub fn new() -> Self {
        Self {
            transitions: Vec::new(),
            total_reward: 0.0,
            length: 0,
        }
    }

    /// Append a transition to the episode.
    pub fn push(&mut self, transition: Transition) {
        self.transitions.push(transition);
    }

    /// Finalise the episode: compute \[`total_reward`\] and \[`length`\] from the
    /// stored transitions.
    pub fn finish(&mut self) {
        self.length = self.transitions.len();
        self.total_reward = self.transitions.iter().map(|t| t.reward).sum();
    }

    /// Compute Monte-Carlo (discounted) returns for each time step.
    ///
    /// G_t = r_t + γ · r_{t+1} + γ² · r_{t+2} + …
    ///
    /// # Arguments
    /// * `gamma` — discount factor ∈ [0, 1].
    ///
    /// # Returns
    /// A vector of length `self.transitions.len()` where element `i` is G_i.
    pub fn compute_returns(&self, gamma: f32) -> Vec<f32> {
        let n = self.transitions.len();
        if n == 0 {
            return Vec::new();
        }

        let mut returns = vec![0.0_f32; n];
        let mut cumulative = 0.0_f32;

        // Back-propagate from the last step.
        for i in (0..n).rev() {
            cumulative = self.transitions[i].reward + gamma * cumulative;
            // If the episode terminated at step i, do not bootstrap beyond it.
            if self.transitions[i].done {
                cumulative = self.transitions[i].reward;
            }
            returns[i] = cumulative;
        }

        returns
    }

    /// Compute Generalised Advantage Estimates (GAE-λ) for each time step.
    ///
    /// δ_t = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)
    /// A_t = δ_t + (γλ) · δ_{t+1} + (γλ)² · δ_{t+2} + …
    ///
    /// # Arguments
    /// * `gamma`  — discount factor.
    /// * `lambda` — GAE smoothing parameter (0 → TD(0), 1 → Monte-Carlo).
    /// * `values` — state-value estimates V(s_t); must have length
    ///              `self.transitions.len() + 1` (the extra element is
    ///              V(s_{T}) for bootstrapping).  If the vector length equals
    ///              `self.transitions.len()`, the terminal value is assumed 0.
    ///
    /// # Returns
    /// A vector of advantage estimates of the same length as `transitions`.
    pub fn compute_advantages(&self, gamma: f32, lambda: f32, values: &[f32]) -> Vec<f32> {
        let n = self.transitions.len();
        if n == 0 {
            return Vec::new();
        }

        let mut advantages = vec![0.0_f32; n];
        let mut running_gae = 0.0_f32;

        for i in (0..n).rev() {
            let done_mask = if self.transitions[i].done {
                0.0_f32
            } else {
                1.0
            };
            let v_next = if i + 1 < values.len() {
                values[i + 1]
            } else {
                0.0
            };
            let v_curr = if i < values.len() { values[i] } else { 0.0 };

            let delta = self.transitions[i].reward + gamma * v_next * done_mask - v_curr;
            running_gae = delta + gamma * lambda * done_mask * running_gae;
            advantages[i] = running_gae;
        }

        advantages
    }
}

impl Default for Episode {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EpisodeBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregates multiple finished episodes.
///
/// Used by batch on-policy algorithms (e.g. PPO) that collect an entire batch
/// of rollouts before performing a gradient update.
#[derive(Debug)]
pub struct EpisodeBuffer {
    episodes: Vec<Episode>,
    max_episodes: usize,
}

impl EpisodeBuffer {
    /// Create a new `EpisodeBuffer` with the given maximum capacity.
    pub fn new(max_episodes: usize) -> Self {
        Self {
            episodes: Vec::with_capacity(max_episodes),
            max_episodes,
        }
    }

    /// Append a finished episode to the buffer, evicting the oldest if full.
    pub fn push_episode(&mut self, episode: Episode) {
        if self.max_episodes > 0 && self.episodes.len() >= self.max_episodes {
            self.episodes.remove(0);
        }
        self.episodes.push(episode);
    }

    /// Iterate over every transition across all stored episodes.
    pub fn all_transitions(&self) -> Vec<&Transition> {
        self.episodes
            .iter()
            .flat_map(|ep| ep.transitions.iter())
            .collect()
    }

    /// Clear all stored episodes.
    pub fn clear(&mut self) {
        self.episodes.clear();
    }

    /// Mean episode total reward across all stored episodes, or `None` if empty.
    pub fn mean_reward(&self) -> Option<f32> {
        if self.episodes.is_empty() {
            return None;
        }
        let sum: f32 = self.episodes.iter().map(|ep| ep.total_reward).sum();
        Some(sum / self.episodes.len() as f32)
    }

    /// Mean episode length across all stored episodes, or `None` if empty.
    pub fn mean_length(&self) -> Option<f32> {
        if self.episodes.is_empty() {
            return None;
        }
        let sum: f32 = self.episodes.iter().map(|ep| ep.length as f32).sum();
        Some(sum / self.episodes.len() as f32)
    }

    /// Number of episodes currently stored.
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// Returns `true` if no episodes are stored.
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_episode(rewards: &[f32], dones: &[bool]) -> Episode {
        let mut ep = Episode::new();
        for (&r, &d) in rewards.iter().zip(dones.iter()) {
            ep.push(Transition {
                state: vec![0.0],
                action: 0,
                reward: r,
                next_state: vec![0.0],
                done: d,
            });
        }
        ep.finish();
        ep
    }

    // ── Episode ───────────────────────────────────────────────────────────────

    #[test]
    fn test_episode_new_is_empty() {
        let ep = Episode::new();
        assert_eq!(ep.length, 0);
        assert_eq!(ep.total_reward, 0.0);
        assert!(ep.transitions.is_empty());
    }

    #[test]
    fn test_episode_push_and_finish() {
        let ep = make_episode(&[1.0, 2.0, 3.0], &[false, false, true]);
        assert_eq!(ep.length, 3);
        assert!((ep.total_reward - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_episode_compute_returns_no_discount() {
        // γ = 1: returns[0] = 1+2+3 = 6, returns[1] = 2+3 = 5, returns[2] = 3
        let ep = make_episode(&[1.0, 2.0, 3.0], &[false, false, false]);
        let returns = ep.compute_returns(1.0);
        assert!((returns[0] - 6.0).abs() < 1e-5, "returns[0]={}", returns[0]);
        assert!((returns[1] - 5.0).abs() < 1e-5, "returns[1]={}", returns[1]);
        assert!((returns[2] - 3.0).abs() < 1e-5, "returns[2]={}", returns[2]);
    }

    #[test]
    fn test_episode_compute_returns_with_discount() {
        // γ = 0.9, rewards = [1, 1, 1]
        // G[2] = 1
        // G[1] = 1 + 0.9 = 1.9
        // G[0] = 1 + 0.9*1.9 = 1 + 1.71 = 2.71
        let ep = make_episode(&[1.0, 1.0, 1.0], &[false, false, false]);
        let returns = ep.compute_returns(0.9);
        assert!((returns[2] - 1.0).abs() < 1e-5);
        assert!((returns[1] - 1.9).abs() < 1e-5, "got {}", returns[1]);
        assert!((returns[0] - 2.71).abs() < 1e-4, "got {}", returns[0]);
    }

    #[test]
    fn test_episode_compute_returns_empty() {
        let ep = Episode::new();
        assert!(ep.compute_returns(0.99).is_empty());
    }

    #[test]
    fn test_episode_compute_advantages_gae() {
        // 3-step episode, all done=false.
        // values = [0.5, 0.5, 0.5, 0.0]  (last element is V(s_T))
        // rewards = [1.0, 1.0, 1.0]
        // δ_2 = 1.0 + 0.99*0.0 - 0.5 = 0.5
        // δ_1 = 1.0 + 0.99*0.5 - 0.5 = 0.5 + 0.495 = 0.995
        // δ_0 = 1.0 + 0.99*0.5 - 0.5 = 0.995
        // A_2 = 0.5
        // A_1 = 0.995 + 0.99*0.95*0.5 = ...
        let ep = make_episode(&[1.0, 1.0, 1.0], &[false, false, false]);
        let values = [0.5, 0.5, 0.5, 0.0];
        let adv = ep.compute_advantages(0.99, 0.95, &values);
        assert_eq!(adv.len(), 3);
        // All advantages should be positive given rewards > baseline.
        for &a in &adv {
            assert!(a > 0.0, "advantage should be positive, got {}", a);
        }
    }

    #[test]
    fn test_episode_compute_advantages_done_mask() {
        // Episode ends at step 1 (done=true), so next-state value at step 1 is ignored.
        let ep = make_episode(&[1.0, 1.0], &[false, true]);
        let values = [0.0, 0.0, 999.0]; // V(s_T) = 999 — should be masked out.
        let adv = ep.compute_advantages(0.99, 0.95, &values);
        // adv[1]: delta = 1.0 + 0.99*999*0 - 0.0 = 1.0 => A[1] = 1.0
        assert!((adv[1] - 1.0).abs() < 1e-5, "adv[1]={}", adv[1]);
    }

    // ── EpisodeBuffer ─────────────────────────────────────────────────────────

    #[test]
    fn test_episode_buffer_push_and_len() {
        let mut buf = EpisodeBuffer::new(5);
        assert_eq!(buf.len(), 0);
        buf.push_episode(make_episode(&[1.0], &[true]));
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_episode_buffer_mean_reward_empty() {
        let buf = EpisodeBuffer::new(10);
        assert!(buf.mean_reward().is_none());
    }

    #[test]
    fn test_episode_buffer_mean_reward() {
        let mut buf = EpisodeBuffer::new(10);
        buf.push_episode(make_episode(&[1.0, 2.0], &[false, true])); // total = 3
        buf.push_episode(make_episode(&[4.0, 5.0], &[false, true])); // total = 9
        let mean = buf.mean_reward().expect("non-empty buffer");
        assert!((mean - 6.0).abs() < 1e-5, "mean={}", mean);
    }

    #[test]
    fn test_episode_buffer_mean_length() {
        let mut buf = EpisodeBuffer::new(10);
        buf.push_episode(make_episode(&[1.0, 2.0], &[false, true])); // len = 2
        buf.push_episode(make_episode(&[1.0, 2.0, 3.0], &[false, false, true])); // len = 3
        let ml = buf.mean_length().expect("non-empty");
        assert!((ml - 2.5).abs() < 1e-5, "mean_length={}", ml);
    }

    #[test]
    fn test_episode_buffer_all_transitions() {
        let mut buf = EpisodeBuffer::new(10);
        buf.push_episode(make_episode(&[1.0, 2.0], &[false, true]));
        buf.push_episode(make_episode(&[3.0], &[true]));
        let all = buf.all_transitions();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_episode_buffer_clear() {
        let mut buf = EpisodeBuffer::new(10);
        buf.push_episode(make_episode(&[1.0], &[true]));
        buf.clear();
        assert!(buf.is_empty());
        assert!(buf.mean_reward().is_none());
    }

    #[test]
    fn test_episode_buffer_evicts_oldest_when_full() {
        let mut buf = EpisodeBuffer::new(2);
        buf.push_episode(make_episode(&[1.0], &[true]));
        buf.push_episode(make_episode(&[2.0], &[true]));
        buf.push_episode(make_episode(&[3.0], &[true])); // evicts first
        assert_eq!(buf.len(), 2);
        // Total rewards should be 2 and 3.
        let mean = buf.mean_reward().expect("non-empty");
        assert!((mean - 2.5).abs() < 1e-5, "mean={}", mean);
    }
}
