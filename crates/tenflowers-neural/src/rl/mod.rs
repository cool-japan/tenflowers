//! # Reinforcement Learning (Track M)
//!
//! Pure-Rust reinforcement learning primitives for building DQN, REINFORCE,
//! SAC, and PPO-style agents.
//!
//! ## Modules
//!
//! - [`types`]           — Shared type aliases, [`RolloutBuffer`], and [`SumTree`].
//! - [`replay_buffer`]   — Uniform and prioritised experience replay buffers.
//! - [`episode`]         — Episode recording and return/advantage computation.
//! - [`policy_gradient`] — REINFORCE, PPO, DQN loss utilities, and action
//!   selection strategies.
//! - [`dqn`]             — DQN, Double-DQN, and Dueling-DQN agent utilities.
//! - [`sac`]             — Soft Actor-Critic loss functions and agent helpers.
//!
//! ## Quick Example
//!
//! ```rust,ignore
//! use tenflowers_neural::rl::{
//!     replay_buffer::{ReplayBuffer, Transition},
//!     episode::Episode,
//!     policy_gradient::{reinforce_loss, normalize_advantages},
//!     dqn::{DqnAgent, QConfig},
//!     sac::{SacAgent, SacConfig},
//! };
//!
//! // Build a replay buffer.
//! let mut buf = ReplayBuffer::new(10_000)?;
//!
//! // Record a transition and push it.
//! buf.push(Transition {
//!     state: vec![0.0; 8],
//!     action: 2,
//!     reward: 1.0,
//!     next_state: vec![0.1; 8],
//!     done: false,
//! });
//!
//! // Sample a mini-batch.
//! let batch = buf.sample(32, 42)?;
//! ```

pub mod dqn;
pub mod episode;
pub mod policy_gradient;
pub mod replay_buffer;
pub mod sac;
pub mod types;

pub use dqn::{DqnAgent, DuelingHeads, QConfig};
pub use episode::{Episode, EpisodeBuffer};
pub use policy_gradient::{
    boltzmann_action, categorical_entropy, dqn_loss, dqn_td_target, epsilon_greedy,
    normalize_advantages, ppo_clip_loss, ppo_loss, reinforce_loss, PpoLossComponents,
};
pub use replay_buffer::{PrioritizedReplayBuffer, ReplayBuffer, Transition};
pub use sac::{SacAgent, SacConfig};
pub use types::{
    ActionIdx, Done, Reward, RolloutBuffer, StateVec, SumTree, TypedEpisode, TypedTransition,
};
