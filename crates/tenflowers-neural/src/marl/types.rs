//! Core types, traits, and shared infrastructure for MARL.
//!
//! Contains: Experience, Agent trait, MultiAgentEnv trait,
//! SharedReplayBuffer, MultiAgentTrainer, MultiAgentTrainerConfig.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// § 1. Core Interfaces & Shared Types
// ─────────────────────────────────────────────────────────────────────────────

/// A single transition experienced by one agent in a multi-agent system.
#[derive(Debug, Clone)]
pub struct Experience {
    /// Local observation received by this agent.
    pub obs: Vec<f64>,
    /// Action taken.
    pub action: Vec<f64>,
    /// Scalar reward received.
    pub reward: f64,
    /// Next local observation.
    pub next_obs: Vec<f64>,
    /// Whether the episode ended.
    pub done: bool,
    /// Which agent generated this experience.
    pub agent_id: usize,
}

impl Experience {
    /// Construct a new experience tuple.
    pub fn new(
        obs: Vec<f64>,
        action: Vec<f64>,
        reward: f64,
        next_obs: Vec<f64>,
        done: bool,
        agent_id: usize,
    ) -> Self {
        Self {
            obs,
            action,
            reward,
            next_obs,
            done,
            agent_id,
        }
    }
}

/// Agent trait: observe → act, and learn from individual experience tuples.
pub trait Agent: Send + Sync {
    /// Select an action for the given local observation.
    fn act(&self, obs: &[f64]) -> Vec<f64>;

    /// Incorporate a single experience into the agent's learning state.
    fn update(&mut self, experience: &Experience);

    /// Unique identifier for this agent.
    fn id(&self) -> usize;
}

/// Multi-agent environment interface.
pub trait MultiAgentEnv: Send {
    /// Advance the environment by one step given each agent's action.
    ///
    /// Returns `(observations, rewards, done)` where observations and rewards
    /// are indexed by agent.
    fn step(&mut self, actions: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, bool);

    /// Reset the environment and return initial observations per agent.
    fn reset(&mut self) -> Vec<Vec<f64>>;

    /// Number of agents in this environment.
    fn num_agents(&self) -> usize;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 2. Shared Replay Buffer
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe replay buffer shared across agents.
///
/// Wraps a ring-buffer behind an `Arc<Mutex<…>>` so that multiple agents
/// can push transitions concurrently from different threads.
#[derive(Debug, Clone)]
pub struct SharedReplayBuffer {
    inner: Arc<Mutex<SharedReplayBufferInner>>,
}

#[derive(Debug)]
struct SharedReplayBufferInner {
    buffer: VecDeque<Experience>,
    capacity: usize,
}

impl SharedReplayBuffer {
    /// Create a new buffer with the given capacity.
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(TensorError::invalid_argument(
                "SharedReplayBuffer capacity must be > 0".into(),
            ));
        }
        Ok(Self {
            inner: Arc::new(Mutex::new(SharedReplayBufferInner {
                buffer: VecDeque::with_capacity(capacity),
                capacity,
            })),
        })
    }

    /// Push one experience; evicts the oldest when full.
    pub fn push(&self, exp: Experience) -> Result<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| TensorError::other(format!("SharedReplayBuffer lock poisoned: {e}")))?;
        if guard.buffer.len() == guard.capacity {
            guard.buffer.pop_front();
        }
        guard.buffer.push_back(exp);
        Ok(())
    }

    /// Current number of stored experiences.
    pub fn len(&self) -> Result<usize> {
        let guard = self
            .inner
            .lock()
            .map_err(|e| TensorError::other(format!("SharedReplayBuffer lock poisoned: {e}")))?;
        Ok(guard.buffer.len())
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Sample `n` experiences uniformly at random using `seed` for reproducibility.
    pub fn sample(&self, n: usize, seed: u64) -> Result<Vec<Experience>> {
        let guard = self
            .inner
            .lock()
            .map_err(|e| TensorError::other(format!("SharedReplayBuffer lock poisoned: {e}")))?;
        let len = guard.buffer.len();
        if len == 0 {
            return Err(TensorError::invalid_argument(
                "Cannot sample from empty SharedReplayBuffer".into(),
            ));
        }
        let n = n.min(len);
        let mut rng = StdRng::seed_from_u64(seed);
        let items: Vec<usize> = (0..n).map(|_| rng.random_range(0..len)).collect();
        let slice: Vec<&Experience> = guard.buffer.iter().collect();
        Ok(items.iter().map(|&i| slice[i].clone()).collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 3. Multi-Agent Trainer
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for `MultiAgentTrainer`.
#[derive(Debug, Clone)]
pub struct MultiAgentTrainerConfig {
    /// Batch size used to sample from the replay buffer per training step.
    pub batch_size: usize,
    /// Minimum experiences in the buffer before training begins.
    pub min_replay_size: usize,
    /// Whether agents share a single replay buffer or each have their own.
    pub shared_buffer: bool,
}

impl Default for MultiAgentTrainerConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            min_replay_size: 100,
            shared_buffer: true,
        }
    }
}

/// Coordinates training of N heterogeneous agents, with shared or independent
/// replay buffers.
pub struct MultiAgentTrainer {
    agents: Vec<Box<dyn Agent>>,
    shared: Option<SharedReplayBuffer>,
    independent: Vec<SharedReplayBuffer>,
    config: MultiAgentTrainerConfig,
    step_count: u64,
}

impl MultiAgentTrainer {
    /// Construct from a set of agents and configuration.
    pub fn new(agents: Vec<Box<dyn Agent>>, config: MultiAgentTrainerConfig) -> Result<Self> {
        let n = agents.len();
        let (shared, independent) = if config.shared_buffer {
            (Some(SharedReplayBuffer::new(10_000)?), Vec::new())
        } else {
            let bufs: Result<Vec<_>> = (0..n).map(|_| SharedReplayBuffer::new(10_000)).collect();
            (None, bufs?)
        };
        Ok(Self {
            agents,
            shared,
            independent,
            config,
            step_count: 0,
        })
    }

    /// Number of agents managed by this trainer.
    pub fn num_agents(&self) -> usize {
        self.agents.len()
    }

    /// Store a transition in the appropriate buffer.
    pub fn record(&self, exp: Experience) -> Result<()> {
        if let Some(buf) = &self.shared {
            buf.push(exp)
        } else {
            let id = exp.agent_id;
            let buf = self.independent.get(id).ok_or_else(|| {
                TensorError::invalid_argument(format!("No buffer for agent {id}"))
            })?;
            buf.push(exp)
        }
    }

    /// Run one training step: sample from buffer and call `update` on each agent.
    pub fn train_step(&mut self) -> Result<()> {
        self.step_count += 1;
        let seed = self.step_count;
        if let Some(buf) = &self.shared {
            if buf.len()? < self.config.min_replay_size {
                return Ok(());
            }
            let batch = buf.sample(self.config.batch_size, seed)?;
            for agent in &mut self.agents {
                let id = agent.id();
                for exp in batch.iter().filter(|e| e.agent_id == id) {
                    agent.update(exp);
                }
            }
        } else {
            for (i, agent) in self.agents.iter_mut().enumerate() {
                let buf = &self.independent[i];
                if buf.len()? < self.config.min_replay_size {
                    continue;
                }
                let batch = buf.sample(self.config.batch_size, seed + i as u64)?;
                for exp in &batch {
                    agent.update(exp);
                }
            }
        }
        Ok(())
    }

    /// Current training step count.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }
}
