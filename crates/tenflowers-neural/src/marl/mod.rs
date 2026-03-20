//! # Multi-Agent Reinforcement Learning (Track Z-MARL)
//!
//! Production-grade MARL algorithms including QMIX, MADDPG, communication
//! protocols (CommNet, TarMAC, ATOC), and cooperative task structures
//! (VDN, COMA, WQMIX, difference rewards).
//!
//! ## Algorithms
//!
//! - **QMIX** — Monotonic value function factorization via hypernetwork mixing.
//! - **MADDPG** — Multi-agent DDPG with centralized training, decentralized execution.
//! - **CommNet** — Continuous mean-field communication between agents.
//! - **TarMAC** — Targeted communication with soft attention over agent messages.
//! - **ATOC** — Attention-based communication trigger.
//! - **COMA** — Counterfactual Multi-Agent policy gradients.
//! - **VDN** — Value Decomposition Networks (sum-mixing).
//! - **WQMIX** — Weighted QMIX (optimistic/central mixing).

pub mod communication;
pub mod cooperative;
pub mod maddpg;
pub mod qmix;
mod tests;
pub mod types;

// § Core types and traits
pub use types::{
    Agent, Experience, MultiAgentEnv, MultiAgentTrainer, MultiAgentTrainerConfig,
    SharedReplayBuffer,
};

// § QMIX
pub use qmix::{AgentQNetwork, MixingNetwork, QmixAgent, QmixTrainer};

// § MADDPG
pub use maddpg::{MaddpgActor, MaddpgAgent, MaddpgCritic, MaddpgTrainer, OuNoise};

// § Communication
pub use communication::{
    AtocAgent, CommChannel, CommNet, MessageDecoder, MessageEncoder, TarmacAgent,
};

// § Cooperative
pub use cooperative::{ComaTrainer, CreditAssignment, TeamRewardShaper, VdnMixer, WqmixTrainer};
