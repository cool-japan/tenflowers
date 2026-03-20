//! Communication protocols for multi-agent systems.
//!
//! Contains: CommChannel, MessageEncoder, MessageDecoder,
//! CommNet, AtocAgent, TarmacAgent.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// § 6. Communication Protocols
// ─────────────────────────────────────────────────────────────────────────────

/// Differentiable communication channel: encodes an agent's hidden state into a
/// message vector and can decode incoming messages back into a representation.
#[derive(Debug, Clone)]
pub struct CommChannel {
    pub msg_dim: usize,
    pub hidden_dim: usize,
    encode_w: Vec<f64>,
    encode_b: Vec<f64>,
    decode_w: Vec<f64>,
    decode_b: Vec<f64>,
}

impl CommChannel {
    /// Create encode/decode projection matrices.
    pub fn new(hidden_dim: usize, msg_dim: usize, seed: u64) -> Result<Self> {
        if hidden_dim == 0 || msg_dim == 0 {
            return Err(TensorError::invalid_argument(
                "CommChannel dimensions must be > 0".into(),
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let se = (2.0_f64 / hidden_dim as f64).sqrt();
        let sd = (2.0_f64 / msg_dim as f64).sqrt();
        let encode_w = (0..msg_dim * hidden_dim)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * se)
            .collect();
        let encode_b = vec![0.0_f64; msg_dim];
        let decode_w = (0..hidden_dim * msg_dim)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * sd)
            .collect();
        let decode_b = vec![0.0_f64; hidden_dim];
        Ok(Self {
            msg_dim,
            hidden_dim,
            encode_w,
            encode_b,
            decode_w,
            decode_b,
        })
    }

    /// Encode a hidden state vector into a message.
    pub fn encode(&self, hidden: &[f64]) -> Result<Vec<f64>> {
        if hidden.len() != self.hidden_dim {
            return Err(TensorError::invalid_argument(format!(
                "CommChannel encode: expected hidden_dim={} got {}",
                self.hidden_dim,
                hidden.len()
            )));
        }
        let mut msg = vec![0.0_f64; self.msg_dim];
        for i in 0..self.msg_dim {
            let mut v = self.encode_b[i];
            for j in 0..self.hidden_dim {
                v += self.encode_w[i * self.hidden_dim + j] * hidden[j];
            }
            msg[i] = v.tanh();
        }
        Ok(msg)
    }

    /// Decode a message back into a hidden-space representation.
    pub fn decode(&self, msg: &[f64]) -> Result<Vec<f64>> {
        if msg.len() != self.msg_dim {
            return Err(TensorError::invalid_argument(format!(
                "CommChannel decode: expected msg_dim={} got {}",
                self.msg_dim,
                msg.len()
            )));
        }
        let mut h = vec![0.0_f64; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut v = self.decode_b[i];
            for j in 0..self.msg_dim {
                v += self.decode_w[i * self.msg_dim + j] * msg[j];
            }
            h[i] = v.tanh();
        }
        Ok(h)
    }
}

/// Encode agent local observations into messages (linear projection + tanh).
#[derive(Debug, Clone)]
pub struct MessageEncoder {
    obs_dim: usize,
    msg_dim: usize,
    w: Vec<f64>,
    b: Vec<f64>,
}

impl MessageEncoder {
    /// Create encoder.
    pub fn new(obs_dim: usize, msg_dim: usize, seed: u64) -> Result<Self> {
        if obs_dim == 0 || msg_dim == 0 {
            return Err(TensorError::invalid_argument(
                "MessageEncoder: dims must be > 0".into(),
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (2.0_f64 / obs_dim as f64).sqrt();
        let w = (0..msg_dim * obs_dim)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
            .collect();
        let b = vec![0.0_f64; msg_dim];
        Ok(Self {
            obs_dim,
            msg_dim,
            w,
            b,
        })
    }

    /// Encode observation into message.
    pub fn encode(&self, obs: &[f64]) -> Result<Vec<f64>> {
        if obs.len() != self.obs_dim {
            return Err(TensorError::invalid_argument(format!(
                "MessageEncoder: obs_dim mismatch {} vs {}",
                self.obs_dim,
                obs.len()
            )));
        }
        let mut msg = vec![0.0_f64; self.msg_dim];
        for i in 0..self.msg_dim {
            let mut v = self.b[i];
            for j in 0..self.obs_dim {
                v += self.w[i * self.obs_dim + j] * obs[j];
            }
            msg[i] = v.tanh();
        }
        Ok(msg)
    }
}

/// Decode a received message into a local representation.
#[derive(Debug, Clone)]
pub struct MessageDecoder {
    msg_dim: usize,
    out_dim: usize,
    w: Vec<f64>,
    b: Vec<f64>,
}

impl MessageDecoder {
    /// Create decoder.
    pub fn new(msg_dim: usize, out_dim: usize, seed: u64) -> Result<Self> {
        if msg_dim == 0 || out_dim == 0 {
            return Err(TensorError::invalid_argument(
                "MessageDecoder: dims must be > 0".into(),
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (2.0_f64 / msg_dim as f64).sqrt();
        let w = (0..out_dim * msg_dim)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
            .collect();
        let b = vec![0.0_f64; out_dim];
        Ok(Self {
            msg_dim,
            out_dim,
            w,
            b,
        })
    }

    /// Decode message into output representation.
    pub fn decode(&self, msg: &[f64]) -> Result<Vec<f64>> {
        if msg.len() != self.msg_dim {
            return Err(TensorError::invalid_argument(format!(
                "MessageDecoder: msg_dim mismatch {} vs {}",
                self.msg_dim,
                msg.len()
            )));
        }
        let mut out = vec![0.0_f64; self.out_dim];
        for i in 0..self.out_dim {
            let mut v = self.b[i];
            for j in 0..self.msg_dim {
                v += self.w[i * self.msg_dim + j] * msg[j];
            }
            out[i] = v.tanh();
        }
        Ok(out)
    }
}

/// CommNet: continuous communication via mean-field message passing.
///
/// Each agent broadcasts a hidden state; every agent receives the mean
/// of all other agents' messages and incorporates it into its next hidden.
#[derive(Debug, Clone)]
pub struct CommNet {
    pub n_agents: usize,
    channels: Vec<CommChannel>,
}

impl CommNet {
    /// Create one `CommChannel` per agent.
    pub fn new(n_agents: usize, hidden_dim: usize, msg_dim: usize, seed: u64) -> Result<Self> {
        if n_agents == 0 {
            return Err(TensorError::invalid_argument(
                "CommNet requires n_agents > 0".into(),
            ));
        }
        let channels: Result<Vec<_>> = (0..n_agents)
            .map(|i| CommChannel::new(hidden_dim, msg_dim, seed + i as u64 * 37))
            .collect();
        Ok(Self {
            n_agents,
            channels: channels?,
        })
    }

    /// One communication round: each agent encodes, then receives mean message.
    ///
    /// Returns the decoded incoming message for each agent.
    pub fn communicate(&self, hiddens: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if hiddens.len() != self.n_agents {
            return Err(TensorError::invalid_argument(format!(
                "CommNet: expected {} hiddens, got {}",
                self.n_agents,
                hiddens.len()
            )));
        }
        // Encode all
        let msgs: Result<Vec<Vec<f64>>> = hiddens
            .iter()
            .zip(&self.channels)
            .map(|(h, ch)| ch.encode(h))
            .collect();
        let msgs = msgs?;
        let msg_dim = msgs[0].len();
        // Compute mean message for each agent (excluding self)
        let mut results = Vec::with_capacity(self.n_agents);
        for i in 0..self.n_agents {
            let mut mean = vec![0.0_f64; msg_dim];
            let count = (self.n_agents - 1).max(1) as f64;
            for (j, msg) in msgs.iter().enumerate() {
                if j != i {
                    for (m, v) in mean.iter_mut().zip(msg.iter()) {
                        *m += v / count;
                    }
                }
            }
            let decoded = self.channels[i].decode(&mean)?;
            results.push(decoded);
        }
        Ok(results)
    }

    /// Message dimension.
    pub fn msg_dim(&self) -> usize {
        if self.channels.is_empty() {
            0
        } else {
            self.channels[0].msg_dim
        }
    }
}

/// ATOC: Attention-based communication — each agent decides whether to
/// communicate based on a learned attention weight over a thought vector.
#[derive(Debug, Clone)]
pub struct AtocAgent {
    pub agent_id: usize,
    obs_dim: usize,
    thought_dim: usize,
    msg_dim: usize,
    /// Thought network: obs → thought_dim (tanh)
    w_thought: Vec<f64>,
    b_thought: Vec<f64>,
    /// Attention query: thought_dim → 1 (sigmoid for gating)
    w_attn: Vec<f64>,
    b_attn: f64,
    pub channel: CommChannel,
}

impl AtocAgent {
    /// Create ATOC agent.
    pub fn new(
        agent_id: usize,
        obs_dim: usize,
        thought_dim: usize,
        msg_dim: usize,
        seed: u64,
    ) -> Result<Self> {
        if obs_dim == 0 || thought_dim == 0 || msg_dim == 0 {
            return Err(TensorError::invalid_argument(
                "AtocAgent: dims must be > 0".into(),
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let s1 = (2.0_f64 / obs_dim as f64).sqrt();
        let s2 = (2.0_f64 / thought_dim as f64).sqrt();
        let w_thought: Vec<f64> = (0..thought_dim * obs_dim)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * s1)
            .collect();
        let b_thought = vec![0.0_f64; thought_dim];
        let w_attn: Vec<f64> = (0..thought_dim)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * s2)
            .collect();
        let b_attn = 0.0_f64;
        let channel = CommChannel::new(thought_dim, msg_dim, seed + 999)?;
        Ok(Self {
            agent_id,
            obs_dim,
            thought_dim,
            msg_dim,
            w_thought,
            b_thought,
            w_attn,
            b_attn,
            channel,
        })
    }

    /// Compute thought vector from observation.
    pub fn compute_thought(&self, obs: &[f64]) -> Result<Vec<f64>> {
        if obs.len() != self.obs_dim {
            return Err(TensorError::invalid_argument(format!(
                "AtocAgent: obs_dim mismatch {} vs {}",
                self.obs_dim,
                obs.len()
            )));
        }
        let mut t = vec![0.0_f64; self.thought_dim];
        for i in 0..self.thought_dim {
            let mut v = self.b_thought[i];
            for j in 0..self.obs_dim {
                v += self.w_thought[i * self.obs_dim + j] * obs[j];
            }
            t[i] = v.tanh();
        }
        Ok(t)
    }

    /// Compute attention gate in \[0,1\] deciding whether to communicate.
    pub fn attention_weight(&self, thought: &[f64]) -> Result<f64> {
        if thought.len() != self.thought_dim {
            return Err(TensorError::invalid_argument(
                "AtocAgent: thought_dim mismatch".into(),
            ));
        }
        let logit: f64 = self
            .w_attn
            .iter()
            .zip(thought)
            .map(|(w, t)| w * t)
            .sum::<f64>()
            + self.b_attn;
        Ok(1.0 / (1.0 + (-logit).exp())) // sigmoid
    }

    /// Encode thought into message for broadcasting.
    pub fn encode_message(&self, thought: &[f64]) -> Result<Vec<f64>> {
        self.channel.encode(thought)
    }
}

/// TarMAC: Targeted multi-agent communication with soft attention.
///
/// Each agent generates a (key, value) pair; receivers compute
/// query-key attention weights and aggregate value vectors.
#[derive(Debug, Clone)]
pub struct TarmacAgent {
    pub agent_id: usize,
    hidden_dim: usize,
    key_dim: usize,
    val_dim: usize,
    /// Key projection: hidden_dim → key_dim
    w_key: Vec<f64>,
    b_key: Vec<f64>,
    /// Value projection: hidden_dim → val_dim
    w_val: Vec<f64>,
    b_val: Vec<f64>,
    /// Query projection: hidden_dim → key_dim
    w_query: Vec<f64>,
    b_query: Vec<f64>,
}

impl TarmacAgent {
    /// Create a TarMAC agent.
    pub fn new(
        agent_id: usize,
        hidden_dim: usize,
        key_dim: usize,
        val_dim: usize,
        seed: u64,
    ) -> Result<Self> {
        if hidden_dim == 0 || key_dim == 0 || val_dim == 0 {
            return Err(TensorError::invalid_argument(
                "TarmacAgent: dims must be > 0".into(),
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let sh = (2.0_f64 / hidden_dim as f64).sqrt();
        let mut mk = |n: usize| -> Vec<f64> {
            let mut r = StdRng::seed_from_u64(rng.random::<u64>());
            let v: Vec<f64> = (0..n)
                .map(|_| (r.random::<f64>() * 2.0 - 1.0) * sh)
                .collect();
            // rng already advanced by seed_from_u64 call above
            v
        };
        let w_key = mk(key_dim * hidden_dim);
        let b_key = vec![0.0_f64; key_dim];
        let w_val = mk(val_dim * hidden_dim);
        let b_val = vec![0.0_f64; val_dim];
        let w_query = mk(key_dim * hidden_dim);
        let b_query = vec![0.0_f64; key_dim];
        Ok(Self {
            agent_id,
            hidden_dim,
            key_dim,
            val_dim,
            w_key,
            b_key,
            w_val,
            b_val,
            w_query,
            b_query,
        })
    }

    fn project(&self, w: &[f64], b: &[f64], out_dim: usize, inp: &[f64]) -> Vec<f64> {
        let in_dim = inp.len();
        let mut out = vec![0.0_f64; out_dim];
        for i in 0..out_dim {
            let mut v = b[i];
            for j in 0..in_dim {
                v += w[i * in_dim + j] * inp[j];
            }
            out[i] = v.tanh();
        }
        out
    }

    /// Compute key vector from hidden state.
    pub fn key(&self, hidden: &[f64]) -> Vec<f64> {
        self.project(&self.w_key, &self.b_key, self.key_dim, hidden)
    }

    /// Compute value vector from hidden state.
    pub fn value(&self, hidden: &[f64]) -> Vec<f64> {
        self.project(&self.w_val, &self.b_val, self.val_dim, hidden)
    }

    /// Compute query vector from hidden state.
    pub fn query(&self, hidden: &[f64]) -> Vec<f64> {
        self.project(&self.w_query, &self.b_query, self.key_dim, hidden)
    }

    /// Aggregate messages from other agents using soft attention.
    ///
    /// Returns the attention-weighted sum of value vectors.
    pub fn aggregate(
        &self,
        my_hidden: &[f64],
        others_hidden: &[Vec<f64>],
        agents: &[TarmacAgent],
    ) -> Result<Vec<f64>> {
        if others_hidden.len() != agents.len() {
            return Err(TensorError::invalid_argument(
                "TarmacAgent::aggregate: length mismatch".into(),
            ));
        }
        if others_hidden.is_empty() {
            return Ok(vec![0.0_f64; self.val_dim]);
        }
        let q = self.query(my_hidden);
        let scale = (self.key_dim as f64).sqrt();
        // Compute attention logits
        let logits: Vec<f64> = agents
            .iter()
            .zip(others_hidden.iter())
            .map(|(ag, h)| {
                let k = ag.key(h);
                let dot: f64 = q.iter().zip(&k).map(|(a, b)| a * b).sum::<f64>();
                dot / scale
            })
            .collect();
        // Stable softmax
        let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_l: Vec<f64> = logits.iter().map(|l| (l - max_l).exp()).collect();
        let sum_exp: f64 = exp_l.iter().sum();
        let attn: Vec<f64> = exp_l.iter().map(|e| e / sum_exp.max(1e-12)).collect();
        // Weighted sum of values
        let mut agg = vec![0.0_f64; self.val_dim];
        for (ag, (h, a)) in agents.iter().zip(others_hidden.iter().zip(&attn)) {
            let v = ag.value(h);
            for (acc, vi) in agg.iter_mut().zip(&v) {
                *acc += a * vi;
            }
        }
        Ok(agg)
    }
}
