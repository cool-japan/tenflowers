//! Temporal Graph Network (TGN) — Rossi et al. 2020.
//!
//! TimeEncoder, NodeMemory, MemoryUpdateModule, TemporalGraphNetwork.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

use super::types::{
    linear, matvec, rand_mat, relu, sigmoid, softmax, tanh_act, vecadd, vecmul, zero_vec,
    MessageFunction, TemporalEdge, TgnConfig,
};

// ─────────────────────────────────────────────────────────────────────────────
// TimeEncoder (Xu et al. 2019 — "time2vec"-style cosine encoding)
// ─────────────────────────────────────────────────────────────────────────────

/// Encodes a scalar time delta using learnable frequencies:
///
/// `out_k = cos(w_k · Δt + b_k)` for k = 0..dim.
#[derive(Debug, Clone)]
pub struct TimeEncoder {
    /// Frequency weights \[dim\].
    pub w: Vec<f64>,
    /// Phase biases \[dim\].
    pub b: Vec<f64>,
    /// Output dimension.
    pub dim: usize,
}

impl TimeEncoder {
    /// Create a new TimeEncoder with `dim` frequency components.
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        // Initialise frequencies log-uniformly in [0.01, 100].
        let w: Vec<f64> = (0..dim)
            .map(|i| {
                let log_scale = (i as f64 + 1.0) / dim as f64 * 4.0 - 2.0;
                10.0_f64.powf(log_scale) + rng.random::<f64>() * 0.01
            })
            .collect();
        let b: Vec<f64> = (0..dim)
            .map(|_| rng.random::<f64>() * std::f64::consts::PI * 2.0)
            .collect();
        Self { w, b, dim }
    }

    /// Encode a time delta into a `dim`-dimensional feature vector.
    pub fn encode(&self, delta_t: f64) -> Vec<f64> {
        self.w
            .iter()
            .zip(self.b.iter())
            .map(|(&wi, &bi)| (wi * delta_t + bi).cos())
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NodeMemory
// ─────────────────────────────────────────────────────────────────────────────

/// Per-node memory state maintained by the TGN.
#[derive(Debug, Clone)]
pub struct NodeMemory {
    /// Memory vector for each node.
    pub states: Vec<Vec<f64>>,
    /// Timestamp of the last memory update for each node.
    pub last_update: Vec<f64>,
    /// Memory dimension.
    pub memory_dim: usize,
}

impl NodeMemory {
    /// Allocate node memory for `n_nodes` nodes.
    pub fn new(n_nodes: usize, memory_dim: usize) -> Self {
        Self {
            states: vec![vec![0.0; memory_dim]; n_nodes],
            last_update: vec![0.0; n_nodes],
            memory_dim,
        }
    }

    /// Retrieve the memory state and last-update time for node `i`.
    pub fn get(&self, node: usize) -> Result<(&Vec<f64>, f64)> {
        if node >= self.states.len() {
            return Err(TensorError::invalid_argument(format!(
                "NodeMemory::get — node index {node} out of range (n_nodes={})",
                self.states.len()
            )));
        }
        Ok((&self.states[node], self.last_update[node]))
    }

    /// Overwrite the memory state for node `i`.
    pub fn set(&mut self, node: usize, state: Vec<f64>, time: f64) -> Result<()> {
        if node >= self.states.len() {
            return Err(TensorError::invalid_argument(format!(
                "NodeMemory::set — node index {node} out of range (n_nodes={})",
                self.states.len()
            )));
        }
        self.states[node] = state;
        self.last_update[node] = time;
        Ok(())
    }

    /// Reset all node memories to zero.
    pub fn reset(&mut self) {
        for s in &mut self.states {
            s.fill(0.0);
        }
        self.last_update.fill(0.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoryUpdateModule — GRU-based update
// ─────────────────────────────────────────────────────────────────────────────

/// GRU cell used to update node memories given new messages.
///
/// `h_new = GRU(message, h_old)`
#[derive(Debug, Clone)]
pub struct MemoryUpdateModule {
    /// Input dimension (message size).
    pub msg_dim: usize,
    /// Hidden state / memory dimension.
    pub mem_dim: usize,
    // Reset gate.
    wr: Vec<Vec<f64>>,
    ur: Vec<Vec<f64>>,
    br: Vec<f64>,
    // Update gate.
    wz: Vec<Vec<f64>>,
    uz: Vec<Vec<f64>>,
    bz: Vec<f64>,
    // New gate.
    wn: Vec<Vec<f64>>,
    un: Vec<Vec<f64>>,
    bn: Vec<f64>,
}

impl MemoryUpdateModule {
    /// Create a new GRU memory updater.
    pub fn new(msg_dim: usize, mem_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let s = (2.0 / (msg_dim + mem_dim) as f64).sqrt();
        Self {
            msg_dim,
            mem_dim,
            wr: rand_mat(mem_dim, msg_dim, s, &mut rng),
            ur: rand_mat(mem_dim, mem_dim, s, &mut rng),
            br: zero_vec(mem_dim),
            wz: rand_mat(mem_dim, msg_dim, s, &mut rng),
            uz: rand_mat(mem_dim, mem_dim, s, &mut rng),
            bz: zero_vec(mem_dim),
            wn: rand_mat(mem_dim, msg_dim, s, &mut rng),
            un: rand_mat(mem_dim, mem_dim, s, &mut rng),
            bn: zero_vec(mem_dim),
        }
    }

    /// Run one GRU step: update `h` given `message`.
    pub fn step(&self, message: &[f64], h: &[f64]) -> Result<Vec<f64>> {
        if message.len() != self.msg_dim {
            return Err(TensorError::invalid_argument(format!(
                "MemoryUpdateModule::step — message dim {} != expected {}",
                message.len(),
                self.msg_dim
            )));
        }
        if h.len() != self.mem_dim {
            return Err(TensorError::invalid_argument(format!(
                "MemoryUpdateModule::step — hidden dim {} != expected {}",
                h.len(),
                self.mem_dim
            )));
        }
        // r = sigmoid(Wr·m + Ur·h + br)
        let r_pre = vecadd(
            &vecadd(&matvec(&self.wr, message), &matvec(&self.ur, h)),
            &self.br,
        );
        let r: Vec<f64> = r_pre.iter().map(|&x| sigmoid(x)).collect();
        // z = sigmoid(Wz·m + Uz·h + bz)
        let z_pre = vecadd(
            &vecadd(&matvec(&self.wz, message), &matvec(&self.uz, h)),
            &self.bz,
        );
        let z: Vec<f64> = z_pre.iter().map(|&x| sigmoid(x)).collect();
        // n = tanh(Wn·m + Un·(r⊙h) + bn)
        let rh = vecmul(&r, h);
        let n_pre = vecadd(
            &vecadd(&matvec(&self.wn, message), &matvec(&self.un, &rh)),
            &self.bn,
        );
        let n: Vec<f64> = n_pre.iter().map(|&x| tanh_act(x)).collect();
        // h_new = (1-z)⊙n + z⊙h
        let h_new: Vec<f64> = z
            .iter()
            .zip(n.iter().zip(h.iter()))
            .map(|(&zi, (&ni, &hi))| (1.0 - zi) * ni + zi * hi)
            .collect();
        Ok(h_new)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TemporalGraphNetwork
// ─────────────────────────────────────────────────────────────────────────────

/// Full Temporal Graph Network.
///
/// Implements the TGN architecture (Rossi et al. 2020):
///   memory module → message function → GRU update → graph-attention embedding.
#[derive(Debug, Clone)]
pub struct TemporalGraphNetwork {
    /// Configuration.
    pub config: TgnConfig,
    /// Per-node memories.
    pub memory: NodeMemory,
    /// Time feature encoder.
    pub time_encoder: TimeEncoder,
    /// GRU memory updater.
    pub memory_updater: MemoryUpdateModule,
    // MLP message projection weights (used when MessageFunction::Mlp).
    w_msg: Vec<Vec<f64>>,
    b_msg: Vec<f64>,
    // Graph-attention weights [n_heads × emb_dim × key_dim].
    w_query: Vec<Vec<f64>>,
    w_key: Vec<Vec<f64>>,
    w_val: Vec<Vec<f64>>,
    // Output projection.
    w_out: Vec<Vec<f64>>,
    b_out: Vec<f64>,
}

impl TemporalGraphNetwork {
    /// Create a new TGN.
    pub fn new(config: TgnConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let s_init = 0.1;
        let memory_dim = config.memory_dim;
        let time_dim = config.time_enc_dim;
        let edge_dim = config.edge_feat_dim;
        let emb_dim = config.emb_dim;

        // Message dim: concat(mem_i, mem_j, time_enc, edge_feat)
        let msg_raw_dim = memory_dim * 2 + time_dim + edge_dim;
        let msg_proj_dim = if config.message_fn == MessageFunction::Mlp {
            memory_dim
        } else {
            msg_raw_dim
        };

        let w_msg = rand_mat(msg_proj_dim, msg_raw_dim, s_init, &mut rng);
        let b_msg = zero_vec(msg_proj_dim);

        let key_dim = emb_dim;
        let w_query = rand_mat(key_dim, memory_dim + time_dim, s_init, &mut rng);
        let w_key = rand_mat(key_dim, memory_dim + time_dim, s_init, &mut rng);
        let w_val = rand_mat(emb_dim, memory_dim + time_dim, s_init, &mut rng);

        let w_out = rand_mat(emb_dim, emb_dim, s_init, &mut rng);
        let b_out = zero_vec(emb_dim);

        let memory_updater =
            MemoryUpdateModule::new(msg_proj_dim, memory_dim, seed.wrapping_add(1));
        let time_encoder = TimeEncoder::new(time_dim, seed.wrapping_add(2));

        Self {
            config: config.clone(),
            memory: NodeMemory::new(config.n_nodes, memory_dim),
            time_encoder,
            memory_updater,
            w_msg,
            b_msg,
            w_query,
            w_key,
            w_val,
            w_out,
            b_out,
        }
    }

    /// Build and apply a message from a temporal edge, updating both endpoint memories.
    pub fn process_edge(&mut self, edge: &TemporalEdge) -> Result<()> {
        let (mem_src, t_src) = {
            let (s, t) = self.memory.get(edge.src)?;
            (s.clone(), t)
        };
        let (mem_dst, t_dst) = {
            let (s, t) = self.memory.get(edge.dst)?;
            (s.clone(), t)
        };

        let dt_src = (edge.time - t_src).max(0.0);
        let dt_dst = (edge.time - t_dst).max(0.0);
        let enc_src = self.time_encoder.encode(dt_src);
        let enc_dst = self.time_encoder.encode(dt_dst);

        // Build raw messages (for both directions).
        let mut raw_src: Vec<f64> =
            Vec::with_capacity(mem_src.len() + mem_dst.len() + enc_src.len() + edge.features.len());
        raw_src.extend_from_slice(&mem_src);
        raw_src.extend_from_slice(&mem_dst);
        raw_src.extend_from_slice(&enc_src);
        raw_src.extend_from_slice(&edge.features);

        let mut raw_dst: Vec<f64> =
            Vec::with_capacity(mem_dst.len() + mem_src.len() + enc_dst.len() + edge.features.len());
        raw_dst.extend_from_slice(&mem_dst);
        raw_dst.extend_from_slice(&mem_src);
        raw_dst.extend_from_slice(&enc_dst);
        raw_dst.extend_from_slice(&edge.features);

        let msg_src = self.project_message(&raw_src);
        let msg_dst = self.project_message(&raw_dst);

        let new_mem_src = self.memory_updater.step(&msg_src, &mem_src)?;
        let new_mem_dst = self.memory_updater.step(&msg_dst, &mem_dst)?;

        self.memory.set(edge.src, new_mem_src, edge.time)?;
        self.memory.set(edge.dst, new_mem_dst, edge.time)?;
        Ok(())
    }

    /// Project raw message through identity or MLP.
    fn project_message(&self, raw: &[f64]) -> Vec<f64> {
        match self.config.message_fn {
            MessageFunction::Identity => raw.to_vec(),
            MessageFunction::Mlp => {
                let pre = linear(&self.w_msg, &self.b_msg, raw);
                pre.iter().map(|&x| relu(x)).collect()
            }
        }
    }

    /// Compute temporal graph-attention embedding for node `node` at time `t`,
    /// given a slice of (neighbour_node, edge_feat, edge_time) tuples.
    pub fn embed(
        &self,
        node: usize,
        t: f64,
        neighbours: &[(usize, Vec<f64>, f64)],
    ) -> Result<Vec<f64>> {
        let (mem_q, t_q) = self.memory.get(node)?;
        let dt_q = (t - t_q).max(0.0);
        let enc_q = self.time_encoder.encode(dt_q);
        let mut qin: Vec<f64> = mem_q.clone();
        qin.extend_from_slice(&enc_q);
        let query = matvec(&self.w_query, &qin);

        if neighbours.is_empty() {
            return Ok(matvec(&self.w_out, &matvec(&self.w_val, &qin)));
        }

        // Compute keys, values and attention scores.
        let mut attn_scores: Vec<f64> = Vec::with_capacity(neighbours.len());
        let mut vals: Vec<Vec<f64>> = Vec::with_capacity(neighbours.len());

        let key_dim = query.len();
        for (nb_node, _ef, _nb_t) in neighbours {
            let (mem_nb, t_nb) = self.memory.get(*nb_node)?;
            let dt_nb = (t - t_nb).max(0.0);
            let enc_nb = self.time_encoder.encode(dt_nb);
            let mut kin: Vec<f64> = mem_nb.clone();
            kin.extend_from_slice(&enc_nb);
            let key = matvec(&self.w_key, &kin);
            let score: f64 = query
                .iter()
                .zip(key.iter())
                .map(|(&q, &k)| q * k)
                .sum::<f64>()
                / (key_dim as f64).sqrt();
            attn_scores.push(score);
            vals.push(matvec(&self.w_val, &kin));
        }

        let attn = softmax(&attn_scores);
        let mut ctx = vec![0.0; self.config.emb_dim];
        for (a, v) in attn.iter().zip(vals.iter()) {
            for (ci, vi) in ctx.iter_mut().zip(v.iter()) {
                *ci += a * vi;
            }
        }

        let out = linear(&self.w_out, &self.b_out, &ctx);
        Ok(out)
    }
}
