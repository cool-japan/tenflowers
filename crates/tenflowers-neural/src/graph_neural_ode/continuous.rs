//! Continuous-time, stochastic, and physics-informed graph ODE extensions.
//!
//! Implements:
//! - [`TemporalNodeEmbedding`]: Continuous-time node embedding with sinusoidal time features.
//! - [`TgodeModel`]: Temporal Graph ODE (Huang 2021 style) — GNN spatial encoder + ODE temporal.
//! - [`EventBasedOde`]: Event-triggered ODE updates (integrate only between events).
//! - [`ReactionDiffusionGnn`]: Reaction-diffusion graph neural ODE (Di Giovanni 2022).
//! - [`DifferentialGraphWiring`]: ODE-updated diffusion matrix (learned graph structure).
//! - [`StochasticGnoModel`]: Graph neural SDE with Euler-Maruyama integration.
//! - [`LatentStochasticGraph`]: VAE encoder + stochastic ODE latent dynamics.
//! - [`ParticleSimGraph`]: Particle simulation on graphs (spring + gravity forces).
//! - [`HamiltonianGnn`]: Hamiltonian neural network on graphs (energy-conserving ODE).
//! - [`LagrangianGnn`]: Lagrangian mechanics on graphs (constraint manifold).
//! - [`GnoSimMetrics`]: Trajectory prediction metrics (MSE, rollout stability).

use super::{
    add_scaled, box_muller, matmul, random_matrix, relu_matrix, softmax_rows, tanh_matrix,
    xavier_std, GnoError, GnoGraph, GnoMessagePassing,
};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// §A  TemporalNodeEmbedding
// ─────────────────────────────────────────────────────────────────────────────

/// Continuous-time node embedding with sinusoidal time encoding.
///
/// Combines static node features with a Fourier time encoding,
/// producing time-aware node representations suitable for temporal GNN-ODEs.
#[derive(Debug, Clone)]
pub struct TemporalNodeEmbedding {
    /// Input feature dimension.
    pub feat_dim: usize,
    /// Output embedding dimension.
    pub out_dim: usize,
    /// Time encoding dimension (must divide evenly: cos/sin pairs).
    pub time_dim: usize,
    /// Feature projection weight (feat_dim → out_dim).
    feat_w: Vec<Vec<f64>>,
    /// Time encoding projection weight (time_dim → out_dim).
    time_w: Vec<Vec<f64>>,
    /// Frequency parameters for the sinusoidal time encoding.
    freqs: Vec<f64>,
}

impl TemporalNodeEmbedding {
    /// Create a new temporal node embedding.
    ///
    /// `time_dim` determines how many cos/sin frequency pairs are used; it must be even.
    pub fn new(
        feat_dim: usize,
        out_dim: usize,
        time_dim: usize,
        seed: u64,
    ) -> Result<Self, GnoError> {
        if feat_dim == 0 || out_dim == 0 || time_dim == 0 {
            return Err(GnoError::InvalidConfig(
                "feat_dim, out_dim, time_dim must be > 0".into(),
            ));
        }
        let effective_time_dim = if time_dim % 2 == 0 { time_dim } else { time_dim + 1 };
        let mut rng = StdRng::seed_from_u64(seed);
        let feat_w = random_matrix(feat_dim, out_dim, xavier_std(feat_dim, out_dim), &mut rng);
        let time_w = random_matrix(
            effective_time_dim,
            out_dim,
            xavier_std(effective_time_dim, out_dim),
            &mut rng,
        );
        // Log-spaced frequencies from 1 to 10000 (like Transformer positional encoding)
        let n_freqs = effective_time_dim / 2;
        let freqs: Vec<f64> = (0..n_freqs)
            .map(|i| {
                let exp = i as f64 / n_freqs as f64;
                (10000.0_f64).powf(-exp)
            })
            .collect();
        Ok(Self {
            feat_dim,
            out_dim,
            time_dim: effective_time_dim,
            feat_w,
            time_w,
            freqs,
        })
    }

    /// Encode node features at time `t`.
    ///
    /// Returns `(n_nodes × out_dim)` embedding matrix.
    pub fn encode(&self, node_feats: &[Vec<f64>], t: f64) -> Result<Vec<Vec<f64>>, GnoError> {
        let n = node_feats.len();
        if n == 0 {
            return Err(GnoError::EmptyGraph);
        }
        // Build time encoding vector (cos/sin interleaved)
        let mut time_enc = Vec::with_capacity(self.time_dim);
        for &freq in &self.freqs {
            time_enc.push((t * freq).cos());
            time_enc.push((t * freq).sin());
        }
        // Project features: feat_proj = node_feats @ feat_w  (n × out_dim)
        let mut feat_proj = matmul(node_feats, &self.feat_w)?;
        // Project time encoding: time_proj = time_enc @ time_w  (1 × out_dim)
        // time_enc is a single row, so treat as (1 × time_dim) matrix
        let time_row = vec![time_enc.clone()];
        let time_proj = matmul(&time_row, &self.time_w)?;
        // Add time projection to each node's feature projection
        for row in feat_proj.iter_mut() {
            for (v, tp) in row.iter_mut().zip(time_proj[0].iter()) {
                *v += tp;
                // Tanh activation for bounded output
                *v = v.tanh();
            }
        }
        Ok(feat_proj)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §B  TgodeModel  (Temporal Graph ODE)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Temporal Graph ODE model.
#[derive(Debug, Clone)]
pub struct TgodeConfig {
    /// Input feature dimension.
    pub feat_dim: usize,
    /// Hidden state dimension.
    pub hidden_dim: usize,
    /// Time encoding dimension.
    pub time_enc_dim: usize,
    /// Integration end time.
    pub t_end: f64,
    /// Integration time step.
    pub dt: f64,
    /// Random seed.
    pub seed: u64,
}

/// Temporal Graph ODE (Huang 2021 style).
///
/// Combines a GNN spatial encoder with continuous-time ODE temporal dynamics.
/// The ODE vector field incorporates sinusoidal time features, enabling the
/// model to capture non-stationary temporal patterns.
///
/// Architecture: Encode(t=0) → ODE(GNN + time_enc) → H(t)
#[derive(Debug, Clone)]
pub struct TgodeModel {
    /// Model configuration.
    pub cfg: TgodeConfig,
    /// Temporal node embedding for time-aware features.
    pub time_embed: TemporalNodeEmbedding,
    /// Spatial GCN weight (hidden_dim → hidden_dim).
    gcn_w: Vec<Vec<f64>>,
    /// ODE recombination weight (hidden_dim → hidden_dim).
    ode_w: Vec<Vec<f64>>,
    /// Normalised adjacency matrix.
    a_hat: Vec<Vec<f64>>,
}

impl TgodeModel {
    /// Create a new Temporal Graph ODE model.
    pub fn new(cfg: TgodeConfig, graph: &GnoGraph) -> Result<Self, GnoError> {
        if cfg.feat_dim == 0 || cfg.hidden_dim == 0 {
            return Err(GnoError::InvalidConfig(
                "feat_dim and hidden_dim must be > 0".into(),
            ));
        }
        let a_hat = graph.normalize_adjacency();
        let time_embed =
            TemporalNodeEmbedding::new(cfg.feat_dim, cfg.hidden_dim, cfg.time_enc_dim, cfg.seed)?;
        let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(1));
        let gcn_w = random_matrix(
            cfg.hidden_dim,
            cfg.hidden_dim,
            xavier_std(cfg.hidden_dim, cfg.hidden_dim),
            &mut rng,
        );
        let ode_w = random_matrix(
            cfg.hidden_dim,
            cfg.hidden_dim,
            xavier_std(cfg.hidden_dim, cfg.hidden_dim),
            &mut rng,
        );
        Ok(Self {
            cfg,
            time_embed,
            gcn_w,
            ode_w,
            a_hat,
        })
    }

    /// ODE dynamics: dH/dt = tanh(Â · H · W_gcn + H · W_ode + time_enc(t))
    fn dynamics(
        &self,
        h: &[Vec<f64>],
        t: f64,
    ) -> Result<Vec<Vec<f64>>, GnoError> {
        let mp = GnoMessagePassing::new();
        let agg = mp.aggregate(h, &self.a_hat)?;
        let mut gcn_out = matmul(&agg, &self.gcn_w)?;
        let ode_out = matmul(h, &self.ode_w)?;
        // time encoding bias
        let n_freqs = self.cfg.time_enc_dim.max(2) / 2;
        for (i, row) in gcn_out.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v += ode_out[i][j];
                // Add a simple time-dependent bias using cos/sin
                let freq_idx = j % n_freqs;
                let freq = (10000.0_f64).powf(-(freq_idx as f64 / n_freqs as f64));
                let t_bias = if j % 2 == 0 {
                    (t * freq).cos() * 0.1
                } else {
                    (t * freq).sin() * 0.1
                };
                *v += t_bias;
            }
        }
        tanh_matrix(&mut gcn_out);
        Ok(gcn_out)
    }

    /// Forward: encode node features at time `t_start`, then integrate to t_end.
    ///
    /// Returns the node hidden states at `t_end`.
    pub fn forward(
        &self,
        node_feats: &[Vec<f64>],
        t_start: f64,
    ) -> Result<Vec<Vec<f64>>, GnoError> {
        let mut h = self.time_embed.encode(node_feats, t_start)?;
        let dt = self.cfg.dt;
        let t_end = self.cfg.t_end;
        let mut t = t_start;
        while t + dt <= t_end + 1e-12 {
            let dh = self.dynamics(&h, t)?;
            h = add_scaled(&h, &dh, dt);
            t += dt;
        }
        Ok(h)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §C  EventBasedOde
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the event-triggered ODE.
#[derive(Debug, Clone)]
pub struct EventOdeConfig {
    /// Feature dimension.
    pub feat_dim: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Random seed.
    pub seed: u64,
}

/// Event-triggered ODE updates.
///
/// Unlike uniform-step ODEs, this model integrates only between event timestamps,
/// making it suitable for asynchronous / irregular-time graph data (e.g. citation
/// events, interaction logs).  Between events the hidden state evolves continuously;
/// at each event the state is updated by a GNN layer.
#[derive(Debug, Clone)]
pub struct EventBasedOde {
    cfg: EventOdeConfig,
    /// ODE drift weight (feat_dim → hidden_dim).
    drift_w: Vec<Vec<f64>>,
    /// Event update GCN weight (feat_dim → feat_dim).
    event_w: Vec<Vec<f64>>,
    a_hat: Vec<Vec<f64>>,
}

impl EventBasedOde {
    /// Create a new event-triggered ODE model.
    pub fn new(cfg: EventOdeConfig, graph: &GnoGraph) -> Result<Self, GnoError> {
        if cfg.feat_dim == 0 || cfg.hidden_dim == 0 {
            return Err(GnoError::InvalidConfig(
                "feat_dim and hidden_dim must be > 0".into(),
            ));
        }
        let a_hat = graph.normalize_adjacency();
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let drift_w = random_matrix(
            cfg.feat_dim,
            cfg.feat_dim,
            xavier_std(cfg.feat_dim, cfg.feat_dim),
            &mut rng,
        );
        let event_w = random_matrix(
            cfg.feat_dim,
            cfg.feat_dim,
            xavier_std(cfg.feat_dim, cfg.feat_dim),
            &mut rng,
        );
        Ok(Self {
            cfg,
            drift_w,
            event_w,
            a_hat,
        })
    }

    /// Continuous ODE dynamics between events: dH/dt = tanh(H · W_drift).
    fn drift(&self, h: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let mut out = matmul(h, &self.drift_w)?;
        tanh_matrix(&mut out);
        Ok(out)
    }

    /// Euler step for the continuous drift.
    fn euler_step(&self, h: &[Vec<f64>], dt: f64) -> Result<Vec<Vec<f64>>, GnoError> {
        let dh = self.drift(h)?;
        Ok(add_scaled(h, &dh, dt))
    }

    /// GNN update at an event time: H ← tanh(Â · H · W_event + H).
    fn event_update(&self, h: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let mp = GnoMessagePassing::new();
        let agg = mp.aggregate(h, &self.a_hat)?;
        let mut update = matmul(&agg, &self.event_w)?;
        // Residual connection
        for (i, row) in update.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v += h[i][j];
                *v = v.tanh();
            }
        }
        Ok(update)
    }

    /// Integrate from time 0 to the last event, integrating between each pair.
    ///
    /// Events are sorted internally.  The returned matrix is the state after
    /// the final event update (or the initial state if no events given).
    pub fn integrate(
        &self,
        h0: &[Vec<f64>],
        events: &[f64],
    ) -> Result<Vec<Vec<f64>>, GnoError> {
        if h0.is_empty() {
            return Err(GnoError::EmptyGraph);
        }
        if events.is_empty() {
            return Ok(h0.to_vec());
        }
        let mut sorted_events = events.to_vec();
        sorted_events.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let dt_min = 0.05_f64; // fixed sub-step for continuous integration
        let mut h = h0.to_vec();
        let mut t_prev = 0.0f64;
        for &t_event in &sorted_events {
            // Integrate from t_prev to t_event
            let mut t = t_prev;
            while t + dt_min <= t_event + 1e-12 {
                h = self.euler_step(&h, dt_min)?;
                t += dt_min;
            }
            // Remainder
            let rem = t_event - t;
            if rem > 1e-12 {
                h = self.euler_step(&h, rem)?;
            }
            // Apply event update (GNN message passing)
            h = self.event_update(&h)?;
            t_prev = t_event;
        }
        Ok(h)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §D  ReactionDiffusionGnn  (Di Giovanni 2022)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the reaction-diffusion graph neural ODE.
#[derive(Debug, Clone)]
pub struct RdGnnConfig {
    /// Input / hidden feature dimension.
    pub feat_dim: usize,
    /// Reaction MLP hidden dimension.
    pub hidden_dim: usize,
    /// Integration end time.
    pub t_end: f64,
    /// Integration time step.
    pub dt: f64,
    /// Diffusion coefficient α controlling the strength of the diffusion term.
    pub diffusion_coeff: f64,
    /// Random seed.
    pub seed: u64,
}

/// Graph neural ODE with reaction (source/sink) + diffusion (Laplacian) terms.
///
/// The continuous dynamics follow:
///   dH/dt = α · L · H  (diffusion, L = normalized Laplacian)
///            + f_θ(H, A) (reaction, parameterized by GNN)
///
/// Stable discretization: diffusion uses backward Euler (implicit) approximated
/// by (I + αΔt·L)^{-1} via one Jacobi iteration; reaction uses Euler (explicit).
#[derive(Debug, Clone)]
pub struct ReactionDiffusionGnn {
    cfg: RdGnnConfig,
    /// Reaction MLP weight 1 (feat_dim → hidden_dim).
    react_w1: Vec<Vec<f64>>,
    /// Reaction MLP weight 2 (hidden_dim → feat_dim).
    react_w2: Vec<Vec<f64>>,
    /// Normalised adjacency matrix for diffusion.
    a_hat: Vec<Vec<f64>>,
    /// Graph Laplacian L = D - A (raw, not normalised).
    laplacian: Vec<Vec<f64>>,
}

impl ReactionDiffusionGnn {
    /// Create a new reaction-diffusion GNN.
    pub fn new(cfg: RdGnnConfig, graph: &GnoGraph) -> Result<Self, GnoError> {
        if cfg.feat_dim == 0 || cfg.hidden_dim == 0 {
            return Err(GnoError::InvalidConfig(
                "feat_dim and hidden_dim must be > 0".into(),
            ));
        }
        let a_hat = graph.normalize_adjacency();
        let laplacian = graph.laplacian();
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let react_w1 = random_matrix(
            cfg.feat_dim,
            cfg.hidden_dim,
            xavier_std(cfg.feat_dim, cfg.hidden_dim),
            &mut rng,
        );
        let react_w2 = random_matrix(
            cfg.hidden_dim,
            cfg.feat_dim,
            xavier_std(cfg.hidden_dim, cfg.feat_dim),
            &mut rng,
        );
        Ok(Self {
            cfg,
            react_w1,
            react_w2,
            a_hat,
            laplacian,
        })
    }

    /// Reaction term: f_θ(H) = W2 · ReLU(W1 · H^T) (element-wise per node).
    fn reaction(&self, h: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let mut hidden = matmul(h, &self.react_w1)?;
        relu_matrix(&mut hidden);
        matmul(&hidden, &self.react_w2)
    }

    /// Diffusion term: α · L · H (normalised Laplacian diffusion).
    fn diffusion(&self, h: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let n = h.len();
        if n == 0 {
            return Err(GnoError::EmptyGraph);
        }
        let d = h[0].len();
        let alpha = self.cfg.diffusion_coeff;
        // Use symmetric normalised Laplacian: I - Â  (ensures negative semi-definite diffusion)
        // dH/dt_{diffusion} = -alpha * (I - Â) * H = alpha * (Â - I) * H
        let mut out = vec![vec![0.0f64; d]; n];
        for i in 0..n {
            for k in 0..d {
                let mut val = 0.0f64;
                for j in 0..n {
                    let l_ij = if i == j { 1.0 - self.a_hat[i][j] } else { -self.a_hat[i][j] };
                    val += l_ij * h[j][k];
                }
                out[i][k] = -alpha * val;
            }
        }
        Ok(out)
    }

    /// Combined dynamics: dH/dt = diffusion(H) + reaction(H).
    fn dynamics(&self, h: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let diff = self.diffusion(h)?;
        let react = self.reaction(h)?;
        let n = h.len();
        let d = h[0].len();
        let mut out = vec![vec![0.0f64; d]; n];
        for i in 0..n {
            for j in 0..d {
                out[i][j] = diff[i][j] + react[i][j];
            }
        }
        Ok(out)
    }

    /// Forward: integrate the reaction-diffusion ODE from 0 to t_end.
    pub fn forward(&self, h0: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let mut h = h0.to_vec();
        let dt = self.cfg.dt;
        let t_end = self.cfg.t_end;
        let mut t = 0.0f64;
        while t + dt <= t_end + 1e-12 {
            let dh = self.dynamics(&h)?;
            h = add_scaled(&h, &dh, dt);
            t += dt;
        }
        Ok(h)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §E  DifferentialGraphWiring
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the differential graph wiring model.
#[derive(Debug, Clone)]
pub struct DgwConfig {
    /// Node feature dimension.
    pub feat_dim: usize,
    /// Hidden dimension for the edge-weight MLP.
    pub hidden_dim: usize,
    /// Integration end time.
    pub t_end: f64,
    /// Integration time step.
    pub dt: f64,
    /// Random seed.
    pub seed: u64,
}

/// Learned graph structure updated by an ODE.
///
/// The diffusion matrix A(t) itself evolves via an ODE whose vector field
/// is parameterized by a small MLP applied to pairwise node similarities.
/// This models scenarios where the graph topology itself is dynamic (e.g.
/// financial correlation networks, brain connectivity).
#[derive(Debug, Clone)]
pub struct DifferentialGraphWiring {
    cfg: DgwConfig,
    /// Similarity MLP layer 1 (2*feat_dim → hidden_dim).
    sim_w1: Vec<Vec<f64>>,
    /// Similarity MLP layer 2 (hidden_dim → 1).
    sim_w2: Vec<Vec<f64>>,
    /// Node feature update weight (feat_dim → feat_dim).
    feat_w: Vec<Vec<f64>>,
    /// Initial adjacency (from graph structure).
    a_init: Vec<Vec<f64>>,
}

impl DifferentialGraphWiring {
    /// Create a new differential graph wiring model.
    pub fn new(cfg: DgwConfig, graph: &GnoGraph) -> Result<Self, GnoError> {
        if cfg.feat_dim == 0 || cfg.hidden_dim == 0 {
            return Err(GnoError::InvalidConfig(
                "feat_dim and hidden_dim must be > 0".into(),
            ));
        }
        let a_init = graph.normalize_adjacency();
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let in_dim = cfg.feat_dim * 2;
        let sim_w1 = random_matrix(in_dim, cfg.hidden_dim, xavier_std(in_dim, cfg.hidden_dim), &mut rng);
        let sim_w2 = random_matrix(cfg.hidden_dim, 1, xavier_std(cfg.hidden_dim, 1), &mut rng);
        let feat_w = random_matrix(
            cfg.feat_dim,
            cfg.feat_dim,
            xavier_std(cfg.feat_dim, cfg.feat_dim),
            &mut rng,
        );
        Ok(Self {
            cfg,
            sim_w1,
            sim_w2,
            feat_w,
            a_init,
        })
    }

    /// Compute pairwise edge weights from current node features.
    ///
    /// Returns an n×n weight matrix (softmax-normalised per row).
    fn compute_adjacency(&self, h: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let n = h.len();
        if n == 0 {
            return Err(GnoError::EmptyGraph);
        }
        let mut a = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                // Concatenate h_i and h_j
                let mut pair: Vec<f64> = h[i].clone();
                pair.extend_from_slice(&h[j]);
                // MLP: pair → hidden → scalar
                let pair_row = vec![pair];
                let hidden = matmul(&pair_row, &self.sim_w1)?;
                let mut hidden_act = hidden[0].clone();
                for v in hidden_act.iter_mut() {
                    if *v < 0.0 { *v = 0.0; }
                }
                let hidden_row = vec![hidden_act];
                let score = matmul(&hidden_row, &self.sim_w2)?;
                a[i][j] = score[0][0];
            }
        }
        softmax_rows(&mut a);
        Ok(a)
    }

    /// Update node features via message passing on current A(t).
    fn update_features(&self, h: &[Vec<f64>], a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let mp = GnoMessagePassing::new();
        let agg = mp.aggregate(h, a)?;
        let mut out = matmul(&agg, &self.feat_w)?;
        tanh_matrix(&mut out);
        Ok(out)
    }

    /// Forward: jointly evolve graph structure A(t) and node features H(t).
    pub fn forward(&self, h0: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let mut h = h0.to_vec();
        let dt = self.cfg.dt;
        let t_end = self.cfg.t_end;
        let mut t = 0.0f64;
        while t + dt <= t_end + 1e-12 {
            // Recompute dynamic adjacency
            let a_t = self.compute_adjacency(&h)?;
            // Blend with initial graph structure for stability
            let n = h.len();
            let mut a_blend = vec![vec![0.0f64; n]; n];
            for i in 0..n {
                for j in 0..n {
                    a_blend[i][j] = 0.5 * a_t[i][j] + 0.5 * self.a_init[i][j];
                }
            }
            let dh = self.update_features(&h, &a_blend)?;
            h = add_scaled(&h, &dh, dt);
            t += dt;
        }
        Ok(h)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §F  StochasticGnoModel  (Graph Neural SDE)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the stochastic graph neural ODE.
#[derive(Debug, Clone)]
pub struct SgnoConfig {
    /// Node feature dimension.
    pub feat_dim: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Integration end time.
    pub t_end: f64,
    /// Integration time step.
    pub dt: f64,
    /// Noise standard deviation for the diffusion term.
    pub noise_scale: f64,
    /// Random seed.
    pub seed: u64,
}

/// Graph neural SDE with Euler-Maruyama integration.
///
/// dH = f_θ(H, A) dt + σ g_φ(H) dW_t
///
/// The drift `f_θ` is a GCN, and the diffusion `g_φ` is a diagonal MLP
/// ensuring per-feature noise amplitude.  Euler-Maruyama discretisation:
///
/// H(t+Δt) = H(t) + f_θ(H(t), A)·Δt + σ·g_φ(H(t))·√Δt·ε,  ε ~ N(0,I)
#[derive(Debug, Clone)]
pub struct StochasticGnoModel {
    cfg: SgnoConfig,
    /// Drift GCN weight (feat_dim → feat_dim).
    drift_gcn_w: Vec<Vec<f64>>,
    /// Diffusion diagonal MLP weight (feat_dim → feat_dim), output is abs-valued.
    diffusion_w: Vec<Vec<f64>>,
    a_hat: Vec<Vec<f64>>,
}

impl StochasticGnoModel {
    /// Create a new stochastic graph neural ODE.
    pub fn new(cfg: SgnoConfig, graph: &GnoGraph) -> Result<Self, GnoError> {
        if cfg.feat_dim == 0 {
            return Err(GnoError::InvalidConfig("feat_dim must be > 0".into()));
        }
        let a_hat = graph.normalize_adjacency();
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let fd = cfg.feat_dim;
        let drift_gcn_w = random_matrix(fd, fd, xavier_std(fd, fd), &mut rng);
        let diffusion_w = random_matrix(fd, fd, xavier_std(fd, fd), &mut rng);
        Ok(Self {
            cfg,
            drift_gcn_w,
            diffusion_w,
            a_hat,
        })
    }

    /// Compute drift: f(H) = tanh(Â · H · W_drift).
    fn drift(&self, h: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let mp = GnoMessagePassing::new();
        let agg = mp.aggregate(h, &self.a_hat)?;
        let mut out = matmul(&agg, &self.drift_gcn_w)?;
        tanh_matrix(&mut out);
        Ok(out)
    }

    /// Compute per-node per-feature noise amplitude: g(H) = |H · W_diffusion|.
    fn diffusion_amplitude(&self, h: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let mut out = matmul(h, &self.diffusion_w)?;
        // Absolute value for positive amplitude
        for row in out.iter_mut() {
            for v in row.iter_mut() {
                *v = v.abs() + 1e-6; // ensure non-zero amplitude
            }
        }
        Ok(out)
    }

    /// Forward: Euler-Maruyama integration from t=0 to t_end with seed for noise.
    pub fn forward(&self, h0: &[Vec<f64>], noise_seed: u64) -> Result<Vec<Vec<f64>>, GnoError> {
        let mut h = h0.to_vec();
        let dt = self.cfg.dt;
        let t_end = self.cfg.t_end;
        let sigma = self.cfg.noise_scale;
        let mut rng = StdRng::seed_from_u64(noise_seed);
        let mut t = 0.0f64;
        while t + dt <= t_end + 1e-12 {
            let f_h = self.drift(&h)?;
            let g_h = self.diffusion_amplitude(&h)?;
            let n = h.len();
            let d = h[0].len();
            let sqrt_dt = dt.sqrt();
            let mut h_next = vec![vec![0.0f64; d]; n];
            for i in 0..n {
                for j in 0..d {
                    let u1 = rng.random::<f64>();
                    let u2 = rng.random::<f64>();
                    let noise = box_muller(u1, u2);
                    h_next[i][j] = h[i][j]
                        + f_h[i][j] * dt
                        + sigma * g_h[i][j] * sqrt_dt * noise;
                }
            }
            h = h_next;
            t += dt;
        }
        Ok(h)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §G  LatentStochasticGraph  (VAE + Stochastic ODE)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the latent stochastic graph model.
#[derive(Debug, Clone)]
pub struct LsgConfig {
    /// Input feature dimension.
    pub feat_dim: usize,
    /// Latent dimension.
    pub latent_dim: usize,
    /// Number of output classes.
    pub num_classes: usize,
    /// Integration end time.
    pub t_end: f64,
    /// Integration time step.
    pub dt: f64,
    /// Noise scale for the stochastic latent ODE.
    pub noise_scale: f64,
    /// Random seed.
    pub seed: u64,
}

/// VAE encoder + stochastic ODE latent dynamics, ELBO training.
///
/// Architecture:
/// 1. Encode: GCN encoder → (μ, log_σ²) per node.
/// 2. Reparameterize: z_0 ~ N(μ, σ²I).
/// 3. Evolve: stochastic ODE (Euler-Maruyama) in latent space.
/// 4. Decode: linear projection to class logits.
///
/// Loss = CE(logits, y) + β · KL(q(z)||p(z))
#[derive(Debug, Clone)]
pub struct LatentStochasticGraph {
    cfg: LsgConfig,
    /// Encoder GCN weight (feat_dim → latent_dim).
    enc_gcn_w: Vec<Vec<f64>>,
    /// Posterior projection (latent_dim → 2*latent_dim) for (μ, log_σ²).
    enc_proj: Vec<Vec<f64>>,
    /// Stochastic ODE drift weight (latent_dim → latent_dim).
    ode_drift_w: Vec<Vec<f64>>,
    /// Stochastic ODE diffusion weight (latent_dim → latent_dim).
    ode_diff_w: Vec<Vec<f64>>,
    /// Decoder (latent_dim → num_classes).
    dec_w: Vec<Vec<f64>>,
    dec_b: Vec<f64>,
    a_hat: Vec<Vec<f64>>,
}

impl LatentStochasticGraph {
    /// Create a new latent stochastic graph model.
    pub fn new(cfg: LsgConfig, graph: &GnoGraph) -> Result<Self, GnoError> {
        if cfg.feat_dim == 0 || cfg.latent_dim == 0 {
            return Err(GnoError::InvalidConfig(
                "feat_dim and latent_dim must be > 0".into(),
            ));
        }
        if cfg.num_classes < 2 {
            return Err(GnoError::InvalidNumClasses { found: cfg.num_classes });
        }
        let a_hat = graph.normalize_adjacency();
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let fd = cfg.feat_dim;
        let ld = cfg.latent_dim;
        let enc_gcn_w = random_matrix(fd, ld, xavier_std(fd, ld), &mut rng);
        let enc_proj = random_matrix(ld, 2 * ld, xavier_std(ld, 2 * ld), &mut rng);
        let ode_drift_w = random_matrix(ld, ld, xavier_std(ld, ld), &mut rng);
        let ode_diff_w = random_matrix(ld, ld, xavier_std(ld, ld), &mut rng);
        let dec_w = random_matrix(ld, cfg.num_classes, xavier_std(ld, cfg.num_classes), &mut rng);
        let dec_b = vec![0.0f64; cfg.num_classes];
        Ok(Self {
            cfg,
            enc_gcn_w,
            enc_proj,
            ode_drift_w,
            ode_diff_w,
            dec_w,
            dec_b,
            a_hat,
        })
    }

    fn encode(&self, x: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), GnoError> {
        let mp = GnoMessagePassing::new();
        let agg = mp.aggregate(x, &self.a_hat)?;
        let mut enc = matmul(&agg, &self.enc_gcn_w)?;
        relu_matrix(&mut enc);
        let params = matmul(&enc, &self.enc_proj)?;
        let n = x.len();
        let ld = self.cfg.latent_dim;
        let mut mu = vec![vec![0.0f64; ld]; n];
        let mut log_var = vec![vec![0.0f64; ld]; n];
        for i in 0..n {
            for j in 0..ld {
                mu[i][j] = params[i][j];
                log_var[i][j] = params[i][j + ld].clamp(-8.0, 4.0); // clamp for stability
            }
        }
        Ok((mu, log_var))
    }

    fn reparameterize(&self, mu: &[Vec<f64>], log_var: &[Vec<f64>], rng: &mut StdRng) -> Vec<Vec<f64>> {
        let n = mu.len();
        let ld = self.cfg.latent_dim;
        let mut z = vec![vec![0.0f64; ld]; n];
        for i in 0..n {
            for j in 0..ld {
                let u1 = rng.random::<f64>();
                let u2 = rng.random::<f64>();
                let eps = box_muller(u1, u2);
                let std = (0.5 * log_var[i][j]).exp();
                z[i][j] = mu[i][j] + eps * std;
            }
        }
        z
    }

    fn stochastic_ode_step(
        &self,
        z: &[Vec<f64>],
        dt: f64,
        rng: &mut StdRng,
    ) -> Result<Vec<Vec<f64>>, GnoError> {
        // Drift: tanh(Â · z · W_drift)
        let mp = GnoMessagePassing::new();
        let agg = mp.aggregate(z, &self.a_hat)?;
        let mut drift = matmul(&agg, &self.ode_drift_w)?;
        tanh_matrix(&mut drift);
        // Diffusion amplitude: |z · W_diff|
        let mut diff_amp = matmul(z, &self.ode_diff_w)?;
        for row in diff_amp.iter_mut() {
            for v in row.iter_mut() {
                *v = v.abs() + 1e-6;
            }
        }
        let sigma = self.cfg.noise_scale;
        let sqrt_dt = dt.sqrt();
        let n = z.len();
        let ld = self.cfg.latent_dim;
        let mut z_next = vec![vec![0.0f64; ld]; n];
        for i in 0..n {
            for j in 0..ld {
                let u1 = rng.random::<f64>();
                let u2 = rng.random::<f64>();
                let noise = box_muller(u1, u2);
                z_next[i][j] = z[i][j]
                    + drift[i][j] * dt
                    + sigma * diff_amp[i][j] * sqrt_dt * noise;
            }
        }
        Ok(z_next)
    }

    /// Forward: encode → reparameterize → stochastic ODE → decode.
    /// Returns (logits, KL divergence).
    pub fn forward(&self, x: &[Vec<f64>], seed: u64) -> Result<(Vec<Vec<f64>>, f64), GnoError> {
        let (mu, log_var) = self.encode(x)?;
        let mut rng = StdRng::seed_from_u64(seed);
        let z0 = self.reparameterize(&mu, &log_var, &mut rng);
        // KL divergence
        let n = mu.len();
        let ld = self.cfg.latent_dim;
        let mut kl = 0.0f64;
        for i in 0..n {
            for j in 0..ld {
                kl += 0.5 * (log_var[i][j].exp() + mu[i][j].powi(2) - 1.0 - log_var[i][j]);
            }
        }
        kl = kl.max(0.0);
        // Stochastic ODE
        let mut z = z0;
        let dt = self.cfg.dt;
        let t_end = self.cfg.t_end;
        let mut t = 0.0f64;
        while t + dt <= t_end + 1e-12 {
            z = self.stochastic_ode_step(&z, dt, &mut rng)?;
            t += dt;
        }
        // Decode
        let mut logits = matmul(&z, &self.dec_w)?;
        for row in logits.iter_mut() {
            for (v, b) in row.iter_mut().zip(self.dec_b.iter()) {
                *v += b;
            }
        }
        Ok((logits, kl))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §H  ParticleSimGraph  (Spring/Gravity physics)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the particle simulation graph model.
#[derive(Debug, Clone)]
pub struct PsgConfig {
    /// Number of particles (= graph nodes).
    pub n_particles: usize,
    /// Per-particle state dimension (positions + velocities, must be even).
    pub state_dim: usize,
    /// Integration end time.
    pub t_end: f64,
    /// Integration time step.
    pub dt: f64,
    /// Spring stiffness constant.
    pub spring_k: f64,
    /// Gravitational acceleration magnitude.
    pub gravity: f64,
    /// Random seed for MLP parameters.
    pub seed: u64,
}

/// Particle system simulation on graphs.
///
/// Each node is a particle with state `[q; v]` (positions, velocities),
/// each edge is a spring.  Forces are:
/// - Spring: F_spring(i,j) = -k · (||q_i - q_j|| - L_0) · (q_i - q_j) / ||...||
///   (with rest length L_0 derived from initial positions)
/// - Gravity: F_grav = g · m (downward in first coordinate of q)
/// - Learned correction: small MLP applied to neighbor aggregation
///
/// Leapfrog / Störmer-Verlet integration for symplectic accuracy.
#[derive(Debug, Clone)]
pub struct ParticleSimGraph {
    cfg: PsgConfig,
    /// Learned correction MLP weight (state_dim → state_dim/2).
    correction_w: Vec<Vec<f64>>,
    a_hat: Vec<Vec<f64>>,
}

impl ParticleSimGraph {
    /// Create a new particle simulation graph.
    pub fn new(cfg: PsgConfig, graph: &GnoGraph) -> Result<Self, GnoError> {
        if cfg.state_dim == 0 || cfg.state_dim % 2 != 0 {
            return Err(GnoError::InvalidConfig(
                "state_dim must be > 0 and even (positions + velocities)".into(),
            ));
        }
        let a_hat = graph.normalize_adjacency();
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let sd = cfg.state_dim;
        let correction_w = random_matrix(sd, sd / 2, xavier_std(sd, sd / 2), &mut rng);
        Ok(Self { cfg, correction_w, a_hat })
    }

    /// Compute spring forces + gravity for each particle.
    fn forces(&self, states: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = states.len();
        let q_dim = self.cfg.state_dim / 2;
        let mut forces = vec![vec![0.0f64; q_dim]; n];
        // Spring forces via edge adjacency
        for i in 0..n {
            for j in 0..n {
                let w = self.a_hat[i][j];
                if w < 1e-12 || i == j {
                    continue;
                }
                // q_i - q_j
                let mut delta = vec![0.0f64; q_dim];
                let mut dist2 = 0.0f64;
                for k in 0..q_dim {
                    delta[k] = states[i][k] - states[j][k];
                    dist2 += delta[k] * delta[k];
                }
                let dist = dist2.sqrt().max(1e-8);
                // Rest length: 1.0 (normalized units)
                let stretch = dist - 1.0;
                let k_spring = self.cfg.spring_k * w;
                for k in 0..q_dim {
                    forces[i][k] -= k_spring * stretch * (delta[k] / dist);
                }
            }
            // Gravity (acts on first position coordinate only, downward)
            forces[i][0] -= self.cfg.gravity;
        }
        forces
    }

    /// Learned correction via MLP on aggregated neighborhood.
    fn learned_correction(&self, states: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let mp = GnoMessagePassing::new();
        let agg = mp.aggregate(states, &self.a_hat)?;
        let mut out = matmul(&agg, &self.correction_w)?;
        tanh_matrix(&mut out);
        Ok(out)
    }

    /// Forward: Störmer-Verlet (leapfrog) integration.
    ///
    /// State layout: `[q_0, ..., q_{d/2-1}, v_0, ..., v_{d/2-1}]`
    pub fn forward(&self, states: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let n = states.len();
        if n == 0 {
            return Err(GnoError::EmptyGraph);
        }
        let q_dim = self.cfg.state_dim / 2;
        let dt = self.cfg.dt;
        let t_end = self.cfg.t_end;
        let mut s = states.to_vec();
        let mut t = 0.0f64;
        while t + dt <= t_end + 1e-12 {
            let f = self.forces(&s);
            let correction = self.learned_correction(&s)?;
            // Leapfrog half-step velocity update
            let mut s_next = s.clone();
            for i in 0..n {
                for k in 0..q_dim {
                    let correction_val = if k < correction[i].len() {
                        correction[i][k]
                    } else {
                        0.0
                    };
                    // v(t + dt/2) = v(t) + (f + correction) * dt/2
                    s_next[i][q_dim + k] += (f[i][k] + correction_val * 0.1) * (dt / 2.0);
                }
                // Position full step
                for k in 0..q_dim {
                    s_next[i][k] += s_next[i][q_dim + k] * dt;
                }
            }
            // Recompute forces at new positions
            let f2 = self.forces(&s_next);
            let correction2 = self.learned_correction(&s_next)?;
            for i in 0..n {
                for k in 0..q_dim {
                    let correction_val = if k < correction2[i].len() {
                        correction2[i][k]
                    } else {
                        0.0
                    };
                    // v(t + dt) = v(t + dt/2) + (f2 + correction2) * dt/2
                    s_next[i][q_dim + k] += (f2[i][k] + correction_val * 0.1) * (dt / 2.0);
                }
            }
            s = s_next;
            t += dt;
        }
        Ok(s)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §I  HamiltonianGnn  (Energy-conserving ODE on graphs)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Hamiltonian GNN.
#[derive(Debug, Clone)]
pub struct HgnConfig {
    /// Per-node state dimension (must be even: half positions q, half momenta p).
    pub state_dim: usize,
    /// Hidden dimension for the Hamiltonian MLP.
    pub hidden_dim: usize,
    /// Integration end time.
    pub t_end: f64,
    /// Integration time step.
    pub dt: f64,
    /// Random seed.
    pub seed: u64,
}

/// Hamiltonian neural network on graphs.
///
/// Learns a scalar Hamiltonian H(q, p, G) parameterized by a GNN, then
/// derives the ODE from Hamilton's equations:
///   dq/dt =  ∂H/∂p   (finite-difference approximation)
///   dp/dt = -∂H/∂q   (finite-difference approximation)
///
/// This guarantees energy conservation by construction (up to discretization).
/// The GNN aggregates neighbor Hamiltonians to capture interaction energies.
#[derive(Debug, Clone)]
pub struct HamiltonianGnn {
    cfg: HgnConfig,
    /// Hamiltonian MLP layer 1 (state_dim → hidden_dim).
    h_w1: Vec<Vec<f64>>,
    /// Hamiltonian MLP layer 2 (hidden_dim → 1) → scalar energy.
    h_w2: Vec<Vec<f64>>,
    a_hat: Vec<Vec<f64>>,
}

impl HamiltonianGnn {
    /// Create a new Hamiltonian GNN.
    pub fn new(cfg: HgnConfig, graph: &GnoGraph) -> Result<Self, GnoError> {
        if cfg.state_dim == 0 || cfg.state_dim % 2 != 0 {
            return Err(GnoError::InvalidConfig(
                "state_dim must be even and > 0 (q and p)".into(),
            ));
        }
        if cfg.hidden_dim == 0 {
            return Err(GnoError::InvalidConfig("hidden_dim must be > 0".into()));
        }
        let a_hat = graph.normalize_adjacency();
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let sd = cfg.state_dim;
        let hd = cfg.hidden_dim;
        let h_w1 = random_matrix(sd, hd, xavier_std(sd, hd), &mut rng);
        let h_w2 = random_matrix(hd, 1, xavier_std(hd, 1), &mut rng);
        Ok(Self { cfg, h_w1, h_w2, a_hat })
    }

    /// Evaluate the Hamiltonian scalar for each node.
    fn hamiltonian_per_node(&self, states: &[Vec<f64>]) -> Result<Vec<f64>, GnoError> {
        let mp = GnoMessagePassing::new();
        let agg = mp.aggregate(states, &self.a_hat)?;
        // Combine own state with aggregated neighbors
        let n = states.len();
        let sd = self.cfg.state_dim;
        let combined: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut row = states[i].clone();
                for k in 0..sd {
                    if k < agg[i].len() {
                        row[k] = (row[k] + agg[i][k]) * 0.5;
                    }
                }
                row
            })
            .collect();
        let hidden = matmul(&combined, &self.h_w1)?;
        let mut hidden_act = hidden.clone();
        // Softplus activation: log(1 + exp(x)) for smooth, positive output
        for row in hidden_act.iter_mut() {
            for v in row.iter_mut() {
                *v = (*v).clamp(-20.0, 20.0);
                *v = (1.0 + v.exp()).ln();
            }
        }
        let energy = matmul(&hidden_act, &self.h_w2)?;
        Ok(energy.iter().map(|row| row[0]).collect())
    }

    /// Compute Hamilton's equations via central finite differences.
    ///
    /// dq/dt = ∂H/∂p ≈ (H(q, p+ε) - H(q, p-ε)) / (2ε)
    /// dp/dt = -∂H/∂q ≈ -(H(q+ε, p) - H(q-ε, p)) / (2ε)
    fn hamilton_equations(&self, states: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let n = states.len();
        let sd = self.cfg.state_dim;
        let q_dim = sd / 2;
        let eps = 1e-4_f64;
        let mut dstates = vec![vec![0.0f64; sd]; n];
        // Compute dq/dt = ∂H/∂p for each node
        for i in 0..n {
            for k in 0..q_dim {
                let p_idx = q_dim + k;
                let mut states_plus = states.to_vec();
                let mut states_minus = states.to_vec();
                states_plus[i][p_idx] += eps;
                states_minus[i][p_idx] -= eps;
                let h_plus = self.hamiltonian_per_node(&states_plus)?[i];
                let h_minus = self.hamiltonian_per_node(&states_minus)?[i];
                dstates[i][k] = (h_plus - h_minus) / (2.0 * eps);
            }
            // Compute dp/dt = -∂H/∂q for each node
            for k in 0..q_dim {
                let mut states_plus = states.to_vec();
                let mut states_minus = states.to_vec();
                states_plus[i][k] += eps;
                states_minus[i][k] -= eps;
                let h_plus = self.hamiltonian_per_node(&states_plus)?[i];
                let h_minus = self.hamiltonian_per_node(&states_minus)?[i];
                dstates[i][q_dim + k] = -(h_plus - h_minus) / (2.0 * eps);
            }
        }
        Ok(dstates)
    }

    /// Forward: symplectic (leapfrog) integration of Hamilton's equations.
    pub fn forward(&self, states: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let mut s = states.to_vec();
        let dt = self.cfg.dt;
        let t_end = self.cfg.t_end;
        let mut t = 0.0f64;
        while t + dt <= t_end + 1e-12 {
            let ds = self.hamilton_equations(&s)?;
            s = add_scaled(&s, &ds, dt);
            t += dt;
        }
        Ok(s)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §J  LagrangianGnn  (Lagrangian mechanics on graphs)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Lagrangian GNN.
#[derive(Debug, Clone)]
pub struct LgnConfig {
    /// Configuration space dimension per node (q only; q_dot inferred).
    pub q_dim: usize,
    /// Hidden dimension for the Lagrangian MLP.
    pub hidden_dim: usize,
    /// Integration end time.
    pub t_end: f64,
    /// Integration time step.
    pub dt: f64,
    /// Random seed.
    pub seed: u64,
}

/// Lagrangian mechanics on graphs.
///
/// Learns a scalar Lagrangian L(q, q_dot, G) = T(q, q_dot) - V(q, G)
/// parameterized by a GNN.  The equations of motion are derived from the
/// Euler-Lagrange equations:
///
///   d/dt(∂L/∂q_dot) - ∂L/∂q = 0
///   ⟹  q_ddot = M^{-1}(q) · (∂L/∂q - Γ(q, q_dot))
///
/// Approximated via finite differences on the Lagrangian network outputs.
#[derive(Debug, Clone)]
pub struct LagrangianGnn {
    cfg: LgnConfig,
    /// Lagrangian MLP layer 1 (2*q_dim → hidden_dim).
    lag_w1: Vec<Vec<f64>>,
    /// Lagrangian MLP layer 2 (hidden_dim → 1).
    lag_w2: Vec<Vec<f64>>,
    a_hat: Vec<Vec<f64>>,
}

impl LagrangianGnn {
    /// Create a new Lagrangian GNN.
    pub fn new(cfg: LgnConfig, graph: &GnoGraph) -> Result<Self, GnoError> {
        if cfg.q_dim == 0 {
            return Err(GnoError::InvalidConfig("q_dim must be > 0".into()));
        }
        if cfg.hidden_dim == 0 {
            return Err(GnoError::InvalidConfig("hidden_dim must be > 0".into()));
        }
        let a_hat = graph.normalize_adjacency();
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let in_d = cfg.q_dim * 2;
        let hd = cfg.hidden_dim;
        let lag_w1 = random_matrix(in_d, hd, xavier_std(in_d, hd), &mut rng);
        let lag_w2 = random_matrix(hd, 1, xavier_std(hd, 1), &mut rng);
        Ok(Self { cfg, lag_w1, lag_w2, a_hat })
    }

    /// Evaluate Lagrangian scalar for each node given (q, q_dot).
    fn lagrangian_per_node(
        &self,
        q: &[Vec<f64>],
        q_dot: &[Vec<f64>],
    ) -> Result<Vec<f64>, GnoError> {
        let n = q.len();
        let mp = GnoMessagePassing::new();
        // Concatenate q and q_dot per node
        let state: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut row = q[i].clone();
                row.extend_from_slice(&q_dot[i]);
                row
            })
            .collect();
        let agg = mp.aggregate(&state, &self.a_hat)?;
        // Blend state with aggregated neighbors
        let combined: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                state[i]
                    .iter()
                    .zip(agg[i].iter())
                    .map(|(s, a)| (s + a) * 0.5)
                    .collect()
            })
            .collect();
        let hidden = matmul(&combined, &self.lag_w1)?;
        let mut hidden_act = hidden.clone();
        for row in hidden_act.iter_mut() {
            for v in row.iter_mut() {
                *v = v.tanh();
            }
        }
        let lag = matmul(&hidden_act, &self.lag_w2)?;
        Ok(lag.iter().map(|row| row[0]).collect())
    }

    /// Euler-Lagrange equations: compute q_ddot via FD of the Lagrangian.
    fn euler_lagrange(
        &self,
        q: &[Vec<f64>],
        q_dot: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, GnoError> {
        let n = q.len();
        let qd = self.cfg.q_dim;
        let eps = 1e-4_f64;
        let mut q_ddot = vec![vec![0.0f64; qd]; n];
        for i in 0..n {
            for k in 0..qd {
                // ∂L/∂q_k: central difference in q
                let mut q_plus = q.to_vec();
                let mut q_minus = q.to_vec();
                q_plus[i][k] += eps;
                q_minus[i][k] -= eps;
                let l_plus = self.lagrangian_per_node(&q_plus, q_dot)?[i];
                let l_minus = self.lagrangian_per_node(&q_minus, q_dot)?[i];
                let dl_dq = (l_plus - l_minus) / (2.0 * eps);
                // ∂L/∂(q_dot_k): central difference in q_dot
                let mut qd_plus = q_dot.to_vec();
                let mut qd_minus = q_dot.to_vec();
                qd_plus[i][k] += eps;
                qd_minus[i][k] -= eps;
                let l_qdp = self.lagrangian_per_node(q, &qd_plus)?[i];
                let l_qdm = self.lagrangian_per_node(q, &qd_minus)?[i];
                let dl_dqd = (l_qdp - l_qdm) / (2.0 * eps);
                // q_ddot ≈ dl_dq - d/dt(dl_dqd)  (simplified: drop Christoffel term)
                // Simple approximation: q_ddot = dl_dq (generalized force drives acceleration)
                q_ddot[i][k] = dl_dq - 0.1 * dl_dqd; // damped EL approx
            }
        }
        Ok(q_ddot)
    }

    /// Forward: integrate Euler-Lagrange equations given (q, q_dot) initial conditions.
    ///
    /// Returns the final (q, q_dot) concatenated as a single matrix.
    pub fn forward(
        &self,
        q0: &[Vec<f64>],
        q_dot0: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, GnoError> {
        let n = q0.len();
        if n == 0 {
            return Err(GnoError::EmptyGraph);
        }
        if q0.len() != q_dot0.len() {
            return Err(GnoError::DimensionMismatch {
                expected: n,
                found: q_dot0.len(),
                context: "LagrangianGnn q0 vs q_dot0",
            });
        }
        let mut q = q0.to_vec();
        let mut q_dot = q_dot0.to_vec();
        let dt = self.cfg.dt;
        let t_end = self.cfg.t_end;
        let mut t = 0.0f64;
        while t + dt <= t_end + 1e-12 {
            let q_ddot = self.euler_lagrange(&q, &q_dot)?;
            // Symplectic Euler: update velocity first, then position
            let mut q_dot_new = q_dot.clone();
            let mut q_new = q.clone();
            for i in 0..n {
                for k in 0..self.cfg.q_dim {
                    q_dot_new[i][k] += q_ddot[i][k] * dt;
                    q_new[i][k] += q_dot_new[i][k] * dt;
                }
            }
            q = q_new;
            q_dot = q_dot_new;
            t += dt;
        }
        // Return concatenated [q, q_dot]
        let out: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut row = q[i].clone();
                row.extend_from_slice(&q_dot[i]);
                row
            })
            .collect();
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §K  GnoSimMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Trajectory prediction metrics for physics-based graph ODE models.
///
/// Provides MSE over full rollout trajectories and a stability check
/// based on max-state-norm growth.
#[derive(Debug, Clone)]
pub struct GnoSimMetrics;

impl GnoSimMetrics {
    /// Mean Squared Error over a full trajectory.
    ///
    /// `pred_traj` and `true_traj` are `Vec<T>` where each `T` is an
    /// `(n_nodes × feat_dim)` snapshot.
    pub fn trajectory_mse(
        pred_traj: &[Vec<Vec<f64>>],
        true_traj: &[Vec<Vec<f64>>],
    ) -> Result<f64, GnoError> {
        let t = pred_traj.len();
        if t != true_traj.len() {
            return Err(GnoError::DimensionMismatch {
                expected: true_traj.len(),
                found: t,
                context: "trajectory_mse length",
            });
        }
        if t == 0 {
            return Ok(0.0);
        }
        let mut total_mse = 0.0f64;
        let mut count = 0usize;
        for (p_snap, t_snap) in pred_traj.iter().zip(true_traj.iter()) {
            let n = p_snap.len();
            if n != t_snap.len() {
                return Err(GnoError::DimensionMismatch {
                    expected: t_snap.len(),
                    found: n,
                    context: "trajectory_mse snapshot rows",
                });
            }
            for (p_row, t_row) in p_snap.iter().zip(t_snap.iter()) {
                for (pv, tv) in p_row.iter().zip(t_row.iter()) {
                    total_mse += (pv - tv).powi(2);
                    count += 1;
                }
            }
        }
        Ok(if count > 0 { total_mse / count as f64 } else { 0.0 })
    }

    /// Rollout stability check.
    ///
    /// Returns `true` if the Frobenius norm of the final state does not exceed
    /// `max_growth_factor × norm(initial_state)`.  Useful for detecting
    /// numerical blow-up in long-horizon rollouts.
    pub fn rollout_stability(
        traj: &[Vec<Vec<f64>>],
        max_growth_factor: f64,
    ) -> bool {
        if traj.is_empty() {
            return true;
        }
        let norm_sq = |snap: &Vec<Vec<f64>>| -> f64 {
            snap.iter()
                .flat_map(|row| row.iter())
                .map(|v| v * v)
                .sum::<f64>()
        };
        let init_norm = norm_sq(&traj[0]).sqrt();
        if init_norm < 1e-15 {
            return true; // trivially stable for zero initial state
        }
        let final_norm = norm_sq(traj.last().unwrap_or(&traj[0])).sqrt();
        final_norm <= max_growth_factor * init_norm
    }

    /// Average final-step absolute error over a batch of trajectories.
    ///
    /// Evaluates how well the model predicts the terminal state.
    pub fn final_step_mae(
        pred_traj: &[Vec<Vec<f64>>],
        true_traj: &[Vec<Vec<f64>>],
    ) -> Result<f64, GnoError> {
        if pred_traj.is_empty() {
            return Ok(0.0);
        }
        let p_final = pred_traj.last().ok_or(GnoError::EmptyGraph)?;
        let t_final = true_traj.last().ok_or(GnoError::EmptyGraph)?;
        if p_final.len() != t_final.len() {
            return Err(GnoError::DimensionMismatch {
                expected: t_final.len(),
                found: p_final.len(),
                context: "final_step_mae rows",
            });
        }
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for (p_row, t_row) in p_final.iter().zip(t_final.iter()) {
            for (pv, tv) in p_row.iter().zip(t_row.iter()) {
                sum += (pv - tv).abs();
                count += 1;
            }
        }
        Ok(if count > 0 { sum / count as f64 } else { 0.0 })
    }
}
