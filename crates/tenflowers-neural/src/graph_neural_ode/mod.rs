//! Graph Neural ODEs — continuous-depth GNN dynamics on graph-structured data.
//!
//! Implements several continuous-time graph neural network architectures:
//!
//! - [`GnoGraph`]: Sparse weighted graph structure (adjacency list).
//! - [`GnoMessagePassing`]: Base sparse message-passing aggregation.
//! - [`GnoGcnLayer`]: Graph Convolutional Network layer (Â·H·W, Xavier init).
//! - [`GnoGatLayer`]: Multi-head Graph Attention Network layer.
//! - [`GnoGraphODE`]: ODE dynamics `dH/dt = f_θ(H, t, A)` backed by a GCN.
//! - [`GraphOdeSolver`]: Euler and RK4 integrators for graph ODE systems.
//! - [`GrandModel`]: GRAND — attention-based graph diffusion (Chamberlain 2021).
//! - [`CgnnModel`]: Continuous Graph Neural Network with latent ODE (RK4).
//! - [`GraphOdeModel`]: GraphODE with GCN encoder/decoder; arbitrary t-interpolation.
//! - [`StGnnOde`]: Spatio-Temporal GNN-ODE for traffic/time-series prediction.
//! - [`LatentGraphOde`]: Variational latent graph ODE (GRU encoder → z_0 → ODE).
//! - [`GnoNodeClassifier`]: Node-classification wrapper around GRAND / GraphODE.
//! - [`GraphOdeMetrics`]: MAE, R², node-classification accuracy, RTE.
//!
//! All computation is pure Rust (`f64`), no unsafe code, no `unwrap()`.

pub mod extensions;
pub use extensions::*;

pub mod continuous;
pub use continuous::*;

#[cfg(test)]
mod tests;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise in graph neural ODE operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GnoError {
    /// Node index is out of range.
    NodeOutOfRange { index: usize, num_nodes: usize },
    /// Matrix/vector dimensions do not agree.
    DimensionMismatch {
        expected: usize,
        found: usize,
        context: &'static str,
    },
    /// The graph has no nodes.
    EmptyGraph,
    /// Time span is invalid (t_end ≤ t_start or dt ≤ 0).
    InvalidTimeSpan { t_start: f64, t_end: f64 },
    /// Number of classes must be ≥ 2.
    InvalidNumClasses { found: usize },
    /// Layer configuration is invalid.
    InvalidConfig(String),
}

impl fmt::Display for GnoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GnoError::NodeOutOfRange { index, num_nodes } => {
                write!(
                    f,
                    "node index {index} out of range for graph with {num_nodes} nodes"
                )
            }
            GnoError::DimensionMismatch {
                expected,
                found,
                context,
            } => {
                write!(
                    f,
                    "dimension mismatch in {context}: expected {expected}, found {found}"
                )
            }
            GnoError::EmptyGraph => write!(f, "graph has no nodes"),
            GnoError::InvalidTimeSpan { t_start, t_end } => {
                write!(f, "invalid time span [{t_start}, {t_end}]")
            }
            GnoError::InvalidNumClasses { found } => {
                write!(f, "num_classes must be ≥ 2, got {found}")
            }
            GnoError::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for GnoError {}

// ─────────────────────────────────────────────────────────────────────────────
// Utility functions (pub(crate) so submodules can use them)
// ─────────────────────────────────────────────────────────────────────────────

/// Box-Muller transform: produces one N(0,1) sample from two U(0,1) inputs.
#[inline]
pub(crate) fn box_muller(u1: f64, u2: f64) -> f64 {
    let u1 = u1.max(1e-12);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Sample a 2-D matrix of N(0, std) values.
pub(crate) fn random_matrix(rows: usize, cols: usize, std: f64, rng: &mut StdRng) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    let u1 = rng.random::<f64>();
                    let u2 = rng.random::<f64>();
                    box_muller(u1, u2) * std
                })
                .collect()
        })
        .collect()
}

/// Xavier (Glorot) uniform init std: sqrt(2 / (fan_in + fan_out)).
#[inline]
pub(crate) fn xavier_std(fan_in: usize, fan_out: usize) -> f64 {
    (2.0 / (fan_in + fan_out) as f64).sqrt()
}

/// Matrix multiply: (m×k) · (k×n) → (m×n)
pub(crate) fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
    let m = a.len();
    if m == 0 {
        return Ok(Vec::new());
    }
    let k = a[0].len();
    let k2 = b.len();
    if k != k2 {
        return Err(GnoError::DimensionMismatch {
            expected: k,
            found: k2,
            context: "matmul k",
        });
    }
    let n = if b.is_empty() { 0 } else { b[0].len() };
    let mut out = vec![vec![0.0f64; n]; m];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i][p];
            for j in 0..n {
                out[i][j] += aip * b[p][j];
            }
        }
    }
    Ok(out)
}

/// Element-wise ReLU on a flat matrix.
pub(crate) fn relu_matrix(h: &mut [Vec<f64>]) {
    for row in h.iter_mut() {
        for v in row.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
    }
}

/// Tanh element-wise.
pub(crate) fn tanh_matrix(h: &mut [Vec<f64>]) {
    for row in h.iter_mut() {
        for v in row.iter_mut() {
            *v = v.tanh();
        }
    }
}

/// Row-wise softmax.
pub(crate) fn softmax_rows(h: &mut [Vec<f64>]) {
    for row in h.iter_mut() {
        let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = row.iter().map(|x| (x - max).exp()).sum();
        let sum = sum.max(1e-12);
        for v in row.iter_mut() {
            *v = (*v - max).exp() / sum;
        }
    }
}

/// Add matrix b into a in-place.
#[allow(dead_code)]
pub(crate) fn mat_add_inplace(a: &mut [Vec<f64>], b: &[Vec<f64>]) {
    for (row_a, row_b) in a.iter_mut().zip(b.iter()) {
        for (va, vb) in row_a.iter_mut().zip(row_b.iter()) {
            *va += vb;
        }
    }
}

/// Scale matrix in-place.
#[allow(dead_code)]
pub(crate) fn mat_scale_inplace(a: &mut [Vec<f64>], s: f64) {
    for row in a.iter_mut() {
        for v in row.iter_mut() {
            *v *= s;
        }
    }
}

/// Clone matrix and scale.
#[allow(dead_code)]
pub(crate) fn mat_scale(a: &[Vec<f64>], s: f64) -> Vec<Vec<f64>> {
    a.iter()
        .map(|row| row.iter().map(|v| v * s).collect())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  GnoGraph
// ─────────────────────────────────────────────────────────────────────────────

/// Sparse weighted graph: adjacency list representation.
///
/// Each entry `adj[u]` is a list of `(v, weight)` pairs.
#[derive(Debug, Clone)]
pub struct GnoGraph {
    adj: Vec<Vec<(usize, f64)>>,
}

impl GnoGraph {
    /// Create a new empty graph with `n` nodes.
    pub fn new(n: usize) -> Self {
        Self {
            adj: vec![Vec::new(); n],
        }
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.adj.len()
    }

    /// Add an undirected weighted edge (u, v, w).  Returns error if index OOB.
    pub fn add_edge(&mut self, u: usize, v: usize, w: f64) -> Result<(), GnoError> {
        let n = self.adj.len();
        if u >= n {
            return Err(GnoError::NodeOutOfRange {
                index: u,
                num_nodes: n,
            });
        }
        if v >= n {
            return Err(GnoError::NodeOutOfRange {
                index: v,
                num_nodes: n,
            });
        }
        self.adj[u].push((v, w));
        if u != v {
            self.adj[v].push((u, w));
        }
        Ok(())
    }

    /// Degree of node `u`.
    pub fn degree(&self, u: usize) -> f64 {
        self.adj[u].iter().map(|(_, w)| w).sum()
    }

    /// Compute D^{-1/2} A D^{-1/2} (symmetric normalisation with self-loops).
    /// Returns a dense n×n matrix.
    pub fn normalize_adjacency(&self) -> Vec<Vec<f64>> {
        let n = self.adj.len();
        // Add self-loops: degree includes the self-loop weight 1.
        let mut deg = vec![1.0f64; n]; // self-loop
        for u in 0..n {
            for (_, w) in &self.adj[u] {
                deg[u] += w;
            }
        }
        let dinv_sqrt: Vec<f64> = deg.iter().map(|d| 1.0 / d.sqrt()).collect();
        let mut a_hat = vec![vec![0.0f64; n]; n];
        // Self-loops
        for i in 0..n {
            a_hat[i][i] = dinv_sqrt[i] * 1.0 * dinv_sqrt[i];
        }
        // Edges
        for u in 0..n {
            for (v, w) in &self.adj[u] {
                a_hat[u][*v] += dinv_sqrt[u] * w * dinv_sqrt[*v];
            }
        }
        a_hat
    }

    /// Graph Laplacian L = D - A (using raw weights, no self-loops).
    pub fn laplacian(&self) -> Vec<Vec<f64>> {
        let n = self.adj.len();
        let mut lap = vec![vec![0.0f64; n]; n];
        for u in 0..n {
            let d = self.degree(u);
            lap[u][u] = d;
            for (v, w) in &self.adj[u] {
                lap[u][*v] -= w;
            }
        }
        lap
    }

    /// Adjacency list (read-only).
    pub fn adj(&self) -> &[Vec<(usize, f64)>] {
        &self.adj
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  GnoMessagePassing
// ─────────────────────────────────────────────────────────────────────────────

/// Base message-passing aggregation: sparse mat-vec for each feature column.
///
/// `aggregate(node_feats, adj)` computes  Â · H  where  Â = normalize_adjacency.
#[derive(Debug, Clone)]
pub struct GnoMessagePassing;

impl GnoMessagePassing {
    /// Create a new message-passing aggregator.
    pub fn new() -> Self {
        Self
    }

    /// Aggregate node features via sparse Â · H.
    ///
    /// `node_feats`: shape (n_nodes, feat_dim)
    /// `a_hat`:      shape (n_nodes, n_nodes) — normalised adjacency
    pub fn aggregate(
        &self,
        node_feats: &[Vec<f64>],
        a_hat: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, GnoError> {
        let n = node_feats.len();
        if n == 0 {
            return Err(GnoError::EmptyGraph);
        }
        let d = node_feats[0].len();
        if a_hat.len() != n {
            return Err(GnoError::DimensionMismatch {
                expected: n,
                found: a_hat.len(),
                context: "aggregate a_hat rows",
            });
        }
        let mut out = vec![vec![0.0f64; d]; n];
        for i in 0..n {
            for j in 0..n {
                let w = a_hat[i][j];
                if w.abs() < 1e-15 {
                    continue;
                }
                for k in 0..d {
                    out[i][k] += w * node_feats[j][k];
                }
            }
        }
        Ok(out)
    }
}

impl Default for GnoMessagePassing {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  GnoGcnLayer
// ─────────────────────────────────────────────────────────────────────────────

/// One GCN layer: H' = σ(Â · H · W)
///
/// Weight W is Xavier-initialised.
#[derive(Debug, Clone)]
pub struct GnoGcnLayer {
    /// Weight matrix (in_dim × out_dim).
    pub weight: Vec<Vec<f64>>,
    /// Input feature dimension.
    pub in_dim: usize,
    /// Output feature dimension.
    pub out_dim: usize,
    mp: GnoMessagePassing,
}

impl GnoGcnLayer {
    /// Create a new GCN layer, seed for reproducibility.
    pub fn new(in_dim: usize, out_dim: usize, seed: u64) -> Result<Self, GnoError> {
        if in_dim == 0 || out_dim == 0 {
            return Err(GnoError::InvalidConfig(
                "in_dim and out_dim must be > 0".into(),
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let std = xavier_std(in_dim, out_dim);
        let weight = random_matrix(in_dim, out_dim, std, &mut rng);
        Ok(Self {
            weight,
            in_dim,
            out_dim,
            mp: GnoMessagePassing::new(),
        })
    }

    /// Forward pass: H' = ReLU(Â · H · W)
    pub fn forward(
        &self,
        node_feats: &[Vec<f64>],
        a_hat: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, GnoError> {
        let aggregated = self.mp.aggregate(node_feats, a_hat)?;
        let mut out = matmul(&aggregated, &self.weight)?;
        relu_matrix(&mut out);
        Ok(out)
    }

    /// Forward without activation (raw linear output).
    pub fn forward_linear(
        &self,
        node_feats: &[Vec<f64>],
        a_hat: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, GnoError> {
        let aggregated = self.mp.aggregate(node_feats, a_hat)?;
        matmul(&aggregated, &self.weight)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  GnoGraphODE
// ─────────────────────────────────────────────────────────────────────────────

/// ODE dynamics: dH/dt = f_θ(H, t, A) backed by a GnoGcnLayer.
///
/// The vector field is `f_θ(H, t) = GCN(H, Â)` (time-invariant here;
/// the `t` parameter is kept for generality).
#[derive(Debug, Clone)]
pub struct GnoGraphODE {
    /// The GCN layer defining the ODE vector field.
    pub layer: GnoGcnLayer,
    /// Normalised adjacency matrix.
    pub a_hat: Vec<Vec<f64>>,
}

impl GnoGraphODE {
    /// Create a new graph ODE from dimensions and adjacency.
    pub fn new(
        in_dim: usize,
        hidden_dim: usize,
        a_hat: Vec<Vec<f64>>,
        seed: u64,
    ) -> Result<Self, GnoError> {
        let layer = GnoGcnLayer::new(in_dim, hidden_dim, seed)?;
        Ok(Self { layer, a_hat })
    }

    /// Evaluate dH/dt = GCN(H, Â).
    pub fn dynamics(&self, h: &[Vec<f64>], _t: f64) -> Result<Vec<Vec<f64>>, GnoError> {
        self.layer.forward(h, &self.a_hat)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  GraphOdeSolver
// ─────────────────────────────────────────────────────────────────────────────

/// ODE integration method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GnoOdeMethod {
    /// First-order Euler method.
    Euler,
    /// Classical 4th-order Runge-Kutta method.
    Rk4,
}

/// Euler and RK4 integrators for `GnoGraphODE`.
#[derive(Debug, Clone)]
pub struct GraphOdeSolver {
    /// The integration method to use.
    pub method: GnoOdeMethod,
}

impl GraphOdeSolver {
    /// Create a new solver with the given method.
    pub fn new(method: GnoOdeMethod) -> Self {
        Self { method }
    }

    /// One integration step: H(t + dt) ← H(t).
    pub fn step(
        &self,
        ode: &GnoGraphODE,
        h: &[Vec<f64>],
        t: f64,
        dt: f64,
    ) -> Result<Vec<Vec<f64>>, GnoError> {
        match self.method {
            GnoOdeMethod::Euler => self.euler_step(ode, h, t, dt),
            GnoOdeMethod::Rk4 => self.rk4_step(ode, h, t, dt),
        }
    }

    fn euler_step(
        &self,
        ode: &GnoGraphODE,
        h: &[Vec<f64>],
        t: f64,
        dt: f64,
    ) -> Result<Vec<Vec<f64>>, GnoError> {
        let k = ode.dynamics(h, t)?;
        let mut h_next = h.to_vec();
        for (row, k_row) in h_next.iter_mut().zip(k.iter()) {
            for (v, kv) in row.iter_mut().zip(k_row.iter()) {
                *v += dt * kv;
            }
        }
        Ok(h_next)
    }

    fn rk4_step(
        &self,
        ode: &GnoGraphODE,
        h: &[Vec<f64>],
        t: f64,
        dt: f64,
    ) -> Result<Vec<Vec<f64>>, GnoError> {
        let k1 = ode.dynamics(h, t)?;

        let h2 = add_scaled(h, &k1, dt / 2.0);
        let k2 = ode.dynamics(&h2, t + dt / 2.0)?;

        let h3 = add_scaled(h, &k2, dt / 2.0);
        let k3 = ode.dynamics(&h3, t + dt / 2.0)?;

        let h4 = add_scaled(h, &k3, dt);
        let k4 = ode.dynamics(&h4, t + dt)?;

        // h + dt/6 * (k1 + 2k2 + 2k3 + k4)
        let n = h.len();
        let d = if n > 0 { h[0].len() } else { 0 };
        let mut h_next = vec![vec![0.0f64; d]; n];
        for i in 0..n {
            for j in 0..d {
                h_next[i][j] =
                    h[i][j] + (dt / 6.0) * (k1[i][j] + 2.0 * k2[i][j] + 2.0 * k3[i][j] + k4[i][j]);
            }
        }
        Ok(h_next)
    }

    /// Integrate from `t_start` to `t_end` with fixed step `dt`.
    /// Returns a trajectory: Vec over time steps of node-feature matrices.
    pub fn solve(
        &self,
        ode: &GnoGraphODE,
        h0: Vec<Vec<f64>>,
        t_start: f64,
        t_end: f64,
        dt: f64,
    ) -> Result<Vec<Vec<Vec<f64>>>, GnoError> {
        if t_end <= t_start || dt <= 0.0 {
            return Err(GnoError::InvalidTimeSpan { t_start, t_end });
        }
        let mut trajectory = Vec::new();
        let mut h = h0;
        let mut t = t_start;
        trajectory.push(h.clone());
        while t + dt <= t_end + 1e-12 {
            h = self.step(ode, &h, t, dt)?;
            t += dt;
            trajectory.push(h.clone());
        }
        Ok(trajectory)
    }
}

/// Helper: new_h[i][j] = h[i][j] + scale * delta[i][j]
pub(crate) fn add_scaled(h: &[Vec<f64>], delta: &[Vec<f64>], scale: f64) -> Vec<Vec<f64>> {
    h.iter()
        .zip(delta.iter())
        .map(|(row, drow)| {
            row.iter()
                .zip(drow.iter())
                .map(|(v, d)| v + scale * d)
                .collect()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  GrandModel  (Graph Neural Diffusion, Chamberlain 2021)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the GRAND model.
#[derive(Debug, Clone)]
pub struct GrandConfig {
    /// Number of nodes in the graph.
    pub num_nodes: usize,
    /// Input feature dimension per node.
    pub feat_dim: usize,
    /// Hidden dimension for attention.
    pub hidden_dim: usize,
    /// Number of output classes.
    pub num_classes: usize,
    /// Integration end time.
    pub t_end: f64,
    /// Integration time step.
    pub dt: f64,
    /// Random seed.
    pub seed: u64,
}

/// GRAND: attention-based graph diffusion.
///
/// ∂H/∂t = div(A(H) · ∇H)
///
/// Attention: A_ij = softmax(h_i W_Q (W_K^T h_j) / sqrt(d_head))
/// Dynamics used: Euler integration on [0, T].
#[derive(Debug, Clone)]
pub struct GrandModel {
    /// Model configuration.
    pub cfg: GrandConfig,
    /// Query projection (feat_dim × hidden_dim).
    pub w_q: Vec<Vec<f64>>,
    /// Key projection (feat_dim × hidden_dim).
    pub w_k: Vec<Vec<f64>>,
    /// Classification head (feat_dim × num_classes).
    pub cls_weight: Vec<Vec<f64>>,
}

impl GrandModel {
    /// Create a new GRAND model.
    pub fn new(cfg: GrandConfig) -> Result<Self, GnoError> {
        if cfg.num_classes < 2 {
            return Err(GnoError::InvalidNumClasses {
                found: cfg.num_classes,
            });
        }
        if cfg.num_nodes == 0 {
            return Err(GnoError::EmptyGraph);
        }
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let std_qk = xavier_std(cfg.feat_dim, cfg.hidden_dim);
        let w_q = random_matrix(cfg.feat_dim, cfg.hidden_dim, std_qk, &mut rng);
        let w_k = random_matrix(cfg.feat_dim, cfg.hidden_dim, std_qk, &mut rng);
        // cls_weight maps feat_dim → num_classes (diffusion output stays in feat_dim space)
        let std_cls = xavier_std(cfg.feat_dim, cfg.num_classes);
        let cls_weight = random_matrix(cfg.feat_dim, cfg.num_classes, std_cls, &mut rng);
        Ok(Self {
            cfg,
            w_q,
            w_k,
            cls_weight,
        })
    }

    /// Compute attention-weighted adjacency A(H): n×n matrix.
    fn attention_adjacency(&self, h: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let n = h.len();
        let d = self.cfg.hidden_dim;
        // Q = H W_Q  (n × d)
        let q = matmul(h, &self.w_q)?;
        // K = H W_K  (n × d)
        let k = matmul(h, &self.w_k)?;
        let scale = (d as f64).sqrt().max(1e-8);
        // Raw attention scores
        let mut attn = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0f64;
                for dk in 0..d {
                    dot += q[i][dk] * k[j][dk];
                }
                attn[i][j] = dot / scale;
            }
        }
        softmax_rows(&mut attn);
        Ok(attn)
    }

    /// Diffusion step: dH/dt = (A(H) - I) · H  (Laplacian-style diffusion).
    fn dynamics(&self, h: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let n = h.len();
        let a = self.attention_adjacency(h)?;
        // dH = A·H - H
        let ah = matmul(&a, h)?;
        let mut dh = ah;
        for i in 0..n {
            for j in 0..h[0].len() {
                dh[i][j] -= h[i][j];
            }
        }
        Ok(dh)
    }

    /// Forward: run Euler ODE, return logits for each node.
    pub fn forward(&self, h0: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let mut h = h0.to_vec();
        let dt = self.cfg.dt;
        let t_end = self.cfg.t_end;
        let mut t = 0.0f64;
        while t + dt <= t_end + 1e-12 {
            let dh = self.dynamics(&h)?;
            let mut h_new = h.clone();
            for i in 0..h_new.len() {
                for j in 0..h_new[i].len() {
                    h_new[i][j] += dt * dh[i][j];
                }
            }
            h = h_new;
            t += dt;
        }
        // Classification head: logits = H · cls_weight
        let logits = matmul(&h, &self.cls_weight)?;
        Ok(logits)
    }

    /// Predict class labels (argmax of logits).
    pub fn predict(&self, h0: &[Vec<f64>]) -> Result<Vec<usize>, GnoError> {
        let logits = self.forward(h0)?;
        logits
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .ok_or(GnoError::EmptyGraph)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  CgnnModel  (Continuous Graph Neural Network)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for CGNN.
#[derive(Debug, Clone)]
pub struct CgnnConfig {
    /// Input feature dimension.
    pub feat_dim: usize,
    /// Hidden layer dimension.
    pub hidden_dim: usize,
    /// Number of MLP layers.
    pub num_layers: usize,
    /// Integration end time.
    pub t_end: f64,
    /// Integration time step.
    pub dt: f64,
    /// Random seed.
    pub seed: u64,
}

/// Continuous GNN: latent ODE on graph, RK4 solver.
///
/// h_v(t) = ODE_v(t, {h_u(0)}_{u ∈ N(v)})
/// The node update function is an MLP(concat(h_v, aggregated_neighbours)).
#[derive(Debug, Clone)]
pub struct CgnnModel {
    /// Model configuration.
    pub cfg: CgnnConfig,
    /// MLP weights: list of (W, b) per layer.
    pub mlp_weights: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    /// Normalised adjacency matrix.
    pub a_hat: Vec<Vec<f64>>,
}

impl CgnnModel {
    /// Create a new CGNN model.
    pub fn new(cfg: CgnnConfig, graph: &GnoGraph) -> Result<Self, GnoError> {
        if cfg.feat_dim == 0 || cfg.hidden_dim == 0 {
            return Err(GnoError::InvalidConfig(
                "feat_dim and hidden_dim must be > 0".into(),
            ));
        }
        let a_hat = graph.normalize_adjacency();
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let mut mlp_weights = Vec::new();
        // Input: concat(h_v, agg_v) → 2*feat_dim → hidden_dim → ... → feat_dim
        let layer_dims: Vec<usize> = {
            let mut dims = vec![cfg.feat_dim * 2];
            for _ in 0..cfg.num_layers.saturating_sub(1) {
                dims.push(cfg.hidden_dim);
            }
            dims.push(cfg.feat_dim);
            dims
        };
        for i in 0..layer_dims.len() - 1 {
            let in_d = layer_dims[i];
            let out_d = layer_dims[i + 1];
            let std = xavier_std(in_d, out_d);
            let w = random_matrix(in_d, out_d, std, &mut rng);
            let b = vec![0.0f64; out_d];
            mlp_weights.push((w, b));
        }
        Ok(Self {
            cfg,
            mlp_weights,
            a_hat,
        })
    }

    /// MLP forward on a single vector.
    fn mlp_forward(&self, x: &[f64]) -> Vec<f64> {
        let mut h = x.to_vec();
        let n_layers = self.mlp_weights.len();
        for (idx, (w, b)) in self.mlp_weights.iter().enumerate() {
            let out_d = b.len();
            let mut out = vec![0.0f64; out_d];
            for (j, bj) in b.iter().enumerate().take(out_d) {
                let in_d = h.len();
                let mut val = *bj;
                for p in 0..in_d {
                    if p < w.len() && j < w[p].len() {
                        val += h[p] * w[p][j];
                    }
                }
                out[j] = val;
            }
            // ReLU except last layer
            if idx < n_layers - 1 {
                for v in out.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
            h = out;
        }
        h
    }

    /// Node ODE dynamics: dH/dt = MLP(concat(H, Â·H)).
    fn dynamics(&self, h: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let mp = GnoMessagePassing::new();
        let agg = mp.aggregate(h, &self.a_hat)?;
        let n = h.len();
        let mut dh = Vec::with_capacity(n);
        for i in 0..n {
            let mut x: Vec<f64> = h[i].clone();
            x.extend_from_slice(&agg[i]);
            dh.push(self.mlp_forward(&x));
        }
        Ok(dh)
    }

    fn rk4_step(&self, h: &[Vec<f64>], _t: f64, dt: f64) -> Result<Vec<Vec<f64>>, GnoError> {
        let k1 = self.dynamics(h)?;
        let h2 = add_scaled(h, &k1, dt / 2.0);
        let k2 = self.dynamics(&h2)?;
        let h3 = add_scaled(h, &k2, dt / 2.0);
        let k3 = self.dynamics(&h3)?;
        let h4 = add_scaled(h, &k3, dt);
        let k4 = self.dynamics(&h4)?;
        let n = h.len();
        let d = if n > 0 { h[0].len() } else { 0 };
        let mut h_next = vec![vec![0.0f64; d]; n];
        for i in 0..n {
            for j in 0..d {
                h_next[i][j] =
                    h[i][j] + (dt / 6.0) * (k1[i][j] + 2.0 * k2[i][j] + 2.0 * k3[i][j] + k4[i][j]);
            }
        }
        Ok(h_next)
    }

    /// Run the continuous GNN, return final node features.
    pub fn forward(&self, h0: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let mut h = h0.to_vec();
        let dt = self.cfg.dt;
        let t_end = self.cfg.t_end;
        let mut t = 0.0f64;
        while t + dt <= t_end + 1e-12 {
            h = self.rk4_step(&h, t, dt)?;
            t += dt;
        }
        Ok(h)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  GraphOdeModel  (Huang 2021)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for GraphODE.
#[derive(Debug, Clone)]
pub struct GraphOdeConfig {
    /// Input feature dimension.
    pub feat_dim: usize,
    /// Latent space dimension.
    pub latent_dim: usize,
    /// Number of output classes.
    pub num_classes: usize,
    /// Integration end time.
    pub t_end: f64,
    /// Integration time step.
    pub dt: f64,
    /// Random seed.
    pub seed: u64,
}

/// GraphODE: encode initial conditions → solve ODE → decode at arbitrary t.
#[derive(Debug, Clone)]
pub struct GraphOdeModel {
    /// Model configuration.
    pub cfg: GraphOdeConfig,
    /// GCN encoder (feat_dim → latent_dim).
    pub encoder: GnoGcnLayer,
    /// Linear decoder (latent_dim → num_classes).
    pub decoder_weight: Vec<Vec<f64>>,
    /// Decoder bias.
    pub decoder_bias: Vec<f64>,
    /// ODE dynamics weight (latent_dim → latent_dim).
    pub ode_weight: Vec<Vec<f64>>,
    /// Normalised adjacency matrix.
    pub a_hat: Vec<Vec<f64>>,
}

impl GraphOdeModel {
    /// Create a new GraphODE model.
    pub fn new(cfg: GraphOdeConfig, graph: &GnoGraph) -> Result<Self, GnoError> {
        if cfg.feat_dim == 0 || cfg.latent_dim == 0 {
            return Err(GnoError::InvalidConfig(
                "feat_dim and latent_dim must be > 0".into(),
            ));
        }
        if cfg.num_classes < 2 {
            return Err(GnoError::InvalidNumClasses {
                found: cfg.num_classes,
            });
        }
        let a_hat = graph.normalize_adjacency();
        let encoder = GnoGcnLayer::new(cfg.feat_dim, cfg.latent_dim, cfg.seed)?;
        let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(1));
        let dec_std = xavier_std(cfg.latent_dim, cfg.num_classes);
        let decoder_weight = random_matrix(cfg.latent_dim, cfg.num_classes, dec_std, &mut rng);
        let decoder_bias = vec![0.0f64; cfg.num_classes];
        let ode_std = xavier_std(cfg.latent_dim, cfg.latent_dim);
        let ode_weight = random_matrix(cfg.latent_dim, cfg.latent_dim, ode_std, &mut rng);
        Ok(Self {
            cfg,
            encoder,
            decoder_weight,
            decoder_bias,
            a_hat,
            ode_weight,
        })
    }

    /// ODE dynamics: dZ/dt = tanh(Â · Z · W_ode).
    fn ode_dynamics(&self, z: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let mp = GnoMessagePassing::new();
        let agg = mp.aggregate(z, &self.a_hat)?;
        let mut out = matmul(&agg, &self.ode_weight)?;
        tanh_matrix(&mut out);
        Ok(out)
    }

    fn euler_step(&self, z: &[Vec<f64>], dt: f64) -> Result<Vec<Vec<f64>>, GnoError> {
        let dz = self.ode_dynamics(z)?;
        Ok(add_scaled(z, &dz, dt))
    }

    fn rk4_step(&self, z: &[Vec<f64>], dt: f64) -> Result<Vec<Vec<f64>>, GnoError> {
        let k1 = self.ode_dynamics(z)?;
        let z2 = add_scaled(z, &k1, dt / 2.0);
        let k2 = self.ode_dynamics(&z2)?;
        let z3 = add_scaled(z, &k2, dt / 2.0);
        let k3 = self.ode_dynamics(&z3)?;
        let z4 = add_scaled(z, &k3, dt);
        let k4 = self.ode_dynamics(&z4)?;
        let n = z.len();
        let d = if n > 0 { z[0].len() } else { 0 };
        let mut z_next = vec![vec![0.0f64; d]; n];
        for i in 0..n {
            for j in 0..d {
                z_next[i][j] =
                    z[i][j] + (dt / 6.0) * (k1[i][j] + 2.0 * k2[i][j] + 2.0 * k3[i][j] + k4[i][j]);
            }
        }
        Ok(z_next)
    }

    /// Decode latent z to logits.
    fn decode(&self, z: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let mut out = matmul(z, &self.decoder_weight)?;
        for row in out.iter_mut() {
            for (v, b) in row.iter_mut().zip(self.decoder_bias.iter()) {
                *v += b;
            }
        }
        Ok(out)
    }

    /// Encode → solve ODE → decode at t_end. Returns logits.
    pub fn forward(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let z0 = self.encoder.forward(x, &self.a_hat)?;
        let logits = self.forward_from_latent(&z0, self.cfg.t_end, GnoOdeMethod::Rk4)?;
        Ok(logits)
    }

    /// Interpolate at arbitrary t.
    pub fn forward_from_latent(
        &self,
        z0: &[Vec<f64>],
        t_query: f64,
        method: GnoOdeMethod,
    ) -> Result<Vec<Vec<f64>>, GnoError> {
        let dt = self.cfg.dt;
        if t_query < 0.0 {
            return Err(GnoError::InvalidTimeSpan {
                t_start: 0.0,
                t_end: t_query,
            });
        }
        let mut z = z0.to_vec();
        let mut t = 0.0f64;
        while t + dt <= t_query + 1e-12 {
            z = match method {
                GnoOdeMethod::Euler => self.euler_step(&z, dt)?,
                GnoOdeMethod::Rk4 => self.rk4_step(&z, dt)?,
            };
            t += dt;
        }
        self.decode(&z)
    }

    /// Predict class labels.
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<usize>, GnoError> {
        let logits = self.forward(x)?;
        logits
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .ok_or(GnoError::EmptyGraph)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  StGnnOde  (Spatio-Temporal GNN-ODE)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the spatio-temporal GNN-ODE.
#[derive(Debug, Clone)]
pub struct StGnnOdeConfig {
    /// Number of graph nodes.
    pub num_nodes: usize,
    /// Input feature dimension.
    pub feat_dim: usize,
    /// Spatial GCN hidden size.
    pub spatial_hidden: usize,
    /// Temporal attention hidden size.
    pub temporal_hidden: usize,
    /// Integration end time.
    pub t_end: f64,
    /// Integration time step.
    pub dt: f64,
    /// Forecast horizon (number of future steps to predict).
    pub horizon: usize,
    /// Random seed.
    pub seed: u64,
}

/// Spatio-Temporal GNN with ODE: h(t+Δt) = h(t) + ODE(h(t), A_spatial, A_temporal, t).
///
/// Spatial: GCN on A_spatial (road network / sensor graph).
/// Temporal: self-attention on the time dimension.
/// Suitable for traffic-speed / sensor forecasting.
#[derive(Debug, Clone)]
pub struct StGnnOde {
    /// Model configuration.
    pub cfg: StGnnOdeConfig,
    /// Spatial GCN weight (feat_dim → spatial_hidden).
    pub spatial_w: Vec<Vec<f64>>,
    /// Temporal attention query projection.
    pub t_w_q: Vec<Vec<f64>>,
    /// Temporal attention key projection.
    pub t_w_k: Vec<Vec<f64>>,
    /// Temporal attention value projection.
    pub t_w_v: Vec<Vec<f64>>,
    /// Output projection (spatial_hidden + temporal_hidden → feat_dim).
    pub out_w: Vec<Vec<f64>>,
    /// Forecast MLP (feat_dim → horizon).
    pub forecast_w: Vec<Vec<f64>>,
    /// Normalised adjacency matrix.
    pub a_hat: Vec<Vec<f64>>,
}

impl StGnnOde {
    /// Create a new spatio-temporal GNN-ODE.
    pub fn new(cfg: StGnnOdeConfig, graph: &GnoGraph) -> Result<Self, GnoError> {
        if cfg.feat_dim == 0 {
            return Err(GnoError::InvalidConfig("feat_dim must be > 0".into()));
        }
        let a_hat = graph.normalize_adjacency();
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let spatial_w = random_matrix(
            cfg.feat_dim,
            cfg.spatial_hidden,
            xavier_std(cfg.feat_dim, cfg.spatial_hidden),
            &mut rng,
        );
        let th = cfg.temporal_hidden;
        let t_w_q = random_matrix(cfg.feat_dim, th, xavier_std(cfg.feat_dim, th), &mut rng);
        let t_w_k = random_matrix(cfg.feat_dim, th, xavier_std(cfg.feat_dim, th), &mut rng);
        let t_w_v = random_matrix(cfg.feat_dim, th, xavier_std(cfg.feat_dim, th), &mut rng);
        let combined = cfg.spatial_hidden + th;
        let out_w = random_matrix(
            combined,
            cfg.feat_dim,
            xavier_std(combined, cfg.feat_dim),
            &mut rng,
        );
        let forecast_w = random_matrix(
            cfg.feat_dim,
            cfg.horizon,
            xavier_std(cfg.feat_dim, cfg.horizon),
            &mut rng,
        );
        Ok(Self {
            cfg,
            spatial_w,
            t_w_q,
            t_w_k,
            t_w_v,
            out_w,
            forecast_w,
            a_hat,
        })
    }

    /// Spatial aggregation: Â · H · W_spatial.
    fn spatial_conv(&self, h: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let mp = GnoMessagePassing::new();
        let agg = mp.aggregate(h, &self.a_hat)?;
        let mut out = matmul(&agg, &self.spatial_w)?;
        relu_matrix(&mut out);
        Ok(out)
    }

    /// Temporal self-attention (treating nodes as tokens).
    fn temporal_attn(&self, h: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let q = matmul(h, &self.t_w_q)?;
        let k = matmul(h, &self.t_w_k)?;
        let v = matmul(h, &self.t_w_v)?;
        let n = h.len();
        let d = self.cfg.temporal_hidden;
        let scale = (d as f64).sqrt().max(1e-8);
        let mut attn = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0f64;
                for dk in 0..d {
                    dot += q[i][dk] * k[j][dk];
                }
                attn[i][j] = dot / scale;
            }
        }
        softmax_rows(&mut attn);
        // Output = attn · V
        matmul(&attn, &v)
    }

    /// ODE dynamics for the ST-GNN.
    fn dynamics(&self, h: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let spatial = self.spatial_conv(h)?;
        let temporal = self.temporal_attn(h)?;
        // Concatenate spatial and temporal features
        let n = h.len();
        let combined: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut row = spatial[i].clone();
                row.extend_from_slice(&temporal[i]);
                row
            })
            .collect();
        let mut out = matmul(&combined, &self.out_w)?;
        relu_matrix(&mut out);
        Ok(out)
    }

    /// Forward: run ODE and return forecast for each node.
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
        // Forecast
        let forecast = matmul(&h, &self.forecast_w)?;
        Ok(forecast)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  LatentGraphOde  (Variational)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Latent Graph ODE.
#[derive(Debug, Clone)]
pub struct LatentGraphOdeConfig {
    /// Input feature dimension.
    pub feat_dim: usize,
    /// Latent space dimension.
    pub latent_dim: usize,
    /// GRU hidden dimension.
    pub hidden_dim: usize,
    /// Number of output classes.
    pub num_classes: usize,
    /// Integration end time.
    pub t_end: f64,
    /// Integration time step.
    pub dt: f64,
    /// Random seed.
    pub seed: u64,
}

/// Variational Latent Graph ODE.
///
/// q(z_0 | X) — GRU encoder → (μ, log_σ²) → reparameterize → z_0.
/// z(t) = ODE(z_0, t).
/// p(X | z(t)) — decoder.
#[derive(Debug, Clone)]
pub struct LatentGraphOde {
    /// Model configuration.
    pub cfg: LatentGraphOdeConfig,
    /// GRU update gate weights (feat_dim + latent_dim → latent_dim).
    gru_wz: Vec<Vec<f64>>,
    /// GRU reset gate weights.
    gru_wr: Vec<Vec<f64>>,
    /// GRU candidate hidden state weights.
    gru_wh: Vec<Vec<f64>>,
    /// Posterior head: (latent_dim → 2*latent_dim) for (μ, log_σ²).
    enc_out: Vec<Vec<f64>>,
    /// ODE weight (latent_dim → latent_dim).
    ode_w: Vec<Vec<f64>>,
    /// Decoder weight (latent_dim → num_classes).
    dec_w: Vec<Vec<f64>>,
    /// Decoder bias.
    dec_b: Vec<f64>,
    /// Normalised adjacency matrix.
    a_hat: Vec<Vec<f64>>,
}

impl LatentGraphOde {
    /// Create a new Latent Graph ODE.
    pub fn new(cfg: LatentGraphOdeConfig, graph: &GnoGraph) -> Result<Self, GnoError> {
        if cfg.feat_dim == 0 || cfg.latent_dim == 0 {
            return Err(GnoError::InvalidConfig(
                "feat_dim and latent_dim must be > 0".into(),
            ));
        }
        if cfg.num_classes < 2 {
            return Err(GnoError::InvalidNumClasses {
                found: cfg.num_classes,
            });
        }
        let a_hat = graph.normalize_adjacency();
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let in_d = cfg.feat_dim + cfg.latent_dim;
        let ld = cfg.latent_dim;
        let gru_wz = random_matrix(in_d, ld, xavier_std(in_d, ld), &mut rng);
        let gru_wr = random_matrix(in_d, ld, xavier_std(in_d, ld), &mut rng);
        let gru_wh = random_matrix(in_d, ld, xavier_std(in_d, ld), &mut rng);
        let enc_out = random_matrix(ld, 2 * ld, xavier_std(ld, 2 * ld), &mut rng);
        let ode_w = random_matrix(ld, ld, xavier_std(ld, ld), &mut rng);
        let dec_w = random_matrix(
            ld,
            cfg.num_classes,
            xavier_std(ld, cfg.num_classes),
            &mut rng,
        );
        let dec_b = vec![0.0f64; cfg.num_classes];
        Ok(Self {
            cfg,
            gru_wz,
            gru_wr,
            gru_wh,
            enc_out,
            ode_w,
            dec_w,
            dec_b,
            a_hat,
        })
    }

    /// Sigmoid function.
    #[inline]
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// GRU update for one node (x: input; h: hidden state).
    fn gru_step_vec(&self, x: &[f64], h: &[f64]) -> Vec<f64> {
        let ld = self.cfg.latent_dim;
        let mut xh: Vec<f64> = x.to_vec();
        xh.extend_from_slice(h);
        // z gate
        let mut z = vec![0.0f64; ld];
        for j in 0..ld {
            let mut v = 0.0f64;
            for p in 0..xh.len() {
                if p < self.gru_wz.len() && j < self.gru_wz[p].len() {
                    v += xh[p] * self.gru_wz[p][j];
                }
            }
            z[j] = Self::sigmoid(v);
        }
        // r gate
        let mut r = vec![0.0f64; ld];
        for j in 0..ld {
            let mut v = 0.0f64;
            for p in 0..xh.len() {
                if p < self.gru_wr.len() && j < self.gru_wr[p].len() {
                    v += xh[p] * self.gru_wr[p][j];
                }
            }
            r[j] = Self::sigmoid(v);
        }
        // candidate
        let mut xrh: Vec<f64> = x.to_vec();
        let rh: Vec<f64> = r.iter().zip(h.iter()).map(|(ri, hi)| ri * hi).collect();
        xrh.extend_from_slice(&rh);
        let mut h_tilde = vec![0.0f64; ld];
        for j in 0..ld {
            let mut v = 0.0f64;
            for p in 0..xrh.len() {
                if p < self.gru_wh.len() && j < self.gru_wh[p].len() {
                    v += xrh[p] * self.gru_wh[p][j];
                }
            }
            h_tilde[j] = v.tanh();
        }
        // output
        (0..ld)
            .map(|j| (1.0 - z[j]) * h[j] + z[j] * h_tilde[j])
            .collect()
    }

    /// Encode: run GRU on features → posterior params.
    fn encode(&self, x: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), GnoError> {
        let n = x.len();
        let ld = self.cfg.latent_dim;
        let mut h_gru = vec![vec![0.0f64; ld]; n];
        // Aggregate features for context
        let mp = GnoMessagePassing::new();
        let agg = mp.aggregate(x, &self.a_hat)?;
        for i in 0..n {
            h_gru[i] = self.gru_step_vec(&agg[i], &h_gru[i]);
        }
        // Posterior params
        let enc = matmul(&h_gru, &self.enc_out)?;
        let mut mu = vec![vec![0.0f64; ld]; n];
        let mut log_var = vec![vec![0.0f64; ld]; n];
        for i in 0..n {
            for j in 0..ld {
                mu[i][j] = enc[i][j];
                log_var[i][j] = enc[i][j + ld];
            }
        }
        Ok((mu, log_var))
    }

    /// Reparameterization trick: z = μ + ε·exp(0.5·log_σ²).
    fn reparameterize(
        &self,
        mu: &[Vec<f64>],
        log_var: &[Vec<f64>],
        rng: &mut StdRng,
    ) -> Vec<Vec<f64>> {
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

    /// ODE dynamics for latent z.
    fn ode_dynamics(&self, z: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GnoError> {
        let mp = GnoMessagePassing::new();
        let agg = mp.aggregate(z, &self.a_hat)?;
        let mut out = matmul(&agg, &self.ode_w)?;
        tanh_matrix(&mut out);
        Ok(out)
    }

    fn rk4_step(&self, z: &[Vec<f64>], dt: f64) -> Result<Vec<Vec<f64>>, GnoError> {
        let k1 = self.ode_dynamics(z)?;
        let z2 = add_scaled(z, &k1, dt / 2.0);
        let k2 = self.ode_dynamics(&z2)?;
        let z3 = add_scaled(z, &k2, dt / 2.0);
        let k3 = self.ode_dynamics(&z3)?;
        let z4 = add_scaled(z, &k3, dt);
        let k4 = self.ode_dynamics(&z4)?;
        let n = z.len();
        let d = if n > 0 { z[0].len() } else { 0 };
        let mut z_next = vec![vec![0.0f64; d]; n];
        for i in 0..n {
            for j in 0..d {
                z_next[i][j] =
                    z[i][j] + (dt / 6.0) * (k1[i][j] + 2.0 * k2[i][j] + 2.0 * k3[i][j] + k4[i][j]);
            }
        }
        Ok(z_next)
    }

    /// Forward: encode → reparameterize → ODE → decode. Returns logits + KL divergence.
    pub fn forward(&self, x: &[Vec<f64>], seed: u64) -> Result<(Vec<Vec<f64>>, f64), GnoError> {
        let (mu, log_var) = self.encode(x)?;
        let mut rng = StdRng::seed_from_u64(seed);
        let z0 = self.reparameterize(&mu, &log_var, &mut rng);
        // KL divergence: 0.5 * sum(exp(log_var) + mu² - 1 - log_var)
        let n = mu.len();
        let ld = self.cfg.latent_dim;
        let mut kl = 0.0f64;
        for i in 0..n {
            for j in 0..ld {
                kl += 0.5 * (log_var[i][j].exp() + mu[i][j].powi(2) - 1.0 - log_var[i][j]);
            }
        }
        // Solve ODE
        let mut z = z0;
        let dt = self.cfg.dt;
        let t_end = self.cfg.t_end;
        let mut t = 0.0f64;
        while t + dt <= t_end + 1e-12 {
            z = self.rk4_step(&z, dt)?;
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

    /// Predict class labels (deterministic, seed=0).
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<usize>, GnoError> {
        let (logits, _) = self.forward(x, 42)?;
        logits
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .ok_or(GnoError::EmptyGraph)
            })
            .collect()
    }
}
