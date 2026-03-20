//! Graph Generative Models and Molecular Machine Learning — Track D.
//!
//! Provides graph representation for molecules and graph-based generative
//! models:
//!
//! - [`MolecularGraph`]: Graph representation with atom/bond typing.
//! - \[`VariationalGraphAutoencoder`\] (VGAE): Kipf & Welling 2016 — GCN encoder
//!   with inner-product decoder.
//! - [`GraphRnn`]: Sequential node-by-node graph generation (You et al. 2018).
//! - [`MolecularFingerprint`]: Morgan / atom-pair / topological-torsion descriptors.
//! - [`MpnnLayer`]: Message Passing Neural Network (Gilmer et al. 2017).
//! - [`GraphPropertyPredictor`]: MPNN + readout → molecular property regression.
//!
//! All computation uses `Vec<f32>` buffers; no external tensor type is required.
//! All fallible operations return `Result<_, GraphGenError>`.
//! No `unsafe` code; no `unwrap()` calls.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors arising from graph-generation operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphGenError {
    /// Two buffers have incompatible lengths.
    DimensionMismatch {
        expected: usize,
        found: usize,
        context: &'static str,
    },
    /// A node index is out of range.
    NodeIndexOutOfRange { index: usize, num_nodes: usize },
    /// A requested fingerprint bit-width is zero or otherwise invalid.
    InvalidParameter { name: &'static str, message: String },
    /// The graph has no nodes.
    EmptyGraph,
    /// Weight buffer is the wrong size for the declared dimensions.
    WeightSizeMismatch {
        expected: usize,
        found: usize,
        layer: &'static str,
    },
}

impl std::fmt::Display for GraphGenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphGenError::DimensionMismatch {
                expected,
                found,
                context,
            } => {
                write!(
                    f,
                    "dimension mismatch in {context}: expected {expected}, found {found}"
                )
            }
            GraphGenError::NodeIndexOutOfRange { index, num_nodes } => {
                write!(
                    f,
                    "node index {index} out of range for graph with {num_nodes} nodes"
                )
            }
            GraphGenError::InvalidParameter { name, message } => {
                write!(f, "invalid parameter '{name}': {message}")
            }
            GraphGenError::EmptyGraph => write!(f, "graph has no nodes"),
            GraphGenError::WeightSizeMismatch {
                expected,
                found,
                layer,
            } => {
                write!(
                    f,
                    "weight size mismatch in {layer}: expected {expected}, found {found}"
                )
            }
        }
    }
}

impl std::error::Error for GraphGenError {}

// ─────────────────────────────────────────────────────────────────────────────
// Activation helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    let xc = x.clamp(-88.0, 88.0);
    1.0 / (1.0 + (-xc).exp())
}

// ─────────────────────────────────────────────────────────────────────────────
// Normal sampling — Box-Muller, seeded
// ─────────────────────────────────────────────────────────────────────────────

/// Generate `n` i.i.d. N(0,1) samples using Box-Muller with a deterministic seed.
fn normal_samples(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    let mut i = 0_usize;
    while i < n {
        let u1: f64 = rng.random::<f64>().max(1e-10);
        let u2: f64 = rng.random::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        let z0 = (r * theta.cos()) as f32;
        let z1 = (r * theta.sin()) as f32;
        out.push(z0);
        i += 1;
        if i < n {
            out.push(z1);
            i += 1;
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Xavier weight initialisation
// ─────────────────────────────────────────────────────────────────────────────

fn xavier_uniform(fan_in: usize, fan_out: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt() as f32;
    (0..fan_in * fan_out)
        .map(|_| {
            let u: f32 = rng.random();
            u * 2.0 * limit - limit
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix helpers (row-major)
// ─────────────────────────────────────────────────────────────────────────────

/// Multiply matrix `a` [m×k] by matrix `b` [k×n] → result [m×n].
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for kk in 0..k {
            let aik = a[i * k + kk];
            for j in 0..n {
                c[i * n + j] += aik * b[kk * n + j];
            }
        }
    }
    c
}

/// Apply ReLU elementwise to every element.
fn relu_vec(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = relu(*x);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Atom / bond type enums
// ─────────────────────────────────────────────────────────────────────────────

/// Chemical element identity for a graph node (atom).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AtomType {
    C,
    N,
    O,
    F,
    P,
    S,
    Cl,
    Br,
    I,
    /// Any other element, identified by atomic number.
    Other(u8),
}

/// Chemical bond order / type for a graph edge.
#[derive(Debug, Clone, PartialEq)]
pub enum BondType {
    Single,
    Double,
    Triple,
    Aromatic,
}

impl BondType {
    /// Numerical bond order used when encoding edge features.
    pub fn order(&self) -> f32 {
        match self {
            BondType::Single => 1.0,
            BondType::Double => 2.0,
            BondType::Triple => 3.0,
            BondType::Aromatic => 1.5,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MolecularGraph
// ─────────────────────────────────────────────────────────────────────────────

/// Graph representation of a small molecule.
///
/// Atoms are nodes; bonds are directed edges stored as both (i→j) and (j→i).
/// Node feature matrix is stored row-major: `atom_features[i * atom_feature_dim ..
/// (i+1) * atom_feature_dim]` is the feature vector for atom `i`.
#[derive(Debug, Clone)]
pub struct MolecularGraph {
    /// Number of atoms (nodes).
    pub num_atoms: usize,
    /// Flat node feature matrix [num_atoms × atom_feature_dim].
    pub atom_features: Vec<f32>,
    /// Undirected edges with associated bond type.
    pub bond_types: Vec<(usize, usize, BondType)>,
    /// `adjacency_list[i]` holds the neighbour indices of atom `i`.
    pub adjacency_list: Vec<Vec<usize>>,
    /// Number of features per atom.
    pub atom_feature_dim: usize,
}

impl MolecularGraph {
    /// Create an empty graph with `num_atoms` nodes and zero-initialised features.
    pub fn new(num_atoms: usize, atom_feature_dim: usize) -> Self {
        MolecularGraph {
            num_atoms,
            atom_features: vec![0.0_f32; num_atoms * atom_feature_dim],
            bond_types: Vec::new(),
            adjacency_list: vec![Vec::new(); num_atoms],
            atom_feature_dim,
        }
    }

    /// Add a (potentially directed) undirected edge between atoms `i` and `j`.
    ///
    /// Silently ignores self-loops (`i == j`).  Duplicate edges are allowed
    /// (callers are responsible for deduplication when needed).
    pub fn add_edge(&mut self, i: usize, j: usize, bond_type: BondType) {
        if i == j || i >= self.num_atoms || j >= self.num_atoms {
            return;
        }
        self.bond_types.push((i, j, bond_type));
        self.adjacency_list[i].push(j);
        self.adjacency_list[j].push(i);
    }

    /// Return the degree (number of neighbours) of atom `i`.
    pub fn degree(&self, i: usize) -> usize {
        if i >= self.num_atoms {
            return 0;
        }
        self.adjacency_list[i].len()
    }

    /// Build the flat row-major [N×N] binary adjacency matrix.
    pub fn adjacency_matrix(&self) -> Vec<f32> {
        let n = self.num_atoms;
        let mut a = vec![0.0_f32; n * n];
        for (i, j, _) in &self.bond_types {
            a[i * n + j] = 1.0;
            a[j * n + i] = 1.0;
        }
        a
    }

    /// Compute the graph Laplacian L = D − A as a flat [N×N] matrix.
    pub fn laplacian(&self) -> Vec<f32> {
        let n = self.num_atoms;
        let a = self.adjacency_matrix();
        let mut l = vec![0.0_f32; n * n];
        for i in 0..n {
            let deg = self.degree(i) as f32;
            l[i * n + i] = deg;
            for j in 0..n {
                l[i * n + j] -= a[i * n + j];
            }
        }
        l
    }

    /// Check whether the graph is plausible: no self-loops in the adjacency
    /// list, all node indices in bounds, max degree ≤ 8 (heuristic for
    /// drug-like molecules).
    pub fn is_valid(&self) -> bool {
        if self.num_atoms == 0 {
            return false;
        }
        for (i, neighbours) in self.adjacency_list.iter().enumerate() {
            for &j in neighbours {
                if i == j || j >= self.num_atoms {
                    return false;
                }
            }
            if neighbours.len() > 8 {
                return false;
            }
        }
        for (i, j, _) in &self.bond_types {
            if i == j || *i >= self.num_atoms || *j >= self.num_atoms {
                return false;
            }
        }
        true
    }

    /// One-hot encode `atom_type` against `allowed_types`.
    ///
    /// Returns a vector of length `allowed_types.len() + 1`; the last position
    /// is the "other/unknown" bucket used when `atom_type` is not in the list.
    pub fn encode_atom_one_hot(atom_type: &AtomType, allowed_types: &[AtomType]) -> Vec<f32> {
        let dim = allowed_types.len() + 1;
        let mut v = vec![0.0_f32; dim];
        for (k, at) in allowed_types.iter().enumerate() {
            if at == atom_type {
                v[k] = 1.0;
                return v;
            }
        }
        // Not found → "other" bucket at the last position.
        v[dim - 1] = 1.0;
        v
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Variational Graph Autoencoder (VGAE)
// ─────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters for the VGAE encoder.
#[derive(Debug, Clone)]
pub struct VgaeConfig {
    /// Dimensionality of the input node features.
    pub input_features: usize,
    /// Hidden GCN layer width.
    pub hidden_dim: usize,
    /// Latent space dimensionality (per node).
    pub latent_dim: usize,
    /// Number of GCN layers before the final μ / log σ² projections.
    pub num_gcn_layers: usize,
}

/// GCN-based encoder for VGAE.
///
/// Applies `num_gcn_layers` symmetric-normalised GCN updates, then projects
/// to (μ, log σ²) in latent space.
pub struct VgaeEncoder {
    pub config: VgaeConfig,
    /// GCN weight matrices.  Layer 0 is [hidden_dim × input_features];
    /// subsequent layers are [hidden_dim × hidden_dim].
    pub gcn_weights: Vec<Vec<f32>>,
    /// μ projection: [latent_dim × hidden_dim].
    pub mu_w: Vec<f32>,
    /// log σ² projection: [latent_dim × hidden_dim].
    pub log_var_w: Vec<f32>,
}

impl VgaeEncoder {
    /// Create a VGAE encoder with Xavier-initialised weights.
    pub fn new(config: VgaeConfig) -> Self {
        let VgaeConfig {
            input_features,
            hidden_dim,
            latent_dim,
            num_gcn_layers,
        } = config.clone();

        let mut gcn_weights = Vec::with_capacity(num_gcn_layers);
        for layer_idx in 0..num_gcn_layers {
            let fan_in = if layer_idx == 0 {
                input_features
            } else {
                hidden_dim
            };
            gcn_weights.push(xavier_uniform(fan_in, hidden_dim, 1000 + layer_idx as u64));
        }
        let mu_w = xavier_uniform(hidden_dim, latent_dim, 2000);
        let log_var_w = xavier_uniform(hidden_dim, latent_dim, 3000);

        VgaeEncoder {
            config,
            gcn_weights,
            mu_w,
            log_var_w,
        }
    }

    /// Symmetrically normalised adjacency: Â = D̃^{-1/2} Ã D̃^{-1/2}
    /// where Ã = A + I (self-loops added).
    fn normalised_adjacency(adj: &[f32], n: usize) -> Vec<f32> {
        // Compute degree with self-loops
        let mut deg = vec![0.0_f32; n];
        for i in 0..n {
            let mut d = 1.0_f32; // self-loop
            for j in 0..n {
                d += adj[i * n + j];
            }
            deg[i] = d;
        }
        let mut a_hat = vec![0.0_f32; n * n];
        for i in 0..n {
            let di = deg[i].sqrt().max(1e-8);
            // Self-loop contribution
            a_hat[i * n + i] = 1.0 / (di * di);
            // Edge contributions
            for j in 0..n {
                if adj[i * n + j] > 0.5 {
                    let dj = deg[j].sqrt().max(1e-8);
                    a_hat[i * n + j] = 1.0 / (di * dj);
                }
            }
        }
        a_hat
    }

    /// Run the GCN encoder.
    ///
    /// # Arguments
    /// - `x`: node features [num_nodes × input_features]
    /// - `adj`: binary adjacency matrix [num_nodes × num_nodes]
    /// - `num_nodes`: N
    ///
    /// # Returns
    /// `(mu, log_var)` each of shape [num_nodes × latent_dim].
    pub fn encode(
        &self,
        x: &[f32],
        adj: &[f32],
        num_nodes: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), GraphGenError> {
        let cfg = &self.config;
        if x.len() != num_nodes * cfg.input_features {
            return Err(GraphGenError::DimensionMismatch {
                expected: num_nodes * cfg.input_features,
                found: x.len(),
                context: "VgaeEncoder::encode x",
            });
        }
        if adj.len() != num_nodes * num_nodes {
            return Err(GraphGenError::DimensionMismatch {
                expected: num_nodes * num_nodes,
                found: adj.len(),
                context: "VgaeEncoder::encode adj",
            });
        }

        let a_hat = Self::normalised_adjacency(adj, num_nodes);
        // h: [num_nodes × in_dim] initially
        let mut h = x.to_vec();
        let mut current_in = cfg.input_features;

        for (layer_idx, w) in self.gcn_weights.iter().enumerate() {
            // Verify weight size
            let expected_w = current_in * cfg.hidden_dim;
            if w.len() != expected_w {
                return Err(GraphGenError::WeightSizeMismatch {
                    expected: expected_w,
                    found: w.len(),
                    layer: "VgaeEncoder GCN",
                });
            }
            // h_new = relu(A_hat H W)  — compute A_hat H first [N × current_in]
            // then multiply by W [current_in × hidden_dim]
            let ah = matmul(&a_hat, &h, num_nodes, num_nodes, current_in);
            let mut h_new = matmul(&ah, w, num_nodes, current_in, cfg.hidden_dim);
            // Apply ReLU except on the last GCN layer (before mu/log_var heads)
            if layer_idx < self.gcn_weights.len() - 1 {
                relu_vec(&mut h_new);
            }
            h = h_new;
            current_in = cfg.hidden_dim;
        }

        // mu = H W_mu  [N × latent_dim]
        let mu = matmul(&h, &self.mu_w, num_nodes, cfg.hidden_dim, cfg.latent_dim);
        // log_var = H W_log_var  [N × latent_dim]
        let log_var = matmul(
            &h,
            &self.log_var_w,
            num_nodes,
            cfg.hidden_dim,
            cfg.latent_dim,
        );

        Ok((mu, log_var))
    }

    /// Reparameterisation: z_i = μ_i + ε_i · exp(log_var_i / 2), ε ∼ N(0,I).
    pub fn reparameterize(
        mu: &[f32],
        log_var: &[f32],
        seed: u64,
    ) -> Result<Vec<f32>, GraphGenError> {
        let n = mu.len();
        if log_var.len() != n {
            return Err(GraphGenError::DimensionMismatch {
                expected: n,
                found: log_var.len(),
                context: "reparameterize log_var",
            });
        }
        let eps = normal_samples(n, seed);
        let z = mu
            .iter()
            .zip(log_var.iter())
            .zip(eps.iter())
            .map(|((&m, &lv), &e)| m + e * (lv * 0.5).exp())
            .collect();
        Ok(z)
    }

    /// Inner-product decoder: Â_{ij} = sigmoid(z_i · z_j).
    ///
    /// # Returns
    /// Flat [N×N] matrix of edge probabilities.
    pub fn decode(
        z: &[f32],
        num_nodes: usize,
        latent_dim: usize,
    ) -> Result<Vec<f32>, GraphGenError> {
        if z.len() != num_nodes * latent_dim {
            return Err(GraphGenError::DimensionMismatch {
                expected: num_nodes * latent_dim,
                found: z.len(),
                context: "decode z",
            });
        }
        // Compute Z Z^T [N × N] explicitly:
        // (Z Z^T)[i,j] = Σ_k z[i,k] * z[j,k]
        let mut zzt = vec![0.0_f32; num_nodes * num_nodes];
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                let mut dot = 0.0_f32;
                for k in 0..latent_dim {
                    dot += z[i * latent_dim + k] * z[j * latent_dim + k];
                }
                zzt[i * num_nodes + j] = dot;
            }
        }
        // Apply sigmoid elementwise
        Ok(zzt.into_iter().map(sigmoid).collect())
    }

    /// Compute VGAE loss = (BCE, KL) pair.
    ///
    /// - BCE is computed with positive-weight correction for sparse graphs:
    ///   `pos_weight = (N²−|E|) / |E|` (clipped to [1, 10]).
    /// - KL = 0.5 · Σ_i (1 + log_var_i − μ_i² − exp(log_var_i)).
    pub fn loss(
        adj_true: &[f32],
        adj_pred: &[f32],
        mu: &[f32],
        log_var: &[f32],
    ) -> Result<(f32, f32), GraphGenError> {
        let n = adj_true.len();
        if adj_pred.len() != n {
            return Err(GraphGenError::DimensionMismatch {
                expected: n,
                found: adj_pred.len(),
                context: "loss adj_pred",
            });
        }

        // Positive-weight: ratio of negatives to positives for class balance.
        let num_pos: f32 = adj_true.iter().sum();
        let num_neg = n as f32 - num_pos;
        let pos_weight = if num_pos > 0.5 {
            (num_neg / num_pos).clamp(1.0, 10.0)
        } else {
            1.0
        };

        let mut bce = 0.0_f32;
        for (&a, &p) in adj_true.iter().zip(adj_pred.iter()) {
            let p_safe = p.clamp(1e-7, 1.0 - 1e-7);
            bce -= a * pos_weight * p_safe.ln() + (1.0 - a) * (1.0 - p_safe).ln();
        }
        bce /= n as f32;

        // KL divergence
        if mu.len() != log_var.len() {
            return Err(GraphGenError::DimensionMismatch {
                expected: mu.len(),
                found: log_var.len(),
                context: "loss mu/log_var lengths",
            });
        }
        let kl: f32 = -mu
            .iter()
            .zip(log_var.iter())
            .map(|(&m, &lv)| 0.5 * (1.0 + lv - m * m - lv.exp()))
            .sum::<f32>(); // KL is negative ELBO term → negate
        let kl = kl / mu.len() as f32;

        Ok((bce, kl))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GraphRNN
// ─────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters for GraphRNN sequential generation.
#[derive(Debug, Clone)]
pub struct GraphRnnConfig {
    /// Maximum number of nodes to generate.
    pub max_nodes: usize,
    /// Dimensionality of node feature vectors.
    pub node_feature_dim: usize,
    /// Hidden-state dimensionality for the node-level GRU.
    pub hidden_dim: usize,
    /// Hidden-state dimensionality for the edge-level GRU.
    pub edge_hidden_dim: usize,
}

/// Sequential graph generator — You et al. (2018).
///
/// Generates a graph node-by-node.  For each new node, an edge-level GRU
/// autoregressively decides whether to connect the new node to each previous
/// node.
///
/// Weight layout for `node_gru_w`:
///   GRU has 3 gates (r, u, n); each gate has `[hidden_dim × (input + hidden)]`.
///   Total = 3 × hidden_dim × (node_feature_dim + hidden_dim).
///
/// Weight layout for `edge_gru_w`:
///   Same pattern with `edge_hidden_dim` and input = 1 (binary edge signal).
pub struct GraphRnn {
    pub config: GraphRnnConfig,
    /// Node-level GRU weights [3 × hidden_dim × (node_feature_dim + hidden_dim)].
    pub node_gru_w: Vec<f32>,
    /// Edge-level GRU weights [3 × edge_hidden_dim × (1 + edge_hidden_dim)].
    pub edge_gru_w: Vec<f32>,
    /// Edge MLP first layer: [edge_hidden_dim × edge_hidden_dim].
    pub edge_mlp_w: Vec<f32>,
    /// Edge MLP first-layer bias: \[edge_hidden_dim\].
    pub edge_mlp_b: Vec<f32>,
    /// Node feature MLP: [node_feature_dim × hidden_dim].
    pub node_mlp_w: Vec<f32>,
    /// Node feature MLP bias: \[node_feature_dim\].
    pub node_mlp_b: Vec<f32>,
}

impl GraphRnn {
    /// Create a new GraphRNN with Xavier-initialised weights.
    pub fn new(config: GraphRnnConfig) -> Self {
        let h = config.hidden_dim;
        let eh = config.edge_hidden_dim;
        let nf = config.node_feature_dim;

        // Node GRU: 3 gates × hidden_dim × (nf + h)
        let node_gru_w = xavier_uniform(3 * (nf + h), h, 100);
        // Edge GRU: 3 gates × edge_hidden_dim × (1 + eh)
        let edge_gru_w = xavier_uniform(3 * (1 + eh), eh, 200);
        // Edge MLP
        let edge_mlp_w = xavier_uniform(eh, eh, 300);
        let edge_mlp_b = vec![0.0_f32; eh];
        // Node MLP
        let node_mlp_w = xavier_uniform(h, nf, 400);
        let node_mlp_b = vec![0.0_f32; nf];

        GraphRnn {
            config,
            node_gru_w,
            edge_gru_w,
            edge_mlp_w,
            edge_mlp_b,
            node_mlp_w,
            node_mlp_b,
        }
    }

    /// GRU step: h' = GRU(h_prev, x) using the provided weight matrix.
    ///
    /// Weight matrix `w` is stored as three vertically concatenated blocks
    /// (reset, update, new-gate), each of shape [hidden_dim × (input + hidden)].
    ///
    /// `input_dim` = `x.len()`, `hidden_dim` = `h_prev.len()`.
    fn gru_step(h_prev: &[f32], x: &[f32], w: &[f32], hidden_dim: usize) -> Vec<f32> {
        let input_dim = x.len();
        let concat_dim = input_dim + hidden_dim;
        // Concatenate input and previous hidden state
        let mut xh = Vec::with_capacity(concat_dim);
        xh.extend_from_slice(x);
        xh.extend_from_slice(h_prev);

        // Each gate block has shape [hidden_dim × concat_dim]
        let block = hidden_dim * concat_dim;

        let mut r = vec![0.0_f32; hidden_dim];
        let mut u = vec![0.0_f32; hidden_dim];
        let mut n = vec![0.0_f32; hidden_dim];

        // Reset gate
        for i in 0..hidden_dim {
            let mut val = 0.0_f32;
            for k in 0..concat_dim {
                val += xh[k] * w[i * concat_dim + k];
            }
            r[i] = sigmoid(val);
        }
        // Update gate
        for i in 0..hidden_dim {
            let mut val = 0.0_f32;
            for k in 0..concat_dim {
                val += xh[k] * w[block + i * concat_dim + k];
            }
            u[i] = sigmoid(val);
        }
        // New gate  (uses r ⊙ h_prev in place of plain h_prev)
        // Concatenate x with r ⊙ h_prev
        let mut xrh = Vec::with_capacity(concat_dim);
        xrh.extend_from_slice(x);
        for (ri, hi) in r.iter().zip(h_prev.iter()) {
            xrh.push(ri * hi);
        }
        for i in 0..hidden_dim {
            let mut val = 0.0_f32;
            for k in 0..concat_dim {
                val += xrh[k] * w[2 * block + i * concat_dim + k];
            }
            n[i] = val.tanh();
        }
        // h' = (1 − u) ⊙ n + u ⊙ h_prev
        (0..hidden_dim)
            .map(|i| (1.0 - u[i]) * n[i] + u[i] * h_prev[i])
            .collect()
    }

    /// Node-level GRU update.
    pub fn node_gru_step(&self, h_prev: &[f32], x: &[f32]) -> Vec<f32> {
        Self::gru_step(h_prev, x, &self.node_gru_w, self.config.hidden_dim)
    }

    /// Edge-level GRU update (input is a scalar stored as a 1-element slice).
    fn edge_gru_step(&self, h_prev: &[f32], x: &[f32]) -> Vec<f32> {
        Self::gru_step(h_prev, x, &self.edge_gru_w, self.config.edge_hidden_dim)
    }

    /// Convert edge hidden state to Bernoulli probability via a two-layer MLP.
    pub fn edge_probability(&self, edge_h: &[f32]) -> f32 {
        let eh = self.config.edge_hidden_dim;
        // Layer 1: edge_h → relu → intermediate [eh]
        let mut mid = vec![0.0_f32; eh];
        for j in 0..eh {
            let mut val = self.edge_mlp_b[j];
            for k in 0..eh {
                // w layout: [out_j × in_k] row-major
                val += edge_h[k] * self.edge_mlp_w[j * eh + k];
            }
            mid[j] = relu(val);
        }
        // Layer 2: sum → single logit (simple dot product with all-ones output)
        let logit: f32 = mid.iter().sum::<f32>() / eh as f32;
        sigmoid(logit)
    }

    /// Project node hidden state to node feature vector.
    fn node_features(&self, h: &[f32]) -> Vec<f32> {
        let nf = self.config.node_feature_dim;
        let hd = self.config.hidden_dim;
        let mut out = self.node_mlp_b.clone();
        for j in 0..nf {
            for k in 0..hd {
                out[j] += h[k] * self.node_mlp_w[j * hd + k];
            }
        }
        // Apply sigmoid so features are in [0,1]
        out.iter().map(|&v| sigmoid(v)).collect()
    }

    /// Sample a graph with `max_nodes` nodes.
    ///
    /// The generation process:
    /// 1. Start with an empty graph.
    /// 2. For each new node `t`, run the node GRU to produce an embedding.
    /// 3. Run the edge GRU over all previous nodes to decide connectivity.
    /// 4. Stop early if no edges are added (generation complete).
    pub fn sample_graph(
        &self,
        max_nodes: usize,
        rng: &mut StdRng,
    ) -> Result<MolecularGraph, GraphGenError> {
        if max_nodes == 0 {
            return Err(GraphGenError::InvalidParameter {
                name: "max_nodes",
                message: "must be >= 1".to_string(),
            });
        }
        let h_dim = self.config.hidden_dim;
        let eh_dim = self.config.edge_hidden_dim;
        let nf_dim = self.config.node_feature_dim;

        // We don't know final size ahead of time; build incrementally.
        let mut graph = MolecularGraph::new(max_nodes, nf_dim);
        let mut node_count = 0_usize;

        // Node-level hidden state
        let mut node_h = vec![0.0_f32; h_dim];
        // Node embeddings accumulated so far
        let mut node_embeddings: Vec<Vec<f32>> = Vec::new();

        for _t in 0..max_nodes {
            // Input to node GRU: zero vector (could be last edge-level summary)
            let node_input = vec![0.0_f32; nf_dim];
            node_h = self.node_gru_step(&node_h, &node_input);

            // Generate features for this node
            let features = self.node_features(&node_h);
            let base = node_count * nf_dim;
            for (k, &fv) in features.iter().enumerate() {
                graph.atom_features[base + k] = fv;
            }
            node_embeddings.push(node_h.clone());
            node_count += 1;

            if node_count == 1 {
                // No previous nodes to connect to
                continue;
            }

            // Edge-level GRU over previous nodes in reverse order
            let mut edge_h = vec![0.0_f32; eh_dim];
            let mut added_edge = false;
            for prev in (0..node_count - 1).rev() {
                // Input: concatenation-summary derived from node embedding dot product
                let dot: f32 = node_h
                    .iter()
                    .zip(node_embeddings[prev].iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>()
                    / h_dim as f32;
                let x_edge = vec![dot.tanh()];
                edge_h = self.edge_gru_step(&edge_h, &x_edge);

                let prob = self.edge_probability(&edge_h);
                let sample: f32 = rng.random();
                if sample < prob {
                    graph.add_edge(node_count - 1, prev, BondType::Single);
                    added_edge = true;
                }
            }
            // If no edges were generated for this node and graph is non-trivial, stop.
            if !added_edge && node_count > 2 {
                break;
            }
        }

        // Trim graph to actual node_count
        graph.num_atoms = node_count;
        graph.atom_features.truncate(node_count * nf_dim);
        graph.adjacency_list.truncate(node_count);

        Ok(graph)
    }

    /// Public convenience: seed-based graph generation.
    pub fn generate(&self, seed: u64, num_nodes: usize) -> Result<MolecularGraph, GraphGenError> {
        let mut rng = StdRng::seed_from_u64(seed);
        self.sample_graph(num_nodes, &mut rng)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Molecular fingerprints
// ─────────────────────────────────────────────────────────────────────────────

/// Computes molecular fingerprint representations for graph-level tasks.
pub struct MolecularFingerprint;

impl MolecularFingerprint {
    /// Simplified Morgan / ECFP-style circular fingerprint.
    ///
    /// Algorithm:
    /// 1. Initialise each atom identifier as a hash of its feature index.
    /// 2. For `radius` rounds, hash each atom's ID with sorted neighbour IDs.
    /// 3. Fold all collected identifiers into a `bits`-length binary vector.
    pub fn morgan_fingerprint(
        graph: &MolecularGraph,
        radius: usize,
        bits: usize,
    ) -> Result<Vec<f32>, GraphGenError> {
        if bits == 0 {
            return Err(GraphGenError::InvalidParameter {
                name: "bits",
                message: "must be >= 1".to_string(),
            });
        }
        if graph.num_atoms == 0 {
            return Err(GraphGenError::EmptyGraph);
        }

        let n = graph.num_atoms;
        let fd = graph.atom_feature_dim;

        // Initialise identifiers: hash of the atom's features
        let mut ids: Vec<u64> = (0..n)
            .map(|i| {
                let feats = &graph.atom_features[i * fd..(i + 1) * fd];
                Self::hash_features(feats)
            })
            .collect();

        let mut all_ids: Vec<u64> = ids.clone();

        for _ in 0..radius {
            let mut new_ids = ids.clone();
            for i in 0..n {
                let mut nbr_ids: Vec<u64> =
                    graph.adjacency_list[i].iter().map(|&j| ids[j]).collect();
                nbr_ids.sort_unstable();
                // Mix: start with atom id, fold in neighbours
                let mut h = ids[i];
                for &nid in &nbr_ids {
                    h = Self::mix64(h, nid);
                }
                new_ids[i] = h;
                all_ids.push(h);
            }
            ids = new_ids;
        }

        // Fold into bits-length binary vector
        let mut fp = vec![0.0_f32; bits];
        for id in all_ids {
            let pos = (id as usize) % bits;
            fp[pos] = 1.0;
        }
        Ok(fp)
    }

    /// Atom-pair fingerprint: for each pair (i, j), record atom types and
    /// shortest-path distance (approximated as topological distance from
    /// adjacency list BFS).
    pub fn atom_pair_fingerprint(graph: &MolecularGraph) -> Vec<f32> {
        let n = graph.num_atoms;
        if n == 0 {
            return Vec::new();
        }
        // BFS all-pairs shortest paths (capped at n)
        let dists = Self::all_pairs_bfs(graph);

        let fd = graph.atom_feature_dim;
        // For each pair encode: feat_i XOR-hash, feat_j XOR-hash, distance
        // Pack into a fixed-size vector of length n*(n-1)/2 * 3
        let num_pairs = n * (n - 1) / 2;
        let mut fp = Vec::with_capacity(num_pairs * 3);
        for i in 0..n {
            for j in (i + 1)..n {
                let hi = Self::hash_features(&graph.atom_features[i * fd..(i + 1) * fd]);
                let hj = Self::hash_features(&graph.atom_features[j * fd..(j + 1) * fd]);
                let dist = dists[i * n + j];
                // Normalise to [0,1]
                fp.push((hi % 256) as f32 / 255.0);
                fp.push((hj % 256) as f32 / 255.0);
                fp.push((dist as f32) / (n as f32).max(1.0));
            }
        }
        fp
    }

    /// Topological-torsion fingerprint: enumerate all paths of length 4 in the
    /// graph and hash each path's atom sequence.
    /// Returns a 256-bit binary vector.
    pub fn topological_torsion(graph: &MolecularGraph) -> Vec<f32> {
        const BITS: usize = 256;
        let mut fp = vec![0.0_f32; BITS];
        let n = graph.num_atoms;
        if n < 4 {
            return fp;
        }
        let fd = graph.atom_feature_dim;
        // Enumerate paths of exactly 4 nodes with no revisiting
        for a in 0..n {
            for &b in &graph.adjacency_list[a] {
                if b == a {
                    continue;
                }
                for &c in &graph.adjacency_list[b] {
                    if c == a || c == b {
                        continue;
                    }
                    for &d in &graph.adjacency_list[c] {
                        if d == a || d == b || d == c {
                            continue;
                        }
                        // Hash the 4-atom path
                        let ha = Self::hash_features(&graph.atom_features[a * fd..(a + 1) * fd]);
                        let hb = Self::hash_features(&graph.atom_features[b * fd..(b + 1) * fd]);
                        let hc = Self::hash_features(&graph.atom_features[c * fd..(c + 1) * fd]);
                        let hd = Self::hash_features(&graph.atom_features[d * fd..(d + 1) * fd]);
                        let h = Self::mix64(Self::mix64(Self::mix64(ha, hb), hc), hd);
                        fp[(h as usize) % BITS] = 1.0;
                    }
                }
            }
        }
        fp
    }

    // ---- internal helpers ----

    /// Hash a feature slice to a u64 using FNV-1a inspired mixing.
    fn hash_features(feats: &[f32]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
        for &v in feats {
            let bits = v.to_bits() as u64;
            h ^= bits;
            h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV prime
        }
        h
    }

    /// Avalanche-style 64-bit hash mixing.
    fn mix64(a: u64, b: u64) -> u64 {
        let mut h = a.wrapping_add(b).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        h ^= h >> 30;
        h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h ^= h >> 27;
        h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
        h ^= h >> 31;
        h
    }

    /// BFS-based all-pairs shortest-path distances.  Returns flat [N×N].
    fn all_pairs_bfs(graph: &MolecularGraph) -> Vec<usize> {
        let n = graph.num_atoms;
        let mut dists = vec![n + 1; n * n]; // unreachable = n+1
        for s in 0..n {
            dists[s * n + s] = 0;
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(s);
            while let Some(u) = queue.pop_front() {
                for &v in &graph.adjacency_list[u] {
                    if dists[s * n + v] > n {
                        dists[s * n + v] = dists[s * n + u] + 1;
                        queue.push_back(v);
                    }
                }
            }
        }
        dists
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Message Passing Neural Network (MPNN)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for an MPNN layer.
#[derive(Debug, Clone)]
pub struct MpnnConfig {
    /// Dimensionality of node features.
    pub node_features: usize,
    /// Dimensionality of edge feature vectors.
    pub edge_features: usize,
    /// Hidden / output dimensionality.
    pub hidden_dim: usize,
    /// Number of message-passing rounds.
    pub num_message_steps: usize,
}

/// Single MPNN layer (Gilmer et al., 2017).
///
/// Message function `M(h_v, h_w, e_vw)` is a two-layer MLP:
///   input = [h_v ‖ h_w ‖ e_vw]  → W1 → ReLU → W2 → message.
///
/// Update function `U(h_v, Σm)` is a GRU step.
pub struct MpnnLayer {
    pub config: MpnnConfig,
    /// First message MLP weight: [(node_features+edge_features) × hidden_dim].
    ///
    /// Input concatenates (h_v + h_w) × node_features/2 each + e_vw.
    /// Full input dim = node_features + node_features + edge_features.
    pub msg_w1: Vec<f32>,
    /// Second message MLP weight: [hidden_dim × hidden_dim].
    pub msg_w2: Vec<f32>,
    /// GRU update weights (3 gates, each [hidden_dim × 2*hidden_dim]).
    pub gru_w: Vec<f32>,
}

impl MpnnLayer {
    /// Create an MPNN layer with Xavier-initialised weights.
    pub fn new(config: MpnnConfig) -> Self {
        let nf = config.node_features;
        let ef = config.edge_features;
        let hd = config.hidden_dim;

        // Message MLP layer 1: input dim = nf + nf + ef = 2*nf + ef
        // (concatenation of both endpoint features and edge features)
        let msg_in_dim = nf + nf + ef;
        let msg_w1 = xavier_uniform(msg_in_dim, hd, 500);
        let msg_w2 = xavier_uniform(hd, hd, 600);

        // GRU: 3 gates × [hd × (hd + hd)] = 3 × hd × 2*hd
        let gru_w = xavier_uniform(3 * 2 * hd, hd, 700);

        MpnnLayer {
            config,
            msg_w1,
            msg_w2,
            gru_w,
        }
    }

    /// Compute message m_{vw} = MLP([h_v ‖ h_w ‖ e_vw]).
    pub fn message(
        &self,
        h_v: &[f32],
        h_w: &[f32],
        e_vw: &[f32],
    ) -> Result<Vec<f32>, GraphGenError> {
        let nf = self.config.node_features;
        let ef = self.config.edge_features;
        let hd = self.config.hidden_dim;

        if h_v.len() != nf {
            return Err(GraphGenError::DimensionMismatch {
                expected: nf,
                found: h_v.len(),
                context: "message h_v",
            });
        }
        if h_w.len() != nf {
            return Err(GraphGenError::DimensionMismatch {
                expected: nf,
                found: h_w.len(),
                context: "message h_w",
            });
        }
        if e_vw.len() != ef {
            return Err(GraphGenError::DimensionMismatch {
                expected: ef,
                found: e_vw.len(),
                context: "message e_vw",
            });
        }

        let msg_in_dim = nf + nf + ef;
        // Build concatenated input
        let mut inp = Vec::with_capacity(msg_in_dim);
        inp.extend_from_slice(h_v);
        inp.extend_from_slice(h_w);
        inp.extend_from_slice(e_vw);

        // Layer 1
        let mut mid = vec![0.0_f32; hd];
        for j in 0..hd {
            for k in 0..msg_in_dim {
                mid[j] += inp[k] * self.msg_w1[k * hd + j];
            }
            mid[j] = relu(mid[j]);
        }
        // Layer 2
        let mut out = vec![0.0_f32; hd];
        for j in 0..hd {
            for k in 0..hd {
                out[j] += mid[k] * self.msg_w2[k * hd + j];
            }
        }
        Ok(out)
    }

    /// Sum-aggregate a list of message vectors → one vector of size `hidden_dim`.
    pub fn aggregate(&self, messages: &[Vec<f32>]) -> Vec<f32> {
        let hd = self.config.hidden_dim;
        let mut agg = vec![0.0_f32; hd];
        for msg in messages {
            for (a, &m) in agg.iter_mut().zip(msg.iter()) {
                *a += m;
            }
        }
        agg
    }

    /// GRU update: h' = GRU(h, aggregated_messages).
    pub fn update_gru(&self, h: &[f32], aggregated: &[f32]) -> Result<Vec<f32>, GraphGenError> {
        let hd = self.config.hidden_dim;
        if h.len() != hd {
            return Err(GraphGenError::DimensionMismatch {
                expected: hd,
                found: h.len(),
                context: "update_gru h",
            });
        }
        if aggregated.len() != hd {
            return Err(GraphGenError::DimensionMismatch {
                expected: hd,
                found: aggregated.len(),
                context: "update_gru aggregated",
            });
        }
        // GRU step: treat aggregated as "input" and h as previous hidden state
        Ok(GraphRnn::gru_step(h, aggregated, &self.gru_w, hd))
    }

    /// Run `num_message_steps` rounds of message passing.
    ///
    /// `adj_with_edge_features`: list of (node_i, node_j, edge_features).
    /// `h`: initial node features [num_nodes × node_features].
    ///
    /// Returns updated node features [num_nodes × hidden_dim].
    pub fn forward(
        &self,
        h: &[f32],
        adj_with_edge_features: &[(usize, usize, Vec<f32>)],
        num_nodes: usize,
    ) -> Result<Vec<f32>, GraphGenError> {
        let nf = self.config.node_features;
        let hd = self.config.hidden_dim;

        if h.len() != num_nodes * nf {
            return Err(GraphGenError::DimensionMismatch {
                expected: num_nodes * nf,
                found: h.len(),
                context: "MpnnLayer::forward h",
            });
        }
        if num_nodes == 0 {
            return Err(GraphGenError::EmptyGraph);
        }

        // Project input node features to hidden_dim via message network
        // (on the first pass use zero hidden states)
        let ef = self.config.edge_features;

        // Pad or project input features if nf != hd
        // Use zero-padded / truncated copies for the initial hidden state
        let mut node_h: Vec<Vec<f32>> = (0..num_nodes)
            .map(|i| {
                let feat = &h[i * nf..(i + 1) * nf];
                // Project: if nf >= hd, take first hd; else zero-pad
                let mut hv = vec![0.0_f32; hd];
                let copy_len = feat.len().min(hd);
                hv[..copy_len].copy_from_slice(&feat[..copy_len]);
                hv
            })
            .collect();

        for _step in 0..self.config.num_message_steps {
            let mut new_h = node_h.clone();
            // Collect messages for each node
            let mut mailboxes: Vec<Vec<Vec<f32>>> = vec![Vec::new(); num_nodes];

            for (vi, wi, evw) in adj_with_edge_features {
                let vi = *vi;
                let wi = *wi;
                if vi >= num_nodes || wi >= num_nodes {
                    return Err(GraphGenError::NodeIndexOutOfRange {
                        index: vi.max(wi),
                        num_nodes,
                    });
                }
                if evw.len() != ef {
                    return Err(GraphGenError::DimensionMismatch {
                        expected: ef,
                        found: evw.len(),
                        context: "MpnnLayer::forward edge features",
                    });
                }
                // Project node hidden states back to node_features for message fn
                let h_v_proj: Vec<f32> = {
                    let hv = &node_h[vi];
                    let mut p = vec![0.0_f32; nf];
                    let copy = hv.len().min(nf);
                    p[..copy].copy_from_slice(&hv[..copy]);
                    p
                };
                let h_w_proj: Vec<f32> = {
                    let hw = &node_h[wi];
                    let mut p = vec![0.0_f32; nf];
                    let copy = hw.len().min(nf);
                    p[..copy].copy_from_slice(&hw[..copy]);
                    p
                };
                let msg = self.message(&h_v_proj, &h_w_proj, evw)?;
                mailboxes[vi].push(msg);
            }

            for v in 0..num_nodes {
                let agg = self.aggregate(&mailboxes[v]);
                new_h[v] = self.update_gru(&node_h[v], &agg)?;
            }
            node_h = new_h;
        }

        // Flatten [num_nodes × hidden_dim]
        let mut out = Vec::with_capacity(num_nodes * hd);
        for hv in node_h {
            out.extend_from_slice(&hv);
        }
        Ok(out)
    }

    /// Sum-pool all node hidden states to a single graph-level vector.
    ///
    /// `h`: [num_nodes × hidden_dim] → output: \[hidden_dim\].
    pub fn readout_sum(
        h: &[f32],
        num_nodes: usize,
        hidden_dim: usize,
    ) -> Result<Vec<f32>, GraphGenError> {
        if num_nodes == 0 {
            return Err(GraphGenError::EmptyGraph);
        }
        if h.len() != num_nodes * hidden_dim {
            return Err(GraphGenError::DimensionMismatch {
                expected: num_nodes * hidden_dim,
                found: h.len(),
                context: "readout_sum h",
            });
        }
        let mut pool = vec![0.0_f32; hidden_dim];
        for i in 0..num_nodes {
            for j in 0..hidden_dim {
                pool[j] += h[i * hidden_dim + j];
            }
        }
        Ok(pool)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph Property Predictor
// ─────────────────────────────────────────────────────────────────────────────

/// MPNN-based molecular property predictor.
///
/// Architecture: MPNN (multiple message-passing rounds) → sum readout → MLP →
/// output_dim predictions.
pub struct GraphPropertyPredictor {
    pub mpnn: MpnnLayer,
    /// Readout MLP weight: [hidden_dim × output_dim].
    pub readout_w: Vec<f32>,
    /// Readout bias: \[output_dim\].
    pub readout_b: Vec<f32>,
    /// Number of output properties.
    pub output_dim: usize,
}

impl GraphPropertyPredictor {
    /// Create a predictor with Xavier-initialised readout weights.
    pub fn new(mpnn_config: MpnnConfig, output_dim: usize) -> Result<Self, GraphGenError> {
        if output_dim == 0 {
            return Err(GraphGenError::InvalidParameter {
                name: "output_dim",
                message: "must be >= 1".to_string(),
            });
        }
        let hd = mpnn_config.hidden_dim;
        let readout_w = xavier_uniform(hd, output_dim, 800);
        let readout_b = vec![0.0_f32; output_dim];
        let mpnn = MpnnLayer::new(mpnn_config);
        Ok(GraphPropertyPredictor {
            mpnn,
            readout_w,
            readout_b,
            output_dim,
        })
    }

    /// Predict molecular properties for a given graph.
    ///
    /// Edge features are derived from bond types in the graph (bond order as
    /// a single float).
    pub fn predict(&self, graph: &MolecularGraph) -> Result<Vec<f32>, GraphGenError> {
        if graph.num_atoms == 0 {
            return Err(GraphGenError::EmptyGraph);
        }
        let ef = self.mpnn.config.edge_features;
        // Build edge list with features from bond types
        let adj_with_ef: Vec<(usize, usize, Vec<f32>)> = graph
            .bond_types
            .iter()
            .map(|(i, j, bt)| {
                // Edge feature: [bond_order, 0, 0, ...] padded to edge_features
                let mut ef_vec = vec![0.0_f32; ef];
                if ef > 0 {
                    ef_vec[0] = bt.order();
                }
                (*i, *j, ef_vec)
            })
            .collect();

        // Run message passing
        let h_out = self
            .mpnn
            .forward(&graph.atom_features, &adj_with_ef, graph.num_atoms)?;

        // Sum readout
        let graph_repr =
            MpnnLayer::readout_sum(&h_out, graph.num_atoms, self.mpnn.config.hidden_dim)?;

        // Linear projection to output
        let hd = self.mpnn.config.hidden_dim;
        let od = self.output_dim;
        let mut pred = self.readout_b.clone();
        for j in 0..od {
            for k in 0..hd {
                pred[j] += graph_repr[k] * self.readout_w[k * od + j];
            }
        }
        Ok(pred)
    }

    /// MSE loss between predictions and targets.
    pub fn property_loss(preds: &[f32], targets: &[f32]) -> Result<f32, GraphGenError> {
        if preds.len() != targets.len() {
            return Err(GraphGenError::DimensionMismatch {
                expected: targets.len(),
                found: preds.len(),
                context: "property_loss preds/targets",
            });
        }
        if preds.is_empty() {
            return Ok(0.0);
        }
        let mse = preds
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| (p - t) * (p - t))
            .sum::<f32>()
            / preds.len() as f32;
        Ok(mse)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── MolecularGraph ────────────────────────────────────────────────────

    #[test]
    fn test_molecular_graph_add_edge_and_degree() {
        let mut g = MolecularGraph::new(4, 3);
        g.add_edge(0, 1, BondType::Single);
        g.add_edge(1, 2, BondType::Double);
        g.add_edge(2, 3, BondType::Aromatic);
        assert_eq!(g.degree(0), 1);
        assert_eq!(g.degree(1), 2);
        assert_eq!(g.degree(2), 2);
        assert_eq!(g.degree(3), 1);
    }

    #[test]
    fn test_molecular_graph_self_loop_ignored() {
        let mut g = MolecularGraph::new(3, 2);
        g.add_edge(1, 1, BondType::Single); // self-loop — must be ignored
        assert_eq!(g.degree(1), 0);
        assert!(g.bond_types.is_empty());
    }

    #[test]
    fn test_molecular_graph_adjacency_matrix_symmetry() {
        let mut g = MolecularGraph::new(3, 1);
        g.add_edge(0, 1, BondType::Single);
        g.add_edge(1, 2, BondType::Double);
        let a = g.adjacency_matrix();
        // Must be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(a[i * 3 + j], a[j * 3 + i], "A[{i},{j}] != A[{j},{i}]");
            }
        }
        // Diagonal must be zero (no self-loops)
        for i in 0..3 {
            assert_eq!(a[i * 3 + i], 0.0);
        }
    }

    #[test]
    fn test_molecular_graph_adjacency_values() {
        let mut g = MolecularGraph::new(3, 1);
        g.add_edge(0, 2, BondType::Triple);
        let a = g.adjacency_matrix();
        assert_eq!(a[2], 1.0); // row 0, col 2
        assert_eq!(a[2 * 3], 1.0); // row 2, col 0
        assert_eq!(a[1], 0.0); // row 0, col 1
    }

    #[test]
    fn test_laplacian_diagonal_equals_degree() {
        let mut g = MolecularGraph::new(4, 1);
        g.add_edge(0, 1, BondType::Single);
        g.add_edge(0, 2, BondType::Single);
        g.add_edge(1, 3, BondType::Double);
        let l = g.laplacian();
        let n = 4;
        // Diagonal L[i,i] = degree(i)
        assert_eq!(l[0], g.degree(0) as f32); // l[0*n + 0]
        assert_eq!(l[n + 1], g.degree(1) as f32);
        assert_eq!(l[2 * n + 2], g.degree(2) as f32);
        assert_eq!(l[3 * n + 3], g.degree(3) as f32);
    }

    #[test]
    fn test_laplacian_off_diagonal_is_negative_adjacency() {
        let mut g = MolecularGraph::new(3, 1);
        g.add_edge(0, 1, BondType::Single);
        g.add_edge(1, 2, BondType::Single);
        let l = g.laplacian();
        let a = g.adjacency_matrix();
        let n = 3;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    assert_eq!(
                        l[i * n + j],
                        -a[i * n + j],
                        "L[{i},{j}] should equal -A[{i},{j}]"
                    );
                }
            }
        }
    }

    #[test]
    fn test_is_valid_no_self_loops() {
        let mut g = MolecularGraph::new(3, 2);
        g.add_edge(0, 1, BondType::Single);
        assert!(g.is_valid());
    }

    #[test]
    fn test_encode_atom_one_hot_correct_dimension() {
        let allowed = vec![AtomType::C, AtomType::N, AtomType::O];
        let enc = MolecularGraph::encode_atom_one_hot(&AtomType::N, &allowed);
        // Length = allowed + 1 (other bucket)
        assert_eq!(enc.len(), 4);
    }

    #[test]
    fn test_encode_atom_one_hot_sum_is_one() {
        let allowed = vec![AtomType::C, AtomType::N, AtomType::O, AtomType::F];
        let enc = MolecularGraph::encode_atom_one_hot(&AtomType::O, &allowed);
        let sum: f32 = enc.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "one-hot sum should be 1, got {sum}"
        );
    }

    #[test]
    fn test_encode_atom_one_hot_unknown_goes_to_last_bucket() {
        let allowed = vec![AtomType::C, AtomType::N];
        let enc = MolecularGraph::encode_atom_one_hot(&AtomType::Br, &allowed);
        // Last element should be 1
        assert_eq!(enc[enc.len() - 1], 1.0);
        assert_eq!(enc[0], 0.0);
        assert_eq!(enc[1], 0.0);
    }

    // ── VGAE ─────────────────────────────────────────────────────────────

    #[test]
    fn test_vgae_encode_output_shapes() {
        let config = VgaeConfig {
            input_features: 4,
            hidden_dim: 8,
            latent_dim: 3,
            num_gcn_layers: 2,
        };
        let encoder = VgaeEncoder::new(config);
        let n = 5;
        let x = vec![0.1_f32; n * 4];
        let mut g = MolecularGraph::new(n, 1);
        g.add_edge(0, 1, BondType::Single);
        g.add_edge(1, 2, BondType::Single);
        let adj = g.adjacency_matrix();
        let (mu, log_var) = encoder.encode(&x, &adj, n).expect("encode should succeed");
        assert_eq!(mu.len(), n * 3);
        assert_eq!(log_var.len(), n * 3);
    }

    #[test]
    fn test_vgae_reparameterize_output_shape() {
        let mu = vec![0.0_f32; 6];
        let log_var = vec![0.0_f32; 6];
        let z = VgaeEncoder::reparameterize(&mu, &log_var, 42).expect("reparameterize");
        assert_eq!(z.len(), 6);
    }

    #[test]
    fn test_vgae_reparameterize_values_are_finite() {
        let mu = vec![1.0_f32, -1.0, 2.0];
        let log_var = vec![-1.0_f32, 0.5, -2.0];
        let z = VgaeEncoder::reparameterize(&mu, &log_var, 7).expect("reparameterize");
        for (i, v) in z.iter().enumerate() {
            assert!(v.is_finite(), "z[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn test_vgae_decode_is_n_times_n_and_in_zero_one() {
        let n = 4;
        let latent_dim = 3;
        let z = vec![0.5_f32; n * latent_dim];
        let a_hat = VgaeEncoder::decode(&z, n, latent_dim).expect("decode");
        assert_eq!(a_hat.len(), n * n);
        for (i, v) in a_hat.iter().enumerate() {
            assert!(*v >= 0.0 && *v <= 1.0, "a_hat[{i}] = {v} is out of [0,1]");
        }
    }

    #[test]
    fn test_vgae_kl_zero_for_standard_normal() {
        // mu=0, log_var=0 → KL should be ~0 (≤ small numerical noise)
        let n = 10;
        let mu = vec![0.0_f32; n];
        let log_var = vec![0.0_f32; n];
        let adj_true = vec![0.0_f32; n * n];
        let adj_pred = vec![0.5_f32; n * n];
        let (_, kl) =
            VgaeEncoder::loss(&adj_true, &adj_pred, &mu, &log_var).expect("loss computation");
        assert!(kl.abs() < 1e-4, "KL should be ~0 for N(0,I), got {kl}");
    }

    #[test]
    fn test_vgae_bce_is_finite() {
        let n = 3;
        let mu = vec![0.1_f32; n * 2];
        let log_var = vec![-0.5_f32; n * 2];
        let adj_true = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let adj_pred = vec![0.2_f32; 9];
        let (bce, kl) =
            VgaeEncoder::loss(&adj_true, &adj_pred, &mu, &log_var).expect("loss computation");
        assert!(bce.is_finite(), "BCE is not finite: {bce}");
        assert!(kl.is_finite(), "KL is not finite: {kl}");
    }

    // ── Morgan fingerprint ───────────────────────────────────────────────

    #[test]
    fn test_morgan_fingerprint_deterministic() {
        let mut g = MolecularGraph::new(3, 2);
        g.atom_features = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
        g.add_edge(0, 1, BondType::Single);
        g.add_edge(1, 2, BondType::Double);
        let fp1 = MolecularFingerprint::morgan_fingerprint(&g, 2, 64).expect("morgan fp");
        let fp2 =
            MolecularFingerprint::morgan_fingerprint(&g, 2, 64).expect("morgan fp second call");
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_morgan_fingerprint_different_atoms_differ() {
        // Use clearly distinct continuous values to avoid hash collisions in the folded
        // fingerprint.  We verify by using a wide (2048-bit) fingerprint and structurally
        // distinct graphs (different connectivity + very different feature magnitudes).
        let mut g1 = MolecularGraph::new(3, 4);
        // C-like features
        g1.atom_features = vec![
            1.0, 0.0, 0.0, 0.0, // atom 0
            1.0, 0.0, 0.0, 0.0, // atom 1
            1.0, 0.0, 0.0, 0.0, // atom 2
        ];
        g1.add_edge(0, 1, BondType::Single);
        g1.add_edge(1, 2, BondType::Single);

        let mut g2 = MolecularGraph::new(3, 4);
        // N-like features — clearly different
        g2.atom_features = vec![
            0.0, 1.0, 0.0, 0.0, // atom 0
            0.0, 1.0, 0.0, 0.0, // atom 1
            0.0, 1.0, 0.0, 0.0, // atom 2
        ];
        g2.add_edge(0, 1, BondType::Double);
        g2.add_edge(1, 2, BondType::Double);

        let fp1 = MolecularFingerprint::morgan_fingerprint(&g1, 2, 2048).expect("fp1");
        let fp2 = MolecularFingerprint::morgan_fingerprint(&g2, 2, 2048).expect("fp2");
        assert_ne!(
            fp1, fp2,
            "fingerprints should differ for structurally distinct graphs"
        );
    }

    #[test]
    fn test_morgan_fingerprint_length() {
        let g = MolecularGraph::new(4, 3);
        let bits = 128;
        let fp = MolecularFingerprint::morgan_fingerprint(&g, 1, bits).expect("fp");
        assert_eq!(fp.len(), bits);
    }

    #[test]
    fn test_atom_pair_fingerprint_finite_values() {
        let mut g = MolecularGraph::new(3, 2);
        g.atom_features = vec![1.0, 0.5, 0.3, 0.7, 0.0, 1.0];
        g.add_edge(0, 1, BondType::Single);
        let fp = MolecularFingerprint::atom_pair_fingerprint(&g);
        for (i, v) in fp.iter().enumerate() {
            assert!(v.is_finite(), "atom_pair_fp[{i}] is not finite: {v}");
        }
        // Expected length: C(3,2) pairs × 3 elements = 3 × 3 = 9
        assert_eq!(fp.len(), 9);
    }

    #[test]
    fn test_topological_torsion_finite() {
        let mut g = MolecularGraph::new(6, 2);
        g.atom_features = vec![0.1_f32; 12];
        for i in 0..5 {
            g.add_edge(i, i + 1, BondType::Single);
        }
        let fp = MolecularFingerprint::topological_torsion(&g);
        assert_eq!(fp.len(), 256);
        for v in &fp {
            assert!(v.is_finite());
        }
    }

    // ── GraphRNN ──────────────────────────────────────────────────────────

    #[test]
    fn test_graphrnn_generate_no_self_loops() {
        let config = GraphRnnConfig {
            max_nodes: 6,
            node_feature_dim: 4,
            hidden_dim: 8,
            edge_hidden_dim: 6,
        };
        let rnn = GraphRnn::new(config);
        let graph = rnn.generate(42, 6).expect("generate");
        assert!(graph.is_valid(), "generated graph must be valid");
    }

    #[test]
    fn test_graphrnn_generate_has_nodes() {
        let config = GraphRnnConfig {
            max_nodes: 5,
            node_feature_dim: 3,
            hidden_dim: 6,
            edge_hidden_dim: 4,
        };
        let rnn = GraphRnn::new(config);
        let graph = rnn.generate(1, 5).expect("generate");
        assert!(graph.num_atoms >= 1);
    }

    // ── MpnnLayer ─────────────────────────────────────────────────────────

    #[test]
    fn test_mpnn_message_output_shape() {
        let config = MpnnConfig {
            node_features: 4,
            edge_features: 2,
            hidden_dim: 8,
            num_message_steps: 1,
        };
        let mpnn = MpnnLayer::new(config);
        let h_v = vec![0.1_f32; 4];
        let h_w = vec![0.2_f32; 4];
        let e = vec![1.0_f32; 2];
        let msg = mpnn.message(&h_v, &h_w, &e).expect("message");
        assert_eq!(msg.len(), 8);
    }

    #[test]
    fn test_mpnn_aggregate_returns_correct_shape() {
        let config = MpnnConfig {
            node_features: 3,
            edge_features: 2,
            hidden_dim: 5,
            num_message_steps: 1,
        };
        let mpnn = MpnnLayer::new(config);
        let messages = vec![vec![1.0_f32; 5], vec![2.0_f32; 5]];
        let agg = mpnn.aggregate(&messages);
        assert_eq!(agg.len(), 5);
        // Sum should be elementwise 3.0
        for v in &agg {
            assert!((v - 3.0).abs() < 1e-5, "agg element should be 3.0, got {v}");
        }
    }

    #[test]
    fn test_mpnn_forward_output_shape() {
        let config = MpnnConfig {
            node_features: 4,
            edge_features: 2,
            hidden_dim: 6,
            num_message_steps: 2,
        };
        let mpnn = MpnnLayer::new(config);
        let num_nodes = 3;
        let h = vec![0.1_f32; num_nodes * 4];
        let adj: Vec<(usize, usize, Vec<f32>)> = vec![
            (0, 1, vec![1.0, 0.0]),
            (1, 0, vec![1.0, 0.0]),
            (1, 2, vec![0.0, 1.0]),
            (2, 1, vec![0.0, 1.0]),
        ];
        let h_out = mpnn.forward(&h, &adj, num_nodes).expect("forward");
        assert_eq!(h_out.len(), num_nodes * 6);
    }

    #[test]
    fn test_mpnn_readout_sum_shape() {
        let h = vec![1.0_f32; 4 * 5]; // 4 nodes × 5-dim
        let pool = MpnnLayer::readout_sum(&h, 4, 5).expect("readout_sum");
        assert_eq!(pool.len(), 5);
        // Each element should be 4.0
        for v in &pool {
            assert!(
                (v - 4.0).abs() < 1e-5,
                "readout element should be 4.0, got {v}"
            );
        }
    }

    // ── GraphPropertyPredictor ────────────────────────────────────────────

    #[test]
    fn test_graph_property_predictor_output_shape() {
        let mpnn_config = MpnnConfig {
            node_features: 4,
            edge_features: 1, // bond order
            hidden_dim: 8,
            num_message_steps: 2,
        };
        let predictor = GraphPropertyPredictor::new(mpnn_config, 3).expect("predictor");
        let mut g = MolecularGraph::new(4, 4);
        g.atom_features = vec![0.1_f32; 16];
        g.add_edge(0, 1, BondType::Single);
        g.add_edge(1, 2, BondType::Double);
        g.add_edge(2, 3, BondType::Aromatic);
        let preds = predictor.predict(&g).expect("predict");
        assert_eq!(preds.len(), 3, "output should have 3 properties");
    }

    #[test]
    fn test_graph_property_predictor_mse_loss() {
        let preds = vec![1.0_f32, 2.0, 3.0];
        let targets = vec![1.0_f32, 2.0, 3.0];
        let loss = GraphPropertyPredictor::property_loss(&preds, &targets).expect("mse");
        assert!(
            loss.abs() < 1e-6,
            "MSE of identical vectors should be 0, got {loss}"
        );
    }

    #[test]
    fn test_graph_property_predictor_mse_nonzero() {
        let preds = vec![0.0_f32, 0.0];
        let targets = vec![1.0_f32, 1.0];
        let loss = GraphPropertyPredictor::property_loss(&preds, &targets).expect("mse");
        assert!((loss - 1.0).abs() < 1e-5, "MSE should be 1.0, got {loss}");
    }

    #[test]
    fn test_mpnn_empty_edge_list() {
        let config = MpnnConfig {
            node_features: 3,
            edge_features: 2,
            hidden_dim: 4,
            num_message_steps: 1,
        };
        let mpnn = MpnnLayer::new(config);
        let h = vec![0.5_f32; 2 * 3];
        let adj: Vec<(usize, usize, Vec<f32>)> = vec![];
        let out = mpnn.forward(&h, &adj, 2).expect("forward with no edges");
        assert_eq!(out.len(), 2 * 4);
    }

    #[test]
    fn test_vgae_decode_approx_symmetric() {
        // For random z, A_hat = sigmoid(Z Z^T) must be symmetric
        let n = 4;
        let latent_dim = 2;
        let z = vec![
            0.1, 0.2, // node 0
            0.3, -0.1, // node 1
            -0.2, 0.4, // node 2
            0.5, 0.1, // node 3
        ];
        let a_hat = VgaeEncoder::decode(&z, n, latent_dim).expect("decode");
        for i in 0..n {
            for j in 0..n {
                let diff = (a_hat[i * n + j] - a_hat[j * n + i]).abs();
                assert!(
                    diff < 1e-5,
                    "A_hat[{i},{j}]={} != A_hat[{j},{i}]={}",
                    a_hat[i * n + j],
                    a_hat[j * n + i]
                );
            }
        }
    }
}
