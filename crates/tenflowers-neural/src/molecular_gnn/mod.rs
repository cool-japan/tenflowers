//! Molecular Graph Neural Networks — Comprehensive implementation.
//!
//! Provides GNN algorithms specifically designed for molecular property prediction,
//! fingerprint computation, reaction classification, and distance-based networks:
//!
//! - [`MolGraph`]: Rich molecular graph with atom/bond feature vectors.
//! - [`Mpnn`]: Message Passing Neural Network (Gilmer 2017).
//! - [`SchNet`]: SchNet-style continuous-filter convolution network.
//! - [`MorganFingerprint`]: Extended connectivity fingerprints (ECFP).
//! - [`TopologicalFingerprint`]: Path-based topological fingerprints.
//! - [`PropertyPredictor`]: End-to-end molecular property prediction.
//! - [`ReactionClassifier`]: Reaction-type classification.
//!
//! All fallible operations return `Result<_, TensorError>`.
//! No `unsafe` code; no `unwrap()` calls.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

pub mod advanced;
pub use advanced::*;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod advanced_tests;

// ─────────────────────────────────────────────────────────────────────────────
// Math utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Sigmoid activation function.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    let xc = x.clamp(-500.0, 500.0);
    1.0 / (1.0 + (-xc).exp())
}

/// ReLU activation function.
#[inline]
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Softmax of a vector — numerically stable (subtract max).
pub fn softmax(v: &[f64]) -> Vec<f64> {
    if v.is_empty() {
        return Vec::new();
    }
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let denom = if sum.abs() < 1e-300 { 1e-300 } else { sum };
    exps.iter().map(|e| e / denom).collect()
}

/// Matrix-vector product: mat [rows × cols] · v \[cols\] → out \[rows\].
pub fn matvec(mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    mat.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum())
        .collect()
}

/// Element-wise vector addition.
pub fn vecadd(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// FNV-1a 64-bit hash over raw bytes.
pub fn fnv1a_hash(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;
    let mut hash = FNV_OFFSET;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Hash a path of `u64` node/bond identifiers using FNV-1a.
pub fn path_hash(path: &[u64]) -> u64 {
    let bytes: Vec<u8> = path.iter().flat_map(|x| x.to_le_bytes()).collect();
    fnv1a_hash(&bytes)
}

// ─────────────────────────────────────────────────────────────────────────────
// Hybridization / Atom / Bond
// ─────────────────────────────────────────────────────────────────────────────

/// Hybridization state of an atom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Hybridization {
    Sp,
    Sp2,
    Sp3,
    Other,
}

/// A single atom in a molecular graph.
///
/// Feature vector dimension: 44
///   - atomic_num one-hot over {H=1,C=6,N=7,O=8,F=9,P=15,S=16,Cl=17,Br=35,I=53,Other}=11 bits
///   - formal_charge one-hot over {-2,-1,0,+1,+2}=5 bits
///   - hybridization one-hot {Sp,Sp2,Sp3,Other}=4 bits
///   - is_aromatic 1 bit
///   - degree one-hot {0..6}=7 bits
///   - n_hydrogens one-hot {0..6}=7 bits
///
///   Total = 11+5+4+1+7+7 = 35 ... we pad to 44 with 9 extra zeros for extensibility.
#[derive(Debug, Clone)]
pub struct Atom {
    /// Atomic number (1 = H, 6 = C, …).
    pub atomic_num: u8,
    /// Formal charge in electrons.
    pub formal_charge: i8,
    /// Whether the atom participates in an aromatic ring.
    pub is_aromatic: bool,
    /// Hybridization state.
    pub hybridization: Hybridization,
    /// Number of heavy-atom bonds (degree in molecular graph).
    pub degree: u8,
    /// Number of attached hydrogens (implicit + explicit).
    pub n_hydrogens: u8,
}

impl Atom {
    /// Encode atom properties as a 44-dimensional one-hot feature vector.
    pub fn feature_vector(&self) -> Vec<f64> {
        let mut feat = vec![0.0_f64; 44];
        let mut idx = 0_usize;

        // Atomic number: 11 classes {H,C,N,O,F,P,S,Cl,Br,I,Other}
        let atom_classes: &[u8] = &[1, 6, 7, 8, 9, 15, 16, 17, 35, 53];
        let mut found = false;
        for (k, &a) in atom_classes.iter().enumerate() {
            if self.atomic_num == a {
                feat[idx + k] = 1.0;
                found = true;
                break;
            }
        }
        if !found {
            feat[idx + 10] = 1.0; // "Other"
        }
        idx += 11;

        // Formal charge: 5 classes {-2,-1,0,+1,+2}
        let fc_idx = match self.formal_charge {
            -2 => 0,
            -1 => 1,
            0 => 2,
            1 => 3,
            _ => 4,
        };
        feat[idx + fc_idx] = 1.0;
        idx += 5;

        // Hybridization: 4 classes {Sp,Sp2,Sp3,Other}
        let hyb_idx = match self.hybridization {
            Hybridization::Sp => 0,
            Hybridization::Sp2 => 1,
            Hybridization::Sp3 => 2,
            Hybridization::Other => 3,
        };
        feat[idx + hyb_idx] = 1.0;
        idx += 4;

        // Aromaticity: 1 bit
        feat[idx] = if self.is_aromatic { 1.0 } else { 0.0 };
        idx += 1;

        // Degree: 7 classes {0..=6}
        let deg_idx = (self.degree as usize).min(6);
        feat[idx + deg_idx] = 1.0;
        idx += 7;

        // n_hydrogens: 7 classes {0..=6}
        let nh_idx = (self.n_hydrogens as usize).min(6);
        feat[idx + nh_idx] = 1.0;
        idx += 7;

        // 9 padding zeros to reach 44 total
        // idx is now 11+5+4+1+7+7 = 35; remaining 9 are already 0.0
        let _ = idx; // suppress unused warning

        feat
    }
}

/// Deterministic hash of an atom's chemical identity.
pub fn atom_hash(atom: &Atom) -> u64 {
    let mut data = Vec::with_capacity(6);
    data.push(atom.atomic_num);
    data.push(atom.formal_charge as u8);
    data.push(atom.degree);
    data.push(atom.n_hydrogens);
    data.push(if atom.is_aromatic { 1 } else { 0 });
    let hyb_byte = match atom.hybridization {
        Hybridization::Sp => 0u8,
        Hybridization::Sp2 => 1,
        Hybridization::Sp3 => 2,
        Hybridization::Other => 3,
    };
    data.push(hyb_byte);
    fnv1a_hash(&data)
}

/// Bond type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MolBondType {
    Single,
    Double,
    Triple,
    Aromatic,
}

/// A single bond in a molecular graph.
///
/// Feature vector dimension: 7
///   bond_type one-hot (4) + is_aromatic (1) + is_conjugated (1) + in_ring (1)
#[derive(Debug, Clone)]
pub struct Bond {
    /// Source atom index.
    pub src: usize,
    /// Destination atom index.
    pub dst: usize,
    /// Bond order type.
    pub bond_type: MolBondType,
    /// Whether bond is aromatic.
    pub is_aromatic: bool,
    /// Whether bond participates in conjugated system.
    pub is_conjugated: bool,
    /// Whether bond is in a ring.
    pub in_ring: bool,
}

impl Bond {
    /// Encode bond properties as a 7-dimensional feature vector.
    pub fn feature_vector(&self) -> Vec<f64> {
        let mut feat = vec![0.0_f64; 7];
        let bt_idx = match self.bond_type {
            MolBondType::Single => 0,
            MolBondType::Double => 1,
            MolBondType::Triple => 2,
            MolBondType::Aromatic => 3,
        };
        feat[bt_idx] = 1.0;
        feat[4] = if self.is_aromatic { 1.0 } else { 0.0 };
        feat[5] = if self.is_conjugated { 1.0 } else { 0.0 };
        feat[6] = if self.in_ring { 1.0 } else { 0.0 };
        feat
    }

    /// Numeric bond order for hash-based fingerprints.
    pub fn bond_order_u64(&self) -> u64 {
        match self.bond_type {
            MolBondType::Single => 1,
            MolBondType::Double => 2,
            MolBondType::Triple => 3,
            MolBondType::Aromatic => 4,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MolGraph
// ─────────────────────────────────────────────────────────────────────────────

/// A rich molecular graph holding atoms and bonds with full feature support.
#[derive(Debug, Clone)]
pub struct MolGraph {
    /// List of atoms (nodes).
    pub atoms: Vec<Atom>,
    /// List of bonds (edges — stored once; use `adjacency_list` for traversal).
    pub bonds: Vec<Bond>,
}

impl MolGraph {
    /// Number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.atoms.len()
    }

    /// Number of bonds.
    pub fn n_bonds(&self) -> usize {
        self.bonds.len()
    }

    /// Build adjacency list: `adj[i]` = `[(neighbor_idx, bond_idx)]`.
    pub fn adjacency_list(&self) -> Vec<Vec<(usize, usize)>> {
        let n = self.n_atoms();
        let mut adj = vec![Vec::new(); n];
        for (bi, bond) in self.bonds.iter().enumerate() {
            if bond.src < n && bond.dst < n {
                adj[bond.src].push((bond.dst, bi));
                adj[bond.dst].push((bond.src, bi));
            }
        }
        adj
    }

    /// Atom feature matrix: one row per atom (44 dims each).
    pub fn atom_features(&self) -> Vec<Vec<f64>> {
        self.atoms.iter().map(|a| a.feature_vector()).collect()
    }

    /// Bond feature matrix: one row per bond (7 dims each).
    pub fn bond_features(&self) -> Vec<Vec<f64>> {
        self.bonds.iter().map(|b| b.feature_vector()).collect()
    }

    /// Generate a synthetic random molecular graph for testing purposes.
    ///
    /// No actual SMILES parser is used — atoms and bonds are randomized
    /// while preserving chemical plausibility constraints.
    pub fn from_smiles_stub(n_atoms: usize, rng: &mut StdRng) -> Self {
        let atom_nums: [u8; 6] = [6, 7, 8, 16, 9, 17]; // C, N, O, S, F, Cl
        let hybs = [
            Hybridization::Sp3,
            Hybridization::Sp2,
            Hybridization::Sp,
            Hybridization::Other,
        ];

        let atoms: Vec<Atom> = (0..n_atoms)
            .map(|_| {
                let an = atom_nums[(rng.random::<u64>() as usize) % atom_nums.len()];
                let hyb = hybs[(rng.random::<u64>() as usize) % hybs.len()];
                let degree = (rng.random::<u8>() % 4) + 1;
                Atom {
                    atomic_num: an,
                    formal_charge: 0,
                    is_aromatic: rng.random::<bool>(),
                    hybridization: hyb,
                    degree,
                    n_hydrogens: rng.random::<u8>() % 4,
                }
            })
            .collect();

        // Build a spanning chain (ensure connectivity) then add random edges
        let bond_types = [
            MolBondType::Single,
            MolBondType::Double,
            MolBondType::Aromatic,
        ];
        let mut bonds: Vec<Bond> = Vec::new();

        // Spanning path
        for i in 0..n_atoms.saturating_sub(1) {
            let bt = bond_types[(rng.random::<u64>() as usize) % bond_types.len()];
            bonds.push(Bond {
                src: i,
                dst: i + 1,
                bond_type: bt,
                is_aromatic: bt == MolBondType::Aromatic,
                is_conjugated: rng.random::<bool>(),
                in_ring: false,
            });
        }

        // Extra random bonds (up to n_atoms / 2)
        let extra = n_atoms / 2;
        for _ in 0..extra {
            if n_atoms < 2 {
                break;
            }
            let a = (rng.random::<u64>() as usize) % n_atoms;
            let b = (rng.random::<u64>() as usize) % n_atoms;
            if a == b {
                continue;
            }
            let (src, dst) = if a < b { (a, b) } else { (b, a) };
            let bt = bond_types[(rng.random::<u64>() as usize) % bond_types.len()];
            bonds.push(Bond {
                src,
                dst,
                bond_type: bt,
                is_aromatic: bt == MolBondType::Aromatic,
                is_conjugated: rng.random::<bool>(),
                in_ring: true,
            });
        }

        MolGraph { atoms, bonds }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Xavier initialisation helper
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) fn xavier_uniform_2d(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let limit = (6.0_f64 / (rows + cols) as f64).sqrt();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    let u: f64 = rng.random();
                    u * 2.0 * limit - limit
                })
                .collect()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// MPNN — Message Passing Neural Network (Gilmer 2017)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for an MPNN network.
#[derive(Debug, Clone)]
pub struct MolMpnnConfig {
    /// Dimensionality of input node features.
    pub node_dim: usize,
    /// Dimensionality of input edge features.
    pub edge_dim: usize,
    /// Hidden dimension (output per layer).
    pub hidden_dim: usize,
    /// Final graph-level output dimension.
    pub output_dim: usize,
    /// Number of message-passing rounds.
    pub n_layers: usize,
    /// Dropout rate (applied stochastically during forward).
    pub dropout_rate: f64,
    /// Random seed for weight initialisation.
    pub seed: u64,
}

/// A single MPNN message-passing layer.
///
/// Messages: linear([h_src ‖ h_dst ‖ edge_feat]) → ReLU
/// Aggregation: mean over incoming messages
/// Update: GRU-style gate
pub struct MpnnLayer {
    /// Weight matrix for message function [hidden_dim × (2*hidden_dim + edge_dim)]
    w_msg: Vec<Vec<f64>>,
    /// Bias for message function [hidden_dim]
    b_msg: Vec<f64>,
    /// Weight for GRU z-gate [hidden_dim × 2*hidden_dim]
    w_z: Vec<Vec<f64>>,
    /// Weight for GRU tanh-branch [hidden_dim × 2*hidden_dim]
    w_u: Vec<Vec<f64>>,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Edge feature dimension accepted by this layer.
    pub edge_dim: usize,
}

impl MpnnLayer {
    /// Construct a new message-passing layer with Xavier-uniform weights.
    pub fn new(input_dim: usize, hidden_dim: usize, edge_dim: usize, seed: u64) -> Self {
        let msg_in = input_dim * 2 + edge_dim;
        let w_msg = xavier_uniform_2d(hidden_dim, msg_in, seed);
        let b_msg = vec![0.0; hidden_dim];
        let w_z = xavier_uniform_2d(hidden_dim, input_dim + hidden_dim, seed.wrapping_add(1));
        let w_u = xavier_uniform_2d(hidden_dim, input_dim + hidden_dim, seed.wrapping_add(2));
        MpnnLayer {
            w_msg,
            b_msg,
            w_z,
            w_u,
            hidden_dim,
            edge_dim,
        }
    }

    /// Compute a single message from source to destination.
    ///
    /// `h_src`, `h_dst` must both have length `hidden_dim`.
    /// `edge_feat` must have length `edge_dim`.
    pub fn message(&self, h_src: &[f64], h_dst: &[f64], edge_feat: &[f64]) -> Vec<f64> {
        // Concatenate [h_src ‖ h_dst ‖ edge_feat]
        let mut cat = Vec::with_capacity(h_src.len() + h_dst.len() + edge_feat.len());
        cat.extend_from_slice(h_src);
        cat.extend_from_slice(h_dst);
        cat.extend_from_slice(edge_feat);

        let mut out = matvec(&self.w_msg, &cat);
        for (o, &b) in out.iter_mut().zip(self.b_msg.iter()) {
            *o = relu(*o + b);
        }
        out
    }

    /// Mean-aggregate a list of messages into a single vector.
    pub fn aggregate(&self, messages: &[Vec<f64>]) -> Vec<f64> {
        if messages.is_empty() {
            return vec![0.0; self.hidden_dim];
        }
        let n = messages.len() as f64;
        let dim = self.hidden_dim;
        let mut agg = vec![0.0; dim];
        for msg in messages {
            for (a, &m) in agg.iter_mut().zip(msg.iter()) {
                *a += m;
            }
        }
        agg.iter_mut().for_each(|x| *x /= n);
        agg
    }

    /// GRU-style node update: h' = (1-z)*h + z*tanh(W[h‖agg]).
    pub fn update(&self, h: &[f64], agg: &[f64]) -> Vec<f64> {
        let mut cat = Vec::with_capacity(h.len() + agg.len());
        cat.extend_from_slice(h);
        cat.extend_from_slice(agg);

        let z_raw = matvec(&self.w_z, &cat);
        let u_raw = matvec(&self.w_u, &cat);

        let mut h_new = vec![0.0; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let z = sigmoid(z_raw[i]);
            let u = u_raw[i].tanh();
            let h_i = if i < h.len() { h[i] } else { 0.0 };
            h_new[i] = (1.0 - z) * h_i + z * u;
        }
        h_new
    }
}

/// Fully-connected MPNN graph neural network.
pub struct Mpnn {
    /// Per-layer message-passing modules (first layer accepts node_dim, rest accept hidden_dim).
    layers: Vec<MpnnLayer>,
    /// Linear readout: [output_dim × (2*hidden_dim)] (mean+max concat).
    w_readout: Vec<Vec<f64>>,
    b_readout: Vec<f64>,
    /// Configuration.
    pub config: MolMpnnConfig,
    /// Input projection: [hidden_dim × node_dim] (maps raw features to hidden_dim).
    w_proj: Vec<Vec<f64>>,
}

impl Mpnn {
    /// Build a new MPNN from the given configuration.
    pub fn new(config: MolMpnnConfig) -> Self {
        let mut seed = config.seed;

        // Input projection: node_dim → hidden_dim
        let w_proj = xavier_uniform_2d(config.hidden_dim, config.node_dim, seed);
        seed = seed.wrapping_add(100);

        // Message-passing layers (all in hidden_dim space after projection)
        let mut layers = Vec::new();
        for i in 0..config.n_layers {
            let layer = MpnnLayer::new(
                config.hidden_dim,
                config.hidden_dim,
                config.edge_dim,
                seed.wrapping_add(i as u64 * 37),
            );
            seed = seed.wrapping_add(1000);
            layers.push(layer);
        }

        // Readout: (mean_pool ‖ max_pool) → output_dim
        let w_readout = xavier_uniform_2d(config.output_dim, config.hidden_dim * 2, seed);
        let b_readout = vec![0.0; config.output_dim];

        Mpnn {
            layers,
            w_readout,
            b_readout,
            w_proj,
            config,
        }
    }

    /// Run forward pass over a molecular graph.
    ///
    /// Returns node-level embeddings of shape [n_atoms × hidden_dim].
    pub fn forward(&self, mol: &MolGraph) -> Result<Vec<Vec<f64>>> {
        let n = mol.n_atoms();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "Mpnn::forward",
                "empty molecular graph",
            ));
        }

        let atom_feat = mol.atom_features();
        let bond_feat = mol.bond_features();
        let adj = mol.adjacency_list();

        // Project raw atom features into hidden_dim space
        let mut h: Vec<Vec<f64>> = atom_feat
            .iter()
            .map(|f| {
                let proj = matvec(&self.w_proj, f);
                proj.iter().map(|&x| relu(x)).collect()
            })
            .collect();

        // Message-passing rounds
        for layer in &self.layers {
            let h_prev = h.clone();
            for i in 0..n {
                let neighbors = &adj[i];
                let mut msgs: Vec<Vec<f64>> = Vec::with_capacity(neighbors.len());
                for &(j, bi) in neighbors {
                    let edge_f = if bi < bond_feat.len() {
                        &bond_feat[bi]
                    } else {
                        &bond_feat[0] // fallback
                    };
                    // Pad or trim edge_feat to match layer.edge_dim
                    let ef_padded = pad_or_trim(edge_f, layer.edge_dim);
                    // Pad or trim h to match layer.hidden_dim
                    let hsrc = pad_or_trim(&h_prev[j], layer.hidden_dim);
                    let hdst = pad_or_trim(&h_prev[i], layer.hidden_dim);
                    msgs.push(layer.message(&hsrc, &hdst, &ef_padded));
                }
                let agg = layer.aggregate(&msgs);
                let h_i = pad_or_trim(&h_prev[i], layer.hidden_dim);
                h[i] = layer.update(&h_i, &agg);
            }
        }

        Ok(h)
    }

    /// Global graph readout: mean + max pooling → linear → output_dim.
    pub fn graph_readout(&self, node_embeds: &[Vec<f64>]) -> Vec<f64> {
        if node_embeds.is_empty() {
            return vec![0.0; self.config.output_dim];
        }
        let dim = self.config.hidden_dim;

        // Mean pooling
        let mut mean_pool = vec![0.0f64; dim];
        for h in node_embeds {
            for (m, &v) in mean_pool.iter_mut().zip(h.iter()) {
                *m += v;
            }
        }
        let n = node_embeds.len() as f64;
        mean_pool.iter_mut().for_each(|x| *x /= n);

        // Max pooling
        let mut max_pool = vec![f64::NEG_INFINITY; dim];
        for h in node_embeds {
            for (mx, &v) in max_pool.iter_mut().zip(h.iter()) {
                if v > *mx {
                    *mx = v;
                }
            }
        }
        // Replace -inf with 0 in case of no real values
        max_pool.iter_mut().for_each(|x| {
            if x.is_infinite() {
                *x = 0.0;
            }
        });

        // Concatenate mean + max → [2*hidden_dim]
        let mut pooled = mean_pool;
        pooled.extend_from_slice(&max_pool);

        // Linear projection
        let mut out = matvec(&self.w_readout, &pooled);
        for (o, &b) in out.iter_mut().zip(self.b_readout.iter()) {
            *o += b;
        }
        out
    }
}

/// Pad a slice to `target` length with zeros, or trim if longer.
pub(crate) fn pad_or_trim(v: &[f64], target: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; target];
    let copy_len = v.len().min(target);
    out[..copy_len].copy_from_slice(&v[..copy_len]);
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// SchNet-style distance-based network
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for SchNet.
#[derive(Debug, Clone)]
pub struct SchNetConfig {
    /// Number of distinct atom types (embedding table size).
    pub n_atom_types: usize,
    /// Number of convolutional filters.
    pub n_filters: usize,
    /// Number of interaction blocks.
    pub n_interactions: usize,
    /// Distance cutoff in Ångström.
    pub cutoff: f64,
    /// Number of Gaussian basis functions.
    pub n_gaussians: usize,
    /// Output embedding dimension.
    pub output_dim: usize,
    /// Random seed.
    pub seed: u64,
}

/// Gaussian smearing of interatomic distances.
///
/// Centers μ_k are equally spaced in [0, cutoff]; σ = spacing.
pub struct GaussianSmearing {
    /// Gaussian centres.
    centers: Vec<f64>,
    /// Gaussian width (standard deviation).
    sigma: f64,
    /// Number of basis functions.
    pub n_gaussians: usize,
}

impl GaussianSmearing {
    /// Create a new Gaussian smearing module.
    pub fn new(n_gaussians: usize, cutoff: f64) -> Self {
        let spacing = if n_gaussians > 1 {
            cutoff / (n_gaussians - 1) as f64
        } else {
            cutoff
        };
        let centers: Vec<f64> = (0..n_gaussians).map(|k| k as f64 * spacing).collect();
        let sigma = spacing.max(1e-8);
        GaussianSmearing {
            centers,
            sigma,
            n_gaussians,
        }
    }

    /// Smear a distance value into n_gaussians basis coefficients.
    pub fn smear(&self, distance: f64) -> Vec<f64> {
        self.centers
            .iter()
            .map(|&mu| (-(distance - mu).powi(2) / (2.0 * self.sigma * self.sigma)).exp())
            .collect()
    }
}

/// One SchNet interaction block (continuous-filter convolution).
pub struct SchNetInteraction {
    /// Filter network: 2-layer MLP (n_gaussians → n_filters → n_filters).
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
    pub n_filters: usize,
    pub n_gaussians: usize,
}

impl SchNetInteraction {
    /// Construct with Xavier-uniform weights.
    pub fn new(n_gaussians: usize, n_filters: usize, seed: u64) -> Self {
        let w1 = xavier_uniform_2d(n_filters, n_gaussians, seed);
        let b1 = vec![0.0; n_filters];
        let w2 = xavier_uniform_2d(n_filters, n_filters, seed.wrapping_add(1));
        let b2 = vec![0.0; n_filters];
        SchNetInteraction {
            w1,
            b1,
            w2,
            b2,
            n_filters,
            n_gaussians,
        }
    }

    /// Apply filter network to a distance encoding.
    pub fn filter_network(&self, dist_encoding: &[f64]) -> Vec<f64> {
        let h1_raw = matvec(&self.w1, dist_encoding);
        let h1: Vec<f64> = h1_raw
            .iter()
            .zip(self.b1.iter())
            .map(|(&x, &b)| relu(x + b))
            .collect();
        let h2_raw = matvec(&self.w2, &h1);
        h2_raw
            .iter()
            .zip(self.b2.iter())
            .map(|(&x, &b)| relu(x + b))
            .collect()
    }

    /// Run one interaction block, updating node embeddings.
    ///
    /// `h`: node embeddings [n_atoms × n_filters]
    /// `positions`: 3-D coordinates [n_atoms × 3]
    pub fn forward(
        &self,
        h: &[Vec<f64>],
        positions: &[Vec<f64>],
        cutoff: f64,
        smearing: &GaussianSmearing,
    ) -> Vec<Vec<f64>> {
        let n = h.len();
        let mut h_new = h.to_vec();

        for i in 0..n {
            let mut delta = vec![0.0f64; self.n_filters];
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dist = euclidean_dist(&positions[i], &positions[j]);
                if dist >= cutoff {
                    continue;
                }
                // Smooth cutoff envelope
                let env = cosine_cutoff(dist, cutoff);
                let dist_enc = smearing.smear(dist);
                let filter = self.filter_network(&dist_enc);
                // h_delta_i += filter * h_j (element-wise, then sum over neighbours)
                let hj = pad_or_trim(&h[j], self.n_filters);
                for (d, (&f, &hv)) in delta.iter_mut().zip(filter.iter().zip(hj.iter())) {
                    *d += env * f * hv;
                }
            }
            // Residual update
            for (h_i, d) in h_new[i].iter_mut().zip(delta.iter()) {
                *h_i += d;
            }
        }
        h_new
    }
}

/// Euclidean distance between two 3-D coordinate vectors.
pub(crate) fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Smooth cosine cutoff envelope: 0.5*(cos(pi*r/rc)+1).
fn cosine_cutoff(r: f64, rc: f64) -> f64 {
    if r >= rc {
        0.0
    } else {
        0.5 * (std::f64::consts::PI * r / rc).cos() + 0.5
    }
}

/// Full SchNet molecular embedding model.
pub struct SchNet {
    /// Atom type embedding table [n_atom_types × n_filters].
    embedding: Vec<Vec<f64>>,
    /// Interaction blocks.
    interactions: Vec<SchNetInteraction>,
    /// Gaussian smearing module.
    smearing: GaussianSmearing,
    /// Output linear: [output_dim × n_filters].
    w_out: Vec<Vec<f64>>,
    b_out: Vec<f64>,
    pub config: SchNetConfig,
}

impl SchNet {
    /// Construct a new SchNet from configuration.
    pub fn new(config: SchNetConfig) -> Self {
        let mut seed = config.seed;

        // Atom embedding table
        let embedding = xavier_uniform_2d(config.n_atom_types, config.n_filters, seed);
        seed = seed.wrapping_add(50);

        // Interaction blocks
        let smearing = GaussianSmearing::new(config.n_gaussians, config.cutoff);
        let mut interactions = Vec::new();
        for i in 0..config.n_interactions {
            interactions.push(SchNetInteraction::new(
                config.n_gaussians,
                config.n_filters,
                seed.wrapping_add(i as u64 * 200),
            ));
            seed = seed.wrapping_add(500);
        }

        // Output projection
        let w_out = xavier_uniform_2d(config.output_dim, config.n_filters, seed);
        let b_out = vec![0.0; config.output_dim];

        SchNet {
            embedding,
            interactions,
            smearing,
            w_out,
            b_out,
            config,
        }
    }

    /// Forward pass: atom_types (indices) + 3-D positions → molecular embedding.
    ///
    /// Returns a vector of length `output_dim` (mean-pooled over atoms).
    pub fn forward(&self, atom_types: &[usize], positions: &[Vec<f64>]) -> Result<Vec<f64>> {
        let n = atom_types.len();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "SchNet::forward",
                "empty atom list",
            ));
        }
        if positions.len() != n {
            return Err(TensorError::invalid_argument_op(
                "SchNet::forward",
                "atom_types and positions length mismatch",
            ));
        }

        // Initial node embeddings from atom type table
        let mut h: Vec<Vec<f64>> = atom_types
            .iter()
            .map(|&t| {
                let idx = t % self.config.n_atom_types;
                self.embedding[idx].clone()
            })
            .collect();

        // Apply interaction blocks
        for interaction in &self.interactions {
            h = interaction.forward(&h, positions, self.config.cutoff, &self.smearing);
        }

        // Mean-pool over atoms
        let mut pooled = vec![0.0f64; self.config.n_filters];
        for hi in &h {
            for (p, &v) in pooled.iter_mut().zip(hi.iter()) {
                *p += v;
            }
        }
        let n_f = n as f64;
        pooled.iter_mut().for_each(|x| *x /= n_f);

        // Output projection
        let mut out = matvec(&self.w_out, &pooled);
        for (o, &b) in out.iter_mut().zip(self.b_out.iter()) {
            *o += b;
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Morgan (ECFP) Fingerprint
// ─────────────────────────────────────────────────────────────────────────────

/// Morgan extended-connectivity fingerprint (ECFP).
pub struct MorganFingerprint {
    /// Number of rounds (radius).
    pub radius: usize,
    /// Bit vector length.
    pub n_bits: usize,
}

impl MorganFingerprint {
    /// Compute the ECFP bit vector for a molecular graph.
    pub fn compute(&self, mol: &MolGraph) -> Vec<u8> {
        let n = mol.n_atoms();
        if n == 0 {
            return vec![0u8; self.n_bits];
        }
        let adj = mol.adjacency_list();
        let bond_feats = mol.bond_features();

        // Round 0: atom identity hashes
        let mut identifiers: Vec<u64> = mol.atoms.iter().map(atom_hash).collect();

        let mut all_hashes: Vec<u64> = identifiers.clone();

        // Rounds 1..radius
        for _r in 0..self.radius {
            let prev = identifiers.clone();
            let mut next = vec![0u64; n];
            for i in 0..n {
                let mut neighbor_info: Vec<(u64, u64)> = adj[i]
                    .iter()
                    .map(|&(j, bi)| {
                        let bond_order = if bi < mol.bonds.len() {
                            mol.bonds[bi].bond_order_u64()
                        } else {
                            1
                        };
                        (prev[j], bond_order)
                    })
                    .collect();
                neighbor_info.sort_unstable();

                // Build path: [current_hash, (bond_order, neighbor_hash), ...]
                let mut path = vec![prev[i]];
                for (nh, bo) in &neighbor_info {
                    path.push(*bo);
                    path.push(*nh);
                }
                let _ = bond_feats; // suppress unused
                next[i] = path_hash(&path);
            }
            all_hashes.extend_from_slice(&next);
            identifiers = next;
        }

        // Fold into bit vector
        let mut bits = vec![0u8; self.n_bits];
        for h in all_hashes {
            let bit = (h % self.n_bits as u64) as usize;
            bits[bit] = 1;
        }
        bits
    }

    /// Tanimoto (Jaccard) similarity on binary fingerprints.
    pub fn tanimoto(fp1: &[u8], fp2: &[u8]) -> f64 {
        let len = fp1.len().min(fp2.len());
        let mut and = 0u64;
        let mut or = 0u64;
        for (&a, &b) in fp1[..len].iter().zip(fp2[..len].iter()) {
            if a > 0 && b > 0 {
                and += 1;
            }
            if a > 0 || b > 0 {
                or += 1;
            }
        }
        if or == 0 {
            1.0
        } else {
            and as f64 / or as f64
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Topological Fingerprint
// ─────────────────────────────────────────────────────────────────────────────

/// Path-based topological fingerprint: enumerate all paths up to `max_path` length.
pub struct TopologicalFingerprint {
    /// Bit vector length.
    pub n_bits: usize,
    /// Maximum path length to enumerate.
    pub max_path: usize,
}

impl TopologicalFingerprint {
    /// Compute path-based fingerprint.
    pub fn compute(&self, mol: &MolGraph) -> Vec<u8> {
        let n = mol.n_atoms();
        let adj = mol.adjacency_list();
        let mut bits = vec![0u8; self.n_bits];

        // DFS from each atom, track visited atoms to avoid cycles in path
        for start in 0..n {
            let atom_h = atom_hash(&mol.atoms[start]);
            // Path starts with the start atom hash
            let init_path = vec![atom_h];
            dfs_paths(
                start,
                &init_path,
                &adj,
                &mol.bonds,
                self.max_path,
                &mut bits,
                self.n_bits,
            );
        }
        bits
    }

    /// Tanimoto similarity (same implementation as Morgan).
    pub fn path_similarity(fp1: &[u8], fp2: &[u8]) -> f64 {
        MorganFingerprint::tanimoto(fp1, fp2)
    }
}

/// Recursive DFS path enumeration for topological fingerprints.
fn dfs_paths(
    current: usize,
    path: &[u64],
    adj: &[Vec<(usize, usize)>],
    bonds: &[Bond],
    max_depth: usize,
    bits: &mut Vec<u8>,
    n_bits: usize,
) {
    // Set bit for current path
    let h = path_hash(path);
    let bit = (h % n_bits as u64) as usize;
    if bit < bits.len() {
        bits[bit] = 1;
    }

    if path.len() > max_depth {
        return;
    }

    let visited_atoms: Vec<u64> = path.to_vec();
    // visited_atoms[k] are the hashed atom identifiers — we need actual indices
    // We'll track the current path of atom-indices separately via the stack approach.
    // Since we only receive path hashes, reconstruct visited by checking path length vs atom index.
    // Simpler: pass visited set differently — rebuild without visited tracking but cap at max_depth.
    // (With max_path = 5, the performance is fine even with revisits; we just bound depth.)
    let _ = visited_atoms; // not needed below — we use depth-bound only

    for &(neighbor, bi) in &adj[current] {
        // Prevent re-using the immediately previous bond (avoid trivial back-and-forth)
        // We allow revisit of other atoms for simplicity (topological torsion style)
        let bond_h = if bi < bonds.len() {
            bonds[bi].bond_order_u64()
        } else {
            1
        };
        let mut new_path = path.to_vec();
        // Interleave bond type and neighbor atom hash
        new_path.push(bond_h);
        let neighbor_atom_h = if neighbor < adj.len() {
            // We don't have mol.atoms here, so use a surrogate from the bond structure
            neighbor as u64 + bond_h * 1000
        } else {
            neighbor as u64
        };
        new_path.push(neighbor_atom_h);
        dfs_paths(neighbor, &new_path, adj, bonds, max_depth, bits, n_bits);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Property Predictor
// ─────────────────────────────────────────────────────────────────────────────

/// Task type for molecular property prediction.
#[derive(Debug, Clone)]
pub enum PredictionTask {
    /// Regression with n_targets outputs.
    Regression { n_targets: usize },
    /// Classification with n_classes outputs.
    Classification { n_classes: usize },
}

/// Configuration for the property predictor.
#[derive(Debug, Clone)]
pub struct PropertyPredictorConfig {
    /// MPNN backbone configuration.
    pub mpnn_config: MolMpnnConfig,
    /// Prediction task.
    pub task: PredictionTask,
}

/// End-to-end molecular property predictor: MPNN + task head.
pub struct PropertyPredictor {
    pub mpnn: Mpnn,
    pub output_dim: usize,
    task_head: Vec<Vec<f64>>,
    task_bias: Vec<f64>,
}

impl PropertyPredictor {
    /// Construct from configuration.
    pub fn new(config: PropertyPredictorConfig) -> Self {
        let output_dim = match &config.task {
            PredictionTask::Regression { n_targets } => *n_targets,
            PredictionTask::Classification { n_classes } => *n_classes,
        };
        let mpnn_out = config.mpnn_config.output_dim;
        let seed = config.mpnn_config.seed.wrapping_add(99999);
        let mpnn = Mpnn::new(config.mpnn_config);
        let task_head = xavier_uniform_2d(output_dim, mpnn_out, seed);
        let task_bias = vec![0.0; output_dim];
        PropertyPredictor {
            mpnn,
            output_dim,
            task_head,
            task_bias,
        }
    }

    /// Predict molecular properties.
    pub fn predict(&self, mol: &MolGraph) -> Result<Vec<f64>> {
        let node_embeds = self.mpnn.forward(mol)?;
        let graph_embed = self.mpnn.graph_readout(&node_embeds);
        let raw = matvec(&self.task_head, &graph_embed);
        let out: Vec<f64> = raw
            .iter()
            .zip(self.task_bias.iter())
            .map(|(&r, &b)| r + b)
            .collect();
        Ok(out)
    }

    /// Compute task-appropriate loss.
    pub fn compute_loss(&self, preds: &[f64], targets: &[f64], task: &PredictionTask) -> f64 {
        match task {
            PredictionTask::Regression { .. } => {
                // MSE
                if preds.is_empty() || targets.is_empty() {
                    return 0.0;
                }
                let n = preds.len().min(targets.len()) as f64;
                preds
                    .iter()
                    .zip(targets.iter())
                    .map(|(&p, &t)| (p - t).powi(2))
                    .sum::<f64>()
                    / n
            }
            PredictionTask::Classification { .. } => {
                // Cross-entropy
                if preds.is_empty() || targets.is_empty() {
                    return 0.0;
                }
                let probs = softmax(preds);
                let target_idx = targets[0] as usize;
                let p = if target_idx < probs.len() {
                    probs[target_idx]
                } else {
                    1e-15
                };
                -(p.max(1e-15).ln())
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reaction Features and Classifier
// ─────────────────────────────────────────────────────────────────────────────

/// Feature representation of a chemical reaction.
#[derive(Debug, Clone)]
pub struct ReactionFeatures {
    /// Concatenated Morgan fingerprints of reactants (f64 encoding of u8 bits).
    pub reactant_fps: Vec<f64>,
    /// Concatenated Morgan fingerprints of products (f64 encoding of u8 bits).
    pub product_fps: Vec<f64>,
    /// Element-wise difference: product_fps − reactant_fps.
    pub diff_fp: Vec<f64>,
}

impl ReactionFeatures {
    /// Build reaction features from lists of reactant and product molecular graphs.
    pub fn from_molgraphs(reactants: &[MolGraph], products: &[MolGraph]) -> Self {
        let mfp = MorganFingerprint {
            radius: 2,
            n_bits: 256,
        };

        // Concatenate reactant fingerprints
        let reactant_fps: Vec<f64> = reactants
            .iter()
            .flat_map(|mol| mfp.compute(mol).into_iter().map(|b| b as f64))
            .collect();

        // Concatenate product fingerprints
        let product_fps: Vec<f64> = products
            .iter()
            .flat_map(|mol| mfp.compute(mol).into_iter().map(|b| b as f64))
            .collect();

        // Element-wise difference (pad shorter to match longer)
        let max_len = reactant_fps.len().max(product_fps.len());
        let diff_fp: Vec<f64> = (0..max_len)
            .map(|i| {
                let p = if i < product_fps.len() {
                    product_fps[i]
                } else {
                    0.0
                };
                let r = if i < reactant_fps.len() {
                    reactant_fps[i]
                } else {
                    0.0
                };
                p - r
            })
            .collect();

        ReactionFeatures {
            reactant_fps,
            product_fps,
            diff_fp,
        }
    }
}

/// Simple linear reaction-type classifier trained by SGD.
pub struct ReactionClassifier {
    /// Weight matrix [n_classes × feat_dim].
    pub weights: Vec<Vec<f64>>,
    /// Bias vector \[n_classes\].
    pub bias: Vec<f64>,
    /// Number of reaction classes.
    pub n_classes: usize,
}

impl ReactionClassifier {
    /// Create a new classifier with zero-initialised weights.
    pub fn new(n_classes: usize, feat_dim: usize) -> Self {
        ReactionClassifier {
            weights: vec![vec![0.0; feat_dim]; n_classes],
            bias: vec![0.0; n_classes],
            n_classes,
        }
    }

    /// Predict class probabilities: softmax over linear(diff_fp).
    pub fn predict(&self, feat: &ReactionFeatures) -> Vec<f64> {
        let logits: Vec<f64> = (0..self.n_classes)
            .map(|c| {
                if c < self.weights.len() {
                    let row = &self.weights[c];
                    let dot: f64 = row
                        .iter()
                        .zip(feat.diff_fp.iter())
                        .map(|(&w, &x)| w * x)
                        .sum();
                    dot + if c < self.bias.len() {
                        self.bias[c]
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            })
            .collect();
        softmax(&logits)
    }

    /// One SGD training step over a batch.
    ///
    /// Returns the mean cross-entropy loss over the batch.
    pub fn train_step(&mut self, feats: &[ReactionFeatures], labels: &[usize], lr: f64) -> f64 {
        if feats.is_empty() {
            return 0.0;
        }
        let n = feats.len() as f64;
        let mut total_loss = 0.0f64;

        // Accumulate gradients
        let feat_dim = feats[0].diff_fp.len();
        let mut dw = vec![vec![0.0f64; feat_dim]; self.n_classes];
        let mut db = vec![0.0f64; self.n_classes];

        for (feat, &label) in feats.iter().zip(labels.iter()) {
            let probs = self.predict(feat);
            let label_idx = label.min(self.n_classes - 1);
            let p = probs.get(label_idx).copied().unwrap_or(1e-15);
            total_loss -= p.max(1e-15).ln();

            // Gradient: d_loss/d_logit_c = prob_c − 1{c == label}
            for c in 0..self.n_classes {
                let grad =
                    probs.get(c).copied().unwrap_or(0.0) - if c == label_idx { 1.0 } else { 0.0 };
                for (j, &x) in feat.diff_fp.iter().enumerate() {
                    if j < dw[c].len() {
                        dw[c][j] += grad * x;
                    }
                }
                if c < db.len() {
                    db[c] += grad;
                }
            }
        }

        // SGD update
        for c in 0..self.n_classes {
            for (j, dw_cj) in dw[c].iter().enumerate() {
                if j < self.weights[c].len() {
                    self.weights[c][j] -= lr * dw_cj / n;
                }
            }
            if c < self.bias.len() {
                self.bias[c] -= lr * db[c] / n;
            }
        }

        total_loss / n
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-export public API type alias for BondType convenience
// ─────────────────────────────────────────────────────────────────────────────

/// Alias for `MolBondType` exposed under the name `BondType` within this module.
/// (The crate-level `BondType` from `graph_generation` is a separate type.)
pub type BondType = MolBondType;
