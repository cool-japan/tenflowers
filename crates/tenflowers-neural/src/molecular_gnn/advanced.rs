//! Advanced molecular machine learning algorithms.
//!
//! This module provides cutting-edge algorithms for:
//!
//! - **3D Molecular Geometry Learning**: [`DimeNetLayer`] (Gasteiger 2020 directional message
//!   passing with Bessel basis + spherical harmonics), [`ComENetLayer`] (torsion-angle-enhanced
//!   messages), [`EquivariantMolNet`] (EGNN-style E(3)-equivariant network).
//!
//! - **Molecular Generation**: [`JunctionTreeVae`] (Jin 2018 tree-decomposed VAE),
//!   [`GraphVae`] (Simonovsky 2018 probabilistic graph generation),
//!   [`MolecularFlowModel`] (RealNVP normalizing flow on fingerprints),
//!   [`MolDruglikenessFilter`] (Lipinski RO5 + Veber rules).
//!
//! - **Molecular Property Prediction Ensemble**: [`AttentiveFp`] (Xiong 2019 attentive
//!   fingerprint), [`MolBert`] (BERT masked-atom pretraining on SMILES),
//!   [`MultiTaskMolNet`] (multi-property prediction), [`UncertaintyMolPredictor`] (MC Dropout).
//!
//! - **Reaction Prediction**: [`ReactionGraph`] (atom-mapped reaction hypergraph),
//!   [`LocalMapper`] (template-free atom mapping), [`RetrosynthesisPredictor`]
//!   (template-based retrosynthesis), [`ReactionYieldPredictor`] (yield prediction).
//!
//! All fallible operations return `Result<_, TensorError>`.
//! No `unsafe` code; no `unwrap()` calls.

use super::{matvec, pad_or_trim, relu, sigmoid, softmax, xavier_uniform_2d,
            MolGraph, MolMpnnConfig, MorganFingerprint, Mpnn};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// 3D Molecular Geometry — DimeNet-style
// ─────────────────────────────────────────────────────────────────────────────

/// 3D molecular conformer: atom positions plus bond/angle topology.
#[derive(Debug, Clone)]
pub struct MolConformer {
    /// 3D Cartesian coordinates for each atom (Ångström).
    pub positions: Vec<[f64; 3]>,
    /// Edges as (i, j) pairs — directed.
    pub edges: Vec<(usize, usize)>,
    /// Atomic numbers for each atom.
    pub atom_types: Vec<u8>,
}

impl MolConformer {
    /// Construct a new conformer.
    pub fn new(positions: Vec<[f64; 3]>, edges: Vec<(usize, usize)>, atom_types: Vec<u8>) -> Self {
        MolConformer { positions, edges, atom_types }
    }

    /// Euclidean distance between atoms `i` and `j`.
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        let a = &self.positions[i];
        let b = &self.positions[j];
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    /// Bond angle ∠(i–j–k) in radians (angle at atom j).
    pub fn bond_angle(&self, i: usize, j: usize, k: usize) -> f64 {
        let pj = &self.positions[j];
        let v1 = [
            self.positions[i][0] - pj[0],
            self.positions[i][1] - pj[1],
            self.positions[i][2] - pj[2],
        ];
        let v2 = [
            self.positions[k][0] - pj[0],
            self.positions[k][1] - pj[1],
            self.positions[k][2] - pj[2],
        ];
        let dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
        let n1 = (v1[0].powi(2) + v1[1].powi(2) + v1[2].powi(2)).sqrt().max(1e-10);
        let n2 = (v2[0].powi(2) + v2[1].powi(2) + v2[2].powi(2)).sqrt().max(1e-10);
        (dot / (n1 * n2)).clamp(-1.0, 1.0).acos()
    }

    /// Torsion angle ∠(i–j–k–l) in radians (dihedral).
    pub fn torsion_angle(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        let p = [
            &self.positions[i],
            &self.positions[j],
            &self.positions[k],
            &self.positions[l],
        ];
        // Vectors along backbone
        let b1 = [p[1][0]-p[0][0], p[1][1]-p[0][1], p[1][2]-p[0][2]];
        let b2 = [p[2][0]-p[1][0], p[2][1]-p[1][1], p[2][2]-p[1][2]];
        let b3 = [p[3][0]-p[2][0], p[3][1]-p[2][1], p[3][2]-p[2][2]];
        let cross = |a: [f64; 3], b: [f64; 3]| -> [f64; 3] {
            [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
        };
        let dot3 = |a: [f64; 3], b: [f64; 3]| a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
        let n1 = cross(b1, b2);
        let n2 = cross(b2, b3);
        let m = cross(n1, b2);
        let nb2 = dot3(b2, b2).sqrt().max(1e-10);
        let b2n = [b2[0]/nb2, b2[1]/nb2, b2[2]/nb2];
        let m2 = cross(n1, b2n);
        dot3(m2, n2).atan2(dot3(n1, n2))
    }

    /// Generate a random synthetic conformer for testing.
    pub fn random_stub(n_atoms: usize, rng: &mut StdRng) -> Self {
        let positions: Vec<[f64; 3]> = (0..n_atoms)
            .map(|_| {
                [
                    rng.random::<f64>() * 5.0,
                    rng.random::<f64>() * 5.0,
                    rng.random::<f64>() * 5.0,
                ]
            })
            .collect();
        let atom_types: Vec<u8> = (0..n_atoms).map(|_| [6u8, 7, 8, 16][(rng.random::<u64>() as usize) % 4]).collect();
        let mut edges = Vec::new();
        for i in 0..n_atoms.saturating_sub(1) {
            edges.push((i, i + 1));
            edges.push((i + 1, i));
        }
        MolConformer { positions, edges, atom_types }
    }
}

/// Bessel basis functions for DimeNet-style distance encoding.
///
/// Uses envelope-damped sinc functions: e(r) · sin(nπr/r_cut) / r.
pub struct BesselBasis {
    /// Cutoff radius.
    pub cutoff: f64,
    /// Number of basis functions.
    pub n_basis: usize,
}

impl BesselBasis {
    /// Create a new Bessel basis.
    pub fn new(n_basis: usize, cutoff: f64) -> Self {
        BesselBasis { cutoff, n_basis }
    }

    /// Smooth polynomial envelope: 1 - (n+1)(n+2)/2 * r^n + n(n+2) * r^(n+1) - n(n+1)/2 * r^(n+2).
    fn envelope(&self, r: f64) -> f64 {
        let p = 5.0_f64;
        let d = r / self.cutoff;
        if d >= 1.0 {
            return 0.0;
        }
        1.0 - (p + 1.0) * (p + 2.0) / 2.0 * d.powf(p)
            + p * (p + 2.0) * d.powf(p + 1.0)
            - p * (p + 1.0) / 2.0 * d.powf(p + 2.0)
    }

    /// Encode a distance into `n_basis` Bessel features.
    pub fn encode(&self, r: f64) -> Vec<f64> {
        let env = self.envelope(r);
        (1..=self.n_basis)
            .map(|n| {
                let freq = n as f64 * std::f64::consts::PI / self.cutoff;
                if r < 1e-10 {
                    // Limit as r→0: sinc → 1
                    env * freq
                } else {
                    env * (freq * r).sin() / r
                }
            })
            .collect()
    }
}

/// DimeNet interaction layer: directional message passing with Bessel + spherical harmonics.
///
/// Implements the core of Gasteiger et al. 2020 (simplified to l=0 scalar + l=1 directional).
pub struct DimeNetLayer {
    /// Bessel basis for radial encoding.
    pub basis: BesselBasis,
    /// Weight matrices for message transformation.
    w_rbf: Vec<Vec<f64>>,
    w_dir: Vec<Vec<f64>>,
    w_out: Vec<Vec<f64>>,
    b_out: Vec<f64>,
    /// Hidden embedding dimension.
    pub hidden_dim: usize,
}

impl DimeNetLayer {
    /// Construct with Xavier-uniform weights.
    pub fn new(n_basis: usize, hidden_dim: usize, cutoff: f64, seed: u64) -> Self {
        let basis = BesselBasis::new(n_basis, cutoff);
        let w_rbf = xavier_uniform_2d(hidden_dim, n_basis, seed);
        // l=1 directional: 3 spherical harmonics (x, y, z)
        let w_dir = xavier_uniform_2d(hidden_dim, 3, seed.wrapping_add(1));
        let w_out = xavier_uniform_2d(hidden_dim, hidden_dim * 2, seed.wrapping_add(2));
        let b_out = vec![0.0; hidden_dim];
        DimeNetLayer { basis, w_rbf, w_dir, w_out, b_out, hidden_dim }
    }

    /// Unit direction vector from atom i to atom j.
    fn unit_direction(conf: &MolConformer, i: usize, j: usize) -> [f64; 3] {
        let pi = &conf.positions[i];
        let pj = &conf.positions[j];
        let dx = pj[0] - pi[0];
        let dy = pj[1] - pi[1];
        let dz = pj[2] - pi[2];
        let norm = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-10);
        [dx / norm, dy / norm, dz / norm]
    }

    /// Forward pass: embed atoms using directional messages.
    ///
    /// Returns node embeddings of shape [n_atoms × hidden_dim].
    pub fn forward(&self, conf: &MolConformer) -> Result<Vec<Vec<f64>>> {
        let n = conf.positions.len();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "DimeNetLayer::forward",
                "empty conformer",
            ));
        }

        // Build adjacency
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(i, j) in &conf.edges {
            if i < n && j < n {
                adj[i].push(j);
            }
        }

        let mut h: Vec<Vec<f64>> = vec![vec![0.0; self.hidden_dim]; n];

        for i in 0..n {
            let mut msg_sum = vec![0.0f64; self.hidden_dim];
            let mut count = 0usize;

            for &j in &adj[i] {
                let r = conf.distance(i, j);
                if r >= self.basis.cutoff {
                    continue;
                }

                // Radial component
                let rbf = self.basis.encode(r);
                let radial = matvec(&self.w_rbf, &rbf);

                // Directional component (l=1 spherical harmonics ≡ unit direction)
                let dir = Self::unit_direction(conf, i, j);
                let directional = matvec(&self.w_dir, &dir);

                // Combine
                let mut combined = Vec::with_capacity(self.hidden_dim * 2);
                combined.extend_from_slice(&radial);
                combined.extend_from_slice(&directional);

                let msg = matvec(&self.w_out, &combined);
                for (s, &m) in msg_sum.iter_mut().zip(msg.iter()) {
                    *s += relu(m);
                }
                count += 1;
            }

            if count > 0 {
                let c = count as f64;
                for (hi, (&b, s)) in h[i].iter_mut().zip(self.b_out.iter().zip(msg_sum.iter())) {
                    *hi = relu(s / c + b);
                }
            }
        }

        Ok(h)
    }
}

/// ComENet layer: completeness-enhanced message passing with torsion angles.
///
/// Extends DimeNet by incorporating torsion angles for full geometric completeness.
pub struct ComENetLayer {
    /// Bessel basis for distances.
    pub basis: BesselBasis,
    w_dist: Vec<Vec<f64>>,
    w_angle: Vec<Vec<f64>>,
    w_torsion: Vec<Vec<f64>>,
    w_update: Vec<Vec<f64>>,
    b_update: Vec<f64>,
    /// Output embedding dimension.
    pub hidden_dim: usize,
}

impl ComENetLayer {
    /// Construct a new ComENet layer.
    pub fn new(n_basis: usize, hidden_dim: usize, cutoff: f64, seed: u64) -> Self {
        let basis = BesselBasis::new(n_basis, cutoff);
        let w_dist = xavier_uniform_2d(hidden_dim, n_basis, seed);
        // Angle encoded as [cos(θ), sin(θ)] → 2 features
        let w_angle = xavier_uniform_2d(hidden_dim, 2, seed.wrapping_add(1));
        // Torsion encoded as [cos(φ), sin(φ)] → 2 features
        let w_torsion = xavier_uniform_2d(hidden_dim, 2, seed.wrapping_add(2));
        let w_update = xavier_uniform_2d(hidden_dim, hidden_dim, seed.wrapping_add(3));
        let b_update = vec![0.0; hidden_dim];
        ComENetLayer { basis, w_dist, w_angle, w_torsion, w_update, b_update, hidden_dim }
    }

    /// Forward pass incorporating distances, bond angles, and torsion angles.
    pub fn forward(&self, conf: &MolConformer) -> Result<Vec<Vec<f64>>> {
        let n = conf.positions.len();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "ComENetLayer::forward",
                "empty conformer",
            ));
        }

        // Adjacency list
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(i, j) in &conf.edges {
            if i < n && j < n {
                adj[i].push(j);
            }
        }

        let mut h: Vec<Vec<f64>> = vec![vec![0.0; self.hidden_dim]; n];

        for i in 0..n {
            let neighbors_i = adj[i].clone();
            let mut agg = vec![0.0f64; self.hidden_dim];
            let mut count = 0usize;

            for &j in &neighbors_i {
                let r_ij = conf.distance(i, j);
                if r_ij >= self.basis.cutoff {
                    continue;
                }

                // Radial features
                let rbf = self.basis.encode(r_ij);
                let dist_feat = matvec(&self.w_dist, &rbf);

                // Bond angle: use first other neighbor k≠j for angle(j–i–k)
                let angle_feat = neighbors_i.iter()
                    .find(|&&k| k != j)
                    .map(|&k| {
                        let theta = conf.bond_angle(j, i, k);
                        let enc = [theta.cos(), theta.sin()];
                        matvec(&self.w_angle, &enc)
                    })
                    .unwrap_or_else(|| vec![0.0; self.hidden_dim]);

                // Torsion: use first neighbor l of j (l≠i)
                let torsion_feat = adj[j].iter()
                    .find(|&&l| l != i)
                    .map(|&l| {
                        let phi = if n > 3 && l < n {
                            // Pick an anchor atom a≠j, i
                            let a = (0..n).find(|&x| x != j && x != i && x != l).unwrap_or(0);
                            conf.torsion_angle(a, i, j, l)
                        } else {
                            0.0
                        };
                        let enc = [phi.cos(), phi.sin()];
                        matvec(&self.w_torsion, &enc)
                    })
                    .unwrap_or_else(|| vec![0.0; self.hidden_dim]);

                // Sum features element-wise
                for k in 0..self.hidden_dim {
                    let val = dist_feat.get(k).copied().unwrap_or(0.0)
                        + angle_feat.get(k).copied().unwrap_or(0.0)
                        + torsion_feat.get(k).copied().unwrap_or(0.0);
                    agg[k] += relu(val);
                }
                count += 1;
            }

            if count > 0 {
                let c = count as f64;
                for a in agg.iter_mut() {
                    *a /= c;
                }
                let updated = matvec(&self.w_update, &agg);
                for (hi, (&b, u)) in h[i].iter_mut().zip(self.b_update.iter().zip(updated.iter())) {
                    *hi = relu(u + b);
                }
            }
        }

        Ok(h)
    }
}

/// E(3)-equivariant molecular network for energy and force prediction (EGNN-style).
///
/// Implements simplified EGNN (Satorras 2021): node features + coordinate updates.
pub struct EquivariantMolNet {
    w_msg: Vec<Vec<f64>>,
    b_msg: Vec<f64>,
    w_coord: Vec<Vec<f64>>,
    w_node: Vec<Vec<f64>>,
    b_node: Vec<f64>,
    w_energy: Vec<Vec<f64>>,
    b_energy: Vec<f64>,
    /// Number of EGNN layers.
    pub n_layers: usize,
    /// Feature dimension.
    pub feat_dim: usize,
}

impl EquivariantMolNet {
    /// Construct an equivariant molecular net.
    pub fn new(feat_dim: usize, n_layers: usize, seed: u64) -> Self {
        // Message: [feat_i ‖ feat_j ‖ dist²] → feat_dim
        let w_msg = xavier_uniform_2d(feat_dim, feat_dim * 2 + 1, seed);
        let b_msg = vec![0.0; feat_dim];
        // Coordinate update: feat_dim → 1 (scalar weight for displacement)
        let w_coord = xavier_uniform_2d(1, feat_dim, seed.wrapping_add(1));
        // Node update: [feat ‖ agg] → feat_dim
        let w_node = xavier_uniform_2d(feat_dim, feat_dim * 2, seed.wrapping_add(2));
        let b_node = vec![0.0; feat_dim];
        // Energy readout: feat_dim → 1
        let w_energy = xavier_uniform_2d(1, feat_dim, seed.wrapping_add(3));
        let b_energy = vec![0.0; 1];
        EquivariantMolNet { w_msg, b_msg, w_coord, w_node, b_node, w_energy, b_energy, n_layers, feat_dim }
    }

    /// Forward pass: predict per-atom energies and optionally forces.
    ///
    /// Returns `(total_energy, per_atom_energies)`.
    pub fn forward(&self, conf: &MolConformer, _atom_feat_dim: usize) -> Result<(f64, Vec<f64>)> {
        let n = conf.positions.len();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "EquivariantMolNet::forward",
                "empty conformer",
            ));
        }

        // Initialize features from atom type one-hot (padded/trimmed to feat_dim)
        let mut h: Vec<Vec<f64>> = conf.atom_types.iter().map(|&at| {
            let mut f = vec![0.0; self.feat_dim];
            let idx = (at as usize).min(self.feat_dim - 1);
            f[idx] = 1.0;
            f
        }).collect();

        let mut coords: Vec<[f64; 3]> = conf.positions.clone();

        // Build full adjacency (all pairs within some radius — use all edges)
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(i, j) in &conf.edges {
            if i < n && j < n {
                adj[i].push(j);
            }
        }

        for _layer in 0..self.n_layers {
            let h_prev = h.clone();
            let coords_prev = coords.clone();
            let mut coord_delta: Vec<[f64; 3]> = vec![[0.0; 3]; n];

            for i in 0..n {
                let mut agg = vec![0.0f64; self.feat_dim];
                let mut n_nbrs = 0usize;

                for &j in &adj[i] {
                    let dp = [
                        coords_prev[i][0] - coords_prev[j][0],
                        coords_prev[i][1] - coords_prev[j][1],
                        coords_prev[i][2] - coords_prev[j][2],
                    ];
                    let dist_sq = dp[0].powi(2) + dp[1].powi(2) + dp[2].powi(2);

                    // Message
                    let hi = &h_prev[i];
                    let hj = &h_prev[j];
                    let mut inp = Vec::with_capacity(self.feat_dim * 2 + 1);
                    inp.extend_from_slice(hi);
                    inp.extend_from_slice(hj);
                    inp.push(dist_sq);

                    let inp_padded = pad_or_trim(&inp, self.w_msg[0].len());
                    let m_raw = matvec(&self.w_msg, &inp_padded);
                    let m: Vec<f64> = m_raw.iter().zip(self.b_msg.iter()).map(|(&x, &b)| relu(x + b)).collect();

                    // Coordinate update weight (scalar)
                    let w = matvec(&self.w_coord, &m);
                    let w_scalar = w.first().copied().unwrap_or(0.0).tanh();
                    coord_delta[i][0] += w_scalar * dp[0];
                    coord_delta[i][1] += w_scalar * dp[1];
                    coord_delta[i][2] += w_scalar * dp[2];

                    for (a, &mv) in agg.iter_mut().zip(m.iter()) {
                        *a += mv;
                    }
                    n_nbrs += 1;
                }

                if n_nbrs > 0 {
                    let c = n_nbrs as f64;
                    for a in agg.iter_mut() { *a /= c; }
                }

                // Node update
                let mut cat = Vec::with_capacity(self.feat_dim * 2);
                cat.extend_from_slice(&h_prev[i]);
                cat.extend_from_slice(&agg);
                let upd_raw = matvec(&self.w_node, &cat);
                h[i] = upd_raw.iter().zip(self.b_node.iter()).map(|(&x, &b)| relu(x + b)).collect();
            }

            // Apply coordinate updates (small step to maintain near-E3 equivariance)
            for i in 0..n {
                coords[i][0] += 0.01 * coord_delta[i][0];
                coords[i][1] += 0.01 * coord_delta[i][1];
                coords[i][2] += 0.01 * coord_delta[i][2];
            }
        }

        // Per-atom energy readout
        let per_atom: Vec<f64> = h.iter().map(|hi| {
            let e_raw = matvec(&self.w_energy, hi);
            e_raw.first().copied().unwrap_or(0.0) + self.b_energy.first().copied().unwrap_or(0.0)
        }).collect();

        let total: f64 = per_atom.iter().sum();
        Ok((total, per_atom))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Molecular Generation
// ─────────────────────────────────────────────────────────────────────────────

/// Fragment vocabulary entry for junction tree VAE.
#[derive(Debug, Clone)]
pub struct JtVocabEntry {
    /// Fragment identifier (e.g., ring index or chain motif).
    pub id: usize,
    /// Feature vector for this fragment.
    pub features: Vec<f64>,
}

/// Junction Tree VAE for molecular generation (Jin 2018).
///
/// Decomposes molecules into ring/chain vocabulary fragments, encodes a tree
/// structure, and decodes via beam search to generate valid molecules.
pub struct JunctionTreeVae {
    /// Vocabulary of molecular fragments.
    pub vocab: Vec<JtVocabEntry>,
    /// Latent dimension.
    pub latent_dim: usize,
    /// Fragment feature dimension.
    pub frag_dim: usize,
    // Encoder: fragment_features → mean/logvar
    w_enc_mean: Vec<Vec<f64>>,
    w_enc_logvar: Vec<Vec<f64>>,
    // Tree decoder: latent → fragment logits
    w_dec_tree: Vec<Vec<f64>>,
    b_dec_tree: Vec<f64>,
    // Graph decoder: latent + fragment → atom/bond logits
    w_dec_graph: Vec<Vec<f64>>,
    b_dec_graph: Vec<f64>,
}

impl JunctionTreeVae {
    /// Construct a Junction Tree VAE.
    pub fn new(vocab_size: usize, frag_dim: usize, latent_dim: usize, seed: u64) -> Self {
        let vocab: Vec<JtVocabEntry> = (0..vocab_size).map(|i| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
            JtVocabEntry {
                id: i,
                features: (0..frag_dim).map(|_| rng.random::<f64>() * 0.1).collect(),
            }
        }).collect();

        let w_enc_mean = xavier_uniform_2d(latent_dim, frag_dim, seed);
        let w_enc_logvar = xavier_uniform_2d(latent_dim, frag_dim, seed.wrapping_add(1));
        let w_dec_tree = xavier_uniform_2d(vocab_size, latent_dim, seed.wrapping_add(2));
        let b_dec_tree = vec![0.0; vocab_size];
        let w_dec_graph = xavier_uniform_2d(frag_dim, latent_dim, seed.wrapping_add(3));
        let b_dec_graph = vec![0.0; frag_dim];

        JunctionTreeVae { vocab, latent_dim, frag_dim, w_enc_mean, w_enc_logvar,
                          w_dec_tree, b_dec_tree, w_dec_graph, b_dec_graph }
    }

    /// Encode a molecule (represented as its fragment feature vector) to (mean, logvar).
    pub fn encode(&self, frag_feat: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        let f = pad_or_trim(frag_feat, self.frag_dim);
        let mean = matvec(&self.w_enc_mean, &f);
        let logvar = matvec(&self.w_enc_logvar, &f);
        Ok((mean, logvar))
    }

    /// Reparameterization trick: z = mean + eps * exp(0.5*logvar).
    pub fn reparameterize(&self, mean: &[f64], logvar: &[f64], rng: &mut StdRng) -> Vec<f64> {
        mean.iter().zip(logvar.iter()).map(|(&m, &lv)| {
            let eps: f64 = rng.random::<f64>() * 2.0 - 1.0; // approximate N(0,1)
            m + eps * (0.5 * lv).exp()
        }).collect()
    }

    /// Decode latent vector to fragment sequence via beam search (beam_width=3).
    pub fn decode_tree(&self, z: &[f64], max_steps: usize) -> Vec<usize> {
        let z_padded = pad_or_trim(z, self.latent_dim);
        let logits_raw = matvec(&self.w_dec_tree, &z_padded);
        let logits: Vec<f64> = logits_raw.iter().zip(self.b_dec_tree.iter())
            .map(|(&x, &b)| x + b).collect();
        let probs = softmax(&logits);

        // Greedy beam search (simplified — top-3 at each step)
        let beam_width = 3.min(self.vocab.len());
        let mut beam: Vec<(Vec<usize>, f64)> = {
            let mut indexed: Vec<(usize, f64)> = probs.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed[..beam_width].iter().map(|&(idx, p)| (vec![idx], p.ln())).collect()
        };

        for _step in 1..max_steps {
            let mut next_beam: Vec<(Vec<usize>, f64)> = Vec::new();
            for (seq, score) in &beam {
                // Use last fragment to condition next step (simplified: just re-score)
                let last_idx = seq.last().copied().unwrap_or(0);
                let cond_feat = self.vocab.get(last_idx)
                    .map(|e| e.features.clone())
                    .unwrap_or_else(|| vec![0.0; self.frag_dim]);
                let combined: Vec<f64> = z_padded.iter().zip(cond_feat.iter())
                    .map(|(&a, &b)| a + b * 0.1).collect();
                let combined_padded = pad_or_trim(&combined, self.latent_dim);
                let step_logits_raw = matvec(&self.w_dec_tree, &combined_padded);
                let step_logits: Vec<f64> = step_logits_raw.iter().zip(self.b_dec_tree.iter())
                    .map(|(&x, &b)| x + b).collect();
                let step_probs = softmax(&step_logits);
                let mut indexed: Vec<(usize, f64)> = step_probs.iter().cloned().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for &(idx, p) in indexed[..beam_width.min(indexed.len())].iter() {
                    let mut new_seq = seq.clone();
                    new_seq.push(idx);
                    next_beam.push((new_seq, score + p.max(1e-15).ln()));
                }
            }
            next_beam.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            next_beam.truncate(beam_width);
            beam = next_beam;
        }

        beam.into_iter().next().map(|(seq, _)| seq).unwrap_or_default()
    }

    /// KL divergence: KL(N(mean, exp(logvar)) ‖ N(0, I)).
    pub fn kl_loss(&self, mean: &[f64], logvar: &[f64]) -> f64 {
        mean.iter().zip(logvar.iter())
            .map(|(&m, &lv)| -0.5 * (1.0 + lv - m.powi(2) - lv.exp()))
            .sum::<f64>()
    }
}

/// Graph VAE for molecular generation (Simonovsky 2018).
///
/// Generates adjacency matrix and node features via a probabilistic VAE
/// with graph matching loss.
pub struct GraphVae {
    /// Maximum number of atoms.
    pub max_atoms: usize,
    /// Node feature dimension.
    pub node_dim: usize,
    /// Latent dimension.
    pub latent_dim: usize,
    w_enc_mean: Vec<Vec<f64>>,
    w_enc_logvar: Vec<Vec<f64>>,
    w_dec_adj: Vec<Vec<f64>>,
    w_dec_feat: Vec<Vec<f64>>,
    b_dec_feat: Vec<f64>,
}

impl GraphVae {
    /// Construct a new Graph VAE.
    pub fn new(max_atoms: usize, node_dim: usize, latent_dim: usize, seed: u64) -> Self {
        let input_dim = max_atoms * node_dim;
        let w_enc_mean = xavier_uniform_2d(latent_dim, input_dim.max(1), seed);
        let w_enc_logvar = xavier_uniform_2d(latent_dim, input_dim.max(1), seed.wrapping_add(1));
        let w_dec_adj = xavier_uniform_2d(max_atoms * max_atoms, latent_dim, seed.wrapping_add(2));
        let w_dec_feat = xavier_uniform_2d(max_atoms * node_dim, latent_dim, seed.wrapping_add(3));
        let b_dec_feat = vec![0.0; max_atoms * node_dim];
        GraphVae { max_atoms, node_dim, latent_dim, w_enc_mean, w_enc_logvar, w_dec_adj, w_dec_feat, b_dec_feat }
    }

    /// Encode a molecule's feature matrix (flattened) to (mean, logvar).
    pub fn encode(&self, node_feats_flat: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let f = pad_or_trim(node_feats_flat, self.max_atoms * self.node_dim);
        let mean = matvec(&self.w_enc_mean, &f);
        let logvar = matvec(&self.w_enc_logvar, &f);
        (mean, logvar)
    }

    /// Decode latent vector to adjacency probabilities + node features.
    ///
    /// Returns `(adj_probs [max_atoms²], node_features_flat [max_atoms × node_dim])`.
    pub fn decode(&self, z: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let z_p = pad_or_trim(z, self.latent_dim);
        let adj_raw = matvec(&self.w_dec_adj, &z_p);
        let adj_probs: Vec<f64> = adj_raw.iter().map(|&x| sigmoid(x)).collect();
        let feat_raw = matvec(&self.w_dec_feat, &z_p);
        let node_feats: Vec<f64> = feat_raw.iter().zip(self.b_dec_feat.iter())
            .map(|(&x, &b)| relu(x + b)).collect();
        (adj_probs, node_feats)
    }

    /// Sample a binary adjacency matrix from probabilities (Bernoulli).
    pub fn sample_adjacency(&self, adj_probs: &[f64], rng: &mut StdRng) -> Vec<Vec<bool>> {
        let mut adj = vec![vec![false; self.max_atoms]; self.max_atoms];
        for i in 0..self.max_atoms {
            for j in (i + 1)..self.max_atoms {
                let idx = i * self.max_atoms + j;
                let p = adj_probs.get(idx).copied().unwrap_or(0.0);
                if rng.random::<f64>() < p {
                    adj[i][j] = true;
                    adj[j][i] = true;
                }
            }
        }
        adj
    }

    /// Graph matching loss: Frobenius norm between target and reconstructed adjacency.
    pub fn graph_matching_loss(&self, target_adj: &[Vec<bool>], pred_probs: &[f64]) -> f64 {
        let mut loss = 0.0f64;
        let n = target_adj.len().min(self.max_atoms);
        for i in 0..n {
            for j in 0..n {
                let t = if target_adj[i].get(j).copied().unwrap_or(false) { 1.0 } else { 0.0 };
                let p = pred_probs.get(i * self.max_atoms + j).copied().unwrap_or(0.0).clamp(1e-7, 1.0 - 1e-7);
                loss -= t * p.ln() + (1.0 - t) * (1.0 - p).ln();
            }
        }
        loss
    }
}

/// Normalizing flow on molecular fingerprint space (RealNVP).
///
/// Learns an invertible mapping from fingerprints to a Gaussian latent space.
pub struct MolecularFlowModel {
    /// Number of affine coupling layers.
    pub n_layers: usize,
    /// Fingerprint dimension.
    pub fp_dim: usize,
    // Scale networks per coupling layer
    scale_nets: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    // Translation networks per coupling layer
    translate_nets: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
}

impl MolecularFlowModel {
    /// Construct a RealNVP flow model on molecular fingerprints.
    pub fn new(fp_dim: usize, n_layers: usize, hidden_dim: usize, seed: u64) -> Self {
        let half = fp_dim / 2;
        let mut scale_nets = Vec::new();
        let mut translate_nets = Vec::new();
        for i in 0..n_layers {
            let s = xavier_uniform_2d(half, hidden_dim, seed.wrapping_add(i as u64 * 100));
            let sb = vec![0.0; half];
            scale_nets.push((s, sb));
            let t = xavier_uniform_2d(half, hidden_dim, seed.wrapping_add(i as u64 * 100 + 50));
            let tb = vec![0.0; half];
            translate_nets.push((t, tb));
        }
        MolecularFlowModel { n_layers, fp_dim, scale_nets, translate_nets }
    }

    /// Forward pass: fingerprint → latent (with log-determinant).
    ///
    /// Returns `(z, log_det_jacobian)`.
    pub fn forward(&self, fp: &[f64]) -> Result<(Vec<f64>, f64)> {
        if fp.len() < 2 {
            return Err(TensorError::invalid_argument_op(
                "MolecularFlowModel::forward",
                "fingerprint too short",
            ));
        }
        let half = self.fp_dim / 2;
        let mut x = pad_or_trim(fp, self.fp_dim);
        let mut log_det = 0.0f64;

        for layer in 0..self.n_layers {
            // Split x into x1, x2
            let x1 = x[..half].to_vec();
            let x2 = x[half..].to_vec();

            // Scale and translate conditioned on x1
            let (ref ws, ref bs) = self.scale_nets[layer];
            let (ref wt, ref bt) = self.translate_nets[layer];
            let h1 = pad_or_trim(&x1, ws.first().map(|r| r.len()).unwrap_or(1));
            let s_raw = matvec(ws, &h1);
            let s: Vec<f64> = s_raw.iter().zip(bs.iter()).map(|(&r, &b)| (r + b).tanh()).collect();
            let t_raw = matvec(wt, &h1);
            let t: Vec<f64> = t_raw.iter().zip(bt.iter()).map(|(&r, &b)| r + b).collect();

            // Apply affine transform to x2
            let x2_new: Vec<f64> = x2.iter().enumerate().map(|(i, &v)| {
                let si = s.get(i).copied().unwrap_or(0.0);
                let ti = t.get(i).copied().unwrap_or(0.0);
                v * si.exp() + ti
            }).collect();

            // Accumulate log-det
            let ld: f64 = s.iter().sum();
            log_det += ld;

            // Alternate which half is transformed
            if layer % 2 == 0 {
                x = [x1, x2_new].concat();
            } else {
                x = [x2_new, x1].concat();
            }
        }

        Ok((x, log_det))
    }

    /// Log-likelihood under standard Gaussian: -0.5 * sum(z^2) + log_det - 0.5*D*ln(2π).
    pub fn log_likelihood(&self, z: &[f64], log_det: f64) -> f64 {
        let d = z.len() as f64;
        let sq_sum: f64 = z.iter().map(|&v| v * v).sum();
        -0.5 * sq_sum + log_det - 0.5 * d * (2.0 * std::f64::consts::PI).ln()
    }
}

/// Drug-likeness filter implementing Lipinski's Rule of Five and Veber's rules.
///
/// Used to filter generated molecules for oral bioavailability.
#[derive(Debug, Clone)]
pub struct MolDruglikenessFilter {
    /// MW threshold (default 500 Da).
    pub mw_threshold: f64,
    /// HBD threshold (default 5).
    pub hbd_threshold: usize,
    /// HBA threshold (default 10).
    pub hba_threshold: usize,
    /// LogP threshold (default 5.0).
    pub logp_threshold: f64,
    /// Rotatable bonds threshold for Veber (default 10).
    pub rotbond_threshold: usize,
    /// TPSA threshold for Veber in Ų (default 140).
    pub tpsa_threshold: f64,
}

impl MolDruglikenessFilter {
    /// Create a filter with Lipinski RO5 + Veber defaults.
    pub fn new() -> Self {
        MolDruglikenessFilter {
            mw_threshold: 500.0,
            hbd_threshold: 5,
            hba_threshold: 10,
            logp_threshold: 5.0,
            rotbond_threshold: 10,
            tpsa_threshold: 140.0,
        }
    }

    /// Check Lipinski's Rule of Five from molecular descriptors.
    ///
    /// At most 1 violation allowed (Lipinski lenient interpretation).
    pub fn passes_ro5(&self, mw: f64, hbd: usize, hba: usize, logp: f64) -> bool {
        let violations = [
            mw > self.mw_threshold,
            hbd > self.hbd_threshold,
            hba > self.hba_threshold,
            logp > self.logp_threshold,
        ].iter().filter(|&&v| v).count();
        violations <= 1
    }

    /// Check Veber rules (oral bioavailability).
    pub fn passes_veber(&self, rotbonds: usize, tpsa: f64) -> bool {
        rotbonds <= self.rotbond_threshold && tpsa <= self.tpsa_threshold
    }

    /// Estimate approximate MW from atom count (heuristic: avg heavy atom mass ≈ 12.5 Da).
    pub fn estimate_mw(n_heavy_atoms: usize) -> f64 {
        n_heavy_atoms as f64 * 12.5
    }

    /// Estimate HBD from molecular graph (N and O atoms with hydrogens).
    pub fn estimate_hbd(mol: &MolGraph) -> usize {
        mol.atoms.iter().filter(|a| (a.atomic_num == 7 || a.atomic_num == 8) && a.n_hydrogens > 0).count()
    }

    /// Estimate HBA from molecular graph (N and O atoms).
    pub fn estimate_hba(mol: &MolGraph) -> usize {
        mol.atoms.iter().filter(|a| a.atomic_num == 7 || a.atomic_num == 8).count()
    }

    /// Full druglikeness assessment of a molecular graph.
    pub fn assess(&self, mol: &MolGraph, logp: f64, rotbonds: usize, tpsa: f64) -> DruglikenessReport {
        let mw = Self::estimate_mw(mol.n_atoms());
        let hbd = Self::estimate_hbd(mol);
        let hba = Self::estimate_hba(mol);
        let ro5 = self.passes_ro5(mw, hbd, hba, logp);
        let veber = self.passes_veber(rotbonds, tpsa);
        DruglikenessReport { mw, hbd, hba, logp, rotbonds, tpsa, passes_ro5: ro5, passes_veber: veber }
    }
}

impl Default for MolDruglikenessFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Druglikeness assessment report.
#[derive(Debug, Clone)]
pub struct DruglikenessReport {
    /// Estimated molecular weight (Da).
    pub mw: f64,
    /// Hydrogen bond donors.
    pub hbd: usize,
    /// Hydrogen bond acceptors.
    pub hba: usize,
    /// LogP (octanol-water partition coefficient).
    pub logp: f64,
    /// Rotatable bond count.
    pub rotbonds: usize,
    /// Topological polar surface area (Ų).
    pub tpsa: f64,
    /// Passes Lipinski Rule of Five.
    pub passes_ro5: bool,
    /// Passes Veber rules.
    pub passes_veber: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Molecular Property Prediction Ensemble
// ─────────────────────────────────────────────────────────────────────────────

/// AttentiveFP: graph attention with atom-level and molecule-level attention (Xiong 2019).
pub struct AttentiveFp {
    /// Atom-level attention weights [hidden_dim × hidden_dim].
    w_atom_attn: Vec<Vec<f64>>,
    /// Molecule-level attention (super-node).
    w_mol_attn: Vec<Vec<f64>>,
    w_transform: Vec<Vec<f64>>,
    b_transform: Vec<f64>,
    w_readout: Vec<Vec<f64>>,
    b_readout: Vec<f64>,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// Number of attention layers.
    pub n_layers: usize,
}

impl AttentiveFp {
    /// Construct AttentiveFP.
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, n_layers: usize, seed: u64) -> Self {
        let w_atom_attn = xavier_uniform_2d(hidden_dim, hidden_dim, seed);
        let w_mol_attn = xavier_uniform_2d(1, hidden_dim, seed.wrapping_add(1));
        let w_transform = xavier_uniform_2d(hidden_dim, input_dim, seed.wrapping_add(2));
        let b_transform = vec![0.0; hidden_dim];
        let w_readout = xavier_uniform_2d(output_dim, hidden_dim, seed.wrapping_add(3));
        let b_readout = vec![0.0; output_dim];
        AttentiveFp { w_atom_attn, w_mol_attn, w_transform, b_transform, w_readout, b_readout,
                      hidden_dim, output_dim, n_layers }
    }

    /// Forward pass: returns graph-level prediction of shape \[output_dim\].
    pub fn forward(&self, mol: &MolGraph) -> Result<Vec<f64>> {
        let n = mol.n_atoms();
        if n == 0 {
            return Err(TensorError::invalid_argument_op("AttentiveFp::forward", "empty graph"));
        }

        let atom_feats = mol.atom_features();
        let adj = mol.adjacency_list();

        // Initial transformation
        let mut h: Vec<Vec<f64>> = atom_feats.iter().map(|f| {
            let p = pad_or_trim(f, self.w_transform.first().map(|r| r.len()).unwrap_or(1));
            let raw = matvec(&self.w_transform, &p);
            raw.iter().zip(self.b_transform.iter()).map(|(&x, &b)| relu(x + b)).collect()
        }).collect();

        for _layer in 0..self.n_layers {
            let h_prev = h.clone();

            // Atom-level attention
            for i in 0..n {
                let neighbors = &adj[i];
                if neighbors.is_empty() {
                    continue;
                }

                // Compute attention scores for neighbors
                let mut scores = Vec::with_capacity(neighbors.len());
                for &(j, _) in neighbors {
                    let hj = pad_or_trim(&h_prev[j], self.hidden_dim);
                    let q = matvec(&self.w_atom_attn, &hj);
                    let hi = pad_or_trim(&h_prev[i], self.hidden_dim);
                    let score: f64 = q.iter().zip(hi.iter()).map(|(&a, &b)| a * b).sum();
                    scores.push(score);
                }

                // Softmax over neighbors
                let attn = softmax(&scores);

                // Weighted sum of neighbor features
                let mut ctx = vec![0.0f64; self.hidden_dim];
                for (k, &(j, _)) in neighbors.iter().enumerate() {
                    let hj = pad_or_trim(&h_prev[j], self.hidden_dim);
                    let w = attn.get(k).copied().unwrap_or(0.0);
                    for (c, &v) in ctx.iter_mut().zip(hj.iter()) {
                        *c += w * v;
                    }
                }

                // GRU-style update: h_i = relu(h_i + ctx)
                for (hi_v, c) in h[i].iter_mut().zip(ctx.iter()) {
                    *hi_v = relu(*hi_v + *c);
                }
            }
        }

        // Molecule-level attention pooling
        let scores: Vec<f64> = h.iter().map(|hi| {
            let h_padded = pad_or_trim(hi, self.hidden_dim);
            let s = matvec(&self.w_mol_attn, &h_padded);
            s.first().copied().unwrap_or(0.0)
        }).collect();
        let attn = softmax(&scores);

        let mut mol_embed = vec![0.0f64; self.hidden_dim];
        for (i, &w) in attn.iter().enumerate() {
            let hi = pad_or_trim(&h[i], self.hidden_dim);
            for (m, &v) in mol_embed.iter_mut().zip(hi.iter()) {
                *m += w * v;
            }
        }

        // Final prediction
        let out_raw = matvec(&self.w_readout, &mol_embed);
        let out: Vec<f64> = out_raw.iter().zip(self.b_readout.iter()).map(|(&x, &b)| x + b).collect();
        Ok(out)
    }
}

/// MolBERT: BERT-style pretraining for molecular SMILES with masked atom prediction.
///
/// Encodes molecules as token sequences and predicts masked atom types.
pub struct MolBert {
    /// Vocabulary size (atom types + special tokens).
    pub vocab_size: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    // Token embedding table
    token_embed: Vec<Vec<f64>>,
    // Per-layer QKV projection
    w_qkv: Vec<Vec<Vec<f64>>>,
    // Per-layer FFN
    w_ffn1: Vec<Vec<Vec<f64>>>,
    w_ffn2: Vec<Vec<Vec<f64>>>,
    b_ffn1: Vec<Vec<f64>>,
    b_ffn2: Vec<Vec<f64>>,
    // MLM head
    w_mlm: Vec<Vec<f64>>,
    b_mlm: Vec<f64>,
}

impl MolBert {
    /// Construct a MolBERT model.
    pub fn new(vocab_size: usize, embed_dim: usize, n_layers: usize, n_heads: usize, seed: u64) -> Self {
        let token_embed = xavier_uniform_2d(vocab_size, embed_dim, seed);
        let head_dim = (embed_dim / n_heads).max(1);
        let mut w_qkv = Vec::new();
        let mut w_ffn1 = Vec::new();
        let mut w_ffn2 = Vec::new();
        let mut b_ffn1 = Vec::new();
        let mut b_ffn2 = Vec::new();
        for l in 0..n_layers {
            let s = seed.wrapping_add(l as u64 * 1000);
            // QKV for all heads flattened: [3 * head_dim * n_heads × embed_dim]
            w_qkv.push(xavier_uniform_2d(3 * head_dim * n_heads, embed_dim, s));
            w_ffn1.push(xavier_uniform_2d(embed_dim * 4, embed_dim, s.wrapping_add(1)));
            w_ffn2.push(xavier_uniform_2d(embed_dim, embed_dim * 4, s.wrapping_add(2)));
            b_ffn1.push(vec![0.0; embed_dim * 4]);
            b_ffn2.push(vec![0.0; embed_dim]);
        }
        let w_mlm = xavier_uniform_2d(vocab_size, embed_dim, seed.wrapping_add(9999));
        let b_mlm = vec![0.0; vocab_size];
        MolBert { vocab_size, embed_dim, n_layers, n_heads, token_embed, w_qkv, w_ffn1, w_ffn2, b_ffn1, b_ffn2, w_mlm, b_mlm }
    }

    /// Encode a sequence of token ids to contextualized embeddings.
    pub fn encode(&self, token_ids: &[usize]) -> Result<Vec<Vec<f64>>> {
        if token_ids.is_empty() {
            return Err(TensorError::invalid_argument_op("MolBert::encode", "empty token sequence"));
        }

        // Token embeddings
        let mut h: Vec<Vec<f64>> = token_ids.iter().map(|&tid| {
            let idx = tid % self.vocab_size;
            self.token_embed[idx].clone()
        }).collect();

        let seq_len = h.len();
        let head_dim = (self.embed_dim / self.n_heads).max(1);

        for l in 0..self.n_layers {
            let h_prev = h.clone();

            // Self-attention (simplified: full-sequence, single query per token)
            for i in 0..seq_len {
                let hi = pad_or_trim(&h_prev[i], self.embed_dim);
                let qkv_all = matvec(&self.w_qkv[l], &hi);
                // Split into Q, K, V (each head_dim * n_heads)
                let total = head_dim * self.n_heads;
                let q = &qkv_all[..total.min(qkv_all.len())];
                let k_start = total.min(qkv_all.len());
                // k slice is recomputed per-neighbor below for attention scoring
                let v_start = (k_start + total).min(qkv_all.len());
                // v slice is recomputed per-neighbor below for value aggregation

                // Attention scores over all positions
                let mut scores = Vec::with_capacity(seq_len);
                for j in 0..seq_len {
                    let hj = pad_or_trim(&h_prev[j], self.embed_dim);
                    let kj_all = matvec(&self.w_qkv[l], &hj);
                    let kj = &kj_all[k_start..(k_start + total).min(kj_all.len())];
                    let dot: f64 = q.iter().zip(kj.iter()).map(|(&a, &b)| a * b).sum();
                    scores.push(dot / (head_dim as f64).sqrt().max(1e-8));
                }
                let attn = softmax(&scores);

                // Weighted value aggregation (v_start points into qkv_all for per-neighbor recomputation)
                let mut ctx = vec![0.0f64; self.embed_dim];
                for (j, &w) in attn.iter().enumerate() {
                    let hj = pad_or_trim(&h_prev[j], self.embed_dim);
                    let vj_all = matvec(&self.w_qkv[l], &hj);
                    let vj = &vj_all[v_start..(v_start + self.embed_dim).min(vj_all.len())];
                    for (c, &val) in ctx.iter_mut().zip(vj.iter()) {
                        *c += w * val;
                    }
                }

                // Residual + FFN
                let ctx_padded = pad_or_trim(&ctx, self.embed_dim);
                let res: Vec<f64> = h_prev[i].iter().zip(ctx_padded.iter()).map(|(&a, &b)| a + b).collect();
                let ffn1 = matvec(&self.w_ffn1[l], &res);
                let ffn1_act: Vec<f64> = ffn1.iter().zip(self.b_ffn1[l].iter())
                    .map(|(&x, &b)| relu(x + b)).collect();
                let ffn2 = matvec(&self.w_ffn2[l], &ffn1_act);
                h[i] = ffn2.iter().zip(self.b_ffn2[l].iter()).zip(res.iter())
                    .map(|((&x, &b), &r)| x + b + r).collect();

            }
        }

        Ok(h)
    }

    /// Predict masked token logits for a given sequence.
    pub fn predict_masked(&self, token_ids: &[usize], mask_positions: &[usize]) -> Result<Vec<Vec<f64>>> {
        let embeddings = self.encode(token_ids)?;
        let mut out = Vec::new();
        for &pos in mask_positions {
            if pos >= embeddings.len() {
                return Err(TensorError::invalid_argument_op("MolBert::predict_masked", "mask position out of range"));
            }
            let logits_raw = matvec(&self.w_mlm, &embeddings[pos]);
            let logits: Vec<f64> = logits_raw.iter().zip(self.b_mlm.iter()).map(|(&x, &b)| x + b).collect();
            out.push(logits);
        }
        Ok(out)
    }
}

/// Multi-task molecular property prediction network.
///
/// Simultaneously predicts logP, aqueous solubility, toxicity, and binding affinity.
pub struct MultiTaskMolNet {
    /// MPNN backbone.
    pub backbone: Mpnn,
    /// Number of tasks.
    pub n_tasks: usize,
    task_heads: Vec<Vec<Vec<f64>>>,
    task_biases: Vec<Vec<f64>>,
    /// Task names for reporting.
    pub task_names: Vec<String>,
}

impl MultiTaskMolNet {
    /// Construct a multi-task net with 4 standard ADMET tasks.
    pub fn new(mpnn_config: MolMpnnConfig) -> Self {
        let n_tasks = 4;
        let mpnn_out = mpnn_config.output_dim;
        let seed = mpnn_config.seed;
        let backbone = Mpnn::new(mpnn_config);
        let task_names = vec![
            "logP".to_string(),
            "aqueous_solubility".to_string(),
            "toxicity".to_string(),
            "binding_affinity".to_string(),
        ];
        let mut task_heads = Vec::new();
        let mut task_biases = Vec::new();
        for t in 0..n_tasks {
            task_heads.push(xavier_uniform_2d(1, mpnn_out, seed.wrapping_add(t as u64 * 500)));
            task_biases.push(vec![0.0; 1]);
        }
        MultiTaskMolNet { backbone, n_tasks, task_heads, task_biases, task_names }
    }

    /// Predict all tasks for a molecule.
    ///
    /// Returns a vector of length `n_tasks` with per-task predictions.
    pub fn predict(&self, mol: &MolGraph) -> Result<Vec<f64>> {
        let node_embeds = self.backbone.forward(mol)?;
        let graph_embed = self.backbone.graph_readout(&node_embeds);
        let preds: Vec<f64> = (0..self.n_tasks).map(|t| {
            let raw = matvec(&self.task_heads[t], &graph_embed);
            raw.first().copied().unwrap_or(0.0) + self.task_biases[t].first().copied().unwrap_or(0.0)
        }).collect();
        Ok(preds)
    }

    /// Compute multi-task MSE loss.
    pub fn loss(&self, preds: &[f64], targets: &[f64]) -> f64 {
        let n = preds.len().min(targets.len()) as f64;
        if n == 0.0 {
            return 0.0;
        }
        preds.iter().zip(targets.iter()).map(|(&p, &t)| (p - t).powi(2)).sum::<f64>() / n
    }
}

/// Uncertainty-aware molecular property predictor using MC Dropout.
///
/// Runs `n_samples` stochastic forward passes to estimate predictive uncertainty.
pub struct UncertaintyMolPredictor {
    /// Inner multi-task network.
    pub net: MultiTaskMolNet,
    /// Dropout rate for MC dropout.
    pub dropout_rate: f64,
    /// Number of MC samples for uncertainty estimation.
    pub n_samples: usize,
}

impl UncertaintyMolPredictor {
    /// Construct an uncertainty predictor.
    pub fn new(mpnn_config: MolMpnnConfig, dropout_rate: f64, n_samples: usize) -> Self {
        let net = MultiTaskMolNet::new(mpnn_config);
        UncertaintyMolPredictor { net, dropout_rate, n_samples }
    }

    /// Predict with uncertainty: returns `(mean, variance)` per task.
    pub fn predict_with_uncertainty(&self, mol: &MolGraph, rng: &mut StdRng) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut all_preds: Vec<Vec<f64>> = Vec::with_capacity(self.n_samples);

        for _ in 0..self.n_samples {
            // Stochastic prediction: apply dropout mask to graph embedding
            let node_embeds = self.net.backbone.forward(mol)?;
            let mut graph_embed = self.net.backbone.graph_readout(&node_embeds);

            // Apply MC dropout
            for v in graph_embed.iter_mut() {
                if rng.random::<f64>() < self.dropout_rate {
                    *v = 0.0;
                } else {
                    *v /= 1.0 - self.dropout_rate;
                }
            }

            let preds: Vec<f64> = (0..self.net.n_tasks).map(|t| {
                let raw = matvec(&self.net.task_heads[t], &graph_embed);
                raw.first().copied().unwrap_or(0.0) + self.net.task_biases[t].first().copied().unwrap_or(0.0)
            }).collect();
            all_preds.push(preds);
        }

        let n_tasks = self.net.n_tasks;
        let n = self.n_samples as f64;

        // Mean
        let mean: Vec<f64> = (0..n_tasks).map(|t| {
            all_preds.iter().map(|p| p.get(t).copied().unwrap_or(0.0)).sum::<f64>() / n
        }).collect();

        // Variance
        let variance: Vec<f64> = (0..n_tasks).map(|t| {
            let m = mean[t];
            all_preds.iter().map(|p| {
                let v = p.get(t).copied().unwrap_or(0.0);
                (v - m).powi(2)
            }).sum::<f64>() / n
        }).collect();

        Ok((mean, variance))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reaction Prediction
// ─────────────────────────────────────────────────────────────────────────────

/// Atom-mapped reaction graph connecting reactants, reagents, and products.
#[derive(Debug, Clone)]
pub struct ReactionGraph {
    /// Reactant molecule graphs.
    pub reactants: Vec<MolGraph>,
    /// Reagent molecule graphs (catalysts, solvents).
    pub reagents: Vec<MolGraph>,
    /// Product molecule graphs.
    pub products: Vec<MolGraph>,
    /// Atom mapping: (reactant_mol_idx, atom_idx) → (product_mol_idx, atom_idx).
    pub atom_map: Vec<((usize, usize), (usize, usize))>,
}

impl ReactionGraph {
    /// Construct a reaction graph from reactant/product lists.
    pub fn new(reactants: Vec<MolGraph>, reagents: Vec<MolGraph>, products: Vec<MolGraph>) -> Self {
        ReactionGraph { reactants, reagents, products, atom_map: Vec::new() }
    }

    /// Number of atoms across all reactants.
    pub fn n_reactant_atoms(&self) -> usize {
        self.reactants.iter().map(|m| m.n_atoms()).sum()
    }

    /// Compute reaction fingerprint as diff of Morgan fingerprints.
    pub fn reaction_fingerprint(&self, n_bits: usize) -> Vec<f64> {
        let mfp = MorganFingerprint { radius: 2, n_bits };
        let r_fp: Vec<u8> = {
            let mut acc = vec![0u8; n_bits];
            for mol in &self.reactants {
                let fp = mfp.compute(mol);
                for (a, &b) in acc.iter_mut().zip(fp.iter()) {
                    *a = a.saturating_add(b);
                }
            }
            acc
        };
        let p_fp: Vec<u8> = {
            let mut acc = vec![0u8; n_bits];
            for mol in &self.products {
                let fp = mfp.compute(mol);
                for (a, &b) in acc.iter_mut().zip(fp.iter()) {
                    *a = a.saturating_add(b);
                }
            }
            acc
        };
        r_fp.iter().zip(p_fp.iter()).map(|(&r, &p)| p as f64 - r as f64).collect()
    }
}

/// Template-free atom mapper inspired by Schwaller 2021 (LocalMapper).
///
/// Uses graph matching on reactant-product atom feature similarity.
pub struct LocalMapper {
    /// Similarity threshold for accepting a mapping.
    pub threshold: f64,
}

impl LocalMapper {
    /// Construct a LocalMapper with a given similarity threshold.
    pub fn new(threshold: f64) -> Self {
        LocalMapper { threshold }
    }

    /// Compute atom-level feature similarity matrix between reactant and product.
    ///
    /// Returns `(n_reactant_atoms × n_product_atoms)` similarity matrix.
    pub fn similarity_matrix(&self, reactant: &MolGraph, product: &MolGraph) -> Vec<Vec<f64>> {
        let r_feats = reactant.atom_features();
        let p_feats = product.atom_features();
        r_feats.iter().map(|rf| {
            p_feats.iter().map(|pf| {
                // Cosine similarity
                let dot: f64 = rf.iter().zip(pf.iter()).map(|(&a, &b)| a * b).sum();
                let nr: f64 = rf.iter().map(|&x| x * x).sum::<f64>().sqrt().max(1e-10);
                let np: f64 = pf.iter().map(|&x| x * x).sum::<f64>().sqrt().max(1e-10);
                dot / (nr * np)
            }).collect()
        }).collect()
    }

    /// Find best atom mapping via greedy Hungarian-like assignment.
    ///
    /// Returns a list of `(reactant_atom_idx, product_atom_idx)` pairs.
    pub fn map_atoms(&self, reactant: &MolGraph, product: &MolGraph) -> Vec<(usize, usize)> {
        let sim = self.similarity_matrix(reactant, product);
        let nr = reactant.n_atoms();
        let np = product.n_atoms();
        let mut assigned_p = vec![false; np];
        let mut mapping = Vec::new();

        // Greedy: for each reactant atom, find best unassigned product atom
        for i in 0..nr {
            let mut best_j = None;
            let mut best_s = f64::NEG_INFINITY;
            for j in 0..np {
                if !assigned_p[j] {
                    let s = sim[i].get(j).copied().unwrap_or(0.0);
                    if s > best_s {
                        best_s = s;
                        best_j = Some(j);
                    }
                }
            }
            if let Some(j) = best_j {
                if best_s >= self.threshold {
                    assigned_p[j] = true;
                    mapping.push((i, j));
                }
            }
        }
        mapping
    }
}

/// Template-based retrosynthesis predictor.
///
/// Scores known reaction templates against a target molecule and returns ranked candidates.
pub struct RetrosynthesisPredictor {
    /// Template fingerprints [n_templates × n_bits].
    pub template_fps: Vec<Vec<f64>>,
    /// Template names.
    pub template_names: Vec<String>,
    /// Fingerprint dimension.
    pub n_bits: usize,
    w_score: Vec<Vec<f64>>,
    b_score: Vec<f64>,
}

impl RetrosynthesisPredictor {
    /// Construct a retrosynthesis predictor with `n_templates` templates.
    pub fn new(n_templates: usize, n_bits: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let template_fps: Vec<Vec<f64>> = (0..n_templates)
            .map(|_| (0..n_bits).map(|_| if rng.random::<bool>() { 1.0 } else { 0.0 }).collect())
            .collect();
        let template_names: Vec<String> = (0..n_templates).map(|i| format!("template_{i}")).collect();
        let w_score = xavier_uniform_2d(n_templates, n_bits, seed.wrapping_add(1));
        let b_score = vec![0.0; n_templates];
        RetrosynthesisPredictor { template_fps, template_names, n_bits, w_score, b_score }
    }

    /// Score all templates for a target molecule.
    ///
    /// Returns template indices sorted by score (highest first).
    pub fn rank_templates(&self, mol: &MolGraph) -> Vec<(usize, f64)> {
        let mfp = MorganFingerprint { radius: 2, n_bits: self.n_bits };
        let fp: Vec<f64> = mfp.compute(mol).into_iter().map(|b| b as f64).collect();
        let fp_padded = pad_or_trim(&fp, self.n_bits);
        let scores_raw = matvec(&self.w_score, &fp_padded);
        let mut scored: Vec<(usize, f64)> = scores_raw.iter().zip(self.b_score.iter())
            .enumerate()
            .map(|(i, (&s, &b))| (i, s + b))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Return top-k template predictions.
    pub fn top_k_templates(&self, mol: &MolGraph, k: usize) -> Vec<(String, f64)> {
        let ranked = self.rank_templates(mol);
        let probs = {
            let scores: Vec<f64> = ranked.iter().map(|&(_, s)| s).collect();
            softmax(&scores)
        };
        ranked.into_iter().take(k).enumerate().map(|(i, (idx, _))| {
            let name = self.template_names.get(idx).cloned().unwrap_or_else(|| format!("template_{idx}"));
            (name, probs.get(i).copied().unwrap_or(0.0))
        }).collect()
    }
}

/// Reaction yield predictor from reactant/reagent features.
pub struct ReactionYieldPredictor {
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
    w_out: Vec<Vec<f64>>,
    b_out: Vec<f64>,
    /// Input feature dimension.
    pub feat_dim: usize,
}

impl ReactionYieldPredictor {
    /// Construct a yield predictor with two hidden layers.
    pub fn new(feat_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        let w1 = xavier_uniform_2d(hidden_dim, feat_dim, seed);
        let b1 = vec![0.0; hidden_dim];
        let w2 = xavier_uniform_2d(hidden_dim, hidden_dim, seed.wrapping_add(1));
        let b2 = vec![0.0; hidden_dim];
        let w_out = xavier_uniform_2d(1, hidden_dim, seed.wrapping_add(2));
        let b_out = vec![0.0; 1];
        ReactionYieldPredictor { w1, b1, w2, b2, w_out, b_out, feat_dim }
    }

    /// Predict yield (0–100%) from a reaction fingerprint.
    pub fn predict(&self, rxn: &ReactionGraph) -> Result<f64> {
        if rxn.reactants.is_empty() {
            return Err(TensorError::invalid_argument_op("ReactionYieldPredictor::predict", "no reactants"));
        }
        let fp = rxn.reaction_fingerprint(self.feat_dim);
        let fp_p = pad_or_trim(&fp, self.feat_dim);

        let h1_raw = matvec(&self.w1, &fp_p);
        let h1: Vec<f64> = h1_raw.iter().zip(self.b1.iter()).map(|(&x, &b)| relu(x + b)).collect();
        let h2_raw = matvec(&self.w2, &h1);
        let h2: Vec<f64> = h2_raw.iter().zip(self.b2.iter()).map(|(&x, &b)| relu(x + b)).collect();
        let out_raw = matvec(&self.w_out, &h2);
        let yield_raw = out_raw.first().copied().unwrap_or(0.0) + self.b_out.first().copied().unwrap_or(0.0);

        // Clamp to [0, 100]
        Ok(sigmoid(yield_raw) * 100.0)
    }

    /// Compute MSE loss between predicted and actual yield (%).
    pub fn loss(&self, predicted: f64, actual: f64) -> f64 {
        (predicted - actual).powi(2)
    }
}
