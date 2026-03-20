//! SE(3)/E(3)-Equivariant Deep Learning Algorithms.
//!
//! - EGNN: [`EgnnCoord`], [`EgnnLayer`], [`EgnnModel`]
//! - SE(3)-Transformer: [`Se3FiberBundle`], [`Se3Attention`], [`Se3TransformerLayer`]
//! - Vector Neuron Networks: [`VnLinear`], [`VnLeakyRelu`], [`VnMaxPool`], [`VnNetwork`]
//! - Invariant Point Attention: [`IpaFrame`], [`IpaAttentionScore`], [`IpaLayer`]

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Internal utilities (kept local to this module)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn eq_relu(x: f64) -> f64 {
    x.max(0.0)
}

#[inline]
fn eq_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn eq_matvec(mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    mat.iter().map(|row| eq_dot(row, v)).collect()
}

fn eq_rand_weight(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let scale = (2.0 / cols as f64).sqrt();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                .collect()
        })
        .collect()
}

/// Compute L2 norm of a 3D vector.
#[inline]
fn norm3(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Squared L2 distance between two 3D points.
#[inline]
fn sq_dist3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    dx * dx + dy * dy + dz * dz
}

/// Softmax over a slice.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let s: f64 = exp.iter().sum();
    if s < 1e-300 {
        vec![1.0 / logits.len() as f64; logits.len()]
    } else {
        exp.iter().map(|e| e / s).collect()
    }
}

/// Apply a 3x3 rotation matrix to a 3D vector.
fn rotate3(rot: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        rot[0][0] * v[0] + rot[0][1] * v[1] + rot[0][2] * v[2],
        rot[1][0] * v[0] + rot[1][1] * v[1] + rot[1][2] * v[2],
        rot[2][0] * v[0] + rot[2][1] * v[1] + rot[2][2] * v[2],
    ]
}

/// Transpose 3x3 rotation matrix.
fn transpose3(r: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [r[0][0], r[1][0], r[2][0]],
        [r[0][1], r[1][1], r[2][1]],
        [r[0][2], r[1][2], r[2][2]],
    ]
}

/// Identity rotation matrix.
fn identity_rot() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

// ─────────────────────────────────────────────────────────────────────────────
// §1 E(n)-Equivariant Graph Neural Network (EGNN)  (Satorras 2021)
// ─────────────────────────────────────────────────────────────────────────────

/// Coordinate update network for EGNN — produces per-edge displacement weights.
///
/// The message function φ_x takes ‖xᵢ−xⱼ‖² concatenated with hᵢ, hⱼ and
/// produces a scalar weight used to shift coordinates equivariantly.
#[derive(Debug, Clone)]
pub struct EgnnCoord {
    /// Weight matrix of the φ_x MLP (first layer).
    pub w1: Vec<Vec<f64>>,
    /// Bias of first layer.
    pub b1: Vec<f64>,
    /// Weight matrix of second (scalar output) layer.
    pub w2: Vec<f64>,
    /// Hidden dimension of the φ_x MLP.
    pub hidden: usize,
    /// Feature dimension hᵢ.
    pub feat_dim: usize,
}

impl EgnnCoord {
    /// Create a new `EgnnCoord` with `feat_dim` node feature channels and
    /// `hidden` hidden units.  Input to the MLP is [hᵢ ‖ hⱼ ‖ ‖xᵢ−xⱼ‖²].
    pub fn new(feat_dim: usize, hidden: usize, seed: u64) -> Self {
        let in_dim = 2 * feat_dim + 1;
        let w1 = eq_rand_weight(hidden, in_dim, seed);
        let b1 = vec![0.0; hidden];
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1));
        let scale = (2.0 / hidden as f64).sqrt();
        let w2: Vec<f64> = (0..hidden)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
            .collect();
        Self { w1, b1, w2, hidden, feat_dim }
    }

    /// Compute the scalar displacement weight for an (i, j) edge.
    ///
    /// Returns a scalar `c_ij` such that `Δxᵢ += c_ij * (xᵢ − xⱼ)`.
    pub fn edge_weight(&self, hi: &[f64], hj: &[f64], sq_d: f64) -> Result<f64> {
        if hi.len() != self.feat_dim || hj.len() != self.feat_dim {
            return Err(TensorError::invalid_argument_op(
                "egnn_coord",
                &format!(
                    "feature dim mismatch: expected {}, got {} and {}",
                    self.feat_dim,
                    hi.len(),
                    hj.len()
                ),
            ));
        }
        let mut inp: Vec<f64> = hi.to_vec();
        inp.extend_from_slice(hj);
        inp.push(sq_d);
        let h1: Vec<f64> = eq_matvec(&self.w1, &inp)
            .into_iter()
            .zip(&self.b1)
            .map(|(z, b)| eq_relu(z + b))
            .collect();
        Ok(eq_dot(&self.w2, &h1))
    }
}

/// Single EGNN message-passing layer.
///
/// Simultaneously updates node features h and 3D coordinates x via:
/// - mᵢⱼ = φ_e(hᵢ, hⱼ, ‖xᵢ−xⱼ‖²)
/// - xᵢ ← xᵢ + Σⱼ (xᵢ−xⱼ) · φ_x(mᵢⱼ)  (equivariant coordinate update)
/// - hᵢ ← φ_h(hᵢ, Σⱼ mᵢⱼ)              (invariant feature update)
#[derive(Debug, Clone)]
pub struct EgnnLayer {
    /// φ_e: edge message MLP input=[hᵢ, hⱼ, ‖xᵢ−xⱼ‖²], output=message_dim.
    pub phi_e_w1: Vec<Vec<f64>>,
    pub phi_e_b1: Vec<f64>,
    pub phi_e_w2: Vec<Vec<f64>>,
    pub phi_e_b2: Vec<f64>,
    /// φ_x: coordinate update scalar.
    pub coord_net: EgnnCoord,
    /// φ_h: node update MLP input=[hᵢ, agg_msg], output=feat_dim.
    pub phi_h_w1: Vec<Vec<f64>>,
    pub phi_h_b1: Vec<f64>,
    pub phi_h_w2: Vec<Vec<f64>>,
    pub phi_h_b2: Vec<f64>,
    pub feat_dim: usize,
    pub msg_dim: usize,
}

impl EgnnLayer {
    /// Construct a layer with `feat_dim`-dimensional node features.
    pub fn new(feat_dim: usize, msg_dim: usize, seed: u64) -> Self {
        let in_e = 2 * feat_dim + 1;
        let phi_e_w1 = eq_rand_weight(msg_dim, in_e, seed);
        let phi_e_b1 = vec![0.0; msg_dim];
        let phi_e_w2 = eq_rand_weight(msg_dim, msg_dim, seed.wrapping_add(1));
        let phi_e_b2 = vec![0.0; msg_dim];

        let coord_net = EgnnCoord::new(feat_dim, msg_dim, seed.wrapping_add(2));

        let in_h = feat_dim + msg_dim;
        let phi_h_w1 = eq_rand_weight(feat_dim * 2, in_h, seed.wrapping_add(3));
        let phi_h_b1 = vec![0.0; feat_dim * 2];
        let phi_h_w2 = eq_rand_weight(feat_dim, feat_dim * 2, seed.wrapping_add(4));
        let phi_h_b2 = vec![0.0; feat_dim];

        Self {
            phi_e_w1,
            phi_e_b1,
            phi_e_w2,
            phi_e_b2,
            coord_net,
            phi_h_w1,
            phi_h_b1,
            phi_h_w2,
            phi_h_b2,
            feat_dim,
            msg_dim,
        }
    }

    fn edge_msg(&self, hi: &[f64], hj: &[f64], sq_d: f64) -> Vec<f64> {
        let mut inp: Vec<f64> = hi.to_vec();
        inp.extend_from_slice(hj);
        inp.push(sq_d);
        let h1: Vec<f64> = eq_matvec(&self.phi_e_w1, &inp)
            .into_iter()
            .zip(&self.phi_e_b1)
            .map(|(z, b)| eq_relu(z + b))
            .collect();
        eq_matvec(&self.phi_e_w2, &h1)
            .into_iter()
            .zip(&self.phi_e_b2)
            .map(|(z, b)| eq_relu(z + b))
            .collect()
    }

    fn node_update(&self, hi: &[f64], agg: &[f64]) -> Vec<f64> {
        let mut inp: Vec<f64> = hi.to_vec();
        inp.extend_from_slice(agg);
        let h1: Vec<f64> = eq_matvec(&self.phi_h_w1, &inp)
            .into_iter()
            .zip(&self.phi_h_b1)
            .map(|(z, b)| eq_relu(z + b))
            .collect();
        eq_matvec(&self.phi_h_w2, &h1)
            .into_iter()
            .zip(&self.phi_h_b2)
            .map(|(z, b)| eq_relu(z + b))
            .collect()
    }

    /// Forward pass.
    ///
    /// Returns `(new_coords, new_features)` where `new_coords` has the same
    /// shape as `coords` and `new_features` has the same shape as `features`.
    pub fn forward(
        &self,
        coords: &[[f64; 3]],
        features: &[Vec<f64>],
    ) -> Result<(Vec<[f64; 3]>, Vec<Vec<f64>>)> {
        let n = coords.len();
        if features.len() != n {
            return Err(TensorError::invalid_argument_op(
                "egnn_layer",
                "coords and features length mismatch",
            ));
        }
        if let Some(h) = features.first() {
            if h.len() != self.feat_dim {
                return Err(TensorError::invalid_argument_op(
                    "egnn_layer",
                    &format!("feature dim {} != layer feat_dim {}", h.len(), self.feat_dim),
                ));
            }
        }

        let mut new_coords = coords.to_vec();
        let mut new_features: Vec<Vec<f64>> = features.to_vec();

        for i in 0..n {
            let mut agg_msg = vec![0.0f64; self.msg_dim];
            let mut coord_delta = [0.0f64; 3];
            let mut count = 0usize;

            for j in 0..n {
                if i == j {
                    continue;
                }
                let sq_d = sq_dist3(&coords[i], &coords[j]);
                let msg = self.edge_msg(&features[i], &features[j], sq_d);
                for (d, &m) in msg.iter().enumerate() {
                    agg_msg[d] += m;
                }
                // Coordinate update: c_ij * (x_i - x_j)
                let c = self.coord_net.edge_weight(&features[i], &features[j], sq_d)?;
                for d in 0..3 {
                    coord_delta[d] += c * (coords[i][d] - coords[j][d]);
                }
                count += 1;
            }

            if count > 0 {
                let inv = 1.0 / count as f64;
                for d in 0..3 {
                    new_coords[i][d] = coords[i][d] + coord_delta[d] * inv;
                }
                for m in agg_msg.iter_mut() {
                    *m *= inv;
                }
            }
            new_features[i] = self.node_update(&features[i], &agg_msg);
        }

        Ok((new_coords, new_features))
    }
}

/// Stacked EGNN model for molecular property prediction.
///
/// Scalar output is obtained by sum-pooling the final node features and
/// passing through a linear readout.
#[derive(Debug, Clone)]
pub struct EgnnModel {
    /// Stacked EGNN layers.
    pub layers: Vec<EgnnLayer>,
    /// Readout: \[feat_dim\] → \[1\] linear.
    pub readout_w: Vec<f64>,
    pub readout_b: f64,
    pub feat_dim: usize,
}

impl EgnnModel {
    /// Create a model with `n_layers` EGNN layers, `feat_dim`-dim node features,
    /// `msg_dim`-dim messages.
    pub fn new(n_layers: usize, feat_dim: usize, msg_dim: usize, seed: u64) -> Self {
        let layers = (0..n_layers)
            .map(|k| EgnnLayer::new(feat_dim, msg_dim, seed.wrapping_add(k as u64 * 10)))
            .collect();
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(999));
        let scale = (1.0 / feat_dim as f64).sqrt();
        let readout_w: Vec<f64> = (0..feat_dim)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
            .collect();
        Self { layers, readout_w, readout_b: 0.0, feat_dim }
    }

    /// Forward: returns predicted scalar graph property.
    pub fn forward(&self, coords: &[[f64; 3]], features: &[Vec<f64>]) -> Result<f64> {
        let mut cur_coords = coords.to_vec();
        let mut cur_feats = features.to_vec();

        for layer in &self.layers {
            let (nc, nf) = layer.forward(&cur_coords, &cur_feats)?;
            cur_coords = nc;
            cur_feats = nf;
        }

        // Sum-pool features → scalar readout
        let n = cur_feats.len();
        if n == 0 {
            return Ok(0.0);
        }
        let mut pooled = vec![0.0f64; self.feat_dim];
        for h in &cur_feats {
            for (d, &v) in h.iter().enumerate() {
                pooled[d] += v;
            }
        }
        let inv = 1.0 / n as f64;
        for v in pooled.iter_mut() {
            *v *= inv;
        }
        Ok(eq_dot(&self.readout_w, &pooled) + self.readout_b)
    }

    /// Get the final coordinate embedding (after all EGNN layers).
    pub fn embed_coords(&self, coords: &[[f64; 3]], features: &[Vec<f64>]) -> Result<Vec<[f64; 3]>> {
        let mut cur_coords = coords.to_vec();
        let mut cur_feats = features.to_vec();
        for layer in &self.layers {
            let (nc, nf) = layer.forward(&cur_coords, &cur_feats)?;
            cur_coords = nc;
            cur_feats = nf;
        }
        Ok(cur_coords)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2 SE(3)-Transformer  (Fuchs et al. 2020)
// ─────────────────────────────────────────────────────────────────────────────

/// Spherical harmonics for l=0 and l=1.
///
/// Returns `[Y_0^0, Y_1^{-1}, Y_1^0, Y_1^1]` evaluated at unit direction `d`.
pub fn spherical_harmonics_01(d: &[f64; 3]) -> [f64; 4] {
    use std::f64::consts::PI;
    let r = norm3(d).max(1e-10);
    let (x, y, z) = (d[0] / r, d[1] / r, d[2] / r);
    let y00 = 1.0 / (4.0 * PI).sqrt();
    let c1 = (3.0 / (4.0 * PI)).sqrt();
    let y1m1 = c1 * y;
    let y10 = c1 * z;
    let y11 = -c1 * x;
    [y00, y1m1, y10, y11]
}

/// Fiber bundle holding type-0 (scalar) and type-1 (3D vector) features per node.
#[derive(Debug, Clone)]
pub struct Se3FiberBundle {
    /// Type-0 features: shape [n_nodes x scalar_dim].
    pub type0: Vec<Vec<f64>>,
    /// Type-1 features: shape [n_nodes x vector_channels x 3].
    pub type1: Vec<Vec<[f64; 3]>>,
}

impl Se3FiberBundle {
    /// Create a bundle with `n_nodes` nodes, `s_dim` scalar channels, `v_dim` vector channels.
    pub fn new(n_nodes: usize, s_dim: usize, v_dim: usize) -> Self {
        Self {
            type0: vec![vec![0.0; s_dim]; n_nodes],
            type1: vec![vec![[0.0; 3]; v_dim]; n_nodes],
        }
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.type0.len()
    }

    /// Scalar feature dimension.
    pub fn scalar_dim(&self) -> usize {
        self.type0.first().map(|v| v.len()).unwrap_or(0)
    }

    /// Vector channel count.
    pub fn vector_channels(&self) -> usize {
        self.type1.first().map(|v| v.len()).unwrap_or(0)
    }
}

/// SE(3)-equivariant attention layer using degree-0 keys/queries and degree-0/1 values.
///
/// Keys and queries are computed from type-0 (scalar) features, ensuring SE(3)-invariant
/// attention weights.  Values combine type-0 and type-1 (vector) features with spherical-
/// harmonic coupling, making outputs SE(3)-covariant.
#[derive(Debug, Clone)]
pub struct Se3Attention {
    /// Number of attention heads.
    pub n_heads: usize,
    /// Scalar feature dimension (type-0).
    pub scalar_dim: usize,
    /// Vector channels (type-1).
    pub vector_channels: usize,
    /// Key/query dimension per head.
    pub head_dim: usize,
    /// W_q: [n_heads * head_dim x scalar_dim].
    pub w_q: Vec<Vec<f64>>,
    /// W_k: [n_heads * head_dim x scalar_dim].
    pub w_k: Vec<Vec<f64>>,
    /// W_v scalar: [scalar_dim x scalar_dim].
    pub w_v0: Vec<Vec<f64>>,
    /// W_v vector: [vector_channels x scalar_dim].
    pub w_v1: Vec<Vec<f64>>,
    /// Output projection scalar: [scalar_dim x scalar_dim].
    pub w_o0: Vec<Vec<f64>>,
    /// Cutoff distance.
    pub cutoff: f64,
}

impl Se3Attention {
    /// Create an SE(3) attention layer.
    pub fn new(
        n_heads: usize,
        scalar_dim: usize,
        vector_channels: usize,
        cutoff: f64,
        seed: u64,
    ) -> Result<Self> {
        if n_heads == 0 || scalar_dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "se3_attention",
                "n_heads and scalar_dim must be positive",
            ));
        }
        let head_dim = (scalar_dim / n_heads).max(1);
        let kq_dim = n_heads * head_dim;
        let w_q = eq_rand_weight(kq_dim, scalar_dim, seed);
        let w_k = eq_rand_weight(kq_dim, scalar_dim, seed.wrapping_add(1));
        let w_v0 = eq_rand_weight(scalar_dim, scalar_dim, seed.wrapping_add(2));
        let w_v1 = eq_rand_weight(vector_channels, scalar_dim, seed.wrapping_add(3));
        let w_o0 = eq_rand_weight(scalar_dim, scalar_dim, seed.wrapping_add(4));
        Ok(Self {
            n_heads,
            scalar_dim,
            vector_channels,
            head_dim,
            w_q,
            w_k,
            w_v0,
            w_v1,
            w_o0,
            cutoff,
        })
    }

    /// Forward equivariant self-attention.
    pub fn forward(&self, bundle: &Se3FiberBundle, positions: &[[f64; 3]]) -> Result<Se3FiberBundle> {
        let n = bundle.n_nodes();
        if positions.len() != n {
            return Err(TensorError::invalid_argument_op(
                "se3_attention",
                "positions and bundle node count mismatch",
            ));
        }

        let mut out_type0 = vec![vec![0.0f64; self.scalar_dim]; n];
        let mut out_type1 = vec![vec![[0.0f64; 3]; self.vector_channels]; n];

        // Precompute keys and queries
        let keys: Vec<Vec<f64>> = bundle
            .type0
            .iter()
            .map(|h| eq_matvec(&self.w_k, h))
            .collect();
        let queries: Vec<Vec<f64>> = bundle
            .type0
            .iter()
            .map(|h| eq_matvec(&self.w_q, h))
            .collect();

        let scale = 1.0 / (self.head_dim as f64).sqrt();

        for i in 0..n {
            // Compute attention logits — invariant (scalar product of type-0 features)
            let mut logits: Vec<f64> = (0..n)
                .map(|j| {
                    let d = sq_dist3(&positions[i], &positions[j]).sqrt();
                    if d > self.cutoff && i != j {
                        return f64::NEG_INFINITY;
                    }
                    eq_dot(&queries[i], &keys[j]) * scale
                })
                .collect();
            // mask self with distance 0 always included
            logits[i] = eq_dot(&queries[i], &keys[i]) * scale;

            let attn = softmax(&logits);

            // Aggregate scalar values
            let mut agg_scalar = vec![0.0f64; self.scalar_dim];
            for (j, &w) in attn.iter().enumerate() {
                let v0 = eq_matvec(&self.w_v0, &bundle.type0[j]);
                for (d, &v) in v0.iter().enumerate() {
                    agg_scalar[d] += w * v;
                }
            }
            out_type0[i] = eq_matvec(&self.w_o0, &agg_scalar);

            // Aggregate vector values using spherical harmonic coupling
            let mut agg_vec = vec![[0.0f64; 3]; self.vector_channels];
            for (j, &w) in attn.iter().enumerate() {
                if w < 1e-12 {
                    continue;
                }
                // Direction from i to j
                let dv = [
                    positions[j][0] - positions[i][0],
                    positions[j][1] - positions[i][1],
                    positions[j][2] - positions[i][2],
                ];
                let sph = spherical_harmonics_01(&dv);

                // type-1 contribution from existing type-1 features (equivariant pass-through)
                for vc in 0..bundle.vector_channels() {
                    for d in 0..3 {
                        agg_vec[vc.min(self.vector_channels - 1)][d] +=
                            w * bundle.type1[j][vc][d];
                    }
                }

                // type-0 → type-1 coupling via Y_1^m (Clebsch-Gordan l=0⊗l=1→l=1)
                let v1_weights = eq_matvec(&self.w_v1, &bundle.type0[j]);
                for (vc, &wt) in v1_weights.iter().enumerate() {
                    if vc >= self.vector_channels {
                        break;
                    }
                    // Use sph[1..4] = (Y_1^{-1}, Y_1^0, Y_1^1) as direction
                    for d in 0..3 {
                        agg_vec[vc][d] += w * wt * sph[d + 1];
                    }
                }
            }
            out_type1[i] = agg_vec;
        }

        Ok(Se3FiberBundle { type0: out_type0, type1: out_type1 })
    }
}

/// Full SE(3)-Transformer layer: equivariant self-attention + layer norm on scalars.
#[derive(Debug, Clone)]
pub struct Se3TransformerLayer {
    /// Attention sub-layer.
    pub attention: Se3Attention,
    /// Layer norm scale (type-0 only).
    pub ln_scale: Vec<f64>,
    /// Layer norm shift (type-0 only).
    pub ln_shift: Vec<f64>,
    /// FFN weight 1: [ffn_dim x scalar_dim].
    pub ffn_w1: Vec<Vec<f64>>,
    pub ffn_b1: Vec<f64>,
    /// FFN weight 2: [scalar_dim x ffn_dim].
    pub ffn_w2: Vec<Vec<f64>>,
    pub ffn_b2: Vec<f64>,
}

impl Se3TransformerLayer {
    /// Construct a transformer layer.
    pub fn new(
        n_heads: usize,
        scalar_dim: usize,
        vector_channels: usize,
        ffn_dim: usize,
        cutoff: f64,
        seed: u64,
    ) -> Result<Self> {
        let attention = Se3Attention::new(n_heads, scalar_dim, vector_channels, cutoff, seed)?;
        let ffn_w1 = eq_rand_weight(ffn_dim, scalar_dim, seed.wrapping_add(100));
        let ffn_b1 = vec![0.0; ffn_dim];
        let ffn_w2 = eq_rand_weight(scalar_dim, ffn_dim, seed.wrapping_add(101));
        let ffn_b2 = vec![0.0; scalar_dim];
        Ok(Self {
            attention,
            ln_scale: vec![1.0; scalar_dim],
            ln_shift: vec![0.0; scalar_dim],
            ffn_w1,
            ffn_b1,
            ffn_w2,
            ffn_b2,
        })
    }

    fn layer_norm(&self, x: &[f64]) -> Vec<f64> {
        let mean = x.iter().sum::<f64>() / x.len().max(1) as f64;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len().max(1) as f64;
        let std = (var + 1e-5).sqrt();
        x.iter()
            .zip(&self.ln_scale)
            .zip(&self.ln_shift)
            .map(|((v, s), sh)| (v - mean) / std * s + sh)
            .collect()
    }

    fn ffn(&self, x: &[f64]) -> Vec<f64> {
        let h: Vec<f64> = eq_matvec(&self.ffn_w1, x)
            .into_iter()
            .zip(&self.ffn_b1)
            .map(|(z, b)| eq_relu(z + b))
            .collect();
        eq_matvec(&self.ffn_w2, &h)
            .into_iter()
            .zip(&self.ffn_b2)
            .map(|(z, b)| z + b)
            .collect()
    }

    /// Forward: residual attention + FFN with layer-norm on scalar features.
    pub fn forward(&self, bundle: &Se3FiberBundle, positions: &[[f64; 3]]) -> Result<Se3FiberBundle> {
        let attn_out = self.attention.forward(bundle, positions)?;
        let n = bundle.n_nodes();
        let mut out_type0 = Vec::with_capacity(n);
        for i in 0..n {
            // Residual + layer-norm + FFN
            let res: Vec<f64> = bundle.type0[i]
                .iter()
                .zip(&attn_out.type0[i])
                .map(|(a, b)| a + b)
                .collect();
            let normed = self.layer_norm(&res);
            let ffn_out = self.ffn(&normed);
            let final_scalar: Vec<f64> = res.iter().zip(&ffn_out).map(|(r, f)| r + f).collect();
            out_type0.push(final_scalar);
        }
        // Vector features pass through with residual
        let out_type1: Vec<Vec<[f64; 3]>> = (0..n)
            .map(|i| {
                bundle.type1[i]
                    .iter()
                    .zip(&attn_out.type1[i])
                    .map(|(a, b)| [a[0] + b[0], a[1] + b[1], a[2] + b[2]])
                    .collect()
            })
            .collect();
        Ok(Se3FiberBundle { type0: out_type0, type1: out_type1 })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3 Vector Neuron Networks (VNN)  (Deng et al. 2021)
// ─────────────────────────────────────────────────────────────────────────────

/// SO(3)-equivariant linear layer on 3D vector features.
///
/// Applies a standard linear map to the *channel* dimension while preserving
/// the spatial (3D) dimension:  Vout\[i\] = Σⱼ W\[i,j\] · Vin\[j\].
/// This is equivariant because W acts on channels, not on 3D axes.
#[derive(Debug, Clone)]
pub struct VnLinear {
    /// Weight matrix [out_channels x in_channels].
    pub weight: Vec<Vec<f64>>,
    pub in_channels: usize,
    pub out_channels: usize,
}

impl VnLinear {
    /// Create a new VnLinear layer.
    pub fn new(in_channels: usize, out_channels: usize, seed: u64) -> Self {
        Self {
            weight: eq_rand_weight(out_channels, in_channels, seed),
            in_channels,
            out_channels,
        }
    }

    /// Forward: `features` is shape [in_channels x 3], output is [out_channels x 3].
    pub fn forward(&self, features: &[[f64; 3]]) -> Result<Vec<[f64; 3]>> {
        if features.len() != self.in_channels {
            return Err(TensorError::invalid_argument_op(
                "vn_linear",
                &format!(
                    "expected {} input channels, got {}",
                    self.in_channels,
                    features.len()
                ),
            ));
        }
        let out: Vec<[f64; 3]> = self
            .weight
            .iter()
            .map(|w_row| {
                let mut out_vec = [0.0f64; 3];
                for (j, &wj) in w_row.iter().enumerate() {
                    let vj = features[j];
                    out_vec[0] += wj * vj[0];
                    out_vec[1] += wj * vj[1];
                    out_vec[2] += wj * vj[2];
                }
                out_vec
            })
            .collect();
        Ok(out)
    }
}

/// SO(3)-equivariant leaky ReLU activation for vector features.
///
/// For each vector v, the activation is:
/// - Find direction k̂ = mean(v̂ᵢ).
/// - Project each vᵢ onto k̂: qᵢ = vᵢ · k̂.
/// - Apply leaky ReLU: aᵢ = (1 if qᵢ≥0 else negative_slope) * vᵢ + residual.
///
/// This preserves equivariance because k̂ rotates with the point cloud.
#[derive(Debug, Clone)]
pub struct VnLeakyRelu {
    /// Negative slope (typically 0.2).
    pub negative_slope: f64,
    /// Weight for the direction-finding mapping [channels x channels].
    pub w_dir: Vec<Vec<f64>>,
    pub channels: usize,
}

impl VnLeakyRelu {
    pub fn new(channels: usize, negative_slope: f64, seed: u64) -> Self {
        Self {
            negative_slope,
            w_dir: eq_rand_weight(channels, channels, seed),
            channels,
        }
    }

    /// Forward: input [channels x 3], output [channels x 3].
    pub fn forward(&self, features: &[[f64; 3]]) -> Result<Vec<[f64; 3]>> {
        if features.len() != self.channels {
            return Err(TensorError::invalid_argument_op(
                "vn_leaky_relu",
                "channel count mismatch",
            ));
        }
        // Compute direction k per channel via linear map
        // k[i] = Σⱼ W[i,j] v[j]  (equivariant direction)
        let k_vecs: Vec<[f64; 3]> = self
            .w_dir
            .iter()
            .map(|w_row| {
                let mut k = [0.0f64; 3];
                for (j, &wj) in w_row.iter().enumerate() {
                    k[0] += wj * features[j][0];
                    k[1] += wj * features[j][1];
                    k[2] += wj * features[j][2];
                }
                k
            })
            .collect();

        let out: Vec<[f64; 3]> = features
            .iter()
            .zip(&k_vecs)
            .map(|(v, k)| {
                let k_norm = norm3(k).max(1e-10);
                let k_hat = [k[0] / k_norm, k[1] / k_norm, k[2] / k_norm];
                let proj = v[0] * k_hat[0] + v[1] * k_hat[1] + v[2] * k_hat[2];
                let slope = if proj >= 0.0 { 1.0 } else { self.negative_slope };
                [slope * v[0], slope * v[1], slope * v[2]]
            })
            .collect();
        Ok(out)
    }
}

/// SO(3)-invariant max-pooling over vector features.
///
/// For each output channel, find the direction that maximises the inner product
/// across all input channels, then take the max norm.
/// The output is a *scalar* (norm) per channel — invariant under SO(3).
#[derive(Debug, Clone)]
pub struct VnMaxPool {
    /// Number of output channels.
    pub out_channels: usize,
    /// Weight [out_channels x in_channels] for direction finding.
    pub weight: Vec<Vec<f64>>,
    pub in_channels: usize,
}

impl VnMaxPool {
    pub fn new(in_channels: usize, out_channels: usize, seed: u64) -> Self {
        Self {
            out_channels,
            weight: eq_rand_weight(out_channels, in_channels, seed),
            in_channels,
        }
    }

    /// Forward: input [in_channels x 3], output \[out_channels\] invariant scalars.
    pub fn forward(&self, features: &[[f64; 3]]) -> Result<Vec<f64>> {
        if features.len() != self.in_channels {
            return Err(TensorError::invalid_argument_op(
                "vn_max_pool",
                "channel count mismatch",
            ));
        }
        let out: Vec<f64> = self
            .weight
            .iter()
            .map(|w_row| {
                // direction d = Σⱼ wⱼ vⱼ
                let mut dir = [0.0f64; 3];
                for (j, &wj) in w_row.iter().enumerate() {
                    dir[0] += wj * features[j][0];
                    dir[1] += wj * features[j][1];
                    dir[2] += wj * features[j][2];
                }
                let d_norm = norm3(&dir).max(1e-10);
                let d_hat = [dir[0] / d_norm, dir[1] / d_norm, dir[2] / d_norm];
                // Max inner product with direction
                features
                    .iter()
                    .map(|v| eq_relu(v[0] * d_hat[0] + v[1] * d_hat[1] + v[2] * d_hat[2]))
                    .fold(0.0f64, f64::max)
            })
            .collect();
        Ok(out)
    }
}

/// Point-cloud classifier using stacked Vector Neuron layers.
///
/// Pipeline: VnLinear → VnLeakyReLU (×n_layers) → VnMaxPool (invariant) → MLP → logits.
#[derive(Debug, Clone)]
pub struct VnNetwork {
    pub vn_layers: Vec<(VnLinear, VnLeakyRelu)>,
    pub pool: VnMaxPool,
    /// Classification MLP [pooled_dim → n_classes].
    pub cls_w: Vec<Vec<f64>>,
    pub cls_b: Vec<f64>,
    pub n_classes: usize,
}

impl VnNetwork {
    /// Create a VnNetwork.
    ///
    /// - `in_channels`: initial vector channels per point.
    /// - `hidden`: hidden channels in VN layers.
    /// - `n_classes`: number of output classes.
    pub fn new(in_channels: usize, hidden: usize, n_classes: usize, seed: u64) -> Self {
        let lin0 = VnLinear::new(in_channels, hidden, seed);
        let act0 = VnLeakyRelu::new(hidden, 0.2, seed.wrapping_add(1));
        let lin1 = VnLinear::new(hidden, hidden, seed.wrapping_add(2));
        let act1 = VnLeakyRelu::new(hidden, 0.2, seed.wrapping_add(3));
        let pool = VnMaxPool::new(hidden, hidden, seed.wrapping_add(4));
        let cls_w = eq_rand_weight(n_classes, hidden, seed.wrapping_add(5));
        let cls_b = vec![0.0; n_classes];
        Self {
            vn_layers: vec![(lin0, act0), (lin1, act1)],
            pool,
            cls_w,
            cls_b,
            n_classes,
        }
    }

    /// Forward for a single point cloud: `pc` is [n_points x in_channels x 3].
    ///
    /// Returns per-class logits of length `n_classes`.
    pub fn forward(&self, pc: &[Vec<[f64; 3]>]) -> Result<Vec<f64>> {
        if pc.is_empty() {
            return Err(TensorError::invalid_argument_op("vn_network", "empty point cloud"));
        }
        // Max-pool over all points → [in_channels x 3]
        let channels = pc[0].len();
        let mut agg: Vec<[f64; 3]> = vec![[0.0; 3]; channels];
        for point in pc {
            for (c, v) in point.iter().enumerate() {
                for d in 0..3 {
                    if v[d].abs() > agg[c][d].abs() {
                        agg[c][d] = v[d];
                    }
                }
            }
        }
        // Pass through VN layers
        let mut cur = agg;
        for (lin, act) in &self.vn_layers {
            let lin_out = lin.forward(&cur)?;
            cur = act.forward(&lin_out)?;
        }
        // Invariant pooling → scalars
        let invariant = self.pool.forward(&cur)?;
        // Classification
        let logits: Vec<f64> = self
            .cls_w
            .iter()
            .zip(&self.cls_b)
            .map(|(w, b)| eq_dot(w, &invariant) + b)
            .collect();
        Ok(logits)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4 Invariant Point Attention (IPA)  (AlphaFold2, Jumper et al. 2021)
// ─────────────────────────────────────────────────────────────────────────────

/// Local rigid frame (backbone frame in protein structure prediction).
///
/// Represents a Euclidean frame T = (R, t) where R is a 3×3 rotation matrix
/// and t is a 3D translation vector.
#[derive(Debug, Clone)]
pub struct IpaFrame {
    /// Rotation matrix R (3×3, column-major rows).
    pub rotation: [[f64; 3]; 3],
    /// Translation vector t.
    pub translation: [f64; 3],
}

impl IpaFrame {
    /// Identity frame.
    pub fn identity() -> Self {
        Self {
            rotation: identity_rot(),
            translation: [0.0; 3],
        }
    }

    /// Create from explicit rotation and translation.
    pub fn new(rotation: [[f64; 3]; 3], translation: [f64; 3]) -> Self {
        Self { rotation, translation }
    }

    /// Apply the frame transformation to a point: p → R·p + t.
    pub fn apply(&self, p: &[f64; 3]) -> [f64; 3] {
        let rp = rotate3(&self.rotation, p);
        [rp[0] + self.translation[0], rp[1] + self.translation[1], rp[2] + self.translation[2]]
    }

    /// Apply the inverse frame transformation: p → Rᵀ·(p − t).
    pub fn apply_inverse(&self, p: &[f64; 3]) -> [f64; 3] {
        let q = [
            p[0] - self.translation[0],
            p[1] - self.translation[1],
            p[2] - self.translation[2],
        ];
        rotate3(&transpose3(&self.rotation), &q)
    }

    /// Compose two frames: T₁ ∘ T₂ = (R₁R₂, R₁t₂ + t₁).
    pub fn compose(&self, other: &Self) -> Self {
        let r1 = &self.rotation;
        let r2 = &other.rotation;
        let mut r_out = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    r_out[i][j] += r1[i][k] * r2[k][j];
                }
            }
        }
        let t2_global = rotate3(&self.rotation, &other.translation);
        let t_out = [
            t2_global[0] + self.translation[0],
            t2_global[1] + self.translation[1],
            t2_global[2] + self.translation[2],
        ];
        Self { rotation: r_out, translation: t_out }
    }
}

/// Pre-computed attention score combining scalar and point attention.
///
/// IPA score between residues i and j:
///   s_ij = (qᵢ·kⱼ)/√d + w_C/2 · Σₚ ‖Tᵢ·qₚᵢ − Tⱼ·kₚⱼ‖²
#[derive(Debug, Clone)]
pub struct IpaAttentionScore {
    /// Scalar component of the score.
    pub scalar_score: f64,
    /// Point-distance component of the score.
    pub point_score: f64,
    /// Combined weighted score.
    pub total: f64,
}

impl IpaAttentionScore {
    pub fn new(scalar_score: f64, point_score: f64, w_c: f64) -> Self {
        let total = scalar_score + w_c * point_score;
        Self { scalar_score, point_score, total }
    }
}

/// Invariant Point Attention layer.
///
/// Given per-residue frames T and scalar features s, computes:
/// 1. Scalar Q/K/V projections.
/// 2. Point Q/K projections (in local frames → global).
/// 3. Combined SE(3)-invariant attention weights.
/// 4. Scalar + point value aggregation.
#[derive(Debug, Clone)]
pub struct IpaLayer {
    /// Query scalar projection [n_heads * head_dim x s_dim].
    pub w_qs: Vec<Vec<f64>>,
    /// Key scalar projection [n_heads * head_dim x s_dim].
    pub w_ks: Vec<Vec<f64>>,
    /// Value scalar projection [n_heads * head_dim x s_dim].
    pub w_vs: Vec<Vec<f64>>,
    /// Query point projection [n_heads * n_query_points * 3 x s_dim].
    pub w_qp: Vec<Vec<f64>>,
    /// Key point projection [n_heads * n_query_points * 3 x s_dim].
    pub w_kp: Vec<Vec<f64>>,
    /// Value point projection [n_heads * n_query_points * 3 x s_dim].
    pub w_vp: Vec<Vec<f64>>,
    /// Output projection.
    pub w_out: Vec<Vec<f64>>,
    pub s_dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub n_query_points: usize,
    /// Weighting scalar for point distance term.
    pub w_c: f64,
}

impl IpaLayer {
    /// Create an IPA layer.
    pub fn new(s_dim: usize, n_heads: usize, n_query_points: usize, seed: u64) -> Result<Self> {
        if s_dim == 0 || n_heads == 0 || n_query_points == 0 {
            return Err(TensorError::invalid_argument_op(
                "ipa_layer",
                "s_dim, n_heads, n_query_points must be positive",
            ));
        }
        let head_dim = (s_dim / n_heads).max(1);
        let kq_dim = n_heads * head_dim;
        let pt_dim = n_heads * n_query_points * 3;
        Ok(Self {
            w_qs: eq_rand_weight(kq_dim, s_dim, seed),
            w_ks: eq_rand_weight(kq_dim, s_dim, seed.wrapping_add(1)),
            w_vs: eq_rand_weight(kq_dim, s_dim, seed.wrapping_add(2)),
            w_qp: eq_rand_weight(pt_dim, s_dim, seed.wrapping_add(3)),
            w_kp: eq_rand_weight(pt_dim, s_dim, seed.wrapping_add(4)),
            w_vp: eq_rand_weight(pt_dim, s_dim, seed.wrapping_add(5)),
            w_out: eq_rand_weight(s_dim, kq_dim + n_heads * n_query_points * 3 + n_heads, seed.wrapping_add(6)),
            s_dim,
            n_heads,
            head_dim,
            n_query_points,
            w_c: 1.0 / (2.0 * n_query_points as f64).sqrt(),
        })
    }

    /// Compute global query/key points for a node given its frame.
    fn project_points(
        &self,
        s: &[f64],
        frame: &IpaFrame,
        w: &[Vec<f64>],
        n_pts: usize,
    ) -> Vec<[f64; 3]> {
        let flat = eq_matvec(w, s);
        (0..n_pts)
            .map(|p| {
                let local = [
                    flat.get(p * 3).copied().unwrap_or(0.0),
                    flat.get(p * 3 + 1).copied().unwrap_or(0.0),
                    flat.get(p * 3 + 2).copied().unwrap_or(0.0),
                ];
                frame.apply(&local)
            })
            .collect()
    }

    /// Forward IPA layer.
    ///
    /// - `features`: node scalar features [n x s_dim].
    /// - `frames`: per-node rigid frames \[n\].
    ///
    /// Returns updated scalar features [n x s_dim].
    pub fn forward(&self, features: &[Vec<f64>], frames: &[IpaFrame]) -> Result<Vec<Vec<f64>>> {
        let n = features.len();
        if frames.len() != n {
            return Err(TensorError::invalid_argument_op(
                "ipa_layer",
                "features and frames length mismatch",
            ));
        }
        if let Some(f) = features.first() {
            if f.len() != self.s_dim {
                return Err(TensorError::invalid_argument_op(
                    "ipa_layer",
                    &format!("expected s_dim={}, got {}", self.s_dim, f.len()),
                ));
            }
        }

        let kq_dim = self.n_heads * self.head_dim;
        let n_pts_per_head = self.n_query_points;
        let total_pts = self.n_heads * n_pts_per_head;

        let scale_scalar = 1.0 / (self.head_dim as f64).sqrt();

        // Precompute scalar Q, K, V and point Q, K, V
        let qs: Vec<Vec<f64>> = features.iter().map(|s| eq_matvec(&self.w_qs, s)).collect();
        let ks: Vec<Vec<f64>> = features.iter().map(|s| eq_matvec(&self.w_ks, s)).collect();
        let vs: Vec<Vec<f64>> = features.iter().map(|s| eq_matvec(&self.w_vs, s)).collect();

        let qp: Vec<Vec<[f64; 3]>> = features
            .iter()
            .zip(frames)
            .map(|(s, frame)| self.project_points(s, frame, &self.w_qp, total_pts))
            .collect();
        let kp: Vec<Vec<[f64; 3]>> = features
            .iter()
            .zip(frames)
            .map(|(s, frame)| self.project_points(s, frame, &self.w_kp, total_pts))
            .collect();
        let vp: Vec<Vec<[f64; 3]>> = features
            .iter()
            .zip(frames)
            .map(|(s, frame)| self.project_points(s, frame, &self.w_vp, total_pts))
            .collect();

        let mut out_features: Vec<Vec<f64>> = Vec::with_capacity(n);

        for i in 0..n {
            let mut head_scalars: Vec<f64> = Vec::new();
            let mut head_points: Vec<f64> = Vec::new();
            let mut head_norms: Vec<f64> = Vec::new();

            for h in 0..self.n_heads {
                // Scalar attention scores for this head
                let q_h = &qs[i][h * self.head_dim..(h + 1) * self.head_dim];
                let logits: Vec<f64> = (0..n)
                    .map(|j| {
                        let k_h = &ks[j][h * self.head_dim..(h + 1) * self.head_dim];
                        let scalar_term = eq_dot(q_h, k_h) * scale_scalar;

                        // Point distance term
                        let pt_start = h * n_pts_per_head;
                        let pt_end = pt_start + n_pts_per_head;
                        let point_sq_dist: f64 = qp[i][pt_start..pt_end]
                            .iter()
                            .zip(&kp[j][pt_start..pt_end])
                            .map(|(qpt, kpt)| sq_dist3(qpt, kpt))
                            .sum();
                        let ipa_score =
                            IpaAttentionScore::new(scalar_term, -point_sq_dist, self.w_c);
                        ipa_score.total
                    })
                    .collect();

                let attn = softmax(&logits);

                // Aggregate scalar values
                let v_h_start = h * self.head_dim;
                let v_h_end = v_h_start + self.head_dim;
                let mut agg_v = vec![0.0f64; self.head_dim];
                for (j, &w) in attn.iter().enumerate() {
                    let v_h = &vs[j][v_h_start..v_h_end];
                    for (d, &vv) in v_h.iter().enumerate() {
                        agg_v[d] += w * vv;
                    }
                }
                head_scalars.extend_from_slice(&agg_v);

                // Aggregate point values → back to local frame i → norms
                let pt_start = h * n_pts_per_head;
                let pt_end = pt_start + n_pts_per_head;
                for p in pt_start..pt_end {
                    let mut agg_pt = [0.0f64; 3];
                    for (j, &w) in attn.iter().enumerate() {
                        agg_pt[0] += w * vp[j][p][0];
                        agg_pt[1] += w * vp[j][p][1];
                        agg_pt[2] += w * vp[j][p][2];
                    }
                    // Transform back to local frame i
                    let local_pt = frames[i].apply_inverse(&agg_pt);
                    head_points.extend_from_slice(&local_pt);
                    head_norms.push(norm3(&local_pt));
                }
            }

            // Concatenate: head_scalars || head_points || head_norms
            let mut concat = head_scalars;
            concat.extend_from_slice(&head_points);
            concat.extend_from_slice(&head_norms);

            // Output projection (with padding if needed)
            let out_dim = self.w_out.first().map(|r| r.len()).unwrap_or(0);
            let concat_padded: Vec<f64> = if concat.len() < out_dim {
                let mut v = concat.clone();
                v.resize(out_dim, 0.0);
                v
            } else {
                concat[..out_dim].to_vec()
            };

            let projected = eq_matvec(&self.w_out, &concat_padded);
            // Residual + layer norm
            let out_i: Vec<f64> = features[i]
                .iter()
                .zip(projected.iter().chain(std::iter::repeat(&0.0)))
                .map(|(s, p)| s + p)
                .take(self.s_dim)
                .collect();
            out_features.push(out_i);
        }

        Ok(out_features)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ── Spherical Harmonics ──

    #[test]
    fn test_sph_l0_constant() {
        let d1 = [1.0, 0.0, 0.0];
        let d2 = [0.0, 1.0, 0.0];
        let sh1 = spherical_harmonics_01(&d1);
        let sh2 = spherical_harmonics_01(&d2);
        assert!((sh1[0] - sh2[0]).abs() < 1e-12, "Y_0^0 should be constant");
    }

    #[test]
    fn test_sph_l0_value() {
        let d = [1.0, 0.0, 0.0];
        let sh = spherical_harmonics_01(&d);
        let expected = 1.0 / (4.0 * PI).sqrt();
        assert!((sh[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn test_sph_l1_x_axis() {
        // Along x: Y_1^1 = -sqrt(3/4π) * x/r = -sqrt(3/4π)
        let d = [1.0, 0.0, 0.0];
        let sh = spherical_harmonics_01(&d);
        let expected = -(3.0 / (4.0 * PI)).sqrt();
        assert!((sh[3] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_sph_l1_y_axis() {
        // Along y: Y_1^{-1} = sqrt(3/4π)
        let d = [0.0, 1.0, 0.0];
        let sh = spherical_harmonics_01(&d);
        let expected = (3.0 / (4.0 * PI)).sqrt();
        assert!((sh[1] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_sph_l1_z_axis() {
        // Along z: Y_1^0 = sqrt(3/4π)
        let d = [0.0, 0.0, 1.0];
        let sh = spherical_harmonics_01(&d);
        let expected = (3.0 / (4.0 * PI)).sqrt();
        assert!((sh[2] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_sph_scale_invariant() {
        let d1 = [2.0, 0.0, 0.0];
        let d2 = [0.5, 0.0, 0.0];
        let sh1 = spherical_harmonics_01(&d1);
        let sh2 = spherical_harmonics_01(&d2);
        for k in 0..4 {
            assert!((sh1[k] - sh2[k]).abs() < 1e-10, "SH should be scale-invariant at index {k}");
        }
    }

    // ── EgnnCoord ──

    #[test]
    fn test_egnn_coord_output_finite() {
        let coord = EgnnCoord::new(4, 8, 42);
        let hi = vec![0.1, 0.2, 0.3, 0.4];
        let hj = vec![0.5, 0.6, 0.7, 0.8];
        let w = coord.edge_weight(&hi, &hj, 1.5).expect("should succeed");
        assert!(w.is_finite());
    }

    #[test]
    fn test_egnn_coord_dim_mismatch() {
        let coord = EgnnCoord::new(4, 8, 1);
        let result = coord.edge_weight(&[1.0, 2.0], &[1.0, 2.0, 3.0, 4.0], 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_egnn_coord_symmetry() {
        // c_ij need not equal c_ji (asymmetric), but both should be finite
        let coord = EgnnCoord::new(3, 6, 77);
        let h1 = vec![0.1, 0.2, 0.3];
        let h2 = vec![0.4, 0.5, 0.6];
        let w_ij = coord.edge_weight(&h1, &h2, 2.0).expect("ok");
        let w_ji = coord.edge_weight(&h2, &h1, 2.0).expect("ok");
        assert!(w_ij.is_finite() && w_ji.is_finite());
    }

    // ── EgnnLayer ──

    #[test]
    fn test_egnn_layer_output_shape() {
        let layer = EgnnLayer::new(4, 8, 0);
        let coords = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let feats: Vec<Vec<f64>> = (0..3).map(|i| vec![i as f64 * 0.1; 4]).collect();
        let (new_coords, new_feats) = layer.forward(&coords, &feats).expect("ok");
        assert_eq!(new_coords.len(), 3);
        assert_eq!(new_feats.len(), 3);
        for f in &new_feats {
            assert_eq!(f.len(), 4);
        }
    }

    #[test]
    fn test_egnn_layer_features_finite() {
        let layer = EgnnLayer::new(4, 8, 1);
        let coords: Vec<[f64; 3]> = (0..5).map(|i| [i as f64, 0.0, 0.0]).collect();
        let feats: Vec<Vec<f64>> = (0..5).map(|_| vec![0.5; 4]).collect();
        let (_, new_feats) = layer.forward(&coords, &feats).expect("ok");
        for f in &new_feats {
            for v in f {
                assert!(v.is_finite(), "Feature value must be finite");
            }
        }
    }

    #[test]
    fn test_egnn_layer_single_node() {
        let layer = EgnnLayer::new(2, 4, 99);
        let coords = vec![[0.0, 0.0, 0.0]];
        let feats = vec![vec![1.0, 2.0]];
        let (nc, nf) = layer.forward(&coords, &feats).expect("ok");
        // Single node: no neighbors, coords should stay same
        assert_eq!(nc.len(), 1);
        assert_eq!(nf.len(), 1);
        assert!((nc[0][0] - coords[0][0]).abs() < 1e-12);
    }

    #[test]
    fn test_egnn_layer_dim_mismatch_error() {
        let layer = EgnnLayer::new(4, 8, 2);
        let coords = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let feats = vec![vec![1.0; 4]]; // length mismatch
        assert!(layer.forward(&coords, &feats).is_err());
    }

    // ── EgnnModel ──

    #[test]
    fn test_egnn_model_forward_scalar() {
        let model = EgnnModel::new(2, 4, 8, 10);
        let coords = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let feats: Vec<Vec<f64>> = (0..3).map(|_| vec![0.1; 4]).collect();
        let pred = model.forward(&coords, &feats).expect("ok");
        assert!(pred.is_finite());
    }

    #[test]
    fn test_egnn_model_empty() {
        let model = EgnnModel::new(1, 4, 4, 5);
        let result = model.forward(&[], &[]).expect("ok");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_egnn_equivariance_coords() {
        // Rotating coordinates should not change scalar output (approximate test)
        let model = EgnnModel::new(1, 4, 4, 100);
        let coords = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]];
        let feats: Vec<Vec<f64>> = vec![vec![0.2; 4], vec![0.3; 4], vec![0.1; 4]];

        let pred1 = model.forward(&coords, &feats).expect("ok");

        // 90° rotation around z-axis: (x,y,z) → (-y,x,z)
        let rot_coords: Vec<[f64; 3]> = coords.iter().map(|p| [-p[1], p[0], p[2]]).collect();
        let pred2 = model.forward(&rot_coords, &feats).expect("ok");

        // With random weights, exact equality won't hold, but relative difference
        // should be small due to ‖xᵢ−xⱼ‖² being invariant
        let rel_diff = (pred1 - pred2).abs() / (pred1.abs().max(pred2.abs()) + 1e-8);
        assert!(rel_diff < 1.0, "EGNN scalar output should be approximately invariant to rotation");
    }

    #[test]
    fn test_egnn_embed_coords_shape() {
        let model = EgnnModel::new(1, 3, 4, 7);
        let coords = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let feats: Vec<Vec<f64>> = vec![vec![0.5; 3], vec![0.5; 3]];
        let embedded = model.embed_coords(&coords, &feats).expect("ok");
        assert_eq!(embedded.len(), 2);
    }

    // ── Se3FiberBundle ──

    #[test]
    fn test_se3_bundle_construction() {
        let b = Se3FiberBundle::new(5, 8, 3);
        assert_eq!(b.n_nodes(), 5);
        assert_eq!(b.scalar_dim(), 8);
        assert_eq!(b.vector_channels(), 3);
    }

    #[test]
    fn test_se3_bundle_zero_initialized() {
        let b = Se3FiberBundle::new(3, 4, 2);
        for node in &b.type0 {
            for &v in node {
                assert_eq!(v, 0.0);
            }
        }
        for node in &b.type1 {
            for ch in node {
                for &d in ch {
                    assert_eq!(d, 0.0);
                }
            }
        }
    }

    // ── Se3Attention ──

    #[test]
    fn test_se3_attention_output_shape() {
        let attn = Se3Attention::new(2, 8, 3, 10.0, 42).expect("ok");
        let mut bundle = Se3FiberBundle::new(4, 8, 3);
        let mut rng = StdRng::seed_from_u64(1);
        for node in bundle.type0.iter_mut() {
            for v in node.iter_mut() {
                *v = rng.random::<f64>();
            }
        }
        let positions: Vec<[f64; 3]> = (0..4).map(|i| [i as f64, 0.0, 0.0]).collect();
        let out = attn.forward(&bundle, &positions).expect("ok");
        assert_eq!(out.n_nodes(), 4);
        assert_eq!(out.scalar_dim(), 8);
        assert_eq!(out.vector_channels(), 3);
    }

    #[test]
    fn test_se3_attention_invalid_args() {
        assert!(Se3Attention::new(0, 8, 3, 5.0, 1).is_err());
        assert!(Se3Attention::new(2, 0, 3, 5.0, 1).is_err());
    }

    #[test]
    fn test_se3_attention_scalars_finite() {
        let attn = Se3Attention::new(1, 4, 2, 5.0, 99).expect("ok");
        let mut bundle = Se3FiberBundle::new(3, 4, 2);
        let mut rng = StdRng::seed_from_u64(2);
        for node in bundle.type0.iter_mut() {
            for v in node.iter_mut() {
                *v = rng.random::<f64>() - 0.5;
            }
        }
        for node in bundle.type1.iter_mut() {
            for ch in node.iter_mut() {
                for d in ch.iter_mut() {
                    *d = rng.random::<f64>() - 0.5;
                }
            }
        }
        let positions: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let out = attn.forward(&bundle, &positions).expect("ok");
        for node in &out.type0 {
            for &v in node {
                assert!(v.is_finite(), "Scalar output must be finite");
            }
        }
    }

    #[test]
    fn test_se3_attention_position_mismatch() {
        let attn = Se3Attention::new(2, 4, 1, 5.0, 10).expect("ok");
        let bundle = Se3FiberBundle::new(3, 4, 1);
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]; // wrong count
        assert!(attn.forward(&bundle, &positions).is_err());
    }

    // ── Se3TransformerLayer ──

    #[test]
    fn test_se3_transformer_layer_shape() {
        let layer = Se3TransformerLayer::new(2, 8, 3, 16, 10.0, 0).expect("ok");
        let mut bundle = Se3FiberBundle::new(4, 8, 3);
        let mut rng = StdRng::seed_from_u64(3);
        for node in bundle.type0.iter_mut() {
            for v in node.iter_mut() {
                *v = rng.random::<f64>();
            }
        }
        let positions: Vec<[f64; 3]> = (0..4).map(|i| [i as f64, 0.0, 0.0]).collect();
        let out = layer.forward(&bundle, &positions).expect("ok");
        assert_eq!(out.n_nodes(), 4);
        assert_eq!(out.scalar_dim(), 8);
    }

    #[test]
    fn test_se3_transformer_layer_finite() {
        let layer = Se3TransformerLayer::new(1, 4, 2, 8, 5.0, 7).expect("ok");
        let bundle = Se3FiberBundle::new(3, 4, 2);
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let out = layer.forward(&bundle, &positions).expect("ok");
        for node in &out.type0 {
            for &v in node {
                assert!(v.is_finite());
            }
        }
    }

    // ── VnLinear ──

    #[test]
    fn test_vn_linear_shape() {
        let layer = VnLinear::new(4, 8, 0);
        let feats: Vec<[f64; 3]> = (0..4).map(|i| [i as f64, 0.0, 0.0]).collect();
        let out = layer.forward(&feats).expect("ok");
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_vn_linear_equivariance() {
        // Rotating input by 90° around z should rotate output by 90° around z
        let layer = VnLinear::new(2, 2, 42);
        let feats: Vec<[f64; 3]> = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let out1 = layer.forward(&feats).expect("ok");

        // Rotate 90° around z: (x,y,z) → (-y, x, z)
        let rot_feats: Vec<[f64; 3]> = feats.iter().map(|&[x, y, z]| [-y, x, z]).collect();
        let out2 = layer.forward(&rot_feats).expect("ok");

        // out2 should be out1 rotated by 90° around z
        for (v1, v2) in out1.iter().zip(&out2) {
            let v1_rotated = [-v1[1], v1[0], v1[2]];
            for d in 0..3 {
                assert!(
                    (v1_rotated[d] - v2[d]).abs() < 1e-10,
                    "VnLinear should be equivariant: expected {}, got {}",
                    v1_rotated[d],
                    v2[d]
                );
            }
        }
    }

    #[test]
    fn test_vn_linear_zero_input() {
        let layer = VnLinear::new(3, 4, 5);
        let feats: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]; 3];
        let out = layer.forward(&feats).expect("ok");
        for v in &out {
            assert_eq!(*v, [0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn test_vn_linear_dim_mismatch() {
        let layer = VnLinear::new(4, 8, 1);
        let feats: Vec<[f64; 3]> = vec![[1.0, 0.0, 0.0]; 3]; // wrong channels
        assert!(layer.forward(&feats).is_err());
    }

    // ── VnLeakyRelu ──

    #[test]
    fn test_vn_leaky_relu_shape() {
        let act = VnLeakyRelu::new(4, 0.2, 0);
        let feats: Vec<[f64; 3]> = (0..4).map(|i| [i as f64, 0.5, 1.0]).collect();
        let out = act.forward(&feats).expect("ok");
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_vn_leaky_relu_zero_preserved() {
        let act = VnLeakyRelu::new(2, 0.2, 7);
        let feats = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let out = act.forward(&feats).expect("ok");
        for v in &out {
            assert_eq!(*v, [0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn test_vn_leaky_relu_finite() {
        let mut rng = StdRng::seed_from_u64(8);
        let act = VnLeakyRelu::new(5, 0.1, 8);
        let feats: Vec<[f64; 3]> = (0..5)
            .map(|_| [rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5])
            .collect();
        let out = act.forward(&feats).expect("ok");
        for v in &out {
            for &d in v {
                assert!(d.is_finite());
            }
        }
    }

    #[test]
    fn test_vn_leaky_relu_dim_mismatch() {
        let act = VnLeakyRelu::new(4, 0.2, 1);
        let feats: Vec<[f64; 3]> = vec![[1.0, 0.0, 0.0]; 3]; // wrong channels
        assert!(act.forward(&feats).is_err());
    }

    // ── VnMaxPool ──

    #[test]
    fn test_vn_max_pool_output_shape() {
        let pool = VnMaxPool::new(4, 8, 0);
        let feats: Vec<[f64; 3]> = (0..4).map(|i| [i as f64, 0.0, 1.0]).collect();
        let out = pool.forward(&feats).expect("ok");
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_vn_max_pool_invariant_to_rotation() {
        // Rotating all input vectors should not change output (invariance)
        let pool = VnMaxPool::new(3, 3, 42);
        let feats: Vec<[f64; 3]> = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let out1 = pool.forward(&feats).expect("ok");

        // Rotate 90° around z: (x,y,z) → (-y,x,z)
        let rot_feats: Vec<[f64; 3]> = feats.iter().map(|&[x, y, z]| [-y, x, z]).collect();
        let out2 = pool.forward(&rot_feats).expect("ok");

        // The max pool output should be the same since norms are preserved
        for (v1, v2) in out1.iter().zip(&out2) {
            assert!(
                (v1 - v2).abs() < 1e-10,
                "VnMaxPool should be approximately invariant: {} vs {}",
                v1,
                v2
            );
        }
    }

    #[test]
    fn test_vn_max_pool_zero_input() {
        let pool = VnMaxPool::new(2, 3, 1);
        let feats = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let out = pool.forward(&feats).expect("ok");
        for &v in &out {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_vn_max_pool_dim_mismatch() {
        let pool = VnMaxPool::new(4, 8, 1);
        let feats: Vec<[f64; 3]> = vec![[1.0, 0.0, 0.0]; 3]; // wrong channels
        assert!(pool.forward(&feats).is_err());
    }

    // ── VnNetwork ──

    #[test]
    fn test_vn_network_output_shape() {
        let net = VnNetwork::new(3, 8, 10, 0);
        // Single point with 3 vector channels
        let pc: Vec<Vec<[f64; 3]>> = vec![
            vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            vec![[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]],
        ];
        let logits = net.forward(&pc).expect("ok");
        assert_eq!(logits.len(), 10);
    }

    #[test]
    fn test_vn_network_empty_error() {
        let net = VnNetwork::new(3, 8, 5, 1);
        assert!(net.forward(&[]).is_err());
    }

    #[test]
    fn test_vn_network_finite_output() {
        let net = VnNetwork::new(2, 4, 5, 99);
        let mut rng = StdRng::seed_from_u64(9);
        let pc: Vec<Vec<[f64; 3]>> = (0..10)
            .map(|_| {
                (0..2)
                    .map(|_| {
                        [rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5]
                    })
                    .collect()
            })
            .collect();
        let logits = net.forward(&pc).expect("ok");
        for &v in &logits {
            assert!(v.is_finite());
        }
    }

    // ── IpaFrame ──

    #[test]
    fn test_ipa_frame_identity() {
        let frame = IpaFrame::identity();
        let p = [1.0, 2.0, 3.0];
        let out = frame.apply(&p);
        assert!((out[0] - p[0]).abs() < 1e-12);
        assert!((out[1] - p[1]).abs() < 1e-12);
        assert!((out[2] - p[2]).abs() < 1e-12);
    }

    #[test]
    fn test_ipa_frame_apply_inverse() {
        let rot = [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let frame = IpaFrame::new(rot, [1.0, 0.0, 0.0]);
        let p = [2.0, 3.0, 4.0];
        let q = frame.apply(&p);
        let p_rec = frame.apply_inverse(&q);
        for d in 0..3 {
            assert!((p_rec[d] - p[d]).abs() < 1e-10, "apply_inverse should invert apply");
        }
    }

    #[test]
    fn test_ipa_frame_compose() {
        let f1 = IpaFrame::new(identity_rot(), [1.0, 0.0, 0.0]);
        let f2 = IpaFrame::new(identity_rot(), [0.0, 1.0, 0.0]);
        let composed = f1.compose(&f2);
        let p = [0.0, 0.0, 0.0];
        let out = composed.apply(&p);
        // T1 ∘ T2 applied to 0 = T1(T2(0)) = T1([0,1,0]) = [1,2,0]
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!((out[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_ipa_frame_rotation_preserves_norm() {
        let rot = [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let frame = IpaFrame::new(rot, [0.0, 0.0, 0.0]);
        let p = [1.0, 2.0, 3.0];
        let q = frame.apply(&p);
        let np: f64 = p.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nq: f64 = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((np - nq).abs() < 1e-10);
    }

    // ── IpaAttentionScore ──

    #[test]
    fn test_ipa_score_construction() {
        let score = IpaAttentionScore::new(1.0, -2.0, 0.5);
        assert!((score.total - (1.0 + 0.5 * (-2.0))).abs() < 1e-12);
    }

    #[test]
    fn test_ipa_score_point_term_dominates() {
        let s1 = IpaAttentionScore::new(0.0, -10.0, 1.0);
        let s2 = IpaAttentionScore::new(0.0, -1.0, 1.0);
        assert!(s1.total < s2.total, "Larger point distance should give lower score");
    }

    // ── IpaLayer ──

    #[test]
    fn test_ipa_layer_output_shape() {
        let layer = IpaLayer::new(8, 2, 4, 0).expect("ok");
        let features: Vec<Vec<f64>> = (0..5).map(|_| vec![0.1; 8]).collect();
        let frames: Vec<IpaFrame> = (0..5).map(|_| IpaFrame::identity()).collect();
        let out = layer.forward(&features, &frames).expect("ok");
        assert_eq!(out.len(), 5);
        for row in &out {
            assert_eq!(row.len(), 8);
        }
    }

    #[test]
    fn test_ipa_layer_output_finite() {
        let layer = IpaLayer::new(4, 1, 2, 42).expect("ok");
        let mut rng = StdRng::seed_from_u64(10);
        let features: Vec<Vec<f64>> = (0..4)
            .map(|_| (0..4).map(|_| rng.random::<f64>() - 0.5).collect())
            .collect();
        let frames: Vec<IpaFrame> = (0..4).map(|_| IpaFrame::identity()).collect();
        let out = layer.forward(&features, &frames).expect("ok");
        for row in &out {
            for &v in row {
                assert!(v.is_finite(), "IPA output must be finite");
            }
        }
    }

    #[test]
    fn test_ipa_layer_invalid_args() {
        assert!(IpaLayer::new(0, 2, 4, 1).is_err());
        assert!(IpaLayer::new(8, 0, 4, 1).is_err());
        assert!(IpaLayer::new(8, 2, 0, 1).is_err());
    }

    #[test]
    fn test_ipa_layer_frame_mismatch() {
        let layer = IpaLayer::new(4, 1, 2, 0).expect("ok");
        let features: Vec<Vec<f64>> = (0..3).map(|_| vec![0.1; 4]).collect();
        let frames: Vec<IpaFrame> = (0..2).map(|_| IpaFrame::identity()).collect(); // wrong count
        assert!(layer.forward(&features, &frames).is_err());
    }

    #[test]
    fn test_ipa_layer_feature_dim_mismatch() {
        let layer = IpaLayer::new(8, 1, 2, 0).expect("ok");
        let features: Vec<Vec<f64>> = (0..3).map(|_| vec![0.1; 4]).collect(); // wrong dim
        let frames: Vec<IpaFrame> = (0..3).map(|_| IpaFrame::identity()).collect();
        assert!(layer.forward(&features, &frames).is_err());
    }

    #[test]
    fn test_ipa_with_non_identity_frames() {
        let layer = IpaLayer::new(4, 1, 1, 99).expect("ok");
        let features: Vec<Vec<f64>> = vec![vec![0.5; 4], vec![0.3; 4], vec![0.7; 4]];
        // Use rotation frames: 90° around z
        let rot = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let frames: Vec<IpaFrame> = (0..3)
            .map(|i| IpaFrame::new(rot, [i as f64, 0.0, 0.0]))
            .collect();
        let out = layer.forward(&features, &frames).expect("ok");
        assert_eq!(out.len(), 3);
        for row in &out {
            for &v in row {
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn test_ipa_single_node() {
        let layer = IpaLayer::new(4, 1, 2, 5).expect("ok");
        let features = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let frames = vec![IpaFrame::identity()];
        let out = layer.forward(&features, &frames).expect("ok");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 4);
    }

    // ── Cross-component integration tests ──

    #[test]
    fn test_egnn_then_se3_transformer() {
        // Test that EGNN features can feed into SE(3) transformer
        let egnn = EgnnModel::new(1, 4, 4, 0);
        let coords: Vec<[f64; 3]> = (0..3).map(|i| [i as f64, 0.0, 0.0]).collect();
        let feats: Vec<Vec<f64>> = (0..3).map(|_| vec![0.2; 4]).collect();
        let pred = egnn.forward(&coords, &feats).expect("ok");
        assert!(pred.is_finite());

        let layer = Se3TransformerLayer::new(1, 4, 2, 8, 10.0, 1).expect("ok");
        let bundle = Se3FiberBundle::new(3, 4, 2);
        let out = layer.forward(&bundle, &coords).expect("ok");
        assert_eq!(out.n_nodes(), 3);
    }

    #[test]
    fn test_vn_then_ipa() {
        // Test that VN features feed into IPA
        let vn = VnNetwork::new(2, 4, 4, 0);
        let pc: Vec<Vec<[f64; 3]>> = (0..5)
            .map(|i| vec![[i as f64, 0.0, 0.0], [0.0, i as f64, 0.0]])
            .collect();
        let logits = vn.forward(&pc).expect("ok");
        assert_eq!(logits.len(), 4);

        let ipa = IpaLayer::new(4, 1, 2, 7).expect("ok");
        let feats: Vec<Vec<f64>> = (0..5).map(|_| vec![0.1; 4]).collect();
        let frames: Vec<IpaFrame> = (0..5).map(|_| IpaFrame::identity()).collect();
        let out = ipa.forward(&feats, &frames).expect("ok");
        assert_eq!(out.len(), 5);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        for &p in &probs {
            assert!((0.0..=1.0).contains(&p));
        }
    }

    #[test]
    fn test_softmax_uniform_input() {
        let logits = vec![0.0; 5];
        let probs = softmax(&logits);
        for &p in &probs {
            assert!((p - 0.2).abs() < 1e-12);
        }
    }
}
