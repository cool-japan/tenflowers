//! Geometric Deep Learning & Equivariant Networks.
//!
//! - Point Cloud: [`PointCloud`], [`KnnGraph`], [`PointNetLayer`], [`PointNetPlusPlus`], [`DgcnnLayer`]
//! - SE(3)-Equivariant: [`Vector3`], [`SO3Features`], [`TFNLayer`], [`EquivariantReadout`], [`SchNetEquivariant`]
//! - Mesh: [`TriangleMesh`], [`MeshConvLayer`], [`ChebMeshConv`], [`SurfacePooling`], [`MeshAutoEncoder`]
//! - Manifold: [`RiemannianOptimizer`], [`HyperbolicEmbedding`], [`LorentzModel`], [`HyperbolicLinear`], [`GeodesicDistance`]
//! - Topology: [`SimplicialComplex`], [`SimplicialConv`], [`PersistenceDiagram`], [`RipsFiltration`], [`TopoLoss`]
//! - EGNN/SE3/VNN/IPA: see [`equivariant`]

pub mod equivariant;
pub use equivariant::*;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn dist3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn matvec(mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    mat.iter().map(|row| dot(row, v)).collect()
}

fn normalize_vec(v: &mut [f64]) {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n > 1e-300 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (n, p, m) = (a.len(), b.first().map(|r| r.len()).unwrap_or(0), b.len());
    (0..n)
        .map(|i| {
            (0..p)
                .map(|j| {
                    (0..m)
                        .map(|k| a[i].get(k).copied().unwrap_or(0.0) * b[k][j])
                        .sum()
                })
                .collect()
        })
        .collect()
}

fn rand_weight(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f64>> {
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

// ─────────────────────────────────────────────────────────────────────────────
// §1 Point Cloud
// ─────────────────────────────────────────────────────────────────────────────

/// N points in 3D with optional per-point features.
#[derive(Debug, Clone)]
pub struct PointCloud {
    pub points: Vec<[f64; 3]>,
    pub features: Option<Vec<Vec<f64>>>,
}

impl PointCloud {
    pub fn new(points: Vec<[f64; 3]>) -> Self {
        Self {
            points,
            features: None,
        }
    }

    pub fn with_features(mut self, features: Vec<Vec<f64>>) -> Result<Self> {
        if features.len() != self.points.len() {
            return Err(TensorError::invalid_argument_op(
                "geometric_dl",
                &format!(
                    "features len {} != points len {}",
                    features.len(),
                    self.points.len()
                ),
            ));
        }
        self.features = Some(features);
        Ok(self)
    }

    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    pub fn feature_dim(&self) -> usize {
        self.features
            .as_ref()
            .and_then(|f| f.first())
            .map(|v| v.len())
            .unwrap_or(0)
    }
}

/// K-nearest neighbor graph builder.
#[derive(Debug, Clone)]
pub struct KnnGraph;

impl KnnGraph {
    pub fn build(pc: &PointCloud, k: usize) -> Vec<Vec<usize>> {
        let n = pc.num_points();
        let k_c = k.min(n.saturating_sub(1));
        pc.points
            .iter()
            .enumerate()
            .map(|(i, pi)| {
                let mut dists: Vec<(usize, f64)> = pc
                    .points
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(j, pj)| (j, dist3(pi, pj)))
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                dists.iter().take(k_c).map(|(j, _)| *j).collect()
            })
            .collect()
    }
}

/// Per-point MLP (shared weights).
#[derive(Debug, Clone)]
pub struct PointNetLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weight: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
}

impl PointNetLayer {
    pub fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        Self {
            in_dim,
            out_dim,
            weight: rand_weight(out_dim, in_dim, seed),
            bias: vec![0.0; out_dim],
        }
    }

    pub fn forward(&self, pc: &PointCloud) -> Result<Vec<Vec<f64>>> {
        let expected = 3 + pc.feature_dim();
        if self.in_dim != expected {
            return Err(TensorError::invalid_argument_op(
                "geometric_dl",
                &format!("PointNetLayer in_dim={} expected {}", self.in_dim, expected),
            ));
        }
        pc.points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut x = p.to_vec();
                if let Some(f) = &pc.features {
                    x.extend_from_slice(&f[i]);
                }
                let raw = matvec(&self.weight, &x);
                Ok(raw
                    .iter()
                    .zip(&self.bias)
                    .map(|(r, b)| relu(r + b))
                    .collect())
            })
            .collect()
    }
}

fn fps(pc: &PointCloud, n_samples: usize) -> Vec<usize> {
    let n = pc.num_points();
    if n_samples == 0 || n == 0 {
        return Vec::new();
    }
    let ns = n_samples.min(n);
    let mut selected = vec![0usize];
    let mut min_dists = vec![f64::INFINITY; n];
    for _ in 1..ns {
        let last = *selected.last().expect("non-empty");
        for (j, d) in min_dists.iter_mut().enumerate() {
            let dist = dist3(&pc.points[last], &pc.points[j]);
            if dist < *d {
                *d = dist;
            }
        }
        let next = min_dists
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        selected.push(next);
    }
    selected
}

fn ball_query(pc: &PointCloud, center: usize, radius: f64, max_pts: usize) -> Vec<usize> {
    pc.points
        .iter()
        .enumerate()
        .filter(|(j, pj)| *j != center && dist3(&pc.points[center], pj) <= radius)
        .take(max_pts)
        .map(|(j, _)| j)
        .collect()
}

/// Hierarchical PointNet++ (FPS + ball query + per-group PointNet).
#[derive(Debug, Clone)]
pub struct PointNetPlusPlus {
    pub n_samples: usize,
    pub radius: f64,
    pub max_pts: usize,
    pub local_layer: PointNetLayer,
}

impl PointNetPlusPlus {
    pub fn new(
        n_samples: usize,
        radius: f64,
        max_pts: usize,
        in_dim: usize,
        out_dim: usize,
        seed: u64,
    ) -> Self {
        Self {
            n_samples,
            radius,
            max_pts,
            local_layer: PointNetLayer::new(in_dim, out_dim, seed),
        }
    }

    pub fn forward(&self, pc: &PointCloud) -> Result<(PointCloud, Vec<Vec<f64>>)> {
        let anchors = fps(pc, self.n_samples);
        let out_dim = self.local_layer.out_dim;
        let mut group_features = Vec::with_capacity(anchors.len());
        for &a in &anchors {
            let neighbors = ball_query(pc, a, self.radius, self.max_pts);
            let indices = if neighbors.is_empty() {
                vec![a]
            } else {
                neighbors
            };
            let pts: Vec<[f64; 3]> = indices.iter().map(|&j| pc.points[j]).collect();
            let feats_opt = pc
                .features
                .as_ref()
                .map(|f| indices.iter().map(|&j| f[j].clone()).collect::<Vec<_>>());
            let group_pc = PointCloud {
                points: pts,
                features: feats_opt,
            };
            let feats = self.local_layer.forward(&group_pc)?;
            let mut pooled = vec![f64::NEG_INFINITY; out_dim];
            for fv in &feats {
                for (d, &val) in fv.iter().enumerate() {
                    if val > pooled[d] {
                        pooled[d] = val;
                    }
                }
            }
            for v in pooled.iter_mut() {
                if v.is_infinite() {
                    *v = 0.0;
                }
            }
            group_features.push(pooled);
        }
        let sampled_pc = PointCloud {
            points: anchors.iter().map(|&a| pc.points[a]).collect(),
            features: None,
        };
        Ok((sampled_pc, group_features))
    }
}

/// Dynamic Graph CNN — EdgeConv layer.
#[derive(Debug, Clone)]
pub struct DgcnnLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub k: usize,
    pub weight: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
}

impl DgcnnLayer {
    pub fn new(in_dim: usize, out_dim: usize, k: usize, seed: u64) -> Self {
        Self {
            in_dim,
            out_dim,
            k,
            weight: rand_weight(out_dim, 2 * in_dim, seed),
            bias: vec![0.0; out_dim],
        }
    }

    pub fn edge_conv(
        &self,
        features: &[Vec<f64>],
        knn_graph: &[Vec<usize>],
    ) -> Result<Vec<Vec<f64>>> {
        let n = features.len();
        if knn_graph.len() != n {
            return Err(TensorError::invalid_argument_op(
                "geometric_dl",
                &format!("features len {} != knn_graph len {}", n, knn_graph.len()),
            ));
        }
        (0..n)
            .map(|i| {
                let xi = &features[i];
                if xi.len() != self.in_dim {
                    return Err(TensorError::invalid_argument_op(
                        "geometric_dl",
                        &format!("feature dim {} != in_dim {}", xi.len(), self.in_dim),
                    ));
                }
                let fallback = vec![i];
                let neighbors = if knn_graph[i].is_empty() {
                    &fallback
                } else {
                    &knn_graph[i]
                };
                let mut agg = vec![f64::NEG_INFINITY; self.out_dim];
                for &j in neighbors {
                    let xj = &features[j.min(n - 1)];
                    let mut edge = xi.clone();
                    edge.extend(xj.iter().zip(xi.iter()).map(|(b, a)| b - a));
                    let raw = matvec(&self.weight, &edge);
                    for (d, r) in raw.iter().enumerate() {
                        let val = relu(r + self.bias[d]);
                        if val > agg[d] {
                            agg[d] = val;
                        }
                    }
                }
                for v in agg.iter_mut() {
                    if v.is_infinite() {
                        *v = 0.0;
                    }
                }
                Ok(agg)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2 SE(3)-Equivariant
// ─────────────────────────────────────────────────────────────────────────────

/// 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    pub fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
    pub fn add(&self, o: &Self) -> Self {
        Self {
            x: self.x + o.x,
            y: self.y + o.y,
            z: self.z + o.z,
        }
    }
    pub fn scale(&self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
    pub fn dot(&self, o: &Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    pub fn cross(&self, o: &Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }
    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    pub fn normalized(&self) -> Self {
        let n = self.norm();
        if n < 1e-300 {
            *self
        } else {
            self.scale(1.0 / n)
        }
    }
    pub fn as_array(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

impl From<[f64; 3]> for Vector3 {
    fn from(a: [f64; 3]) -> Self {
        Self::new(a[0], a[1], a[2])
    }
}

/// Type-0 (scalar) and type-1 (vector) SO(3) features.
#[derive(Debug, Clone)]
pub struct SO3Features {
    pub scalars: Vec<f64>,
    pub vectors: Vec<Vector3>,
}

impl SO3Features {
    pub fn new(scalars: Vec<f64>, vectors: Vec<Vector3>) -> Self {
        Self { scalars, vectors }
    }
    pub fn zero(ns: usize, nv: usize) -> Self {
        Self {
            scalars: vec![0.0; ns],
            vectors: vec![Vector3::zero(); nv],
        }
    }
    pub fn add(&self, o: &Self) -> Result<Self> {
        if self.scalars.len() != o.scalars.len() || self.vectors.len() != o.vectors.len() {
            return Err(TensorError::invalid_argument_op(
                "geometric_dl",
                "SO3Features dimension mismatch",
            ));
        }
        Ok(Self {
            scalars: self
                .scalars
                .iter()
                .zip(&o.scalars)
                .map(|(a, b)| a + b)
                .collect(),
            vectors: self
                .vectors
                .iter()
                .zip(&o.vectors)
                .map(|(a, b)| a.add(b))
                .collect(),
        })
    }
}

/// Tensor Field Network (TFN) — l=0 and l=1 irreps only.
#[derive(Debug, Clone)]
pub struct TFNLayer {
    pub n_rbf: usize,
    pub rbf_centres: Vec<f64>,
    pub rbf_gamma: f64,
    pub cutoff: f64,
    pub w_scalar: Vec<f64>,
    pub w_vector: Vec<f64>,
}

impl TFNLayer {
    pub fn new(n_rbf: usize, cutoff: f64, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let rbf_centres: Vec<f64> = (0..n_rbf)
            .map(|i| (i as f64 + 0.5) * cutoff / n_rbf as f64)
            .collect();
        let rbf_gamma = 1.0 / (cutoff / n_rbf as f64).powi(2);
        let w_scalar: Vec<f64> = (0..n_rbf).map(|_| rng.random::<f64>() * 0.1).collect();
        let w_vector: Vec<f64> = (0..n_rbf).map(|_| rng.random::<f64>() * 0.1).collect();
        Self {
            n_rbf,
            rbf_centres,
            rbf_gamma,
            cutoff,
            w_scalar,
            w_vector,
        }
    }

    fn rbf(&self, r: f64) -> Vec<f64> {
        self.rbf_centres
            .iter()
            .map(|&mu| (-self.rbf_gamma * (r - mu).powi(2)).exp())
            .collect()
    }

    pub fn basis(&self, r: f64, vec: &[f64; 3]) -> (f64, [f64; 3]) {
        let phi = self.rbf(r);
        let scalar = dot(&self.w_scalar, &phi);
        let w_vec_sum = dot(&self.w_vector, &phi);
        let norm = (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
            .sqrt()
            .max(1e-10);
        (
            scalar,
            [
                vec[0] / norm * w_vec_sum,
                vec[1] / norm * w_vec_sum,
                vec[2] / norm * w_vec_sum,
            ],
        )
    }

    pub fn forward(
        &self,
        positions: &[[f64; 3]],
        features_scalar: &[f64],
        features_vec: &[[f64; 3]],
    ) -> Result<(Vec<f64>, Vec<[f64; 3]>)> {
        let n = positions.len();
        if features_scalar.len() != n || features_vec.len() != n {
            return Err(TensorError::invalid_argument_op(
                "geometric_dl",
                "TFNLayer: position/feature length mismatch",
            ));
        }
        let mut out_scalar = vec![0.0f64; n];
        let mut out_vec = vec![[0.0f64; 3]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dv = [
                    positions[j][0] - positions[i][0],
                    positions[j][1] - positions[i][1],
                    positions[j][2] - positions[i][2],
                ];
                let r = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
                if r > self.cutoff {
                    continue;
                }
                let (sb, vb) = self.basis(r, &dv);
                out_scalar[i] += sb * features_scalar[j];
                for d in 0..3 {
                    out_vec[i][d] += vb[d] * features_scalar[j] + sb * features_vec[j][d];
                }
            }
        }
        Ok((out_scalar, out_vec))
    }
}

/// Pool equivariant features to graph level.
#[derive(Debug, Clone)]
pub struct EquivariantReadout {
    pub use_mean_for_vectors: bool,
}

impl EquivariantReadout {
    pub fn new(use_mean_for_vectors: bool) -> Self {
        Self {
            use_mean_for_vectors,
        }
    }
    pub fn pool(&self, scalars: &[f64], vectors: &[[f64; 3]]) -> Result<(f64, [f64; 3])> {
        if scalars.len() != vectors.len() {
            return Err(TensorError::invalid_argument_op(
                "geometric_dl",
                "EquivariantReadout: length mismatch",
            ));
        }
        let n = scalars.len();
        let s_sum: f64 = scalars.iter().sum();
        let mut vs = [0.0f64; 3];
        for v in vectors {
            for d in 0..3 {
                vs[d] += v[d];
            }
        }
        if self.use_mean_for_vectors && n > 0 {
            let f = 1.0 / n as f64;
            for d in 0..3 {
                vs[d] *= f;
            }
        }
        Ok((s_sum, vs))
    }
}

/// SchNet-style SE(3)-invariant network.
#[derive(Debug, Clone)]
pub struct SchNetEquivariant {
    pub n_rbf: usize,
    pub rbf_centres: Vec<f64>,
    pub rbf_gamma: f64,
    pub cutoff: f64,
    pub interaction_w: Vec<Vec<f64>>,
    pub hidden_dim: usize,
}

impl SchNetEquivariant {
    pub fn new(n_rbf: usize, cutoff: f64, hidden_dim: usize, seed: u64) -> Self {
        let rbf_centres: Vec<f64> = (0..n_rbf)
            .map(|i| (i as f64 + 0.5) * cutoff / n_rbf as f64)
            .collect();
        let rbf_gamma = 1.0 / (cutoff / n_rbf as f64).powi(2);
        Self {
            n_rbf,
            rbf_centres,
            rbf_gamma,
            cutoff,
            interaction_w: rand_weight(hidden_dim, n_rbf, seed),
            hidden_dim,
        }
    }

    fn rbf(&self, r: f64) -> Vec<f64> {
        self.rbf_centres
            .iter()
            .map(|&mu| (-self.rbf_gamma * (r - mu).powi(2)).exp())
            .collect()
    }
    fn cutoff_fn(&self, r: f64) -> f64 {
        if r >= self.cutoff {
            0.0
        } else {
            0.5 * (1.0 + (std::f64::consts::PI * r / self.cutoff).cos())
        }
    }

    pub fn forward(&self, positions: &[[f64; 3]], embeddings: &[f64]) -> Result<Vec<f64>> {
        let n = positions.len();
        if embeddings.len() != n {
            return Err(TensorError::invalid_argument_op(
                "geometric_dl",
                "SchNetEquivariant: length mismatch",
            ));
        }
        let mut out = embeddings.to_vec();
        for i in 0..n {
            let mut msg = vec![0.0f64; self.hidden_dim];
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dv = [
                    positions[j][0] - positions[i][0],
                    positions[j][1] - positions[i][1],
                    positions[j][2] - positions[i][2],
                ];
                let r = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
                if r > self.cutoff {
                    continue;
                }
                let phi = self.rbf(r);
                let env = self.cutoff_fn(r);
                let filter = matvec(&self.interaction_w, &phi);
                for d in 0..self.hidden_dim {
                    msg[d] += filter[d] * env * embeddings[j];
                }
            }
            out[i] += msg.iter().sum::<f64>() / self.hidden_dim as f64;
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3 Mesh Processing
// ─────────────────────────────────────────────────────────────────────────────

/// Triangle mesh.
#[derive(Debug, Clone)]
pub struct TriangleMesh {
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<[usize; 3]>,
}

impl TriangleMesh {
    pub fn new(vertices: Vec<[f64; 3]>, faces: Vec<[usize; 3]>) -> Self {
        Self { vertices, faces }
    }

    pub fn normals(&self) -> Vec<[f64; 3]> {
        self.faces
            .iter()
            .map(|&[i, j, k]| {
                let (vi, vj, vk) = (self.vertices[i], self.vertices[j], self.vertices[k]);
                let (e1, e2) = (
                    [vj[0] - vi[0], vj[1] - vi[1], vj[2] - vi[2]],
                    [vk[0] - vi[0], vk[1] - vi[1], vk[2] - vi[2]],
                );
                [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ]
            })
            .collect()
    }

    pub fn areas(&self) -> Vec<f64> {
        self.normals()
            .iter()
            .map(|n| 0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt())
            .collect()
    }

    pub fn laplacian_matrix(&self) -> Vec<Vec<f64>> {
        let v = self.vertices.len();
        let mut w = vec![vec![0.0f64; v]; v];
        for &[i, j, k] in &self.faces {
            let (pi, pj, pk) = (
                Vector3::from(self.vertices[i]),
                Vector3::from(self.vertices[j]),
                Vector3::from(self.vertices[k]),
            );
            let cot_i = cot_angle(&pk.add(&pi.scale(-1.0)), &pj.add(&pi.scale(-1.0)));
            let cot_j = cot_angle(&pi.add(&pj.scale(-1.0)), &pk.add(&pj.scale(-1.0)));
            let cot_k = cot_angle(&pi.add(&pk.scale(-1.0)), &pj.add(&pk.scale(-1.0)));
            w[j][k] += 0.5 * cot_i;
            w[k][j] += 0.5 * cot_i;
            w[i][k] += 0.5 * cot_j;
            w[k][i] += 0.5 * cot_j;
            w[i][j] += 0.5 * cot_k;
            w[j][i] += 0.5 * cot_k;
        }
        let degree: Vec<f64> = (0..v).map(|i| w[i].iter().sum()).collect();
        let mut l = vec![vec![0.0f64; v]; v];
        for i in 0..v {
            let di = if degree[i].abs() < 1e-12 {
                1.0
            } else {
                degree[i].sqrt()
            };
            for j in 0..v {
                let dj = if degree[j].abs() < 1e-12 {
                    1.0
                } else {
                    degree[j].sqrt()
                };
                let sym = w[i][j] / (di * dj);
                l[i][j] = if i == j { 1.0 - sym } else { -sym };
            }
        }
        l
    }
}

fn cot_angle(u: &Vector3, v: &Vector3) -> f64 {
    let cos = u.dot(v);
    let sin = u.cross(v).norm();
    if sin.abs() < 1e-12 {
        0.0
    } else {
        cos / sin
    }
}

/// Spectral mesh convolution.
#[derive(Debug, Clone)]
pub struct MeshConvLayer {
    pub in_channels: usize,
    pub out_channels: usize,
    pub weight: Vec<Vec<f64>>,
}

impl MeshConvLayer {
    pub fn new(in_channels: usize, out_channels: usize, seed: u64) -> Self {
        Self {
            in_channels,
            out_channels,
            weight: rand_weight(out_channels, in_channels, seed),
        }
    }
    pub fn forward(&self, x: &[Vec<f64>], laplacian: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let v = x.len();
        if laplacian.len() != v {
            return Err(TensorError::invalid_argument_op(
                "geometric_dl",
                "MeshConvLayer: Laplacian size mismatch",
            ));
        }
        let mut lx = vec![vec![0.0f64; self.in_channels]; v];
        for i in 0..v {
            for j in 0..v {
                let l_ij = laplacian[i][j];
                if l_ij.abs() < 1e-14 {
                    continue;
                }
                for c in 0..self.in_channels {
                    lx[i][c] += l_ij * x[j].get(c).copied().unwrap_or(0.0);
                }
            }
        }
        lx.iter()
            .map(|row| Ok(self.weight.iter().map(|w| relu(dot(w, row))).collect()))
            .collect()
    }
}

/// Chebyshev polynomial mesh convolution.
#[derive(Debug, Clone)]
pub struct ChebMeshConv {
    pub k: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub theta: Vec<Vec<Vec<f64>>>,
}

impl ChebMeshConv {
    pub fn new(k: usize, in_channels: usize, out_channels: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (2.0 / (k * in_channels) as f64).sqrt();
        let theta = (0..k)
            .map(|_| {
                (0..out_channels)
                    .map(|_| {
                        (0..in_channels)
                            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                            .collect()
                    })
                    .collect()
            })
            .collect();
        Self {
            k,
            in_channels,
            out_channels,
            theta,
        }
    }

    pub fn forward(&self, x: &[Vec<f64>], laplacian: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let v = x.len();
        if v == 0 {
            return Ok(Vec::new());
        }
        if laplacian.len() != v {
            return Err(TensorError::invalid_argument_op(
                "geometric_dl",
                "ChebMeshConv: Laplacian size mismatch",
            ));
        }
        let mut l_tilde = laplacian.to_vec();
        for i in 0..v {
            l_tilde[i][i] -= 1.0;
        }
        let mut t_prev = x.to_vec();
        let mut t_curr = mat_mul(&l_tilde, x);
        let mut out = vec![vec![0.0f64; self.out_channels]; v];
        for i in 0..v {
            for o in 0..self.out_channels {
                out[i][o] += dot(&self.theta[0][o], &t_prev[i]);
            }
        }
        for ki in 1..self.k {
            for i in 0..v {
                for o in 0..self.out_channels {
                    out[i][o] += dot(&self.theta[ki][o], &t_curr[i]);
                }
            }
            if ki + 1 < self.k {
                let t_next = mat_mul(&l_tilde, &t_curr);
                let t_new: Vec<Vec<f64>> = t_next
                    .iter()
                    .zip(&t_prev)
                    .map(|(tn, tp)| tn.iter().zip(tp).map(|(a, b)| 2.0 * a - b).collect())
                    .collect();
                t_prev = t_curr;
                t_curr = t_new;
            }
        }
        for row in out.iter_mut() {
            for v in row.iter_mut() {
                *v = relu(*v);
            }
        }
        Ok(out)
    }
}

/// Vertex cluster pooling.
#[derive(Debug, Clone)]
pub struct SurfacePooling {
    pub n_clusters: usize,
}

impl SurfacePooling {
    pub fn new(n_clusters: usize) -> Self {
        Self { n_clusters }
    }

    pub fn pool(
        &self,
        x: &[Vec<f64>],
        positions: &[[f64; 3]],
    ) -> Result<(Vec<Vec<f64>>, Vec<usize>)> {
        let v = x.len();
        if v == 0 || self.n_clusters == 0 {
            return Err(TensorError::invalid_argument_op(
                "geometric_dl",
                "SurfacePooling: empty input",
            ));
        }
        let nc = self.n_clusters.min(v);
        let channels = x.first().map(|r| r.len()).unwrap_or(0);
        let step = (v as f64 / nc as f64).max(1.0);
        let centres: Vec<[f64; 3]> = (0..nc)
            .map(|k| positions[((k as f64 * step) as usize).min(v - 1)])
            .collect();
        let cluster_ids: Vec<usize> = positions
            .iter()
            .map(|p| {
                centres
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        dist3(p, a)
                            .partial_cmp(&dist3(p, b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(k, _)| k)
                    .unwrap_or(0)
            })
            .collect();
        let mut sums = vec![vec![0.0f64; channels]; nc];
        let mut counts = vec![0usize; nc];
        for (i, &k) in cluster_ids.iter().enumerate() {
            counts[k] += 1;
            for c in 0..channels {
                sums[k][c] += x[i].get(c).copied().unwrap_or(0.0);
            }
        }
        let pooled = sums
            .iter()
            .zip(&counts)
            .map(|(s, &cnt)| {
                let f = if cnt == 0 { 1.0 } else { 1.0 / cnt as f64 };
                s.iter().map(|v| v * f).collect()
            })
            .collect();
        Ok((pooled, cluster_ids))
    }
}

/// Mesh autoencoder.
#[derive(Debug, Clone)]
pub struct MeshAutoEncoder {
    pub encoder_conv: ChebMeshConv,
    pub pooling: SurfacePooling,
    pub decoder_conv: ChebMeshConv,
    pub latent_dim: usize,
}

impl MeshAutoEncoder {
    pub fn new(
        in_channels: usize,
        latent_dim: usize,
        n_clusters: usize,
        k: usize,
        seed: u64,
    ) -> Self {
        Self {
            encoder_conv: ChebMeshConv::new(k, in_channels, latent_dim, seed),
            pooling: SurfacePooling::new(n_clusters),
            decoder_conv: ChebMeshConv::new(k, latent_dim, in_channels, seed.wrapping_add(1)),
            latent_dim,
        }
    }
    pub fn encode(
        &self,
        x: &[Vec<f64>],
        laplacian: &[Vec<f64>],
        positions: &[[f64; 3]],
    ) -> Result<Vec<Vec<f64>>> {
        let enc = self.encoder_conv.forward(x, laplacian)?;
        let (pooled, _) = self.pooling.pool(&enc, positions)?;
        Ok(pooled)
    }
    pub fn decode(
        &self,
        latent: &[Vec<f64>],
        laplacian: &[Vec<f64>],
        cluster_ids: &[usize],
    ) -> Result<Vec<Vec<f64>>> {
        let v = cluster_ids.len();
        let unpooled: Vec<Vec<f64>> = cluster_ids
            .iter()
            .map(|&k| {
                latent
                    .get(k)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; self.latent_dim])
            })
            .collect();
        if unpooled.len() != v || laplacian.len() != v {
            return Err(TensorError::invalid_argument_op(
                "geometric_dl",
                "MeshAutoEncoder::decode: dimension mismatch",
            ));
        }
        self.decoder_conv.forward(&unpooled, laplacian)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4 Manifold Learning
// ─────────────────────────────────────────────────────────────────────────────

/// Manifold type for [`RiemannianOptimizer`].
#[derive(Debug, Clone, PartialEq)]
pub enum Manifold {
    Sphere,
    Spd { dim: usize },
}

/// Projected gradient descent on a Riemannian manifold.
#[derive(Debug, Clone)]
pub struct RiemannianOptimizer {
    pub lr: f64,
    pub manifold: Manifold,
}

impl RiemannianOptimizer {
    pub fn new(lr: f64, manifold: Manifold) -> Self {
        Self { lr, manifold }
    }
    pub fn step(&self, point: &[f64], grad: &[f64]) -> Result<Vec<f64>> {
        if point.len() != grad.len() {
            return Err(TensorError::invalid_argument_op(
                "geometric_dl",
                "RiemannianOptimizer: length mismatch",
            ));
        }
        match &self.manifold {
            Manifold::Sphere => {
                let inner = dot(point, grad);
                let rg: Vec<f64> = grad.iter().zip(point).map(|(g, p)| g - inner * p).collect();
                let mut np: Vec<f64> = point
                    .iter()
                    .zip(&rg)
                    .map(|(p, g)| p - self.lr * g)
                    .collect();
                normalize_vec(&mut np);
                Ok(np)
            }
            Manifold::Spd { dim } => {
                let n = *dim;
                if point.len() != n * n {
                    return Err(TensorError::invalid_argument_op(
                        "geometric_dl",
                        "RiemannianOptimizer(SPD): point must be dim²",
                    ));
                }
                let mut np: Vec<f64> = point
                    .iter()
                    .zip(grad)
                    .map(|(p, g)| p - self.lr * g)
                    .collect();
                for i in 0..n {
                    for j in (i + 1)..n {
                        let avg = (np[i * n + j] + np[j * n + i]) * 0.5;
                        np[i * n + j] = avg;
                        np[j * n + i] = avg;
                    }
                }
                for i in 0..n {
                    let d = &mut np[i * n + i];
                    if *d < 1e-6 {
                        *d = 1e-6;
                    }
                }
                Ok(np)
            }
        }
    }
}

/// Poincaré disk model.
#[derive(Debug, Clone)]
pub struct HyperbolicEmbedding {
    pub c: f64,
}

impl HyperbolicEmbedding {
    pub fn new(c: f64) -> Result<Self> {
        if c <= 0.0 {
            return Err(TensorError::invalid_argument_op(
                "geometric_dl",
                "HyperbolicEmbedding: c must be positive",
            ));
        }
        Ok(Self { c })
    }

    pub fn dist(&self, u: &[f64], v: &[f64]) -> Result<f64> {
        if u.len() != v.len() {
            return Err(TensorError::invalid_argument_op(
                "geometric_dl",
                "HyperbolicEmbedding::dist: dimension mismatch",
            ));
        }
        let u_sq: f64 = u.iter().map(|x| x * x).sum();
        let v_sq: f64 = v.iter().map(|x| x * x).sum();
        let uv_sq: f64 = u.iter().zip(v).map(|(a, b)| (a - b).powi(2)).sum();
        let num = 2.0 * self.c * uv_sq;
        let denom = ((1.0 - self.c * u_sq) * (1.0 - self.c * v_sq)).max(1e-14);
        Ok((1.0 / self.c.sqrt()) * (1.0 + num / denom).max(1.0).acosh())
    }

    pub fn exp_map(&self, x: &[f64], v: &[f64]) -> Result<Vec<f64>> {
        if x.len() != v.len() {
            return Err(TensorError::invalid_argument_op(
                "geometric_dl",
                "exp_map: dimension mismatch",
            ));
        }
        let x_sq: f64 = x.iter().map(|xi| xi * xi).sum();
        let lambda = 2.0 / (1.0 - self.c * x_sq).max(1e-14);
        let v_norm = v.iter().map(|vi| vi * vi).sum::<f64>().sqrt().max(1e-14);
        let arg = (lambda * self.c.sqrt() * v_norm / 2.0).tanh();
        let coeff = arg / (self.c.sqrt() * v_norm);
        let y: Vec<f64> = v.iter().map(|vi| coeff * vi).collect();
        self.mobius_add(x, &y)
    }

    pub fn log_map(&self, x: &[f64], y: &[f64]) -> Result<Vec<f64>> {
        if x.len() != y.len() {
            return Err(TensorError::invalid_argument_op(
                "geometric_dl",
                "log_map: dimension mismatch",
            ));
        }
        let x_sq: f64 = x.iter().map(|xi| xi * xi).sum();
        let lambda = 2.0 / (1.0 - self.c * x_sq).max(1e-14);
        let neg_x: Vec<f64> = x.iter().map(|xi| -xi).collect();
        let z = self.mobius_add(&neg_x, y)?;
        let z_norm = z.iter().map(|zi| zi * zi).sum::<f64>().sqrt().max(1e-14);
        let arg = (self.c.sqrt() * z_norm).atanh();
        let coeff = 2.0 * arg / (lambda * self.c.sqrt() * z_norm);
        Ok(z.iter().map(|zi| coeff * zi).collect())
    }

    pub fn mobius_add(&self, x: &[f64], y: &[f64]) -> Result<Vec<f64>> {
        let x_sq: f64 = x.iter().map(|xi| xi * xi).sum();
        let y_sq: f64 = y.iter().map(|yi| yi * yi).sum();
        let xy: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();
        let nc_x = 1.0 + 2.0 * self.c * xy + self.c * y_sq;
        let nc_y = 1.0 - self.c * x_sq;
        let denom = (1.0 + 2.0 * self.c * xy + self.c * self.c * x_sq * y_sq).max(1e-14);
        Ok(x.iter()
            .zip(y)
            .map(|(xi, yi)| (nc_x * xi + nc_y * yi) / denom)
            .collect())
    }
}

/// Lorentz (hyperboloid) model.
#[derive(Debug, Clone)]
pub struct LorentzModel {
    pub k: f64,
}

impl LorentzModel {
    pub fn new(k: f64) -> Result<Self> {
        if k <= 0.0 {
            return Err(TensorError::invalid_argument_op(
                "geometric_dl",
                "LorentzModel: k must be positive",
            ));
        }
        Ok(Self { k })
    }
    pub fn minkowski_dot(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() || x.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "geometric_dl",
                "LorentzModel: dimension mismatch",
            ));
        }
        Ok(-x[0] * y[0] + x[1..].iter().zip(&y[1..]).map(|(a, b)| a * b).sum::<f64>())
    }
    pub fn dist(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        let inner = self.minkowski_dot(x, y)?;
        Ok(self.k.sqrt() * (-inner / self.k).max(1.0).acosh())
    }
    pub fn project_onto_hyperboloid(&self, x: &[f64]) -> Vec<f64> {
        if x.is_empty() {
            return Vec::new();
        }
        let spatial_sq: f64 = x[1..].iter().map(|xi| xi * xi).sum();
        let mut r = vec![(self.k + spatial_sq).sqrt()];
        r.extend_from_slice(&x[1..]);
        r
    }
}

/// Möbius linear layer in Poincaré ball.
#[derive(Debug, Clone)]
pub struct HyperbolicLinear {
    pub embedding: HyperbolicEmbedding,
    pub weight: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
}

impl HyperbolicLinear {
    pub fn new(in_dim: usize, out_dim: usize, c: f64, seed: u64) -> Result<Self> {
        let embedding = HyperbolicEmbedding::new(c)?;
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (1.0 / in_dim as f64).sqrt() * 0.01;
        let weight = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                    .collect()
            })
            .collect();
        Ok(Self {
            embedding,
            weight,
            bias: vec![0.0; out_dim],
        })
    }
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>> {
        let zero = vec![0.0; x.len()];
        let tangent = self.embedding.log_map(&zero, x)?;
        let linear: Vec<f64> = self
            .weight
            .iter()
            .map(|w_row| dot(w_row, &tangent))
            .collect();
        let zero_out = vec![0.0; self.weight.len()];
        let mapped = self.embedding.exp_map(&zero_out, &linear)?;
        self.embedding.mobius_add(&mapped, &self.bias)
    }
}

/// Pairwise geodesic distances.
#[derive(Debug, Clone)]
pub struct GeodesicDistance {
    pub embedding: HyperbolicEmbedding,
}

impl GeodesicDistance {
    pub fn new(c: f64) -> Result<Self> {
        Ok(Self {
            embedding: HyperbolicEmbedding::new(c)?,
        })
    }
    pub fn pairwise(&self, points: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = points.len();
        let mut dists = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = self.embedding.dist(&points[i], &points[j])?;
                dists[i][j] = d;
                dists[j][i] = d;
            }
        }
        Ok(dists)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5 Topological Deep Learning
// ─────────────────────────────────────────────────────────────────────────────

/// Simplicial complex: nodes, edges, triangles.
#[derive(Debug, Clone)]
pub struct SimplicialComplex {
    pub n_nodes: usize,
    pub edges: Vec<[usize; 2]>,
    pub triangles: Vec<[usize; 3]>,
}

impl SimplicialComplex {
    pub fn new(n_nodes: usize, edges: Vec<[usize; 2]>, triangles: Vec<[usize; 3]>) -> Self {
        Self {
            n_nodes,
            edges,
            triangles,
        }
    }

    pub fn boundary_matrix(&self, dim: usize) -> Vec<Vec<i32>> {
        match dim {
            1 => {
                let ne = self.edges.len();
                let mut b = vec![vec![0i32; ne]; self.n_nodes];
                for (e_idx, &[i, j]) in self.edges.iter().enumerate() {
                    if j < self.n_nodes {
                        b[j][e_idx] = 1;
                    }
                    if i < self.n_nodes {
                        b[i][e_idx] = -1;
                    }
                }
                b
            }
            2 => {
                let (ne, nt) = (self.edges.len(), self.triangles.len());
                let mut b = vec![vec![0i32; nt]; ne];
                let edge_idx: std::collections::HashMap<[usize; 2], usize> = self
                    .edges
                    .iter()
                    .enumerate()
                    .map(|(i, &e)| (e, i))
                    .collect();
                for (t_idx, &[a, bv, c]) in self.triangles.iter().enumerate() {
                    for (edge, sign) in &[([a, bv], 1i32), ([a, c], -1), ([bv, c], 1)] {
                        if let Some(&e_idx) = edge_idx.get(edge) {
                            b[e_idx][t_idx] = *sign;
                        }
                    }
                }
                b
            }
            _ => Vec::new(),
        }
    }

    pub fn hodge_laplacian(&self, k: usize) -> Vec<Vec<f64>> {
        match k {
            0 => {
                let b1 = self.boundary_matrix(1);
                let (n, ne) = (self.n_nodes, self.edges.len());
                let mut l = vec![vec![0.0f64; n]; n];
                for i in 0..n {
                    for j in 0..n {
                        for e in 0..ne {
                            l[i][j] += (b1[i][e] as f64) * (b1[j][e] as f64);
                        }
                    }
                }
                l
            }
            1 => {
                let (b1, b2) = (self.boundary_matrix(1), self.boundary_matrix(2));
                let (ne, nn, nt) = (self.edges.len(), self.n_nodes, self.triangles.len());
                let mut l = vec![vec![0.0f64; ne]; ne];
                for i in 0..ne {
                    for j in 0..ne {
                        for v in 0..nn {
                            l[i][j] += (b1[v][i] as f64) * (b1[v][j] as f64);
                        }
                        for t in 0..nt {
                            l[i][j] += (b2[i][t] as f64) * (b2[j][t] as f64);
                        }
                    }
                }
                l
            }
            _ => Vec::new(),
        }
    }
}

/// Message passing on simplicial complex.
#[derive(Debug, Clone)]
pub struct SimplicialConv {
    pub dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub weight: Vec<Vec<f64>>,
}

impl SimplicialConv {
    pub fn new(dim: usize, in_channels: usize, out_channels: usize, seed: u64) -> Self {
        Self {
            dim,
            in_channels,
            out_channels,
            weight: rand_weight(out_channels, in_channels, seed),
        }
    }
    pub fn forward(&self, x: &[Vec<f64>], complex: &SimplicialComplex) -> Result<Vec<Vec<f64>>> {
        let laplacian = complex.hodge_laplacian(self.dim);
        let lx = mat_mul(&laplacian, x);
        lx.iter()
            .map(|row| Ok(self.weight.iter().map(|w| relu(dot(w, row))).collect()))
            .collect()
    }
}

/// Birth-death pairs for a topological dimension.
#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    pub dim: usize,
    pub pairs: Vec<(f64, f64)>,
}

impl PersistenceDiagram {
    pub fn new(dim: usize, pairs: Vec<(f64, f64)>) -> Self {
        Self { dim, pairs }
    }
    pub fn total_persistence(&self) -> f64 {
        self.pairs.iter().map(|(b, d)| (d - b).max(0.0)).sum()
    }
    pub fn betti_number(&self, threshold: f64) -> usize {
        self.pairs
            .iter()
            .filter(|(b, d)| (d - b) > threshold)
            .count()
    }
}

/// Vietoris-Rips filtration (0-dimensional).
#[derive(Debug, Clone)]
pub struct RipsFiltration {
    pub max_scale: f64,
}

impl RipsFiltration {
    pub fn new(max_scale: f64) -> Self {
        Self { max_scale }
    }

    pub fn compute(&self, dist_matrix: &[Vec<f64>]) -> Result<PersistenceDiagram> {
        let n = dist_matrix.len();
        if n == 0 {
            return Ok(PersistenceDiagram::new(0, Vec::new()));
        }
        let mut edges: Vec<(f64, usize, usize)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let d = dist_matrix[i].get(j).copied().unwrap_or(f64::INFINITY);
                if d <= self.max_scale {
                    edges.push((d, i, j));
                }
            }
        }
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank = vec![0usize; n];
        let birth = vec![0.0f64; n];
        let mut pairs: Vec<(f64, f64)> = Vec::new();

        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }
        for (dist, i, j) in &edges {
            let (ri, rj) = (find(&mut parent, *i), find(&mut parent, *j));
            if ri == rj {
                continue;
            }
            let (survivor, dead) = if rank[ri] >= rank[rj] {
                (ri, rj)
            } else {
                (rj, ri)
            };
            pairs.push((birth[dead], *dist));
            parent[dead] = survivor;
            if rank[ri] == rank[rj] {
                rank[survivor] += 1;
            }
        }
        Ok(PersistenceDiagram::new(0, pairs))
    }
}

/// Differentiable topology loss.
#[derive(Debug, Clone)]
pub struct TopoLoss {
    pub target_betti_0: usize,
    pub threshold: f64,
    pub weight: f64,
}

impl TopoLoss {
    pub fn new(target_betti_0: usize, threshold: f64, weight: f64) -> Self {
        Self {
            target_betti_0,
            threshold,
            weight,
        }
    }

    pub fn compute(&self, diagram: &PersistenceDiagram) -> f64 {
        let betti = diagram.betti_number(self.threshold);
        let betti_diff = (betti as f64 - self.target_betti_0 as f64).powi(2);
        let noise_loss: f64 = diagram
            .pairs
            .iter()
            .filter(|(b, d)| (d - b) <= self.threshold)
            .map(|(b, d)| (d - b).powi(2))
            .sum();
        self.weight * (betti_diff + noise_loss)
    }

    pub fn compute_from_distances(&self, dist_matrix: &[Vec<f64>], max_scale: f64) -> Result<f64> {
        Ok(self.compute(&RipsFiltration::new(max_scale).compute(dist_matrix)?))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_cloud_construction() {
        let pc = PointCloud::new(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        assert_eq!(pc.num_points(), 3);
        assert!(pc.features.is_none());
    }

    #[test]
    fn test_point_cloud_with_features() {
        let pc = PointCloud::new(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            .with_features(vec![vec![1.0, 2.0], vec![3.0, 4.0]])
            .expect("computation failed");
        assert_eq!(pc.feature_dim(), 2);
    }

    #[test]
    fn test_knn_graph_k_neighbors() {
        let pc = PointCloud::new(vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]);
        let knn = KnnGraph::build(&pc, 2);
        assert_eq!(knn.len(), 4);
        for nb in &knn {
            assert_eq!(nb.len(), 2);
        }
    }

    #[test]
    fn test_knn_graph_self_excluded() {
        let pc = PointCloud::new(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let knn = KnnGraph::build(&pc, 1);
        for (i, nb) in knn.iter().enumerate() {
            for &j in nb {
                assert_ne!(i, j);
            }
        }
    }

    #[test]
    fn test_pointnet_layer_shape() {
        let pc = PointCloud::new(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let out = PointNetLayer::new(3, 8, 42).forward(&pc).expect("operation should succeed");
        assert_eq!(out.len(), 3);
        for row in &out {
            assert_eq!(row.len(), 8);
        }
    }

    #[test]
    fn test_pointnet_plusplus_hierarchical() {
        let pts: Vec<[f64; 3]> = (0..10).map(|i| [i as f64, 0.0, 0.0]).collect();
        let (sampled, feats) = PointNetPlusPlus::new(4, 1.5, 3, 3, 16, 0)
            .forward(&PointCloud::new(pts))
            .expect("computation failed");
        assert_eq!(sampled.num_points(), 4);
        assert_eq!(feats.len(), 4);
        for f in &feats {
            assert_eq!(f.len(), 16);
        }
    }

    #[test]
    fn test_dgcnn_edge_conv_shape() {
        let pc = PointCloud::new((0..5).map(|i| [i as f64, 0.0, 0.0]).collect());
        let knn = KnnGraph::build(&pc, 2);
        let feats: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64; 4]).collect();
        let out = DgcnnLayer::new(4, 8, 2, 99)
            .edge_conv(&feats, &knn)
            .expect("computation failed");
        assert_eq!(out.len(), 5);
        for row in &out {
            assert_eq!(row.len(), 8);
        }
    }

    #[test]
    fn test_vector3_cross_product() {
        let w = Vector3::new(1.0, 0.0, 0.0).cross(&Vector3::new(0.0, 1.0, 0.0));
        assert!((w.z - 1.0).abs() < 1e-12 && w.x.abs() < 1e-12 && w.y.abs() < 1e-12);
    }

    #[test]
    fn test_vector3_operations() {
        let u = Vector3::new(3.0, 4.0, 0.0);
        assert!((u.norm() - 5.0).abs() < 1e-12);
        assert!((u.normalized().norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_so3_features() {
        let s = SO3Features::new(vec![1.0, 2.0], vec![Vector3::new(1.0, 0.0, 0.0)]);
        let t = SO3Features::new(vec![3.0, 4.0], vec![Vector3::new(0.0, 1.0, 0.0)]);
        let sum = s.add(&t).expect("operation should succeed");
        assert!((sum.scalars[0] - 4.0).abs() < 1e-12);
        assert!((sum.vectors[0].y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_tfn_basis_output() {
        let layer = TFNLayer::new(8, 5.0, 7);
        let (scalar, vb) = layer.basis(1.0, &[1.0, 0.0, 0.0]);
        assert!(scalar.is_finite() && vb[0].is_finite() && vb[1].is_finite() && vb[2].is_finite());
    }

    #[test]
    fn test_tfn_layer_forward() {
        let layer = TFNLayer::new(4, 3.0, 11);
        let (out_s, out_v) = layer
            .forward(
                &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                &[1.0, 2.0, 3.0],
                &[[0.1, 0.0, 0.0]; 3],
            )
            .expect("computation failed");
        assert_eq!(out_s.len(), 3);
        assert_eq!(out_v.len(), 3);
    }

    #[test]
    fn test_equivariant_readout() {
        let (s, v) = EquivariantReadout::new(true)
            .pool(
                &[1.0, 2.0, 3.0],
                &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            )
            .expect("computation failed");
        assert!((s - 6.0).abs() < 1e-12);
        assert!((v[0] - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_schnet_equivariant() {
        let out = SchNetEquivariant::new(8, 5.0, 16, 42)
            .forward(
                &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                &[1.0, 1.0, 1.0],
            )
            .expect("computation failed");
        assert_eq!(out.len(), 3);
        for v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_triangle_mesh_normals() {
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0, 1, 2]],
        );
        let normals = mesh.normals();
        assert_eq!(normals.len(), 1);
        assert!(normals[0][2] > 0.0);
    }

    #[test]
    fn test_mesh_areas_positive() {
        let mesh = TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
            ],
            vec![[0, 1, 2], [1, 3, 2]],
        );
        for a in mesh.areas() {
            assert!(a > 0.0);
        }
    }

    #[test]
    fn test_mesh_laplacian_symmetric() {
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            vec![[0, 1, 2]],
        );
        let l = mesh.laplacian_matrix();
        for i in 0..3 {
            for j in 0..3 {
                assert!((l[i][j] - l[j][i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_cheb_mesh_conv_shape() {
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            vec![[0, 1, 2]],
        );
        let out = ChebMeshConv::new(3, 2, 4, 1)
            .forward(
                &[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]],
                &mesh.laplacian_matrix(),
            )
            .expect("computation failed");
        assert_eq!(out.len(), 3);
        for row in &out {
            assert_eq!(row.len(), 4);
        }
    }

    #[test]
    fn test_surface_pooling() {
        let positions: Vec<[f64; 3]> = (0..6).map(|i| [i as f64, 0.0, 0.0]).collect();
        let x: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64, (i * 2) as f64]).collect();
        let (pooled, ids) = SurfacePooling::new(2).pool(&x, &positions).expect("operation should succeed");
        assert_eq!(pooled.len(), 2);
        assert_eq!(ids.len(), 6);
    }

    #[test]
    fn test_poincare_distance_positive() {
        let d = HyperbolicEmbedding::new(1.0)
            .expect("computation failed")
            .dist(&[0.1, 0.2], &[0.3, 0.1])
            .expect("computation failed");
        assert!(d > 0.0);
    }

    #[test]
    fn test_poincare_self_distance_zero() {
        let d = HyperbolicEmbedding::new(1.0)
            .expect("computation failed")
            .dist(&[0.1, 0.2, 0.3], &[0.1, 0.2, 0.3])
            .expect("computation failed");
        assert!(d < 1e-10);
    }

    #[test]
    fn test_exp_log_inverse() {
        let emb = HyperbolicEmbedding::new(1.0).expect("operation should succeed");
        let v = vec![0.1, 0.05];
        let y = emb.exp_map(&[0.0, 0.0], &v).expect("operation should succeed");
        let v_rec = emb.log_map(&[0.0, 0.0], &y).expect("operation should succeed");
        assert!((v_rec[0] - v[0]).abs() < 1e-8 && (v_rec[1] - v[1]).abs() < 1e-8);
    }

    #[test]
    fn test_lorentz_inner_product() {
        let model = LorentzModel::new(1.0).expect("operation should succeed");
        let x = model.project_onto_hyperboloid(&[0.0, 1.0, 0.0]);
        assert!((model.minkowski_dot(&x, &x).expect("operation should succeed") + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbolic_linear() {
        let out = HyperbolicLinear::new(3, 4, 0.5, 77)
            .expect("computation failed")
            .forward(&[0.1, 0.1, 0.1])
            .expect("computation failed");
        assert_eq!(out.len(), 4);
        for v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_simplicial_complex_boundary() {
        let sc = SimplicialComplex::new(3, vec![[0, 1], [0, 2], [1, 2]], vec![[0, 1, 2]]);
        let b1 = sc.boundary_matrix(1);
        assert_eq!(b1.len(), 3);
        for e in 0..3 {
            assert_eq!((0..3).filter(|&v| b1[v][e] == 1).count(), 1);
            assert_eq!((0..3).filter(|&v| b1[v][e] == -1).count(), 1);
        }
    }

    #[test]
    fn test_boundary_matrix_dim0() {
        assert!(SimplicialComplex::new(3, vec![[0, 1]], vec![])
            .boundary_matrix(0)
            .is_empty());
    }

    #[test]
    fn test_boundary_matrix_dim2() {
        let sc = SimplicialComplex::new(3, vec![[0, 1], [0, 2], [1, 2]], vec![[0, 1, 2]]);
        let b2 = sc.boundary_matrix(2);
        assert_eq!(b2.len(), 3);
        assert_eq!(b2[0].len(), 1);
    }

    #[test]
    fn test_rips_filtration_components() {
        let dists = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0],
        ];
        let diagram = RipsFiltration::new(3.0).compute(&dists).expect("operation should succeed");
        assert_eq!(diagram.dim, 0);
        assert_eq!(diagram.pairs.len(), 2);
    }

    #[test]
    fn test_persistence_diagram() {
        let pd = PersistenceDiagram::new(0, vec![(0.0, 1.0), (0.0, 0.5), (0.0, 10.0)]);
        assert!((pd.total_persistence() - 11.5).abs() < 1e-12);
        assert_eq!(pd.betti_number(0.8), 2);
    }

    #[test]
    fn test_topo_loss() {
        let loss = TopoLoss::new(1, 0.5, 1.0)
            .compute_from_distances(
                &[
                    vec![0.0, 0.1, 2.0],
                    vec![0.1, 0.0, 2.0],
                    vec![2.0, 2.0, 0.0],
                ],
                3.0,
            )
            .expect("computation failed");
        assert!(loss >= 0.0 && loss.is_finite());
    }

    #[test]
    fn test_topo_loss_target_betti() {
        let pd = PersistenceDiagram::new(0, vec![(0.0, 1.0)]);
        assert!(TopoLoss::new(1, 0.5, 1.0).compute(&pd) < 1e-12);
    }

    #[test]
    fn test_simplicial_conv_shape() {
        let sc = SimplicialComplex::new(4, vec![[0, 1], [1, 2], [2, 3]], vec![]);
        let out = SimplicialConv::new(0, 2, 4, 5)
            .forward(&vec![vec![1.0, 0.0]; 4], &sc)
            .expect("computation failed");
        assert_eq!(out.len(), 4);
        for row in &out {
            assert_eq!(row.len(), 4);
        }
    }

    #[test]
    fn test_mesh_autoencoder_encode_decode() {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ];
        let mesh = TriangleMesh::new(
            vertices.clone(),
            vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
        );
        let laplacian = mesh.laplacian_matrix();
        let x = vec![vec![1.0, 0.5]; 4];
        let ae = MeshAutoEncoder::new(2, 8, 2, 2, 13);
        let latent = ae.encode(&x, &laplacian, &vertices).expect("operation should succeed");
        assert_eq!(latent.len(), 2);
        let (_, ids) = ae.pooling.pool(&x, &vertices).expect("operation should succeed");
        let recon = ae.decode(&latent, &laplacian, &ids).expect("operation should succeed");
        assert_eq!(recon.len(), 4);
    }

    #[test]
    fn test_fps_unique_count() {
        let pts: Vec<[f64; 3]> = (0..20).map(|i| [i as f64, 0.0, 0.0]).collect();
        let selected = fps(&PointCloud::new(pts), 5);
        assert_eq!(selected.len(), 5);
        let mut s = selected.clone();
        s.sort_unstable();
        s.dedup();
        assert_eq!(s.len(), 5);
    }

    #[test]
    fn test_lorentz_invalid_curvature() {
        assert!(LorentzModel::new(-1.0).is_err() && LorentzModel::new(0.0).is_err());
    }

    #[test]
    fn test_hyperbolic_invalid_curvature() {
        assert!(HyperbolicEmbedding::new(0.0).is_err() && HyperbolicEmbedding::new(-2.0).is_err());
    }

    #[test]
    fn test_knn_clamped_to_available() {
        let pc = PointCloud::new(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let knn = KnnGraph::build(&pc, 10);
        for nb in &knn {
            assert_eq!(nb.len(), 1);
        }
    }

    #[test]
    fn test_riemannian_sphere_step() {
        let new_pt = RiemannianOptimizer::new(0.01, Manifold::Sphere)
            .step(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0])
            .expect("computation failed");
        let norm: f64 = new_pt.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_geodesic_distance_pairwise() {
        let gd = GeodesicDistance::new(1.0).expect("operation should succeed");
        let dists = gd
            .pairwise(&[vec![0.1, 0.0], vec![0.2, 0.0], vec![0.0, 0.1]])
            .expect("computation failed");
        assert_eq!(dists.len(), 3);
        for i in 0..3 {
            assert!(dists[i][i] < 1e-10);
        }
        for i in 0..3 {
            for j in 0..3 {
                assert!((dists[i][j] - dists[j][i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_lorentz_distance_positive() {
        let model = LorentzModel::new(1.0).expect("operation should succeed");
        let x = model.project_onto_hyperboloid(&[0.0, 1.0, 0.0]);
        let y = model.project_onto_hyperboloid(&[0.0, 0.0, 1.0]);
        assert!(model.dist(&x, &y).expect("operation should succeed") > 0.0);
    }
}
