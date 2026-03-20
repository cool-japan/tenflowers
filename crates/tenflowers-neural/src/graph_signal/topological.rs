//! §5 Topological Signal Processing.
//!
//! Provides Hodge Laplacians, edge flows, simplicial signal denoising,
//! and TDA feature extraction.

use super::spectral::GraphLaplacian;
use super::utils::{dot, mat_add, mat_mul, matvec};

// ─────────────────────────────────────────────────────────────────────────────
// Simplicial complex data
// ─────────────────────────────────────────────────────────────────────────────

/// A simplicial complex with 0-, 1-, 2-simplices.
#[derive(Debug, Clone)]
pub struct SimplicialComplexData {
    pub n_vertices: usize,
    pub edges: Vec<(usize, usize)>,
    pub triangles: Vec<(usize, usize, usize)>,
}

impl SimplicialComplexData {
    pub fn new(
        n_vertices: usize,
        edges: Vec<(usize, usize)>,
        triangles: Vec<(usize, usize, usize)>,
    ) -> Self {
        Self {
            n_vertices,
            edges,
            triangles,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hodge Laplacians
// ─────────────────────────────────────────────────────────────────────────────

/// Hodge Laplacians L_0 (on vertices), L_1 (on edges), L_2 (on triangles).
#[derive(Debug, Clone)]
pub struct HodgeLaplacian;

impl HodgeLaplacian {
    /// Compute L_0 (standard graph Laplacian on vertices).
    pub fn l0(sc: &SimplicialComplexData) -> Vec<Vec<f64>> {
        let n = sc.n_vertices;
        let mut adj = vec![vec![0.0; n]; n];
        for &(u, v) in &sc.edges {
            if u < n && v < n {
                adj[u][v] = 1.0;
                adj[v][u] = 1.0;
            }
        }
        GraphLaplacian::compute(&adj)
    }

    /// Compute boundary matrix B1: [n_vertices x n_edges].
    pub(super) fn boundary1(sc: &SimplicialComplexData) -> Vec<Vec<f64>> {
        let n_v = sc.n_vertices;
        let n_e = sc.edges.len();
        let mut b1 = vec![vec![0.0; n_e]; n_v];
        for (k, &(u, v)) in sc.edges.iter().enumerate() {
            if u < n_v && v < n_v {
                b1[u][k] = -1.0;
                b1[v][k] = 1.0;
            }
        }
        b1
    }

    /// Compute boundary matrix B2: [n_edges x n_triangles].
    pub(super) fn boundary2(sc: &SimplicialComplexData) -> Vec<Vec<f64>> {
        let n_e = sc.edges.len();
        let n_t = sc.triangles.len();
        let mut b2 = vec![vec![0.0; n_t]; n_e];
        for (t, &(u, v, w)) in sc.triangles.iter().enumerate() {
            let tri_edges = [(u, v), (v, w), (u, w)];
            let signs = [1.0_f64, 1.0, -1.0];
            for (te, s) in tri_edges.iter().zip(signs.iter()) {
                for (k, &(eu, ev)) in sc.edges.iter().enumerate() {
                    if (eu == te.0 && ev == te.1) || (eu == te.1 && ev == te.0) {
                        b2[k][t] += s;
                        break;
                    }
                }
            }
        }
        b2
    }

    /// Compute L_1 = B1^T B1 + B2 B2^T (Hodge-1 Laplacian).
    pub fn l1(sc: &SimplicialComplexData) -> Vec<Vec<f64>> {
        let n_e = sc.edges.len();
        if n_e == 0 {
            return Vec::new();
        }
        let b1 = Self::boundary1(sc);
        let b2 = Self::boundary2(sc);

        let b1t: Vec<Vec<f64>> = (0..n_e)
            .map(|e| (0..sc.n_vertices).map(|v| b1[v][e]).collect())
            .collect();
        let b1t_b1 = mat_mul(&b1t, &b1);

        let n_t = sc.triangles.len();
        let b2t: Vec<Vec<f64>> = (0..n_t)
            .map(|t| (0..n_e).map(|e| b2[e][t]).collect())
            .collect();
        let b2_b2t = mat_mul(&b2, &b2t);

        mat_add(&b1t_b1, &b2_b2t)
    }

    /// Compute L_2 = B2^T B2 (Hodge-2 Laplacian on triangles).
    pub fn l2(sc: &SimplicialComplexData) -> Vec<Vec<f64>> {
        let n_t = sc.triangles.len();
        if n_t == 0 {
            return Vec::new();
        }
        let b2 = Self::boundary2(sc);
        let b2t: Vec<Vec<f64>> = (0..n_t)
            .map(|t| (0..sc.edges.len()).map(|e| b2[e][t]).collect())
            .collect();
        mat_mul(&b2t, &b2)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge flows
// ─────────────────────────────────────────────────────────────────────────────

/// Signals on edges: gradient, curl, divergence.
#[derive(Debug, Clone)]
pub struct EdgeFlow {
    pub sc: SimplicialComplexData,
}

impl EdgeFlow {
    pub fn new(sc: SimplicialComplexData) -> Self {
        Self { sc }
    }

    /// Gradient: maps vertex signal to edge signal (B1^T v).
    pub fn gradient(&self, vertex_signal: &[f64]) -> Vec<f64> {
        let b1 = HodgeLaplacian::boundary1(&self.sc);
        let n_e = self.sc.edges.len();
        (0..n_e)
            .map(|e| {
                (0..self.sc.n_vertices)
                    .map(|v| b1[v][e] * vertex_signal.get(v).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect()
    }

    /// Divergence: maps edge signal to vertex signal (B1 f).
    pub fn divergence(&self, edge_signal: &[f64]) -> Vec<f64> {
        let b1 = HodgeLaplacian::boundary1(&self.sc);
        (0..self.sc.n_vertices)
            .map(|v| dot(&b1[v], edge_signal))
            .collect()
    }

    /// Curl: maps edge signal to triangle signal (B2^T f).
    pub fn curl(&self, edge_signal: &[f64]) -> Vec<f64> {
        let b2 = HodgeLaplacian::boundary2(&self.sc);
        let n_t = self.sc.triangles.len();
        let n_e = self.sc.edges.len();
        (0..n_t)
            .map(|t| {
                (0..n_e)
                    .map(|e| b2[e][t] * edge_signal.get(e).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Simplicial signal denoising
// ─────────────────────────────────────────────────────────────────────────────

/// Denoise signal on simplicial complex via Hodge regularization.
#[derive(Debug, Clone)]
pub struct SimplicialSignalDenoise {
    pub lambda: f64,
    pub iters: usize,
    pub lr: f64,
}

impl SimplicialSignalDenoise {
    pub fn new(lambda: f64, iters: usize, lr: f64) -> Self {
        Self { lambda, iters, lr }
    }

    /// Denoise edge signal: min_f ||f - y||² + lambda * f^T L_1 f.
    pub fn denoise(&self, noisy: &[f64], sc: &SimplicialComplexData) -> Vec<f64> {
        let l1 = HodgeLaplacian::l1(sc);
        if l1.is_empty() {
            return noisy.to_vec();
        }
        let mut f = noisy.to_vec();
        for _ in 0..self.iters {
            let l1f = matvec(&l1, &f);
            for (i, fi) in f.iter_mut().enumerate() {
                let grad = 2.0 * (*fi - noisy.get(i).copied().unwrap_or(0.0))
                    + 2.0 * self.lambda * l1f.get(i).copied().unwrap_or(0.0);
                *fi -= self.lr * grad;
            }
        }
        f
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TDA features
// ─────────────────────────────────────────────────────────────────────────────

/// Extract TDA features: Betti numbers at multiple filtration scales.
#[derive(Debug, Clone)]
pub struct TdaFeatureExtractor {
    pub thresholds: Vec<f64>,
}

impl TdaFeatureExtractor {
    pub fn new(thresholds: Vec<f64>) -> Self {
        Self { thresholds }
    }

    /// Compute Betti numbers beta_0, beta_1 at each threshold.
    pub fn extract(&self, adj: &[Vec<f64>]) -> Vec<[usize; 2]> {
        let n = adj.len();
        self.thresholds
            .iter()
            .map(|&t| {
                let edges: Vec<(usize, usize)> = (0..n)
                    .flat_map(|i| {
                        (i + 1..n).filter_map(
                            move |j| {
                                if adj[i][j] >= t {
                                    Some((i, j))
                                } else {
                                    None
                                }
                            },
                        )
                    })
                    .collect();
                let beta_0 = Self::connected_components(n, &edges);
                let e = edges.len();
                let beta_1 = if e >= n.saturating_sub(beta_0) {
                    e - n + beta_0
                } else {
                    0
                };
                [beta_0, beta_1]
            })
            .collect()
    }

    fn connected_components(n: usize, edges: &[(usize, usize)]) -> usize {
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }
        let mut comps = n;
        for &(u, v) in edges {
            let pu = find(&mut parent, u);
            let pv = find(&mut parent, v);
            if pu != pv {
                parent[pu] = pv;
                comps -= 1;
            }
        }
        comps
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Persistent homology
// ─────────────────────────────────────────────────────────────────────────────

/// A persistence pair (birth filtration value, death filtration value).
#[derive(Debug, Clone)]
pub struct PersistencePair {
    pub birth: f64,
    pub death: f64,
    pub dimension: usize,
}

impl PersistencePair {
    /// Persistence (lifetime) of the topological feature.
    pub fn persistence(&self) -> f64 {
        (self.death - self.birth).max(0.0)
    }
}

/// Differentiable persistent homology layer via Rips filtration.
#[derive(Debug, Clone)]
pub struct PersistentHomologyLayer {
    /// Regularization weight for persistence loss.
    pub reg: f64,
}

impl PersistentHomologyLayer {
    pub fn new(reg: f64) -> Self {
        Self { reg }
    }

    /// Compute persistence diagram from edge weights (Rips filtration, dim 0).
    pub fn compute_diagram(&self, adj: &[Vec<f64>]) -> Vec<PersistencePair> {
        let n = adj.len();
        if n == 0 {
            return Vec::new();
        }

        let mut edge_vals: Vec<(f64, usize, usize)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if adj[i][j] > 0.0 {
                    edge_vals.push((adj[i][j], i, j));
                }
            }
        }
        edge_vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut parent: Vec<usize> = (0..n).collect();
        let mut birth: Vec<f64> = vec![0.0; n];
        let mut pairs: Vec<PersistencePair> = Vec::new();

        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        for (w, u, v) in &edge_vals {
            let pu = find(&mut parent, *u);
            let pv = find(&mut parent, *v);
            if pu != pv {
                let (survivor, dying) = if birth[pu] <= birth[pv] {
                    (pu, pv)
                } else {
                    (pv, pu)
                };
                pairs.push(PersistencePair {
                    birth: birth[dying],
                    death: *w,
                    dimension: 0,
                });
                parent[dying] = survivor;
                birth[survivor] = birth[survivor].min(birth[dying]);
            }
        }

        // Essential class (one connected component survives to ∞, capped at 1.0)
        for i in 0..n {
            if find(&mut parent, i) == i && birth[i] == 0.0 {
                pairs.push(PersistencePair {
                    birth: 0.0,
                    death: 1.0,
                    dimension: 0,
                });
                break;
            }
        }

        pairs
    }

    /// Compute total persistence loss for gradient-based optimization.
    pub fn persistence_loss(&self, diagram: &[PersistencePair]) -> f64 {
        diagram
            .iter()
            .map(|p| {
                let pers = p.persistence();
                self.reg * pers * pers
            })
            .sum()
    }
}
