//! Advanced Riemannian geometry: Poincaré ball (hyperbolic space), geometric flows,
//! and extended metrics.

use super::*;
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ═══════════════════════════════════════════════════════════════════════════
// §A  Poincaré Ball (Hyperbolic Space)
// ═══════════════════════════════════════════════════════════════════════════

/// Poincaré ball model of n-dimensional hyperbolic space with curvature c > 0.
///
/// Points live inside the open ball `‖x‖² < 1/c`. The canonical choice is c = 1.
/// The Poincaré ball is the natural embedding space for hierarchical data (trees,
/// graphs with hierarchical structure) because it can represent exponentially many
/// nodes in polynomial space.
#[derive(Debug, Clone)]
pub struct PoincareBall {
    /// Ambient dimension of the hyperbolic space.
    pub dim: usize,
    /// Curvature parameter (c > 0). Default: 1.0.
    pub curvature: f64,
}

impl PoincareBall {
    /// Create a Poincaré ball with given dimension and curvature.
    pub fn new(dim: usize, curvature: f64) -> Self {
        assert!(curvature > 0.0, "curvature must be positive");
        Self { dim, curvature }
    }

    /// Conformal factor λ_c(x) = 2 / (1 − c ‖x‖²).
    ///
    /// Used to scale between Euclidean and hyperbolic distances.
    pub fn lambda_x(&self, x: &[f64]) -> f64 {
        let norm_sq: f64 = x.iter().map(|xi| xi * xi).sum();
        2.0 / (1.0 - self.curvature * norm_sq).max(1e-15)
    }

    /// Möbius addition in the Poincaré ball: x ⊕_c y.
    ///
    /// This is the group operation for the Poincaré ball model:
    /// (1 + 2c⟨x,y⟩ + c‖y‖²)x + (1 − c‖x‖²)y
    /// ────────────────────────────────────────────
    /// 1 + 2c⟨x,y⟩ + c²‖x‖²‖y‖²
    pub fn mobius_add(&self, x: &[f64], y: &[f64]) -> Vec<f64> {
        let c = self.curvature;
        let nx2: f64 = x.iter().map(|xi| xi * xi).sum();
        let ny2: f64 = y.iter().map(|yi| yi * yi).sum();
        let xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();

        let num_factor_x = 1.0 + 2.0 * c * xy + c * ny2;
        let num_factor_y = 1.0 - c * nx2;
        let denom = (1.0 + 2.0 * c * xy + c * c * nx2 * ny2).max(1e-15);

        x.iter()
            .zip(y.iter())
            .map(|(xi, yi)| (num_factor_x * xi + num_factor_y * yi) / denom)
            .collect()
    }

    /// Exponential map at origin 0: exp_0^c(v).
    ///
    /// Maps tangent vector v at the origin to a point on the Poincaré ball:
    /// exp_0^c(v) = tanh(√c ‖v‖) / (√c ‖v‖) * v
    pub fn exp_map_origin(&self, v: &[f64]) -> Vec<f64> {
        let c = self.curvature;
        let norm_v = v.iter().map(|vi| vi * vi).sum::<f64>().sqrt().max(1e-15);
        let sqrt_c = c.sqrt();
        let tanh_val = (sqrt_c * norm_v).tanh();
        let factor = tanh_val / (sqrt_c * norm_v);
        v.iter().map(|vi| factor * vi).collect()
    }

    /// Logarithmic map at origin 0: log_0^c(y).
    ///
    /// Maps a ball point y back to a tangent vector at origin:
    /// log_0^c(y) = artanh(√c ‖y‖) / (√c ‖y‖) * y
    pub fn log_map_origin(&self, y: &[f64]) -> Vec<f64> {
        let c = self.curvature;
        let norm_y = y.iter().map(|yi| yi * yi).sum::<f64>().sqrt().max(1e-15);
        let sqrt_c = c.sqrt();
        let atanh_val = (sqrt_c * norm_y).min(1.0 - 1e-7).atanh();
        let factor = atanh_val / (sqrt_c * norm_y);
        y.iter().map(|yi| factor * yi).collect()
    }

    /// Geodesic distance d_c(x, y) = (2/√c) artanh(√c ‖(−x) ⊕_c y‖).
    pub fn geodesic_distance(&self, x: &[f64], y: &[f64]) -> f64 {
        let c = self.curvature;
        let neg_x: Vec<f64> = x.iter().map(|xi| -xi).collect();
        let diff = self.mobius_add(&neg_x, y);
        let norm = diff.iter().map(|di| di * di).sum::<f64>().sqrt();
        let sqrt_c = c.sqrt();
        (2.0 / sqrt_c) * (sqrt_c * norm).min(1.0 - 1e-7).atanh()
    }

    /// Project a point onto the Poincaré ball (clamp if outside due to numerical drift).
    pub fn project(&self, x: &[f64]) -> Vec<f64> {
        let c = self.curvature;
        let norm_sq: f64 = x.iter().map(|xi| xi * xi).sum();
        let max_norm = ((1.0 / c) - 1e-5).sqrt();
        let norm = norm_sq.sqrt();
        if norm < max_norm {
            x.to_vec()
        } else {
            x.iter().map(|xi| xi * max_norm / norm).collect()
        }
    }
}

/// Learnable embeddings in the Poincaré ball, trained via Riemannian SGD.
///
/// Encodes hierarchical relationships: nearby items in the hierarchy are close
/// in the Poincaré ball, while the hierarchy depth maps to radial distance.
#[derive(Debug, Clone)]
pub struct HyperbolicEmbedding {
    /// The Poincaré ball manifold.
    pub ball: PoincareBall,
    /// Embedding vectors (n_items × dim), stored flat in row-major order.
    pub embeddings: Vec<f64>,
    /// Number of items to embed.
    pub n_items: usize,
    /// Learning rate for Riemannian SGD.
    pub lr: f64,
}

impl HyperbolicEmbedding {
    /// Create random embeddings near the origin (scale 0.01 to start inside ball).
    pub fn new(n_items: usize, dim: usize, curvature: f64, lr: f64, seed: u64) -> Self {
        let ball = PoincareBall::new(dim, curvature);
        let mut rng = StdRng::seed_from_u64(seed);
        let embeddings: Vec<f64> = (0..n_items * dim)
            .map(|_| rng.random::<f64>() * 0.01 - 0.005)
            .collect();
        Self {
            ball,
            embeddings,
            n_items,
            lr,
        }
    }

    /// Get the embedding slice for item i.
    pub fn get(&self, i: usize) -> &[f64] {
        let d = self.ball.dim;
        &self.embeddings[i * d..(i + 1) * d]
    }

    /// Riemannian SGD update for item i with Euclidean gradient.
    ///
    /// The Riemannian gradient = (λ_x / 2)² × Euclidean gradient.
    /// Update: x ← proj(x ⊕ exp_0(−lr * rgrad / λ_x)).
    pub fn rsgd_step(&mut self, i: usize, euclidean_grad: &[f64]) {
        let d = self.ball.dim;
        let start = i * d;
        let x: Vec<f64> = self.embeddings[start..start + d].to_vec();
        let lambda = self.ball.lambda_x(&x);
        let scale = (lambda / 2.0).powi(2);

        // Riemannian gradient
        let rgrad: Vec<f64> = euclidean_grad.iter().map(|g| scale * g).collect();
        // Step direction (gradient descent)
        let step: Vec<f64> = rgrad.iter().map(|g| -self.lr * g).collect();
        // Transport step to tangent at x and apply via Möbius addition
        let v_scaled: Vec<f64> = step.iter().map(|vi| vi / lambda.max(1e-15)).collect();
        let new_x_pre = self
            .ball
            .mobius_add(&x, &self.ball.exp_map_origin(&v_scaled));
        let new_x = self.ball.project(&new_x_pre);
        for (j, val) in new_x.iter().enumerate() {
            self.embeddings[start + j] = *val;
        }
    }

    /// Compute pairwise hyperbolic distance between items i and j.
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        self.ball.geodesic_distance(self.get(i), self.get(j))
    }
}

/// Hyperbolic fully-connected layer via Möbius transform.
///
/// Implements a linear layer operating in the Poincaré ball:
/// 1. Apply Euclidean weight matrix W to the log-mapped input.
/// 2. Exp-map the result back to the ball.
/// 3. Möbius-add the bias b.
///
/// Reference: Ganea et al. (2018) "Hyperbolic Neural Networks".
#[derive(Debug, Clone)]
pub struct HyperbolicLinear {
    /// Poincaré ball for the output space.
    pub ball: PoincareBall,
    /// Weight matrix (out_dim × in_dim) in Euclidean space.
    pub weights: Vec<f64>,
    /// Bias vector on the Poincaré ball.
    pub bias: Vec<f64>,
    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
}

impl HyperbolicLinear {
    /// Create a new hyperbolic linear layer with Xavier initialization.
    pub fn new(in_dim: usize, out_dim: usize, curvature: f64, seed: u64) -> Self {
        let ball = PoincareBall::new(out_dim, curvature);
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (6.0_f64 / (in_dim + out_dim) as f64).sqrt();
        let weights: Vec<f64> = (0..out_dim * in_dim)
            .map(|_| rng.random::<f64>() * 2.0 * scale - scale)
            .collect();
        let bias = vec![0.0_f64; out_dim];
        Self {
            ball,
            weights,
            bias,
            in_dim,
            out_dim,
        }
    }

    /// Forward pass: maps input x (Euclidean in_dim) to output on Poincaré ball.
    ///
    /// Pipeline: Euclidean linear → exp_map(origin) → Möbius add bias → project.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        // Euclidean linear map W * x
        let mut wx = vec![0.0_f64; self.out_dim];
        for o in 0..self.out_dim {
            for i in 0..self.in_dim {
                wx[o] += self.weights[o * self.in_dim + i] * x[i];
            }
        }
        // Exp-map linear result from origin onto ball
        let y_pre = self.ball.exp_map_origin(&wx);
        // Add bias via Möbius addition
        let y = self.ball.mobius_add(&y_pre, &self.bias);
        self.ball.project(&y)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §B  Geometric Flows on Graphs
// ═══════════════════════════════════════════════════════════════════════════

/// Ollivier-Ricci curvature for graph edges using Wasserstein-1 distance.
///
/// For edge (i, j), the curvature measures how quickly geodesics spread:
/// κ(i, j) = 1 − W₁(μ_i, μ_j) / d(i, j)
/// where μ_x is the uniform distribution over x's neighbors.
///
/// Reference: Ollivier (2009) "Ricci curvature of Markov chains on metric spaces".
#[derive(Debug, Clone)]
pub struct OllivierRicciCurvature {
    /// Adjacency list: edges\[i\] = Vec<(neighbor_j, weight)>.
    pub edges: Vec<Vec<(usize, f64)>>,
    /// Number of nodes in the graph.
    pub n_nodes: usize,
}

impl OllivierRicciCurvature {
    /// Create from adjacency list representation.
    pub fn new(n_nodes: usize, edges: Vec<Vec<(usize, f64)>>) -> Self {
        Self { n_nodes, edges }
    }

    /// Build neighbor probability distribution for node i (uniform over neighbors).
    fn neighbor_dist(&self, i: usize) -> Vec<(usize, f64)> {
        let neighbors = &self.edges[i];
        if neighbors.is_empty() {
            // Self-loop as degenerate distribution
            return vec![(i, 1.0)];
        }
        let n = neighbors.len() as f64;
        neighbors.iter().map(|&(j, _)| (j, 1.0 / n)).collect()
    }

    /// Compute shortest path distance d(i, j) via Dijkstra's algorithm.
    fn graph_distance(&self, src: usize, dst: usize) -> f64 {
        if src == dst {
            return 0.0;
        }
        let mut dist = vec![f64::INFINITY; self.n_nodes];
        dist[src] = 0.0;
        let mut visited = vec![false; self.n_nodes];
        for _ in 0..self.n_nodes {
            // Find closest unvisited node
            let u = (0..self.n_nodes).filter(|&k| !visited[k]).min_by(|&a, &b| {
                dist[a]
                    .partial_cmp(&dist[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let u = match u {
                Some(u) => u,
                None => break,
            };
            if dist[u].is_infinite() {
                break;
            }
            visited[u] = true;
            if u == dst {
                return dist[dst];
            }
            for &(v, w) in &self.edges[u] {
                if dist[u] + w < dist[v] {
                    dist[v] = dist[u] + w;
                }
            }
        }
        dist[dst]
    }

    /// Approximate Wasserstein-1 distance between neighbor distributions of i and j.
    ///
    /// Uses the primal formulation with graph distances as ground metric.
    fn wasserstein1(&self, mu_i: &[(usize, f64)], mu_j: &[(usize, f64)]) -> f64 {
        let mut total = 0.0_f64;
        for &(a, pa) in mu_i {
            let mut row_cost = 0.0_f64;
            let mut row_mass = 0.0_f64;
            for &(b, pb) in mu_j {
                let d = self.graph_distance(a, b);
                row_cost += pb * d;
                row_mass += pb;
            }
            if row_mass > 0.0 {
                total += pa * row_cost / row_mass;
            }
        }
        total
    }

    /// Compute Ollivier-Ricci curvature for edge (i, j).
    ///
    /// Returns a value in (−∞, 1]: positive curvature → clustering/clique structure,
    /// negative curvature → tree-like/expander structure.
    pub fn curvature(&self, i: usize, j: usize) -> f64 {
        let d_ij = self.graph_distance(i, j);
        if d_ij < 1e-12 {
            return 0.0;
        }
        let mu_i = self.neighbor_dist(i);
        let mu_j = self.neighbor_dist(j);
        let w1 = self.wasserstein1(&mu_i, &mu_j);
        1.0 - w1 / d_ij
    }
}

/// Discrete Ricci flow on graphs for geometry-aware graph processing.
///
/// Updates edge weights proportional to the deficit from a target curvature:
/// `w_{ij}^{t+1} = w_{ij}^t * (1 − lr * (κ_{ij}^t − κ_target))`
///
/// Converges to a graph where all edges have the target curvature.
/// Reference: Lin, Lu, Yau (2011) "Ricci curvature of graphs".
#[derive(Debug, Clone)]
pub struct RicciFlow {
    /// Ollivier-Ricci curvature computer.
    pub ricci: OllivierRicciCurvature,
    /// Target Ricci curvature (0 = flat, positive = sphere-like, negative = hyperbolic-like).
    pub target_curvature: f64,
    /// Flow step size (learning rate).
    pub lr: f64,
}

impl RicciFlow {
    /// Create a new Ricci flow on the given graph.
    pub fn new(
        n_nodes: usize,
        edges: Vec<Vec<(usize, f64)>>,
        target_curvature: f64,
        lr: f64,
    ) -> Self {
        Self {
            ricci: OllivierRicciCurvature::new(n_nodes, edges),
            target_curvature,
            lr,
        }
    }

    /// Perform one step of discrete Ricci flow (updates all edge weights).
    pub fn step(&mut self) {
        let n = self.ricci.n_nodes;
        for i in 0..n {
            for ej in 0..self.ricci.edges[i].len() {
                let j = self.ricci.edges[i][ej].0;
                let kappa = self.ricci.curvature(i, j);
                let deficit = kappa - self.target_curvature;
                let old_w = self.ricci.edges[i][ej].1;
                // Flow equation: keep weights positive
                let new_w = (old_w * (1.0 - self.lr * deficit)).max(1e-10);
                self.ricci.edges[i][ej].1 = new_w;
            }
        }
    }

    /// Run `n_steps` of Ricci flow.
    pub fn run(&mut self, n_steps: usize) {
        for _ in 0..n_steps {
            self.step();
        }
    }

    /// Get current edge weight between nodes i and j (if edge exists).
    pub fn weight(&self, i: usize, j: usize) -> Option<f64> {
        self.ricci.edges[i]
            .iter()
            .find(|&&(k, _)| k == j)
            .map(|&(_, w)| w)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §C  Extended RgMetrics
// ═══════════════════════════════════════════════════════════════════════════

/// Extended Riemannian geometry metrics for evaluation and diagnostics.
///
/// Records sectional curvature estimates, geodesic distance errors,
/// and hyperbolic embedding distortions.
#[derive(Debug, Clone, Default)]
pub struct RgMetrics {
    /// Sectional curvature estimates from triangle angle-excess method.
    pub sectional_curvatures: Vec<f64>,
    /// Absolute geodesic distance errors (|predicted − true|).
    pub geodesic_errors: Vec<f64>,
    /// Relative distortions of hyperbolic embeddings (|embedded/original − 1|).
    pub embedding_distortions: Vec<f64>,
}

impl RgMetrics {
    /// Create empty metrics container.
    pub fn new() -> Self {
        Self::default()
    }

    /// Estimate sectional curvature at point p on manifold using two tangent vectors.
    ///
    /// Uses geodesic triangle angle-excess: builds triangle with vertices
    /// p, q = exp_p(ε·u), r = exp_p(ε·v) and measures angle surplus/deficit.
    pub fn estimate_sectional_curvature<M: RiemannianManifold>(
        &mut self,
        manifold: &M,
        p: &[f64],
        u: &[f64],
        v: &[f64],
        epsilon: f64,
    ) -> f64 {
        let q = manifold.exp_map(p, &u.iter().map(|ui| epsilon * ui).collect::<Vec<_>>());
        let r = manifold.exp_map(p, &v.iter().map(|vi| epsilon * vi).collect::<Vec<_>>());

        let d_pq = manifold.geodesic_distance(p, &q);
        let d_pr = manifold.geodesic_distance(p, &r);
        let d_qr = manifold.geodesic_distance(&q, &r);

        // Spherical law of cosines to estimate angle at p
        let cos_flat = if d_pq * d_pr > 1e-12 {
            (d_pq * d_pq + d_pr * d_pr - d_qr * d_qr) / (2.0 * d_pq * d_pr)
        } else {
            1.0
        };
        // Sectional curvature estimate: angle_p − π/2 scaled by ε²
        let excess = cos_flat.clamp(-1.0, 1.0).acos() - std::f64::consts::FRAC_PI_2;
        let curvature = excess / (epsilon * epsilon).max(1e-20);
        self.sectional_curvatures.push(curvature);
        curvature
    }

    /// Record an absolute geodesic distance error |predicted − true|.
    pub fn record_geodesic_error(&mut self, predicted: f64, true_dist: f64) {
        self.geodesic_errors.push((predicted - true_dist).abs());
    }

    /// Mean absolute geodesic error.
    pub fn mean_geodesic_error(&self) -> f64 {
        if self.geodesic_errors.is_empty() {
            return 0.0;
        }
        self.geodesic_errors.iter().sum::<f64>() / self.geodesic_errors.len() as f64
    }

    /// Record embedding distortion: relative error |embedded_dist / original_dist − 1|.
    pub fn record_embedding_distortion(&mut self, original_dist: f64, embedded_dist: f64) {
        let distortion = if original_dist > 1e-12 {
            (embedded_dist / original_dist - 1.0).abs()
        } else {
            embedded_dist.abs()
        };
        self.embedding_distortions.push(distortion);
    }

    /// Mean embedding distortion.
    pub fn mean_embedding_distortion(&self) -> f64 {
        if self.embedding_distortions.is_empty() {
            return 0.0;
        }
        self.embedding_distortions.iter().sum::<f64>() / self.embedding_distortions.len() as f64
    }
}
