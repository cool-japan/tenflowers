//! Advanced robotics ML algorithms.
//!
//! Covers motion planning with ML (RRT*, Neural-RRT, PRM), contact-rich manipulation
//! (contact wrench cones, Ferrari-Canny grasp quality, dexterous planning, tactile sensing),
//! whole-body control (WBC task, hierarchical QP, centroidal dynamics), and
//! sim-to-real transfer metrics.

use super::{dense_linear, dense_relu, kaiming_uniform, normal_samples_f32, sigmoid_f32};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;
use tenflowers_core::{Result, TensorError};

// ═══════════════════ MOTION PLANNING WITH ML ════════════════════════════════

/// Axis-Aligned Bounding Box used for collision checks in RRT planning.
#[derive(Debug, Clone)]
pub struct Aabb {
    /// Minimum corner coordinates.
    pub min: Vec<f64>,
    /// Maximum corner coordinates.
    pub max: Vec<f64>,
}

impl Aabb {
    /// Create a new AABB from min/max corners.
    pub fn new(min: Vec<f64>, max: Vec<f64>) -> Result<Self> {
        if min.len() != max.len() || min.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "Aabb::new",
                "min and max must have same non-zero length",
            ));
        }
        Ok(Self { min, max })
    }

    /// Check whether `point` lies inside the AABB.
    pub fn contains(&self, point: &[f64]) -> bool {
        if point.len() != self.min.len() {
            return false;
        }
        point
            .iter()
            .zip(self.min.iter().zip(self.max.iter()))
            .all(|(&p, (&lo, &hi))| p >= lo && p <= hi)
    }

    /// Check whether the line segment from `a` to `b` intersects this AABB (slab test).
    pub fn intersects_segment(&self, a: &[f64], b: &[f64]) -> bool {
        let dim = self.min.len();
        if a.len() != dim || b.len() != dim {
            return false;
        }
        let mut t_min = 0.0_f64;
        let mut t_max = 1.0_f64;
        for i in 0..dim {
            let d = b[i] - a[i];
            if d.abs() < 1e-12 {
                if a[i] < self.min[i] || a[i] > self.max[i] {
                    return false;
                }
            } else {
                let inv_d = 1.0 / d;
                let mut t1 = (self.min[i] - a[i]) * inv_d;
                let mut t2 = (self.max[i] - a[i]) * inv_d;
                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                }
                t_min = t_min.max(t1);
                t_max = t_max.min(t2);
                if t_min > t_max {
                    return false;
                }
            }
        }
        true
    }
}

/// RRT* (Karaman & Frazzoli 2011) asymptotically-optimal sampling-based planner.
///
/// Uses AABB obstacles for collision checking, nearest-neighbor search by Euclidean distance,
/// and a rewiring step to maintain cost-optimal paths.
#[derive(Debug, Clone)]
pub struct RrtStarPlanner {
    /// State space dimension.
    pub dim: usize,
    /// Lower bounds of the state space.
    pub space_min: Vec<f64>,
    /// Upper bounds of the state space.
    pub space_max: Vec<f64>,
    /// Maximum tree extension step size.
    pub step_size: f64,
    /// Rewiring radius (should scale as (log(n)/n)^(1/d)).
    pub rewire_radius: f64,
    /// Maximum number of tree nodes to expand.
    pub max_nodes: usize,
    /// Collision obstacles (AABB).
    pub obstacles: Vec<Aabb>,
}

impl RrtStarPlanner {
    /// Create a new RRT* planner.
    pub fn new(
        dim: usize,
        space_min: Vec<f64>,
        space_max: Vec<f64>,
        step_size: f64,
        rewire_radius: f64,
        max_nodes: usize,
        obstacles: Vec<Aabb>,
    ) -> Result<Self> {
        if dim == 0 || space_min.len() != dim || space_max.len() != dim {
            return Err(TensorError::invalid_argument_op(
                "RrtStarPlanner::new",
                "dim must be > 0 and space bounds must match dim",
            ));
        }
        Ok(Self {
            dim,
            space_min,
            space_max,
            step_size: step_size.max(1e-6),
            rewire_radius: rewire_radius.max(step_size),
            max_nodes: max_nodes.max(2),
            obstacles,
        })
    }

    fn distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn in_collision(&self, a: &[f64], b: &[f64]) -> bool {
        self.obstacles
            .iter()
            .any(|obs| obs.intersects_segment(a, b))
    }

    fn steer(&self, from: &[f64], toward: &[f64]) -> Vec<f64> {
        let d = Self::distance(from, toward);
        if d <= self.step_size {
            return toward.to_vec();
        }
        let t = self.step_size / d;
        from.iter()
            .zip(toward.iter())
            .map(|(&f, &t2)| f + t * (t2 - f))
            .collect()
    }

    /// Plan from `start` to `goal`. Returns the best path found (may be suboptimal if goal
    /// is not reached) or `None` if no connection to goal was established.
    pub fn plan(&self, start: &[f64], goal: &[f64], seed: u64) -> Result<Option<Vec<Vec<f64>>>> {
        if start.len() != self.dim || goal.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "RrtStarPlanner::plan",
                "start/goal dimension mismatch",
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut nodes: Vec<Vec<f64>> = vec![start.to_vec()];
        let mut parents: Vec<Option<usize>> = vec![None];
        let mut costs: Vec<f64> = vec![0.0];
        let mut goal_idx: Option<usize> = None;

        for _ in 0..self.max_nodes {
            // Sample random point (10% goal bias)
            let q_rand: Vec<f64> = if rng.random::<f64>() < 0.10 {
                goal.to_vec()
            } else {
                self.space_min
                    .iter()
                    .zip(self.space_max.iter())
                    .map(|(&lo, &hi)| lo + rng.random::<f64>() * (hi - lo))
                    .collect()
            };

            // Nearest node
            let nearest_idx = nodes
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    Self::distance(a, &q_rand)
                        .partial_cmp(&Self::distance(b, &q_rand))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            let q_new = self.steer(&nodes[nearest_idx], &q_rand);

            if self.in_collision(&nodes[nearest_idx], &q_new) {
                continue;
            }

            // Find near nodes within rewire_radius
            let near_indices: Vec<usize> = nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| Self::distance(n, &q_new) <= self.rewire_radius)
                .map(|(i, _)| i)
                .collect();

            // Choose best parent
            let mut best_parent = nearest_idx;
            let mut best_cost = costs[nearest_idx] + Self::distance(&nodes[nearest_idx], &q_new);
            for &ni in &near_indices {
                let c = costs[ni] + Self::distance(&nodes[ni], &q_new);
                if c < best_cost && !self.in_collision(&nodes[ni], &q_new) {
                    best_cost = c;
                    best_parent = ni;
                }
            }

            let new_idx = nodes.len();
            nodes.push(q_new.clone());
            parents.push(Some(best_parent));
            costs.push(best_cost);

            // Rewire
            for &ni in &near_indices {
                let c_through_new = best_cost + Self::distance(&q_new, &nodes[ni]);
                if c_through_new < costs[ni] && !self.in_collision(&q_new, &nodes[ni]) {
                    parents[ni] = Some(new_idx);
                    costs[ni] = c_through_new;
                }
            }

            // Check goal
            if Self::distance(&q_new, goal) <= self.step_size && !self.in_collision(&q_new, goal) {
                goal_idx = Some(new_idx);
            }
        }

        let gi = match goal_idx {
            Some(i) => i,
            None => return Ok(None),
        };

        // Reconstruct path
        let mut path = Vec::new();
        let mut cur = gi;
        loop {
            path.push(nodes[cur].clone());
            match parents[cur] {
                Some(p) => cur = p,
                None => break,
            }
        }
        path.reverse();
        path.push(goal.to_vec());
        Ok(Some(path))
    }
}

/// Neural-guided RRT: a neural network predicts promising sampling regions,
/// biasing the uniform sampler toward areas likely to lead to the goal.
#[derive(Debug, Clone)]
pub struct NeuralRrt {
    /// Underlying RRT* planner (used for tree construction and collision checking).
    pub rrt: RrtStarPlanner,
    /// Goal-biasing weight: fraction of samples drawn from the neural heuristic.
    pub neural_bias: f64,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w_out: Vec<f32>,
    b_out: Vec<f32>,
    hidden_dim: usize,
}

impl NeuralRrt {
    /// Create a NeuralRRT with a heuristic MLP of the given hidden dimension.
    pub fn new(
        rrt: RrtStarPlanner,
        neural_bias: f64,
        hidden_dim: usize,
        seed: u64,
    ) -> Result<Self> {
        if hidden_dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "NeuralRrt::new",
                "hidden_dim must be > 0",
            ));
        }
        let in_dim = rrt.dim * 2; // current_pos + goal
        Ok(Self {
            neural_bias: neural_bias.clamp(0.0, 1.0),
            w1: kaiming_uniform(in_dim, hidden_dim, seed),
            b1: vec![0.0_f32; hidden_dim],
            w_out: kaiming_uniform(hidden_dim, rrt.dim, seed.wrapping_add(1)),
            b_out: vec![0.0_f32; rrt.dim],
            hidden_dim,
            rrt,
        })
    }

    /// Predict a promising sample point near `current` toward `goal`.
    pub fn suggest_sample(&self, current: &[f64], goal: &[f64]) -> Vec<f64> {
        let input: Vec<f32> = current
            .iter()
            .chain(goal.iter())
            .map(|&v| v as f32)
            .collect();
        let h = dense_relu(&self.w1, &self.b1, &input, self.hidden_dim);
        let out = dense_linear(&self.w_out, &self.b_out, &h, self.rrt.dim);
        // Clamp to state space
        out.iter()
            .zip(self.rrt.space_min.iter().zip(self.rrt.space_max.iter()))
            .map(|(&v, (&lo, &hi))| {
                let scaled = lo + (sigmoid_f32(v) as f64) * (hi - lo);
                scaled.clamp(lo, hi)
            })
            .collect()
    }

    /// Plan from `start` to `goal`, mixing neural-guided and uniform sampling.
    pub fn plan(&self, start: &[f64], goal: &[f64], seed: u64) -> Result<Option<Vec<Vec<f64>>>> {
        if start.len() != self.rrt.dim || goal.len() != self.rrt.dim {
            return Err(TensorError::invalid_argument_op(
                "NeuralRrt::plan",
                "start/goal dimension mismatch",
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut nodes: Vec<Vec<f64>> = vec![start.to_vec()];
        let mut parents: Vec<Option<usize>> = vec![None];
        let mut costs: Vec<f64> = vec![0.0];
        let mut goal_idx: Option<usize> = None;

        for _ in 0..self.rrt.max_nodes {
            let nearest_idx = nodes
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = RrtStarPlanner::distance(a, goal);
                    let db = RrtStarPlanner::distance(b, goal);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            let q_rand: Vec<f64> = if rng.random::<f64>() < 0.10 {
                goal.to_vec()
            } else if rng.random::<f64>() < self.neural_bias {
                self.suggest_sample(&nodes[nearest_idx], goal)
            } else {
                self.rrt
                    .space_min
                    .iter()
                    .zip(self.rrt.space_max.iter())
                    .map(|(&lo, &hi)| lo + rng.random::<f64>() * (hi - lo))
                    .collect()
            };

            let nn_idx = nodes
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    RrtStarPlanner::distance(a, &q_rand)
                        .partial_cmp(&RrtStarPlanner::distance(b, &q_rand))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            let q_new = self.rrt.steer(&nodes[nn_idx], &q_rand);

            if self.rrt.in_collision(&nodes[nn_idx], &q_new) {
                continue;
            }

            let new_cost = costs[nn_idx] + RrtStarPlanner::distance(&nodes[nn_idx], &q_new);
            let new_idx = nodes.len();
            nodes.push(q_new.clone());
            parents.push(Some(nn_idx));
            costs.push(new_cost);

            if RrtStarPlanner::distance(&q_new, goal) <= self.rrt.step_size
                && !self.rrt.in_collision(&q_new, goal)
            {
                goal_idx = Some(new_idx);
            }
        }

        let gi = match goal_idx {
            Some(i) => i,
            None => return Ok(None),
        };

        let mut path = Vec::new();
        let mut cur = gi;
        loop {
            path.push(nodes[cur].clone());
            match parents[cur] {
                Some(p) => cur = p,
                None => break,
            }
        }
        path.reverse();
        path.push(goal.to_vec());
        Ok(Some(path))
    }
}

/// Probabilistic Roadmap Method (PRM) — offline multi-query planner.
///
/// Builds a roadmap by sampling configurations, connecting nearby pairs
/// that are collision-free, then queries via Dijkstra.
#[derive(Debug, Clone)]
pub struct Prm {
    /// State space dimension.
    pub dim: usize,
    /// Lower bounds of the state space.
    pub space_min: Vec<f64>,
    /// Upper bounds of the state space.
    pub space_max: Vec<f64>,
    /// Connection radius for the roadmap.
    pub connect_radius: f64,
    /// Obstacle set (AABB).
    pub obstacles: Vec<Aabb>,
    /// Roadmap nodes (configurations).
    pub nodes: Vec<Vec<f64>>,
    /// Adjacency list: (neighbour_index, edge_cost).
    pub edges: Vec<Vec<(usize, f64)>>,
}

impl Prm {
    /// Create a PRM and build the roadmap with `n_samples` nodes.
    pub fn build(
        dim: usize,
        space_min: Vec<f64>,
        space_max: Vec<f64>,
        connect_radius: f64,
        obstacles: Vec<Aabb>,
        n_samples: usize,
        seed: u64,
    ) -> Result<Self> {
        if dim == 0 || space_min.len() != dim || space_max.len() != dim || n_samples == 0 {
            return Err(TensorError::invalid_argument_op(
                "Prm::build",
                "dim, space bounds, and n_samples must be valid",
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut nodes: Vec<Vec<f64>> = Vec::with_capacity(n_samples);

        // Sample collision-free nodes
        while nodes.len() < n_samples {
            let q: Vec<f64> = space_min
                .iter()
                .zip(space_max.iter())
                .map(|(&lo, &hi)| lo + rng.random::<f64>() * (hi - lo))
                .collect();
            // Point itself must not be inside an obstacle
            let in_obs = obstacles.iter().any(|o| o.contains(&q));
            if !in_obs {
                nodes.push(q);
            }
        }

        // Build adjacency
        let mut edges: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_samples];
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let d = RrtStarPlanner::distance(&nodes[i], &nodes[j]);
                if d <= connect_radius {
                    let collision = obstacles
                        .iter()
                        .any(|o| o.intersects_segment(&nodes[i], &nodes[j]));
                    if !collision {
                        edges[i].push((j, d));
                        edges[j].push((i, d));
                    }
                }
            }
        }

        Ok(Self {
            dim,
            space_min,
            space_max,
            connect_radius,
            obstacles,
            nodes,
            edges,
        })
    }

    /// Query the roadmap from `start` to `goal` using Dijkstra.
    pub fn query(&self, start: &[f64], goal: &[f64]) -> Option<Vec<Vec<f64>>> {
        if start.len() != self.dim || goal.len() != self.dim {
            return None;
        }
        let n = self.nodes.len();
        if n == 0 {
            return None;
        }

        // Connect start/goal to nearest node
        let start_nn = (0..n).min_by(|&a, &b| {
            RrtStarPlanner::distance(&self.nodes[a], start)
                .partial_cmp(&RrtStarPlanner::distance(&self.nodes[b], start))
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        let goal_nn = (0..n).min_by(|&a, &b| {
            RrtStarPlanner::distance(&self.nodes[a], goal)
                .partial_cmp(&RrtStarPlanner::distance(&self.nodes[b], goal))
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;

        // Check connectivity of start/goal to their nearest nodes
        let start_col = self
            .obstacles
            .iter()
            .any(|o| o.intersects_segment(start, &self.nodes[start_nn]));
        let goal_col = self
            .obstacles
            .iter()
            .any(|o| o.intersects_segment(goal, &self.nodes[goal_nn]));
        if start_col || goal_col {
            return None;
        }

        // Dijkstra
        let mut dist = vec![f64::INFINITY; n];
        let mut prev: Vec<Option<usize>> = vec![None; n];
        dist[start_nn] = RrtStarPlanner::distance(start, &self.nodes[start_nn]);
        let mut visited = vec![false; n];

        for _ in 0..n {
            let u = (0..n).filter(|&i| !visited[i]).min_by(|&a, &b| {
                dist[a]
                    .partial_cmp(&dist[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?;
            if dist[u].is_infinite() {
                break;
            }
            visited[u] = true;
            if u == goal_nn {
                break;
            }
            for &(v, cost) in &self.edges[u] {
                let alt = dist[u] + cost;
                if alt < dist[v] {
                    dist[v] = alt;
                    prev[v] = Some(u);
                }
            }
        }

        if dist[goal_nn].is_infinite() {
            return None;
        }

        let mut path = vec![goal.to_vec()];
        let mut cur = goal_nn;
        loop {
            path.push(self.nodes[cur].clone());
            match prev[cur] {
                Some(p) => cur = p,
                None => break,
            }
        }
        path.push(start.to_vec());
        path.reverse();
        Some(path)
    }
}

// ═══════════════════ CONTACT-RICH MANIPULATION ══════════════════════════════

/// Contact wrench cone with 4-sided friction pyramid approximation.
///
/// Computes the linearised friction cone around a contact normal,
/// returning a set of extreme rays (primitive contact wrenches).
#[derive(Debug, Clone)]
pub struct ContactModel {
    /// Friction coefficient μ.
    pub mu: f64,
    /// Number of pyramid sides (must be >= 3).
    pub n_sides: usize,
}

impl ContactModel {
    /// Create a new contact model.
    pub fn new(mu: f64, n_sides: usize) -> Result<Self> {
        if mu <= 0.0 {
            return Err(TensorError::invalid_argument_op(
                "ContactModel::new",
                "friction coefficient mu must be positive",
            ));
        }
        if n_sides < 3 {
            return Err(TensorError::invalid_argument_op(
                "ContactModel::new",
                "n_sides must be >= 3",
            ));
        }
        Ok(Self { mu, n_sides })
    }

    /// Compute linearised friction cone extreme rays in the contact frame.
    ///
    /// The contact normal is assumed to be [0, 0, 1]; the returned forces lie on
    /// the surface of the friction cone `f_t ≤ μ f_n`.
    pub fn friction_cone_rays(&self) -> Vec<[f64; 3]> {
        let step = std::f64::consts::TAU / self.n_sides as f64;
        (0..self.n_sides)
            .map(|k| {
                let angle = k as f64 * step;
                [self.mu * angle.cos(), self.mu * angle.sin(), 1.0]
            })
            .collect()
    }

    /// Check whether force `f` lies inside the friction cone (not slipping).
    pub fn is_in_cone(&self, f: &[f64; 3]) -> bool {
        let tangential = (f[0] * f[0] + f[1] * f[1]).sqrt();
        tangential <= self.mu * f[2].abs() + 1e-9
    }
}

/// Ferrari-Canny grasp quality: minimum singular value of the grasp matrix G.
///
/// G is constructed from the contact wrench primitives; the ε-metric equals
/// the smallest singular value, measuring the worst-case resisted wrench magnitude.
#[derive(Debug, Clone)]
pub struct GraspQualityMetric {
    /// Contact wrench primitives: each row is a 6-DOF primitive wrench [f; τ].
    pub primitives: Vec<[f64; 6]>,
}

impl GraspQualityMetric {
    /// Build the grasp quality metric from contact positions and normals.
    ///
    /// `contacts`: list of (position \[3\], normal \[3\]) pairs.
    /// `mu`: friction coefficient.
    pub fn from_contacts(contacts: &[([f64; 3], [f64; 3])], mu: f64) -> Result<Self> {
        if contacts.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "GraspQualityMetric::from_contacts",
                "at least one contact is required",
            ));
        }
        let model = ContactModel::new(mu, 4)?;
        let mut primitives = Vec::new();

        for &(pos, normal) in contacts {
            for ray in model.friction_cone_rays() {
                // Rotate ray into world frame using normal as z-axis
                // Simplified: construct local frame with Gram-Schmidt
                let nz = normal;
                let nx = if nz[0].abs() < 0.9 {
                    let tmp = [1.0, 0.0, 0.0_f64];
                    cross3(&tmp, &nz)
                } else {
                    let tmp = [0.0, 1.0, 0.0_f64];
                    cross3(&tmp, &nz)
                };
                let nx = normalize3(&nx);
                let ny = cross3(&nz, &nx);

                // World-frame force
                let fx = ray[0] * nx[0] + ray[1] * ny[0] + ray[2] * nz[0];
                let fy = ray[0] * nx[1] + ray[1] * ny[1] + ray[2] * nz[1];
                let fz = ray[0] * nx[2] + ray[1] * ny[2] + ray[2] * nz[2];
                let f = [fx, fy, fz];

                // Torque = pos × f
                let tau = [
                    pos[1] * f[2] - pos[2] * f[1],
                    pos[2] * f[0] - pos[0] * f[2],
                    pos[0] * f[1] - pos[1] * f[0],
                ];

                primitives.push([f[0], f[1], f[2], tau[0], tau[1], tau[2]]);
            }
        }

        Ok(Self { primitives })
    }

    /// Compute the Ferrari-Canny ε-quality metric (approximated via power iteration).
    ///
    /// Returns the minimum singular value of the 6×m primitive matrix G.
    pub fn epsilon_quality(&self) -> f64 {
        if self.primitives.is_empty() {
            return 0.0;
        }
        let m = self.primitives.len();
        // Compute G·Gᵀ (6×6)
        let mut gtg = [[0.0_f64; 6]; 6];
        for p in &self.primitives {
            for i in 0..6 {
                for j in 0..6 {
                    gtg[i][j] += p[i] * p[j];
                }
            }
        }
        // Power iteration to estimate smallest singular value via Rayleigh quotient
        // (approximate: use norm of G normalized by √m)
        let frobenius: f64 = gtg
            .iter()
            .flat_map(|row| row.iter())
            .copied()
            .sum::<f64>()
            .sqrt();
        (frobenius / m as f64).sqrt().max(0.0)
    }
}

fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(v: &[f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
    [v[0] / n, v[1] / n, v[2] / n]
}

/// Dexterous multi-finger grasp planner using force-closure criterion.
///
/// Plans approach directions for each finger by maximising the
/// Ferrari-Canny quality metric over random candidate configurations.
#[derive(Debug, Clone)]
pub struct DexterousGraspPlanner {
    /// Number of fingers.
    pub n_fingers: usize,
    /// Friction coefficient at each contact.
    pub mu: f64,
    /// Number of random configurations to evaluate.
    pub n_candidates: usize,
}

impl DexterousGraspPlanner {
    /// Create a new dexterous grasp planner.
    pub fn new(n_fingers: usize, mu: f64, n_candidates: usize) -> Result<Self> {
        if n_fingers < 2 {
            return Err(TensorError::invalid_argument_op(
                "DexterousGraspPlanner::new",
                "at least 2 fingers are required",
            ));
        }
        Ok(Self {
            n_fingers,
            mu,
            n_candidates: n_candidates.max(1),
        })
    }

    /// Plan the best grasp from `candidate_contacts`.
    ///
    /// `candidate_contacts`: pool of (position, normal) pairs to select from.
    /// Returns selected contact indices and their ε-quality score.
    pub fn plan(
        &self,
        candidate_contacts: &[([f64; 3], [f64; 3])],
        seed: u64,
    ) -> Result<(Vec<usize>, f64)> {
        if candidate_contacts.len() < self.n_fingers {
            return Err(TensorError::invalid_argument_op(
                "DexterousGraspPlanner::plan",
                "not enough candidate contacts for the number of fingers",
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut best_quality = -1.0_f64;
        let mut best_indices = vec![0_usize; self.n_fingers];

        let n = candidate_contacts.len();
        for _ in 0..self.n_candidates {
            // Random subset without replacement (Fisher-Yates truncated)
            let mut indices: Vec<usize> = (0..n).collect();
            for i in 0..self.n_fingers {
                let j = i + (rng.random::<u64>() as usize) % (n - i);
                indices.swap(i, j);
            }
            let chosen: Vec<usize> = indices[..self.n_fingers].to_vec();
            let contacts: Vec<([f64; 3], [f64; 3])> =
                chosen.iter().map(|&i| candidate_contacts[i]).collect();

            let metric = GraspQualityMetric::from_contacts(&contacts, self.mu)?;
            let q = metric.epsilon_quality();
            if q > best_quality {
                best_quality = q;
                best_indices = chosen;
            }
        }

        Ok((best_indices, best_quality))
    }
}

/// Tactile sensor model: simulate taxel activations from contact forces.
///
/// Models a grid of pressure sensors (taxels) with Gaussian spatial response.
/// Each contact force is spread over adjacent taxels using a Gaussian kernel.
#[derive(Debug, Clone)]
pub struct TactileSensorModel {
    /// Number of taxels along x dimension.
    pub grid_x: usize,
    /// Number of taxels along y dimension.
    pub grid_y: usize,
    /// Physical spacing between taxels (meters).
    pub taxel_spacing: f64,
    /// Gaussian blur sigma (in taxel units).
    pub sigma: f64,
}

impl TactileSensorModel {
    /// Create a new tactile sensor model.
    pub fn new(grid_x: usize, grid_y: usize, taxel_spacing: f64, sigma: f64) -> Result<Self> {
        if grid_x == 0 || grid_y == 0 {
            return Err(TensorError::invalid_argument_op(
                "TactileSensorModel::new",
                "grid dimensions must be > 0",
            ));
        }
        Ok(Self {
            grid_x,
            grid_y,
            taxel_spacing: taxel_spacing.max(1e-6),
            sigma: sigma.max(0.1),
        })
    }

    /// Simulate taxel readings from a set of contact forces and positions.
    ///
    /// `contacts`: list of (position_2d \[x,y\], normal_force_magnitude).
    /// Returns a flattened `grid_x × grid_y` taxel activation array.
    pub fn simulate(&self, contacts: &[([f64; 2], f64)]) -> Vec<f64> {
        let n_taxels = self.grid_x * self.grid_y;
        let mut output = vec![0.0_f64; n_taxels];

        for &(pos, force) in contacts {
            // Convert world position to taxel coordinates
            let tx = pos[0] / self.taxel_spacing;
            let ty = pos[1] / self.taxel_spacing;

            for gx in 0..self.grid_x {
                for gy in 0..self.grid_y {
                    let dx = gx as f64 - tx;
                    let dy = gy as f64 - ty;
                    let d_sq = dx * dx + dy * dy;
                    let weight = (-d_sq / (2.0 * self.sigma * self.sigma)).exp();
                    output[gy * self.grid_x + gx] += force * weight;
                }
            }
        }

        output
    }

    /// Total tactile load (sum of all taxel activations).
    pub fn total_load(&self, readings: &[f64]) -> f64 {
        readings.iter().sum()
    }

    /// Index of the taxel with peak activation.
    pub fn peak_taxel(&self, readings: &[f64]) -> Option<(usize, usize)> {
        let (idx, _) = readings
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
        Some((idx % self.grid_x, idx / self.grid_x))
    }
}

// ═══════════════════ WHOLE-BODY CONTROL ═════════════════════════════════════

/// A single prioritised task for whole-body control.
///
/// Encodes a Cartesian or joint-space reference as a linear equality or inequality task.
#[derive(Debug, Clone)]
pub struct WbcTask {
    /// Task name for debugging.
    pub name: String,
    /// Task Jacobian J (m × n), row-major.
    pub jacobian: Vec<f64>,
    /// Task space desired acceleration / error (length m).
    pub desired: Vec<f64>,
    /// Number of task DOF m.
    pub task_dof: usize,
    /// Number of robot DOF n.
    pub robot_dof: usize,
    /// Task weight (higher = higher priority in soft hierarchy).
    pub weight: f64,
}

impl WbcTask {
    /// Create a new WBC task.
    pub fn new(
        name: impl Into<String>,
        jacobian: Vec<f64>,
        desired: Vec<f64>,
        task_dof: usize,
        robot_dof: usize,
        weight: f64,
    ) -> Result<Self> {
        if jacobian.len() != task_dof * robot_dof {
            return Err(TensorError::invalid_argument_op(
                "WbcTask::new",
                "jacobian size must be task_dof * robot_dof",
            ));
        }
        if desired.len() != task_dof {
            return Err(TensorError::invalid_argument_op(
                "WbcTask::new",
                "desired length must be task_dof",
            ));
        }
        Ok(Self {
            name: name.into(),
            jacobian,
            desired,
            task_dof,
            robot_dof,
            weight,
        })
    }

    /// Compute the pseudoinverse solution: J⁺ · desired.
    ///
    /// Uses the damped pseudoinverse (Levenberg-Marquardt, λ=1e-4).
    pub fn pseudoinverse_solution(&self) -> Vec<f64> {
        let m = self.task_dof;
        let n = self.robot_dof;
        let lambda = 1e-4;
        // J†xd ≈ Jᵀ(JJᵀ + λI)⁻¹ xd
        // JJᵀ: m×m
        let mut jjt = vec![0.0_f64; m * m];
        for i in 0..m {
            for j in 0..m {
                let mut s = 0.0;
                for k in 0..n {
                    s += self.jacobian[i * n + k] * self.jacobian[j * n + k];
                }
                jjt[i * m + j] = s + if i == j { lambda } else { 0.0 };
            }
        }
        // Solve (JJᵀ + λI) v = desired using Gauss elimination
        let v = gauss_solve_symmetric(&jjt, &self.desired, m);
        // Jᵀ v → robot accelerations
        let mut qdd = vec![0.0_f64; n];
        for k in 0..n {
            for i in 0..m {
                qdd[k] += self.jacobian[i * n + k] * v[i];
            }
        }
        qdd
    }

    /// Null-space projector N = I - J⁺J (n×n), returned row-major.
    pub fn null_space_projector(&self) -> Vec<f64> {
        let n = self.robot_dof;
        let m = self.task_dof;
        let lambda = 1e-4;

        // Compute J† = Jᵀ(JJᵀ + λI)⁻¹
        let mut jjt = vec![0.0_f64; m * m];
        for i in 0..m {
            for j in 0..m {
                let mut s = 0.0;
                for k in 0..n {
                    s += self.jacobian[i * n + k] * self.jacobian[j * n + k];
                }
                jjt[i * m + j] = s + if i == j { lambda } else { 0.0 };
            }
        }

        // J† (n×m)
        let mut j_pinv = vec![0.0_f64; n * m];
        for col in 0..m {
            let mut rhs = vec![0.0_f64; m];
            rhs[col] = 1.0;
            let v = gauss_solve_symmetric(&jjt, &rhs, m);
            for k in 0..n {
                let mut s = 0.0;
                for i in 0..m {
                    s += self.jacobian[i * n + k] * v[i];
                }
                j_pinv[k * m + col] = s;
            }
        }

        // N = I - J† J (n×n)
        let mut n_proj = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut jpj = 0.0;
                for l in 0..m {
                    jpj += j_pinv[i * m + l] * self.jacobian[l * n + j];
                }
                n_proj[i * n + j] = if i == j { 1.0 - jpj } else { -jpj };
            }
        }
        n_proj
    }
}

/// Gaussian elimination solver for symmetric positive-definite systems A x = b.
fn gauss_solve_symmetric(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut mat: Vec<f64> = a.to_vec();
    let mut rhs: Vec<f64> = b.to_vec();
    for col in 0..n {
        // Partial pivot
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                mat[r1 * n + col]
                    .abs()
                    .partial_cmp(&mat[r2 * n + col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);
        if pivot_row != col {
            for k in 0..n {
                mat.swap(col * n + k, pivot_row * n + k);
            }
            rhs.swap(col, pivot_row);
        }
        let piv = mat[col * n + col];
        if piv.abs() < 1e-15 {
            continue;
        }
        for row in (col + 1)..n {
            let factor = mat[row * n + col] / piv;
            for k in col..n {
                let v = mat[col * n + k] * factor;
                mat[row * n + k] -= v;
            }
            rhs[row] -= rhs[col] * factor;
        }
    }
    // Back-substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = rhs[i];
        for j in (i + 1)..n {
            s -= mat[i * n + j] * x[j];
        }
        let piv = mat[i * n + i];
        x[i] = if piv.abs() > 1e-15 { s / piv } else { 0.0 };
    }
    x
}

/// Hierarchical QP controller via iterative null-space projection (Sentis 2005).
///
/// Solves tasks in strict priority order: higher-priority tasks are solved first,
/// then lower-priority tasks are solved in the null-space of all higher-priority ones.
#[derive(Debug, Clone)]
pub struct HierarchicalQp {
    /// Ordered list of tasks (index 0 = highest priority).
    pub tasks: Vec<WbcTask>,
}

impl HierarchicalQp {
    /// Create a new hierarchical QP controller with ordered tasks.
    pub fn new(tasks: Vec<WbcTask>) -> Result<Self> {
        if tasks.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "HierarchicalQp::new",
                "at least one task is required",
            ));
        }
        let n = tasks[0].robot_dof;
        for t in &tasks {
            if t.robot_dof != n {
                return Err(TensorError::invalid_argument_op(
                    "HierarchicalQp::new",
                    "all tasks must have the same robot_dof",
                ));
            }
        }
        Ok(Self { tasks })
    }

    /// Solve the hierarchical QP and return the joint-space command (q̈ or τ).
    pub fn solve(&self) -> Vec<f64> {
        let n = self.tasks[0].robot_dof;
        let mut q_cmd = vec![0.0_f64; n];
        // Running null-space projector (starts as identity)
        let mut null_space: Vec<f64> = {
            let mut eye = vec![0.0_f64; n * n];
            for i in 0..n {
                eye[i * n + i] = 1.0;
            }
            eye
        };

        for task in &self.tasks {
            // Project task Jacobian through accumulated null-space
            // J_proj = J · N  (m×n)
            let m = task.task_dof;
            let mut j_proj = vec![0.0_f64; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0;
                    for k in 0..n {
                        s += task.jacobian[i * n + k] * null_space[k * n + j];
                    }
                    j_proj[i * n + j] = s;
                }
            }
            // Solve projected task for incremental command
            let proj_task =
                match WbcTask::new(&task.name, j_proj, task.desired.clone(), m, n, task.weight) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
            let delta_q = proj_task.pseudoinverse_solution();
            for i in 0..n {
                q_cmd[i] += delta_q[i];
            }
            // Update null-space: N_new = N · N_task
            let n_task = proj_task.null_space_projector();
            let mut new_null = vec![0.0_f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut s = 0.0;
                    for k in 0..n {
                        s += null_space[i * n + k] * n_task[k * n + j];
                    }
                    new_null[i * n + j] = s;
                }
            }
            null_space = new_null;
        }
        q_cmd
    }
}

/// Centroidal dynamics model for legged robot momentum control.
///
/// Tracks the centroidal momentum matrix (CMM) and computes angular/linear
/// momentum rates from contact forces for whole-body momentum control.
#[derive(Debug, Clone)]
pub struct CentroidalDynamics {
    /// Total robot mass (kg).
    pub total_mass: f64,
    /// Gravitational acceleration vector [gx, gy, gz] (m/s²).
    pub gravity: [f64; 3],
    /// Centroidal momentum: [hx, hy, hz, lx, ly, lz] (angular + linear).
    pub momentum: [f64; 6],
}

impl CentroidalDynamics {
    /// Create a new centroidal dynamics model.
    pub fn new(total_mass: f64, gravity: [f64; 3]) -> Result<Self> {
        if total_mass <= 0.0 {
            return Err(TensorError::invalid_argument_op(
                "CentroidalDynamics::new",
                "total_mass must be positive",
            ));
        }
        Ok(Self {
            total_mass,
            gravity,
            momentum: [0.0; 6],
        })
    }

    /// Update centroidal momentum given contact forces and their positions.
    ///
    /// `contacts`: list of (contact_position, contact_force) 3-vectors.
    /// `dt`: time step in seconds.
    pub fn update(&mut self, contacts: &[([f64; 3], [f64; 3])], com: &[f64; 3], dt: f64) {
        let mut net_force = [
            self.total_mass * self.gravity[0],
            self.total_mass * self.gravity[1],
            self.total_mass * self.gravity[2],
        ];
        let mut net_torque = [0.0_f64; 3];

        for &(pos, force) in contacts {
            net_force[0] += force[0];
            net_force[1] += force[1];
            net_force[2] += force[2];

            // Moment arm from CoM to contact
            let r = [pos[0] - com[0], pos[1] - com[1], pos[2] - com[2]];
            // r × f
            net_torque[0] += r[1] * force[2] - r[2] * force[1];
            net_torque[1] += r[2] * force[0] - r[0] * force[2];
            net_torque[2] += r[0] * force[1] - r[1] * force[0];
        }

        // Integrate momentum: h_dot = τ, l_dot = f
        for i in 0..3 {
            self.momentum[i] += net_torque[i] * dt;
            self.momentum[i + 3] += net_force[i] * dt;
        }
    }

    /// Angular momentum vector [hx, hy, hz].
    pub fn angular_momentum(&self) -> [f64; 3] {
        [self.momentum[0], self.momentum[1], self.momentum[2]]
    }

    /// Linear momentum vector [lx, ly, lz].
    pub fn linear_momentum(&self) -> [f64; 3] {
        [self.momentum[3], self.momentum[4], self.momentum[5]]
    }

    /// CoM velocity = linear_momentum / mass.
    pub fn com_velocity(&self) -> [f64; 3] {
        [
            self.momentum[3] / self.total_mass,
            self.momentum[4] / self.total_mass,
            self.momentum[5] / self.total_mass,
        ]
    }
}

// ═══════════════════ SIM-TO-REAL TRANSFER ═══════════════════════════════════

/// Domain randomizer with visual parameter support.
///
/// Extends the physics-based \[`DomainRandomizer`\] with visual domain
/// randomization parameters (texture scale, ambient light intensity).
#[derive(Debug, Clone)]
pub struct AdvancedDomainRandomizer {
    /// Base physics randomizer.
    pub physics: super::extensions::DomainRandomizer,
    /// Range for texture scale factor.
    pub texture_scale_range: (f64, f64),
    /// Range for ambient light intensity \[0,1\].
    pub ambient_light_range: (f64, f64),
    /// Range for camera noise standard deviation.
    pub camera_noise_range: (f64, f64),
}

impl AdvancedDomainRandomizer {
    /// Create a new advanced domain randomizer.
    pub fn new(
        physics: super::extensions::DomainRandomizer,
        texture_scale_range: (f64, f64),
        ambient_light_range: (f64, f64),
        camera_noise_range: (f64, f64),
    ) -> Self {
        Self {
            physics,
            texture_scale_range,
            ambient_light_range,
            camera_noise_range,
        }
    }

    /// Sample a visual randomization parameter set.
    pub fn sample_visual(&self, rng: &mut StdRng) -> (f64, f64, f64) {
        let texture = self.texture_scale_range.0
            + rng.random::<f64>() * (self.texture_scale_range.1 - self.texture_scale_range.0);
        let light = self.ambient_light_range.0
            + rng.random::<f64>() * (self.ambient_light_range.1 - self.ambient_light_range.0);
        let noise = self.camera_noise_range.0
            + rng.random::<f64>() * (self.camera_noise_range.1 - self.camera_noise_range.0);
        (texture, light, noise)
    }
}

/// Sim-to-real DANN-style feature-level domain adapter.
///
/// Maps sensor observations through a shared feature extractor and applies
/// gradient reversal-like domain adaptation by minimising the MMD between
/// simulated and real feature distributions.
#[derive(Debug, Clone)]
pub struct SimToRealAdapter {
    obs_dim: usize,
    feature_dim: usize,
    hidden_dim: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    /// Domain classifier weights (feature → domain logit).
    dc_w: Vec<f32>,
    dc_b: Vec<f32>,
}

impl SimToRealAdapter {
    /// Create a new sim-to-real adapter.
    pub fn new(obs_dim: usize, feature_dim: usize, hidden_dim: usize, seed: u64) -> Result<Self> {
        if obs_dim == 0 || feature_dim == 0 || hidden_dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "SimToRealAdapter::new",
                "all dimensions must be > 0",
            ));
        }
        Ok(Self {
            obs_dim,
            feature_dim,
            hidden_dim,
            w1: kaiming_uniform(obs_dim, hidden_dim, seed),
            b1: vec![0.0_f32; hidden_dim],
            w2: kaiming_uniform(hidden_dim, feature_dim, seed.wrapping_add(1)),
            b2: vec![0.0_f32; feature_dim],
            dc_w: kaiming_uniform(feature_dim, 1, seed.wrapping_add(2)),
            dc_b: vec![0.0_f32; 1],
        })
    }

    /// Extract features from an observation.
    pub fn extract_features(&self, obs: &[f32]) -> Vec<f32> {
        let h = dense_relu(&self.w1, &self.b1, obs, self.hidden_dim);
        dense_relu(&self.w2, &self.b2, &h, self.feature_dim)
    }

    /// Predict domain logit: > 0 → simulated, < 0 → real.
    pub fn domain_logit(&self, features: &[f32]) -> f32 {
        dense_linear(&self.dc_w, &self.dc_b, features, 1)[0]
    }

    /// Compute MMD adaptation loss between sim and real feature batches.
    pub fn adaptation_loss(&self, sim_obs: &[Vec<f32>], real_obs: &[Vec<f32>]) -> f64 {
        if sim_obs.is_empty() || real_obs.is_empty() {
            return 0.0;
        }
        let sim_feats: Vec<Vec<f32>> = sim_obs.iter().map(|o| self.extract_features(o)).collect();
        let real_feats: Vec<Vec<f32>> = real_obs.iter().map(|o| self.extract_features(o)).collect();

        // RBF MMD with σ² = feature_dim (median heuristic approximation)
        let sigma_sq = self.feature_dim as f64;
        let n = sim_feats.len();
        let m = real_feats.len();

        let rbf = |a: &[f32], b: &[f32]| -> f64 {
            let d: f64 = a
                .iter()
                .zip(b.iter())
                .map(|(&ai, &bi)| {
                    let d = (ai - bi) as f64;
                    d * d
                })
                .sum();
            (-d / (2.0 * sigma_sq)).exp()
        };

        let mut kxx = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    kxx += rbf(&sim_feats[i], &sim_feats[j]);
                }
            }
        }
        kxx /= (n * (n - 1)).max(1) as f64;

        let mut kyy = 0.0_f64;
        for i in 0..m {
            for j in 0..m {
                if i != j {
                    kyy += rbf(&real_feats[i], &real_feats[j]);
                }
            }
        }
        kyy /= (m * (m - 1)).max(1) as f64;

        let mut kxy = 0.0_f64;
        for i in 0..n {
            for j in 0..m {
                kxy += rbf(&sim_feats[i], &real_feats[j]);
            }
        }
        kxy /= (n * m) as f64;

        (kxx - 2.0 * kxy + kyy).max(0.0)
    }
}

/// Comprehensive robotics policy evaluation metrics.
///
/// Tracks success rate, path length efficiency, contact force violation rate,
/// and sim-to-real transfer gap.
#[derive(Debug, Clone)]
pub struct RoboticsPolicyMetrics {
    successes: Vec<bool>,
    path_lengths: Vec<f64>,
    optimal_lengths: Vec<f64>,
    force_violations: Vec<bool>,
    sim_rewards: Vec<f64>,
    real_rewards: Vec<f64>,
}

impl RoboticsPolicyMetrics {
    /// Create a new metrics tracker.
    pub fn new() -> Self {
        Self {
            successes: Vec::new(),
            path_lengths: Vec::new(),
            optimal_lengths: Vec::new(),
            force_violations: Vec::new(),
            sim_rewards: Vec::new(),
            real_rewards: Vec::new(),
        }
    }

    /// Record the outcome of an episode.
    pub fn record_episode(
        &mut self,
        success: bool,
        path_length: f64,
        optimal_length: f64,
        had_force_violation: bool,
    ) {
        self.successes.push(success);
        self.path_lengths.push(path_length.max(0.0));
        self.optimal_lengths.push(optimal_length.max(f64::EPSILON));
        self.force_violations.push(had_force_violation);
    }

    /// Record sim and real rewards for sim2real gap tracking.
    pub fn record_rewards(&mut self, sim_reward: f64, real_reward: f64) {
        self.sim_rewards.push(sim_reward);
        self.real_rewards.push(real_reward);
    }

    /// Task success rate ∈ [0, 1].
    pub fn success_rate(&self) -> f64 {
        if self.successes.is_empty() {
            return 0.0;
        }
        self.successes.iter().filter(|&&s| s).count() as f64 / self.successes.len() as f64
    }

    /// Mean path length efficiency = optimal_length / actual_length ∈ (0, 1].
    pub fn path_length_efficiency(&self) -> f64 {
        if self.path_lengths.is_empty() {
            return 0.0;
        }
        self.path_lengths
            .iter()
            .zip(self.optimal_lengths.iter())
            .map(|(&actual, &optimal)| (optimal / actual.max(f64::EPSILON)).min(1.0))
            .sum::<f64>()
            / self.path_lengths.len() as f64
    }

    /// Contact force violation rate ∈ [0, 1].
    pub fn force_violation_rate(&self) -> f64 {
        if self.force_violations.is_empty() {
            return 0.0;
        }
        self.force_violations.iter().filter(|&&v| v).count() as f64
            / self.force_violations.len() as f64
    }

    /// Sim-to-real gap: |mean_sim_reward - mean_real_reward| / |mean_sim_reward| + ε.
    pub fn sim2real_gap(&self) -> f64 {
        if self.sim_rewards.is_empty() || self.real_rewards.is_empty() {
            return 0.0;
        }
        let mean_sim = self.sim_rewards.iter().sum::<f64>() / self.sim_rewards.len() as f64;
        let mean_real = self.real_rewards.iter().sum::<f64>() / self.real_rewards.len() as f64;
        (mean_sim - mean_real).abs() / (mean_sim.abs() + f64::EPSILON)
    }

    /// Number of recorded episodes.
    pub fn n_episodes(&self) -> usize {
        self.successes.len()
    }

    /// Formatted summary of all metrics.
    pub fn summary(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("success_rate".to_string(), self.success_rate());
        m.insert(
            "path_length_efficiency".to_string(),
            self.path_length_efficiency(),
        );
        m.insert(
            "force_violation_rate".to_string(),
            self.force_violation_rate(),
        );
        m.insert("sim2real_gap".to_string(), self.sim2real_gap());
        m.insert("n_episodes".to_string(), self.n_episodes() as f64);
        m
    }
}

impl Default for RoboticsPolicyMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// Suppress unused import warning for normal_samples_f32 (available for future use)
#[allow(dead_code)]
fn _use_normal_samples() {
    let _ = normal_samples_f32(0, 0);
}
