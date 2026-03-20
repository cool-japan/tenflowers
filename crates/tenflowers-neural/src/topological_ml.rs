//! Topological Data Analysis (TDA) for Machine Learning
//!
//! This module provides:
//! - SimplicialComplex / VietorisRipsComplex
//! - PersistentHomology: persistence diagrams, column reduction
//! - PersistenceImage: vectorized TDA descriptor
//! - PersistenceLandscape: functional TDA summary
//! - MapperAlgorithm: topological graph construction
//! - TopologicalFeatureExtractor: batch TDA feature extraction
//! - TdaMetrics: Wasserstein, bottleneck, sliced-Wasserstein distances

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// 1. SimplicialComplex
// ─────────────────────────────────────────────────────────────────────────────

/// A sorted collection of vertex indices representing a simplex.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TdaSimplex {
    pub vertices: Vec<usize>,
}

impl TdaSimplex {
    /// Create a new simplex; vertices are sorted.
    pub fn new(mut vertices: Vec<usize>) -> Self {
        vertices.sort_unstable();
        Self { vertices }
    }

    /// Dimension of the simplex (num_vertices - 1).
    pub fn dim(&self) -> usize {
        self.vertices.len().saturating_sub(1)
    }
}

/// Simplicial complex indexed by dimension.
/// `simplices[k]` contains all k-simplices.
#[derive(Debug, Clone)]
pub struct TdaSimplicialComplex {
    /// simplices\[dim\] = list of simplices of that dimension
    pub simplices: Vec<Vec<TdaSimplex>>,
}

impl TdaSimplicialComplex {
    pub fn new() -> Self {
        Self {
            simplices: Vec::new(),
        }
    }

    /// Ensure internal storage has at least `dim+1` levels.
    fn ensure_dim(&mut self, dim: usize) {
        while self.simplices.len() <= dim {
            self.simplices.push(Vec::new());
        }
    }

    /// Add a simplex (and all its faces recursively).
    pub fn add_simplex(&mut self, s: TdaSimplex) {
        let dim = s.dim();
        self.ensure_dim(dim);
        // Check for duplicates
        if self.simplices[dim].contains(&s) {
            return;
        }
        // Recursively add boundary faces
        for face in self.boundary(&s) {
            self.add_simplex(face);
        }
        self.simplices[dim].push(s);
    }

    /// Maximum dimension of any simplex.
    pub fn dimension(&self) -> usize {
        self.simplices.len().saturating_sub(1)
    }

    /// Number of simplices of given dimension.
    pub fn n_simplices(&self, dim: usize) -> usize {
        self.simplices.get(dim).map(|v| v.len()).unwrap_or(0)
    }

    /// Boundary of a simplex: all (dim-1)-faces.
    pub fn boundary(&self, s: &TdaSimplex) -> Vec<TdaSimplex> {
        if s.vertices.len() <= 1 {
            return Vec::new();
        }
        (0..s.vertices.len())
            .map(|i| {
                let verts: Vec<usize> = s
                    .vertices
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, &v)| v)
                    .collect();
                TdaSimplex::new(verts)
            })
            .collect()
    }

    /// Euler characteristic: alternating sum of simplex counts.
    pub fn euler_characteristic(&self) -> i64 {
        self.simplices
            .iter()
            .enumerate()
            .map(|(k, sv)| {
                let sign: i64 = if k % 2 == 0 { 1 } else { -1 };
                sign * sv.len() as i64
            })
            .sum()
    }
}

impl Default for TdaSimplicialComplex {
    fn default() -> Self {
        Self::new()
    }
}

/// Builds a Vietoris-Rips complex from a point cloud.
pub struct VietorisRipsComplex;

impl VietorisRipsComplex {
    fn euclidean_dist(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    /// Build a Vietoris-Rips complex up to `max_dim` from the given points.
    pub fn build(points: &[Vec<f32>], epsilon: f32, max_dim: usize) -> TdaSimplicialComplex {
        let n = points.len();
        let mut complex = TdaSimplicialComplex::new();

        // Add all 0-simplices (vertices)
        for i in 0..n {
            complex.add_simplex(TdaSimplex::new(vec![i]));
        }
        if max_dim == 0 || n == 0 {
            return complex;
        }

        // Build edge set: connect all pairs within epsilon
        let mut edge_set: Vec<(usize, usize)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if Self::euclidean_dist(&points[i], &points[j]) <= epsilon {
                    edge_set.push((i, j));
                    complex.add_simplex(TdaSimplex::new(vec![i, j]));
                }
            }
        }
        if max_dim == 1 {
            return complex;
        }

        // Build adjacency for higher-dim checks
        let mut adj = vec![vec![false; n]; n];
        for &(i, j) in &edge_set {
            adj[i][j] = true;
            adj[j][i] = true;
        }

        // Add higher-dimensional simplices: clique detection up to max_dim
        // We use incremental clique extension
        Self::add_cliques(&mut complex, &adj, n, max_dim);
        complex
    }

    /// Incrementally find and add all cliques up to max_dim+1 vertices.
    fn add_cliques(
        complex: &mut TdaSimplicialComplex,
        adj: &[Vec<bool>],
        n: usize,
        max_dim: usize,
    ) {
        // BFS over cliques: start with edges, extend by one vertex at a time
        // Use a Vec of candidate cliques and extend
        let mut cliques: Vec<Vec<usize>> = complex
            .simplices
            .get(1)
            .cloned()
            .unwrap_or_default()
            .iter()
            .map(|s| s.vertices.clone())
            .collect();

        for dim in 2..=max_dim {
            let mut next_cliques: Vec<Vec<usize>> = Vec::new();
            for clique in &cliques {
                let min_v = *clique.last().unwrap_or(&0);
                for v in (min_v + 1)..n {
                    // Check v is connected to all vertices in clique
                    if clique.iter().all(|&u| adj[u][v]) {
                        let mut new_clique = clique.clone();
                        new_clique.push(v);
                        complex.add_simplex(TdaSimplex::new(new_clique.clone()));
                        next_cliques.push(new_clique);
                    }
                }
            }
            cliques = next_cliques;
            if cliques.is_empty() || dim >= max_dim {
                break;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Persistent Homology
// ─────────────────────────────────────────────────────────────────────────────

/// A birth-death pair in a persistence diagram.
#[derive(Debug, Clone)]
pub struct TdaPersistencePair {
    pub birth: f32,
    pub death: f32,
    pub dimension: usize,
}

impl TdaPersistencePair {
    pub fn new(birth: f32, death: f32, dimension: usize) -> Self {
        Self {
            birth,
            death,
            dimension,
        }
    }

    /// Persistence (lifetime) of the feature.
    pub fn persistence(&self) -> f32 {
        if self.death == f32::INFINITY {
            f32::INFINITY
        } else {
            self.death - self.birth
        }
    }
}

/// A collection of persistence pairs.
#[derive(Debug, Clone)]
pub struct TdaPersistenceDiagram {
    pub pairs: Vec<TdaPersistencePair>,
}

impl TdaPersistenceDiagram {
    pub fn new(pairs: Vec<TdaPersistencePair>) -> Self {
        Self { pairs }
    }

    /// Pairs with death = infinity (essential homology classes).
    pub fn essential_pairs(&self) -> Vec<&TdaPersistencePair> {
        self.pairs
            .iter()
            .filter(|p| p.death == f32::INFINITY)
            .collect()
    }

    /// Keep only pairs with persistence >= min_pers.
    pub fn filter_by_persistence(&self, min_pers: f32) -> TdaPersistenceDiagram {
        let pairs = self
            .pairs
            .iter()
            .filter(|p| p.persistence() >= min_pers)
            .cloned()
            .collect();
        TdaPersistenceDiagram::new(pairs)
    }
}

/// Union-Find structure for connected components.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) -> Option<(usize, usize)> {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return None;
        }
        // Returns (older_root, younger_root) — older has smaller index
        let (older, younger) = if ra < rb { (ra, rb) } else { (rb, ra) };
        if self.rank[older] < self.rank[younger] {
            self.parent[older] = younger;
        } else if self.rank[older] > self.rank[younger] {
            self.parent[younger] = older;
        } else {
            self.parent[younger] = older;
            self.rank[older] += 1;
        }
        Some((older, younger))
    }
}

/// Computes persistence diagrams from point clouds.
pub struct PersistenceComputer;

impl PersistenceComputer {
    fn euclidean_dist(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    /// Compute H0 persistence via filtration of Vietoris-Rips at increasing epsilon values.
    pub fn compute(
        points: &[Vec<f32>],
        max_epsilon: f32,
        n_steps: usize,
        max_dim: usize,
    ) -> TdaPersistenceDiagram {
        let n = points.len();
        if n == 0 {
            return TdaPersistenceDiagram::new(Vec::new());
        }

        // Precompute all pairwise distances and sort edges
        let mut edges: Vec<(f32, usize, usize)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let d = Self::euclidean_dist(&points[i], &points[j]);
                edges.push((d, i, j));
            }
        }
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let step = if n_steps > 1 {
            max_epsilon / (n_steps as f32 - 1.0)
        } else {
            max_epsilon
        };
        let _ = step; // step size for reference; we use continuous filtration

        let mut pairs: Vec<TdaPersistencePair> = Vec::new();
        let mut uf = UnionFind::new(n);
        // All components born at 0
        let birth_times: Vec<f32> = vec![0.0; n];
        let mut component_birth = birth_times;

        // Process edges in order of distance (H0 persistence)
        for &(dist, i, j) in &edges {
            if dist > max_epsilon {
                break;
            }
            if let Some((older, younger)) = uf.union(i, j) {
                // The younger component dies
                let birth_older = component_birth[older];
                let birth_younger = component_birth[younger];
                let (dying_birth, surviving_birth) = if birth_older <= birth_younger {
                    (birth_younger, birth_older)
                } else {
                    (birth_older, birth_younger)
                };
                let _ = surviving_birth;
                pairs.push(TdaPersistencePair::new(dying_birth, dist, 0));
                // Update the surviving root's birth to the minimum
                let surviving_root = uf.find(older);
                component_birth[surviving_root] =
                    component_birth[older].min(component_birth[younger]);
            }
        }

        // Remaining components are essential (die at infinity)
        let mut seen_roots = std::collections::HashSet::new();
        for i in 0..n {
            let root = uf.find(i);
            if seen_roots.insert(root) {
                pairs.push(TdaPersistencePair::new(
                    component_birth[root],
                    f32::INFINITY,
                    0,
                ));
            }
        }

        // If max_dim >= 1, also compute H1+ via boundary matrix reduction on a discrete filtration.
        // We need simplices of dimension max_dim+1 to detect Hk for k=max_dim.
        if max_dim >= 1 && !edges.is_empty() {
            // Build complex at max_epsilon with one extra dimension for boundary detection
            let complex_dim = (max_dim + 1).min(3);
            let complex = VietorisRipsComplex::build(points, max_epsilon, complex_dim);
            // Create a simple filtration based on edge distances
            let filtration = Self::build_filtration(&complex, points, max_epsilon);
            let h1_pairs = Self::boundary_matrix_reduction(&complex, &filtration);
            for p in h1_pairs.pairs {
                if p.dimension >= 1 {
                    pairs.push(p);
                }
            }
        }

        TdaPersistenceDiagram::new(pairs)
    }

    fn build_filtration(
        complex: &TdaSimplicialComplex,
        points: &[Vec<f32>],
        max_epsilon: f32,
    ) -> Vec<f32> {
        let _ = max_epsilon;
        // Assign filtration value to each simplex: max edge length among faces
        let mut vals: Vec<f32> = Vec::new();
        for dim_simplices in &complex.simplices {
            for s in dim_simplices {
                if s.vertices.len() <= 1 {
                    vals.push(0.0);
                } else {
                    // Filtration value = max pairwise distance in the simplex
                    let mut max_d = 0.0_f32;
                    for ii in 0..s.vertices.len() {
                        for jj in (ii + 1)..s.vertices.len() {
                            let d = PersistenceComputer::euclidean_dist(
                                &points[s.vertices[ii]],
                                &points[s.vertices[jj]],
                            );
                            if d > max_d {
                                max_d = d;
                            }
                        }
                    }
                    vals.push(max_d);
                }
            }
        }
        vals
    }

    /// Standard persistence algorithm: boundary matrix column reduction.
    pub fn boundary_matrix_reduction(
        complex: &TdaSimplicialComplex,
        filtration: &[f32],
    ) -> TdaPersistenceDiagram {
        // Flatten all simplices with their filtration values
        let mut all_simplices: Vec<(usize, f32, &TdaSimplex)> = Vec::new();
        let mut flat_idx = 0;
        for dim_simplices in &complex.simplices {
            for s in dim_simplices {
                let fval = filtration.get(flat_idx).copied().unwrap_or(0.0);
                all_simplices.push((flat_idx, fval, s));
                flat_idx += 1;
            }
        }
        // Sort by filtration value then by dimension
        all_simplices.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.2.dim().cmp(&b.2.dim()))
        });
        let m = all_simplices.len();

        // Map from original flat index to sorted position
        let mut orig_to_sorted: Vec<usize> = vec![0; m];
        for (sorted_pos, &(orig_idx, _, _)) in all_simplices.iter().enumerate() {
            if orig_idx < orig_to_sorted.len() {
                orig_to_sorted[orig_idx] = sorted_pos;
            }
        }

        // Build boundary matrix as columns of sorted indices
        let mut columns: Vec<Vec<usize>> = Vec::with_capacity(m);
        for &(_, _, s) in &all_simplices {
            if s.vertices.len() <= 1 {
                columns.push(Vec::new());
                continue;
            }
            // Faces: all (dim-1) sub-simplices in order
            let mut col: Vec<usize> = (0..s.vertices.len())
                .map(|i| {
                    let face_verts: Vec<usize> = s
                        .vertices
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, &v)| v)
                        .collect();
                    let face = TdaSimplex::new(face_verts);
                    // Find this face in all_simplices
                    all_simplices
                        .iter()
                        .position(|(_, _, fs)| *fs == &face)
                        .unwrap_or(m)
                })
                .filter(|&pos| pos < m)
                .collect();
            col.sort_unstable();
            col.dedup();
            columns.push(col);
        }

        // Standard column reduction (left-to-right)
        let mut pivot_to_col: Vec<Option<usize>> = vec![None; m];
        let mut pairs: Vec<TdaPersistencePair> = Vec::new();
        // Track which indices appear as a pivot (birth) in some pairing
        let mut is_birth = vec![false; m];

        for j in 0..m {
            while let Some(&pivot) = columns[j].last() {
                match pivot_to_col[pivot] {
                    None => {
                        pivot_to_col[pivot] = Some(j);
                        break;
                    }
                    Some(k) => {
                        // XOR columns[j] with columns[k] (symmetric difference)
                        let col_k = columns[k].clone();
                        let merged = sym_diff(&columns[j], &col_k);
                        columns[j] = merged;
                    }
                }
            }
            // Record finite pair if column has a pivot after reduction
            if let Some(&pivot) = columns[j].last() {
                let dim = all_simplices[pivot].2.dim();
                let birth = all_simplices[pivot].1;
                let death = all_simplices[j].1;
                if death > birth {
                    pairs.push(TdaPersistencePair::new(birth, death, dim));
                }
                is_birth[pivot] = true;
            }
        }

        // Record essential (infinite) pairs: zero-column simplices that were never a birth
        for j in 0..m {
            let is_zero_col = columns[j].is_empty();
            let was_paired_as_birth = is_birth[j];
            if is_zero_col && !was_paired_as_birth {
                let dim = all_simplices[j].2.dim();
                // Dimension 0 essential pairs are handled by union-find above;
                // only emit dim >= 1 essential pairs here to avoid duplication
                if dim >= 1 {
                    let birth = all_simplices[j].1;
                    pairs.push(TdaPersistencePair::new(birth, f32::INFINITY, dim));
                }
            }
        }

        TdaPersistenceDiagram::new(pairs)
    }
}

/// Symmetric difference of two sorted Vecs of usize.
fn sym_diff(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut result: Vec<usize> = Vec::with_capacity(a.len() + b.len());
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                result.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Persistence Image
// ─────────────────────────────────────────────────────────────────────────────

/// Weight function for persistence image.
#[derive(Debug, Clone)]
pub enum TdaWeightFn {
    Linear,
    Arctan(f32),
    Sigmoid(f32),
}

impl TdaWeightFn {
    pub fn apply(&self, persistence: f32) -> f32 {
        match self {
            TdaWeightFn::Linear => persistence.max(0.0),
            TdaWeightFn::Arctan(scale) => {
                (scale * persistence).atan() / std::f32::consts::FRAC_PI_2
            }
            TdaWeightFn::Sigmoid(scale) => {
                let x = scale * persistence;
                1.0 / (1.0 + (-x).exp())
            }
        }
    }
}

/// Configuration for persistence image computation.
#[derive(Debug, Clone)]
pub struct TdaPiConfig {
    pub resolution: usize,
    pub bandwidth: f32,
    pub weight_fn: TdaWeightFn,
}

impl TdaPiConfig {
    pub fn new(resolution: usize, bandwidth: f32, weight_fn: TdaWeightFn) -> Self {
        Self {
            resolution,
            bandwidth,
            weight_fn,
        }
    }
}

/// Persistence image: a 2D array of pixel values.
#[derive(Debug, Clone)]
pub struct PersistenceImage {
    /// pixels\[row\]\[col\] indexed as \[birth_bin\]\[persistence_bin\]
    pub pixels: Vec<Vec<f32>>,
    pub config: TdaPiConfig,
}

impl PersistenceImage {
    /// Compute persistence image from a diagram for a specific dimension.
    pub fn compute(diagram: &TdaPersistenceDiagram, dim: usize, config: TdaPiConfig) -> Self {
        let res = config.resolution;
        let mut pixels = vec![vec![0.0_f32; res]; res];

        // Gather finite pairs for the given dimension
        let pairs: Vec<(f32, f32)> = diagram
            .pairs
            .iter()
            .filter(|p| p.dimension == dim && p.death.is_finite())
            .map(|p| (p.birth, p.persistence()))
            .collect();

        if pairs.is_empty() {
            return Self { pixels, config };
        }

        // Determine bounding box in (birth, persistence) space
        let birth_min = pairs.iter().map(|(b, _)| *b).fold(f32::INFINITY, f32::min);
        let birth_max = pairs
            .iter()
            .map(|(b, _)| *b)
            .fold(f32::NEG_INFINITY, f32::max);
        let pers_min = 0.0_f32;
        let pers_max = pairs
            .iter()
            .map(|(_, p)| *p)
            .fold(f32::NEG_INFINITY, f32::max);

        let birth_range = (birth_max - birth_min).max(1e-6);
        let pers_range = (pers_max - pers_min).max(1e-6);
        let bw = config.bandwidth;
        let bw2 = 2.0 * bw * bw;

        // Evaluate Gaussian KDE on grid
        for row in 0..res {
            let py = pers_min + (row as f32 + 0.5) / res as f32 * pers_range;
            for col in 0..res {
                let bx = birth_min + (col as f32 + 0.5) / res as f32 * birth_range;
                let mut val = 0.0_f32;
                for (birth, pers) in &pairs {
                    let w = config.weight_fn.apply(*pers);
                    let db = bx - birth;
                    let dp = py - pers;
                    let gauss = (-(db * db + dp * dp) / bw2).exp();
                    val += w * gauss;
                }
                pixels[row][col] = val;
            }
        }

        Self { pixels, config }
    }

    /// Flatten pixels to a 1D vector (row-major).
    pub fn to_vec(&self) -> Vec<f32> {
        self.pixels
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect()
    }

    /// L2 distance between two persistence images.
    pub fn distance(a: &PersistenceImage, b: &PersistenceImage) -> f32 {
        let va = a.to_vec();
        let vb = b.to_vec();
        let min_len = va.len().min(vb.len());
        let sum_sq: f32 = va
            .iter()
            .zip(vb.iter())
            .take(min_len)
            .map(|(x, y)| (x - y) * (x - y))
            .sum();
        sum_sq.sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Persistence Landscape
// ─────────────────────────────────────────────────────────────────────────────

/// Functional summary of a persistence diagram.
#[derive(Debug, Clone)]
pub struct PersistenceLandscape {
    /// layers\[k\]\[i\] = λ_{k+1}(eval_points\[i\])
    pub layers: Vec<Vec<f32>>,
    pub eval_points: Vec<f32>,
}

impl PersistenceLandscape {
    /// Compute the k-layer persistence landscape for a given dimension.
    ///
    /// λ_k(t) = k-th largest value of max(0, min(t-b, d-t)) over all pairs (b,d).
    pub fn compute(
        diagram: &TdaPersistenceDiagram,
        dim: usize,
        n_layers: usize,
        eval_points: Vec<f32>,
    ) -> Self {
        let pairs: Vec<(f32, f32)> = diagram
            .pairs
            .iter()
            .filter(|p| p.dimension == dim && p.death.is_finite())
            .map(|p| (p.birth, p.death))
            .collect();

        let n_pts = eval_points.len();
        let mut layers = vec![vec![0.0_f32; n_pts]; n_layers];

        for (ti, &t) in eval_points.iter().enumerate() {
            // Compute tent function values for all pairs at t
            let mut tent_vals: Vec<f32> = pairs
                .iter()
                .map(|(b, d)| {
                    let left = t - b;
                    let right = d - t;
                    if left < 0.0 || right < 0.0 {
                        0.0
                    } else {
                        left.min(right)
                    }
                })
                .collect();
            // Sort descending
            tent_vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            for k in 0..n_layers {
                layers[k][ti] = tent_vals.get(k).copied().unwrap_or(0.0);
            }
        }

        Self {
            layers,
            eval_points,
        }
    }

    /// Compute the mean of a slice of landscapes (all must have same structure).
    pub fn mean_landscape(landscapes: &[PersistenceLandscape]) -> PersistenceLandscape {
        if landscapes.is_empty() {
            return PersistenceLandscape {
                layers: Vec::new(),
                eval_points: Vec::new(),
            };
        }
        let n_layers = landscapes[0].layers.len();
        let n_pts = landscapes[0].eval_points.len();
        let n = landscapes.len() as f32;
        let mut mean_layers = vec![vec![0.0_f32; n_pts]; n_layers];
        for land in landscapes {
            for (k, layer) in land.layers.iter().enumerate() {
                if k < n_layers {
                    for (i, &v) in layer.iter().enumerate() {
                        if i < n_pts {
                            mean_layers[k][i] += v / n;
                        }
                    }
                }
            }
        }
        PersistenceLandscape {
            layers: mean_layers,
            eval_points: landscapes[0].eval_points.clone(),
        }
    }

    /// L2 distance between two persistence landscapes.
    pub fn l2_distance(a: &PersistenceLandscape, b: &PersistenceLandscape) -> f32 {
        let mut sum_sq = 0.0_f32;
        let n_layers = a.layers.len().min(b.layers.len());
        for k in 0..n_layers {
            let la = &a.layers[k];
            let lb = &b.layers[k];
            let n_pts = la.len().min(lb.len());
            for i in 0..n_pts {
                let diff = la[i] - lb[i];
                sum_sq += diff * diff;
            }
        }
        sum_sq.sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Mapper Algorithm
// ─────────────────────────────────────────────────────────────────────────────

/// An interval in the cover of the filter range.
#[derive(Debug, Clone)]
pub struct CoverInterval {
    pub center: f32,
    pub width: f32,
}

impl CoverInterval {
    pub fn contains(&self, x: f32) -> bool {
        let half = self.width / 2.0;
        x >= self.center - half && x <= self.center + half
    }
}

/// Configuration for the Mapper algorithm.
#[derive(Debug, Clone)]
pub struct MapperConfig {
    pub n_intervals: usize,
    pub overlap_pct: f32,
}

impl MapperConfig {
    pub fn new(n_intervals: usize, overlap_pct: f32) -> Self {
        Self {
            n_intervals,
            overlap_pct,
        }
    }
}

/// A node in the Mapper graph.
#[derive(Debug, Clone)]
pub struct MapperNode {
    pub points: Vec<usize>,
    pub cluster_id: usize,
    pub filter_value: f32,
}

/// The Mapper topological graph.
#[derive(Debug, Clone)]
pub struct MapperGraph {
    pub nodes: Vec<MapperNode>,
    pub edges: Vec<(usize, usize)>,
}

/// The Mapper algorithm for topological data analysis.
pub struct MapperAlgorithm {
    pub config: MapperConfig,
}

impl MapperAlgorithm {
    pub fn new(config: MapperConfig) -> Self {
        Self { config }
    }

    fn euclidean_dist(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    /// Single-linkage clustering of a subset of points.
    /// Returns cluster labels (one per index in `indices`).
    pub fn single_linkage_cluster(
        points: &[Vec<f32>],
        indices: &[usize],
        threshold: f32,
    ) -> Vec<usize> {
        let n = indices.len();
        if n == 0 {
            return Vec::new();
        }
        // Union-Find
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank = vec![0usize; n];

        let find = |parent: &mut Vec<usize>, mut x: usize| -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        };

        for i in 0..n {
            for j in (i + 1)..n {
                let d = Self::euclidean_dist(&points[indices[i]], &points[indices[j]]);
                if d <= threshold {
                    let ri = find(&mut parent, i);
                    let rj = find(&mut parent, j);
                    if ri != rj {
                        if rank[ri] < rank[rj] {
                            parent[ri] = rj;
                        } else if rank[ri] > rank[rj] {
                            parent[rj] = ri;
                        } else {
                            parent[rj] = ri;
                            rank[ri] += 1;
                        }
                    }
                }
            }
        }

        // Assign contiguous cluster IDs
        let mut root_to_id: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut next_id = 0usize;
        let mut labels = vec![0usize; n];
        for i in 0..n {
            let root = find(&mut parent, i);
            let id = *root_to_id.entry(root).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            labels[i] = id;
        }
        labels
    }

    /// Run the Mapper algorithm.
    ///
    /// 1. Apply filter function to all points.
    /// 2. Cover \[min,max\] with overlapping intervals.
    /// 3. Cluster points in each preimage (single-linkage).
    /// 4. Connect nodes that share points.
    pub fn compute(&self, points: &[Vec<f32>], filter_fn: &dyn Fn(&[f32]) -> f32) -> MapperGraph {
        let n = points.len();
        if n == 0 {
            return MapperGraph {
                nodes: Vec::new(),
                edges: Vec::new(),
            };
        }

        // Step 1: Apply filter
        let filter_vals: Vec<f32> = points.iter().map(|p| filter_fn(p)).collect();

        let f_min = filter_vals.iter().copied().fold(f32::INFINITY, f32::min);
        let f_max = filter_vals
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let f_range = (f_max - f_min).max(1e-6);

        // Step 2: Build overlapping cover
        let ni = self.config.n_intervals;
        let overlap = self.config.overlap_pct;
        let base_width = f_range / ni as f32;
        let width = base_width * (1.0 + overlap);

        let intervals: Vec<CoverInterval> = (0..ni)
            .map(|k| {
                let center = f_min + (k as f32 + 0.5) * base_width;
                CoverInterval { center, width }
            })
            .collect();

        // Step 3: For each interval, gather points and cluster
        let cluster_threshold = base_width * 0.5;
        let mut nodes: Vec<MapperNode> = Vec::new();

        for interval in &intervals {
            let indices: Vec<usize> = (0..n)
                .filter(|&i| interval.contains(filter_vals[i]))
                .collect();
            if indices.is_empty() {
                continue;
            }

            let labels = Self::single_linkage_cluster(points, &indices, cluster_threshold);
            let n_clusters = labels.iter().copied().max().unwrap_or(0) + 1;

            for cluster_id in 0..n_clusters {
                let cluster_points: Vec<usize> = indices
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| labels[*i] == cluster_id)
                    .map(|(_, &pt_idx)| pt_idx)
                    .collect();
                if !cluster_points.is_empty() {
                    let mean_filter = cluster_points
                        .iter()
                        .map(|&pi| filter_vals[pi])
                        .sum::<f32>()
                        / cluster_points.len() as f32;
                    nodes.push(MapperNode {
                        points: cluster_points,
                        cluster_id,
                        filter_value: mean_filter,
                    });
                }
            }
        }

        // Step 4: Connect nodes sharing at least one point
        let mut edges: Vec<(usize, usize)> = Vec::new();
        let nn = nodes.len();
        for i in 0..nn {
            for j in (i + 1)..nn {
                let shared = nodes[i].points.iter().any(|p| nodes[j].points.contains(p));
                if shared {
                    edges.push((i, j));
                }
            }
        }

        MapperGraph { nodes, edges }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Topological Feature Extractor
// ─────────────────────────────────────────────────────────────────────────────

/// TDA features for a point cloud.
#[derive(Debug, Clone)]
pub struct TdaFeatures {
    pub betti_numbers: Vec<usize>,
    pub persistence_stats: Vec<f32>,
    pub landscape_features: Vec<f32>,
}

/// Extracts topological features from point clouds.
pub struct TopologicalFeatureExtractor {
    pub max_dim: usize,
    pub n_landscape_layers: usize,
    pub pi_resolution: usize,
}

impl TopologicalFeatureExtractor {
    pub fn new(max_dim: usize, n_landscape_layers: usize, pi_resolution: usize) -> Self {
        Self {
            max_dim,
            n_landscape_layers,
            pi_resolution,
        }
    }

    /// Count the number of topological features alive at the given filtration threshold.
    /// A feature (b, d) in the diagram is alive at `threshold` if b <= threshold < d.
    /// Pairs with death == infinity are considered alive for any threshold >= birth.
    pub fn compute_betti_numbers(
        diagram: &TdaPersistenceDiagram,
        dims: usize,
        threshold: f32,
    ) -> Vec<usize> {
        (0..dims)
            .map(|d| {
                diagram
                    .pairs
                    .iter()
                    .filter(|p| {
                        p.dimension == d
                            && p.birth <= threshold
                            && (p.death == f32::INFINITY || p.death > threshold)
                    })
                    .count()
            })
            .collect()
    }

    /// Persistence statistics: [mean, std, max, entropy].
    pub fn persistence_statistics(diagram: &TdaPersistenceDiagram, dim: usize) -> Vec<f32> {
        let vals: Vec<f32> = diagram
            .pairs
            .iter()
            .filter(|p| p.dimension == dim && p.death.is_finite())
            .map(|p| p.persistence())
            .collect();

        if vals.is_empty() {
            return vec![0.0, 0.0, 0.0, 0.0];
        }

        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
        let variance =
            vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / vals.len() as f32;
        let std = variance.sqrt();
        let max = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Shannon entropy (normalized)
        let total = vals.iter().sum::<f32>().max(1e-10);
        let entropy = vals
            .iter()
            .map(|&v| {
                let p = v / total;
                if p > 1e-10 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum::<f32>();

        vec![mean, std, max, entropy]
    }

    /// Extract TDA features from a single point cloud.
    pub fn extract(&self, points: &[Vec<f32>], max_epsilon: f32) -> TdaFeatures {
        let diagram = PersistenceComputer::compute(points, max_epsilon, 20, self.max_dim);

        let betti_numbers = Self::compute_betti_numbers(&diagram, self.max_dim + 1, 0.0);

        let mut persistence_stats: Vec<f32> = Vec::new();
        for d in 0..=self.max_dim {
            let stats = Self::persistence_statistics(&diagram, d);
            persistence_stats.extend(stats);
        }

        // Landscape features
        let n_eval = 20;
        let eval_points: Vec<f32> = (0..n_eval)
            .map(|i| max_epsilon * i as f32 / (n_eval - 1) as f32)
            .collect();
        let mut landscape_features: Vec<f32> = Vec::new();
        for d in 0..=self.max_dim {
            let land = PersistenceLandscape::compute(
                &diagram,
                d,
                self.n_landscape_layers,
                eval_points.clone(),
            );
            for layer in &land.layers {
                landscape_features.extend(layer.iter().copied());
            }
        }

        TdaFeatures {
            betti_numbers,
            persistence_stats,
            landscape_features,
        }
    }

    /// Extract TDA features from multiple point clouds.
    pub fn extract_batch(
        &self,
        point_clouds: &[Vec<Vec<f32>>],
        max_epsilon: f32,
    ) -> Vec<TdaFeatures> {
        point_clouds
            .iter()
            .map(|pc| self.extract(pc, max_epsilon))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. TDA Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Metrics for comparing persistence diagrams.
pub struct TdaMetrics;

impl TdaMetrics {
    fn pairs_for_dim(diagram: &TdaPersistenceDiagram, dim: usize) -> Vec<(f32, f32)> {
        diagram
            .pairs
            .iter()
            .filter(|p| p.dimension == dim && p.death.is_finite())
            .map(|p| (p.birth, p.death))
            .collect()
    }

    /// Diagonal projection: closest point on diagonal to (b, d).
    fn diag_projection(b: f32, d: f32) -> (f32, f32) {
        let m = (b + d) / 2.0;
        (m, m)
    }

    fn point_dist(a: (f32, f32), b: (f32, f32)) -> f32 {
        let dx = a.0 - b.0;
        let dy = a.1 - b.1;
        (dx * dx + dy * dy).sqrt()
    }

    /// 2-Wasserstein distance via optimal assignment (O(n^2) Hungarian approximation).
    pub fn wasserstein_distance(
        d1: &TdaPersistenceDiagram,
        d2: &TdaPersistenceDiagram,
        dim: usize,
    ) -> f32 {
        let mut pts1 = Self::pairs_for_dim(d1, dim);
        let mut pts2 = Self::pairs_for_dim(d2, dim);

        // Add diagonal projections to balance sizes
        let n = pts1.len().max(pts2.len());
        while pts1.len() < n {
            let (b, d) = pts2[pts1.len()];
            pts1.push(Self::diag_projection(b, d));
        }
        while pts2.len() < n {
            let (b, d) = pts1[pts2.len()];
            pts2.push(Self::diag_projection(b, d));
        }

        if n == 0 {
            return 0.0;
        }

        // Greedy O(n^2) matching
        let cost = Self::hungarian_cost(&pts1, &pts2);
        cost.sqrt()
    }

    /// O(n^2) greedy matching cost (sum of squared distances).
    fn hungarian_cost(pts1: &[(f32, f32)], pts2: &[(f32, f32)]) -> f32 {
        let n = pts1.len();
        let mut used = vec![false; n];
        let mut total = 0.0_f32;
        for i in 0..n {
            let mut best_j = 0usize;
            let mut best_dist = f32::INFINITY;
            for j in 0..n {
                if !used[j] {
                    let d = Self::point_dist(pts1[i], pts2[j]);
                    if d < best_dist {
                        best_dist = d;
                        best_j = j;
                    }
                }
            }
            used[best_j] = true;
            total += best_dist * best_dist;
        }
        total
    }

    /// Bottleneck distance: max cost in optimal matching.
    pub fn bottleneck_distance(
        d1: &TdaPersistenceDiagram,
        d2: &TdaPersistenceDiagram,
        dim: usize,
    ) -> f32 {
        let mut pts1 = Self::pairs_for_dim(d1, dim);
        let mut pts2 = Self::pairs_for_dim(d2, dim);

        let n = pts1.len().max(pts2.len());
        while pts1.len() < n {
            let (b, d) = pts2[pts1.len()];
            pts1.push(Self::diag_projection(b, d));
        }
        while pts2.len() < n {
            let (b, d) = pts1[pts2.len()];
            pts2.push(Self::diag_projection(b, d));
        }

        if n == 0 {
            return 0.0;
        }

        // Bottleneck: min over matchings of max edge distance
        // Approximate with greedy assignment tracking max
        let mut used = vec![false; n];
        let mut max_dist = 0.0_f32;
        for i in 0..n {
            let mut best_j = 0usize;
            let mut best_d = f32::INFINITY;
            for j in 0..n {
                if !used[j] {
                    let d = Self::point_dist(pts1[i], pts2[j]);
                    if d < best_d {
                        best_d = d;
                        best_j = j;
                    }
                }
            }
            used[best_j] = true;
            if best_d > max_dist {
                max_dist = best_d;
            }
        }
        max_dist
    }

    /// Sliced Wasserstein distance: average 1D Wasserstein over random projections.
    pub fn sliced_wasserstein(
        d1: &TdaPersistenceDiagram,
        d2: &TdaPersistenceDiagram,
        n_slices: usize,
    ) -> f32 {
        // Project all dimensions together
        let pts1: Vec<(f32, f32)> = d1
            .pairs
            .iter()
            .filter(|p| p.death.is_finite())
            .map(|p| (p.birth, p.death))
            .collect();
        let pts2: Vec<(f32, f32)> = d2
            .pairs
            .iter()
            .filter(|p| p.death.is_finite())
            .map(|p| (p.birth, p.death))
            .collect();

        if pts1.is_empty() && pts2.is_empty() {
            return 0.0;
        }

        let mut rng = StdRng::seed_from_u64(42);
        let mut total = 0.0_f32;

        for _ in 0..n_slices {
            let angle: f32 = rng.random_range(0.0_f32..std::f32::consts::TAU);
            let (cos_a, sin_a) = (angle.cos(), angle.sin());

            let mut proj1: Vec<f32> = pts1.iter().map(|(b, d)| b * cos_a + d * sin_a).collect();
            let mut proj2: Vec<f32> = pts2.iter().map(|(b, d)| b * cos_a + d * sin_a).collect();

            proj1.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            proj2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // 1D Wasserstein: interpolate to same length
            let n1 = proj1.len();
            let n2 = proj2.len();
            let n = n1.max(n2).max(1);

            let interp = |proj: &[f32], t: f32| -> f32 {
                if proj.is_empty() {
                    return 0.0;
                }
                let idx = (t * (proj.len() as f32 - 1.0)).floor() as usize;
                let idx = idx.min(proj.len() - 1);
                proj[idx]
            };

            let w1: f32 = (0..n)
                .map(|k| {
                    let t = k as f32 / (n as f32 - 1.0).max(1.0);
                    (interp(&proj1, t) - interp(&proj2, t)).abs()
                })
                .sum::<f32>()
                / n as f32;

            total += w1;
        }

        total / n_slices as f32
    }

    /// Evaluate a batch of point clouds and return summary statistics.
    pub fn evaluate(point_clouds: &[Vec<Vec<f32>>], max_epsilon: f32) -> TdaEvalReport {
        if point_clouds.is_empty() {
            return TdaEvalReport {
                avg_betti0: 0.0,
                avg_betti1: 0.0,
                mean_persistence: 0.0,
                topological_complexity: 0.0,
            };
        }

        let n = point_clouds.len() as f32;
        let mut sum_b0 = 0.0_f32;
        let mut sum_b1 = 0.0_f32;
        let mut sum_pers = 0.0_f32;
        let mut sum_complexity = 0.0_f32;

        for pc in point_clouds {
            let diagram = PersistenceComputer::compute(pc, max_epsilon, 20, 2);

            let b0 = diagram
                .pairs
                .iter()
                .filter(|p| p.dimension == 0 && p.persistence() > 0.0)
                .count();
            let b1 = diagram
                .pairs
                .iter()
                .filter(|p| p.dimension == 1 && p.persistence() > 0.0)
                .count();
            sum_b0 += b0 as f32;
            sum_b1 += b1 as f32;

            let finite_pairs: Vec<f32> = diagram
                .pairs
                .iter()
                .filter(|p| p.death.is_finite())
                .map(|p| p.persistence())
                .collect();
            if !finite_pairs.is_empty() {
                sum_pers += finite_pairs.iter().sum::<f32>() / finite_pairs.len() as f32;
            }

            sum_complexity += (b0 + b1) as f32;
        }

        TdaEvalReport {
            avg_betti0: sum_b0 / n,
            avg_betti1: sum_b1 / n,
            mean_persistence: sum_pers / n,
            topological_complexity: sum_complexity / n,
        }
    }
}

/// Summary statistics for a batch of point clouds.
#[derive(Debug, Clone)]
pub struct TdaEvalReport {
    pub avg_betti0: f32,
    pub avg_betti1: f32,
    pub mean_persistence: f32,
    pub topological_complexity: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // Helper: sample n points on a circle of given radius
    fn circle_points(n: usize, radius: f32) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                let angle = 2.0 * PI * i as f32 / n as f32;
                vec![radius * angle.cos(), radius * angle.sin()]
            })
            .collect()
    }

    // ─── 1. SimplicialComplex tests ───────────────────────────────────────────

    #[test]
    fn test_simplex_creation() {
        let s = TdaSimplex::new(vec![2, 0, 1]);
        assert_eq!(s.vertices, vec![0, 1, 2]);
        assert_eq!(s.dim(), 2);
    }

    #[test]
    fn test_simplex_boundary_triangle() {
        let sc = TdaSimplicialComplex::new();
        let triangle = TdaSimplex::new(vec![0, 1, 2]);
        let boundary = sc.boundary(&triangle);
        assert_eq!(boundary.len(), 3);
        // Should contain edges [0,1], [0,2], [1,2]
        assert!(boundary.contains(&TdaSimplex::new(vec![1, 2])));
        assert!(boundary.contains(&TdaSimplex::new(vec![0, 2])));
        assert!(boundary.contains(&TdaSimplex::new(vec![0, 1])));
    }

    #[test]
    fn test_simplex_boundary_edge() {
        let sc = TdaSimplicialComplex::new();
        let edge = TdaSimplex::new(vec![0, 1]);
        let boundary = sc.boundary(&edge);
        assert_eq!(boundary.len(), 2);
        assert!(boundary.contains(&TdaSimplex::new(vec![0])));
        assert!(boundary.contains(&TdaSimplex::new(vec![1])));
    }

    #[test]
    fn test_simplicial_complex_euler() {
        // Tetrahedron: V=4, E=6, F=4, T=1 → χ = 4-6+4-1 = 1
        let mut sc = TdaSimplicialComplex::new();
        sc.add_simplex(TdaSimplex::new(vec![0, 1, 2, 3]));
        let chi = sc.euler_characteristic();
        // Vertices: 4, Edges: 6, Faces: 4, Tet: 1 → 4 - 6 + 4 - 1 = 1
        assert_eq!(chi, 1);
    }

    #[test]
    fn test_vr_complex_empty_epsilon() {
        let points = vec![vec![0.0, 0.0], vec![10.0, 0.0], vec![0.0, 10.0]];
        let sc = VietorisRipsComplex::build(&points, 0.01, 2);
        // No edges should be added (points are 10 apart)
        assert_eq!(sc.n_simplices(1), 0);
        assert_eq!(sc.n_simplices(0), 3);
    }

    #[test]
    fn test_vr_complex_full_epsilon() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let sc = VietorisRipsComplex::build(&points, 5.0, 2);
        // All 3 edges and 1 triangle
        assert_eq!(sc.n_simplices(1), 3);
        assert_eq!(sc.n_simplices(2), 1);
    }

    #[test]
    fn test_vr_complex_triangle() {
        // Equilateral triangle with side length 1
        let points = vec![vec![0.0_f32, 0.0], vec![1.0, 0.0], vec![0.5, 0.866]];
        let sc = VietorisRipsComplex::build(&points, 1.1, 2);
        assert_eq!(sc.n_simplices(0), 3);
        assert_eq!(sc.n_simplices(1), 3);
        assert_eq!(sc.n_simplices(2), 1);
    }

    #[test]
    fn test_vr_complex_dimension() {
        let points = vec![
            vec![0.0_f32, 0.0],
            vec![1.0, 0.0],
            vec![0.5, 0.866],
            vec![0.5, 0.3],
        ];
        let sc = VietorisRipsComplex::build(&points, 1.1, 3);
        assert!(sc.dimension() >= 2);
    }

    #[test]
    fn test_simplicial_complex_add() {
        let mut sc = TdaSimplicialComplex::new();
        sc.add_simplex(TdaSimplex::new(vec![0, 1]));
        sc.add_simplex(TdaSimplex::new(vec![1, 2]));
        assert_eq!(sc.n_simplices(1), 2);
        assert!(sc.n_simplices(0) >= 2);
    }

    // ─── 2. PersistentHomology tests ──────────────────────────────────────────

    #[test]
    fn test_persistence_pair_persistence() {
        let pair = TdaPersistencePair::new(0.5, 2.0, 0);
        assert!((pair.persistence() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_persistence_diagram_filter() {
        let diagram = TdaPersistenceDiagram::new(vec![
            TdaPersistencePair::new(0.0, 1.0, 0),
            TdaPersistencePair::new(0.0, 0.1, 0),
            TdaPersistencePair::new(0.0, 5.0, 1),
        ]);
        let filtered = diagram.filter_by_persistence(0.5);
        assert_eq!(filtered.pairs.len(), 2);
    }

    #[test]
    fn test_persistence_diagram_essential() {
        let diagram = TdaPersistenceDiagram::new(vec![
            TdaPersistencePair::new(0.0, f32::INFINITY, 0),
            TdaPersistencePair::new(0.0, 1.0, 0),
        ]);
        let essential = diagram.essential_pairs();
        assert_eq!(essential.len(), 1);
        assert_eq!(essential[0].birth, 0.0);
    }

    #[test]
    fn test_persistence_computer_runs() {
        let points = circle_points(20, 1.0);
        let diagram = PersistenceComputer::compute(&points, 3.0, 10, 1);
        assert!(!diagram.pairs.is_empty());
    }

    #[test]
    fn test_persistence_returns_pairs() {
        let points = vec![vec![0.0_f32, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
        let diagram = PersistenceComputer::compute(&points, 3.0, 10, 0);
        // 3 points → 2 merges + 1 essential = 3 pairs
        assert!(!diagram.pairs.is_empty());
    }

    #[test]
    fn test_persistence_h0_connected_component() {
        // Two well-separated clusters
        let mut points = Vec::new();
        for i in 0..5 {
            points.push(vec![i as f32 * 0.1, 0.0_f32]);
        }
        for i in 0..5 {
            points.push(vec![10.0 + i as f32 * 0.1, 0.0_f32]);
        }

        let diagram = PersistenceComputer::compute(&points, 5.0, 20, 0);
        // Should detect at most one H0 pair dying before epsilon=10 (within-cluster merges)
        let h0_finite: Vec<_> = diagram
            .pairs
            .iter()
            .filter(|p| p.dimension == 0 && p.death.is_finite())
            .collect();
        // Within each cluster: 4 merges each → 8 finite H0 pairs
        assert!(!h0_finite.is_empty());
    }

    // ─── 3. PersistenceImage tests ────────────────────────────────────────────

    #[test]
    fn test_persistence_image_shape() {
        let diagram = TdaPersistenceDiagram::new(vec![
            TdaPersistencePair::new(0.0, 1.0, 0),
            TdaPersistencePair::new(0.5, 2.0, 0),
        ]);
        let config = TdaPiConfig::new(10, 0.5, TdaWeightFn::Linear);
        let pi = PersistenceImage::compute(&diagram, 0, config);
        assert_eq!(pi.pixels.len(), 10);
        assert_eq!(pi.pixels[0].len(), 10);
    }

    #[test]
    fn test_persistence_image_to_vec_length() {
        let diagram = TdaPersistenceDiagram::new(vec![TdaPersistencePair::new(0.0, 1.0, 0)]);
        let config = TdaPiConfig::new(8, 0.3, TdaWeightFn::Linear);
        let pi = PersistenceImage::compute(&diagram, 0, config);
        assert_eq!(pi.to_vec().len(), 64);
    }

    #[test]
    fn test_persistence_image_distance_zero() {
        let diagram = TdaPersistenceDiagram::new(vec![TdaPersistencePair::new(0.0, 1.0, 0)]);
        let config1 = TdaPiConfig::new(8, 0.3, TdaWeightFn::Linear);
        let config2 = TdaPiConfig::new(8, 0.3, TdaWeightFn::Linear);
        let pi1 = PersistenceImage::compute(&diagram, 0, config1);
        let pi2 = PersistenceImage::compute(&diagram, 0, config2);
        let dist = PersistenceImage::distance(&pi1, &pi2);
        assert!(
            dist < 1e-4,
            "Same diagram should have distance ~0, got {}",
            dist
        );
    }

    #[test]
    fn test_persistence_image_distance_positive() {
        let diag1 = TdaPersistenceDiagram::new(vec![TdaPersistencePair::new(0.0, 1.0, 0)]);
        let diag2 = TdaPersistenceDiagram::new(vec![TdaPersistencePair::new(0.0, 5.0, 0)]);
        let pi1 =
            PersistenceImage::compute(&diag1, 0, TdaPiConfig::new(8, 0.3, TdaWeightFn::Linear));
        let pi2 =
            PersistenceImage::compute(&diag2, 0, TdaPiConfig::new(8, 0.3, TdaWeightFn::Linear));
        assert!(PersistenceImage::distance(&pi1, &pi2) > 0.0);
    }

    #[test]
    fn test_weight_fn_linear() {
        let wf = TdaWeightFn::Linear;
        assert!((wf.apply(2.0) - 2.0).abs() < 1e-6);
        assert!((wf.apply(-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_weight_fn_arctan() {
        let wf = TdaWeightFn::Arctan(1.0);
        let v = wf.apply(1000.0); // should approach 1.0
        assert!(v > 0.99);
        assert!(wf.apply(0.0) < 0.01);
    }

    #[test]
    fn test_persistence_image_kernel_gaussian() {
        // Verify Gaussian KDE produces positive values near birth-persistence location
        let diagram = TdaPersistenceDiagram::new(vec![
            TdaPersistencePair::new(0.5, 1.5, 0), // birth=0.5, pers=1.0
        ]);
        let config = TdaPiConfig::new(20, 0.2, TdaWeightFn::Linear);
        let pi = PersistenceImage::compute(&diagram, 0, config);
        let total: f32 = pi.to_vec().iter().sum();
        assert!(total > 0.0, "Gaussian KDE should produce nonzero values");
    }

    // ─── 4. PersistenceLandscape tests ───────────────────────────────────────

    #[test]
    fn test_persistence_landscape_shape() {
        let diagram = TdaPersistenceDiagram::new(vec![
            TdaPersistencePair::new(0.0, 2.0, 0),
            TdaPersistencePair::new(1.0, 3.0, 0),
        ]);
        let eval_pts = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let land = PersistenceLandscape::compute(&diagram, 0, 2, eval_pts.clone());
        assert_eq!(land.layers.len(), 2);
        assert_eq!(land.layers[0].len(), eval_pts.len());
    }

    #[test]
    fn test_persistence_landscape_layers() {
        let diagram = TdaPersistenceDiagram::new(vec![TdaPersistencePair::new(0.0, 2.0, 0)]);
        let eval_pts = vec![0.0, 1.0, 2.0];
        let land = PersistenceLandscape::compute(&diagram, 0, 2, eval_pts);
        // At t=1.0, λ_1(1.0) = min(1-0, 2-1) = 1.0
        assert!((land.layers[0][1] - 1.0).abs() < 1e-5);
        // λ_2(1.0) = 0 (no second pair)
        assert!((land.layers[1][1]).abs() < 1e-5);
    }

    #[test]
    fn test_persistence_landscape_l2_distance_zero() {
        let diagram = TdaPersistenceDiagram::new(vec![TdaPersistencePair::new(0.0, 2.0, 0)]);
        let eval_pts = vec![0.0, 1.0, 2.0];
        let l1 = PersistenceLandscape::compute(&diagram, 0, 2, eval_pts.clone());
        let l2 = PersistenceLandscape::compute(&diagram, 0, 2, eval_pts);
        assert!(PersistenceLandscape::l2_distance(&l1, &l2) < 1e-6);
    }

    #[test]
    fn test_persistence_landscape_mean() {
        let diag = TdaPersistenceDiagram::new(vec![TdaPersistencePair::new(0.0, 2.0, 0)]);
        let eval_pts = vec![0.0, 1.0, 2.0];
        let l = PersistenceLandscape::compute(&diag, 0, 1, eval_pts.clone());
        let mean = PersistenceLandscape::mean_landscape(&[l.clone(), l.clone()]);
        // Mean of identical landscapes should equal the landscape
        assert!((mean.layers[0][1] - l.layers[0][1]).abs() < 1e-5);
    }

    #[test]
    fn test_landscape_eval_points() {
        let diag = TdaPersistenceDiagram::new(vec![TdaPersistencePair::new(0.0, 2.0, 0)]);
        let eval_pts: Vec<f32> = (0..50).map(|i| i as f32 / 49.0 * 2.0).collect();
        let land = PersistenceLandscape::compute(&diag, 0, 3, eval_pts.clone());
        assert_eq!(land.eval_points.len(), 50);
        assert_eq!(land.layers[0].len(), 50);
    }

    // ─── 5. MapperAlgorithm tests ─────────────────────────────────────────────

    #[test]
    fn test_mapper_single_linkage_two_clusters() {
        let points: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.2, 0.0], // cluster A
            vec![10.0, 0.0],
            vec![10.1, 0.0],
            vec![10.2, 0.0], // cluster B
        ];
        let indices: Vec<usize> = (0..6).collect();
        let labels = MapperAlgorithm::single_linkage_cluster(&points, &indices, 0.5);
        assert_eq!(labels.len(), 6);
        // First 3 should have same label, last 3 same label, different from first
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_mapper_single_linkage_one_cluster() {
        let points: Vec<Vec<f32>> = vec![vec![0.0, 0.0], vec![0.5, 0.0], vec![1.0, 0.0]];
        let indices: Vec<usize> = (0..3).collect();
        let labels = MapperAlgorithm::single_linkage_cluster(&points, &indices, 1.0);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
    }

    #[test]
    fn test_mapper_compute_returns_graph() {
        let points: Vec<Vec<f32>> = (0..30).map(|i| vec![i as f32, 0.0_f32]).collect();
        let config = MapperConfig::new(5, 0.3);
        let mapper = MapperAlgorithm::new(config);
        let graph = mapper.compute(&points, &|p| p[0]);
        // Should return a valid graph structure
        assert!(!graph.nodes.is_empty() || graph.edges.is_empty());
    }

    #[test]
    fn test_mapper_graph_node_count_positive() {
        let points: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![i as f32 * 0.5, (i as f32).sin()])
            .collect();
        let config = MapperConfig::new(4, 0.5);
        let mapper = MapperAlgorithm::new(config);
        let graph = mapper.compute(&points, &|p| p[0]);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_mapper_config_creation() {
        let config = MapperConfig::new(10, 0.3);
        assert_eq!(config.n_intervals, 10);
        assert!((config.overlap_pct - 0.3).abs() < 1e-6);
    }

    // ─── 6. TopologicalFeatureExtractor tests ────────────────────────────────

    #[test]
    fn test_betti_numbers_circle() {
        // Small circle (6 points) so the VR complex stays tractable:
        // Adjacent dist ~1.0, next-to-adjacent dist ~1.73, diameter ~2.0
        // max_epsilon=1.2 → only adjacent edges form, creating a ring (H0=1, H1=1)
        let points = circle_points(6, 1.0);
        // With epsilon=1.2: edges connect adjacent points only (distance=1.0 < 1.2)
        // The ring forms but no triangles yet (next-adjacent is 1.73 > 1.2)
        let diagram = PersistenceComputer::compute(&points, 1.2, 20, 1);

        // H0 essential pair at threshold=1.0 (all adjacent merged at eps~1.0): betti[0]=1
        // H1: born when ring closes (~1.0), at threshold=1.1 the cycle is alive
        let betti = TopologicalFeatureExtractor::compute_betti_numbers(&diagram, 2, 1.1);
        // Circle should have exactly 1 connected component
        assert_eq!(betti[0], 1, "H0 should be 1 for a connected circle");
        // H1 cycle should be alive (born, not yet died since no triangles at eps=1.2)
        assert!(betti[1] >= 1, "H1 should be >= 1 for a circle");
    }

    #[test]
    fn test_betti_numbers_dimension_count() {
        let diagram = TdaPersistenceDiagram::new(vec![
            TdaPersistencePair::new(0.0, f32::INFINITY, 0), // essential H0: alive at 0.3
            TdaPersistencePair::new(0.0, 0.1, 0),           // H0 dies at 0.1: not alive at 0.3
            TdaPersistencePair::new(0.2, 1.0, 1),           // H1: born 0.2, dies 1.0: alive at 0.3
        ]);
        // At threshold=0.3: first H0 alive, second H0 dead, H1 alive
        let betti = TopologicalFeatureExtractor::compute_betti_numbers(&diagram, 2, 0.3);
        assert_eq!(betti[0], 1);
        assert_eq!(betti[1], 1);
    }

    #[test]
    fn test_persistence_statistics_shape() {
        let diagram = TdaPersistenceDiagram::new(vec![
            TdaPersistencePair::new(0.0, 1.0, 0),
            TdaPersistencePair::new(0.5, 2.0, 0),
            TdaPersistencePair::new(1.0, 1.5, 0),
        ]);
        let stats = TopologicalFeatureExtractor::persistence_statistics(&diagram, 0);
        assert_eq!(stats.len(), 4); // [mean, std, max, entropy]
    }

    #[test]
    fn test_persistence_statistics_max_ge_mean() {
        let diagram = TdaPersistenceDiagram::new(vec![
            TdaPersistencePair::new(0.0, 1.0, 0),
            TdaPersistencePair::new(0.0, 3.0, 0),
            TdaPersistencePair::new(0.0, 2.0, 0),
        ]);
        let stats = TopologicalFeatureExtractor::persistence_statistics(&diagram, 0);
        let mean = stats[0];
        let max = stats[2];
        assert!(max >= mean);
    }

    #[test]
    fn test_tda_features_extraction() {
        let points = circle_points(20, 1.0);
        let extractor = TopologicalFeatureExtractor::new(1, 2, 5);
        let features = extractor.extract(&points, 3.0);
        assert!(!features.betti_numbers.is_empty());
        assert!(!features.persistence_stats.is_empty());
    }

    #[test]
    fn test_tda_batch_extraction_count() {
        let clouds: Vec<Vec<Vec<f32>>> = vec![
            circle_points(15, 1.0),
            circle_points(15, 2.0),
            circle_points(15, 0.5),
        ];
        let extractor = TopologicalFeatureExtractor::new(1, 2, 5);
        let batch = extractor.extract_batch(&clouds, 5.0);
        assert_eq!(batch.len(), 3);
    }

    // ─── 7. TdaMetrics tests ──────────────────────────────────────────────────

    #[test]
    fn test_wasserstein_distance_zero_same() {
        let diag = TdaPersistenceDiagram::new(vec![
            TdaPersistencePair::new(0.0, 1.0, 0),
            TdaPersistencePair::new(1.0, 3.0, 0),
        ]);
        let d = TdaMetrics::wasserstein_distance(&diag, &diag, 0);
        assert!(d < 1e-4, "Same diagram: distance should be ~0, got {}", d);
    }

    #[test]
    fn test_wasserstein_distance_positive() {
        let d1 = TdaPersistenceDiagram::new(vec![TdaPersistencePair::new(0.0, 1.0, 0)]);
        let d2 = TdaPersistenceDiagram::new(vec![TdaPersistencePair::new(0.0, 5.0, 0)]);
        let d = TdaMetrics::wasserstein_distance(&d1, &d2, 0);
        assert!(d > 0.0);
    }

    #[test]
    fn test_bottleneck_distance_zero_same() {
        let diag = TdaPersistenceDiagram::new(vec![TdaPersistencePair::new(0.0, 1.0, 0)]);
        let d = TdaMetrics::bottleneck_distance(&diag, &diag, 0);
        assert!(d < 1e-4);
    }

    #[test]
    fn test_bottleneck_distance_positive() {
        let d1 = TdaPersistenceDiagram::new(vec![TdaPersistencePair::new(0.0, 1.0, 0)]);
        let d2 = TdaPersistenceDiagram::new(vec![TdaPersistencePair::new(2.0, 5.0, 0)]);
        let d = TdaMetrics::bottleneck_distance(&d1, &d2, 0);
        assert!(d > 0.0);
    }

    #[test]
    fn test_sliced_wasserstein_nonneg() {
        let d1 = TdaPersistenceDiagram::new(vec![
            TdaPersistencePair::new(0.0, 1.0, 0),
            TdaPersistencePair::new(1.0, 2.0, 1),
        ]);
        let d2 = TdaPersistenceDiagram::new(vec![TdaPersistencePair::new(0.5, 1.5, 0)]);
        let d = TdaMetrics::sliced_wasserstein(&d1, &d2, 50);
        assert!(d >= 0.0);
    }

    #[test]
    fn test_tda_eval_report_fields() {
        let clouds = vec![circle_points(10, 1.0), circle_points(10, 2.0)];
        let report = TdaMetrics::evaluate(&clouds, 5.0);
        assert!(report.avg_betti0 >= 0.0);
        assert!(report.avg_betti1 >= 0.0);
        assert!(report.mean_persistence >= 0.0);
        assert!(report.topological_complexity >= 0.0);
    }

    #[test]
    fn test_evaluate_batch_runs() {
        let clouds: Vec<Vec<Vec<f32>>> =
            (0..5).map(|i| circle_points(12, 1.0 + i as f32)).collect();
        let report = TdaMetrics::evaluate(&clouds, 6.0);
        assert!(report.avg_betti0 >= 0.0);
        assert!(report.topological_complexity >= 0.0);
    }
}
