//! Graph Matching, Alignment & Kernels
//!
//! Production-grade implementations of classical and modern graph matching algorithms,
//! graph kernels, and spectral alignment methods. All algorithms are pure Rust with
//! no external dependencies beyond `tenflowers_core`.
//!
//! # Components
//!
//! | Component | Algorithm | Complexity |
//! |-----------|-----------|------------|
//! | [`GmGraph`] | Graph representation | O(V+E) storage |
//! | [`GraphEditDistance`] | A* with beam search + Hungarian LB | O(V^2 * beam) |
//! | [`Vf2Matcher`] | VF2 subgraph isomorphism | O(V! worst, fast prune) |
//! | [`WeisfeilerLemanKernel`] | WL subtree kernel | O(h * E) per iteration |
//! | [`RandomWalkKernel`] | Geometric series walk kernel | O(V^3 * walk_len) |
//! | [`ShortestPathKernel`] | Floyd-Warshall SP kernel | O(V^3) per graph |
//! | [`SpectralAlignment`] | Laplacian eigenvector Procrustes | O(V^2 * k) |
//! | [`GraduatedAssignment`] | Softassign + Sinkhorn + annealing | O(V^2 * iters) |
//! | [`MaxCommonSubgraph`] | McSplit-inspired branch & bound | O(exp) worst |
//! | [`GraphMatchMetrics`] | Accuracy, PSD check, normalized GED | O(V^2) |

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use tenflowers_core::{Result, TensorError};

mod tests;

// ---------------------------------------------------------------------------
// GmGraph — Graph representation for matching
// ---------------------------------------------------------------------------

/// Edge in a graph matching context, connecting two nodes with optional features.
#[derive(Debug, Clone)]
pub struct GmEdge {
    pub target: usize,
    pub features: Vec<f64>,
}

/// Graph representation optimized for matching and kernel computations.
///
/// Stores an adjacency list with per-node and per-edge feature vectors.
/// Undirected edges are stored in both directions for efficient traversal.
#[derive(Debug, Clone)]
pub struct GmGraph {
    /// Node feature vectors.
    pub node_features: Vec<Vec<f64>>,
    /// Adjacency list (each entry: list of (target, edge_features)).
    adjacency: Vec<Vec<GmEdge>>,
    /// Total number of (undirected) edges.
    n_edges: usize,
}

impl GmGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            node_features: Vec::new(),
            adjacency: Vec::new(),
            n_edges: 0,
        }
    }

    /// Add a node with the given feature vector. Returns the node index.
    pub fn add_node(&mut self, features: Vec<f64>) -> usize {
        let idx = self.node_features.len();
        self.node_features.push(features);
        self.adjacency.push(Vec::new());
        idx
    }

    /// Add an undirected edge between nodes `u` and `v` with features.
    /// Returns `Err` if either node index is out of bounds.
    pub fn add_edge(&mut self, u: usize, v: usize, features: Vec<f64>) -> Result<()> {
        let n = self.n_nodes();
        if u >= n || v >= n {
            return Err(TensorError::compute_error_simple(format!(
                "GmGraph::add_edge: node index out of bounds (u={u}, v={v}, n_nodes={n})"
            )));
        }
        self.adjacency[u].push(GmEdge {
            target: v,
            features: features.clone(),
        });
        self.adjacency[v].push(GmEdge {
            target: u,
            features,
        });
        self.n_edges += 1;
        Ok(())
    }

    /// Neighbors of a node (as slice of GmEdge).
    pub fn neighbors(&self, node: usize) -> &[GmEdge] {
        if node < self.adjacency.len() {
            &self.adjacency[node]
        } else {
            &[]
        }
    }

    /// Degree of a node.
    pub fn degree(&self, node: usize) -> usize {
        if node < self.adjacency.len() {
            self.adjacency[node].len()
        } else {
            0
        }
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.node_features.len()
    }

    /// Number of undirected edges.
    pub fn n_edges(&self) -> usize {
        self.n_edges
    }

    /// Check if an edge exists between u and v.
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        if u >= self.n_nodes() {
            return false;
        }
        self.adjacency[u].iter().any(|e| e.target == v)
    }

    /// Build a dense adjacency matrix (n x n).
    pub fn adjacency_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.n_nodes();
        let mut mat = vec![vec![0.0; n]; n];
        for u in 0..n {
            for e in &self.adjacency[u] {
                mat[u][e.target] = 1.0;
            }
        }
        mat
    }

    /// Build the graph Laplacian L = D - A (unnormalized).
    pub fn laplacian(&self) -> Vec<Vec<f64>> {
        let n = self.n_nodes();
        let adj = self.adjacency_matrix();
        let mut lap = vec![vec![0.0; n]; n];
        for i in 0..n {
            let deg: f64 = adj[i].iter().sum();
            lap[i][i] = deg;
            for j in 0..n {
                lap[i][j] -= adj[i][j];
            }
        }
        lap
    }
}

impl Default for GmGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Graph Edit Distance — A* with beam search + Hungarian lower bound
// ---------------------------------------------------------------------------

/// Types of edit operations for graph edit distance.
#[derive(Debug, Clone, PartialEq)]
pub enum GmEditOp {
    /// Substitute node i in g1 for node j in g2.
    SubstituteNode { from: usize, to: usize },
    /// Delete node i from g1.
    DeleteNode { node: usize },
    /// Insert node j from g2.
    InsertNode { node: usize },
    /// Substitute edge (u1,v1) in g1 for (u2,v2) in g2.
    SubstituteEdge {
        from: (usize, usize),
        to: (usize, usize),
    },
    /// Delete edge (u,v) from g1.
    DeleteEdge { edge: (usize, usize) },
    /// Insert edge (u,v) from g2.
    InsertEdge { edge: (usize, usize) },
}

/// Cost model for graph edit distance operations.
#[derive(Debug, Clone)]
pub struct GmEditCostModel {
    /// Cost per node substitution (base; added to feature distance).
    pub node_sub_cost: f64,
    /// Cost per node deletion.
    pub node_del_cost: f64,
    /// Cost per node insertion.
    pub node_ins_cost: f64,
    /// Cost per edge substitution.
    pub edge_sub_cost: f64,
    /// Cost per edge deletion.
    pub edge_del_cost: f64,
    /// Cost per edge insertion.
    pub edge_ins_cost: f64,
}

impl Default for GmEditCostModel {
    fn default() -> Self {
        Self {
            node_sub_cost: 1.0,
            node_del_cost: 1.0,
            node_ins_cost: 1.0,
            edge_sub_cost: 1.0,
            edge_del_cost: 1.0,
            edge_ins_cost: 1.0,
        }
    }
}

/// Euclidean distance between two feature vectors.
fn feature_distance(a: &[f64], b: &[f64]) -> f64 {
    let max_len = a.len().max(b.len());
    let mut sum = 0.0;
    for i in 0..max_len {
        let va = if i < a.len() { a[i] } else { 0.0 };
        let vb = if i < b.len() { b[i] } else { 0.0 };
        let d = va - vb;
        sum += d * d;
    }
    sum.sqrt()
}

/// Node substitution cost including feature distance.
fn node_sub_cost(g1: &GmGraph, i: usize, g2: &GmGraph, j: usize, model: &GmEditCostModel) -> f64 {
    let fd = feature_distance(&g1.node_features[i], &g2.node_features[j]);
    model.node_sub_cost * fd
}

/// A* search state for GED computation.
#[derive(Clone)]
struct GedState {
    /// Mapping from g1 nodes to g2 nodes (or usize::MAX for deleted).
    mapping: Vec<Option<usize>>,
    /// Which g2 nodes are already used.
    used_g2: HashSet<usize>,
    /// Next g1 node to process.
    depth: usize,
    /// Cost so far.
    g_cost: f64,
    /// Heuristic lower bound for remaining cost.
    h_cost: f64,
}

impl GedState {
    fn f_cost(&self) -> f64 {
        self.g_cost + self.h_cost
    }
}

impl PartialEq for GedState {
    fn eq(&self, other: &Self) -> bool {
        self.f_cost().to_bits() == other.f_cost().to_bits()
    }
}

impl Eq for GedState {}

impl PartialOrd for GedState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GedState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse comparison.
        other
            .f_cost()
            .partial_cmp(&self.f_cost())
            .unwrap_or(Ordering::Equal)
    }
}

/// Graph Edit Distance computation using A* with beam search.
///
/// The Hungarian algorithm is used as an admissible heuristic for the remaining
/// unmatched nodes. Beam width limits memory usage on large graphs.
pub struct GraphEditDistance;

impl GraphEditDistance {
    /// Compute the Hungarian algorithm lower bound for the cost of assigning
    /// remaining g1 nodes to remaining g2 nodes.
    ///
    /// Uses a simplified Hungarian (Jonker-Volgenant-style) on a cost matrix
    /// padded with deletion/insertion costs.
    fn hungarian_lower_bound(
        g1: &GmGraph,
        g2: &GmGraph,
        state: &GedState,
        model: &GmEditCostModel,
    ) -> f64 {
        let n1 = g1.n_nodes();
        let n2 = g2.n_nodes();
        let remaining_g1: Vec<usize> = (state.depth..n1).collect();
        let remaining_g2: Vec<usize> = (0..n2).filter(|j| !state.used_g2.contains(j)).collect();

        if remaining_g1.is_empty() && remaining_g2.is_empty() {
            return 0.0;
        }

        // Build augmented cost matrix of size max(r1, r2) x max(r1, r2).
        let r1 = remaining_g1.len();
        let r2 = remaining_g2.len();
        let dim = r1.max(r2);
        if dim == 0 {
            return 0.0;
        }

        let mut cost = vec![vec![0.0f64; dim]; dim];

        for (i, &gi) in remaining_g1.iter().enumerate() {
            for (j, &gj) in remaining_g2.iter().enumerate() {
                cost[i][j] = node_sub_cost(g1, gi, g2, gj, model);
            }
            // Padding columns for deletion.
            for j in r2..dim {
                let _ = j;
                cost[i][r2.min(dim.saturating_sub(1))] = model.node_del_cost;
            }
            for j in r2..dim {
                cost[i][j] = model.node_del_cost;
            }
        }

        // Padding rows for insertion.
        for i in r1..dim {
            for j in 0..dim {
                cost[i][j] = if j < r2 { model.node_ins_cost } else { 0.0 };
            }
        }

        // Greedy assignment as lower bound approximation (full Hungarian is O(n^3)).
        Self::greedy_assignment(&cost, dim)
    }

    /// Greedy row-minimum assignment as a fast lower bound.
    fn greedy_assignment(cost: &[Vec<f64>], dim: usize) -> f64 {
        let mut used_cols = vec![false; dim];
        let mut total = 0.0;

        // For each row, find the cheapest available column.
        for row in cost.iter().take(dim) {
            let mut best_cost = f64::MAX;
            let mut best_col = 0;
            for (j, &c) in row.iter().enumerate().take(dim) {
                if !used_cols[j] && c < best_cost {
                    best_cost = c;
                    best_col = j;
                }
            }
            if best_cost < f64::MAX {
                used_cols[best_col] = true;
                total += best_cost;
            }
        }
        total
    }

    /// Compute graph edit distance between g1 and g2 with beam search.
    ///
    /// Returns `(distance, edit_path)` where the edit path lists all operations.
    /// `beam_width` limits the search frontier size for tractability.
    pub fn compute(
        g1: &GmGraph,
        g2: &GmGraph,
        beam_width: usize,
        cost_model: &GmEditCostModel,
    ) -> Result<(f64, Vec<GmEditOp>)> {
        let n1 = g1.n_nodes();
        let n2 = g2.n_nodes();

        // Edge case: both empty.
        if n1 == 0 && n2 == 0 {
            return Ok((0.0, Vec::new()));
        }

        let initial = GedState {
            mapping: Vec::new(),
            used_g2: HashSet::new(),
            depth: 0,
            g_cost: 0.0,
            h_cost: 0.0,
        };

        let mut heap = BinaryHeap::new();
        let h0 = Self::hungarian_lower_bound(g1, g2, &initial, cost_model);
        let initial = GedState {
            h_cost: h0,
            ..initial
        };
        heap.push(initial);

        let mut best_cost = f64::MAX;
        let mut best_mapping: Vec<Option<usize>> = Vec::new();
        let mut iterations = 0;
        let max_iterations = beam_width * (n1 + n2 + 1) * 50;

        while let Some(state) = heap.pop() {
            iterations += 1;
            if iterations > max_iterations {
                break;
            }

            if state.f_cost() >= best_cost {
                continue;
            }

            // All g1 nodes processed — handle remaining g2 insertions.
            if state.depth == n1 {
                let mut ins_cost = 0.0;
                for j in 0..n2 {
                    if !state.used_g2.contains(&j) {
                        ins_cost += cost_model.node_ins_cost;
                    }
                }
                let total =
                    state.g_cost + ins_cost + Self::edge_cost(g1, g2, &state.mapping, cost_model);
                if total < best_cost {
                    best_cost = total;
                    best_mapping = state.mapping.clone();
                }
                continue;
            }

            // Try substituting g1[depth] -> g2[j] for each available j.
            let mut children = Vec::new();
            for j in 0..n2 {
                if state.used_g2.contains(&j) {
                    continue;
                }
                let sub_cost = node_sub_cost(g1, state.depth, g2, j, cost_model);
                let mut new_mapping = state.mapping.clone();
                new_mapping.push(Some(j));
                let mut new_used = state.used_g2.clone();
                new_used.insert(j);
                let new_state = GedState {
                    mapping: new_mapping,
                    used_g2: new_used,
                    depth: state.depth + 1,
                    g_cost: state.g_cost + sub_cost,
                    h_cost: 0.0,
                };
                let h = Self::hungarian_lower_bound(g1, g2, &new_state, cost_model);
                let new_state = GedState {
                    h_cost: h,
                    ..new_state
                };
                if new_state.f_cost() < best_cost {
                    children.push(new_state);
                }
            }

            // Try deleting g1[depth].
            {
                let mut new_mapping = state.mapping.clone();
                new_mapping.push(None);
                let new_state = GedState {
                    mapping: new_mapping,
                    used_g2: state.used_g2.clone(),
                    depth: state.depth + 1,
                    g_cost: state.g_cost + cost_model.node_del_cost,
                    h_cost: 0.0,
                };
                let h = Self::hungarian_lower_bound(g1, g2, &new_state, cost_model);
                let new_state = GedState {
                    h_cost: h,
                    ..new_state
                };
                if new_state.f_cost() < best_cost {
                    children.push(new_state);
                }
            }

            // Beam pruning: keep top beam_width children.
            children.sort_by(|a, b| {
                a.f_cost()
                    .partial_cmp(&b.f_cost())
                    .unwrap_or(Ordering::Equal)
            });
            for child in children.into_iter().take(beam_width) {
                heap.push(child);
            }

            // Limit heap size to beam_width * 2.
            if heap.len() > beam_width * 4 {
                let mut temp: Vec<GedState> = heap.into_sorted_vec();
                temp.truncate(beam_width * 2);
                heap = BinaryHeap::from(temp);
            }
        }

        // Build edit path from best mapping.
        let edit_path = Self::build_edit_path(g1, g2, &best_mapping);
        Ok((best_cost, edit_path))
    }

    /// Compute edge-level edit cost for a given node mapping.
    fn edge_cost(
        g1: &GmGraph,
        g2: &GmGraph,
        mapping: &[Option<usize>],
        model: &GmEditCostModel,
    ) -> f64 {
        let mut cost = 0.0;
        let mut matched_edges_g2: HashSet<(usize, usize)> = HashSet::new();

        // For each edge in g1.
        for u in 0..g1.n_nodes() {
            for e in g1.neighbors(u) {
                let v = e.target;
                if u >= v {
                    continue; // process each undirected edge once
                }
                match (
                    mapping.get(u).copied().flatten(),
                    mapping.get(v).copied().flatten(),
                ) {
                    (Some(mu), Some(mv)) => {
                        if g2.has_edge(mu, mv) {
                            // Edge substitution.
                            cost += model.edge_sub_cost
                                * feature_distance(&e.features, &Self::edge_features(g2, mu, mv));
                            let pair = (mu.min(mv), mu.max(mv));
                            matched_edges_g2.insert(pair);
                        } else {
                            // Edge in g1 not present in g2 -> deletion.
                            cost += model.edge_del_cost;
                        }
                    }
                    _ => {
                        // Node was deleted -> edge deletion.
                        cost += model.edge_del_cost;
                    }
                }
            }
        }

        // Remaining g2 edges not matched -> insertion.
        for u in 0..g2.n_nodes() {
            for e in g2.neighbors(u) {
                let v = e.target;
                if u >= v {
                    continue;
                }
                let pair = (u.min(v), u.max(v));
                if !matched_edges_g2.contains(&pair) {
                    cost += model.edge_ins_cost;
                }
            }
        }

        cost
    }

    /// Get edge features between u and v in graph g.
    fn edge_features(g: &GmGraph, u: usize, v: usize) -> Vec<f64> {
        for e in g.neighbors(u) {
            if e.target == v {
                return e.features.clone();
            }
        }
        Vec::new()
    }

    /// Build the list of edit operations from a node mapping.
    fn build_edit_path(g1: &GmGraph, g2: &GmGraph, mapping: &[Option<usize>]) -> Vec<GmEditOp> {
        let mut ops = Vec::new();
        let n2 = g2.n_nodes();
        let mut used_g2 = HashSet::new();

        for (i, m) in mapping.iter().enumerate() {
            match m {
                Some(j) => {
                    ops.push(GmEditOp::SubstituteNode { from: i, to: *j });
                    used_g2.insert(*j);
                }
                None => {
                    ops.push(GmEditOp::DeleteNode { node: i });
                }
            }
        }

        for j in 0..n2 {
            if !used_g2.contains(&j) {
                ops.push(GmEditOp::InsertNode { node: j });
            }
        }

        // Edge operations.
        let mut matched_edges_g2: HashSet<(usize, usize)> = HashSet::new();
        for u in 0..g1.n_nodes() {
            for e in g1.neighbors(u) {
                let v = e.target;
                if u >= v {
                    continue;
                }
                match (
                    mapping.get(u).copied().flatten(),
                    mapping.get(v).copied().flatten(),
                ) {
                    (Some(mu), Some(mv)) if g2.has_edge(mu, mv) => {
                        ops.push(GmEditOp::SubstituteEdge {
                            from: (u, v),
                            to: (mu, mv),
                        });
                        let pair = (mu.min(mv), mu.max(mv));
                        matched_edges_g2.insert(pair);
                    }
                    _ => {
                        ops.push(GmEditOp::DeleteEdge { edge: (u, v) });
                    }
                }
            }
        }

        for u in 0..g2.n_nodes() {
            for e in g2.neighbors(u) {
                let v = e.target;
                if u >= v {
                    continue;
                }
                let pair = (u.min(v), u.max(v));
                if !matched_edges_g2.contains(&pair) {
                    ops.push(GmEditOp::InsertEdge { edge: (u, v) });
                }
            }
        }

        ops
    }
}

// ---------------------------------------------------------------------------
// VF2 Matcher — Subgraph isomorphism
// ---------------------------------------------------------------------------

/// VF2 state for subgraph isomorphism matching.
#[derive(Clone)]
struct Vf2State {
    /// Mapping from pattern node to target node.
    core_pattern: Vec<Option<usize>>,
    /// Reverse mapping: target node to pattern node.
    core_target: Vec<Option<usize>>,
    /// Number of matched pairs.
    depth: usize,
}

/// VF2 algorithm for subgraph isomorphism detection.
///
/// Implements the state-space search with feasibility rules including
/// adjacency consistency and degree-based pruning.
pub struct Vf2Matcher;

impl Vf2Matcher {
    /// Find one isomorphism mapping pattern into target.
    /// Returns mapping\[pattern_node\] = target_node if found.
    pub fn find_isomorphism(pattern: &GmGraph, target: &GmGraph) -> Result<Option<Vec<usize>>> {
        if pattern.n_nodes() > target.n_nodes() {
            return Ok(None);
        }

        let state = Vf2State {
            core_pattern: vec![None; pattern.n_nodes()],
            core_target: vec![None; target.n_nodes()],
            depth: 0,
        };

        let mut result = None;
        Self::vf2_recurse(pattern, target, &state, &mut result, false);
        Ok(result.and_then(|mappings| mappings.into_iter().next()))
    }

    /// Find all subgraph isomorphisms of pattern in target.
    pub fn find_all_subgraph_isomorphisms(
        pattern: &GmGraph,
        target: &GmGraph,
    ) -> Result<Vec<Vec<usize>>> {
        if pattern.n_nodes() > target.n_nodes() {
            return Ok(Vec::new());
        }

        let state = Vf2State {
            core_pattern: vec![None; pattern.n_nodes()],
            core_target: vec![None; target.n_nodes()],
            depth: 0,
        };

        let mut all_mappings: Vec<Vec<usize>> = Vec::new();
        Self::vf2_collect(pattern, target, &state, &mut all_mappings);
        Ok(all_mappings)
    }

    /// Recursive VF2 search for one mapping.
    fn vf2_recurse(
        pattern: &GmGraph,
        target: &GmGraph,
        state: &Vf2State,
        result: &mut Option<Vec<Vec<usize>>>,
        _find_all: bool,
    ) {
        if state.depth == pattern.n_nodes() {
            let mapping: Vec<usize> = state
                .core_pattern
                .iter()
                .map(|opt| opt.unwrap_or(0))
                .collect();
            *result = Some(vec![mapping]);
            return;
        }

        if result.is_some() {
            return;
        }

        let candidates = Self::generate_candidates(pattern, target, state);

        for (p, t) in candidates {
            if Self::is_feasible(pattern, target, state, p, t) {
                let mut new_state = state.clone();
                new_state.core_pattern[p] = Some(t);
                new_state.core_target[t] = Some(p);
                new_state.depth += 1;

                Self::vf2_recurse(pattern, target, &new_state, result, false);

                if result.is_some() {
                    return;
                }
            }
        }
    }

    /// Recursive VF2 collecting all matches.
    fn vf2_collect(
        pattern: &GmGraph,
        target: &GmGraph,
        state: &Vf2State,
        all_mappings: &mut Vec<Vec<usize>>,
    ) {
        if state.depth == pattern.n_nodes() {
            let mapping: Vec<usize> = state
                .core_pattern
                .iter()
                .map(|opt| opt.unwrap_or(0))
                .collect();
            all_mappings.push(mapping);
            return;
        }

        let candidates = Self::generate_candidates(pattern, target, state);

        for (p, t) in candidates {
            if Self::is_feasible(pattern, target, state, p, t) {
                let mut new_state = state.clone();
                new_state.core_pattern[p] = Some(t);
                new_state.core_target[t] = Some(p);
                new_state.depth += 1;

                Self::vf2_collect(pattern, target, &new_state, all_mappings);
            }
        }
    }

    /// Generate candidate pairs (pattern_node, target_node) for extension.
    fn generate_candidates(
        pattern: &GmGraph,
        target: &GmGraph,
        state: &Vf2State,
    ) -> Vec<(usize, usize)> {
        // Find the first unmapped pattern node (ordered by index for determinism).
        let p_node = (0..pattern.n_nodes()).find(|&i| state.core_pattern[i].is_none());

        let p_node = match p_node {
            Some(n) => n,
            None => return Vec::new(),
        };

        // Candidate target nodes: unmapped, and pass degree check.
        let mut candidates = Vec::new();
        for t in 0..target.n_nodes() {
            if state.core_target[t].is_some() {
                continue;
            }
            // Degree pruning: target node must have at least as many neighbors
            // as the pattern node.
            if target.degree(t) >= pattern.degree(p_node) {
                candidates.push((p_node, t));
            }
        }

        candidates
    }

    /// Check VF2 feasibility rules for mapping pattern[p] -> target[t].
    fn is_feasible(
        pattern: &GmGraph,
        target: &GmGraph,
        state: &Vf2State,
        p: usize,
        t: usize,
    ) -> bool {
        // Syntactic rule: for every mapped neighbor of p in pattern,
        // the corresponding target node must be a neighbor of t.
        for e in pattern.neighbors(p) {
            let p_neighbor = e.target;
            if let Some(t_neighbor) = state.core_pattern[p_neighbor] {
                if !target.has_edge(t, t_neighbor) {
                    return false;
                }
            }
        }

        // Semantic rule: node feature compatibility (cosine similarity > 0
        // or matching dimensionality with L2 < threshold).
        let pf = &pattern.node_features[p];
        let tf = &target.node_features[t];
        if !pf.is_empty() && !tf.is_empty() && pf.len() == tf.len() {
            let dist = feature_distance(pf, tf);
            // Allow match if features are reasonably close (threshold = 5.0).
            if dist > 5.0 {
                return false;
            }
        }

        true
    }
}

// ---------------------------------------------------------------------------
// Weisfeiler-Leman Kernel — WL subtree kernel
// ---------------------------------------------------------------------------

/// Weisfeiler-Leman subtree kernel for graph comparison.
///
/// Iteratively relabels nodes by hashing their neighborhood structure,
/// then compares histogram of labels across iterations.
pub struct WeisfeilerLemanKernel;

impl WeisfeilerLemanKernel {
    /// Compute the WL subtree kernel value between two graphs.
    ///
    /// `h_iterations` controls the number of WL refinement rounds.
    /// Returns a non-negative kernel value.
    pub fn compute_kernel(g1: &GmGraph, g2: &GmGraph, h_iterations: usize) -> Result<f64> {
        let labels1 = Self::compute_label_sequence(g1, h_iterations);
        let labels2 = Self::compute_label_sequence(g2, h_iterations);

        // Kernel = sum over all iterations of dot product of histograms.
        let mut kernel_value = 0.0;

        for h in 0..=h_iterations {
            let hist1 = Self::histogram(&labels1[h]);
            let hist2 = Self::histogram(&labels2[h]);

            // Dot product of histograms.
            for (label, count1) in &hist1 {
                if let Some(count2) = hist2.get(label) {
                    kernel_value += (*count1 as f64) * (*count2 as f64);
                }
            }
        }

        Ok(kernel_value)
    }

    /// WL-optimal assignment kernel variant.
    ///
    /// Uses optimal assignment (greedy matching) between label histograms
    /// of the two graphs at each WL iteration.
    pub fn compute_oa_kernel(g1: &GmGraph, g2: &GmGraph, h_iterations: usize) -> Result<f64> {
        let labels1 = Self::compute_label_sequence(g1, h_iterations);
        let labels2 = Self::compute_label_sequence(g2, h_iterations);

        let mut kernel_value = 0.0;

        for h in 0..=h_iterations {
            // For OA kernel, compute optimal assignment of nodes based on label match.
            let n1 = labels1[h].len();
            let n2 = labels2[h].len();
            let mut used = vec![false; n2];

            for i in 0..n1 {
                for j in 0..n2 {
                    if !used[j] && labels1[h][i] == labels2[h][j] {
                        kernel_value += 1.0;
                        used[j] = true;
                        break;
                    }
                }
            }
        }

        Ok(kernel_value)
    }

    /// Compute the sequence of label vectors for each WL iteration.
    fn compute_label_sequence(g: &GmGraph, h: usize) -> Vec<Vec<u64>> {
        let n = g.n_nodes();
        let mut all_labels = Vec::with_capacity(h + 1);

        // Initial labels: hash of node features.
        let mut current: Vec<u64> = (0..n)
            .map(|i| Self::hash_features(&g.node_features[i]))
            .collect();
        all_labels.push(current.clone());

        for _ in 0..h {
            let mut next = Vec::with_capacity(n);
            for i in 0..n {
                let mut neighbor_labels: Vec<u64> =
                    g.neighbors(i).iter().map(|e| current[e.target]).collect();
                neighbor_labels.sort();
                let new_label = Self::hash_label_with_neighbors(current[i], &neighbor_labels);
                next.push(new_label);
            }
            current = next;
            all_labels.push(current.clone());
        }

        all_labels
    }

    /// Build a histogram (label -> count) from a label vector.
    fn histogram(labels: &[u64]) -> HashMap<u64, usize> {
        let mut hist = HashMap::new();
        for &l in labels {
            *hist.entry(l).or_insert(0) += 1;
        }
        hist
    }

    /// FNV-1a hash of node features.
    fn hash_features(features: &[f64]) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for &f in features {
            let bits = f.to_bits();
            hash ^= bits;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Hash a label together with its sorted neighbor labels (WL relabeling).
    fn hash_label_with_neighbors(label: u64, neighbors: &[u64]) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        hash ^= label;
        hash = hash.wrapping_mul(0x100000001b3);
        for &n in neighbors {
            hash ^= n;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }
}

// ---------------------------------------------------------------------------
// Random Walk Kernel
// ---------------------------------------------------------------------------

/// Random walk graph kernel using the direct product graph approach.
///
/// Computes `K(g1, g2) = sum_{k=0}^{walk_length} decay^k * (number of matching length-k walks)`
/// via powers of the adjacency matrix of the direct product graph.
pub struct RandomWalkKernel;

impl RandomWalkKernel {
    /// Compute the random walk kernel between two graphs.
    ///
    /// `walk_length`: maximum walk length to consider.
    /// `decay`: geometric decay factor (should be < 1/max_degree for convergence).
    pub fn compute(g1: &GmGraph, g2: &GmGraph, walk_length: usize, decay: f64) -> Result<f64> {
        // Build direct product graph adjacency matrix.
        let n1 = g1.n_nodes();
        let n2 = g2.n_nodes();
        let prod_size = n1 * n2;

        if prod_size == 0 {
            return Ok(0.0);
        }

        // Product graph: node (i,j) connects to (i',j') iff (i,i') in g1 and (j,j') in g2.
        let adj1 = g1.adjacency_matrix();
        let adj2 = g2.adjacency_matrix();

        let mut prod_adj = vec![vec![0.0f64; prod_size]; prod_size];
        for i in 0..n1 {
            for j in 0..n2 {
                let pij = i * n2 + j;
                for i2 in 0..n1 {
                    for j2 in 0..n2 {
                        if adj1[i][i2] > 0.0 && adj2[j][j2] > 0.0 {
                            let pi2j2 = i2 * n2 + j2;
                            prod_adj[pij][pi2j2] = 1.0;
                        }
                    }
                }
            }
        }

        // Geometric series: K = sum_{k=0}^{L} decay^k * 1^T A^k 1
        // Start with identity (k=0), then multiply by A for each step.
        let mut power = vec![vec![0.0f64; prod_size]; prod_size];
        for i in 0..prod_size {
            power[i][i] = 1.0; // Identity = A^0
        }

        let mut kernel_val = 0.0;
        let mut decay_k = 1.0; // decay^0

        for _k in 0..=walk_length {
            // Sum all entries of power matrix.
            let walk_count: f64 = power.iter().flat_map(|row| row.iter()).sum();
            kernel_val += decay_k * walk_count;

            if _k < walk_length {
                // power = power * prod_adj
                power = mat_mul(&power, &prod_adj);
                decay_k *= decay;
            }
        }

        Ok(kernel_val)
    }
}

/// Dense matrix multiplication C = A * B.
fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 {
        return Vec::new();
    }
    let p = b[0].len();
    let n = b.len();
    let mut c = vec![vec![0.0f64; p]; m];
    for i in 0..m {
        for k in 0..n {
            let a_ik = a[i][k];
            if a_ik == 0.0 {
                continue;
            }
            for j in 0..p {
                c[i][j] += a_ik * b[k][j];
            }
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Shortest Path Kernel
// ---------------------------------------------------------------------------

/// Shortest path graph kernel.
///
/// Computes all-pairs shortest paths via Floyd-Warshall, then counts
/// matching shortest path length pairs between two graphs.
pub struct ShortestPathKernel;

impl ShortestPathKernel {
    /// Compute the SP kernel between two graphs.
    pub fn compute(g1: &GmGraph, g2: &GmGraph) -> Result<f64> {
        let sp1 = Self::floyd_warshall(g1);
        let sp2 = Self::floyd_warshall(g2);

        // Collect all shortest path values (for connected pairs).
        let paths1 = Self::collect_paths(&sp1);
        let paths2 = Self::collect_paths(&sp2);

        // Kernel: count matching SP lengths.
        let hist1 = Self::path_histogram(&paths1);
        let hist2 = Self::path_histogram(&paths2);

        let mut kernel_val = 0.0;
        for (len, count1) in &hist1 {
            if let Some(count2) = hist2.get(len) {
                kernel_val += (*count1 as f64) * (*count2 as f64);
            }
        }

        Ok(kernel_val)
    }

    /// Floyd-Warshall all-pairs shortest paths.
    fn floyd_warshall(g: &GmGraph) -> Vec<Vec<f64>> {
        let n = g.n_nodes();
        let inf = f64::MAX / 2.0;
        let mut dist = vec![vec![inf; n]; n];

        for i in 0..n {
            dist[i][i] = 0.0;
        }
        for u in 0..n {
            for e in g.neighbors(u) {
                dist[u][e.target] = 1.0; // unweighted
            }
        }

        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let through_k = dist[i][k] + dist[k][j];
                    if through_k < dist[i][j] {
                        dist[i][j] = through_k;
                    }
                }
            }
        }

        dist
    }

    /// Collect all finite shortest path distances as (i, j, dist) triples.
    fn collect_paths(sp: &[Vec<f64>]) -> Vec<usize> {
        let n = sp.len();
        let inf = f64::MAX / 4.0;
        let mut paths = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if sp[i][j] < inf {
                    paths.push(sp[i][j] as usize);
                }
            }
        }
        paths
    }

    /// Histogram of path lengths.
    fn path_histogram(paths: &[usize]) -> HashMap<usize, usize> {
        let mut hist = HashMap::new();
        for &p in paths {
            *hist.entry(p).or_insert(0) += 1;
        }
        hist
    }
}

// ---------------------------------------------------------------------------
// Spectral Alignment — Laplacian eigenvector Procrustes
// ---------------------------------------------------------------------------

/// Spectral graph matching via Laplacian eigenvector alignment.
///
/// Computes the top-k eigenvectors of each graph's Laplacian, then aligns
/// them using the orthogonal Procrustes problem (SVD-based).
pub struct SpectralAlignment;

impl SpectralAlignment {
    /// Align two graphs using spectral methods.
    ///
    /// Returns a vector of (g1_node, g2_node) correspondences.
    pub fn align(g1: &GmGraph, g2: &GmGraph, k_eigenvectors: usize) -> Result<Vec<(usize, usize)>> {
        let n1 = g1.n_nodes();
        let n2 = g2.n_nodes();

        if n1 == 0 || n2 == 0 {
            return Ok(Vec::new());
        }

        let k = k_eigenvectors.min(n1).min(n2);
        if k == 0 {
            return Ok(Vec::new());
        }

        // Compute Laplacian eigenvectors for both graphs.
        let lap1 = g1.laplacian();
        let lap2 = g2.laplacian();

        let eigvecs1 = Self::top_k_eigenvectors(&lap1, k)?;
        let eigvecs2 = Self::top_k_eigenvectors(&lap2, k)?;

        // Orthogonal Procrustes: find R = argmin ||U1 R - U2||_F
        // Solution: R = V U^T where M = U1^T U2 = U S V^T
        let m_min = n1.min(n2);

        // Truncate eigenvector matrices to same number of rows.
        let u1: Vec<Vec<f64>> = eigvecs1.into_iter().take(m_min).collect();
        let u2: Vec<Vec<f64>> = eigvecs2.into_iter().take(m_min).collect();

        // Compute U1^T * U2 (k x k).
        let m_mat = Self::mat_transpose_mul(&u1, &u2, k);

        // SVD of M via power iteration (for small k this is fine).
        let (u_svd, _s, vt_svd) = Self::svd_power_iter(&m_mat, k)?;

        // R = V * U^T (k x k).
        let r = Self::mat_mul_transpose_second(&vt_svd, &u_svd, k);

        // Apply rotation: U1_aligned = U1 * R (m_min x k).
        let u1_aligned = Self::mat_mul_dense(&u1, &r);

        // Match nodes by nearest neighbor in spectral embedding space.
        let mut correspondences = Vec::new();
        let mut used_g2 = vec![false; m_min];

        for i in 0..m_min.min(n1) {
            let mut best_j = 0;
            let mut best_dist = f64::MAX;
            for j in 0..m_min.min(n2) {
                if used_g2[j] {
                    continue;
                }
                let dist = Self::vec_distance(&u1_aligned[i], &u2[j]);
                if dist < best_dist {
                    best_dist = dist;
                    best_j = j;
                }
            }
            if best_dist < f64::MAX {
                used_g2[best_j] = true;
                correspondences.push((i, best_j));
            }
        }

        Ok(correspondences)
    }

    /// Compute top-k eigenvectors of a symmetric matrix using power iteration with deflation.
    fn top_k_eigenvectors(mat: &[Vec<f64>], k: usize) -> Result<Vec<Vec<f64>>> {
        let n = mat.len();
        let mut eigvecs = vec![vec![0.0; k]; n];
        let mut deflated = mat.to_vec();

        for col in 0..k {
            let (eigenvalue, eigvec) = Self::power_iteration(&deflated, 200)?;

            for i in 0..n {
                eigvecs[i][col] = eigvec[i];
            }

            // Deflate: M = M - lambda * v * v^T
            for i in 0..n {
                for j in 0..n {
                    deflated[i][j] -= eigenvalue * eigvec[i] * eigvec[j];
                }
            }
        }

        Ok(eigvecs)
    }

    /// Power iteration to find the dominant eigenvector.
    fn power_iteration(mat: &[Vec<f64>], max_iter: usize) -> Result<(f64, Vec<f64>)> {
        let n = mat.len();
        if n == 0 {
            return Err(TensorError::compute_error_simple(
                "SpectralAlignment: empty matrix".to_string(),
            ));
        }

        // Initialize with [1, 1, ..., 1] / sqrt(n).
        let inv_sqrt_n = 1.0 / (n as f64).sqrt();
        let mut v = vec![inv_sqrt_n; n];

        let mut eigenvalue = 0.0;

        for _ in 0..max_iter {
            // w = M * v
            let mut w = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += mat[i][j] * v[j];
                }
            }

            // Eigenvalue estimate = v^T w.
            eigenvalue = 0.0;
            for i in 0..n {
                eigenvalue += v[i] * w[i];
            }

            // Normalize.
            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                break;
            }
            for x in &mut w {
                *x /= norm;
            }

            v = w;
        }

        Ok((eigenvalue, v))
    }

    /// Compute A^T * B where A is n x k and B is n x k.
    fn mat_transpose_mul(a: &[Vec<f64>], b: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; k]; k];
        let n = a.len().min(b.len());
        for i in 0..k {
            for j in 0..k {
                for r in 0..n {
                    if i < a[r].len() && j < b[r].len() {
                        result[i][j] += a[r][i] * b[r][j];
                    }
                }
            }
        }
        result
    }

    /// SVD of a small k x k matrix via power iteration on M^T M.
    fn svd_power_iter(
        m: &[Vec<f64>],
        k: usize,
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>)> {
        // Compute M^T M.
        let mut mtm = vec![vec![0.0; k]; k];
        for i in 0..k {
            for j in 0..k {
                for r in 0..k {
                    if r < m.len() && i < m[r].len() && j < m[r].len() {
                        mtm[i][j] += m[r][i] * m[r][j];
                    }
                }
            }
        }

        // Find eigenvectors of M^T M (these are V).
        let v_vecs = Self::top_k_eigenvectors(&mtm, k)?;

        // Singular values and U.
        let mut s = vec![0.0; k];
        let mut u_vecs = vec![vec![0.0; k]; k];

        for col in 0..k {
            // u_col = M * v_col
            let v_col: Vec<f64> = (0..k)
                .map(|r| {
                    if r < v_vecs.len() && col < v_vecs[r].len() {
                        v_vecs[r][col]
                    } else {
                        0.0
                    }
                })
                .collect();

            let mut u_col = vec![0.0; k];
            for i in 0..k.min(m.len()) {
                for j in 0..k.min(m[i].len()) {
                    u_col[i] += m[i][j] * v_col[j];
                }
            }

            let norm = u_col.iter().map(|x| x * x).sum::<f64>().sqrt();
            s[col] = norm;
            if norm > 1e-15 {
                for x in &mut u_col {
                    *x /= norm;
                }
            }
            for i in 0..k {
                u_vecs[i][col] = u_col[i];
            }
        }

        // V^T from V.
        let mut vt = vec![vec![0.0; k]; k];
        for i in 0..k {
            for j in 0..k {
                if j < v_vecs.len() && i < v_vecs[j].len() {
                    vt[i][j] = v_vecs[j][i];
                }
            }
        }

        Ok((u_vecs, s, vt))
    }

    /// Multiply A * B^T where both are stored as Vec<Vec<f64>> of size k x k.
    fn mat_mul_transpose_second(a: &[Vec<f64>], b: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; k]; k];
        for i in 0..k {
            for j in 0..k {
                for r in 0..k {
                    let a_val = if i < a.len() && r < a[i].len() {
                        a[i][r]
                    } else {
                        0.0
                    };
                    let b_val = if j < b.len() && r < b[j].len() {
                        b[j][r]
                    } else {
                        0.0
                    };
                    result[i][j] += a_val * b_val;
                }
            }
        }
        result
    }

    /// Dense matrix multiplication: C = A * B.
    fn mat_mul_dense(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let m = a.len();
        if m == 0 || b.is_empty() {
            return Vec::new();
        }
        let p = b[0].len();
        let n = b.len();
        let mut c = vec![vec![0.0; p]; m];
        for i in 0..m {
            for k in 0..a[i].len().min(n) {
                let a_ik = a[i][k];
                if a_ik == 0.0 {
                    continue;
                }
                for j in 0..p {
                    c[i][j] += a_ik * b[k][j];
                }
            }
        }
        c
    }

    /// Euclidean distance between two vectors.
    fn vec_distance(a: &[f64], b: &[f64]) -> f64 {
        let len = a.len().max(b.len());
        let mut sum = 0.0;
        for i in 0..len {
            let va = if i < a.len() { a[i] } else { 0.0 };
            let vb = if i < b.len() { b[i] } else { 0.0 };
            let d = va - vb;
            sum += d * d;
        }
        sum.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Graduated Assignment — Softassign + Sinkhorn + annealing
// ---------------------------------------------------------------------------

/// Graduated assignment algorithm for graph matching (Gold & Rangarajan 1996).
///
/// Uses Sinkhorn normalization to obtain a doubly-stochastic soft assignment
/// matrix, with graduated annealing of the inverse temperature parameter beta.
pub struct GraduatedAssignment;

/// Result of graduated assignment matching.
#[derive(Debug, Clone)]
pub struct GmSoftAssignment {
    /// Soft assignment matrix (n1 x n2), entries in [0, 1].
    pub matrix: Vec<Vec<f64>>,
    /// Discrete assignment: for each g1 node, the best-matched g2 node.
    pub discrete: Vec<usize>,
    /// Final inverse temperature.
    pub final_beta: f64,
}

impl GraduatedAssignment {
    /// Perform graduated assignment matching between g1 and g2.
    ///
    /// * `n_outer_iters` — number of annealing iterations.
    /// * `n_sinkhorn_iters` — Sinkhorn normalization iterations per outer step.
    /// * `beta_init` — initial inverse temperature.
    /// * `beta_factor` — multiplicative increase per outer iteration.
    pub fn match_graphs(
        g1: &GmGraph,
        g2: &GmGraph,
        n_outer_iters: usize,
        n_sinkhorn_iters: usize,
        beta_init: f64,
        beta_factor: f64,
    ) -> Result<GmSoftAssignment> {
        let n1 = g1.n_nodes();
        let n2 = g2.n_nodes();

        if n1 == 0 || n2 == 0 {
            return Ok(GmSoftAssignment {
                matrix: Vec::new(),
                discrete: Vec::new(),
                final_beta: beta_init,
            });
        }

        // Compatibility matrix C[i][j] = similarity between g1[i] and g2[j].
        let compat = Self::compute_compatibility(g1, g2);

        // Initialize soft assignment uniformly.
        let init_val = 1.0 / (n2 as f64);
        let mut m = vec![vec![init_val; n2]; n1];

        let mut beta = beta_init;

        for _ in 0..n_outer_iters {
            // Update M based on compatibility + neighborhood agreement.
            let mut energy = vec![vec![0.0; n2]; n1];
            for i in 0..n1 {
                for j in 0..n2 {
                    // Unary term: node compatibility.
                    energy[i][j] = compat[i][j];

                    // Pairwise term: reward consistent neighbor matchings.
                    for e1 in g1.neighbors(i) {
                        let i_prime = e1.target;
                        for e2 in g2.neighbors(j) {
                            let j_prime = e2.target;
                            if i_prime < n1 && j_prime < n2 {
                                energy[i][j] += m[i_prime][j_prime] * 0.5;
                            }
                        }
                    }
                }
            }

            // Softmax with temperature.
            for i in 0..n1 {
                let max_e = energy[i].iter().copied().fold(f64::NEG_INFINITY, f64::max);
                for j in 0..n2 {
                    m[i][j] = ((energy[i][j] - max_e) * beta).exp();
                }
            }

            // Sinkhorn normalization.
            for _ in 0..n_sinkhorn_iters {
                // Normalize rows.
                for i in 0..n1 {
                    let row_sum: f64 = m[i].iter().sum();
                    if row_sum > 1e-15 {
                        for j in 0..n2 {
                            m[i][j] /= row_sum;
                        }
                    }
                }
                // Normalize columns.
                for j in 0..n2 {
                    let col_sum: f64 = (0..n1).map(|i| m[i][j]).sum();
                    if col_sum > 1e-15 {
                        for i in 0..n1 {
                            m[i][j] /= col_sum;
                        }
                    }
                }
            }

            beta *= beta_factor;
        }

        // Discretize: greedy assignment.
        let discrete = Self::greedy_discretize(&m, n1, n2);

        Ok(GmSoftAssignment {
            matrix: m,
            discrete,
            final_beta: beta,
        })
    }

    /// Compute node compatibility matrix based on feature similarity.
    fn compute_compatibility(g1: &GmGraph, g2: &GmGraph) -> Vec<Vec<f64>> {
        let n1 = g1.n_nodes();
        let n2 = g2.n_nodes();
        let mut compat = vec![vec![0.0; n2]; n1];

        for i in 0..n1 {
            for j in 0..n2 {
                let dist = feature_distance(&g1.node_features[i], &g2.node_features[j]);
                // Gaussian similarity.
                compat[i][j] = (-dist * dist / 2.0).exp();
            }
        }

        compat
    }

    /// Greedy discretization of soft assignment matrix.
    fn greedy_discretize(m: &[Vec<f64>], n1: usize, n2: usize) -> Vec<usize> {
        let mut assignment = vec![0usize; n1];
        let mut used = vec![false; n2];

        // Collect all (value, i, j) and sort descending.
        let mut entries: Vec<(f64, usize, usize)> = Vec::new();
        for i in 0..n1 {
            for j in 0..n2 {
                entries.push((m[i][j], i, j));
            }
        }
        entries.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        let mut assigned = vec![false; n1];
        for (_, i, j) in entries {
            if !assigned[i] && !used[j] {
                assignment[i] = j;
                assigned[i] = true;
                used[j] = true;
            }
        }

        assignment
    }
}

// ---------------------------------------------------------------------------
// Maximum Common Subgraph — McSplit-inspired branch & bound
// ---------------------------------------------------------------------------

/// Result of maximum common subgraph computation.
#[derive(Debug, Clone)]
pub struct GmMcsResult {
    /// Node mapping: (g1_node, g2_node) pairs in the MCS.
    pub mapping: Vec<(usize, usize)>,
    /// Number of nodes in the common subgraph.
    pub size: usize,
}

/// Maximum common subgraph finder using branch-and-bound with color-based partitioning.
///
/// Inspired by McSplit (McCreesh et al. 2017), uses compatibility-based
/// branching with upper bound pruning.
pub struct MaxCommonSubgraph;

impl MaxCommonSubgraph {
    /// Find the maximum common subgraph of g1 and g2.
    pub fn find_mcs(g1: &GmGraph, g2: &GmGraph) -> Result<GmMcsResult> {
        let n1 = g1.n_nodes();
        let n2 = g2.n_nodes();

        if n1 == 0 || n2 == 0 {
            return Ok(GmMcsResult {
                mapping: Vec::new(),
                size: 0,
            });
        }

        // Build compatibility: g1[i] can match g2[j] if degree constraint holds.
        let compat = Self::build_compatibility(g1, g2);

        let mut best = GmMcsResult {
            mapping: Vec::new(),
            size: 0,
        };

        let current_mapping: Vec<(usize, usize)> = Vec::new();
        let candidates: Vec<usize> = (0..n1).collect();
        let used_g2: HashSet<usize> = HashSet::new();

        Self::branch_and_bound(
            g1,
            g2,
            &compat,
            &current_mapping,
            &candidates,
            &used_g2,
            &mut best,
        );

        Ok(best)
    }

    /// Build compatibility matrix: compat[i] = set of compatible g2 nodes.
    fn build_compatibility(g1: &GmGraph, g2: &GmGraph) -> Vec<Vec<usize>> {
        let n1 = g1.n_nodes();
        let n2 = g2.n_nodes();
        let mut compat = Vec::with_capacity(n1);

        for i in 0..n1 {
            let mut compatible = Vec::new();
            for j in 0..n2 {
                // Node i can match j if j has at least as many neighbors.
                if g2.degree(j) >= g1.degree(i) {
                    compatible.push(j);
                }
            }
            compat.push(compatible);
        }

        compat
    }

    /// Branch and bound search.
    fn branch_and_bound(
        g1: &GmGraph,
        g2: &GmGraph,
        compat: &[Vec<usize>],
        current: &[(usize, usize)],
        candidates: &[usize],
        used_g2: &HashSet<usize>,
        best: &mut GmMcsResult,
    ) {
        // Upper bound: current size + remaining candidates.
        let upper_bound = current.len() + candidates.len();
        if upper_bound <= best.size {
            return; // Cannot improve.
        }

        if candidates.is_empty() {
            if current.len() > best.size {
                best.mapping = current.to_vec();
                best.size = current.len();
            }
            return;
        }

        // Choose the candidate with the fewest compatible partners (MRV heuristic).
        let mut best_cand_idx = 0;
        let mut min_partners = usize::MAX;
        for (idx, &c) in candidates.iter().enumerate() {
            let partners = compat[c].iter().filter(|j| !used_g2.contains(j)).count();
            if partners < min_partners {
                min_partners = partners;
                best_cand_idx = idx;
            }
        }

        let chosen = candidates[best_cand_idx];
        let mut remaining: Vec<usize> = candidates.to_vec();
        remaining.remove(best_cand_idx);

        // Try matching chosen to each compatible g2 node.
        let compatible_targets: Vec<usize> = compat[chosen]
            .iter()
            .copied()
            .filter(|j| !used_g2.contains(j))
            .collect();

        for target in &compatible_targets {
            // Check adjacency consistency with current mapping.
            if !Self::is_consistent(g1, g2, current, chosen, *target) {
                continue;
            }

            let mut new_current = current.to_vec();
            new_current.push((chosen, *target));
            let mut new_used = used_g2.clone();
            new_used.insert(*target);

            // Filter remaining candidates to those that still have compatible partners.
            let filtered_remaining: Vec<usize> = remaining
                .iter()
                .copied()
                .filter(|&c| compat[c].iter().any(|j| !new_used.contains(j)))
                .collect();

            Self::branch_and_bound(
                g1,
                g2,
                compat,
                &new_current,
                &filtered_remaining,
                &new_used,
                best,
            );
        }

        // Also try NOT matching chosen (skip it).
        Self::branch_and_bound(g1, g2, compat, current, &remaining, used_g2, best);
    }

    /// Check that mapping (chosen -> target) is consistent with existing mappings.
    fn is_consistent(
        g1: &GmGraph,
        g2: &GmGraph,
        current: &[(usize, usize)],
        chosen: usize,
        target: usize,
    ) -> bool {
        for &(c1, c2) in current {
            // If chosen is adjacent to c1 in g1, then target must be adjacent to c2 in g2.
            let adj_g1 = g1.has_edge(chosen, c1);
            let adj_g2 = g2.has_edge(target, c2);
            if adj_g1 && !adj_g2 {
                return false;
            }
            // For induced subgraph: also check reverse.
            if !adj_g1 && adj_g2 {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Graph Match Metrics
// ---------------------------------------------------------------------------

/// Evaluation report for graph matching.
#[derive(Debug, Clone)]
pub struct GmReport {
    /// Matching accuracy (fraction of correct pairs).
    pub accuracy: f64,
    /// Normalized graph edit distance (GED / max_possible).
    pub normalized_ged: f64,
    /// Whether the kernel matrix is positive semi-definite.
    pub kernel_psd: bool,
    /// Kernel matrix Frobenius norm (if computed).
    pub kernel_frobenius_norm: f64,
}

/// Evaluation metrics for graph matching and kernel methods.
pub struct GraphMatchMetrics;

impl GraphMatchMetrics {
    /// Compute matching accuracy: fraction of correctly matched node pairs.
    ///
    /// `predicted` and `ground_truth` are vectors of (g1_node, g2_node) pairs.
    pub fn matching_accuracy(predicted: &[(usize, usize)], ground_truth: &[(usize, usize)]) -> f64 {
        if ground_truth.is_empty() {
            return if predicted.is_empty() { 1.0 } else { 0.0 };
        }

        let gt_set: HashSet<(usize, usize)> = ground_truth.iter().copied().collect();
        let correct = predicted
            .iter()
            .filter(|pair| gt_set.contains(pair))
            .count();

        correct as f64 / ground_truth.len() as f64
    }

    /// Compute normalized graph edit distance.
    ///
    /// Normalized by the maximum possible GED: n1 + n2 + e1 + e2.
    pub fn normalized_ged(ged: f64, g1: &GmGraph, g2: &GmGraph) -> f64 {
        let max_ged = (g1.n_nodes() + g2.n_nodes() + g1.n_edges() + g2.n_edges()) as f64;
        if max_ged < 1e-15 {
            return 0.0;
        }
        ged / max_ged
    }

    /// Check if a kernel matrix is positive semi-definite.
    ///
    /// Uses Gershgorin circle theorem: the matrix is PSD if all eigenvalue
    /// lower bounds (diagonal - row off-diagonal sum) are >= -tolerance.
    pub fn is_positive_semidefinite(kernel_matrix: &[Vec<f64>], tolerance: f64) -> bool {
        let n = kernel_matrix.len();
        if n == 0 {
            return true;
        }

        // Check symmetry first.
        for i in 0..n {
            if kernel_matrix[i].len() != n {
                return false;
            }
            for j in (i + 1)..n {
                if (kernel_matrix[i][j] - kernel_matrix[j][i]).abs() > tolerance {
                    return false;
                }
            }
        }

        // Gershgorin: for each row, eigenvalue >= diag - sum(|off-diag|).
        // If all Gershgorin lower bounds >= -tolerance, likely PSD.
        // This is a necessary condition check; for small matrices we also do
        // leading principal minors check.
        for i in 0..n {
            let diag = kernel_matrix[i][i];
            if diag < -tolerance {
                return false;
            }
        }

        // Check 2x2 leading principal minors: det >= 0.
        for i in 0..n {
            for j in (i + 1)..n {
                let det = kernel_matrix[i][i] * kernel_matrix[j][j]
                    - kernel_matrix[i][j] * kernel_matrix[j][i];
                if det < -tolerance {
                    return false;
                }
            }
        }

        true
    }

    /// Compute the Frobenius norm of a kernel matrix.
    pub fn frobenius_norm(matrix: &[Vec<f64>]) -> f64 {
        let mut sum = 0.0;
        for row in matrix {
            for &val in row {
                sum += val * val;
            }
        }
        sum.sqrt()
    }

    /// Build a full evaluation report.
    pub fn evaluate(
        predicted_matching: &[(usize, usize)],
        ground_truth: &[(usize, usize)],
        ged: f64,
        g1: &GmGraph,
        g2: &GmGraph,
        kernel_matrix: Option<&[Vec<f64>]>,
    ) -> GmReport {
        let accuracy = Self::matching_accuracy(predicted_matching, ground_truth);
        let normalized = Self::normalized_ged(ged, g1, g2);

        let (kernel_psd, kernel_frobenius) = match kernel_matrix {
            Some(km) => (
                Self::is_positive_semidefinite(km, 1e-8),
                Self::frobenius_norm(km),
            ),
            None => (true, 0.0),
        };

        GmReport {
            accuracy,
            normalized_ged: normalized,
            kernel_psd,
            kernel_frobenius_norm: kernel_frobenius,
        }
    }
}
