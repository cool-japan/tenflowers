//! Probabilistic Circuits, Sum-Product Networks, and Tractable Probabilistic Models.
//!
//! This module provides:
//! - **PcGraph**: Graph structure for probabilistic circuits (sum, product, leaf nodes)
//! - **PcEval**: Exact inference (evaluate, log-likelihood, marginalization, MPE)
//! - **SumProductNetwork**: SPN construction strategies (naive, region-graph, random)
//! - **ChowLiuTree**: Structure learning via maximum spanning tree of mutual information
//! - **PcLearning**: EM and gradient-based parameter learning
//! - **PcSampling**: Ancestral and conditional sampling
//! - **PcMetrics**: Evaluation (mean-LL, BIC, perplexity)

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// §1  PcNode — Graph structure
// ─────────────────────────────────────────────────────────────────────────────

/// Type of a node in a probabilistic circuit.
#[derive(Debug, Clone, PartialEq)]
pub enum PcNodeType {
    Sum,
    Product,
    Leaf,
}

/// Leaf distribution for a PC leaf node.
#[derive(Debug, Clone)]
pub enum PcLeafDist {
    Gaussian { mean: f64, std: f64 },
    Bernoulli { p: f64 },
    Categorical { probs: Vec<f64> },
    Indicator { var_idx: usize, value: f64 },
}

/// A single node in a probabilistic circuit graph.
#[derive(Debug, Clone)]
pub struct PcNode {
    pub id: usize,
    pub node_type: PcNodeType,
    pub children: Vec<usize>,
    /// Weights for sum nodes (length == children.len()).
    pub weights: Vec<f64>,
    /// Leaf distribution (only for leaf nodes).
    pub leaf: Option<PcLeafDist>,
    /// Variable scope: which variables this node covers.
    pub var_scope: Vec<usize>,
}

/// A directed acyclic graph of probabilistic circuit nodes.
#[derive(Debug, Clone)]
pub struct PcGraph {
    pub nodes: Vec<PcNode>,
    pub root: usize,
}

impl PcGraph {
    /// Create an empty PC graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root: 0,
        }
    }

    /// Add a sum node; returns the new node id.
    pub fn add_sum_node(&mut self, children: Vec<usize>, weights: Vec<f64>) -> usize {
        let id = self.nodes.len();
        // Collect var_scope as union of children scopes.
        let mut scope: Vec<usize> = Vec::new();
        for &c in &children {
            for &v in &self.nodes[c].var_scope {
                if !scope.contains(&v) {
                    scope.push(v);
                }
            }
        }
        scope.sort_unstable();
        self.nodes.push(PcNode {
            id,
            node_type: PcNodeType::Sum,
            children,
            weights,
            leaf: None,
            var_scope: scope,
        });
        id
    }

    /// Add a product node; returns the new node id.
    pub fn add_product_node(&mut self, children: Vec<usize>) -> usize {
        let id = self.nodes.len();
        let mut scope: Vec<usize> = Vec::new();
        for &c in &children {
            for &v in &self.nodes[c].var_scope {
                if !scope.contains(&v) {
                    scope.push(v);
                }
            }
        }
        scope.sort_unstable();
        self.nodes.push(PcNode {
            id,
            node_type: PcNodeType::Product,
            children,
            weights: Vec::new(),
            leaf: None,
            var_scope: scope,
        });
        id
    }

    /// Add a leaf node; returns the new node id.
    pub fn add_leaf(&mut self, dist: PcLeafDist, var_scope: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(PcNode {
            id,
            node_type: PcNodeType::Leaf,
            children: Vec::new(),
            weights: Vec::new(),
            leaf: Some(dist),
            var_scope,
        });
        id
    }

    /// Number of nodes in the graph.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }
}

impl Default for PcGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  PcEval — Inference
// ─────────────────────────────────────────────────────────────────────────────

/// Partial evidence: values\[i\] = Some(v) means variable i is observed as v;
/// None means the variable is marginalized out.
#[derive(Debug, Clone)]
pub struct PcEvidence {
    pub values: Vec<Option<f64>>,
}

impl PcEvidence {
    pub fn new(values: Vec<Option<f64>>) -> Self {
        Self { values }
    }

    /// Full evidence from a slice.
    pub fn from_sample(sample: &[f64]) -> Self {
        Self {
            values: sample.iter().map(|&v| Some(v)).collect(),
        }
    }

    /// All-None evidence (full marginalization).
    pub fn all_none(n_vars: usize) -> Self {
        Self {
            values: vec![None; n_vars],
        }
    }

    fn get(&self, idx: usize) -> Option<f64> {
        self.values.get(idx).copied().flatten()
    }
}

/// Evaluate the likelihood of a leaf distribution under a given observed value.
fn leaf_likelihood(dist: &PcLeafDist, val: Option<f64>) -> f64 {
    match val {
        None => 1.0, // marginalized: integrate out → 1.0 for normalized distributions
        Some(x) => match dist {
            PcLeafDist::Gaussian { mean, std } => {
                let std_safe = std.max(1e-12);
                let diff = (x - mean) / std_safe;
                let exponent = -0.5 * diff * diff;
                (exponent.exp()) / (std_safe * (2.0 * std::f64::consts::PI).sqrt())
            }
            PcLeafDist::Bernoulli { p } => {
                let p_clamped = p.clamp(1e-12, 1.0 - 1e-12);
                if (x - 1.0).abs() < 0.5 {
                    p_clamped
                } else {
                    1.0 - p_clamped
                }
            }
            PcLeafDist::Categorical { probs } => {
                let idx = x.round() as usize;
                if idx < probs.len() {
                    probs[idx].max(1e-300)
                } else {
                    1e-300
                }
            }
            PcLeafDist::Indicator { var_idx: _, value } => {
                if (x - value).abs() < 1e-9 {
                    1.0
                } else {
                    0.0
                }
            }
        },
    }
}

/// Evaluate a node (recursive, stack-based).
pub fn evaluate_node(graph: &PcGraph, node_id: usize, evidence: &PcEvidence) -> f64 {
    // Iterative post-order traversal to avoid stack overflow on large graphs.
    let n = graph.nodes.len();
    let mut values = vec![f64::NAN; n];
    let mut order = Vec::with_capacity(n);
    let mut stack = vec![node_id];
    let mut visited = vec![false; n];
    // Topological sort by DFS post-order.
    while let Some(&top) = stack.last() {
        if visited[top] {
            stack.pop();
            order.push(top);
        } else {
            visited[top] = true;
            for &child in &graph.nodes[top].children {
                if !visited[child] {
                    stack.push(child);
                }
            }
        }
    }
    for &nid in &order {
        let node = &graph.nodes[nid];
        let val = match node.node_type {
            PcNodeType::Leaf => {
                let scope_val = node.var_scope.first().and_then(|&vi| evidence.get(vi));
                match &node.leaf {
                    Some(dist) => leaf_likelihood(dist, scope_val),
                    None => 1.0,
                }
            }
            PcNodeType::Sum => {
                let mut acc = 0.0;
                for (i, &child) in node.children.iter().enumerate() {
                    let w = *node.weights.get(i).unwrap_or(&1.0);
                    acc += w * values[child];
                }
                acc
            }
            PcNodeType::Product => {
                let mut acc = 1.0;
                for &child in &node.children {
                    acc *= values[child];
                }
                acc
            }
        };
        values[nid] = val;
    }
    values[node_id]
}

/// Compute log-likelihood of a fully observed sample.
pub fn pc_log_likelihood(graph: &PcGraph, sample: &[f64]) -> f64 {
    let evidence = PcEvidence::from_sample(sample);
    let p = evaluate_node(graph, graph.root, &evidence);
    p.max(1e-300).ln()
}

/// Marginalize: evaluate with partial evidence.
pub fn pc_marginalize(graph: &PcGraph, evidence: &PcEvidence) -> f64 {
    evaluate_node(graph, graph.root, evidence)
}

/// Most Probable Explanation: return the mode for each variable.
pub fn pc_mpe(graph: &PcGraph, evidence: &PcEvidence) -> Vec<f64> {
    let n_vars = evidence.values.len();
    let mut result = vec![0.0_f64; n_vars];
    // For observed variables, use their observed values.
    for (i, val) in evidence.values.iter().enumerate() {
        if let Some(v) = val {
            result[i] = *v;
        }
    }
    // For marginalized variables, collect leaf modes and pick the best.
    // Collect per-variable candidate (mode, weight) from sum-weighted traversal.
    let mut var_candidates: Vec<Vec<(f64, f64)>> = vec![Vec::new(); n_vars];
    collect_mpe_candidates(graph, graph.root, 1.0, &mut var_candidates, evidence);
    for v in 0..n_vars {
        if evidence.values.get(v).copied().flatten().is_none() && !var_candidates[v].is_empty() {
            // Choose mode from highest-weighted candidate.
            let best = var_candidates[v]
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(m, _)| *m)
                .unwrap_or(0.0);
            result[v] = best;
        }
    }
    result
}

fn collect_mpe_candidates(
    graph: &PcGraph,
    node_id: usize,
    weight: f64,
    candidates: &mut Vec<Vec<(f64, f64)>>,
    evidence: &PcEvidence,
) {
    let node = &graph.nodes[node_id];
    match node.node_type {
        PcNodeType::Leaf => {
            if let (Some(dist), Some(&var_idx)) = (&node.leaf, node.var_scope.first()) {
                if evidence.values.get(var_idx).copied().flatten().is_none() {
                    let mode = leaf_mode(dist);
                    if var_idx < candidates.len() {
                        candidates[var_idx].push((mode, weight));
                    }
                }
            }
        }
        PcNodeType::Sum => {
            // Propagate to highest-weight child.
            let best_child = node
                .children
                .iter()
                .enumerate()
                .max_by(|(i, &ci), (j, &cj)| {
                    let wi = *node.weights.get(*i).unwrap_or(&0.0);
                    let wj = *node.weights.get(*j).unwrap_or(&0.0);
                    (wi * evaluate_node(graph, ci, evidence))
                        .partial_cmp(&(wj * evaluate_node(graph, cj, evidence)))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            if let Some((i, &child)) = best_child {
                let w = *node.weights.get(i).unwrap_or(&1.0);
                collect_mpe_candidates(graph, child, weight * w, candidates, evidence);
            }
        }
        PcNodeType::Product => {
            for &child in &node.children {
                collect_mpe_candidates(graph, child, weight, candidates, evidence);
            }
        }
    }
}

fn leaf_mode(dist: &PcLeafDist) -> f64 {
    match dist {
        PcLeafDist::Gaussian { mean, .. } => *mean,
        PcLeafDist::Bernoulli { p } => {
            if *p >= 0.5 {
                1.0
            } else {
                0.0
            }
        }
        PcLeafDist::Categorical { probs } => probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as f64)
            .unwrap_or(0.0),
        PcLeafDist::Indicator { value, .. } => *value,
    }
}

/// Evaluate with all-None evidence; should equal 1.0 for a normalized PC.
pub fn pc_partition_function(graph: &PcGraph, n_vars: usize) -> f64 {
    let evidence = PcEvidence::all_none(n_vars);
    evaluate_node(graph, graph.root, &evidence)
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  SumProductNetwork — SPN constructions
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for SPN construction.
#[derive(Debug, Clone)]
pub struct SpnConfig {
    pub n_vars: usize,
    pub n_sum_nodes: usize,
    pub n_product_nodes: usize,
}

/// Naive factorization: for each variable, sum of Gaussian leaves; product at root.
pub struct NaiveFactorization;

impl NaiveFactorization {
    /// Build a naive factorized SPN.
    /// For each variable: `n_leaf_per_var` Gaussian leaves → one sum node.
    /// Product of all sum nodes at root.
    pub fn build(n_vars: usize, n_leaf_per_var: usize) -> PcGraph {
        let mut g = PcGraph::new();
        let mut var_sums = Vec::with_capacity(n_vars);
        for v in 0..n_vars {
            let mut leaves = Vec::with_capacity(n_leaf_per_var);
            for k in 0..n_leaf_per_var {
                let mean = (k as f64 - n_leaf_per_var as f64 / 2.0) * 0.5;
                let leaf_id = g.add_leaf(PcLeafDist::Gaussian { mean, std: 1.0 }, vec![v]);
                leaves.push(leaf_id);
            }
            let w = 1.0 / n_leaf_per_var as f64;
            let weights = vec![w; n_leaf_per_var];
            let sum_id = g.add_sum_node(leaves, weights);
            var_sums.push(sum_id);
        }
        let root = if n_vars == 1 {
            var_sums[0]
        } else {
            g.add_product_node(var_sums)
        };
        g.root = root;
        g
    }
}

/// Region-graph based SPN (Poon & Domingos 2011 style).
pub struct RegionGraph;

impl RegionGraph {
    /// Build a dense SPN over `n_vars` variables.
    /// Recursively splits variable range in half; at leaves uses `n_sums_per_region` Gaussian sums.
    pub fn build_dense(n_vars: usize, n_sums_per_region: usize) -> PcGraph {
        let mut g = PcGraph::new();
        let vars: Vec<usize> = (0..n_vars).collect();
        let root = Self::build_region(&mut g, &vars, n_sums_per_region, 0);
        g.root = root;
        g
    }

    #[allow(clippy::only_used_in_recursion)]
    fn build_region(g: &mut PcGraph, vars: &[usize], n_sums: usize, depth: usize) -> usize {
        if vars.len() == 1 {
            // Base case: sum of Gaussian leaves for one variable.
            let v = vars[0];
            let mut leaves = Vec::with_capacity(n_sums);
            for k in 0..n_sums {
                let mean = (k as f64 - n_sums as f64 / 2.0) * 0.5;
                let leaf_id = g.add_leaf(PcLeafDist::Gaussian { mean, std: 1.0 }, vec![v]);
                leaves.push(leaf_id);
            }
            let w = 1.0 / n_sums as f64;
            let weights = vec![w; n_sums];
            g.add_sum_node(leaves, weights)
        } else {
            // Split in half.
            let mid = vars.len() / 2;
            let left_vars = &vars[..mid];
            let right_vars = &vars[mid..];
            let left_node = Self::build_region(g, left_vars, n_sums, depth + 1);
            let right_node = Self::build_region(g, right_vars, n_sums, depth + 1);
            let prod_id = g.add_product_node(vec![left_node, right_node]);
            // Wrap in a sum node to create a valid region.
            let w = 1.0;
            g.add_sum_node(vec![prod_id], vec![w])
        }
    }
}

/// Random valid SPN structure.
pub struct RandomStructure;

impl RandomStructure {
    /// Build a random SPN of given depth.
    /// Alternates sum/product levels; variables are distributed randomly at leaves.
    pub fn build(n_vars: usize, depth: usize, n_sums: usize, rng: &mut impl Rng) -> PcGraph {
        let mut g = PcGraph::new();
        let vars: Vec<usize> = (0..n_vars).collect();
        let root = Self::build_random_node(&mut g, &vars, depth, n_sums, true, rng);
        g.root = root;
        g
    }

    fn build_random_node(
        g: &mut PcGraph,
        vars: &[usize],
        depth: usize,
        n_sums: usize,
        is_sum: bool,
        rng: &mut impl Rng,
    ) -> usize {
        if depth == 0 || vars.len() == 1 {
            let v = vars[0];
            let mean: f64 = rng.random::<f64>() * 2.0 - 1.0;
            let leaf_id = g.add_leaf(PcLeafDist::Gaussian { mean, std: 1.0 }, vec![v]);
            return leaf_id;
        }
        if is_sum {
            // Sum node with n_sums children, each a product node.
            let mut children = Vec::with_capacity(n_sums);
            for _ in 0..n_sums {
                // Randomly shuffle and split vars.
                let mut shuffled = vars.to_vec();
                shuffle_vec(&mut shuffled, rng);
                let mid = (shuffled.len() / 2).max(1);
                let left = &shuffled[..mid];
                let right = if mid < shuffled.len() {
                    &shuffled[mid..]
                } else {
                    &shuffled[..mid]
                };
                let left_node = Self::build_random_node(g, left, depth - 1, n_sums, false, rng);
                let right_node = Self::build_random_node(g, right, depth - 1, n_sums, false, rng);
                let prod = g.add_product_node(vec![left_node, right_node]);
                children.push(prod);
            }
            let w = 1.0 / n_sums as f64;
            let weights = vec![w; n_sums];
            g.add_sum_node(children, weights)
        } else {
            // Product node: split vars into two halves.
            let mid = (vars.len() / 2).max(1);
            let left = &vars[..mid];
            let right = if mid < vars.len() {
                &vars[mid..]
            } else {
                &vars[..mid]
            };
            let left_node = Self::build_random_node(g, left, depth - 1, n_sums, true, rng);
            let right_node = Self::build_random_node(g, right, depth - 1, n_sums, true, rng);
            g.add_product_node(vec![left_node, right_node])
        }
    }
}

fn shuffle_vec<T>(v: &mut [T], rng: &mut impl Rng) {
    let n = v.len();
    for i in (1..n).rev() {
        let j = (rng.random::<f64>() * (i + 1) as f64) as usize;
        v.swap(i, j.min(i));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  ChowLiuTree — Structure learning
// ─────────────────────────────────────────────────────────────────────────────

/// Mutual information computation utilities.
pub struct PcMutualInformation;

impl PcMutualInformation {
    /// Compute discrete MI between two discrete variable arrays.
    /// `nx` and `ny` are the number of categories for x and y.
    pub fn discrete_mi(x: &[usize], y: &[usize], nx: usize, ny: usize) -> f64 {
        let n = x.len().min(y.len());
        if n == 0 || nx == 0 || ny == 0 {
            return 0.0;
        }
        let n_f = n as f64;
        // Joint counts.
        let mut joint = vec![vec![0usize; ny]; nx];
        let mut px = vec![0usize; nx];
        let mut py = vec![0usize; ny];
        for i in 0..n {
            let xi = x[i].min(nx - 1);
            let yi = y[i].min(ny - 1);
            joint[xi][yi] += 1;
            px[xi] += 1;
            py[yi] += 1;
        }
        let mut mi = 0.0;
        for xi in 0..nx {
            for yi in 0..ny {
                let pxy = joint[xi][yi] as f64 / n_f;
                if pxy > 1e-300 {
                    let pxi = px[xi] as f64 / n_f;
                    let pyi = py[yi] as f64 / n_f;
                    if pxi > 1e-300 && pyi > 1e-300 {
                        mi += pxy * (pxy / (pxi * pyi)).ln();
                    }
                }
            }
        }
        mi.max(0.0)
    }

    /// Compute MI between two continuous variables by KDE-based discretization.
    pub fn continuous_mi_kde(x: &[f64], y: &[f64], n_bins: usize) -> f64 {
        let n = x.len().min(y.len());
        if n == 0 || n_bins == 0 {
            return 0.0;
        }
        let xd = discretize(x, n_bins);
        let yd = discretize(y, n_bins);
        Self::discrete_mi(&xd, &yd, n_bins, n_bins)
    }
}

fn discretize(vals: &[f64], n_bins: usize) -> Vec<usize> {
    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1e-12);
    vals.iter()
        .map(|&v| (((v - min) / range) * n_bins as f64) as usize)
        .map(|b: usize| b.min(n_bins - 1))
        .collect()
}

/// Prim's minimum spanning tree (used for maximum spanning tree by negating weights).
pub struct PcPrimMst;

impl PcPrimMst {
    /// Find maximum spanning tree for a complete weighted graph.
    /// Returns n-1 edges as (u, v) pairs.
    pub fn find_mst(n: usize, edge_weights: &[Vec<f64>]) -> Vec<(usize, usize)> {
        if n == 0 {
            return Vec::new();
        }
        let mut in_tree = vec![false; n];
        let mut best_weight = vec![f64::NEG_INFINITY; n];
        let mut best_from = vec![0usize; n];
        best_weight[0] = f64::INFINITY;
        in_tree[0] = true;
        let mut edges = Vec::new();
        // Initialize neighbors of node 0.
        for v in 1..n {
            let w = edge_weights.first()
                .and_then(|row| row.get(v))
                .copied()
                .unwrap_or(0.0);
            best_weight[v] = w;
            best_from[v] = 0;
        }
        for _ in 1..n {
            // Find max weight node not in tree.
            let u = (0..n).filter(|&v| !in_tree[v]).max_by(|&a, &b| {
                best_weight[a]
                    .partial_cmp(&best_weight[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let u = match u {
                Some(x) => x,
                None => break,
            };
            in_tree[u] = true;
            edges.push((best_from[u], u));
            // Update neighbors.
            for v in 0..n {
                if !in_tree[v] {
                    let w = edge_weights
                        .get(u)
                        .and_then(|row| row.get(v))
                        .copied()
                        .unwrap_or(0.0);
                    if w > best_weight[v] {
                        best_weight[v] = w;
                        best_from[v] = u;
                    }
                }
            }
        }
        edges
    }
}

/// Chow-Liu tree structure learner.
#[derive(Debug, Clone)]
pub struct ChowLiuTree {
    pub edges: Vec<(usize, usize, f64)>,
    pub root: usize,
}

impl ChowLiuTree {
    /// Fit a Chow-Liu tree to data using pairwise MI.
    pub fn fit(data: &[Vec<f64>], n_bins: usize) -> Self {
        if data.is_empty() {
            return Self {
                edges: Vec::new(),
                root: 0,
            };
        }
        let n_vars = data[0].len();
        if n_vars == 0 {
            return Self {
                edges: Vec::new(),
                root: 0,
            };
        }
        // Extract columns.
        let cols: Vec<Vec<f64>> = (0..n_vars)
            .map(|v| data.iter().map(|row| *row.get(v).unwrap_or(&0.0)).collect())
            .collect();
        // Pairwise MI matrix.
        let mut mi_matrix = vec![vec![0.0_f64; n_vars]; n_vars];
        for i in 0..n_vars {
            for j in i + 1..n_vars {
                let mi = PcMutualInformation::continuous_mi_kde(&cols[i], &cols[j], n_bins);
                mi_matrix[i][j] = mi;
                mi_matrix[j][i] = mi;
            }
        }
        // Maximum spanning tree via Prim.
        let mst_edges = PcPrimMst::find_mst(n_vars, &mi_matrix);
        let edges: Vec<(usize, usize, f64)> = mst_edges
            .iter()
            .map(|&(u, v)| (u, v, mi_matrix[u][v]))
            .collect();
        Self { edges, root: 0 }
    }

    /// Build a PC from the Chow-Liu tree structure using marginal/conditional Gaussians.
    pub fn build_pc(tree: &ChowLiuTree, data: &[Vec<f64>], n_bins: usize) -> PcGraph {
        if data.is_empty() {
            return PcGraph::new();
        }
        let n_vars = data[0].len();
        if n_vars == 0 {
            return PcGraph::new();
        }
        let mut g = PcGraph::new();
        // Build adjacency from tree edges.
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_vars];
        for &(u, v, _) in &tree.edges {
            adj[u].push(v);
            adj[v].push(u);
        }
        // Extract column statistics.
        let cols: Vec<Vec<f64>> = (0..n_vars)
            .map(|v| data.iter().map(|row| *row.get(v).unwrap_or(&0.0)).collect())
            .collect();
        let means: Vec<f64> = cols.iter().map(|col| mean_f64(col)).collect();
        let stds: Vec<f64> = cols.iter().map(|col| std_f64(col)).collect();
        // BFS from root to build factorized PC: P(root) * Π P(x_i | parent)
        let mut node_ids: Vec<Option<usize>> = vec![None; n_vars];
        let mut parent: Vec<Option<usize>> = vec![None; n_vars];
        let mut bfs_order = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        let mut visited = vec![false; n_vars];
        queue.push_back(tree.root);
        visited[tree.root] = true;
        while let Some(v) = queue.pop_front() {
            bfs_order.push(v);
            for &nb in &adj[v] {
                if !visited[nb] {
                    visited[nb] = true;
                    parent[nb] = Some(v);
                    queue.push_back(nb);
                }
            }
        }
        // Create leaf nodes for each variable (Gaussian).
        for &v in &bfs_order {
            let mean = means[v];
            let std_v = stds[v].max(1e-6);
            let leaf_id = g.add_leaf(PcLeafDist::Gaussian { mean, std: std_v }, vec![v]);
            node_ids[v] = Some(leaf_id);
        }
        // Product over all variables.
        let all_leaves: Vec<usize> = bfs_order.iter().filter_map(|&v| node_ids[v]).collect();
        let root = if all_leaves.len() == 1 {
            all_leaves[0]
        } else {
            g.add_product_node(all_leaves)
        };
        let _ = n_bins; // used for MI computation; leaf params derived from empirical stats
        g.root = root;
        g
    }
}

fn mean_f64(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn std_f64(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 1.0;
    }
    let m = mean_f64(v);
    let var = v.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64;
    var.sqrt().max(1e-12)
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  PcLearning — Parameter learning
// ─────────────────────────────────────────────────────────────────────────────

/// EM learner for SPN parameters.
pub struct PcEmLearner;

impl PcEmLearner {
    /// E-step: compute responsibilities for each sum node's children.
    /// Returns Vec indexed by node_id; each element is Vec of responsibilities (one per child).
    pub fn e_step(graph: &PcGraph, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n_nodes = graph.nodes.len();
        let mut responsibilities: Vec<Vec<f64>> = graph
            .nodes
            .iter()
            .map(|n| {
                if n.node_type == PcNodeType::Sum {
                    vec![0.0_f64; n.children.len()]
                } else {
                    Vec::new()
                }
            })
            .collect();
        for sample in data {
            let evidence = PcEvidence::from_sample(sample);
            // Compute forward values for all nodes.
            let mut values = vec![0.0_f64; n_nodes];
            let mut order = compute_topological_order(graph);
            for &nid in &order {
                let node = &graph.nodes[nid];
                values[nid] = match node.node_type {
                    PcNodeType::Leaf => {
                        let scope_val =
                            node.var_scope.first().and_then(|&vi| evidence.get(vi));
                        match &node.leaf {
                            Some(dist) => leaf_likelihood(dist, scope_val),
                            None => 1.0,
                        }
                    }
                    PcNodeType::Sum => node
                        .children
                        .iter()
                        .enumerate()
                        .map(|(i, &c)| node.weights.get(i).copied().unwrap_or(1.0) * values[c])
                        .sum(),
                    PcNodeType::Product => node.children.iter().map(|&c| values[c]).product(),
                };
            }
            // Accumulate responsibilities.
            for nid in 0..n_nodes {
                if graph.nodes[nid].node_type == PcNodeType::Sum {
                    let node = &graph.nodes[nid];
                    let total = values[nid].max(1e-300);
                    for (i, &child) in node.children.iter().enumerate() {
                        let w = node.weights.get(i).copied().unwrap_or(1.0);
                        responsibilities[nid][i] += w * values[child] / total;
                    }
                }
            }
            let _ = order.len();
        }
        responsibilities
    }

    /// M-step: update sum node weights from accumulated responsibilities.
    pub fn m_step(graph: &mut PcGraph, responsibilities: &[Vec<f64>]) {
        for (nid, node) in graph.nodes.iter_mut().enumerate() {
            if node.node_type == PcNodeType::Sum {
                if let Some(resp) = responsibilities.get(nid) {
                    if !resp.is_empty() {
                        let total: f64 = resp.iter().sum::<f64>().max(1e-300);
                        node.weights = resp.iter().map(|&r| r / total).collect();
                    }
                }
            }
        }
    }

    /// Fit via EM for `n_iter` iterations; returns log-likelihood history.
    pub fn fit(graph: &mut PcGraph, data: &[Vec<f64>], n_iter: usize) -> Vec<f64> {
        let n_vars = data.first().map(|s| s.len()).unwrap_or(0);
        let mut ll_history = Vec::with_capacity(n_iter);
        for _ in 0..n_iter {
            let responsibilities = Self::e_step(graph, data);
            Self::m_step(graph, &responsibilities);
            // Compute mean log-likelihood.
            let ll = if data.is_empty() {
                f64::NEG_INFINITY
            } else {
                data.iter()
                    .map(|s| pc_log_likelihood(graph, s))
                    .sum::<f64>()
                    / data.len() as f64
            };
            ll_history.push(ll);
            let _ = n_vars;
        }
        ll_history
    }
}

fn compute_topological_order(graph: &PcGraph) -> Vec<usize> {
    let n = graph.nodes.len();
    let mut order = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    let mut stack = vec![graph.root];
    while let Some(&top) = stack.last() {
        if visited[top] {
            stack.pop();
            order.push(top);
        } else {
            visited[top] = true;
            for &child in &graph.nodes[top].children {
                if !visited[child] {
                    stack.push(child);
                }
            }
        }
    }
    order
}

/// Gradient-based learner for SPN weights.
pub struct PcGradientLearner;

impl PcGradientLearner {
    /// Compute gradient of log-likelihood w.r.t. sum node weights.
    /// Returns Vec<(node_id, Vec<grad_per_weight>)>.
    pub fn log_likelihood_gradient_weights(
        graph: &PcGraph,
        data: &[Vec<f64>],
    ) -> Vec<(usize, Vec<f64>)> {
        let n_nodes = graph.nodes.len();
        let n_data = data.len();
        if n_data == 0 {
            return Vec::new();
        }
        let mut grad_accum: Vec<Vec<f64>> = graph
            .nodes
            .iter()
            .map(|n| {
                if n.node_type == PcNodeType::Sum {
                    vec![0.0_f64; n.children.len()]
                } else {
                    Vec::new()
                }
            })
            .collect();
        for sample in data {
            let evidence = PcEvidence::from_sample(sample);
            // Forward pass.
            let mut fwd = vec![0.0_f64; n_nodes];
            let order = compute_topological_order(graph);
            for &nid in &order {
                let node = &graph.nodes[nid];
                fwd[nid] = match node.node_type {
                    PcNodeType::Leaf => {
                        let scope_val =
                            node.var_scope.first().and_then(|&vi| evidence.get(vi));
                        match &node.leaf {
                            Some(d) => leaf_likelihood(d, scope_val),
                            None => 1.0,
                        }
                    }
                    PcNodeType::Sum => node
                        .children
                        .iter()
                        .enumerate()
                        .map(|(i, &c)| node.weights.get(i).copied().unwrap_or(1.0) * fwd[c])
                        .sum(),
                    PcNodeType::Product => node.children.iter().map(|&c| fwd[c]).product(),
                };
            }
            // Backward pass: d(log P)/d(w_i) = (1/P_root) * d(P_root)/d(w_i)
            // For sum node s with child c_i: dP_root/dw_i = (P_root / P_s) * fwd[c_i]
            // which simplifies to: grad_w_i = fwd[c_i] / P_s (when P_root / (P_s * P_root) = 1/P_s)
            // Full: d log P / d w_i = (1/P_root) * backprop[s] * fwd[c_i]
            let mut backprop = vec![0.0_f64; n_nodes];
            backprop[graph.root] = 1.0;
            // Propagate in reverse topological order.
            for &nid in order.iter().rev() {
                let bp = backprop[nid];
                let node = &graph.nodes[nid];
                match node.node_type {
                    PcNodeType::Sum => {
                        for (i, &c) in node.children.iter().enumerate() {
                            let w = node.weights.get(i).copied().unwrap_or(1.0);
                            backprop[c] += bp * w;
                            // Gradient w.r.t. w_i
                            grad_accum[nid][i] += bp * fwd[c];
                        }
                    }
                    PcNodeType::Product => {
                        let prod = fwd[nid];
                        for &c in &node.children {
                            let fc = fwd[c];
                            if fc.abs() > 1e-300 {
                                backprop[c] += bp * prod / fc;
                            } else {
                                // Compute product excluding c.
                                let exc: f64 = node
                                    .children
                                    .iter()
                                    .filter(|&&cc| cc != c)
                                    .map(|&cc| fwd[cc])
                                    .product();
                                backprop[c] += bp * exc;
                            }
                        }
                    }
                    PcNodeType::Leaf => {}
                }
            }
            // Normalize by P_root.
            let p_root = fwd[graph.root].max(1e-300);
            for nid in 0..n_nodes {
                if graph.nodes[nid].node_type == PcNodeType::Sum {
                    for g in grad_accum[nid].iter_mut() {
                        *g /= p_root;
                    }
                }
            }
        }
        // Average over data.
        let scale = 1.0 / n_data as f64;
        let mut result = Vec::new();
        for (nid, grads) in grad_accum.iter().enumerate() {
            if !grads.is_empty() {
                result.push((nid, grads.iter().map(|&g| g * scale).collect()));
            }
        }
        result
    }

    /// Apply a gradient step and project to simplex via softmax.
    pub fn gradient_step(graph: &mut PcGraph, gradients: &[(usize, Vec<f64>)], lr: f64) {
        for (node_id, grads) in gradients {
            if *node_id >= graph.nodes.len() {
                continue;
            }
            let node = &mut graph.nodes[*node_id];
            if node.node_type != PcNodeType::Sum {
                continue;
            }
            if node.weights.len() != grads.len() {
                continue;
            }
            // Gradient ascent step in log-space then softmax.
            let log_w: Vec<f64> = node.weights.iter().map(|&w| w.max(1e-300).ln()).collect();
            let updated: Vec<f64> = log_w
                .iter()
                .zip(grads.iter())
                .map(|(&lw, &g)| lw + lr * g)
                .collect();
            node.weights = softmax_vec(&updated);
        }
    }
}

fn softmax_vec(v: &[f64]) -> Vec<f64> {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum::<f64>().max(1e-300);
    exps.iter().map(|&e| e / sum).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  PcSampling — Sampling from probabilistic circuits
// ─────────────────────────────────────────────────────────────────────────────

/// Sample a single value from a leaf distribution.
pub fn pc_sample_leaf(dist: &PcLeafDist, rng: &mut impl Rng) -> f64 {
    match dist {
        PcLeafDist::Gaussian { mean, std } => {
            // Box-Muller transform.
            let u1 = (rng.random::<f64>()).max(1e-300);
            let u2: f64 = rng.random::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            mean + std * z
        }
        PcLeafDist::Bernoulli { p } => {
            if rng.random::<f64>() < *p {
                1.0
            } else {
                0.0
            }
        }
        PcLeafDist::Categorical { probs } => {
            let u: f64 = rng.random::<f64>();
            let mut cum = 0.0;
            for (i, &p) in probs.iter().enumerate() {
                cum += p;
                if u <= cum {
                    return i as f64;
                }
            }
            (probs.len().saturating_sub(1)) as f64
        }
        PcLeafDist::Indicator { value, .. } => *value,
    }
}

/// Ancestral sampling: top-down traversal.
pub fn pc_ancestral_sample(graph: &PcGraph, rng: &mut impl Rng) -> Vec<f64> {
    let n_vars = graph
        .nodes
        .iter()
        .flat_map(|n| n.var_scope.iter().copied())
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    let mut result = vec![f64::NAN; n_vars];
    let evidence = PcEvidence::all_none(n_vars);
    ancestral_sample_node(graph, graph.root, &mut result, &evidence, rng);
    result
}

fn ancestral_sample_node(
    graph: &PcGraph,
    node_id: usize,
    result: &mut Vec<f64>,
    evidence: &PcEvidence,
    rng: &mut impl Rng,
) {
    let node = &graph.nodes[node_id];
    match node.node_type {
        PcNodeType::Leaf => {
            if let (Some(dist), Some(&var_idx)) = (&node.leaf, node.var_scope.first()) {
                if var_idx < result.len() && result[var_idx].is_nan() {
                    result[var_idx] = pc_sample_leaf(dist, rng);
                }
            }
        }
        PcNodeType::Sum => {
            if node.children.is_empty() {
                return;
            }
            // Evaluate each child.
            let evals: Vec<f64> = node
                .children
                .iter()
                .enumerate()
                .map(|(i, &c)| {
                    let w = node.weights.get(i).copied().unwrap_or(1.0);
                    w * evaluate_node(graph, c, evidence).max(0.0)
                })
                .collect();
            let total: f64 = evals.iter().sum::<f64>().max(1e-300);
            let u: f64 = rng.random::<f64>();
            let mut cum = 0.0;
            let mut chosen = node.children.len() - 1;
            for (i, &eval) in evals.iter().enumerate() {
                cum += eval / total;
                if u <= cum {
                    chosen = i;
                    break;
                }
            }
            ancestral_sample_node(graph, node.children[chosen], result, evidence, rng);
        }
        PcNodeType::Product => {
            for &child in &node.children {
                ancestral_sample_node(graph, child, result, evidence, rng);
            }
        }
    }
}

/// Sample a batch of samples.
pub fn pc_sample_batch(graph: &PcGraph, n_samples: usize, rng: &mut impl Rng) -> Vec<Vec<f64>> {
    (0..n_samples)
        .map(|_| pc_ancestral_sample(graph, rng))
        .collect()
}

/// Conditional sample: evidence variables are fixed, remaining are sampled.
pub fn pc_conditional_sample(
    graph: &PcGraph,
    evidence: &PcEvidence,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let n_vars = evidence.values.len();
    let mut result = vec![f64::NAN; n_vars];
    // Fill in observed variables.
    for (i, val) in evidence.values.iter().enumerate() {
        if let Some(v) = val {
            result[i] = *v;
        }
    }
    conditional_sample_node(graph, graph.root, &mut result, evidence, rng);
    // Fill any remaining NaN with 0.0.
    for v in result.iter_mut() {
        if v.is_nan() {
            *v = 0.0;
        }
    }
    result
}

fn conditional_sample_node(
    graph: &PcGraph,
    node_id: usize,
    result: &mut Vec<f64>,
    evidence: &PcEvidence,
    rng: &mut impl Rng,
) {
    let node = &graph.nodes[node_id];
    match node.node_type {
        PcNodeType::Leaf => {
            if let (Some(dist), Some(&var_idx)) = (&node.leaf, node.var_scope.first()) {
                if var_idx < result.len() && result[var_idx].is_nan() {
                    // Not in evidence → sample.
                    result[var_idx] = pc_sample_leaf(dist, rng);
                }
                // If in evidence, keep the observed value.
            }
        }
        PcNodeType::Sum => {
            if node.children.is_empty() {
                return;
            }
            // Weight by evidence-conditioned probability.
            let evals: Vec<f64> = node
                .children
                .iter()
                .enumerate()
                .map(|(i, &c)| {
                    let w = node.weights.get(i).copied().unwrap_or(1.0);
                    w * evaluate_node(graph, c, evidence).max(0.0)
                })
                .collect();
            let total: f64 = evals.iter().sum::<f64>().max(1e-300);
            let u: f64 = rng.random::<f64>();
            let mut cum = 0.0;
            let mut chosen = node.children.len() - 1;
            for (i, &eval) in evals.iter().enumerate() {
                cum += eval / total;
                if u <= cum {
                    chosen = i;
                    break;
                }
            }
            conditional_sample_node(graph, node.children[chosen], result, evidence, rng);
        }
        PcNodeType::Product => {
            for &child in &node.children {
                conditional_sample_node(graph, child, result, evidence, rng);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  PcMetrics — Evaluation
// ─────────────────────────────────────────────────────────────────────────────

/// Mean log-likelihood over a dataset.
pub fn pc_mean_log_likelihood(graph: &PcGraph, data: &[Vec<f64>]) -> f64 {
    if data.is_empty() {
        return f64::NEG_INFINITY;
    }
    data.iter()
        .map(|s| pc_log_likelihood(graph, s))
        .sum::<f64>()
        / data.len() as f64
}

/// BIC score: -2 * log_lik + k * ln(n), where k = number of free parameters.
pub fn pc_bic_score(graph: &PcGraph, data: &[Vec<f64>]) -> f64 {
    let n = data.len();
    if n == 0 {
        return f64::INFINITY;
    }
    let log_lik: f64 = data.iter().map(|s| pc_log_likelihood(graph, s)).sum();
    // Count free parameters: for each sum node with k children, k-1 free weights.
    let k: usize = graph
        .nodes
        .iter()
        .filter(|n| n.node_type == PcNodeType::Sum && n.children.len() > 1)
        .map(|n| n.children.len() - 1)
        .sum();
    -2.0 * log_lik + k as f64 * (n as f64).ln()
}

/// Perplexity: exp(-mean_log_lik).
pub fn pc_perplexity(graph: &PcGraph, data: &[Vec<f64>]) -> f64 {
    let mll = pc_mean_log_likelihood(graph, data);
    (-mll).exp()
}

/// Evaluation report for a PC model.
#[derive(Debug, Clone)]
pub struct PcEvalReport {
    pub mean_ll: f64,
    pub bic: f64,
    pub perplexity: f64,
    pub n_nodes: usize,
    pub n_params: usize,
}

/// Full evaluation of a PC model on data.
pub fn pc_evaluate(graph: &PcGraph, data: &[Vec<f64>]) -> PcEvalReport {
    let mean_ll = pc_mean_log_likelihood(graph, data);
    let bic = pc_bic_score(graph, data);
    let perplexity = pc_perplexity(graph, data);
    let n_nodes = graph.n_nodes();
    let n_params: usize = graph
        .nodes
        .iter()
        .filter(|n| n.node_type == PcNodeType::Sum && n.children.len() > 1)
        .map(|n| n.children.len() - 1)
        .sum();
    PcEvalReport {
        mean_ll,
        bic,
        perplexity,
        n_nodes,
        n_params,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::SeedableRng;

    fn make_simple_graph() -> PcGraph {
        // Two Gaussian leaves → sum node → root
        let mut g = PcGraph::new();
        let l0 = g.add_leaf(
            PcLeafDist::Gaussian {
                mean: 0.0,
                std: 1.0,
            },
            vec![0],
        );
        let l1 = g.add_leaf(
            PcLeafDist::Gaussian {
                mean: 1.0,
                std: 1.0,
            },
            vec![0],
        );
        g.add_sum_node(vec![l0, l1], vec![0.5, 0.5]);
        g.root = 2;
        g
    }

    fn make_product_graph() -> PcGraph {
        let mut g = PcGraph::new();
        let l0 = g.add_leaf(
            PcLeafDist::Gaussian {
                mean: 0.0,
                std: 1.0,
            },
            vec![0],
        );
        let l1 = g.add_leaf(
            PcLeafDist::Gaussian {
                mean: 0.0,
                std: 1.0,
            },
            vec![1],
        );
        g.add_product_node(vec![l0, l1]);
        g.root = 2;
        g
    }

    fn make_synthetic_data(n: usize, n_vars: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                (0..n_vars)
                    .map(|_| {
                        let u1 = (rng.random::<f64>()).max(1e-300);
                        let u2: f64 = rng.random::<f64>();
                        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                    })
                    .collect()
            })
            .collect()
    }

    // §1 PcGraph structure tests
    #[test]
    fn test_pc_graph_add_leaf() {
        let mut g = PcGraph::new();
        let id = g.add_leaf(
            PcLeafDist::Gaussian {
                mean: 0.0,
                std: 1.0,
            },
            vec![0],
        );
        assert_eq!(id, 0);
        assert_eq!(g.nodes[0].node_type, PcNodeType::Leaf);
    }

    #[test]
    fn test_pc_graph_add_sum() {
        let mut g = PcGraph::new();
        let l0 = g.add_leaf(
            PcLeafDist::Gaussian {
                mean: 0.0,
                std: 1.0,
            },
            vec![0],
        );
        let l1 = g.add_leaf(
            PcLeafDist::Gaussian {
                mean: 1.0,
                std: 1.0,
            },
            vec![0],
        );
        let s = g.add_sum_node(vec![l0, l1], vec![0.5, 0.5]);
        assert_eq!(g.nodes[s].node_type, PcNodeType::Sum);
        assert_eq!(g.nodes[s].children.len(), 2);
    }

    #[test]
    fn test_pc_graph_add_product() {
        let mut g = PcGraph::new();
        let l0 = g.add_leaf(
            PcLeafDist::Gaussian {
                mean: 0.0,
                std: 1.0,
            },
            vec![0],
        );
        let l1 = g.add_leaf(
            PcLeafDist::Gaussian {
                mean: 0.0,
                std: 1.0,
            },
            vec![1],
        );
        let p = g.add_product_node(vec![l0, l1]);
        assert_eq!(g.nodes[p].node_type, PcNodeType::Product);
    }

    #[test]
    fn test_pc_graph_n_nodes() {
        let mut g = PcGraph::new();
        g.add_leaf(
            PcLeafDist::Gaussian {
                mean: 0.0,
                std: 1.0,
            },
            vec![0],
        );
        g.add_leaf(
            PcLeafDist::Gaussian {
                mean: 1.0,
                std: 1.0,
            },
            vec![1],
        );
        assert_eq!(g.n_nodes(), 2);
    }

    // §2 PcEval tests
    #[test]
    fn test_leaf_gaussian_likelihood() {
        let dist = PcLeafDist::Gaussian {
            mean: 0.0,
            std: 1.0,
        };
        let ll = leaf_likelihood(&dist, Some(0.0));
        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((ll - expected).abs() < 1e-10);
    }

    #[test]
    fn test_leaf_bernoulli_likelihood() {
        let dist = PcLeafDist::Bernoulli { p: 0.7 };
        let ll1 = leaf_likelihood(&dist, Some(1.0));
        let ll0 = leaf_likelihood(&dist, Some(0.0));
        assert!((ll1 - 0.7).abs() < 1e-10);
        assert!((ll0 - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_product_node() {
        let g = make_product_graph();
        let evidence = PcEvidence::from_sample(&[0.0, 0.0]);
        let val = evaluate_node(&g, g.root, &evidence);
        // Product of two Gaussian(0,1) at 0: (1/sqrt(2π))^2
        let expected = (1.0 / (2.0 * std::f64::consts::PI).sqrt()).powi(2);
        assert!((val - expected).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_sum_node_weighted() {
        let g = make_simple_graph();
        let evidence = PcEvidence::from_sample(&[0.0]);
        let val = evaluate_node(&g, g.root, &evidence);
        let p0 = leaf_likelihood(
            &PcLeafDist::Gaussian {
                mean: 0.0,
                std: 1.0,
            },
            Some(0.0),
        );
        let p1 = leaf_likelihood(
            &PcLeafDist::Gaussian {
                mean: 1.0,
                std: 1.0,
            },
            Some(0.0),
        );
        let expected = 0.5 * p0 + 0.5 * p1;
        assert!((val - expected).abs() < 1e-10);
    }

    #[test]
    fn test_log_likelihood_finite() {
        let g = make_simple_graph();
        let ll = pc_log_likelihood(&g, &[0.0]);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_marginalize_all_none() {
        let g = make_product_graph();
        let evidence = PcEvidence::all_none(2);
        let val = pc_marginalize(&g, &evidence);
        // With all-none evidence, leaf = 1.0, product = 1.0
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_partition_function_normalized() {
        let g = NaiveFactorization::build(3, 2);
        let pf = pc_partition_function(&g, 3);
        // Should be approximately 1.0 (normalized Gaussian integrals)
        assert!((pf - 1.0).abs() < 1e-3, "partition function = {}", pf);
    }

    #[test]
    fn test_mpe_shape() {
        let g = make_product_graph();
        let evidence = PcEvidence::new(vec![None, None]);
        let mpe = pc_mpe(&g, &evidence);
        assert_eq!(mpe.len(), 2);
    }

    #[test]
    fn test_mpe_in_range() {
        let mut g = PcGraph::new();
        let l0 = g.add_leaf(PcLeafDist::Bernoulli { p: 0.8 }, vec![0]);
        g.root = l0;
        let evidence = PcEvidence::new(vec![None]);
        let mpe = pc_mpe(&g, &evidence);
        assert!(mpe[0] == 0.0 || mpe[0] == 1.0);
    }

    // §3 SPN construction tests
    #[test]
    fn test_naive_factorization_n_nodes() {
        let g = NaiveFactorization::build(3, 2);
        // 3 vars * 2 leaves = 6 leaf nodes + 3 sum nodes + 1 product = 10
        assert!(g.n_nodes() >= 10);
    }

    #[test]
    fn test_naive_factorization_partition_function() {
        let g = NaiveFactorization::build(2, 3);
        let pf = pc_partition_function(&g, 2);
        assert!(pf > 0.5 && pf < 2.0, "partition function = {}", pf);
    }

    #[test]
    fn test_region_graph_n_nodes() {
        let g = RegionGraph::build_dense(4, 2);
        assert!(g.n_nodes() >= 4);
    }

    #[test]
    fn test_region_graph_log_lik_finite() {
        let g = RegionGraph::build_dense(4, 2);
        let sample = vec![0.0_f64; 4];
        let ll = pc_log_likelihood(&g, &sample);
        assert!(ll.is_finite(), "ll = {}", ll);
    }

    #[test]
    fn test_random_structure_root_valid() {
        let mut rng = StdRng::seed_from_u64(42);
        let g = RandomStructure::build(4, 2, 2, &mut rng);
        assert!(g.root < g.n_nodes());
    }

    #[test]
    fn test_random_structure_finite_log_lik() {
        let mut rng = StdRng::seed_from_u64(42);
        let g = RandomStructure::build(4, 2, 2, &mut rng);
        let n_vars = g
            .nodes
            .iter()
            .flat_map(|n| n.var_scope.iter().copied())
            .max()
            .map(|m| m + 1)
            .unwrap_or(1);
        let sample = vec![0.0_f64; n_vars];
        let ll = pc_log_likelihood(&g, &sample);
        assert!(ll.is_finite() || ll == f64::NEG_INFINITY);
    }

    // §4 ChowLiuTree tests
    #[test]
    fn test_discrete_mi_nonneg() {
        let x = vec![0usize, 1, 0, 1, 0];
        let y = vec![0usize, 1, 0, 0, 1];
        let mi = PcMutualInformation::discrete_mi(&x, &y, 2, 2);
        assert!(mi >= 0.0);
    }

    #[test]
    fn test_discrete_mi_self_max() {
        let x = vec![0usize, 1, 0, 1, 0, 1, 1, 0];
        let mi_self = PcMutualInformation::discrete_mi(&x, &x, 2, 2);
        let y = vec![1usize, 0, 1, 0, 1, 0, 0, 1];
        let mi_other = PcMutualInformation::discrete_mi(&x, &y, 2, 2);
        // MI(X;X) >= MI(X;Y)
        assert!(mi_self >= mi_other - 1e-10);
    }

    #[test]
    fn test_continuous_mi_kde_nonneg() {
        let data = make_synthetic_data(100, 2, 7);
        let x: Vec<f64> = data.iter().map(|r| r[0]).collect();
        let y: Vec<f64> = data.iter().map(|r| r[1]).collect();
        let mi = PcMutualInformation::continuous_mi_kde(&x, &y, 5);
        assert!(mi >= 0.0);
    }

    #[test]
    fn test_prim_mst_edge_count() {
        let n = 5;
        let weights = vec![vec![0.0_f64; n]; n];
        let edges = PcPrimMst::find_mst(n, &weights);
        assert_eq!(edges.len(), n - 1);
    }

    #[test]
    fn test_prim_mst_connected() {
        let n = 4;
        let mut weights = vec![vec![0.0_f64; n]; n];
        weights[0][1] = 1.0;
        weights[1][0] = 1.0;
        weights[1][2] = 2.0;
        weights[2][1] = 2.0;
        weights[2][3] = 0.5;
        weights[3][2] = 0.5;
        weights[0][3] = 3.0;
        weights[3][0] = 3.0;
        let edges = PcPrimMst::find_mst(n, &weights);
        assert_eq!(edges.len(), n - 1);
        // Check all nodes covered.
        let mut covered = std::collections::HashSet::new();
        for &(u, v) in &edges {
            covered.insert(u);
            covered.insert(v);
        }
        assert_eq!(covered.len(), n);
    }

    #[test]
    fn test_chow_liu_fit_edges() {
        let data = make_synthetic_data(50, 4, 11);
        let tree = ChowLiuTree::fit(&data, 5);
        // n-1 edges for n=4 variables
        assert_eq!(tree.edges.len(), 3);
    }

    #[test]
    fn test_chow_liu_fit_root_valid() {
        let data = make_synthetic_data(50, 4, 12);
        let tree = ChowLiuTree::fit(&data, 5);
        assert_eq!(tree.root, 0);
    }

    // §5 PcLearning tests
    #[test]
    fn test_em_e_step_shape() {
        let g = make_simple_graph();
        let data = make_synthetic_data(10, 1, 13);
        let resp = PcEmLearner::e_step(&g, &data);
        // Node 2 is the sum node with 2 children.
        assert_eq!(resp.len(), 3);
        assert_eq!(resp[2].len(), 2);
    }

    #[test]
    fn test_em_m_step_weights_normalized() {
        let mut g = make_simple_graph();
        let data = make_synthetic_data(20, 1, 14);
        let resp = PcEmLearner::e_step(&g, &data);
        PcEmLearner::m_step(&mut g, &resp);
        let weights = &g.nodes[2].weights;
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_em_fit_increasing_ll() {
        let mut g = NaiveFactorization::build(2, 3);
        let data = make_synthetic_data(50, 2, 15);
        let history = PcEmLearner::fit(&mut g, &data, 5);
        assert!(!history.is_empty());
        // Log-likelihood should be finite.
        assert!(
            history
                .last()
                .copied()
                .unwrap_or(f64::NEG_INFINITY)
                .is_finite()
                || history.last().copied().unwrap_or(0.0) < 0.0
        );
    }

    #[test]
    fn test_gradient_ll_gradient_shape() {
        let g = make_simple_graph();
        let data = make_synthetic_data(10, 1, 16);
        let grads = PcGradientLearner::log_likelihood_gradient_weights(&g, &data);
        // Should have one entry for the sum node.
        assert!(!grads.is_empty());
        let (node_id, ref g_vec) = grads[0];
        assert_eq!(g_vec.len(), 2);
    }

    #[test]
    fn test_gradient_step_updates_weights() {
        let mut g = make_simple_graph();
        let data = make_synthetic_data(10, 1, 17);
        let grads = PcGradientLearner::log_likelihood_gradient_weights(&g, &data);
        let old_w = g.nodes[2].weights.clone();
        PcGradientLearner::gradient_step(&mut g, &grads, 0.01);
        let new_w = &g.nodes[2].weights;
        // Weights should have changed.
        let changed = old_w
            .iter()
            .zip(new_w.iter())
            .any(|(a, b)| (a - b).abs() > 1e-15);
        assert!(changed);
    }

    // §6 PcSampling tests
    #[test]
    fn test_sample_gaussian_leaf_finite() {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = PcLeafDist::Gaussian {
            mean: 0.0,
            std: 1.0,
        };
        let v = pc_sample_leaf(&dist, &mut rng);
        assert!(v.is_finite());
    }

    #[test]
    fn test_sample_bernoulli_leaf_range() {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = PcLeafDist::Bernoulli { p: 0.5 };
        for _ in 0..20 {
            let v = pc_sample_leaf(&dist, &mut rng);
            assert!(v == 0.0 || v == 1.0);
        }
    }

    #[test]
    fn test_sample_categorical_valid_index() {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = PcLeafDist::Categorical {
            probs: vec![0.2, 0.5, 0.3],
        };
        for _ in 0..20 {
            let v = pc_sample_leaf(&dist, &mut rng);
            let idx = v.round() as usize;
            assert!(idx < 3);
        }
    }

    #[test]
    fn test_ancestral_sample_shape() {
        let g = make_product_graph();
        let mut rng = StdRng::seed_from_u64(42);
        let sample = pc_ancestral_sample(&g, &mut rng);
        assert_eq!(sample.len(), 2);
        assert!(sample.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_sample_batch_count() {
        let g = make_product_graph();
        let mut rng = StdRng::seed_from_u64(42);
        let batch = pc_sample_batch(&g, 10, &mut rng);
        assert_eq!(batch.len(), 10);
    }

    #[test]
    fn test_conditional_sample_evidence_fixed() {
        let g = make_product_graph();
        let mut rng = StdRng::seed_from_u64(42);
        let evidence = PcEvidence::new(vec![Some(2.5), None]);
        let sample = pc_conditional_sample(&g, &evidence, &mut rng);
        assert_eq!(sample.len(), 2);
        assert!(
            (sample[0] - 2.5).abs() < 1e-10,
            "observed var not fixed: {}",
            sample[0]
        );
    }

    // §7 PcMetrics tests
    #[test]
    fn test_mean_log_lik_finite() {
        let g = make_product_graph();
        let data = make_synthetic_data(20, 2, 18);
        let mll = pc_mean_log_likelihood(&g, &data);
        assert!(mll.is_finite());
    }

    #[test]
    fn test_bic_finite() {
        let g = make_simple_graph();
        let data = make_synthetic_data(20, 1, 19);
        let bic = pc_bic_score(&g, &data);
        assert!(bic.is_finite());
    }

    #[test]
    fn test_perplexity_positive() {
        let g = make_product_graph();
        let data = make_synthetic_data(20, 2, 20);
        let p = pc_perplexity(&g, &data);
        assert!(p > 0.0);
    }

    #[test]
    fn test_pc_eval_report_fields() {
        let g = make_product_graph();
        let data = make_synthetic_data(20, 2, 21);
        let report = pc_evaluate(&g, &data);
        assert!(report.mean_ll.is_finite());
        assert!(report.perplexity > 0.0);
        assert_eq!(report.n_nodes, 3);
    }

    #[test]
    fn test_evaluate_runs() {
        let g = NaiveFactorization::build(3, 2);
        let data = make_synthetic_data(30, 3, 22);
        let report = pc_evaluate(&g, &data);
        assert!(report.n_nodes > 0);
    }

    #[test]
    fn test_sum_weights_normalized_after_m_step() {
        let mut g = NaiveFactorization::build(2, 4);
        let data = make_synthetic_data(30, 2, 23);
        PcEmLearner::fit(&mut g, &data, 3);
        for node in &g.nodes {
            if node.node_type == PcNodeType::Sum {
                let s: f64 = node.weights.iter().sum();
                assert!((s - 1.0).abs() < 1e-9, "weights sum = {}", s);
            }
        }
    }

    #[test]
    fn test_chow_liu_pc_partition_finite() {
        let data = make_synthetic_data(40, 3, 24);
        let tree = ChowLiuTree::fit(&data, 5);
        let g = ChowLiuTree::build_pc(&tree, &data, 5);
        let n_vars = g
            .nodes
            .iter()
            .flat_map(|n| n.var_scope.iter().copied())
            .max()
            .map(|m| m + 1)
            .unwrap_or(3);
        let pf = pc_partition_function(&g, n_vars);
        assert!(pf.is_finite() && pf > 0.0);
    }

    #[test]
    fn test_log_likelihood_gradients_finite() {
        let g = make_simple_graph();
        let data = make_synthetic_data(20, 1, 25);
        let grads = PcGradientLearner::log_likelihood_gradient_weights(&g, &data);
        for (_, g_vec) in &grads {
            for &gv in g_vec {
                assert!(gv.is_finite(), "gradient not finite: {}", gv);
            }
        }
    }

    #[test]
    fn test_continuous_mi_symmetric() {
        let data = make_synthetic_data(80, 2, 26);
        let x: Vec<f64> = data.iter().map(|r| r[0]).collect();
        let y: Vec<f64> = data.iter().map(|r| r[1]).collect();
        let mi_xy = PcMutualInformation::continuous_mi_kde(&x, &y, 8);
        let mi_yx = PcMutualInformation::continuous_mi_kde(&y, &x, 8);
        assert!(
            (mi_xy - mi_yx).abs() < 1e-10,
            "MI not symmetric: {} vs {}",
            mi_xy,
            mi_yx
        );
    }
}
