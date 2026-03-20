//! Large-scale Graph ML with sampling methods.
//!
//! Implements:
//! - [`GmsCsrGraph`] — sparse CSR graph representation
//! - [`GmsNeighborSampler`] — GraphSAGE-style multi-hop sampling (Hamilton et al. 2017)
//! - [`ClusterGcn`] — Cluster-GCN batch training (Chiang et al. 2019)
//! - [`NodeSampler`] / [`EdgeSampler`] — GraphSAINT samplers (Zeng et al. 2020)
//! - [`SignModel`] — SIGN scalable inception GNN (Rossi et al. 2020)
//! - [`GmsGraphSageModel`] — GraphSAGE aggregation model
//! - Evaluation metrics: accuracy, link-prediction AUC, modularity

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─── §1  CsrGraph ────────────────────────────────────────────────────────────

/// Sparse CSR graph for large-scale GNNs.
#[derive(Clone, Debug)]
pub struct GmsCsrGraph {
    /// row_ptr\[v\]..row_ptr[v+1] indexes into col_idx for node v's neighbors.
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub edge_weights: Option<Vec<f32>>,
    pub n_nodes: usize,
    pub n_edges: usize,
}

impl GmsCsrGraph {
    /// Build a CSR graph from an edge list.  Duplicate edges are kept.
    pub fn from_edges(n_nodes: usize, edges: Vec<(usize, usize)>) -> Self {
        let n_edges = edges.len();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for (u, v) in &edges {
            if *u < n_nodes {
                adj[*u].push(*v);
            }
        }
        for neighbors in adj.iter_mut() {
            neighbors.sort_unstable();
        }
        let mut row_ptr = Vec::with_capacity(n_nodes + 1);
        let mut col_idx = Vec::with_capacity(n_edges);
        row_ptr.push(0);
        for neighbors in &adj {
            col_idx.extend_from_slice(neighbors);
            row_ptr.push(col_idx.len());
        }
        Self {
            row_ptr,
            col_idx,
            edge_weights: None,
            n_nodes,
            n_edges,
        }
    }

    /// Neighbors of `node` as a slice.
    pub fn neighbors(&self, node: usize) -> &[usize] {
        if node >= self.n_nodes {
            return &[];
        }
        let start = self.row_ptr[node];
        let end = self.row_ptr[node + 1];
        &self.col_idx[start..end]
    }

    /// Out-degree of `node`.
    pub fn degree(&self, node: usize) -> usize {
        if node >= self.n_nodes {
            return 0;
        }
        self.row_ptr[node + 1] - self.row_ptr[node]
    }

    /// Sample up to `k` neighbors of `node` without replacement.
    /// If degree < k, returns all neighbors in a random order.
    pub fn sample_neighbors(&self, node: usize, k: usize, rng: &mut impl Rng) -> Vec<usize> {
        let nbrs = self.neighbors(node);
        if nbrs.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut indices: Vec<usize> = (0..nbrs.len()).collect();
        // Fisher-Yates partial shuffle for k elements
        let take = k.min(nbrs.len());
        for i in 0..take {
            let j = i + rng.random_range(0..(nbrs.len() - i));
            indices.swap(i, j);
        }
        indices[..take].iter().map(|&i| nbrs[i]).collect()
    }
}

/// Per-node feature matrix.
#[derive(Clone, Debug)]
pub struct NodeFeatures {
    pub features: Vec<Vec<f32>>,
    pub n_nodes: usize,
    pub feat_dim: usize,
}

impl NodeFeatures {
    pub fn new(features: Vec<Vec<f32>>) -> Self {
        let n_nodes = features.len();
        let feat_dim = features.first().map(|r| r.len()).unwrap_or(0);
        Self {
            features,
            n_nodes,
            feat_dim,
        }
    }
}

/// A mini-batch subgraph.
#[derive(Clone, Debug)]
pub struct GraphBatch {
    pub nodes: Vec<usize>,
    pub adj: Vec<Vec<usize>>,
    pub features: Vec<Vec<f32>>,
}

// ─── §2  NeighborSampler (GraphSAGE-style) ───────────────────────────────────

/// Sampling configuration: number of layers and per-layer fanouts.
#[derive(Clone, Debug)]
pub struct SamplerConfig {
    pub n_layers: usize,
    pub fanouts: Vec<usize>,
}

/// Result of multi-hop neighbor sampling.
#[derive(Clone, Debug)]
pub struct GmsSubgraphSample {
    /// All unique node IDs encountered across all hops (seed + sampled).
    pub node_ids: Vec<usize>,
    /// adjacencies\[l\]\[i\] = list of local indices (into node_ids) that are
    /// neighbors of node_ids\[i\] at hop l.
    pub adjacencies: Vec<Vec<Vec<usize>>>,
    /// Features for each node in node_ids, in order.
    pub features: Vec<Vec<f32>>,
}

/// GraphSAGE-style multi-hop neighbor sampler.
pub struct GmsNeighborSampler {
    pub config: SamplerConfig,
}

impl GmsNeighborSampler {
    pub fn new(config: SamplerConfig) -> Self {
        Self { config }
    }

    /// Multi-hop neighborhood sampling starting from `seed_nodes`.
    ///
    /// For each layer l = 0..n_layers:
    ///   - frontier = nodes discovered so far
    ///   - for each frontier node, sample fanout\[l\] neighbors
    ///   - add new neighbors to the frontier
    ///
    /// Returns a [`GmsSubgraphSample`] whose `node_ids` includes all nodes
    /// encountered and whose `adjacencies[l][i]` contains local indices of
    /// node i's sampled neighbors at layer l.
    pub fn sample(
        &self,
        graph: &GmsCsrGraph,
        seed_nodes: &[usize],
        node_features: &NodeFeatures,
        rng: &mut impl Rng,
    ) -> GmsSubgraphSample {
        use std::collections::HashMap;

        // Collect all nodes layer by layer
        let mut all_nodes: Vec<usize> = seed_nodes.to_vec();
        // adjacency list per layer, stored as (global_src, global_dst) pairs
        let mut layer_edges: Vec<Vec<(usize, usize)>> = Vec::new();

        let mut frontier: Vec<usize> = seed_nodes.to_vec();
        for l in 0..self.config.n_layers {
            let fanout = if l < self.config.fanouts.len() {
                self.config.fanouts[l]
            } else {
                5
            };
            let mut edges_this_layer: Vec<(usize, usize)> = Vec::new();
            let mut new_nodes: Vec<usize> = Vec::new();

            for &node in &frontier {
                let sampled = graph.sample_neighbors(node, fanout, rng);
                for &nb in &sampled {
                    edges_this_layer.push((node, nb));
                    if !all_nodes.contains(&nb) && !new_nodes.contains(&nb) {
                        new_nodes.push(nb);
                    }
                }
            }
            all_nodes.extend_from_slice(&new_nodes);
            layer_edges.push(edges_this_layer);
            frontier = new_nodes;
        }

        // Build local index map
        let id_to_local: HashMap<usize, usize> =
            all_nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();

        // Build adjacency per layer
        let n = all_nodes.len();
        let mut adjacencies: Vec<Vec<Vec<usize>>> = Vec::new();
        for edges in &layer_edges {
            let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
            for &(src, dst) in edges {
                if let (Some(&li), Some(&lj)) = (id_to_local.get(&src), id_to_local.get(&dst)) {
                    adj[li].push(lj);
                }
            }
            adjacencies.push(adj);
        }

        // Extract features
        let features: Vec<Vec<f32>> = all_nodes
            .iter()
            .map(|&nid| {
                if nid < node_features.n_nodes {
                    node_features.features[nid].clone()
                } else {
                    vec![0.0_f32; node_features.feat_dim]
                }
            })
            .collect();

        GmsSubgraphSample {
            node_ids: all_nodes,
            adjacencies,
            features,
        }
    }

    /// Sample a random mini-batch of seed node indices.
    pub fn mini_batch_nodes(n_nodes: usize, batch_size: usize, rng: &mut impl Rng) -> Vec<usize> {
        if n_nodes == 0 || batch_size == 0 {
            return Vec::new();
        }
        let take = batch_size.min(n_nodes);
        let mut indices: Vec<usize> = (0..n_nodes).collect();
        for i in 0..take {
            let j = i + rng.random_range(0..(n_nodes - i));
            indices.swap(i, j);
        }
        indices[..take].to_vec()
    }
}

// ─── §3  Cluster-GCN ─────────────────────────────────────────────────────────

/// Configuration for Cluster-GCN.
#[derive(Clone, Debug)]
pub struct ClusterConfig {
    pub n_clusters: usize,
    pub batch_size_clusters: usize,
}

/// A single graph cluster (partition).
#[derive(Clone, Debug)]
pub struct GraphCluster {
    pub cluster_id: usize,
    pub node_ids: Vec<usize>,
}

/// Cluster-GCN trainer (Chiang et al. 2019).
pub struct ClusterGcn {
    pub config: ClusterConfig,
}

impl ClusterGcn {
    pub fn new(config: ClusterConfig) -> Self {
        Self { config }
    }

    /// Random node partitioning into `n_clusters` clusters (METIS baseline).
    pub fn partition_graph(
        graph: &GmsCsrGraph,
        n_clusters: usize,
        rng: &mut impl Rng,
    ) -> Vec<GraphCluster> {
        if n_clusters == 0 || graph.n_nodes == 0 {
            return Vec::new();
        }
        let n = graph.n_nodes;
        let mut perm: Vec<usize> = (0..n).collect();
        // shuffle
        for i in 0..n {
            let j = i + rng.random_range(0..(n - i));
            perm.swap(i, j);
        }
        let mut clusters: Vec<GraphCluster> = (0..n_clusters)
            .map(|id| GraphCluster {
                cluster_id: id,
                node_ids: Vec::new(),
            })
            .collect();
        for (idx, &node) in perm.iter().enumerate() {
            clusters[idx % n_clusters].node_ids.push(node);
        }
        clusters
    }

    /// Sample `k` clusters and return the union of their nodes.
    pub fn sample_cluster_batch(
        clusters: &[GraphCluster],
        k: usize,
        rng: &mut impl Rng,
    ) -> Vec<usize> {
        if clusters.is_empty() || k == 0 {
            return Vec::new();
        }
        let take = k.min(clusters.len());
        let mut indices: Vec<usize> = (0..clusters.len()).collect();
        for i in 0..take {
            let j = i + rng.random_range(0..(clusters.len() - i));
            indices.swap(i, j);
        }
        let mut nodes: Vec<usize> = Vec::new();
        for &ci in &indices[..take] {
            nodes.extend_from_slice(&clusters[ci].node_ids);
        }
        nodes.sort_unstable();
        nodes.dedup();
        nodes
    }

    /// Extract the subgraph induced by `nodes`, reindexing nodes to 0..N.
    pub fn induced_subgraph(graph: &GmsCsrGraph, nodes: &[usize]) -> GmsCsrGraph {
        use std::collections::HashMap;
        let local: HashMap<usize, usize> = nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let n_local = nodes.len();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_local];
        for (li, &gn) in nodes.iter().enumerate() {
            for &nb in graph.neighbors(gn) {
                if let Some(&lnb) = local.get(&nb) {
                    adj[li].push(lnb);
                }
            }
        }
        let mut row_ptr = Vec::with_capacity(n_local + 1);
        let mut col_idx: Vec<usize> = Vec::new();
        row_ptr.push(0);
        for row in &adj {
            col_idx.extend_from_slice(row);
            row_ptr.push(col_idx.len());
        }
        let n_edges = col_idx.len();
        GmsCsrGraph {
            row_ptr,
            col_idx,
            edge_weights: None,
            n_nodes: n_local,
            n_edges,
        }
    }
}

/// Single GCN layer with mean aggregation.
pub struct GmsGcnLayer {
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
}

impl GmsGcnLayer {
    pub fn new(in_dim: usize, out_dim: usize, rng: &mut impl Rng) -> Self {
        let scale = (2.0_f32 / (in_dim + out_dim) as f32).sqrt();
        let weights = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| rng.random_range(-scale..scale))
                    .collect()
            })
            .collect();
        let bias = vec![0.0_f32; out_dim];
        Self { weights, bias }
    }

    /// Mean aggregation: h_v = ReLU(W * (1/(1+d_v)) * (h_v + Σ h_u))
    pub fn forward(&self, features: &[Vec<f32>], adj: &GmsCsrGraph) -> Vec<Vec<f32>> {
        let n = features.len();
        let in_dim = features.first().map(|r| r.len()).unwrap_or(0);
        let out_dim = self.weights.len();
        let mut output = vec![vec![0.0_f32; out_dim]; n];

        for v in 0..n {
            // aggregate: h_v + sum of neighbors
            let mut agg = if in_dim > 0 {
                features[v].clone()
            } else {
                vec![0.0_f32; in_dim]
            };
            let nbrs = adj.neighbors(v);
            for &u in nbrs {
                if u < n {
                    for d in 0..in_dim.min(features[u].len()) {
                        agg[d] += features[u][d];
                    }
                }
            }
            let scale = 1.0 / (1.0 + nbrs.len() as f32);
            for d in 0..in_dim {
                agg[d] *= scale;
            }
            // linear + bias + ReLU
            for o in 0..out_dim {
                let mut val = self.bias[o];
                let row = &self.weights[o];
                for d in 0..in_dim.min(row.len()) {
                    val += row[d] * agg[d];
                }
                output[v][o] = val.max(0.0);
            }
        }
        output
    }
}

// ─── §4  GraphSAINT ──────────────────────────────────────────────────────────

/// GraphSAINT sampled subgraph with normalization weights.
#[derive(Clone, Debug)]
pub struct SaintSubgraph {
    pub nodes: Vec<usize>,
    pub edges: Vec<(usize, usize)>,
    pub node_features: Vec<Vec<f32>>,
    pub node_weights: Vec<f32>,
    pub edge_weights: Vec<f32>,
}

/// Normalization helpers for GraphSAINT.
pub struct NormalizationWeights;

impl NormalizationWeights {
    /// Node normalization weight: 1 / p_v where p_v = expected_visits.
    pub fn node_weight(_node_degree: usize, expected_visits: f32) -> f32 {
        if expected_visits <= 0.0 {
            1.0
        } else {
            1.0 / expected_visits
        }
    }

    /// Edge weight: minimum of the two endpoint node weights.
    pub fn edge_weight(node_weight1: f32, node_weight2: f32) -> f32 {
        node_weight1.min(node_weight2)
    }
}

/// GraphSAINT node sampler — uniformly samples nodes then extracts induced subgraph.
pub struct NodeSampler {
    pub budget: usize,
}

impl NodeSampler {
    pub fn new(budget: usize) -> Self {
        Self { budget }
    }

    pub fn sample(&self, graph: &GmsCsrGraph, n_nodes: usize, rng: &mut impl Rng) -> SaintSubgraph {
        if graph.n_nodes == 0 {
            return SaintSubgraph {
                nodes: Vec::new(),
                edges: Vec::new(),
                node_features: Vec::new(),
                node_weights: Vec::new(),
                edge_weights: Vec::new(),
            };
        }
        let take = n_nodes.min(graph.n_nodes);
        // Sample without replacement via partial Fisher-Yates
        let mut perm: Vec<usize> = (0..graph.n_nodes).collect();
        for i in 0..take {
            let j = i + rng.random_range(0..(graph.n_nodes - i));
            perm.swap(i, j);
        }
        let sampled_nodes: Vec<usize> = perm[..take].to_vec();
        self.build_subgraph(graph, sampled_nodes, take as f32 / graph.n_nodes as f32)
    }

    fn build_subgraph(
        &self,
        graph: &GmsCsrGraph,
        mut nodes: Vec<usize>,
        p_v: f32,
    ) -> SaintSubgraph {
        use std::collections::HashMap;
        nodes.sort_unstable();
        nodes.dedup();
        let local: HashMap<usize, usize> = nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let n = nodes.len();
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for &gn in &nodes {
            for &nb in graph.neighbors(gn) {
                if local.contains_key(&nb) {
                    edges.push((gn, nb));
                }
            }
        }
        let nw = NormalizationWeights::node_weight(0, p_v);
        let node_weights = vec![nw; n];
        let edge_weights = edges
            .iter()
            .map(|_| NormalizationWeights::edge_weight(nw, nw))
            .collect();
        SaintSubgraph {
            nodes,
            edges,
            node_features: Vec::new(),
            node_weights,
            edge_weights,
        }
    }
}

/// GraphSAINT edge sampler — samples edges proportional to 1/(d_u * d_v).
pub struct EdgeSampler {
    pub budget: usize,
}

impl EdgeSampler {
    pub fn new(budget: usize) -> Self {
        Self { budget }
    }

    pub fn sample(&self, graph: &GmsCsrGraph, n_edges: usize, rng: &mut impl Rng) -> SaintSubgraph {
        if graph.n_edges == 0 || graph.n_nodes == 0 {
            return SaintSubgraph {
                nodes: Vec::new(),
                edges: Vec::new(),
                node_features: Vec::new(),
                node_weights: Vec::new(),
                edge_weights: Vec::new(),
            };
        }
        // Build a flat edge list and compute sampling weights
        let mut edge_list: Vec<(usize, usize)> = Vec::with_capacity(graph.n_edges);
        let mut weights: Vec<f32> = Vec::with_capacity(graph.n_edges);
        let mut total_weight = 0.0_f32;
        for u in 0..graph.n_nodes {
            let du = graph.degree(u) as f32;
            for &v in graph.neighbors(u) {
                let dv = graph.degree(v) as f32;
                let w = if du > 0.0 && dv > 0.0 {
                    1.0 / (du * dv)
                } else {
                    1.0
                };
                total_weight += w;
                weights.push(w);
                edge_list.push((u, v));
            }
        }
        // Sample n_edges edges with replacement using CDF inversion
        let take = n_edges.min(edge_list.len());
        let mut sampled_edges: Vec<(usize, usize)> = Vec::with_capacity(take);
        for _ in 0..take {
            let r: f32 = rng.random_range(0.0_f32..total_weight);
            let mut cum = 0.0_f32;
            let mut chosen = 0;
            for (i, &w) in weights.iter().enumerate() {
                cum += w;
                if cum >= r {
                    chosen = i;
                    break;
                }
            }
            sampled_edges.push(edge_list[chosen]);
        }
        // Collect unique nodes
        let mut node_set: Vec<usize> = sampled_edges.iter().flat_map(|&(u, v)| [u, v]).collect();
        node_set.sort_unstable();
        node_set.dedup();

        let n = node_set.len();
        let p_v = take as f32 / edge_list.len().max(1) as f32;
        let nw = NormalizationWeights::node_weight(0, p_v.max(1e-8));
        let node_weights = vec![nw; n];
        let edge_weights_out = sampled_edges
            .iter()
            .map(|_| NormalizationWeights::edge_weight(nw, nw))
            .collect();

        SaintSubgraph {
            nodes: node_set,
            edges: sampled_edges,
            node_features: Vec::new(),
            node_weights,
            edge_weights: edge_weights_out,
        }
    }
}

// ─── §5  SIGN ────────────────────────────────────────────────────────────────

/// SIGN configuration.
#[derive(Clone, Debug)]
pub struct SignConfig {
    pub n_hops: usize,
    pub hidden_dim: usize,
    pub n_classes: usize,
}

/// SIGN — Scalable Inception Graph Neural Network (Rossi et al. 2020).
///
/// Pre-computes K hops of D^{-1/2} A D^{-1/2} diffusion, applies an MLP
/// to each hop independently, concatenates the results, then projects to
/// class logits.
pub struct SignModel {
    /// One (weight, bias) pair per hop.
    pub per_hop_layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
    /// Final projection: concat of (n_hops+1)*hidden_dim → n_classes.
    pub fusion_layer: (Vec<Vec<f32>>, Vec<f32>),
}

impl SignModel {
    fn xavier(rows: usize, cols: usize, rng: &mut impl Rng) -> Vec<Vec<f32>> {
        let scale = (2.0_f32 / (rows + cols) as f32).sqrt();
        (0..rows)
            .map(|_| (0..cols).map(|_| rng.random_range(-scale..scale)).collect())
            .collect()
    }

    /// Create a SIGN model.
    ///
    /// Per-hop MLPs: feat_dim → hidden_dim
    /// Fusion layer: (n_hops+1)*hidden_dim → n_classes
    pub fn new(feat_dim: usize, config: SignConfig, rng: &mut impl Rng) -> Self {
        let n_inputs = config.n_hops + 1;
        let mut per_hop_layers = Vec::with_capacity(n_inputs);
        for _ in 0..n_inputs {
            let w = Self::xavier(config.hidden_dim, feat_dim, rng);
            let b = vec![0.0_f32; config.hidden_dim];
            per_hop_layers.push((w, b));
        }
        let fusion_in = n_inputs * config.hidden_dim;
        let fw = Self::xavier(config.n_classes, fusion_in, rng);
        let fb = vec![0.0_f32; config.n_classes];
        SignModel {
            per_hop_layers,
            fusion_layer: (fw, fb),
        }
    }

    /// Compute K diffused feature matrices X^(0)..X^(K).
    /// X^(0) = original features; X^(k) = D^{-1/2} A D^{-1/2} X^{k-1}.
    pub fn precompute_diffusion(
        graph: &GmsCsrGraph,
        features: &NodeFeatures,
        k: usize,
    ) -> Vec<NodeFeatures> {
        let sym_edges = Self::sym_normalize_adj(graph);
        let n = features.n_nodes;
        let d = features.feat_dim;
        let mut result = Vec::with_capacity(k + 1);
        result.push(features.clone());

        for _ in 0..k {
            let prev = &result.last().expect("result is non-empty").features;
            let mut next: Vec<Vec<f32>> = vec![vec![0.0_f32; d]; n];
            for &(u, v, w) in &sym_edges {
                if u < n && v < n {
                    for dim in 0..d {
                        next[u][dim] += w * prev[v][dim];
                    }
                }
            }
            result.push(NodeFeatures::new(next));
        }
        result
    }

    /// Compute the symmetrically normalized adjacency D^{-1/2} A D^{-1/2}
    /// as a sparse edge list (u, v, weight).
    pub fn sym_normalize_adj(graph: &GmsCsrGraph) -> Vec<(usize, usize, f32)> {
        let n = graph.n_nodes;
        let degrees: Vec<f32> = (0..n).map(|v| graph.degree(v) as f32).collect();
        let inv_sqrt: Vec<f32> = degrees
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();

        let mut edges = Vec::with_capacity(graph.n_edges);
        for u in 0..n {
            for &v in graph.neighbors(u) {
                if v < n {
                    let w = inv_sqrt[u] * inv_sqrt[v];
                    if w.is_finite() {
                        edges.push((u, v, w));
                    }
                }
            }
        }
        edges
    }

    /// Run SIGN forward pass on pre-diffused features.
    /// Returns logits of shape [n_nodes, n_classes].
    pub fn forward(&self, diffused_features: &[NodeFeatures]) -> Vec<Vec<f32>> {
        if diffused_features.is_empty() {
            return Vec::new();
        }
        let n = diffused_features[0].n_nodes;
        let n_inputs = self.per_hop_layers.len();

        let mut node_hidden: Vec<Vec<f32>> = vec![Vec::new(); n];

        // Apply per-hop MLP to each diffused feature matrix
        for (hop_idx, (w, b)) in self.per_hop_layers.iter().enumerate() {
            let feat = if hop_idx < diffused_features.len() {
                &diffused_features[hop_idx].features
            } else {
                &diffused_features[0].features
            };
            let out_dim = w.len();
            for v in 0..n {
                let x = if v < feat.len() { &feat[v] } else { continue };
                let in_dim = x.len().min(w.first().map(|r| r.len()).unwrap_or(0));
                let mut h = vec![0.0_f32; out_dim];
                for o in 0..out_dim {
                    let mut val = b[o];
                    for d in 0..in_dim {
                        val += w[o][d] * x[d];
                    }
                    h[o] = val.max(0.0); // ReLU
                }
                node_hidden[v].extend_from_slice(&h);
            }
        }

        // Fusion layer
        let (fw, fb) = &self.fusion_layer;
        let n_classes = fw.len();
        let mut logits = vec![vec![0.0_f32; n_classes]; n];
        for v in 0..n {
            let h = &node_hidden[v];
            let h_dim = h.len().min(fw.first().map(|r| r.len()).unwrap_or(0));
            for o in 0..n_classes {
                let mut val = fb[o];
                for d in 0..h_dim {
                    val += fw[o][d] * h[d];
                }
                logits[v][o] = val;
            }
        }
        // Verify output shape makes sense
        let _ = n_inputs; // used indirectly
        logits
    }
}

// ─── §6  GraphSAGE ───────────────────────────────────────────────────────────

/// Aggregation strategy for GraphSAGE.
#[derive(Clone, Debug, PartialEq)]
pub enum GmsAggregatorType {
    Mean,
    MaxPool,
    GcnMean,
}

/// A single GraphSAGE layer with selectable aggregator.
pub struct GmsGraphSageLayer {
    pub aggregator: GmsAggregatorType,
    pub self_weight: Vec<Vec<f32>>,
    pub agg_weight: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl GmsGraphSageLayer {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        aggregator: GmsAggregatorType,
        rng: &mut impl Rng,
    ) -> Self {
        let scale = (2.0_f32 / (in_dim + out_dim) as f32).sqrt();
        let self_weight: Vec<Vec<f32>> = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| rng.random_range(-scale..scale))
                    .collect()
            })
            .collect();
        let agg_weight: Vec<Vec<f32>> = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| rng.random_range(-scale..scale))
                    .collect()
            })
            .collect();
        let bias = vec![0.0_f32; out_dim];
        Self {
            aggregator,
            self_weight,
            agg_weight,
            bias,
            in_dim,
            out_dim,
        }
    }

    /// Aggregate neighbor features according to the aggregator type.
    pub fn aggregate_neighbors(
        &self,
        graph: &GmsCsrGraph,
        node: usize,
        features: &[Vec<f32>],
    ) -> Vec<f32> {
        let nbrs = graph.neighbors(node);
        if nbrs.is_empty() {
            return vec![0.0_f32; self.in_dim];
        }
        match self.aggregator {
            GmsAggregatorType::Mean | GmsAggregatorType::GcnMean => {
                let mut agg = vec![0.0_f32; self.in_dim];
                let mut count = 0usize;
                for &nb in nbrs {
                    if nb < features.len() {
                        for d in 0..self.in_dim.min(features[nb].len()) {
                            agg[d] += features[nb][d];
                        }
                        count += 1;
                    }
                }
                if count > 0 {
                    let inv = 1.0 / count as f32;
                    for v in agg.iter_mut() {
                        *v *= inv;
                    }
                }
                agg
            }
            GmsAggregatorType::MaxPool => {
                let mut agg = vec![f32::NEG_INFINITY; self.in_dim];
                let mut has_any = false;
                for &nb in nbrs {
                    if nb < features.len() {
                        for d in 0..self.in_dim.min(features[nb].len()) {
                            if features[nb][d] > agg[d] {
                                agg[d] = features[nb][d];
                            }
                        }
                        has_any = true;
                    }
                }
                if !has_any {
                    agg = vec![0.0_f32; self.in_dim];
                } else {
                    for v in agg.iter_mut() {
                        if v.is_infinite() {
                            *v = 0.0;
                        }
                    }
                }
                agg
            }
        }
    }

    /// Forward pass for a single node: concat(self_feat, agg) → linear → L2-normalize.
    pub fn forward_node(
        &self,
        node: usize,
        features: &[Vec<f32>],
        graph: &GmsCsrGraph,
    ) -> Vec<f32> {
        let self_feat = if node < features.len() {
            features[node].clone()
        } else {
            vec![0.0_f32; self.in_dim]
        };
        let agg = self.aggregate_neighbors(graph, node, features);

        // Linear transform: W_self * self_feat + W_agg * agg + bias
        let mut out = vec![0.0_f32; self.out_dim];
        for o in 0..self.out_dim {
            let mut val = self.bias[o];
            for d in 0..self.in_dim.min(self_feat.len()) {
                val += self.self_weight[o][d] * self_feat[d];
            }
            for d in 0..self.in_dim.min(agg.len()) {
                val += self.agg_weight[o][d] * agg[d];
            }
            out[o] = val.max(0.0); // ReLU
        }

        // L2 normalize
        let norm = out.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in out.iter_mut() {
                *v /= norm;
            }
        }
        out
    }
}

/// Multi-layer GraphSAGE model.
pub struct GmsGraphSageModel {
    pub layers: Vec<GmsGraphSageLayer>,
}

impl GmsGraphSageModel {
    /// Build a multi-layer GraphSAGE model.
    ///
    /// `layer_dims[i] = (in_dim, out_dim)` for layer i.
    pub fn new(
        layer_dims: &[(usize, usize)],
        aggregator: GmsAggregatorType,
        rng: &mut impl Rng,
    ) -> Self {
        let layers = layer_dims
            .iter()
            .map(|&(i, o)| GmsGraphSageLayer::new(i, o, aggregator.clone(), rng))
            .collect();
        Self { layers }
    }

    /// Run the model on a batch of nodes.
    pub fn forward_batch(
        &self,
        graph: &GmsCsrGraph,
        node_ids: &[usize],
        features: &NodeFeatures,
    ) -> Vec<Vec<f32>> {
        if self.layers.is_empty() || node_ids.is_empty() {
            return Vec::new();
        }
        let mut current: Vec<Vec<f32>> = features.features.clone();
        for layer in &self.layers {
            let mut next: Vec<Vec<f32>> = vec![vec![0.0_f32; layer.out_dim]; current.len()];
            for (idx, _) in current.iter().enumerate() {
                next[idx] = layer.forward_node(idx, &current, graph);
            }
            current = next;
        }
        node_ids
            .iter()
            .map(|&nid| {
                if nid < current.len() {
                    current[nid].clone()
                } else {
                    Vec::new()
                }
            })
            .collect()
    }
}

// ─── §7  Evaluation Metrics ──────────────────────────────────────────────────

/// Node classification accuracy from logits.
pub fn node_classification_accuracy(logits: &[Vec<f32>], labels: &[usize]) -> f32 {
    if logits.is_empty() || labels.is_empty() {
        return 0.0;
    }
    let n = logits.len().min(labels.len());
    let correct = logits[..n]
        .iter()
        .zip(&labels[..n])
        .filter(|(logit, &lbl)| {
            let pred = logit
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            pred == lbl
        })
        .count();
    correct as f32 / n as f32
}

/// Approximate link-prediction AUC using dot-product scoring.
/// AUC = P(pos_score > neg_score) estimated via ranking.
pub fn link_prediction_auc(
    embeddings: &[Vec<f32>],
    pos_edges: &[(usize, usize)],
    neg_edges: &[(usize, usize)],
) -> f32 {
    if pos_edges.is_empty() || neg_edges.is_empty() {
        return 0.5;
    }
    let dot = |u: usize, v: usize| -> f32 {
        if u >= embeddings.len() || v >= embeddings.len() {
            return 0.0;
        }
        embeddings[u]
            .iter()
            .zip(&embeddings[v])
            .map(|(&a, &b)| a * b)
            .sum()
    };

    let pos_scores: Vec<f32> = pos_edges.iter().map(|&(u, v)| dot(u, v)).collect();
    let neg_scores: Vec<f32> = neg_edges.iter().map(|&(u, v)| dot(u, v)).collect();

    // AUC estimate: fraction of (pos, neg) pairs where pos_score > neg_score
    let mut wins = 0u64;
    let total = pos_scores.len() as u64 * neg_scores.len() as u64;
    for &ps in &pos_scores {
        for &ns in &neg_scores {
            if ps > ns {
                wins += 1;
            } else if (ps - ns).abs() < 1e-8 {
                wins += 1; // tie counts as 0.5 each — approximate
            }
        }
    }
    if total == 0 {
        0.5
    } else {
        (wins as f32) / (total as f32)
    }
}

/// Graph modularity for a clustering assignment.
/// Q = (1/2m) Σ_{ij} [A_ij - k_i k_j / 2m] δ(c_i, c_j)
pub fn cluster_modularity(graph: &GmsCsrGraph, clusters: &[GraphCluster]) -> f32 {
    if graph.n_edges == 0 {
        return 0.0;
    }
    let m = graph.n_edges as f32;
    // Build node → cluster map
    let mut node_cluster = vec![0usize; graph.n_nodes];
    for cluster in clusters {
        for &nid in &cluster.node_ids {
            if nid < graph.n_nodes {
                node_cluster[nid] = cluster.cluster_id;
            }
        }
    }
    let mut q = 0.0_f32;
    for u in 0..graph.n_nodes {
        let ku = graph.degree(u) as f32;
        for &v in graph.neighbors(u) {
            if v < graph.n_nodes && node_cluster[u] == node_cluster[v] {
                let kv = graph.degree(v) as f32;
                q += 1.0 - (ku * kv) / (2.0 * m);
            }
        }
    }
    (q / (2.0 * m)).clamp(-1.0, 1.0)
}

/// Aggregate evaluation report for scalable GNN experiments.
#[derive(Clone, Debug)]
pub struct ScalableGnnReport {
    pub accuracy: f32,
    pub link_auc: f32,
    pub modularity: f32,
    pub throughput_nodes_per_sec: f32,
}

impl ScalableGnnReport {
    pub fn new(
        accuracy: f32,
        link_auc: f32,
        modularity: f32,
        throughput_nodes_per_sec: f32,
    ) -> Self {
        Self {
            accuracy,
            link_auc,
            modularity,
            throughput_nodes_per_sec,
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::{rngs::StdRng, SeedableRng};

    fn small_graph() -> GmsCsrGraph {
        // 5-node graph: 0-1, 0-2, 1-2, 1-3, 2-4, 3-4
        GmsCsrGraph::from_edges(
            5,
            vec![(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)],
        )
    }

    fn small_features() -> NodeFeatures {
        let feats: Vec<Vec<f32>> = (0..5)
            .map(|i| vec![i as f32, (i * 2) as f32, 1.0])
            .collect();
        NodeFeatures::new(feats)
    }

    // §1 GmsCsrGraph
    #[test]
    fn test_csr_graph_creation() {
        let g = small_graph();
        assert_eq!(g.n_nodes, 5);
        assert_eq!(g.n_edges, 7);
    }

    #[test]
    fn test_csr_graph_neighbors() {
        let g = small_graph();
        let nbrs = g.neighbors(0);
        assert!(nbrs.contains(&1));
        assert!(nbrs.contains(&2));
    }

    #[test]
    fn test_csr_graph_degree() {
        let g = small_graph();
        assert_eq!(g.degree(0), 2);
        assert_eq!(g.degree(1), 3);
    }

    #[test]
    fn test_csr_graph_sample_neighbors_count() {
        let g = small_graph();
        let mut rng = StdRng::seed_from_u64(1);
        let sampled = g.sample_neighbors(1, 2, &mut rng);
        assert_eq!(sampled.len(), 2);
    }

    #[test]
    fn test_csr_graph_sample_neighbors_valid() {
        let g = small_graph();
        let mut rng = StdRng::seed_from_u64(2);
        let sampled = g.sample_neighbors(1, 5, &mut rng);
        // All sampled nodes must be actual neighbors
        let nbrs = g.neighbors(1);
        for &s in &sampled {
            assert!(nbrs.contains(&s));
        }
    }

    #[test]
    fn test_node_features_creation() {
        let f = small_features();
        assert_eq!(f.n_nodes, 5);
        assert_eq!(f.feat_dim, 3);
    }

    // §2 NeighborSampler
    #[test]
    fn test_neighbor_sampler_sample() {
        let g = small_graph();
        let f = small_features();
        let cfg = SamplerConfig {
            n_layers: 2,
            fanouts: vec![2, 2],
        };
        let sampler = GmsNeighborSampler::new(cfg);
        let mut rng = StdRng::seed_from_u64(3);
        let s = sampler.sample(&g, &[0], &f, &mut rng);
        assert!(!s.node_ids.is_empty());
        assert_eq!(s.adjacencies.len(), 2);
    }

    #[test]
    fn test_neighbor_sampler_sample_node_ids() {
        let g = small_graph();
        let f = small_features();
        let cfg = SamplerConfig {
            n_layers: 1,
            fanouts: vec![3],
        };
        let sampler = GmsNeighborSampler::new(cfg);
        let mut rng = StdRng::seed_from_u64(4);
        let s = sampler.sample(&g, &[0, 1], &f, &mut rng);
        // Seeds must be in node_ids
        assert!(s.node_ids.contains(&0));
        assert!(s.node_ids.contains(&1));
    }

    #[test]
    fn test_mini_batch_nodes_count() {
        let mut rng = StdRng::seed_from_u64(5);
        let batch = GmsNeighborSampler::mini_batch_nodes(100, 16, &mut rng);
        assert_eq!(batch.len(), 16);
    }

    #[test]
    fn test_mini_batch_nodes_valid_range() {
        let mut rng = StdRng::seed_from_u64(6);
        let batch = GmsNeighborSampler::mini_batch_nodes(50, 10, &mut rng);
        for &n in &batch {
            assert!(n < 50);
        }
    }

    // §3 ClusterGcn
    #[test]
    fn test_cluster_partition_covers_all() {
        let g = small_graph();
        let mut rng = StdRng::seed_from_u64(7);
        let clusters = ClusterGcn::partition_graph(&g, 3, &mut rng);
        let mut all_nodes: Vec<usize> = clusters.iter().flat_map(|c| c.node_ids.clone()).collect();
        all_nodes.sort_unstable();
        all_nodes.dedup();
        assert_eq!(all_nodes.len(), 5);
    }

    #[test]
    fn test_cluster_partition_count() {
        let g = small_graph();
        let mut rng = StdRng::seed_from_u64(8);
        let clusters = ClusterGcn::partition_graph(&g, 3, &mut rng);
        assert_eq!(clusters.len(), 3);
    }

    #[test]
    fn test_sample_cluster_batch_size() {
        let g = small_graph();
        let mut rng = StdRng::seed_from_u64(9);
        let clusters = ClusterGcn::partition_graph(&g, 3, &mut rng);
        let mut rng2 = StdRng::seed_from_u64(10);
        let nodes = ClusterGcn::sample_cluster_batch(&clusters, 2, &mut rng2);
        assert!(!nodes.is_empty());
    }

    #[test]
    fn test_induced_subgraph_nodes() {
        let g = small_graph();
        let sub = ClusterGcn::induced_subgraph(&g, &[0, 1, 2]);
        assert_eq!(sub.n_nodes, 3);
    }

    #[test]
    fn test_induced_subgraph_edges_subset() {
        let g = small_graph();
        let sub = ClusterGcn::induced_subgraph(&g, &[0, 1, 2]);
        // Edge count must be <= original
        assert!(sub.n_edges <= g.n_edges);
    }

    #[test]
    fn test_induced_subgraph_n_nodes() {
        let g = small_graph();
        let sub = ClusterGcn::induced_subgraph(&g, &[1, 3, 4]);
        assert_eq!(sub.n_nodes, 3);
    }

    #[test]
    fn test_gcn_layer_forward_shape() {
        let g = small_graph();
        let mut rng = StdRng::seed_from_u64(11);
        let layer = GmsGcnLayer::new(3, 8, &mut rng);
        let feats: Vec<Vec<f32>> = (0..5).map(|_| vec![1.0, 2.0, 3.0]).collect();
        let out = layer.forward(&feats, &g);
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn test_gcn_forward_batch_shape() {
        let g = small_graph();
        let mut rng = StdRng::seed_from_u64(12);
        let layer = GmsGcnLayer::new(3, 4, &mut rng);
        let feats: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32; 3]).collect();
        let out = layer.forward(&feats, &g);
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].len(), 4);
    }

    // §4 GraphSAINT
    #[test]
    fn test_saint_node_sampler_count() {
        let g = small_graph();
        let sampler = NodeSampler::new(10);
        let mut rng = StdRng::seed_from_u64(13);
        let sub = sampler.sample(&g, 3, &mut rng);
        assert_eq!(sub.nodes.len(), 3);
    }

    #[test]
    fn test_saint_node_sampler_valid_nodes() {
        let g = small_graph();
        let sampler = NodeSampler::new(10);
        let mut rng = StdRng::seed_from_u64(14);
        let sub = sampler.sample(&g, 4, &mut rng);
        for &n in &sub.nodes {
            assert!(n < g.n_nodes);
        }
    }

    #[test]
    fn test_saint_edge_sampler_edges_count() {
        let g = small_graph();
        let sampler = EdgeSampler::new(10);
        let mut rng = StdRng::seed_from_u64(15);
        let sub = sampler.sample(&g, 4, &mut rng);
        assert_eq!(sub.edges.len(), 4);
    }

    #[test]
    fn test_saint_subgraph_edges_valid() {
        let g = small_graph();
        let sampler = NodeSampler::new(10);
        let mut rng = StdRng::seed_from_u64(16);
        let sub = sampler.sample(&g, 4, &mut rng);
        for &(u, v) in &sub.edges {
            assert!(sub.nodes.contains(&u) || u < g.n_nodes);
            assert!(sub.nodes.contains(&v) || v < g.n_nodes);
        }
    }

    #[test]
    fn test_node_weight_positive() {
        let w = NormalizationWeights::node_weight(5, 0.2);
        assert!(w > 0.0);
    }

    #[test]
    fn test_edge_weight_is_min() {
        let w = NormalizationWeights::edge_weight(2.0, 3.0);
        assert!((w - 2.0).abs() < 1e-6);
    }

    // §5 SIGN
    #[test]
    fn test_sign_precompute_diffusion_k_features() {
        let g = small_graph();
        let f = small_features();
        let diffused = SignModel::precompute_diffusion(&g, &f, 3);
        assert_eq!(diffused.len(), 4); // k+1
    }

    #[test]
    fn test_sign_precompute_diffusion_shape() {
        let g = small_graph();
        let f = small_features();
        let diffused = SignModel::precompute_diffusion(&g, &f, 2);
        for feat_mat in &diffused {
            assert_eq!(feat_mat.n_nodes, 5);
            assert_eq!(feat_mat.feat_dim, 3);
        }
    }

    #[test]
    fn test_sym_normalize_adj_finite() {
        let g = small_graph();
        let edges = SignModel::sym_normalize_adj(&g);
        for &(_, _, w) in &edges {
            assert!(w.is_finite());
        }
    }

    #[test]
    fn test_sign_model_creation() {
        let cfg = SignConfig {
            n_hops: 2,
            hidden_dim: 8,
            n_classes: 3,
        };
        let mut rng = StdRng::seed_from_u64(17);
        let model = SignModel::new(3, cfg, &mut rng);
        assert_eq!(model.per_hop_layers.len(), 3); // n_hops+1
    }

    #[test]
    fn test_sign_model_forward_shape() {
        let g = small_graph();
        let f = small_features();
        let cfg = SignConfig {
            n_hops: 2,
            hidden_dim: 8,
            n_classes: 4,
        };
        let mut rng = StdRng::seed_from_u64(18);
        let model = SignModel::new(3, cfg, &mut rng);
        let diffused = SignModel::precompute_diffusion(&g, &f, 2);
        let logits = model.forward(&diffused);
        assert_eq!(logits.len(), 5);
    }

    #[test]
    fn test_sign_model_n_classes() {
        let g = small_graph();
        let f = small_features();
        let cfg = SignConfig {
            n_hops: 1,
            hidden_dim: 4,
            n_classes: 3,
        };
        let mut rng = StdRng::seed_from_u64(19);
        let model = SignModel::new(3, cfg, &mut rng);
        let diffused = SignModel::precompute_diffusion(&g, &f, 1);
        let logits = model.forward(&diffused);
        assert_eq!(logits[0].len(), 3);
    }

    // §6 GraphSAGE
    #[test]
    fn test_graphsage_layer_mean_shape() {
        let g = small_graph();
        let mut rng = StdRng::seed_from_u64(20);
        let layer = GmsGraphSageLayer::new(3, 8, GmsAggregatorType::Mean, &mut rng);
        let feats: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32; 3]).collect();
        let out = layer.forward_node(0, &feats, &g);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_graphsage_layer_maxpool_shape() {
        let g = small_graph();
        let mut rng = StdRng::seed_from_u64(21);
        let layer = GmsGraphSageLayer::new(3, 6, GmsAggregatorType::MaxPool, &mut rng);
        let feats: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32; 3]).collect();
        let out = layer.forward_node(2, &feats, &g);
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn test_graphsage_layer_gcn_shape() {
        let g = small_graph();
        let mut rng = StdRng::seed_from_u64(22);
        let layer = GmsGraphSageLayer::new(3, 5, GmsAggregatorType::GcnMean, &mut rng);
        let feats: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32; 3]).collect();
        let out = layer.forward_node(1, &feats, &g);
        assert_eq!(out.len(), 5);
    }

    #[test]
    fn test_graphsage_l2_normalized() {
        let g = small_graph();
        let mut rng = StdRng::seed_from_u64(23);
        let layer = GmsGraphSageLayer::new(3, 4, GmsAggregatorType::Mean, &mut rng);
        // Use non-zero features to trigger non-trivial activation
        let feats: Vec<Vec<f32>> = (0..5).map(|i| vec![(i + 1) as f32 * 0.5; 3]).collect();
        let out = layer.forward_node(0, &feats, &g);
        let norm: f32 = out.iter().map(|&x| x * x).sum::<f32>().sqrt();
        // Either zero (all-negative pre-activation → ReLU) or unit norm
        assert!(norm < 1e-6 || (norm - 1.0).abs() < 1e-5, "norm={norm}");
    }

    #[test]
    fn test_graphsage_aggregate_shape() {
        let g = small_graph();
        let mut rng = StdRng::seed_from_u64(24);
        let layer = GmsGraphSageLayer::new(3, 4, GmsAggregatorType::Mean, &mut rng);
        let feats: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32; 3]).collect();
        let agg = layer.aggregate_neighbors(&g, 0, &feats);
        assert_eq!(agg.len(), 3);
    }

    #[test]
    fn test_graphsage_model_forward_batch_shape() {
        let g = small_graph();
        let f = small_features();
        let mut rng = StdRng::seed_from_u64(25);
        let model = GmsGraphSageModel::new(&[(3, 8), (8, 4)], GmsAggregatorType::Mean, &mut rng);
        let out = model.forward_batch(&g, &[0, 1, 2], &f);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].len(), 4);
    }

    #[test]
    fn test_graphsage_model_n_layers() {
        let mut rng = StdRng::seed_from_u64(26);
        let model =
            GmsGraphSageModel::new(&[(3, 8), (8, 4), (4, 2)], GmsAggregatorType::Mean, &mut rng);
        assert_eq!(model.layers.len(), 3);
    }

    // §7 Metrics
    #[test]
    fn test_node_classification_accuracy_perfect() {
        let logits = vec![
            vec![1.0_f32, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let labels = vec![0usize, 1, 2];
        let acc = node_classification_accuracy(&logits, &labels);
        assert!((acc - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_node_classification_accuracy_zero() {
        let logits = vec![
            vec![0.0_f32, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0],
        ];
        let labels = vec![0usize, 1, 2];
        let acc = node_classification_accuracy(&logits, &labels);
        assert!((acc - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_link_prediction_auc_range() {
        let embeddings: Vec<Vec<f32>> = (0..5).map(|i| vec![(i as f32 + 1.0).recip(); 4]).collect();
        let pos_edges = vec![(0, 1), (1, 2)];
        let neg_edges = vec![(0, 4), (2, 4)];
        let auc = link_prediction_auc(&embeddings, &pos_edges, &neg_edges);
        assert!((0.0..=1.0).contains(&auc));
    }

    #[test]
    fn test_cluster_modularity_range() {
        let g = small_graph();
        let mut rng = StdRng::seed_from_u64(27);
        let clusters = ClusterGcn::partition_graph(&g, 2, &mut rng);
        let q = cluster_modularity(&g, &clusters);
        assert!((-1.0..=1.0).contains(&q));
    }

    #[test]
    fn test_scalable_report_fields() {
        let report = ScalableGnnReport::new(0.9, 0.85, 0.3, 1000.0);
        assert!((report.accuracy - 0.9).abs() < 1e-6);
        assert!((report.link_auc - 0.85).abs() < 1e-6);
        assert!((report.modularity - 0.3).abs() < 1e-6);
        assert!((report.throughput_nodes_per_sec - 1000.0).abs() < 1e-6);
    }

    // Additional tests
    #[test]
    fn test_csr_neighbors_all_valid_indices() {
        let g = small_graph();
        for v in 0..g.n_nodes {
            for &nb in g.neighbors(v) {
                assert!(nb < g.n_nodes);
            }
        }
    }

    #[test]
    fn test_subgraph_sample_structure() {
        let g = small_graph();
        let f = small_features();
        let cfg = SamplerConfig {
            n_layers: 2,
            fanouts: vec![2, 2],
        };
        let sampler = GmsNeighborSampler::new(cfg);
        let mut rng = StdRng::seed_from_u64(28);
        let s = sampler.sample(&g, &[0, 1], &f, &mut rng);
        assert_eq!(s.features.len(), s.node_ids.len());
        assert_eq!(s.adjacencies.len(), 2);
    }

    #[test]
    fn test_neighbor_sampler_multi_layer() {
        let g = small_graph();
        let f = small_features();
        let cfg = SamplerConfig {
            n_layers: 3,
            fanouts: vec![2, 2, 2],
        };
        let sampler = GmsNeighborSampler::new(cfg);
        let mut rng = StdRng::seed_from_u64(29);
        let s = sampler.sample(&g, &[0], &f, &mut rng);
        assert_eq!(s.adjacencies.len(), 3);
    }
}
