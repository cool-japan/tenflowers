//! Network Science and Complex Network Analysis with ML
//!
//! Implements graph-theoretic algorithms, community detection, node embeddings,
//! link prediction, robustness analysis, epidemic simulation, motif finding,
//! temporal network analysis, and summary statistics.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// 1. NetworkGraph — sparse adjacency-list graph representation
// ---------------------------------------------------------------------------

/// Sparse graph with weighted adjacency lists.
#[derive(Debug, Clone)]
pub struct NetworkGraph {
    /// Number of nodes (nodes are identified by 0..n_nodes-1).
    pub n_nodes: usize,
    /// Adjacency list: adj\[u\] = [(v, weight), ...]
    pub adj: Vec<Vec<(usize, f32)>>,
    /// Whether edges are directed.
    pub is_directed: bool,
}

impl NetworkGraph {
    /// Create an empty graph with `n_nodes` nodes.
    pub fn new(n_nodes: usize, is_directed: bool) -> Self {
        Self {
            n_nodes,
            adj: vec![Vec::new(); n_nodes],
            is_directed,
        }
    }

    /// Add a weighted edge from `u` to `v`. For undirected graphs, adds both directions.
    pub fn add_edge(&mut self, u: usize, v: usize, w: f32) {
        if u >= self.n_nodes || v >= self.n_nodes {
            return;
        }
        self.adj[u].push((v, w));
        if !self.is_directed && u != v {
            self.adj[v].push((u, w));
        }
    }

    /// Unweighted degree (number of neighbors).
    pub fn degree(&self, node: usize) -> usize {
        if node >= self.n_nodes {
            return 0;
        }
        self.adj[node].len()
    }

    /// Sum of edge weights incident to `node`.
    pub fn weighted_degree(&self, node: usize) -> f32 {
        if node >= self.n_nodes {
            return 0.0;
        }
        self.adj[node].iter().map(|&(_, w)| w).sum()
    }

    /// Slice of neighbors as (node_id, weight) pairs.
    pub fn neighbors(&self, node: usize) -> &[(usize, f32)] {
        if node >= self.n_nodes {
            return &[];
        }
        &self.adj[node]
    }

    /// Total weight of all edges (counts each undirected edge once).
    pub fn total_weight(&self) -> f64 {
        let sum: f32 = self
            .adj
            .iter()
            .flat_map(|row| row.iter().map(|&(_, w)| w))
            .sum();
        if self.is_directed {
            sum as f64
        } else {
            sum as f64 / 2.0
        }
    }

    /// Collect set of neighbors as a `HashSet<usize>`.
    pub fn neighbor_set(&self, node: usize) -> HashSet<usize> {
        self.adj[node].iter().map(|&(v, _)| v).collect()
    }

    /// Check whether edge (u, v) exists.
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        if u >= self.n_nodes {
            return false;
        }
        self.adj[u].iter().any(|&(w, _)| w == v)
    }
}

// Dijkstra shortest paths from a source (returns f32::INFINITY for unreachable).
fn dijkstra(g: &NetworkGraph, src: usize) -> Vec<f32> {
    let n = g.n_nodes;
    let mut dist = vec![f32::INFINITY; n];
    dist[src] = 0.0;

    // (neg_dist, node) — BinaryHeap is a max-heap, so negate to get min-heap.
    let mut heap: BinaryHeap<(OrderedF32, usize)> = BinaryHeap::new();
    heap.push((OrderedF32(-0.0), src));

    while let Some((OrderedF32(neg_d), u)) = heap.pop() {
        let d = -neg_d;
        if d > dist[u] {
            continue;
        }
        for &(v, w) in &g.adj[u] {
            let nd = d + w;
            if nd < dist[v] {
                dist[v] = nd;
                heap.push((OrderedF32(-nd), v));
            }
        }
    }
    dist
}

// BFS shortest paths (unit weights) from a source.
fn bfs_distances(g: &NetworkGraph, src: usize) -> Vec<Option<usize>> {
    let n = g.n_nodes;
    let mut dist = vec![None; n];
    dist[src] = Some(0);
    let mut queue: VecDeque<usize> = VecDeque::new();
    queue.push_back(src);
    while let Some(u) = queue.pop_front() {
        let d = dist[u].unwrap_or(0);
        for &(v, _) in &g.adj[u] {
            if dist[v].is_none() {
                dist[v] = Some(d + 1);
                queue.push_back(v);
            }
        }
    }
    dist
}

// Newtype wrapper for f32 to make it orderable in BinaryHeap.
#[derive(Clone, Copy, PartialEq)]
struct OrderedF32(f32);

impl Eq for OrderedF32 {}

impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// 2. CentralityMeasures
// ---------------------------------------------------------------------------

/// Collection of functions computing node centrality scores.
pub struct CentralityMeasures;

impl CentralityMeasures {
    /// Degree centrality: d(v) / (n-1).
    pub fn degree_centrality(g: &NetworkGraph) -> Vec<f32> {
        let n = g.n_nodes;
        if n <= 1 {
            return vec![0.0; n];
        }
        let norm = (n - 1) as f32;
        (0..n).map(|v| g.degree(v) as f32 / norm).collect()
    }

    /// Closeness centrality using BFS (unweighted) or Dijkstra (weighted).
    pub fn closeness_centrality(g: &NetworkGraph) -> Vec<f32> {
        let n = g.n_nodes;
        (0..n)
            .map(|v| {
                let dists = dijkstra(g, v);
                let (reachable, sum) =
                    dists
                        .iter()
                        .enumerate()
                        .fold((0usize, 0.0f64), |(cnt, s), (u, &d)| {
                            if u != v && d.is_finite() {
                                (cnt + 1, s + d as f64)
                            } else {
                                (cnt, s)
                            }
                        });
                if reachable == 0 || sum == 0.0 {
                    return 0.0;
                }
                // Normalized for disconnected graphs (Wasserman & Faust)
                let norm = reachable as f64 / (n - 1) as f64;
                (reachable as f64 / sum * norm) as f32
            })
            .collect()
    }

    /// Betweenness centrality using Brandes algorithm O(VE).
    pub fn betweenness_centrality(g: &NetworkGraph) -> Vec<f32> {
        let n = g.n_nodes;
        let mut bc = vec![0.0f64; n];

        for s in 0..n {
            let mut stack: Vec<usize> = Vec::new();
            let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
            let mut sigma = vec![0.0f64; n];
            sigma[s] = 1.0;
            let mut dist = vec![-1i64; n];
            dist[s] = 0;
            let mut queue: VecDeque<usize> = VecDeque::new();
            queue.push_back(s);

            while let Some(v) = queue.pop_front() {
                stack.push(v);
                for &(w, _) in &g.adj[v] {
                    if dist[w] < 0 {
                        queue.push_back(w);
                        dist[w] = dist[v] + 1;
                    }
                    if dist[w] == dist[v] + 1 {
                        sigma[w] += sigma[v];
                        pred[w].push(v);
                    }
                }
            }

            let mut delta = vec![0.0f64; n];
            while let Some(w) = stack.pop() {
                for &v in &pred[w] {
                    if sigma[w] > 0.0 {
                        delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                    }
                }
                if w != s {
                    bc[w] += delta[w];
                }
            }
        }

        // Normalize
        let norm = if g.is_directed {
            ((n - 1) * (n - 2)) as f64
        } else {
            ((n - 1) * (n - 2)) as f64 / 2.0
        };
        bc.iter()
            .map(|&v| if norm > 0.0 { (v / norm) as f32 } else { 0.0 })
            .collect()
    }

    /// Eigenvector centrality via power iteration.
    pub fn eigenvector_centrality(g: &NetworkGraph, max_iter: usize) -> Vec<f32> {
        let n = g.n_nodes;
        if n == 0 {
            return vec![];
        }
        let mut x = vec![1.0f64 / (n as f64).sqrt(); n];

        for _ in 0..max_iter {
            let mut x_new = vec![0.0f64; n];
            for u in 0..n {
                for &(v, w) in &g.adj[u] {
                    x_new[u] += w as f64 * x[v];
                }
            }
            // Normalize L2
            let norm: f64 = x_new.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-12 {
                for xi in x_new.iter_mut() {
                    *xi /= norm;
                }
            }
            let diff: f64 = x.iter().zip(x_new.iter()).map(|(a, b)| (a - b).abs()).sum();
            x = x_new;
            if diff < 1e-10 {
                break;
            }
        }
        x.iter().map(|&v| v as f32).collect()
    }

    /// PageRank with damping factor and power iteration.
    pub fn pagerank(g: &NetworkGraph, damping: f32, max_iter: usize) -> Vec<f32> {
        let n = g.n_nodes;
        if n == 0 {
            return vec![];
        }
        let d = damping as f64;
        let mut pr = vec![1.0 / n as f64; n];

        // Out-degree for normalization
        let out_deg: Vec<f64> = (0..n)
            .map(|v| g.adj[v].iter().map(|&(_, w)| w as f64).sum::<f64>())
            .collect();

        for _ in 0..max_iter {
            let mut pr_new = vec![(1.0 - d) / n as f64; n];
            for u in 0..n {
                if out_deg[u] > 0.0 {
                    for &(v, w) in &g.adj[u] {
                        pr_new[v] += d * pr[u] * (w as f64 / out_deg[u]);
                    }
                } else {
                    // Dangling node: distribute equally
                    let share = d * pr[u] / n as f64;
                    for pv in pr_new.iter_mut() {
                        *pv += share;
                    }
                }
            }
            let diff: f64 = pr
                .iter()
                .zip(pr_new.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            pr = pr_new;
            if diff < 1e-10 {
                break;
            }
        }
        pr.iter().map(|&v| v as f32).collect()
    }
}

// ---------------------------------------------------------------------------
// 3. CommunityDetection
// ---------------------------------------------------------------------------

/// Community detection algorithms.
pub struct CommunityDetection;

impl CommunityDetection {
    /// Compute modularity Q for a given partition.
    /// Q = (1/2m) * Σ_{ij} [A_ij - k_i*k_j/(2m)] * δ(c_i, c_j)
    pub fn modularity(g: &NetworkGraph, partition: &[usize]) -> f64 {
        let m2 = 2.0 * g.total_weight();
        if m2 <= 0.0 {
            return 0.0;
        }
        let mut q = 0.0f64;
        for u in 0..g.n_nodes {
            let ku = g.weighted_degree(u) as f64;
            for &(v, w) in &g.adj[u] {
                if partition.get(u) == partition.get(v) {
                    q += w as f64 - ku * g.weighted_degree(v) as f64 / m2;
                }
            }
        }
        q / m2
    }

    /// One pass of Louvain modularity optimization.
    /// Returns the modularity gain achieved.
    pub fn louvain_step(g: &NetworkGraph, partition: &mut [usize]) -> f64 {
        let n = g.n_nodes;
        let m2 = 2.0 * g.total_weight();
        if m2 <= 0.0 {
            return 0.0;
        }

        // Community weighted degrees
        let mut com_degree: HashMap<usize, f64> = HashMap::new();
        for u in 0..n {
            let com = partition[u];
            *com_degree.entry(com).or_insert(0.0) += g.weighted_degree(u) as f64;
        }

        let before = Self::modularity(g, partition);
        let mut improved = true;

        while improved {
            improved = false;
            for u in 0..n {
                let ku = g.weighted_degree(u) as f64;
                let cur_com = partition[u];

                // Weight of edges from u to each neighboring community
                let mut nbr_com_weight: HashMap<usize, f64> = HashMap::new();
                for &(v, w) in g.neighbors(u) {
                    *nbr_com_weight.entry(partition[v]).or_insert(0.0) += w as f64;
                }

                // Remove u from its community
                *com_degree.entry(cur_com).or_insert(0.0) -= ku;
                partition[u] = n + u; // temporary isolated community
                com_degree.insert(n + u, ku);

                // Delta Q for moving u to community c:
                // ΔQ = [k_{u,c}/m - (k_c * k_u)/(2m^2)] - [... same for cur_com removal ...]
                // Simplified gain: k_{u,c}/m - Σ_c (k_c * k_u) / m^2
                let mut best_com = n + u;
                let mut best_gain = 0.0f64;

                let mut candidates: HashSet<usize> = nbr_com_weight.keys().copied().collect();
                candidates.insert(cur_com);

                for &c in &candidates {
                    let kuc = nbr_com_weight.get(&c).copied().unwrap_or(0.0);
                    let kc = com_degree.get(&c).copied().unwrap_or(0.0);
                    let gain = kuc / m2 - (kc * ku) / (m2 * m2);
                    if gain > best_gain {
                        best_gain = gain;
                        best_com = c;
                    }
                }

                // Move u to best community
                partition[u] = best_com;
                *com_degree.entry(best_com).or_insert(0.0) += ku;
                // Remove isolated community entry
                com_degree.remove(&(n + u));

                if best_com != cur_com {
                    improved = true;
                }
            }
        }

        // Remap community IDs to 0..k
        let mut id_map: HashMap<usize, usize> = HashMap::new();
        let mut next_id = 0usize;
        for p in partition.iter_mut() {
            let new_id = *id_map.entry(*p).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            *p = new_id;
        }

        Self::modularity(g, partition) - before
    }

    /// Spectral clustering using the normalized graph Laplacian.
    /// Returns node-to-cluster assignment vector.
    pub fn spectral_clustering_network(g: &NetworkGraph, k: usize) -> Vec<usize> {
        let n = g.n_nodes;
        if n == 0 || k == 0 {
            return vec![0; n];
        }
        let k = k.min(n);

        // Build degree-normalized Laplacian eigenvectors via power iteration
        // D^{-1/2} L D^{-1/2} where L = D - A
        let deg: Vec<f64> = (0..n).map(|u| g.weighted_degree(u) as f64).collect();
        let deg_inv_sqrt: Vec<f64> = deg
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();

        // Compute k leading eigenvectors of D^{-1} A (normalized adjacency) via power iteration
        // This approximates spectral embedding
        let mut embeddings: Vec<Vec<f64>> = Vec::with_capacity(k);
        let mut seed_rng = StdRng::seed_from_u64(42);

        for i in 0..k {
            let mut v: Vec<f64> = (0..n).map(|_| seed_rng.random::<f64>() - 0.5).collect();
            // Orthogonalize against previous vectors (Gram-Schmidt)
            for prev in &embeddings {
                let dot: f64 = v.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                for (vi, pi) in v.iter_mut().zip(prev.iter()) {
                    *vi -= dot * pi;
                }
            }
            // Power iteration: v <- D^{-1} A v, normalized
            for _ in 0..50 {
                let mut v_new = vec![0.0f64; n];
                for u in 0..n {
                    let mut s = 0.0;
                    for &(nb, w) in g.neighbors(u) {
                        s += w as f64 * v[nb];
                    }
                    v_new[u] = if deg[u] > 0.0 { s / deg[u] } else { 0.0 };
                }
                // Re-orthogonalize
                for prev in &embeddings {
                    let dot: f64 = v_new.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                    for (vi, pi) in v_new.iter_mut().zip(prev.iter()) {
                        *vi -= dot * pi;
                    }
                }
                let norm: f64 = v_new.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-12 {
                    for xi in v_new.iter_mut() {
                        *xi /= norm;
                    }
                }
                let diff: f64 = v.iter().zip(v_new.iter()).map(|(a, b)| (a - b).abs()).sum();
                v = v_new;
                if diff < 1e-10 {
                    break;
                }
            }
            // Normalize with degree (normalize rows using D^{-1/2})
            for (u, vi) in v.iter_mut().enumerate() {
                *vi *= deg_inv_sqrt[u];
            }
            embeddings.push(v);
            // suppress unused variable warning for i
            let _ = i;
        }

        // k-means clustering on the embedding rows
        // Each node's feature vector is embeddings[0][node], embeddings[1][node], ...
        let rows: Vec<Vec<f64>> = (0..n)
            .map(|node| embeddings.iter().map(|e| e[node]).collect())
            .collect();

        kmeans_cluster(&rows, k, 100, &mut StdRng::seed_from_u64(99))
    }

    /// Label propagation community detection.
    pub fn label_propagation(g: &NetworkGraph, max_iter: usize, rng: &mut StdRng) -> Vec<usize> {
        let n = g.n_nodes;
        let mut labels: Vec<usize> = (0..n).collect();
        let mut order: Vec<usize> = (0..n).collect();

        for _ in 0..max_iter {
            // Shuffle update order
            for i in (1..n).rev() {
                let j = (rng.random::<u64>() as usize) % (i + 1);
                order.swap(i, j);
            }
            let mut changed = false;
            for &u in &order {
                if g.adj[u].is_empty() {
                    continue;
                }
                // Count label frequencies among neighbors
                let mut freq: HashMap<usize, f64> = HashMap::new();
                for &(v, w) in &g.adj[u] {
                    *freq.entry(labels[v]).or_insert(0.0) += w as f64;
                }
                // Choose label with maximum frequency (break ties by smallest label)
                let best = freq
                    .iter()
                    .max_by(|a, b| {
                        a.1.partial_cmp(b.1)
                            .unwrap_or(Ordering::Equal)
                            .then(b.0.cmp(a.0))
                    })
                    .map(|(&l, _)| l)
                    .unwrap_or(labels[u]);
                if best != labels[u] {
                    labels[u] = best;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // Remap to 0-based sequential IDs
        let mut id_map: HashMap<usize, usize> = HashMap::new();
        let mut next_id = 0usize;
        for l in labels.iter_mut() {
            let new_id = *id_map.entry(*l).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            *l = new_id;
        }
        labels
    }
}

// Simple k-means returning cluster assignments.
fn kmeans_cluster(data: &[Vec<f64>], k: usize, max_iter: usize, rng: &mut StdRng) -> Vec<usize> {
    let n = data.len();
    if n == 0 || k == 0 {
        return vec![0; n];
    }
    let k = k.min(n);
    let dim = data[0].len();

    // Initialize centroids by random selection
    let mut indices: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.random::<u64>() as usize) % (i + 1);
        indices.swap(i, j);
    }
    let mut centroids: Vec<Vec<f64>> = indices[..k].iter().map(|&i| data[i].clone()).collect();
    let mut assignments = vec![0usize; n];

    for _ in 0..max_iter {
        let mut changed = false;
        // Assignment step
        for (i, point) in data.iter().enumerate() {
            let best = (0..k)
                .min_by(|&a, &b| {
                    let da: f64 = point
                        .iter()
                        .zip(centroids[a].iter())
                        .map(|(x, c)| (x - c).powi(2))
                        .sum();
                    let db: f64 = point
                        .iter()
                        .zip(centroids[b].iter())
                        .map(|(x, c)| (x - c).powi(2))
                        .sum();
                    da.partial_cmp(&db).unwrap_or(Ordering::Equal)
                })
                .unwrap_or(0);
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        // Update step
        let mut sums: Vec<Vec<f64>> = vec![vec![0.0; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, point) in data.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for (s, x) in sums[c].iter_mut().zip(point.iter()) {
                *s += x;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    centroids[c][d] = sums[c][d] / counts[c] as f64;
                }
            }
        }
    }
    assignments
}

// ---------------------------------------------------------------------------
// 4. NetworkEmbedding — Node2Vec
// ---------------------------------------------------------------------------

/// Configuration for Node2Vec embedding.
#[derive(Debug, Clone)]
pub struct Node2VecConfig {
    /// Length of each random walk.
    pub walk_len: usize,
    /// Number of walks per node.
    pub n_walks: usize,
    /// Return parameter p (higher = less likely to revisit).
    pub p: f32,
    /// In-out parameter q (higher = DFS-like, lower = BFS-like).
    pub q: f32,
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Context window size for skip-gram.
    pub window: usize,
}

impl Default for Node2VecConfig {
    fn default() -> Self {
        Self {
            walk_len: 80,
            n_walks: 10,
            p: 1.0,
            q: 1.0,
            embed_dim: 64,
            window: 5,
        }
    }
}

/// Generate a biased random walk from `start` for Node2Vec.
pub fn biased_random_walk(
    g: &NetworkGraph,
    start: usize,
    walk_len: usize,
    p: f32,
    q: f32,
    rng: &mut StdRng,
) -> Vec<usize> {
    let mut walk = vec![start];
    if walk_len == 0 || g.adj[start].is_empty() {
        return walk;
    }
    // First step: uniform
    let first_idx = (rng.random::<u64>() as usize) % g.adj[start].len();
    let first = g.adj[start][first_idx].0;
    walk.push(first);

    for _ in 2..walk_len {
        let cur = *walk.last().unwrap_or(&start);
        let prev = walk[walk.len() - 2];
        let nbrs = &g.adj[cur];
        if nbrs.is_empty() {
            break;
        }

        // Compute unnormalized transition probabilities
        let prev_nbrs: HashSet<usize> = g.adj[prev].iter().map(|&(v, _)| v).collect();

        let weights: Vec<f64> = nbrs
            .iter()
            .map(|&(nb, w)| {
                let base = w as f64;
                let alpha = if nb == prev {
                    1.0 / p as f64
                } else if prev_nbrs.contains(&nb) {
                    1.0
                } else {
                    1.0 / q as f64
                };
                base * alpha
            })
            .collect();

        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            break;
        }

        let r: f64 = rng.random::<f64>() * total;
        let mut cum = 0.0;
        let mut chosen = nbrs[0].0;
        for (i, &w) in weights.iter().enumerate() {
            cum += w;
            if r <= cum {
                chosen = nbrs[i].0;
                break;
            }
        }
        walk.push(chosen);
    }
    walk
}

/// One skip-gram negative-sampling SGD update.
pub fn skip_gram_update(
    embeddings: &mut [Vec<f32>],
    center: usize,
    context: &[usize],
    neg_samples: &[usize],
    lr: f32,
) {
    let dim = if let Some(e) = embeddings.first() {
        e.len()
    } else {
        return;
    };
    let n = embeddings.len();

    if center >= n {
        return;
    }

    for &ctx in context {
        if ctx >= n {
            continue;
        }
        // Positive pair: maximize log σ(u · v)
        let dot: f32 = (0..dim)
            .map(|d| embeddings[center][d] * embeddings[ctx][d])
            .sum();
        let sig = sigmoid_f32(dot);
        let grad = (1.0 - sig) * lr;
        // Clone context embedding to avoid borrow issues
        let ctx_embed = embeddings[ctx].clone();
        let center_embed = embeddings[center].clone();
        for d in 0..dim {
            embeddings[center][d] += grad * ctx_embed[d];
            embeddings[ctx][d] += grad * center_embed[d];
        }

        for &neg in neg_samples {
            if neg >= n || neg == ctx {
                continue;
            }
            let dot_neg: f32 = (0..dim)
                .map(|d| embeddings[center][d] * embeddings[neg][d])
                .sum();
            let sig_neg = sigmoid_f32(dot_neg);
            let grad_neg = -sig_neg * lr;
            let neg_embed = embeddings[neg].clone();
            let center_embed2 = embeddings[center].clone();
            for d in 0..dim {
                embeddings[center][d] += grad_neg * neg_embed[d];
                embeddings[neg][d] += grad_neg * center_embed2[d];
            }
        }
    }
}

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Train Node2Vec embeddings.
pub fn node2vec_train(
    g: &NetworkGraph,
    config: &Node2VecConfig,
    rng: &mut StdRng,
) -> Vec<Vec<f32>> {
    let n = g.n_nodes;
    let dim = config.embed_dim;

    // Initialize embeddings ~ U(-0.5/dim, 0.5/dim)
    let scale = 0.5 / dim as f32;
    let mut embeddings: Vec<Vec<f32>> = (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| (rng.random::<f32>() - 0.5) * scale)
                .collect()
        })
        .collect();

    let neg_k = 5usize; // negative samples per positive
    let lr = 0.025f32;

    for node in 0..n {
        for _ in 0..config.n_walks {
            let walk = biased_random_walk(g, node, config.walk_len, config.p, config.q, rng);
            for (i, &center) in walk.iter().enumerate() {
                let win_start = i.saturating_sub(config.window);
                let win_end = (i + config.window + 1).min(walk.len());
                let context: Vec<usize> = walk[win_start..win_end]
                    .iter()
                    .copied()
                    .filter(|&v| v != center)
                    .collect();
                // Random negative samples
                let neg: Vec<usize> = (0..neg_k)
                    .map(|_| (rng.random::<u64>() as usize) % n)
                    .collect();
                skip_gram_update(&mut embeddings, center, &context, &neg, lr);
            }
        }
    }
    embeddings
}

// ---------------------------------------------------------------------------
// 5. LinkPrediction
// ---------------------------------------------------------------------------

/// Structural link prediction scores.
pub struct LinkPrediction;

impl LinkPrediction {
    /// Number of common neighbors between u and v.
    pub fn common_neighbors(g: &NetworkGraph, u: usize, v: usize) -> usize {
        if u >= g.n_nodes || v >= g.n_nodes {
            return 0;
        }
        let nu = g.neighbor_set(u);
        let nv = g.neighbor_set(v);
        nu.intersection(&nv).count()
    }

    /// Jaccard coefficient: |N(u) ∩ N(v)| / |N(u) ∪ N(v)|.
    pub fn jaccard_coefficient(g: &NetworkGraph, u: usize, v: usize) -> f32 {
        if u >= g.n_nodes || v >= g.n_nodes {
            return 0.0;
        }
        let nu = g.neighbor_set(u);
        let nv = g.neighbor_set(v);
        let inter = nu.intersection(&nv).count();
        let union = nu.union(&nv).count();
        if union == 0 {
            0.0
        } else {
            inter as f32 / union as f32
        }
    }

    /// Adamic-Adar score: Σ_{w ∈ N(u)∩N(v)} 1/log(deg(w)).
    pub fn adamic_adar(g: &NetworkGraph, u: usize, v: usize) -> f32 {
        if u >= g.n_nodes || v >= g.n_nodes {
            return 0.0;
        }
        let nu = g.neighbor_set(u);
        let nv = g.neighbor_set(v);
        nu.intersection(&nv)
            .map(|&w| {
                let d = g.degree(w) as f32;
                if d > 1.0 {
                    1.0 / d.ln()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Preferential attachment: deg(u) * deg(v).
    pub fn preferential_attachment(g: &NetworkGraph, u: usize, v: usize) -> f32 {
        (g.degree(u) * g.degree(v)) as f32
    }

    /// Dot-product link score from embeddings.
    pub fn embed_link_score(embed_u: &[f32], embed_v: &[f32]) -> f32 {
        embed_u.iter().zip(embed_v.iter()).map(|(a, b)| a * b).sum()
    }
}

// ---------------------------------------------------------------------------
// 6. NetworkRobustness
// ---------------------------------------------------------------------------

/// Network robustness and structural metrics.
pub struct NetworkRobustness;

impl NetworkRobustness {
    /// Diameter: maximum shortest path length. Returns 0 for disconnected graphs
    /// (or the max among reachable pairs).
    pub fn diameter(g: &NetworkGraph) -> usize {
        let n = g.n_nodes;
        let mut max_dist = 0usize;
        for src in 0..n {
            let dists = bfs_distances(g, src);
            for d in dists.into_iter().flatten() {
                if d > max_dist {
                    max_dist = d;
                }
            }
        }
        max_dist
    }

    /// Average shortest path length (over all reachable pairs).
    pub fn average_path_length(g: &NetworkGraph) -> f32 {
        let n = g.n_nodes;
        if n <= 1 {
            return 0.0;
        }
        let mut total = 0usize;
        let mut count = 0usize;
        for src in 0..n {
            let dists = bfs_distances(g, src);
            for (dst, d) in dists.into_iter().enumerate() {
                if dst != src {
                    if let Some(dist) = d {
                        total += dist;
                        count += 1;
                    }
                }
            }
        }
        if count == 0 {
            0.0
        } else {
            total as f32 / count as f32
        }
    }

    /// Local clustering coefficient of a node.
    pub fn clustering_coefficient(g: &NetworkGraph, node: usize) -> f32 {
        if node >= g.n_nodes {
            return 0.0;
        }
        let nbrs: Vec<usize> = g.adj[node].iter().map(|&(v, _)| v).collect();
        let k = nbrs.len();
        if k < 2 {
            return 0.0;
        }
        let nbr_set: HashSet<usize> = nbrs.iter().copied().collect();
        let triangles: usize = nbrs
            .iter()
            .map(|&u| {
                g.adj[u]
                    .iter()
                    .filter(|&&(v, _)| nbr_set.contains(&v))
                    .count()
            })
            .sum();
        // Each triangle counted twice (once from each end)
        triangles as f32 / (k * (k - 1)) as f32
    }

    /// Global (average) clustering coefficient.
    pub fn global_clustering(g: &NetworkGraph) -> f32 {
        let n = g.n_nodes;
        if n == 0 {
            return 0.0;
        }
        let sum: f32 = (0..n).map(|v| Self::clustering_coefficient(g, v)).sum();
        sum / n as f32
    }

    /// Degree assortativity (Pearson correlation of degrees at each end of edges).
    pub fn assortativity(g: &NetworkGraph) -> f32 {
        let mut deg_pairs: Vec<(f64, f64)> = Vec::new();
        for u in 0..g.n_nodes {
            let du = g.degree(u) as f64;
            for &(v, _) in &g.adj[u] {
                deg_pairs.push((du, g.degree(v) as f64));
            }
        }
        if deg_pairs.is_empty() {
            return 0.0;
        }
        let m = deg_pairs.len() as f64;
        let mean_x = deg_pairs.iter().map(|(x, _)| x).sum::<f64>() / m;
        let mean_y = deg_pairs.iter().map(|(_, y)| y).sum::<f64>() / m;
        let cov: f64 = deg_pairs
            .iter()
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum::<f64>()
            / m;
        let var_x: f64 = deg_pairs
            .iter()
            .map(|(x, _)| (x - mean_x).powi(2))
            .sum::<f64>()
            / m;
        let var_y: f64 = deg_pairs
            .iter()
            .map(|(_, y)| (y - mean_y).powi(2))
            .sum::<f64>()
            / m;
        let denom = (var_x * var_y).sqrt();
        if denom < 1e-12 {
            0.0
        } else {
            (cov / denom) as f32
        }
    }
}

// ---------------------------------------------------------------------------
// 7. EpidemicModel — SIR dynamics on networks
// ---------------------------------------------------------------------------

/// State of SIR epidemic model per node.
#[derive(Debug, Clone)]
pub struct SirState {
    /// Susceptible probability per node.
    pub s: Vec<f32>,
    /// Infected probability per node.
    pub i: Vec<f32>,
    /// Recovered probability per node.
    pub r: Vec<f32>,
}

impl SirState {
    /// Create initial state with all nodes susceptible.
    pub fn new_susceptible(n: usize) -> Self {
        Self {
            s: vec![1.0; n],
            i: vec![0.0; n],
            r: vec![0.0; n],
        }
    }
}

/// Configuration for SIR simulation.
#[derive(Debug, Clone)]
pub struct SirConfig {
    /// Transmission rate per contact.
    pub beta: f32,
    /// Recovery rate.
    pub gamma: f32,
    /// Number of simulation steps.
    pub n_steps: usize,
}

/// Compute one discrete-time SIR step (stochastic).
pub fn sir_step(
    g: &NetworkGraph,
    state: &SirState,
    config: &SirConfig,
    rng: &mut StdRng,
) -> SirState {
    let n = g.n_nodes;
    let mut new_s = state.s.clone();
    let mut new_i = state.i.clone();
    let mut new_r = state.r.clone();

    // Track newly infected this step to avoid double-counting
    let mut newly_infected = vec![false; n];
    let mut newly_recovered = vec![false; n];

    for u in 0..n {
        // Infected node recovers with probability gamma
        if state.i[u] > 0.5 && rng.random::<f32>() < config.gamma {
            newly_recovered[u] = true;
        }
        // Susceptible node gets infected by any infected neighbor
        if state.s[u] > 0.5 {
            for &(v, _) in g.neighbors(u) {
                if state.i[v] > 0.5 && rng.random::<f32>() < config.beta {
                    newly_infected[u] = true;
                    break;
                }
            }
        }
    }

    for u in 0..n {
        if newly_infected[u] {
            new_s[u] = 0.0;
            new_i[u] = 1.0;
        }
        if newly_recovered[u] {
            new_i[u] = 0.0;
            new_r[u] = 1.0;
        }
    }

    SirState {
        s: new_s,
        i: new_i,
        r: new_r,
    }
}

/// Run full SIR simulation and return state at each time step.
pub fn sir_simulate(
    g: &NetworkGraph,
    init_infected: &[usize],
    config: &SirConfig,
    rng: &mut StdRng,
) -> Vec<SirState> {
    let n = g.n_nodes;
    let mut state = SirState::new_susceptible(n);
    for &node in init_infected {
        if node < n {
            state.s[node] = 0.0;
            state.i[node] = 1.0;
        }
    }
    let mut history = Vec::with_capacity(config.n_steps + 1);
    history.push(state.clone());

    for _ in 0..config.n_steps {
        state = sir_step(g, &state, config, rng);
        history.push(state.clone());
        // Early stop if no infected nodes
        if state.i.iter().all(|&x| x < 0.5) {
            break;
        }
    }
    history
}

/// Estimate basic reproduction number R0 = β * <k²> / (`<k>` * γ).
pub fn basic_reproduction_number(g: &NetworkGraph, beta: f32, gamma: f32) -> f32 {
    let n = g.n_nodes;
    if n == 0 || gamma == 0.0 {
        return 0.0;
    }
    let degrees: Vec<f64> = (0..n).map(|v| g.degree(v) as f64).collect();
    let mean_k: f64 = degrees.iter().sum::<f64>() / n as f64;
    let mean_k2: f64 = degrees.iter().map(|k| k * k).sum::<f64>() / n as f64;
    if mean_k == 0.0 {
        return 0.0;
    }
    (beta as f64 * mean_k2 / (mean_k * gamma as f64)) as f32
}

// ---------------------------------------------------------------------------
// 8. NetworkMotifFinder
// ---------------------------------------------------------------------------

/// Find and count small subgraph motifs.
pub struct NetworkMotifFinder;

impl NetworkMotifFinder {
    /// Count total triangles in an undirected graph (each triangle counted once).
    pub fn count_triangles(g: &NetworkGraph) -> usize {
        let n = g.n_nodes;
        let mut count = 0usize;
        for u in 0..n {
            let nu: HashSet<usize> = g.adj[u].iter().map(|&(v, _)| v).collect();
            for &(v, _) in &g.adj[u] {
                if v > u {
                    let common = g.adj[v]
                        .iter()
                        .filter(|&&(w, _)| w > v && nu.contains(&w))
                        .count();
                    count += common;
                }
            }
        }
        count
    }

    /// Count 3-node induced subgraph motifs.
    /// Returns counts for: "chain" (path), "triangle", "star" (1 center, 2 leaves), "empty".
    pub fn count_3node_motifs(g: &NetworkGraph) -> HashMap<String, usize> {
        let n = g.n_nodes;
        let mut counts: HashMap<String, usize> = HashMap::new();
        counts.insert("empty".into(), 0);
        counts.insert("single_edge".into(), 0);
        counts.insert("chain".into(), 0);
        counts.insert("triangle".into(), 0);
        counts.insert("star".into(), 0);

        // Enumerate all combinations of 3 nodes
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    let eij = g.has_edge(i, j) || g.has_edge(j, i);
                    let eik = g.has_edge(i, k) || g.has_edge(k, i);
                    let ejk = g.has_edge(j, k) || g.has_edge(k, j);
                    let edge_count = eij as usize + eik as usize + ejk as usize;
                    let key = match edge_count {
                        0 => "empty",
                        1 => "single_edge",
                        2 => "chain",
                        3 => "triangle",
                        _ => "unknown",
                    };
                    *counts.entry(key.into()).or_insert(0) += 1;
                }
            }
        }
        counts
    }

    /// Brute-force 3-node adjacency-matrix isomorphism check (6 permutations).
    pub fn is_isomorphic_3node(adj: &[[bool; 3]; 3], template: &[[bool; 3]; 3]) -> bool {
        let perms: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        for perm in &perms {
            let mut matches = true;
            'outer: for i in 0..3 {
                for j in 0..3 {
                    if adj[i][j] != template[perm[i]][perm[j]] {
                        matches = false;
                        break 'outer;
                    }
                }
            }
            if matches {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// 9. TemporalNetwork — evolving network representation
// ---------------------------------------------------------------------------

/// A single temporal contact event.
#[derive(Debug, Clone)]
pub struct NetTemporalEdge {
    pub u: usize,
    pub v: usize,
    /// Timestamp.
    pub t: f64,
    /// Edge weight.
    pub w: f32,
}

/// Temporal network storing time-stamped contacts.
#[derive(Debug, Clone)]
pub struct TemporalNetwork {
    pub edges: Vec<NetTemporalEdge>,
    pub n_nodes: usize,
}

impl TemporalNetwork {
    /// Create empty temporal network.
    pub fn new(n_nodes: usize) -> Self {
        Self {
            edges: Vec::new(),
            n_nodes,
        }
    }

    /// Add a temporal edge.
    pub fn add_edge(&mut self, u: usize, v: usize, t: f64, w: f32) {
        self.edges.push(NetTemporalEdge { u, v, t, w });
    }

    /// Snapshot graph at time `t` using edges in [t - window, t].
    pub fn snapshot_at(&self, t: f64, window: f64) -> NetworkGraph {
        let mut g = NetworkGraph::new(self.n_nodes, false);
        for e in &self.edges {
            if e.t >= t - window && e.t <= t {
                g.add_edge(e.u, e.v, e.w);
            }
        }
        g
    }

    /// All contact times between nodes u and v.
    pub fn contact_sequence(&self, u: usize, v: usize) -> Vec<f64> {
        self.edges
            .iter()
            .filter(|e| (e.u == u && e.v == v) || (e.u == v && e.v == u))
            .map(|e| e.t)
            .collect()
    }

    /// Temporal PageRank: rank nodes by activity and temporal walk connectivity.
    /// `beta` is a forgetting factor (higher = more recency-weighted).
    pub fn temporal_pagerank(&self, beta: f32, max_t: f64) -> Vec<f32> {
        let n = self.n_nodes;
        if n == 0 {
            return vec![];
        }

        // Sort edges by time
        let mut sorted_edges: Vec<&NetTemporalEdge> =
            self.edges.iter().filter(|e| e.t <= max_t).collect();
        sorted_edges.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(Ordering::Equal));

        let mut score = vec![1.0f64 / n as f64; n];
        let mut last_active = vec![0.0f64; n];

        for e in sorted_edges {
            if e.u >= n || e.v >= n {
                continue;
            }
            let dt = e.t - last_active[e.u].max(last_active[e.v]);
            let decay = (-beta as f64 * dt.max(0.0)).exp();

            // Temporal walk: score propagates along the edge with decay
            let transfer = decay * score[e.u] * (e.w as f64);
            score[e.v] += transfer;

            last_active[e.u] = e.t;
            last_active[e.v] = e.t;
        }

        // Normalize
        let total: f64 = score.iter().sum();
        if total > 0.0 {
            score.iter().map(|&s| (s / total) as f32).collect()
        } else {
            vec![1.0 / n as f32; n]
        }
    }
}

// ---------------------------------------------------------------------------
// 10. NetworkStatistics
// ---------------------------------------------------------------------------

/// Summary statistics for network analysis.
pub struct NetworkStatistics;

impl NetworkStatistics {
    /// Degree distribution histogram indexed by degree value.
    pub fn degree_distribution(g: &NetworkGraph) -> Vec<usize> {
        let n = g.n_nodes;
        if n == 0 {
            return vec![];
        }
        let max_deg = (0..n).map(|v| g.degree(v)).max().unwrap_or(0);
        let mut hist = vec![0usize; max_deg + 1];
        for v in 0..n {
            let d = g.degree(v);
            hist[d] += 1;
        }
        hist
    }

    /// MLE power-law exponent α = 1 + n / Σ ln(k_i / k_min).
    pub fn power_law_exponent(degrees: &[usize]) -> f32 {
        if degrees.is_empty() {
            return 0.0;
        }
        let k_min = *degrees.iter().filter(|&&k| k >= 1).min().unwrap_or(&1);
        let valid: Vec<usize> = degrees.iter().copied().filter(|&k| k >= k_min).collect();
        let n = valid.len();
        if n == 0 {
            return 0.0;
        }
        let sum_log: f64 = valid.iter().map(|&k| (k as f64 / k_min as f64).ln()).sum();
        if sum_log <= 0.0 {
            return 1.0;
        }
        (1.0 + n as f64 / sum_log) as f32
    }

    /// Small-world coefficient σ = (C/C_rand) / (L/L_rand).
    /// C_rand ≈ `<k>`/n, L_rand ≈ ln(n)/ln(`<k>`).
    pub fn small_world_coefficient(g: &NetworkGraph) -> f32 {
        let n = g.n_nodes;
        if n < 3 {
            return 1.0;
        }
        let mean_k = (0..n).map(|v| g.degree(v)).sum::<usize>() as f64 / n as f64;
        if mean_k < 1.0 {
            return 1.0;
        }

        let c_actual = NetworkRobustness::global_clustering(g) as f64;
        let l_actual = NetworkRobustness::average_path_length(g) as f64;

        let c_rand = mean_k / n as f64;
        let l_rand = (n as f64).ln() / mean_k.ln().max(1e-10);

        if l_rand <= 0.0 || c_rand <= 0.0 || l_actual <= 0.0 {
            return 1.0;
        }
        ((c_actual / c_rand) / (l_actual / l_rand)) as f32
    }

    /// Size of the largest connected component (BFS/DFS).
    pub fn giant_component_size(g: &NetworkGraph) -> usize {
        let n = g.n_nodes;
        let mut visited = vec![false; n];
        let mut max_size = 0usize;

        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut size = 0usize;
            let mut queue: VecDeque<usize> = VecDeque::new();
            queue.push_back(start);
            visited[start] = true;
            while let Some(u) = queue.pop_front() {
                size += 1;
                for &(v, _) in &g.adj[u] {
                    if !visited[v] {
                        visited[v] = true;
                        queue.push_back(v);
                    }
                }
            }
            if size > max_size {
                max_size = size;
            }
        }
        max_size
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_graph() -> NetworkGraph {
        // Triangle 0-1-2-0, with extra node 3 connected to 0
        let mut g = NetworkGraph::new(4, false);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 0, 1.0);
        g.add_edge(0, 3, 1.0);
        g
    }

    fn path_graph(n: usize) -> NetworkGraph {
        let mut g = NetworkGraph::new(n, false);
        for i in 0..n.saturating_sub(1) {
            g.add_edge(i, i + 1, 1.0);
        }
        g
    }

    fn cycle_graph(n: usize) -> NetworkGraph {
        let mut g = NetworkGraph::new(n, false);
        for i in 0..n {
            g.add_edge(i, (i + 1) % n, 1.0);
        }
        g
    }

    // --- NetworkGraph ---
    #[test]
    fn test_graph_creation() {
        let g = NetworkGraph::new(5, false);
        assert_eq!(g.n_nodes, 5);
        assert!(!g.is_directed);
    }

    #[test]
    fn test_add_edge_undirected() {
        let mut g = NetworkGraph::new(3, false);
        g.add_edge(0, 1, 2.0);
        assert_eq!(g.degree(0), 1);
        assert_eq!(g.degree(1), 1);
        assert_eq!(g.degree(2), 0);
    }

    #[test]
    fn test_add_edge_directed() {
        let mut g = NetworkGraph::new(3, true);
        g.add_edge(0, 1, 1.0);
        assert_eq!(g.degree(0), 1);
        assert_eq!(g.degree(1), 0);
    }

    #[test]
    fn test_weighted_degree() {
        let mut g = NetworkGraph::new(2, false);
        g.add_edge(0, 1, 3.5);
        assert!((g.weighted_degree(0) - 3.5).abs() < 1e-5);
    }

    #[test]
    fn test_neighbors() {
        let g = small_graph();
        assert!(g.neighbors(0).iter().any(|&(v, _)| v == 1));
    }

    #[test]
    fn test_has_edge() {
        let g = small_graph();
        assert!(g.has_edge(0, 1));
        assert!(!g.has_edge(1, 3));
    }

    #[test]
    fn test_out_of_bounds_degree() {
        let g = NetworkGraph::new(3, false);
        assert_eq!(g.degree(10), 0);
    }

    // --- CentralityMeasures ---
    #[test]
    fn test_degree_centrality() {
        let g = small_graph(); // n=4
        let dc = CentralityMeasures::degree_centrality(&g);
        // node 0 has degree 3, norm = 3
        assert!((dc[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_closeness_centrality_complete() {
        let mut g = NetworkGraph::new(4, false);
        for i in 0..4 {
            for j in (i + 1)..4 {
                g.add_edge(i, j, 1.0);
            }
        }
        let cc = CentralityMeasures::closeness_centrality(&g);
        for &c in &cc {
            assert!(c > 0.0, "closeness should be positive on complete graph");
        }
    }

    #[test]
    fn test_betweenness_centrality_path() {
        let g = path_graph(5); // 0-1-2-3-4
        let bc = CentralityMeasures::betweenness_centrality(&g);
        // Middle node should have higher betweenness
        assert!(bc[2] > bc[0], "center node should have higher betweenness");
    }

    #[test]
    fn test_eigenvector_centrality_star() {
        // Star graph: center node 0 connected to 1,2,3,4
        let mut g = NetworkGraph::new(5, false);
        for i in 1..5 {
            g.add_edge(0, i, 1.0);
        }
        let ec = CentralityMeasures::eigenvector_centrality(&g, 100);
        // Center should have highest eigenvector centrality
        let max_ec = ec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (ec[0] - max_ec).abs() < 1e-4,
            "center has max eigenvector centrality"
        );
    }

    #[test]
    fn test_pagerank_sums_to_one() {
        let g = small_graph();
        let pr = CentralityMeasures::pagerank(&g, 0.85, 100);
        let total: f32 = pr.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "PageRank should sum to 1, got {}",
            total
        );
    }

    #[test]
    fn test_pagerank_length() {
        let g = small_graph();
        let pr = CentralityMeasures::pagerank(&g, 0.85, 50);
        assert_eq!(pr.len(), 4);
    }

    // --- CommunityDetection ---
    #[test]
    fn test_modularity_trivial() {
        let g = small_graph();
        // All in one community -> Q should be <= 0
        let partition = vec![0usize; 4];
        let q = CommunityDetection::modularity(&g, &partition);
        assert!(q.is_finite());
    }

    #[test]
    fn test_modularity_separated() {
        // Two cliques of 3 nodes each
        let mut g = NetworkGraph::new(6, false);
        for (a, b) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)] {
            g.add_edge(a, b, 1.0);
        }
        g.add_edge(2, 3, 0.01); // weak inter-community link
        let partition = vec![0, 0, 0, 1, 1, 1];
        let q = CommunityDetection::modularity(&g, &partition);
        assert!(
            q > 0.0,
            "modularity should be positive for good partition, got {}",
            q
        );
    }

    #[test]
    fn test_louvain_step() {
        let g = small_graph();
        let mut partition = vec![0usize; 4];
        let gain = CommunityDetection::louvain_step(&g, &mut partition);
        assert!(gain.is_finite());
    }

    #[test]
    fn test_spectral_clustering_output_len() {
        let g = small_graph();
        let labels = CommunityDetection::spectral_clustering_network(&g, 2);
        assert_eq!(labels.len(), 4);
    }

    #[test]
    fn test_spectral_clustering_k_bound() {
        let g = cycle_graph(6);
        let labels = CommunityDetection::spectral_clustering_network(&g, 3);
        assert!(labels.iter().all(|&l| l < 3), "labels should be < k=3");
    }

    #[test]
    fn test_label_propagation() {
        let g = small_graph();
        let mut rng = StdRng::seed_from_u64(42);
        let labels = CommunityDetection::label_propagation(&g, 10, &mut rng);
        assert_eq!(labels.len(), 4);
    }

    // --- LinkPrediction ---
    #[test]
    fn test_common_neighbors() {
        let g = small_graph();
        // 0 and 2 share neighbor 1
        let cn = LinkPrediction::common_neighbors(&g, 0, 2);
        assert!(cn >= 1, "0 and 2 should share neighbor 1, got {}", cn);
    }

    #[test]
    fn test_jaccard_coefficient() {
        let g = small_graph();
        let j = LinkPrediction::jaccard_coefficient(&g, 0, 2);
        assert!((0.0..=1.0).contains(&j));
    }

    #[test]
    fn test_adamic_adar() {
        let g = small_graph();
        let aa = LinkPrediction::adamic_adar(&g, 0, 2);
        assert!(aa >= 0.0);
    }

    #[test]
    fn test_preferential_attachment() {
        let g = small_graph();
        let pa = LinkPrediction::preferential_attachment(&g, 0, 1);
        assert_eq!(pa, (g.degree(0) * g.degree(1)) as f32);
    }

    #[test]
    fn test_embed_link_score() {
        let u = vec![1.0f32, 0.0, 0.0];
        let v = vec![1.0f32, 0.0, 0.0];
        let score = LinkPrediction::embed_link_score(&u, &v);
        assert!((score - 1.0).abs() < 1e-6);
    }

    // --- NetworkRobustness ---
    #[test]
    fn test_diameter_path() {
        let g = path_graph(5);
        assert_eq!(NetworkRobustness::diameter(&g), 4);
    }

    #[test]
    fn test_average_path_length_path() {
        let g = path_graph(3); // 0-1-2
        let apl = NetworkRobustness::average_path_length(&g);
        // Paths: (0,1)=1, (0,2)=2, (1,0)=1, (1,2)=1, (2,0)=2, (2,1)=1 -> avg=8/6=4/3
        assert!((apl - 4.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_clustering_coefficient_triangle() {
        let mut g = NetworkGraph::new(3, false);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 0, 1.0);
        // All nodes form a triangle: CC = 1.0
        let cc = NetworkRobustness::clustering_coefficient(&g, 0);
        assert!((cc - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_global_clustering() {
        let g = small_graph();
        let gc = NetworkRobustness::global_clustering(&g);
        assert!((0.0..=1.0).contains(&gc));
    }

    #[test]
    fn test_assortativity_range() {
        let g = small_graph();
        let a = NetworkRobustness::assortativity(&g);
        assert!((-1.0..=1.0).contains(&a));
    }

    // --- EpidemicModel ---
    #[test]
    fn test_sir_simulate_length() {
        let g = small_graph();
        let config = SirConfig {
            beta: 0.3,
            gamma: 0.1,
            n_steps: 10,
        };
        let mut rng = StdRng::seed_from_u64(1);
        let history = sir_simulate(&g, &[0], &config, &mut rng);
        assert!(!history.is_empty());
        assert!(history.len() <= 11);
    }

    #[test]
    fn test_sir_conservation() {
        let g = small_graph();
        let config = SirConfig {
            beta: 0.5,
            gamma: 0.2,
            n_steps: 5,
        };
        let mut rng = StdRng::seed_from_u64(2);
        let history = sir_simulate(&g, &[0], &config, &mut rng);
        // Each node's (S+I+R) should stay ~1.0
        for state in &history {
            for v in 0..4 {
                let total = state.s[v] + state.i[v] + state.r[v];
                assert!((total - 1.0).abs() < 1e-5, "S+I+R != 1 for node {}", v);
            }
        }
    }

    #[test]
    fn test_basic_reproduction_number() {
        let g = cycle_graph(10);
        let r0 = basic_reproduction_number(&g, 0.3, 0.1);
        assert!(r0 > 0.0);
    }

    // --- NetworkMotifFinder ---
    #[test]
    fn test_count_triangles() {
        let mut g = NetworkGraph::new(3, false);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(0, 2, 1.0);
        assert_eq!(NetworkMotifFinder::count_triangles(&g), 1);
    }

    #[test]
    fn test_count_triangles_path() {
        let g = path_graph(4);
        assert_eq!(NetworkMotifFinder::count_triangles(&g), 0);
    }

    #[test]
    fn test_count_3node_motifs() {
        let mut g = NetworkGraph::new(3, false);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(0, 2, 1.0);
        let counts = NetworkMotifFinder::count_3node_motifs(&g);
        assert_eq!(*counts.get("triangle").unwrap_or(&0), 1);
    }

    #[test]
    fn test_isomorphic_3node_identity() {
        let adj = [
            [false, true, false],
            [true, false, true],
            [false, true, false],
        ];
        assert!(NetworkMotifFinder::is_isomorphic_3node(&adj, &adj));
    }

    #[test]
    fn test_isomorphic_3node_permuted() {
        // Chain: 0-1-2 vs 2-1-0 (same topology)
        let adj1 = [
            [false, true, false],
            [true, false, true],
            [false, true, false],
        ];
        let adj2 = [
            [false, true, false],
            [true, false, true],
            [false, true, false],
        ];
        assert!(NetworkMotifFinder::is_isomorphic_3node(&adj1, &adj2));
    }

    // --- TemporalNetwork ---
    #[test]
    fn test_temporal_snapshot() {
        let mut tn = TemporalNetwork::new(4);
        tn.add_edge(0, 1, 1.0, 1.0);
        tn.add_edge(1, 2, 2.0, 1.0);
        tn.add_edge(2, 3, 5.0, 1.0);
        let snap = tn.snapshot_at(2.5, 2.0);
        // Should include edges at t=1 and t=2, not t=5
        assert!(snap.has_edge(0, 1) || snap.has_edge(1, 0));
        assert!(!snap.has_edge(2, 3) && !snap.has_edge(3, 2));
    }

    #[test]
    fn test_contact_sequence() {
        let mut tn = TemporalNetwork::new(3);
        tn.add_edge(0, 1, 1.0, 1.0);
        tn.add_edge(0, 1, 3.0, 1.0);
        tn.add_edge(1, 2, 2.0, 1.0);
        let contacts = tn.contact_sequence(0, 1);
        assert_eq!(contacts.len(), 2);
    }

    #[test]
    fn test_temporal_pagerank() {
        let mut tn = TemporalNetwork::new(4);
        tn.add_edge(0, 1, 1.0, 1.0);
        tn.add_edge(1, 2, 2.0, 1.0);
        tn.add_edge(2, 3, 3.0, 1.0);
        let pr = tn.temporal_pagerank(0.1, 4.0);
        assert_eq!(pr.len(), 4);
        let total: f32 = pr.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "temporal PR should sum to 1, got {}",
            total
        );
    }

    // --- NetworkStatistics ---
    #[test]
    fn test_degree_distribution() {
        let g = small_graph(); // degrees: 3,2,2,1
        let dist = NetworkStatistics::degree_distribution(&g);
        assert_eq!(dist[1], 1); // one node with degree 1
        assert_eq!(dist[2], 2); // two nodes with degree 2
        assert_eq!(dist[3], 1); // one node with degree 3
    }

    #[test]
    fn test_power_law_exponent() {
        // Generate a degree sequence that should give ~2.5
        let degrees: Vec<usize> = (1..=50).collect();
        let alpha = NetworkStatistics::power_law_exponent(&degrees);
        assert!(alpha > 1.0 && alpha < 4.0, "alpha={}", alpha);
    }

    #[test]
    fn test_giant_component_path() {
        let g = path_graph(5);
        assert_eq!(NetworkStatistics::giant_component_size(&g), 5);
    }

    #[test]
    fn test_giant_component_disconnected() {
        let mut g = NetworkGraph::new(6, false);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        // Nodes 3,4,5 isolated (no edges)
        assert_eq!(NetworkStatistics::giant_component_size(&g), 3);
    }

    #[test]
    fn test_small_world_coefficient() {
        // Use a graph with triangles so clustering > 0
        let mut g = NetworkGraph::new(6, false);
        // Two triangles 0-1-2 and 3-4-5 connected by edge 2-3
        for (a, b) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
            g.add_edge(a, b, 1.0);
        }
        let sw = NetworkStatistics::small_world_coefficient(&g);
        assert!(
            sw.is_finite(),
            "small_world_coefficient should be finite, got {}",
            sw
        );
    }

    // --- Node2Vec ---
    #[test]
    fn test_biased_random_walk_length() {
        let g = cycle_graph(10);
        let mut rng = StdRng::seed_from_u64(7);
        let walk = biased_random_walk(&g, 0, 20, 1.0, 1.0, &mut rng);
        assert!(walk.len() <= 20);
        assert!(!walk.is_empty());
    }

    #[test]
    fn test_node2vec_output_shape() {
        let g = cycle_graph(6);
        let config = Node2VecConfig {
            walk_len: 10,
            n_walks: 2,
            p: 1.0,
            q: 1.0,
            embed_dim: 8,
            window: 2,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let embeds = node2vec_train(&g, &config, &mut rng);
        assert_eq!(embeds.len(), 6);
        assert_eq!(embeds[0].len(), 8);
    }

    #[test]
    fn test_skip_gram_update() {
        let mut embeddings: Vec<Vec<f32>> = vec![vec![0.1; 4]; 5];
        let orig_center = embeddings[0].clone();
        skip_gram_update(&mut embeddings, 0, &[1, 2], &[3, 4], 0.01);
        // Embeddings should have changed
        assert!(embeddings[0] != orig_center || embeddings[1] != vec![0.1f32; 4]);
    }
}
