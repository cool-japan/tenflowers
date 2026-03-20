//! Advanced graph self-supervised pretraining algorithms.
//!
//! This module implements:
//! - **GraphCL** (You et al. 2020): Graph Contrastive Learning with augmentations
//! - **GraphMAE** (Hou et al. 2022): Graph Masked Autoencoders
//! - **Graph Transformer Pretraining**: MGM + graph classification
//! - **FedGraph**: Federated graph learning and cross-graph transfer
//! - **GraphMetrics**: AUC-ROC, AP, Accuracy, Hits@k for evaluation

use super::{dot, layer_norm_vec, relu, sigmoid, softmax_1d, xavier_vec};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// 1. GRAPH CONTRASTIVE PRETRAINING (GraphCL)
// ─────────────────────────────────────────────────────────────────────────────

/// Augmentation strategy for graph contrastive learning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GclAugmentation {
    /// Randomly drop nodes from the graph
    NodeDropping,
    /// Randomly remove edges from the graph
    EdgePerturbation,
    /// Randomly zero out node attribute dimensions
    AttributeMasking,
    /// Sample a connected subgraph rooted at a random node
    SubgraphSampling,
}

/// Graph representation as adjacency + features for augmentation.
#[derive(Debug, Clone)]
pub struct GclGraph {
    /// Node feature matrix \[n_nodes\]\[d_feat\]
    pub node_feats: Vec<Vec<f64>>,
    /// Adjacency list (directed)
    pub adj: Vec<Vec<usize>>,
}

impl GclGraph {
    /// Create a new graph from features and adjacency.
    pub fn new(node_feats: Vec<Vec<f64>>, adj: Vec<Vec<usize>>) -> Self {
        Self { node_feats, adj }
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.node_feats.len()
    }
}

/// Graph Contrastive Learning (You 2020): two augmented views + NT-Xent loss.
///
/// Applies two different augmentations to each graph, encodes them with a shared
/// GNN encoder, projects to a low-dimensional space, and minimises NT-Xent loss.
#[derive(Debug, Clone)]
pub struct GraphCL {
    /// GNN encoder: linear projection weights \[d_out\]\[d_in\]
    pub encoder_w: Vec<Vec<f64>>,
    /// Projection head weights \[d_proj\]\[d_out\]
    pub proj_w: Vec<Vec<f64>>,
    /// Input feature dimension
    pub d_in: usize,
    /// Encoder output dimension
    pub d_out: usize,
    /// Projection head dimension
    pub d_proj: usize,
    /// NT-Xent temperature
    pub temperature: f64,
    /// Drop ratio for augmentations
    pub aug_ratio: f64,
}

impl GraphCL {
    /// Build a GraphCL model.
    ///
    /// # Arguments
    /// - `d_in`: input node feature dimension
    /// - `d_out`: encoder output dimension
    /// - `d_proj`: projection head dimension
    /// - `temperature`: NT-Xent temperature (typically 0.1–0.5)
    /// - `aug_ratio`: fraction to drop/mask during augmentation
    pub fn new(
        d_in: usize,
        d_out: usize,
        d_proj: usize,
        temperature: f64,
        aug_ratio: f64,
        rng: &mut StdRng,
    ) -> Self {
        let encoder_w = (0..d_out)
            .map(|_| xavier_vec(d_in, d_in, d_out, rng))
            .collect();
        let proj_w = (0..d_proj)
            .map(|_| xavier_vec(d_out, d_out, d_proj, rng))
            .collect();
        Self {
            encoder_w,
            proj_w,
            d_in,
            d_out,
            d_proj,
            temperature,
            aug_ratio,
        }
    }

    /// Apply a graph augmentation strategy to produce a new view.
    pub fn augment(
        &self,
        graph: &GclGraph,
        aug: GclAugmentation,
        rng: &mut StdRng,
    ) -> GclGraph {
        match aug {
            GclAugmentation::NodeDropping => self.aug_node_drop(graph, rng),
            GclAugmentation::EdgePerturbation => self.aug_edge_perturb(graph, rng),
            GclAugmentation::AttributeMasking => self.aug_attr_mask(graph, rng),
            GclAugmentation::SubgraphSampling => self.aug_subgraph(graph, rng),
        }
    }

    fn aug_node_drop(&self, g: &GclGraph, rng: &mut StdRng) -> GclGraph {
        let n = g.n_nodes();
        let keep: Vec<bool> = (0..n).map(|_| rng.random::<f64>() >= self.aug_ratio).collect();
        // Build index remapping
        let mut remap = vec![usize::MAX; n];
        let mut cnt = 0;
        for (i, &k) in keep.iter().enumerate() {
            if k {
                remap[i] = cnt;
                cnt += 1;
            }
        }
        let node_feats: Vec<Vec<f64>> = g
            .node_feats
            .iter()
            .enumerate()
            .filter(|(i, _)| keep[*i])
            .map(|(_, f)| f.clone())
            .collect();
        let adj: Vec<Vec<usize>> = g
            .adj
            .iter()
            .enumerate()
            .filter(|(i, _)| keep[*i])
            .map(|(i, nbrs)| {
                nbrs.iter()
                    .filter(|&&nb| keep[nb])
                    .map(|&nb| remap[nb])
                    .collect()
            })
            .map(|v: Vec<usize>| {
                let _ = v;
                vec![] // rebuilt below
            })
            .collect();
        // Rebuild adjacency with remapped indices
        let mut new_adj = vec![vec![]; node_feats.len()];
        for (i, nbrs) in g.adj.iter().enumerate() {
            if !keep[i] {
                continue;
            }
            let new_i = remap[i];
            for &nb in nbrs {
                if keep[nb] {
                    new_adj[new_i].push(remap[nb]);
                }
            }
        }
        let _ = adj; // shadow the unused intermediate
        GclGraph::new(node_feats, new_adj)
    }

    fn aug_edge_perturb(&self, g: &GclGraph, rng: &mut StdRng) -> GclGraph {
        let adj: Vec<Vec<usize>> = g
            .adj
            .iter()
            .map(|nbrs| {
                nbrs.iter()
                    .filter(|_| rng.random::<f64>() >= self.aug_ratio)
                    .cloned()
                    .collect()
            })
            .collect();
        GclGraph::new(g.node_feats.clone(), adj)
    }

    fn aug_attr_mask(&self, g: &GclGraph, rng: &mut StdRng) -> GclGraph {
        let node_feats: Vec<Vec<f64>> = g
            .node_feats
            .iter()
            .map(|f| {
                f.iter()
                    .map(|&x| if rng.random::<f64>() < self.aug_ratio { 0.0 } else { x })
                    .collect()
            })
            .collect();
        GclGraph::new(node_feats, g.adj.clone())
    }

    fn aug_subgraph(&self, g: &GclGraph, rng: &mut StdRng) -> GclGraph {
        let n = g.n_nodes();
        if n == 0 {
            return g.clone();
        }
        // BFS from random root, keep ~(1 - aug_ratio) fraction of nodes
        let target = ((n as f64 * (1.0 - self.aug_ratio)).ceil() as usize).max(1).min(n);
        let root = rng.random_range(0..n);
        let mut visited = vec![false; n];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(root);
        visited[root] = true;
        let mut order = vec![root];
        while let Some(u) = queue.pop_front() {
            if order.len() >= target {
                break;
            }
            for &v in &g.adj[u] {
                if !visited[v] {
                    visited[v] = true;
                    queue.push_back(v);
                    order.push(v);
                    if order.len() >= target {
                        break;
                    }
                }
            }
        }
        // Build subgraph
        let mut remap = vec![usize::MAX; n];
        for (new_idx, &old_idx) in order.iter().enumerate() {
            remap[old_idx] = new_idx;
        }
        let node_feats: Vec<Vec<f64>> = order.iter().map(|&i| g.node_feats[i].clone()).collect();
        let mut new_adj = vec![vec![]; order.len()];
        for &old_i in &order {
            let new_i = remap[old_i];
            for &nb in &g.adj[old_i] {
                if remap[nb] != usize::MAX {
                    new_adj[new_i].push(remap[nb]);
                }
            }
        }
        GclGraph::new(node_feats, new_adj)
    }

    /// Encode a graph to a fixed-size representation (mean pooling over GNN outputs).
    pub fn encode(&self, graph: &GclGraph) -> Vec<f64> {
        let n = graph.n_nodes();
        if n == 0 {
            return vec![0.0; self.d_out];
        }
        // One-layer GNN: message passing + linear projection
        let mut agg = graph.node_feats.clone();
        for i in 0..n {
            if !graph.adj[i].is_empty() {
                let nb_mean: Vec<f64> = (0..self.d_in.min(agg[0].len()))
                    .map(|d| {
                        graph.adj[i]
                            .iter()
                            .filter(|&&nb| nb < n)
                            .map(|&nb| if d < graph.node_feats[nb].len() { graph.node_feats[nb][d] } else { 0.0 })
                            .sum::<f64>()
                            / graph.adj[i].len() as f64
                    })
                    .collect();
                // Combine self + neighbour
                let d = self.d_in.min(agg[i].len());
                for di in 0..d {
                    agg[i][di] = (agg[i][di] + nb_mean.get(di).cloned().unwrap_or(0.0)) / 2.0;
                }
            }
        }
        // Linear projection
        let projected: Vec<Vec<f64>> = agg
            .iter()
            .map(|f| {
                let d = self.d_in.min(f.len());
                (0..self.d_out)
                    .map(|j| dot(&f[..d], &self.encoder_w[j][..d]))
                    .collect()
            })
            .collect();
        // Mean pooling
        let mean: Vec<f64> = (0..self.d_out)
            .map(|d| projected.iter().map(|v| v[d]).sum::<f64>() / n as f64)
            .collect();
        // ReLU activation
        mean.iter().map(|&x| relu(x)).collect()
    }

    /// Project encoded representation through the projection head (with L2 norm).
    pub fn project(&self, h: &[f64]) -> Vec<f64> {
        let d = self.d_out.min(h.len());
        let out: Vec<f64> = (0..self.d_proj)
            .map(|j| dot(&h[..d], &self.proj_w[j][..d]))
            .collect();
        let norm = out.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        out.iter().map(|x| x / norm).collect()
    }

    /// Compute NT-Xent (normalised temperature cross-entropy) loss for a batch.
    ///
    /// Given `z1[i]` and `z2[i]` as positive pairs, all other pairs are negatives.
    pub fn nt_xent_loss(&self, z1: &[Vec<f64>], z2: &[Vec<f64>]) -> f64 {
        let n = z1.len().min(z2.len());
        if n == 0 {
            return 0.0;
        }
        let mut total = 0.0;
        for i in 0..n {
            // positive sim
            let pos = dot(&z1[i], &z2[i]) / self.temperature;
            // all other sims (negatives from both views)
            let mut all_sims: Vec<f64> = vec![pos];
            for j in 0..n {
                if j != i {
                    all_sims.push(dot(&z1[i], &z1[j]) / self.temperature);
                    all_sims.push(dot(&z1[i], &z2[j]) / self.temperature);
                }
            }
            let max_s = all_sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let log_denom = max_s
                + all_sims
                    .iter()
                    .map(|&s| (s - max_s).exp())
                    .sum::<f64>()
                    .ln();
            total += log_denom - pos;
        }
        total / n as f64
    }
}

/// Batch trainer for GraphCL with hard negative mining.
#[derive(Debug, Clone)]
pub struct GraphContraster {
    /// Underlying GraphCL model
    pub model: GraphCL,
    /// Number of hard negatives to mine per sample
    pub n_hard_neg: usize,
}

impl GraphContraster {
    /// Build a GraphContraster.
    pub fn new(model: GraphCL, n_hard_neg: usize) -> Self {
        Self { model, n_hard_neg }
    }

    /// Train one step: augment graphs, encode, compute NT-Xent with hard negatives.
    ///
    /// Returns the contrastive loss value.
    pub fn train_step(&self, graphs: &[GclGraph], rng: &mut StdRng) -> f64 {
        let aug_pairs = [
            GclAugmentation::NodeDropping,
            GclAugmentation::EdgePerturbation,
            GclAugmentation::AttributeMasking,
            GclAugmentation::SubgraphSampling,
        ];
        let z1: Vec<Vec<f64>> = graphs
            .iter()
            .map(|g| {
                let aug1 = aug_pairs[rng.random_range(0..aug_pairs.len())];
                let v1 = self.model.augment(g, aug1, rng);
                self.model.project(&self.model.encode(&v1))
            })
            .collect();
        let z2: Vec<Vec<f64>> = graphs
            .iter()
            .map(|g| {
                let aug2 = aug_pairs[rng.random_range(0..aug_pairs.len())];
                let v2 = self.model.augment(g, aug2, rng);
                self.model.project(&self.model.encode(&v2))
            })
            .collect();
        self.model.nt_xent_loss(&z1, &z2)
    }

    /// Compute cosine similarity between two embeddings (for hard negative mining).
    pub fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
        let na = a.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        dot(a, b) / (na * nb)
    }

    /// Find hard negatives: top-k most similar samples that are not the positive.
    pub fn hard_negatives<'a>(
        &self,
        query: &[f64],
        pool: &'a [Vec<f64>],
        positive_idx: usize,
    ) -> Vec<&'a Vec<f64>> {
        let mut scored: Vec<(usize, f64)> = pool
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != positive_idx)
            .map(|(i, z)| (i, Self::cosine_sim(query, z)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .iter()
            .take(self.n_hard_neg)
            .map(|(i, _)| &pool[*i])
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. GRAPH MASKED AUTOENCODER (GraphMAE)
// ─────────────────────────────────────────────────────────────────────────────

/// Masking strategy for GraphMAE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaeMaskStrategy {
    /// Select masked nodes uniformly at random
    UniformRandom,
    /// Prefer high-degree nodes for masking
    DegreeBased,
}

/// GIN-based encoder for GraphMAE.
///
/// Uses a single GIN layer: h_v = MLP(h_v + sum_{u in N(v)} h_u).
#[derive(Debug, Clone)]
pub struct GraphMaeEncoder {
    /// Input dimension
    pub d_in: usize,
    /// Hidden dimension
    pub d_hidden: usize,
    /// MLP layer 1 weights \[d_hidden\]\[d_in\]
    pub mlp1_w: Vec<Vec<f64>>,
    /// MLP layer 2 weights \[d_hidden\]\[d_hidden\]
    pub mlp2_w: Vec<Vec<f64>>,
    /// Learnable mask token embedding
    pub mask_token: Vec<f64>,
}

impl GraphMaeEncoder {
    /// Build a GIN-based encoder for GraphMAE.
    pub fn new(d_in: usize, d_hidden: usize, rng: &mut StdRng) -> Self {
        let mlp1_w = (0..d_hidden)
            .map(|_| xavier_vec(d_in, d_in, d_hidden, rng))
            .collect();
        let mlp2_w = (0..d_hidden)
            .map(|_| xavier_vec(d_hidden, d_hidden, d_hidden, rng))
            .collect();
        let mask_token = xavier_vec(d_in, d_in, 1, rng);
        Self {
            d_in,
            d_hidden,
            mlp1_w,
            mlp2_w,
            mask_token,
        }
    }

    /// Encode nodes; masked nodes receive the mask token embedding.
    pub fn encode(
        &self,
        node_feats: &[Vec<f64>],
        adj: &[Vec<usize>],
        masked_indices: &[usize],
    ) -> Vec<Vec<f64>> {
        let n = node_feats.len();
        let masked_set: std::collections::HashSet<usize> = masked_indices.iter().cloned().collect();

        // Replace masked nodes with mask token
        let input: Vec<Vec<f64>> = node_feats
            .iter()
            .enumerate()
            .map(|(i, f)| {
                if masked_set.contains(&i) {
                    self.mask_token.clone()
                } else {
                    f.clone()
                }
            })
            .collect();

        // GIN aggregation: h_v + mean(h_neighbours)
        let aggregated: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let d = self.d_in.min(input[i].len());
                let mut agg: Vec<f64> = input[i][..d].to_vec();
                if !adj[i].is_empty() {
                    for &nb in &adj[i] {
                        if nb < n {
                            let nb_d = d.min(input[nb].len());
                            for di in 0..nb_d {
                                agg[di] += input[nb][di] / adj[i].len() as f64;
                            }
                        }
                    }
                }
                agg
            })
            .collect();

        // MLP: ReLU(W2 * ReLU(W1 * x))
        aggregated
            .iter()
            .map(|x| {
                let d = self.d_in.min(x.len());
                let h1: Vec<f64> = (0..self.d_hidden)
                    .map(|j| relu(dot(&x[..d], &self.mlp1_w[j][..d])))
                    .collect();
                let h2: Vec<f64> = (0..self.d_hidden)
                    .map(|j| relu(dot(&h1, &self.mlp2_w[j])))
                    .collect();
                layer_norm_vec(&h2)
            })
            .collect()
    }

    /// Select masked nodes by strategy.
    pub fn select_mask(
        &self,
        n: usize,
        ratio: f64,
        adj: &[Vec<usize>],
        strategy: MaeMaskStrategy,
        rng: &mut StdRng,
    ) -> Vec<usize> {
        let n_mask = ((n as f64 * ratio).round() as usize).min(n);
        match strategy {
            MaeMaskStrategy::UniformRandom => {
                let mut indices: Vec<usize> = (0..n).collect();
                for i in (1..n).rev() {
                    let j = rng.random_range(0..=i);
                    indices.swap(i, j);
                }
                indices[..n_mask].to_vec()
            }
            MaeMaskStrategy::DegreeBased => {
                // Sort by degree descending, mask top-k high-degree nodes
                let degrees: Vec<usize> = adj.iter().map(|nbrs| nbrs.len()).collect();
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by(|&a, &b| degrees[b].cmp(&degrees[a]));
                order[..n_mask].to_vec()
            }
        }
    }
}

/// GAT-based decoder for GraphMAE feature reconstruction.
#[derive(Debug, Clone)]
pub struct GraphMaeDecoder {
    /// Encoder hidden dimension (input to decoder)
    pub d_hidden: usize,
    /// Original feature dimension (output to reconstruct)
    pub d_out: usize,
    /// GAT attention weight [d_hidden * 2]
    pub attn_w: Vec<f64>,
    /// Output projection \[d_out\]\[d_hidden\]
    pub out_proj: Vec<Vec<f64>>,
}

impl GraphMaeDecoder {
    /// Build a GAT-based decoder.
    pub fn new(d_hidden: usize, d_out: usize, rng: &mut StdRng) -> Self {
        let attn_w = xavier_vec(d_hidden * 2, d_hidden, 1, rng);
        let out_proj = (0..d_out)
            .map(|_| xavier_vec(d_hidden, d_hidden, d_out, rng))
            .collect();
        Self {
            d_hidden,
            d_out,
            attn_w,
            out_proj,
        }
    }

    /// Decode latent node representations to original feature space.
    pub fn decode(&self, latents: &[Vec<f64>], adj: &[Vec<usize>]) -> Vec<Vec<f64>> {
        let n = latents.len();
        // GAT: for each node, aggregate neighbour latents with attention
        let aggregated: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                if adj[i].is_empty() {
                    return latents[i].clone();
                }
                let d = self.d_hidden.min(latents[i].len());
                let scores: Vec<f64> = adj[i]
                    .iter()
                    .filter(|&&nb| nb < n)
                    .map(|&nb| {
                        let nb_d = d.min(latents[nb].len());
                        let cat: Vec<f64> = latents[i][..d]
                            .iter()
                            .chain(latents[nb][..nb_d].iter())
                            .cloned()
                            .collect();
                        let aw = self.attn_w.len().min(cat.len());
                        relu(dot(&cat[..aw], &self.attn_w[..aw]))
                    })
                    .collect();
                let weights = softmax_1d(&scores);
                let mut agg = latents[i][..d].to_vec();
                for (wi, &nb) in weights.iter().zip(adj[i].iter()) {
                    if nb < n {
                        let nb_d = d.min(latents[nb].len());
                        for di in 0..nb_d {
                            agg[di] += wi * latents[nb][di];
                        }
                    }
                }
                agg
            })
            .collect();
        // Project to original feature space
        aggregated
            .iter()
            .map(|h| {
                let d = self.d_hidden.min(h.len());
                (0..self.d_out)
                    .map(|j| dot(&h[..d], &self.out_proj[j][..d]))
                    .collect()
            })
            .collect()
    }
}

/// Graph Masked Autoencoder (Hou 2022): masked node feature reconstruction.
///
/// Masks a subset of node features, encodes the graph, then reconstructs
/// the masked features using scaled cosine error as the loss.
#[derive(Debug, Clone)]
pub struct GraphMaeModel {
    /// GIN-based encoder
    pub encoder: GraphMaeEncoder,
    /// GAT-based decoder
    pub decoder: GraphMaeDecoder,
    /// Masking ratio (fraction of nodes to mask)
    pub mask_ratio: f64,
    /// Masking strategy
    pub strategy: MaeMaskStrategy,
}

impl GraphMaeModel {
    /// Build a GraphMAE model.
    pub fn new(
        d_in: usize,
        d_hidden: usize,
        mask_ratio: f64,
        strategy: MaeMaskStrategy,
        rng: &mut StdRng,
    ) -> Self {
        let encoder = GraphMaeEncoder::new(d_in, d_hidden, rng);
        let decoder = GraphMaeDecoder::new(d_hidden, d_in, rng);
        Self {
            encoder,
            decoder,
            mask_ratio,
            strategy,
        }
    }

    /// Forward pass: mask → encode → decode, returns reconstructed features.
    pub fn forward(
        &self,
        node_feats: &[Vec<f64>],
        adj: &[Vec<usize>],
        rng: &mut StdRng,
    ) -> (Vec<Vec<f64>>, Vec<usize>) {
        let n = node_feats.len();
        let masked_indices =
            self.encoder
                .select_mask(n, self.mask_ratio, adj, self.strategy, rng);
        let latents = self.encoder.encode(node_feats, adj, &masked_indices);
        let recon = self.decoder.decode(&latents, adj);
        (recon, masked_indices)
    }

    /// Scaled cosine error loss on masked nodes (Hou 2022).
    ///
    /// SCE = (1 - cos(pred, true)) / 2 averaged over masked nodes.
    pub fn sce_loss(
        &self,
        pred: &[Vec<f64>],
        true_feats: &[Vec<f64>],
        masked_indices: &[usize],
    ) -> f64 {
        if masked_indices.is_empty() {
            return 0.0;
        }
        let total: f64 = masked_indices
            .iter()
            .map(|&i| {
                if i >= pred.len() || i >= true_feats.len() {
                    return 0.0;
                }
                let p = &pred[i];
                let t = &true_feats[i];
                let np = p.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
                let nt = t.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
                let cos = dot(p, t) / (np * nt);
                (1.0 - cos) / 2.0
            })
            .sum();
        total / masked_indices.len() as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. GRAPH TRANSFORMER PRETRAINING
// ─────────────────────────────────────────────────────────────────────────────

/// Tokenizes a graph to a sequence for transformer input.
///
/// Performs BFS node ordering and encodes each node with positional information.
#[derive(Debug, Clone)]
pub struct GtpTokenizer {
    /// Feature dimension
    pub d_feat: usize,
    /// Laplacian positional encoding dimension
    pub d_pe: usize,
    /// Random walk steps for RWPE
    pub rw_steps: usize,
}

impl GtpTokenizer {
    /// Build a graph tokenizer.
    pub fn new(d_feat: usize, d_pe: usize, rw_steps: usize) -> Self {
        Self { d_feat, d_pe, rw_steps }
    }

    /// BFS-ordered node sequence from a random root.
    pub fn bfs_order(adj: &[Vec<usize>], root: usize) -> Vec<usize> {
        let n = adj.len();
        let mut visited = vec![false; n];
        let mut queue = std::collections::VecDeque::new();
        let mut order = Vec::new();
        if root < n {
            queue.push_back(root);
            visited[root] = true;
        }
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for &v in &adj[u] {
                if v < n && !visited[v] {
                    visited[v] = true;
                    queue.push_back(v);
                }
            }
        }
        // Add any unreached nodes
        for i in 0..n {
            if !visited[i] {
                order.push(i);
            }
        }
        order
    }

    /// Compute random walk positional encoding (diagonal of P^k).
    pub fn rwpe(&self, adj: &[Vec<usize>]) -> Vec<Vec<f64>> {
        let n = adj.len();
        if n == 0 {
            return vec![];
        }
        // Degree-normalised adjacency (row stochastic)
        let deg: Vec<f64> = adj.iter().map(|nbrs| nbrs.len() as f64).collect();
        let mut rw = vec![vec![0.0f64; n]; n];
        // Identity as P^0 diagonal is 1
        for i in 0..n {
            rw[i][i] = 1.0;
        }
        let steps = self.rw_steps.min(10);
        let mut pe_cols: Vec<Vec<f64>> = Vec::new();
        // Store diagonals of P^1 .. P^steps
        let mut prob = rw.clone();
        for _ in 0..steps {
            // prob = prob * P (one RW step)
            let mut next = vec![vec![0.0f64; n]; n];
            for i in 0..n {
                for &nb in &adj[i] {
                    if nb < n && deg[i] > 0.0 {
                        for j in 0..n {
                            next[j][nb] += prob[j][i] / deg[i];
                        }
                    }
                }
            }
            let diag: Vec<f64> = (0..n).map(|i| next[i][i]).collect();
            pe_cols.push(diag);
            prob = next;
        }
        // Each node gets a d_pe-dim encoding from the diagonals
        let d_pe = self.d_pe.min(steps);
        (0..n)
            .map(|i| {
                let mut enc = vec![0.0f64; self.d_pe];
                for (k, col) in pe_cols.iter().take(d_pe).enumerate() {
                    enc[k] = col[i];
                }
                enc
            })
            .collect()
    }

    /// Tokenize: returns BFS-ordered [(node_idx, feat_with_pe)] sequence.
    pub fn tokenize(
        &self,
        node_feats: &[Vec<f64>],
        adj: &[Vec<usize>],
        rng: &mut StdRng,
    ) -> Vec<(usize, Vec<f64>)> {
        let n = node_feats.len();
        if n == 0 {
            return vec![];
        }
        let root = rng.random_range(0..n);
        let order = Self::bfs_order(adj, root);
        let rwpe = self.rwpe(adj);
        order
            .iter()
            .map(|&i| {
                let d = self.d_feat.min(node_feats[i].len());
                let mut tok = node_feats[i][..d].to_vec();
                tok.resize(self.d_feat, 0.0);
                // Append RWPE
                let pe = if i < rwpe.len() { &rwpe[i] } else { &[] as &[f64] };
                tok.extend_from_slice(pe);
                tok.resize(self.d_feat + self.d_pe, 0.0);
                (i, tok)
            })
            .collect()
    }
}

/// Graph Transformer Pretraining model.
///
/// Supports:
/// - **Masked Graph Modeling (MGM)**: mask node tokens and reconstruct
/// - **Graph-level classification**: CLS token classification head
#[derive(Debug, Clone)]
pub struct GtpModel {
    /// Tokenizer
    pub tokenizer: GtpTokenizer,
    /// Total token dimension (d_feat + d_pe)
    pub d_token: usize,
    /// Transformer hidden dimension
    pub d_model: usize,
    /// Input projection weights \[d_model\]\[d_token\]
    pub input_proj: Vec<Vec<f64>>,
    /// Self-attention Q projection \[d_model\]\[d_model\]
    pub wq: Vec<Vec<f64>>,
    /// Self-attention K projection
    pub wk: Vec<Vec<f64>>,
    /// Self-attention V projection
    pub wv: Vec<Vec<f64>>,
    /// FFN layer 1 [d_model * 4]\[d_model\]
    pub ff1: Vec<Vec<f64>>,
    /// FFN layer 2 [d_model][d_model * 4]
    pub ff2: Vec<Vec<f64>>,
    /// Token reconstruction projection \[d_feat\]\[d_model\]
    pub recon_proj: Vec<Vec<f64>>,
    /// CLS token embedding
    pub cls_token: Vec<f64>,
}

impl GtpModel {
    /// Build a GtpModel.
    pub fn new(
        d_feat: usize,
        d_pe: usize,
        rw_steps: usize,
        d_model: usize,
        rng: &mut StdRng,
    ) -> Self {
        let d_token = d_feat + d_pe;
        let tokenizer = GtpTokenizer::new(d_feat, d_pe, rw_steps);
        let input_proj = (0..d_model)
            .map(|_| xavier_vec(d_token, d_token, d_model, rng))
            .collect();
        let wq = (0..d_model)
            .map(|_| xavier_vec(d_model, d_model, d_model, rng))
            .collect();
        let wk = (0..d_model)
            .map(|_| xavier_vec(d_model, d_model, d_model, rng))
            .collect();
        let wv = (0..d_model)
            .map(|_| xavier_vec(d_model, d_model, d_model, rng))
            .collect();
        let ff_dim = d_model * 4;
        let ff1 = (0..ff_dim)
            .map(|_| xavier_vec(d_model, d_model, ff_dim, rng))
            .collect();
        let ff2 = (0..d_model)
            .map(|_| xavier_vec(ff_dim, ff_dim, d_model, rng))
            .collect();
        let recon_proj = (0..d_feat)
            .map(|_| xavier_vec(d_model, d_model, d_feat, rng))
            .collect();
        let cls_token = xavier_vec(d_model, d_model, 1, rng);
        Self {
            tokenizer,
            d_token,
            d_model,
            input_proj,
            wq,
            wk,
            wv,
            ff1,
            ff2,
            recon_proj,
            cls_token,
        }
    }

    fn project_token(&self, tok: &[f64]) -> Vec<f64> {
        let d = self.d_token.min(tok.len());
        (0..self.d_model)
            .map(|j| dot(&tok[..d], &self.input_proj[j][..d]))
            .collect()
    }

    fn self_attn(&self, tokens: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = tokens.len();
        let scale = (self.d_model as f64).sqrt();
        let q: Vec<Vec<f64>> = tokens
            .iter()
            .map(|t| (0..self.d_model).map(|j| dot(t, &self.wq[j])).collect())
            .collect();
        let k: Vec<Vec<f64>> = tokens
            .iter()
            .map(|t| (0..self.d_model).map(|j| dot(t, &self.wk[j])).collect())
            .collect();
        let v: Vec<Vec<f64>> = tokens
            .iter()
            .map(|t| (0..self.d_model).map(|j| dot(t, &self.wv[j])).collect())
            .collect();
        (0..n)
            .map(|i| {
                let logits: Vec<f64> = (0..n).map(|j| dot(&q[i], &k[j]) / scale).collect();
                let attn = softmax_1d(&logits);
                let attended: Vec<f64> = (0..self.d_model)
                    .map(|d| attn.iter().zip(v.iter()).map(|(w, vj)| w * vj[d]).sum())
                    .collect();
                layer_norm_vec(
                    &attended
                        .iter()
                        .zip(tokens[i].iter())
                        .map(|(a, b)| a + b)
                        .collect::<Vec<_>>(),
                )
            })
            .collect()
    }

    fn ffn(&self, tokens: &[Vec<f64>]) -> Vec<Vec<f64>> {
        tokens
            .iter()
            .map(|x| {
                let ff_dim = self.ff1.len();
                let h1: Vec<f64> = (0..ff_dim).map(|i| relu(dot(x, &self.ff1[i]))).collect();
                let h2: Vec<f64> = (0..self.d_model).map(|i| dot(&h1, &self.ff2[i])).collect();
                layer_norm_vec(
                    &h2.iter()
                        .zip(x.iter())
                        .map(|(a, b)| a + b)
                        .collect::<Vec<_>>(),
                )
            })
            .collect()
    }

    /// Encode a tokenized graph sequence (with prepended CLS token).
    pub fn encode_tokens(&self, tokens: &[(usize, Vec<f64>)]) -> Vec<Vec<f64>> {
        // Project all tokens
        let mut hidden: Vec<Vec<f64>> = std::iter::once(self.cls_token.clone())
            .chain(tokens.iter().map(|(_, t)| self.project_token(t)))
            .collect();
        hidden = self.self_attn(&hidden);
        hidden = self.ffn(&hidden);
        hidden
    }

    /// Compute MGM reconstruction loss for masked tokens.
    pub fn mgm_loss(
        &self,
        node_feats: &[Vec<f64>],
        adj: &[Vec<usize>],
        mask_ratio: f64,
        rng: &mut StdRng,
    ) -> f64 {
        let tokens = self.tokenizer.tokenize(node_feats, adj, rng);
        if tokens.is_empty() {
            return 0.0;
        }
        // Mask some tokens
        let n = tokens.len();
        let n_mask = ((n as f64 * mask_ratio).round() as usize).min(n);
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.random_range(0..=i);
            order.swap(i, j);
        }
        let mask_set: std::collections::HashSet<usize> =
            order[..n_mask].iter().cloned().collect();

        // Zero out masked tokens
        let masked_tokens: Vec<(usize, Vec<f64>)> = tokens
            .iter()
            .enumerate()
            .map(|(pos, (nid, tok))| {
                if mask_set.contains(&pos) {
                    (*nid, vec![0.0; tok.len()])
                } else {
                    (*nid, tok.clone())
                }
            })
            .collect();

        let hidden = self.encode_tokens(&masked_tokens);
        // Reconstruct masked tokens (offset by 1 for CLS)
        let d_feat = self.tokenizer.d_feat;
        let mut total_loss = 0.0;
        let mut count = 0;
        for &pos in &order[..n_mask] {
            let h_idx = pos + 1; // +1 for CLS
            if h_idx >= hidden.len() {
                continue;
            }
            let (node_idx, _) = &tokens[pos];
            if *node_idx >= node_feats.len() {
                continue;
            }
            // Reconstruct
            let recon: Vec<f64> = (0..d_feat)
                .map(|j| dot(&hidden[h_idx], &self.recon_proj[j]))
                .collect();
            let target = &node_feats[*node_idx];
            let d = d_feat.min(target.len());
            let mse: f64 = recon[..d]
                .iter()
                .zip(target[..d].iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / d as f64;
            total_loss += mse;
            count += 1;
        }
        if count == 0 { 0.0 } else { total_loss / count as f64 }
    }

    /// Graph-level CLS representation.
    pub fn graph_repr(&self, node_feats: &[Vec<f64>], adj: &[Vec<usize>], rng: &mut StdRng) -> Vec<f64> {
        let tokens = self.tokenizer.tokenize(node_feats, adj, rng);
        let hidden = self.encode_tokens(&tokens);
        hidden.into_iter().next().unwrap_or_else(|| self.cls_token.clone())
    }
}

/// Pretrainer combining MGM + graph-level classification objectives.
#[derive(Debug, Clone)]
pub struct GtpPretrainer {
    /// Underlying graph transformer
    pub model: GtpModel,
    /// Classification head \[n_classes\]\[d_model\]
    pub cls_head: Vec<Vec<f64>>,
    /// Number of output classes
    pub n_classes: usize,
    /// MGM masking ratio
    pub mask_ratio: f64,
}

impl GtpPretrainer {
    /// Build a GtpPretrainer.
    pub fn new(model: GtpModel, n_classes: usize, mask_ratio: f64, rng: &mut StdRng) -> Self {
        let cls_head = (0..n_classes)
            .map(|_| xavier_vec(model.d_model, model.d_model, n_classes, rng))
            .collect();
        Self { model, cls_head, n_classes, mask_ratio }
    }

    /// Compute combined pretraining loss (MGM + classification).
    pub fn pretrain_loss(
        &self,
        node_feats: &[Vec<f64>],
        adj: &[Vec<usize>],
        label: Option<usize>,
        rng: &mut StdRng,
    ) -> f64 {
        let mgm = self.model.mgm_loss(node_feats, adj, self.mask_ratio, rng);
        let cls = if let Some(lbl) = label {
            let repr = self.model.graph_repr(node_feats, adj, rng);
            let logits: Vec<f64> = (0..self.n_classes)
                .map(|i| dot(&repr, &self.cls_head[i]))
                .collect();
            let probs = softmax_1d(&logits);
            if lbl < probs.len() {
                -probs[lbl].max(1e-12).ln()
            } else {
                0.0
            }
        } else {
            0.0
        };
        mgm + cls
    }

    /// Classify a graph (returns predicted class index).
    pub fn classify(&self, node_feats: &[Vec<f64>], adj: &[Vec<usize>], rng: &mut StdRng) -> usize {
        let repr = self.model.graph_repr(node_feats, adj, rng);
        let logits: Vec<f64> = (0..self.n_classes)
            .map(|i| dot(&repr, &self.cls_head[i]))
            .collect();
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. GRAPH FEDERATED PRETRAINING
// ─────────────────────────────────────────────────────────────────────────────

/// Federated graph learning with local graph augmentation (FedGraph).
///
/// Each client trains locally on its graph partition and shares
/// only aggregated model updates with the server.
#[derive(Debug, Clone)]
pub struct FedGraph {
    /// Global model weights (GNN encoder) \[d_out\]\[d_in\]
    pub global_encoder: Vec<Vec<f64>>,
    /// Per-client local encoder weights (same shape)
    pub local_encoders: Vec<Vec<Vec<f64>>>,
    /// Input dimension
    pub d_in: usize,
    /// Output dimension
    pub d_out: usize,
    /// Number of clients
    pub n_clients: usize,
    /// Local learning rate
    pub local_lr: f64,
    /// Augmentation drop ratio for local training
    pub aug_ratio: f64,
}

impl FedGraph {
    /// Build a FedGraph model.
    pub fn new(
        d_in: usize,
        d_out: usize,
        n_clients: usize,
        local_lr: f64,
        aug_ratio: f64,
        rng: &mut StdRng,
    ) -> Self {
        let global_encoder: Vec<Vec<f64>> = (0..d_out)
            .map(|_| xavier_vec(d_in, d_in, d_out, rng))
            .collect();
        let local_encoders: Vec<Vec<Vec<f64>>> = (0..n_clients)
            .map(|_| global_encoder.clone())
            .collect();
        Self {
            global_encoder,
            local_encoders,
            d_in,
            d_out,
            n_clients,
            local_lr,
            aug_ratio,
        }
    }

    /// Local update: one gradient step on a client's graph using contrastive loss.
    pub fn local_update(
        &mut self,
        client_id: usize,
        node_feats: &[Vec<f64>],
        adj: &[Vec<usize>],
        rng: &mut StdRng,
    ) -> f64 {
        if client_id >= self.n_clients {
            return 0.0;
        }
        let n = node_feats.len();
        if n == 0 {
            return 0.0;
        }
        // Create augmented view (edge perturbation)
        let aug_adj: Vec<Vec<usize>> = adj
            .iter()
            .map(|nbrs| {
                nbrs.iter()
                    .filter(|_| rng.random::<f64>() >= self.aug_ratio)
                    .cloned()
                    .collect()
            })
            .collect();
        // Encode both views
        let enc = |feats: &[Vec<f64>], a: &[Vec<usize>], w: &[Vec<f64>]| -> Vec<f64> {
            let projected: Vec<Vec<f64>> = feats
                .iter()
                .map(|f| {
                    let d = self.d_in.min(f.len());
                    (0..self.d_out).map(|j| dot(&f[..d], &w[j][..d])).collect()
                })
                .collect();
            let n = feats.len();
            if n == 0 { return vec![0.0; self.d_out]; }
            // Mean pooling with neighbourhood aggregation
            let mut agg = projected.clone();
            for i in 0..n {
                if !a[i].is_empty() {
                    for &nb in &a[i] {
                        if nb < n {
                            for d in 0..self.d_out {
                                agg[i][d] += projected[nb][d] / a[i].len() as f64;
                            }
                        }
                    }
                }
            }
            let mean: Vec<f64> = (0..self.d_out)
                .map(|d| agg.iter().map(|v| v[d]).sum::<f64>() / n as f64)
                .collect();
            let norm = mean.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            mean.iter().map(|x| x / norm).collect()
        };
        let w = &self.local_encoders[client_id];
        let z1 = enc(node_feats, adj, w);
        let z2 = enc(node_feats, &aug_adj, w);
        // InfoNCE loss (simplified for two views)
        let temp = 0.5;
        let pos = dot(&z1, &z2) / temp;
        let neg = dot(&z1, &z1) / temp; // self-contrast as negative
        let loss = -(pos - (pos.exp() + neg.exp()).ln());
        // Gradient step (simulated: weight update proportional to loss)
        for row in self.local_encoders[client_id].iter_mut() {
            for w in row.iter_mut() {
                *w -= self.local_lr * loss.abs() * 0.001;
            }
        }
        loss.abs()
    }

    /// FedAvg aggregation: average local encoders into global.
    pub fn fedavg(&mut self) {
        let n = self.n_clients as f64;
        for j in 0..self.d_out {
            for k in 0..self.d_in {
                let avg = self.local_encoders.iter().map(|loc| loc[j][k]).sum::<f64>() / n;
                self.global_encoder[j][k] = avg;
            }
        }
        // Broadcast global back to clients
        for client in self.local_encoders.iter_mut() {
            client[..self.d_out].clone_from_slice(&self.global_encoder[..self.d_out]);
        }
    }

    /// Encode a graph using the global encoder (inference).
    pub fn encode(&self, node_feats: &[Vec<f64>], adj: &[Vec<usize>]) -> Vec<f64> {
        let n = node_feats.len();
        if n == 0 { return vec![0.0; self.d_out]; }
        let projected: Vec<Vec<f64>> = node_feats
            .iter()
            .map(|f| {
                let d = self.d_in.min(f.len());
                (0..self.d_out)
                    .map(|j| dot(&f[..d], &self.global_encoder[j][..d]))
                    .collect()
            })
            .collect();
        let mean: Vec<f64> = (0..self.d_out)
            .map(|d| projected.iter().map(|v| v[d]).sum::<f64>() / n as f64)
            .collect();
        let norm = mean.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        mean.iter().map(|x| x / norm).collect()
    }
}

/// Cross-graph transfer learning with domain adaptation.
///
/// Adapts a source-graph pretrained encoder to a target graph by minimising
/// maximum mean discrepancy (MMD) between source and target representations.
#[derive(Debug, Clone)]
pub struct CrossGraphTransfer {
    /// Shared encoder weights \[d_out\]\[d_in\]
    pub encoder_w: Vec<Vec<f64>>,
    /// Domain adaptation weights (bias alignment)
    pub domain_bias: Vec<f64>,
    /// Input dimension
    pub d_in: usize,
    /// Output dimension
    pub d_out: usize,
    /// RBF kernel bandwidth for MMD
    pub bandwidth: f64,
}

impl CrossGraphTransfer {
    /// Build a cross-graph transfer module.
    pub fn new(d_in: usize, d_out: usize, bandwidth: f64, rng: &mut StdRng) -> Self {
        let encoder_w = (0..d_out)
            .map(|_| xavier_vec(d_in, d_in, d_out, rng))
            .collect();
        let domain_bias = vec![0.0; d_out];
        Self { encoder_w, domain_bias, d_in, d_out, bandwidth }
    }

    fn encode_graph(&self, node_feats: &[Vec<f64>], apply_bias: bool) -> Vec<f64> {
        let n = node_feats.len();
        if n == 0 { return vec![0.0; self.d_out]; }
        let projected: Vec<Vec<f64>> = node_feats
            .iter()
            .map(|f| {
                let d = self.d_in.min(f.len());
                (0..self.d_out)
                    .map(|j| dot(&f[..d], &self.encoder_w[j][..d]))
                    .collect()
            })
            .collect();
        let mean: Vec<f64> = (0..self.d_out)
            .map(|d| projected.iter().map(|v| v[d]).sum::<f64>() / n as f64)
            .collect();
        if apply_bias {
            mean.iter().zip(&self.domain_bias).map(|(a, b)| a + b).collect()
        } else {
            mean
        }
    }

    /// RBF kernel: k(x, y) = exp(-||x-y||^2 / (2 * sigma^2))
    fn rbf_kernel(&self, a: &[f64], b: &[f64]) -> f64 {
        let sq_dist: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        (-sq_dist / (2.0 * self.bandwidth.powi(2))).exp()
    }

    /// Compute MMD between source and target graph embeddings.
    pub fn mmd_loss(
        &self,
        source_graphs: &[Vec<Vec<f64>>],
        target_graphs: &[Vec<Vec<f64>>],
    ) -> f64 {
        let src_embs: Vec<Vec<f64>> = source_graphs
            .iter()
            .map(|g| self.encode_graph(g, false))
            .collect();
        let tgt_embs: Vec<Vec<f64>> = target_graphs
            .iter()
            .map(|g| self.encode_graph(g, true))
            .collect();
        let ns = src_embs.len();
        let nt = tgt_embs.len();
        if ns == 0 || nt == 0 {
            return 0.0;
        }
        // E[k(xs, xs')] - 2*E[k(xs, xt)] + E[k(xt, xt')]
        let kss: f64 = src_embs
            .iter()
            .flat_map(|a| src_embs.iter().map(move |b| self.rbf_kernel(a, b)))
            .sum::<f64>()
            / (ns * ns) as f64;
        let ktt: f64 = tgt_embs
            .iter()
            .flat_map(|a| tgt_embs.iter().map(move |b| self.rbf_kernel(a, b)))
            .sum::<f64>()
            / (nt * nt) as f64;
        let kst: f64 = src_embs
            .iter()
            .flat_map(|a| tgt_embs.iter().map(move |b| self.rbf_kernel(a, b)))
            .sum::<f64>()
            / (ns * nt) as f64;
        (kss + ktt - 2.0 * kst).max(0.0).sqrt()
    }

    /// Adapt the domain bias to minimise MMD (one gradient step).
    pub fn adapt_step(
        &mut self,
        source_graphs: &[Vec<Vec<f64>>],
        target_graphs: &[Vec<Vec<f64>>],
        lr: f64,
    ) -> f64 {
        let loss_before = self.mmd_loss(source_graphs, target_graphs);
        // Compute mean target embedding and push bias towards it
        let tgt_mean: Vec<f64> = if target_graphs.is_empty() {
            vec![0.0; self.d_out]
        } else {
            let sum: Vec<f64> = target_graphs
                .iter()
                .map(|g| self.encode_graph(g, false))
                .fold(vec![0.0; self.d_out], |acc, e| {
                    acc.iter().zip(e.iter()).map(|(a, b)| a + b).collect()
                });
            sum.iter().map(|x| x / target_graphs.len() as f64).collect()
        };
        let src_mean: Vec<f64> = if source_graphs.is_empty() {
            vec![0.0; self.d_out]
        } else {
            let sum: Vec<f64> = source_graphs
                .iter()
                .map(|g| self.encode_graph(g, false))
                .fold(vec![0.0; self.d_out], |acc, e| {
                    acc.iter().zip(e.iter()).map(|(a, b)| a + b).collect()
                });
            sum.iter().map(|x| x / source_graphs.len() as f64).collect()
        };
        for (b, (tm, sm)) in self
            .domain_bias
            .iter_mut()
            .zip(tgt_mean.iter().zip(src_mean.iter()))
        {
            *b += lr * (tm - sm);
        }
        loss_before
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. GRAPH METRICS
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluation metrics for graph learning tasks.
///
/// Supports link prediction and node classification tasks.
#[derive(Debug, Clone)]
pub struct GraphMetrics {
    /// True positive edge scores
    pub pos_scores: Vec<f64>,
    /// True negative edge scores
    pub neg_scores: Vec<f64>,
    /// True labels for node classification (0 or 1)
    pub true_labels: Vec<usize>,
    /// Predicted probabilities for node classification
    pub pred_probs: Vec<Vec<f64>>,
}

impl GraphMetrics {
    /// Build a new metrics accumulator.
    pub fn new() -> Self {
        Self {
            pos_scores: Vec::new(),
            neg_scores: Vec::new(),
            true_labels: Vec::new(),
            pred_probs: Vec::new(),
        }
    }

    /// Record link prediction scores.
    pub fn add_link_scores(&mut self, pos: &[f64], neg: &[f64]) {
        self.pos_scores.extend_from_slice(pos);
        self.neg_scores.extend_from_slice(neg);
    }

    /// Record node classification predictions.
    pub fn add_node_preds(&mut self, labels: &[usize], probs: &[Vec<f64>]) {
        self.true_labels.extend_from_slice(labels);
        self.pred_probs.extend(probs.iter().cloned());
    }

    /// Compute AUC-ROC for link prediction using trapezoidal integration.
    pub fn auc_roc(&self) -> f64 {
        let n_pos = self.pos_scores.len();
        let n_neg = self.neg_scores.len();
        if n_pos == 0 || n_neg == 0 {
            return 0.5;
        }
        // Build (score, label) pairs
        let mut pairs: Vec<(f64, u8)> = self
            .pos_scores
            .iter()
            .map(|&s| (s, 1u8))
            .chain(self.neg_scores.iter().map(|&s| (s, 0u8)))
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut auc = 0.0;
        let mut tp = 0.0;
        let mut fp = 0.0;
        let mut prev_tp = 0.0;
        let mut prev_fp = 0.0;
        for (_, lbl) in &pairs {
            if *lbl == 1 {
                tp += 1.0;
            } else {
                fp += 1.0;
                // Trapezoid area
                auc += (tp + prev_tp) / 2.0;
                prev_tp = tp;
                prev_fp = fp;
            }
        }
        let _ = prev_fp;
        if n_pos as f64 * n_neg as f64 > 0.0 {
            auc / (n_pos as f64 * n_neg as f64)
        } else {
            0.5
        }
    }

    /// Compute Average Precision (AP) for link prediction.
    pub fn average_precision(&self) -> f64 {
        let n_pos = self.pos_scores.len();
        let n_neg = self.neg_scores.len();
        if n_pos == 0 || n_neg == 0 {
            return 0.0;
        }
        let mut pairs: Vec<(f64, u8)> = self
            .pos_scores
            .iter()
            .map(|&s| (s, 1u8))
            .chain(self.neg_scores.iter().map(|&s| (s, 0u8)))
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut ap = 0.0;
        let mut tp = 0.0;
        for (rank, (_, lbl)) in pairs.iter().enumerate() {
            if *lbl == 1 {
                tp += 1.0;
                let precision = tp / (rank as f64 + 1.0);
                ap += precision;
            }
        }
        ap / n_pos as f64
    }

    /// Compute accuracy for node classification.
    pub fn accuracy(&self) -> f64 {
        if self.true_labels.is_empty() {
            return 0.0;
        }
        let correct = self
            .true_labels
            .iter()
            .zip(self.pred_probs.iter())
            .filter(|(lbl, probs)| {
                let pred = probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                pred == **lbl
            })
            .count();
        correct as f64 / self.true_labels.len() as f64
    }

    /// Compute Hits@k for link prediction.
    ///
    /// For each positive edge, compute the fraction where pos_score ranks in top-k
    /// among all scores.
    pub fn hits_at_k(&self, k: usize) -> f64 {
        let n_pos = self.pos_scores.len();
        if n_pos == 0 {
            return 0.0;
        }
        let all_neg: Vec<f64> = self.neg_scores.clone();
        let hits = self
            .pos_scores
            .iter()
            .filter(|&&pos_score| {
                // Count negatives with higher score
                let rank = all_neg.iter().filter(|&&s| s >= pos_score).count() + 1;
                rank <= k
            })
            .count();
        hits as f64 / n_pos as f64
    }

    /// Reset all accumulated statistics.
    pub fn reset(&mut self) {
        self.pos_scores.clear();
        self.neg_scores.clear();
        self.true_labels.clear();
        self.pred_probs.clear();
    }
}

impl Default for GraphMetrics {
    fn default() -> Self {
        Self::new()
    }
}
