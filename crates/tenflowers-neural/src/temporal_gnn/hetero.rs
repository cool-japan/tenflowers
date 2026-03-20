//! Heterogeneous Temporal Graph components.
//!
//! HeteroTemporalGraph, RelationalTemporalConv, HeteroTgnModel.

use tenflowers_core::{Result, TensorError};

use super::tgn::{MemoryUpdateModule, NodeMemory, TimeEncoder};
use super::types::{linear, rand_mat, zero_vec, HeteroEdgeType, HeteroNodeType, TemporalEdge};
use scirs2_core::random::{rngs::StdRng, SeedableRng};

// ─────────────────────────────────────────────────────────────────────────────
// HeteroTemporalGraph
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-relational temporal graph with typed nodes and edges.
#[derive(Debug, Clone)]
pub struct HeteroTemporalGraph {
    /// Global node type assignments: node_id → type label.
    pub node_types: Vec<HeteroNodeType>,
    /// Temporal edges with relation type.
    pub edges: Vec<(HeteroEdgeType, TemporalEdge)>,
}

impl HeteroTemporalGraph {
    /// Create a new empty heterogeneous temporal graph.
    pub fn new(node_types: Vec<HeteroNodeType>) -> Self {
        Self {
            node_types,
            edges: Vec::new(),
        }
    }

    /// Add a typed temporal edge.
    pub fn add_edge(&mut self, rel: HeteroEdgeType, edge: TemporalEdge) -> Result<()> {
        let n = self.node_types.len();
        if edge.src >= n || edge.dst >= n {
            return Err(TensorError::invalid_argument(format!(
                "HeteroTemporalGraph::add_edge — node index out of range (n_nodes={n})"
            )));
        }
        self.edges.push((rel, edge));
        Ok(())
    }

    /// Return edges of a given relation type.
    pub fn edges_of_type(&self, rel: &HeteroEdgeType) -> Vec<&TemporalEdge> {
        self.edges
            .iter()
            .filter(|(r, _)| r == rel)
            .map(|(_, e)| e)
            .collect()
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.node_types.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RelationalTemporalConv
// ─────────────────────────────────────────────────────────────────────────────

/// Per-relation-type temporal GCN convolution with mean aggregation.
#[derive(Debug, Clone)]
pub struct RelationalTemporalConv {
    pub relation_types: Vec<HeteroEdgeType>,
    /// Weight matrix per relation [out_dim × in_dim].
    pub weights: Vec<Vec<Vec<f64>>>,
    pub biases: Vec<Vec<f64>>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl RelationalTemporalConv {
    /// Create a new RelationalTemporalConv.
    pub fn new(
        relation_types: Vec<HeteroEdgeType>,
        in_dim: usize,
        out_dim: usize,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let s = (2.0 / in_dim as f64).sqrt();
        let weights: Vec<Vec<Vec<f64>>> = relation_types
            .iter()
            .map(|_| rand_mat(out_dim, in_dim, s, &mut rng))
            .collect();
        let biases: Vec<Vec<f64>> = relation_types.iter().map(|_| zero_vec(out_dim)).collect();
        Self {
            relation_types,
            weights,
            biases,
            in_dim,
            out_dim,
        }
    }

    /// Forward pass.
    ///
    /// For each relation, aggregate neighbour features and apply a linear transform.
    /// Outputs are summed across all relation types.
    ///
    /// `node_features`: `[N][in_dim]`.
    /// `graph`: the heterogeneous graph with typed edges.
    /// Returns `[N][out_dim]`.
    pub fn forward(
        &self,
        node_features: &[Vec<f64>],
        graph: &HeteroTemporalGraph,
    ) -> Result<Vec<Vec<f64>>> {
        let n = node_features.len();
        let mut out: Vec<Vec<f64>> = vec![zero_vec(self.out_dim); n];
        for (rel_idx, rel) in self.relation_types.iter().enumerate() {
            let edges = graph.edges_of_type(rel);
            // Count and accumulate neighbour features.
            let mut agg: Vec<Vec<f64>> = vec![zero_vec(self.in_dim); n];
            let mut cnt: Vec<usize> = vec![0; n];
            for e in edges {
                if e.src >= n || e.dst >= n {
                    continue;
                }
                for (ak, &fk) in agg[e.dst].iter_mut().zip(node_features[e.src].iter()) {
                    *ak += fk;
                }
                cnt[e.dst] += 1;
            }
            // Mean-normalise and project.
            for node in 0..n {
                let denom = cnt[node].max(1) as f64;
                let mean_feat: Vec<f64> = agg[node].iter().map(|v| v / denom).collect();
                let proj = linear(&self.weights[rel_idx], &self.biases[rel_idx], &mean_feat);
                for (oi, pi) in out[node].iter_mut().zip(proj.iter()) {
                    *oi += pi;
                }
            }
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HeteroTgnModel — TGN for heterogeneous graphs
// ─────────────────────────────────────────────────────────────────────────────

/// TGN extended to heterogeneous multi-relational temporal graphs.
#[derive(Debug, Clone)]
pub struct HeteroTgnModel {
    pub memory: NodeMemory,
    pub time_encoder: TimeEncoder,
    /// Per-relation GRU updaters.
    pub updaters: Vec<(HeteroEdgeType, MemoryUpdateModule)>,
    pub rel_conv: RelationalTemporalConv,
    pub memory_dim: usize,
    pub time_dim: usize,
}

impl HeteroTgnModel {
    /// Build a HeteroTgnModel.
    pub fn new(
        n_nodes: usize,
        memory_dim: usize,
        relation_types: Vec<HeteroEdgeType>,
        seed: u64,
    ) -> Self {
        let time_dim = 16;
        let msg_dim = memory_dim * 2 + time_dim;
        let updaters: Vec<(HeteroEdgeType, MemoryUpdateModule)> = relation_types
            .iter()
            .enumerate()
            .map(|(i, rel)| {
                (
                    rel.clone(),
                    MemoryUpdateModule::new(msg_dim, memory_dim, seed.wrapping_add(i as u64)),
                )
            })
            .collect();
        let rel_conv = RelationalTemporalConv::new(
            relation_types,
            memory_dim,
            memory_dim,
            seed.wrapping_add(77),
        );
        Self {
            memory: NodeMemory::new(n_nodes, memory_dim),
            time_encoder: TimeEncoder::new(time_dim, seed.wrapping_add(55)),
            updaters,
            rel_conv,
            memory_dim,
            time_dim,
        }
    }

    /// Process a typed temporal edge, updating both endpoint memories.
    pub fn process_edge(&mut self, rel: &HeteroEdgeType, edge: &TemporalEdge) -> Result<()> {
        let updater_idx = self
            .updaters
            .iter()
            .position(|(r, _)| r == rel)
            .ok_or_else(|| {
                TensorError::invalid_argument(format!(
                    "HeteroTgnModel: unknown relation type '{}'",
                    rel.0
                ))
            })?;

        let (mem_src, t_src) = {
            let (s, t) = self.memory.get(edge.src)?;
            (s.clone(), t)
        };
        let (mem_dst, t_dst) = {
            let (s, t) = self.memory.get(edge.dst)?;
            (s.clone(), t)
        };
        let dt_src = (edge.time - t_src).max(0.0);
        let dt_dst = (edge.time - t_dst).max(0.0);
        let enc_src = self.time_encoder.encode(dt_src);
        let enc_dst = self.time_encoder.encode(dt_dst);

        let mut msg_src: Vec<f64> = Vec::with_capacity(self.memory_dim * 2 + self.time_dim);
        msg_src.extend_from_slice(&mem_src);
        msg_src.extend_from_slice(&mem_dst);
        msg_src.extend_from_slice(&enc_src);

        let mut msg_dst: Vec<f64> = Vec::with_capacity(self.memory_dim * 2 + self.time_dim);
        msg_dst.extend_from_slice(&mem_dst);
        msg_dst.extend_from_slice(&mem_src);
        msg_dst.extend_from_slice(&enc_dst);

        let new_mem_src = self.updaters[updater_idx].1.step(&msg_src, &mem_src)?;
        let new_mem_dst = self.updaters[updater_idx].1.step(&msg_dst, &mem_dst)?;

        self.memory.set(edge.src, new_mem_src, edge.time)?;
        self.memory.set(edge.dst, new_mem_dst, edge.time)?;
        Ok(())
    }

    /// Compute node embeddings using relational temporal convolution.
    ///
    /// Returns `[N][memory_dim]`.
    pub fn embed(&self, graph: &HeteroTemporalGraph) -> Result<Vec<Vec<f64>>> {
        let node_feats: Vec<Vec<f64>> = self.memory.states.clone();
        self.rel_conv.forward(&node_feats, graph)
    }
}
