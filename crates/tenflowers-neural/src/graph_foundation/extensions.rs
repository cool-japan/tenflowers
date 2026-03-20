//! Heterogeneous graph transformers, link prediction, and graph generation extensions.
//!
//! This module provides:
//! - **Heterogeneous Graph Transformers**: HGT, R-GCN, CompGCN, HeteroSAGE
//! - **Link Prediction**: ComplEx, RotatE
//! - **Graph Generation**: GraphRNN (node/edge), MoleculeGenerator

use super::{layer_norm_vec, relu, sigmoid, softmax_1d, xavier_vec, dot};
use scirs2_core::random::{rngs::StdRng, Rng};
use scirs2_core::RngExt;
use std::collections::HashMap;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// 4. HETEROGENEOUS GRAPH TRANSFORMERS
// ─────────────────────────────────────────────────────────────────────────────

/// Heterogeneous Graph Transformer layer: type-specific Q/K/V projections.
#[derive(Debug, Clone)]
pub struct HgtLayer {
    /// Feature/model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Per-node-type Q projections: type_name -> [head][d_model * head_dim]
    pub wq_per_type: HashMap<String, Vec<Vec<f64>>>,
    /// Per-node-type K projections
    pub wk_per_type: HashMap<String, Vec<Vec<f64>>>,
    /// Per-node-type V projections
    pub wv_per_type: HashMap<String, Vec<Vec<f64>>>,
    /// Per-edge-type attention weight scalar
    pub edge_type_bias: HashMap<String, f64>,
}

impl HgtLayer {
    /// Build an HGT layer with type-specific projections.
    pub fn new(
        d_model: usize,
        num_heads: usize,
        node_types: &[&str],
        edge_types: &[&str],
        rng: &mut StdRng,
    ) -> Result<Self> {
        if d_model % num_heads != 0 {
            return Err(TensorError::invalid_argument(format!(
                "d_model={d_model} not divisible by num_heads={num_heads}"
            )));
        }
        let head_dim = d_model / num_heads;
        let mut wq_per_type = HashMap::new();
        let mut wk_per_type = HashMap::new();
        let mut wv_per_type = HashMap::new();
        for &nt in node_types {
            let wq = (0..num_heads)
                .map(|_| xavier_vec(d_model * head_dim, d_model, head_dim, rng))
                .collect();
            let wk = (0..num_heads)
                .map(|_| xavier_vec(d_model * head_dim, d_model, head_dim, rng))
                .collect();
            let wv = (0..num_heads)
                .map(|_| xavier_vec(d_model * head_dim, d_model, head_dim, rng))
                .collect();
            wq_per_type.insert(nt.to_string(), wq);
            wk_per_type.insert(nt.to_string(), wk);
            wv_per_type.insert(nt.to_string(), wv);
        }
        let edge_type_bias = edge_types
            .iter()
            .map(|&et| (et.to_string(), rng.random::<f64>() * 0.1))
            .collect();
        Ok(Self {
            d_model,
            num_heads,
            wq_per_type,
            wk_per_type,
            wv_per_type,
            edge_type_bias,
        })
    }

    fn project_type(
        feats: &[Vec<f64>],
        w: &[Vec<f64>],
        d_model: usize,
        head_dim: usize,
    ) -> Vec<Vec<f64>> {
        feats
            .iter()
            .map(|f| {
                (0..head_dim)
                    .map(|j| {
                        (0..d_model.min(f.len()))
                            .map(|k| f[k] * w[0][k * head_dim + j])
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }

    /// Forward: heterogeneous attention across node types and edge types.
    pub fn forward(
        &self,
        node_feats_by_type: &HashMap<String, Vec<Vec<f64>>>,
        edges_by_type: &HashMap<String, Vec<(usize, usize)>>,
    ) -> HashMap<String, Vec<Vec<f64>>> {
        let head_dim = self.d_model / self.num_heads;
        let mut output: HashMap<String, Vec<Vec<f64>>> = node_feats_by_type
            .iter()
            .map(|(t, feats)| (t.clone(), vec![vec![0.0f64; self.d_model]; feats.len()]))
            .collect();

        for (edge_type, edges) in edges_by_type {
            // Parse "src_type-rel-dst_type" convention or use first available types
            let (src_type, dst_type) = Self::parse_edge_type(edge_type, node_feats_by_type);

            let src_feats = match node_feats_by_type.get(&src_type) {
                Some(f) => f,
                None => continue,
            };
            let dst_feats = match node_feats_by_type.get(&dst_type) {
                Some(f) => f,
                None => continue,
            };

            let wq = match self.wq_per_type.get(&dst_type) {
                Some(w) => w,
                None => continue,
            };
            let wk = match self.wk_per_type.get(&src_type) {
                Some(w) => w,
                None => continue,
            };
            let wv = match self.wv_per_type.get(&src_type) {
                Some(w) => w,
                None => continue,
            };

            let edge_bias = self
                .edge_type_bias
                .get(edge_type)
                .cloned()
                .unwrap_or(0.0);

            let q_all = Self::project_type(dst_feats, wq, self.d_model, head_dim);
            let k_all = Self::project_type(src_feats, wk, self.d_model, head_dim);
            let v_all = Self::project_type(src_feats, wv, self.d_model, head_dim);

            let scale = (head_dim as f64).sqrt();

            for &(src, dst) in edges {
                if src >= src_feats.len() || dst >= dst_feats.len() {
                    continue;
                }
                let q = &q_all[dst];
                let k = &k_all[src];
                let v = &v_all[src];

                let score = dot(q, k) / scale + edge_bias;
                let attn = sigmoid(score);

                let dst_out = output
                    .entry(dst_type.clone())
                    .or_insert_with(|| vec![vec![0.0f64; self.d_model]; dst_feats.len()]);
                if dst < dst_out.len() {
                    let hoff = 0;
                    for d in 0..head_dim {
                        dst_out[dst][hoff + d] += attn * v[d];
                    }
                }
            }
        }
        output
            .iter()
            .map(|(t, feats)| (t.clone(), feats.iter().map(|f| layer_norm_vec(f)).collect()))
            .collect()
    }

    fn parse_edge_type(
        edge_type: &str,
        node_feats_by_type: &HashMap<String, Vec<Vec<f64>>>,
    ) -> (String, String) {
        // Convention: "src_type-rel-dst_type"
        let parts: Vec<&str> = edge_type.splitn(3, '-').collect();
        if parts.len() == 3 {
            let src = parts[0].to_string();
            let dst = parts[2].to_string();
            if node_feats_by_type.contains_key(&src) && node_feats_by_type.contains_key(&dst) {
                return (src, dst);
            }
        }
        // Fallback: use first two available types
        let types: Vec<String> = node_feats_by_type.keys().cloned().collect();
        let src = types.first().cloned().unwrap_or_default();
        let dst = types.get(1).cloned().unwrap_or_else(|| src.clone());
        (src, dst)
    }
}

/// Relational GCN with basis decomposition.
#[derive(Debug, Clone)]
pub struct RelationalGcn {
    /// Input feature dimension
    pub d_in: usize,
    /// Output feature dimension
    pub d_out: usize,
    /// Number of relation types
    pub n_relations: usize,
    /// Number of basis matrices
    pub n_bases: usize,
    /// Basis matrices [n_bases][d_in * d_out]
    pub bases: Vec<Vec<f64>>,
    /// Coefficients per relation \[n_relations\]\[n_bases\]
    pub coeffs: Vec<Vec<f64>>,
    /// Self-loop weight [d_in * d_out]
    pub self_w: Vec<f64>,
}

impl RelationalGcn {
    /// Build a relational GCN with basis decomposition.
    pub fn new(
        d_in: usize,
        d_out: usize,
        n_relations: usize,
        n_bases: usize,
        rng: &mut StdRng,
    ) -> Self {
        let bases = (0..n_bases)
            .map(|_| xavier_vec(d_in * d_out, d_in, d_out, rng))
            .collect();
        let coeffs = (0..n_relations)
            .map(|_| xavier_vec(n_bases, n_bases, 1, rng))
            .collect();
        let self_w = xavier_vec(d_in * d_out, d_in, d_out, rng);
        Self {
            d_in,
            d_out,
            n_relations,
            n_bases,
            bases,
            coeffs,
            self_w,
        }
    }

    fn relation_weight(&self, rel: usize) -> Vec<f64> {
        let mut w = vec![0.0f64; self.d_in * self.d_out];
        for b in 0..self.n_bases {
            let c = if rel < self.coeffs.len() && b < self.coeffs[rel].len() {
                self.coeffs[rel][b]
            } else {
                0.0
            };
            for (i, x) in self.bases[b].iter().enumerate() {
                w[i] += c * x;
            }
        }
        w
    }

    /// Forward: aggregate per-relation neighbors.
    pub fn forward(
        &self,
        feats: &[Vec<f64>],
        adj_per_relation: &[Vec<Vec<usize>>],
    ) -> Vec<Vec<f64>> {
        let n = feats.len();
        let mut out = vec![vec![0.0f64; self.d_out]; n];

        // Self connection
        for i in 0..n {
            for j in 0..self.d_out {
                out[i][j] += (0..self.d_in.min(feats[i].len()))
                    .map(|k| feats[i][k] * self.self_w[k * self.d_out + j])
                    .sum::<f64>();
            }
        }

        // Relational aggregation
        for (rel, adj) in adj_per_relation.iter().enumerate().take(self.n_relations) {
            let w = self.relation_weight(rel);
            for (dst, neighbors) in adj.iter().enumerate().take(n) {
                if neighbors.is_empty() {
                    continue;
                }
                let norm = 1.0 / neighbors.len() as f64;
                for &src in neighbors {
                    if src >= n {
                        continue;
                    }
                    for j in 0..self.d_out {
                        out[dst][j] += norm
                            * (0..self.d_in.min(feats[src].len()))
                                .map(|k| feats[src][k] * w[k * self.d_out + j])
                                .sum::<f64>();
                    }
                }
            }
        }
        out.iter().map(|v| layer_norm_vec(v)).collect()
    }
}

/// CompGCN layer: compositional relation aggregation W_o * (h_v + h_r).
#[derive(Debug, Clone)]
pub struct CompGcnLayer {
    /// Feature dimension
    pub d_model: usize,
    /// Node transform weights [d_model * d_model]
    pub node_w: Vec<f64>,
    /// Relation embedding: rel_id -> `Vec<f64>`
    pub rel_embs: Vec<Vec<f64>>,
}

impl CompGcnLayer {
    /// Build a CompGCN layer.
    pub fn new(d_model: usize, n_relations: usize, rng: &mut StdRng) -> Self {
        let node_w = xavier_vec(d_model * d_model, d_model, d_model, rng);
        let rel_embs = (0..n_relations)
            .map(|_| xavier_vec(d_model, d_model, 1, rng))
            .collect();
        Self {
            d_model,
            node_w,
            rel_embs,
        }
    }

    /// Forward: for each edge (src, dst, rel), compose node + relation, aggregate.
    pub fn forward(&self, feats: &[Vec<f64>], edges: &[(usize, usize, usize)]) -> Vec<Vec<f64>> {
        let n = feats.len();
        let mut agg = vec![vec![0.0f64; self.d_model]; n];
        let mut counts = vec![0usize; n];

        for &(src, dst, rel) in edges {
            if src >= n || dst >= n {
                continue;
            }
            let h_r = if rel < self.rel_embs.len() {
                &self.rel_embs[rel]
            } else {
                &self.rel_embs[0]
            };
            let d = self.d_model.min(feats[src].len());
            let composed: Vec<f64> = feats[src][..d]
                .iter()
                .zip(h_r[..d].iter())
                .map(|(a, b)| a + b)
                .collect();
            let transformed: Vec<f64> = (0..self.d_model)
                .map(|j| {
                    (0..d)
                        .map(|k| composed[k] * self.node_w[k * self.d_model + j])
                        .sum()
                })
                .collect();
            for di in 0..self.d_model {
                agg[dst][di] += transformed[di];
            }
            counts[dst] += 1;
        }

        agg.iter_mut().zip(counts.iter()).for_each(|(v, &c)| {
            if c > 0 {
                for x in v.iter_mut() {
                    *x /= c as f64;
                }
            }
        });
        agg.iter().map(|v| layer_norm_vec(v)).collect()
    }
}

/// HeteroSAGE: per-type neighbor sampling + type-specific aggregation.
#[derive(Debug, Clone)]
pub struct HeteroSage {
    /// Feature dimension
    pub d_model: usize,
    /// Per-type aggregation weights
    pub agg_w: HashMap<String, Vec<f64>>,
}

impl HeteroSage {
    /// Build a HeteroSAGE layer.
    pub fn new(d_model: usize, node_types: &[&str], rng: &mut StdRng) -> Self {
        let agg_w = node_types
            .iter()
            .map(|&t| {
                (
                    t.to_string(),
                    xavier_vec(d_model * d_model, d_model, d_model, rng),
                )
            })
            .collect();
        Self { d_model, agg_w }
    }

    /// Aggregate features for nodes of a given type using typed neighbor features.
    pub fn aggregate(
        &self,
        node_type: &str,
        feats: &[Vec<f64>],
        neighbor_feats: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let w = self
            .agg_w
            .get(node_type)
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.d_model * self.d_model]);
        feats
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let mean: Vec<f64> = if neighbor_feats.is_empty() {
                    f.clone()
                } else {
                    let nf: Vec<&Vec<f64>> = neighbor_feats.iter().collect();
                    let d = self.d_model.min(f.len());
                    (0..d)
                        .map(|d_| {
                            nf.iter()
                                .map(|n| if d_ < n.len() { n[d_] } else { 0.0 })
                                .sum::<f64>()
                                / nf.len() as f64
                                + if d_ < f.len() { f[d_] } else { 0.0 }
                        })
                        .collect()
                };
                let d = self.d_model.min(mean.len());
                let out: Vec<f64> = (0..self.d_model)
                    .map(|j| (0..d).map(|k| mean[k] * w[k * self.d_model + j]).sum())
                    .collect();
                let _ = i;
                layer_norm_vec(&out)
            })
            .collect()
    }
}

/// Semantic attention over meta-path embeddings.
#[derive(Debug, Clone)]
pub struct SemanticAttention {
    /// Feature dimension
    pub d_model: usize,
    /// Attention vector \[d_model\]
    pub attn_vec: Vec<f64>,
}

impl SemanticAttention {
    /// Build a semantic attention module.
    pub fn new(d_model: usize, rng: &mut StdRng) -> Self {
        let attn_vec = xavier_vec(d_model, d_model, 1, rng);
        Self { d_model, attn_vec }
    }

    /// Aggregate embeddings from different meta-paths via soft attention.
    pub fn aggregate(&self, path_embeddings: &[Vec<f64>]) -> Vec<f64> {
        if path_embeddings.is_empty() {
            return vec![0.0; self.d_model];
        }
        let d = self.d_model.min(self.attn_vec.len());
        let scores: Vec<f64> = path_embeddings
            .iter()
            .map(|e| {
                (0..d.min(e.len()))
                    .map(|i| e[i] * self.attn_vec[i])
                    .sum::<f64>()
                    .tanh()
            })
            .collect();
        let weights = softmax_1d(&scores);
        (0..self.d_model)
            .map(|di| {
                weights
                    .iter()
                    .zip(path_embeddings.iter())
                    .map(|(w, e)| w * if di < e.len() { e[di] } else { 0.0 })
                    .sum()
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. LINK PREDICTION & GRAPH GENERATION
// ─────────────────────────────────────────────────────────────────────────────

/// ComplEx embedding for link prediction: Re(<h, r, t̄>).
#[derive(Debug, Clone)]
pub struct ComplexLinkPredictor {
    /// Embedding dimension (complex)
    pub dim: usize,
    /// Number of entities
    pub n_entities: usize,
    /// Number of relation types
    pub n_relations: usize,
    /// Entity embeddings: real parts \[n_entities\]\[dim\]
    pub ent_re: Vec<Vec<f64>>,
    /// Entity embeddings: imaginary parts \[n_entities\]\[dim\]
    pub ent_im: Vec<Vec<f64>>,
    /// Relation embeddings: real parts \[n_relations\]\[dim\]
    pub rel_re: Vec<Vec<f64>>,
    /// Relation embeddings: imaginary parts \[n_relations\]\[dim\]
    pub rel_im: Vec<Vec<f64>>,
}

impl ComplexLinkPredictor {
    /// Build a ComplEx link predictor.
    pub fn new(n_entities: usize, n_relations: usize, dim: usize, rng: &mut StdRng) -> Self {
        let ent_re = (0..n_entities)
            .map(|_| xavier_vec(dim, n_entities, dim, rng))
            .collect();
        let ent_im = (0..n_entities)
            .map(|_| xavier_vec(dim, n_entities, dim, rng))
            .collect();
        let rel_re = (0..n_relations)
            .map(|_| xavier_vec(dim, n_relations, dim, rng))
            .collect();
        let rel_im = (0..n_relations)
            .map(|_| xavier_vec(dim, n_relations, dim, rng))
            .collect();
        Self {
            dim,
            n_entities,
            n_relations,
            ent_re,
            ent_im,
            rel_re,
            rel_im,
        }
    }

    /// ComplEx score: Re(<h, r, conj(t)>) = sum_d h_re*r_re*t_re + h_re*r_im*t_im + h_im*r_re*t_im - h_im*r_im*t_re
    pub fn score(&self, h: usize, r: usize, t: usize) -> f64 {
        if h >= self.n_entities || r >= self.n_relations || t >= self.n_entities {
            return 0.0;
        }
        (0..self.dim)
            .map(|d| {
                let hr = self.ent_re[h][d];
                let hi = self.ent_im[h][d];
                let rr = self.rel_re[r][d];
                let ri = self.rel_im[r][d];
                let tr = self.ent_re[t][d];
                let ti = self.ent_im[t][d];
                hr * rr * tr + hr * ri * ti + hi * rr * ti - hi * ri * tr
            })
            .sum()
    }
}

/// RotatE link predictor: entity in complex space, relation = phase rotation.
#[derive(Debug, Clone)]
pub struct RotateLinkPredictor {
    /// Embedding dimension
    pub dim: usize,
    /// Number of entities
    pub n_entities: usize,
    /// Number of relation types
    pub n_relations: usize,
    /// Margin gamma
    pub gamma: f64,
    /// Entity embeddings: real parts \[n\]\[dim\]
    pub ent_re: Vec<Vec<f64>>,
    /// Entity embeddings: imaginary parts \[n\]\[dim\]
    pub ent_im: Vec<Vec<f64>>,
    /// Relation phase \[n_relations\]\[dim\]
    pub rel_phase: Vec<Vec<f64>>,
}

impl RotateLinkPredictor {
    /// Build a RotatE link predictor.
    pub fn new(
        n_entities: usize,
        n_relations: usize,
        dim: usize,
        gamma: f64,
        rng: &mut StdRng,
    ) -> Self {
        use std::f64::consts::PI;
        let ent_re = (0..n_entities)
            .map(|_| xavier_vec(dim, n_entities, dim, rng))
            .collect();
        let ent_im = (0..n_entities)
            .map(|_| xavier_vec(dim, n_entities, dim, rng))
            .collect();
        // Phases uniformly in [-pi, pi]
        let rel_phase = (0..n_relations)
            .map(|_| {
                (0..dim)
                    .map(|_| rng.random::<f64>() * 2.0 * PI - PI)
                    .collect()
            })
            .collect();
        Self {
            dim,
            n_entities,
            n_relations,
            gamma,
            ent_re,
            ent_im,
            rel_phase,
        }
    }

    /// RotatE score: gamma - ||h o r - t||  where o is complex element-wise multiply.
    pub fn score(&self, h: usize, r: usize, t: usize) -> f64 {
        if h >= self.n_entities || r >= self.n_relations || t >= self.n_entities {
            return 0.0;
        }
        let dist_sq: f64 = (0..self.dim)
            .map(|d| {
                let (hr, hi) = (self.ent_re[h][d], self.ent_im[h][d]);
                let phase = self.rel_phase[r][d];
                let (rr, ri) = (phase.cos(), phase.sin());
                // h * r (complex multiply)
                let prod_re = hr * rr - hi * ri;
                let prod_im = hr * ri + hi * rr;
                let (tr, ti) = (self.ent_re[t][d], self.ent_im[t][d]);
                (prod_re - tr).powi(2) + (prod_im - ti).powi(2)
            })
            .sum();
        self.gamma - dist_sq.sqrt()
    }
}

/// Node-level RNN for graph generation: predict node type at each step.
#[derive(Debug, Clone)]
pub struct GraphRnnNode {
    /// Hidden state dimension
    pub d_hidden: usize,
    /// Number of node types to predict
    pub n_node_types: usize,
    /// GRU update gate weight (input)
    pub wz: Vec<f64>,
    /// GRU reset gate weight (input)
    pub wr: Vec<f64>,
    /// GRU candidate hidden weight (input)
    pub wh: Vec<f64>,
    /// GRU update gate weight (hidden)
    pub uz: Vec<f64>,
    /// GRU reset gate weight (hidden)
    pub ur: Vec<f64>,
    /// GRU candidate hidden weight (hidden)
    pub uh: Vec<f64>,
    /// Output projection to node type logits
    pub out_w: Vec<Vec<f64>>,
}

impl GraphRnnNode {
    /// Build a node-level graph RNN.
    pub fn new(d_hidden: usize, n_node_types: usize, rng: &mut StdRng) -> Self {
        let s = d_hidden * d_hidden;
        let wz = xavier_vec(s, d_hidden, d_hidden, rng);
        let wr = xavier_vec(s, d_hidden, d_hidden, rng);
        let wh = xavier_vec(s, d_hidden, d_hidden, rng);
        let uz = xavier_vec(s, d_hidden, d_hidden, rng);
        let ur = xavier_vec(s, d_hidden, d_hidden, rng);
        let uh = xavier_vec(s, d_hidden, d_hidden, rng);
        let out_w = (0..n_node_types)
            .map(|_| xavier_vec(d_hidden, d_hidden, n_node_types, rng))
            .collect();
        Self {
            d_hidden,
            n_node_types,
            wz,
            wr,
            wh,
            uz,
            ur,
            uh,
            out_w,
        }
    }

    fn gru_step(&self, h: &[f64], x: &[f64]) -> Vec<f64> {
        let d = self.d_hidden;
        let z: Vec<f64> = (0..d)
            .map(|j| {
                sigmoid(
                    (0..d.min(x.len()))
                        .map(|k| x[k] * self.wz[k * d + j])
                        .sum::<f64>()
                        + (0..d.min(h.len()))
                            .map(|k| h[k] * self.uz[k * d + j])
                            .sum::<f64>(),
                )
            })
            .collect();
        let r: Vec<f64> = (0..d)
            .map(|j| {
                sigmoid(
                    (0..d.min(x.len()))
                        .map(|k| x[k] * self.wr[k * d + j])
                        .sum::<f64>()
                        + (0..d.min(h.len()))
                            .map(|k| h[k] * self.ur[k * d + j])
                            .sum::<f64>(),
                )
            })
            .collect();
        let rh: Vec<f64> = r.iter().zip(h.iter()).map(|(ri, hi)| ri * hi).collect();
        let h_tilde: Vec<f64> = (0..d)
            .map(|j| {
                ((0..d.min(x.len()))
                    .map(|k| x[k] * self.wh[k * d + j])
                    .sum::<f64>()
                    + (0..d.min(rh.len()))
                        .map(|k| rh[k] * self.uh[k * d + j])
                        .sum::<f64>())
                .tanh()
            })
            .collect();
        z.iter()
            .zip(h.iter())
            .zip(h_tilde.iter())
            .map(|((zi, hi), hti)| (1.0 - zi) * hi + zi * hti)
            .collect()
    }

    /// Generate a sequence of node type probabilities.
    pub fn generate(&self, n_steps: usize) -> Vec<Vec<f64>> {
        let mut h = vec![0.0f64; self.d_hidden];
        let mut results = Vec::new();
        for _ in 0..n_steps {
            let x = vec![0.0f64; self.d_hidden]; // zero input (start token)
            h = self.gru_step(&h, &x);
            let logits: Vec<f64> = (0..self.n_node_types)
                .map(|i| dot(&h, &self.out_w[i]))
                .collect();
            results.push(softmax_1d(&logits));
        }
        results
    }
}

/// Edge-level RNN: for each new node, decide edges to previous nodes.
#[derive(Debug, Clone)]
pub struct GraphRnnEdge {
    /// Hidden state dimension
    pub d_hidden: usize,
    /// Edge existence predictor \[d_hidden\]
    pub edge_pred_w: Vec<f64>,
    /// GRU gate weights (input)
    pub wz: Vec<f64>,
    /// GRU reset weight (input)
    pub wr: Vec<f64>,
    /// GRU candidate weight (input)
    pub wh: Vec<f64>,
    /// GRU gate weights (hidden)
    pub uz: Vec<f64>,
    /// GRU reset weight (hidden)
    pub ur: Vec<f64>,
    /// GRU candidate weight (hidden)
    pub uh: Vec<f64>,
}

impl GraphRnnEdge {
    /// Build an edge-level graph RNN.
    pub fn new(d_hidden: usize, rng: &mut StdRng) -> Self {
        let s = d_hidden * d_hidden;
        Self {
            d_hidden,
            edge_pred_w: xavier_vec(d_hidden, d_hidden, 1, rng),
            wz: xavier_vec(s, d_hidden, d_hidden, rng),
            wr: xavier_vec(s, d_hidden, d_hidden, rng),
            wh: xavier_vec(s, d_hidden, d_hidden, rng),
            uz: xavier_vec(s, d_hidden, d_hidden, rng),
            ur: xavier_vec(s, d_hidden, d_hidden, rng),
            uh: xavier_vec(s, d_hidden, d_hidden, rng),
        }
    }

    fn gru_step(&self, h: &[f64], x: &[f64]) -> Vec<f64> {
        let d = self.d_hidden;
        let z: Vec<f64> = (0..d)
            .map(|j| {
                sigmoid(
                    (0..d.min(x.len()))
                        .map(|k| x[k] * self.wz[k * d + j])
                        .sum::<f64>()
                        + (0..d.min(h.len()))
                            .map(|k| h[k] * self.uz[k * d + j])
                            .sum::<f64>(),
                )
            })
            .collect();
        let r: Vec<f64> = (0..d)
            .map(|j| {
                sigmoid(
                    (0..d.min(x.len()))
                        .map(|k| x[k] * self.wr[k * d + j])
                        .sum::<f64>()
                        + (0..d.min(h.len()))
                            .map(|k| h[k] * self.ur[k * d + j])
                            .sum::<f64>(),
                )
            })
            .collect();
        let rh: Vec<f64> = r.iter().zip(h.iter()).map(|(ri, hi)| ri * hi).collect();
        let h_tilde: Vec<f64> = (0..d)
            .map(|j| {
                ((0..d.min(x.len()))
                    .map(|k| x[k] * self.wh[k * d + j])
                    .sum::<f64>()
                    + (0..d.min(rh.len()))
                        .map(|k| rh[k] * self.uh[k * d + j])
                        .sum::<f64>())
                .tanh()
            })
            .collect();
        z.iter()
            .zip(h.iter())
            .zip(h_tilde.iter())
            .map(|((zi, hi), hti)| (1.0 - zi) * hi + zi * hti)
            .collect()
    }

    /// For a new node at position `node_idx`, predict edges to previous `node_idx` nodes.
    pub fn predict_edges(&self, node_idx: usize, node_hidden: &[f64]) -> Vec<f64> {
        let mut h = node_hidden.to_vec();
        let mut edge_probs = Vec::new();
        for _ in 0..node_idx {
            h = self.gru_step(&h, &vec![0.0; self.d_hidden]);
            edge_probs.push(sigmoid(dot(&h, &self.edge_pred_w)));
        }
        edge_probs
    }
}

/// Molecule generator: junction-tree VAE style (scaffold + fragments).
#[derive(Debug, Clone)]
pub struct MoleculeGenerator {
    /// Latent space dimension
    pub d_latent: usize,
    /// Number of atom types
    pub n_atom_types: usize,
    /// Number of bond types
    pub n_bond_types: usize,
    /// Scaffold encoder weights
    pub scaffold_enc: Vec<Vec<f64>>,
    /// Fragment attachment weights
    pub frag_attach: Vec<Vec<f64>>,
    /// Atom type decoder
    pub atom_dec: Vec<Vec<f64>>,
    /// Bond type decoder
    pub bond_dec: Vec<Vec<f64>>,
}

impl MoleculeGenerator {
    /// Build a molecule generator with junction-tree VAE style.
    pub fn new(
        d_latent: usize,
        n_atom_types: usize,
        n_bond_types: usize,
        rng: &mut StdRng,
    ) -> Self {
        let scaffold_enc = (0..d_latent)
            .map(|_| xavier_vec(d_latent, d_latent, d_latent, rng))
            .collect();
        let frag_attach = (0..d_latent)
            .map(|_| xavier_vec(d_latent, d_latent, d_latent, rng))
            .collect();
        let atom_dec = (0..n_atom_types)
            .map(|_| xavier_vec(d_latent, d_latent, n_atom_types, rng))
            .collect();
        let bond_dec = (0..n_bond_types)
            .map(|_| xavier_vec(d_latent, d_latent, n_bond_types, rng))
            .collect();
        Self {
            d_latent,
            n_atom_types,
            n_bond_types,
            scaffold_enc,
            frag_attach,
            atom_dec,
            bond_dec,
        }
    }

    /// Encode a scaffold graph to latent vector (mean of node projections).
    pub fn encode_scaffold(&self, scaffold_feats: &[Vec<f64>]) -> Vec<f64> {
        if scaffold_feats.is_empty() {
            return vec![0.0; self.d_latent];
        }
        let d = self.d_latent.min(if scaffold_feats.is_empty() {
            0
        } else {
            scaffold_feats[0].len()
        });
        let encoded: Vec<Vec<f64>> = scaffold_feats
            .iter()
            .map(|f| {
                (0..self.d_latent)
                    .map(|j| dot(&f[..d], &self.scaffold_enc[j][..d]))
                    .collect()
            })
            .collect();
        (0..self.d_latent)
            .map(|di| encoded.iter().map(|e| e[di]).sum::<f64>() / encoded.len() as f64)
            .collect()
    }

    /// Predict atom types for new nodes (given scaffold latent).
    pub fn decode_atoms(&self, latent: &[f64], n_atoms: usize) -> Vec<Vec<f64>> {
        let d = self.d_latent.min(latent.len());
        (0..n_atoms)
            .map(|_| {
                let logits: Vec<f64> = (0..self.n_atom_types)
                    .map(|i| dot(&latent[..d], &self.atom_dec[i][..d]))
                    .collect();
                softmax_1d(&logits)
            })
            .collect()
    }

    /// Predict bond types for given pairs.
    pub fn decode_bonds(&self, latent: &[f64], n_bonds: usize) -> Vec<Vec<f64>> {
        let d = self.d_latent.min(latent.len());
        (0..n_bonds)
            .map(|_| {
                let logits: Vec<f64> = (0..self.n_bond_types)
                    .map(|i| dot(&latent[..d], &self.bond_dec[i][..d]))
                    .collect();
                softmax_1d(&logits)
            })
            .collect()
    }
}
