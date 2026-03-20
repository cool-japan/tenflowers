//! DyGFormer-style dynamic graph transformer components.
//!
//! NeighborSampler, CoOccurrenceEncoder, DyGFormerLayer, DynamicGraphTransformer.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use tenflowers_core::{Result, TensorError};

use super::tgn::{NodeMemory, TimeEncoder};
use super::types::{linear, matvec, rand_mat, relu, softmax, vecadd, zero_vec, TemporalEdge};

// ─────────────────────────────────────────────────────────────────────────────
// NeighborSampler
// ─────────────────────────────────────────────────────────────────────────────

/// Samples the most-recent k temporal neighbours for each query node.
#[derive(Debug, Clone)]
pub struct NeighborSampler {
    /// Maximum number of recent neighbours to return.
    pub k: usize,
    /// Edge list sorted by time (most recent first per node).
    edges: Vec<TemporalEdge>,
}

impl NeighborSampler {
    /// Build from a list of temporal edges (will be sorted internally).
    pub fn new(mut edges: Vec<TemporalEdge>, k: usize) -> Self {
        edges.sort_by(|a, b| {
            b.time
                .partial_cmp(&a.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self { k, edges }
    }

    /// Return up to `k` most-recent neighbours of `node` before time `before_time`.
    pub fn sample(&self, node: usize, before_time: f64) -> Vec<&TemporalEdge> {
        self.edges
            .iter()
            .filter(|e| (e.src == node || e.dst == node) && e.time < before_time)
            .take(self.k)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CoOccurrenceEncoder
// ─────────────────────────────────────────────────────────────────────────────

/// Encodes binary co-occurrence patterns of shared neighbours across time windows.
///
/// For each pair (node, neighbour) we compute a binary vector indicating which
/// time-window slots both nodes co-appeared as neighbours of some anchor.
#[derive(Debug, Clone)]
pub struct CoOccurrenceEncoder {
    /// Number of time windows.
    pub n_windows: usize,
    /// Total time span (used to map timestamps → window indices).
    pub time_span: f64,
}

impl CoOccurrenceEncoder {
    /// Create a CoOccurrenceEncoder.
    pub fn new(n_windows: usize, time_span: f64) -> Result<Self> {
        if n_windows == 0 {
            return Err(TensorError::invalid_argument(
                "CoOccurrenceEncoder: n_windows must be > 0".to_string(),
            ));
        }
        if time_span <= 0.0 {
            return Err(TensorError::invalid_argument(
                "CoOccurrenceEncoder: time_span must be > 0".to_string(),
            ));
        }
        Ok(Self {
            n_windows,
            time_span,
        })
    }

    /// Compute co-occurrence encoding: binary indicator per window.
    ///
    /// Returns a `[n_windows]` vector where entry `w == 1.0` if `node_a` and
    /// `node_b` both appeared in window `w` among `edges`.
    pub fn encode(&self, node_a: usize, node_b: usize, edges: &[TemporalEdge]) -> Vec<f64> {
        let window_size = self.time_span / self.n_windows as f64;
        let mut windows_a = vec![false; self.n_windows];
        let mut windows_b = vec![false; self.n_windows];

        for e in edges {
            let involved_a = e.src == node_a || e.dst == node_a;
            let involved_b = e.src == node_b || e.dst == node_b;
            let window_idx = ((e.time / window_size).floor() as isize)
                .rem_euclid(self.n_windows as isize) as usize;
            let window_idx = window_idx.min(self.n_windows - 1);
            if involved_a {
                windows_a[window_idx] = true;
            }
            if involved_b {
                windows_b[window_idx] = true;
            }
        }

        windows_a
            .iter()
            .zip(windows_b.iter())
            .map(|(&a, &b)| if a && b { 1.0 } else { 0.0 })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DyGFormerLayer
// ─────────────────────────────────────────────────────────────────────────────

/// Single DyGFormer layer: Transformer over temporal neighbours enriched with
/// co-occurrence patterns.
#[derive(Debug, Clone)]
pub struct DyGFormerLayer {
    pub input_dim: usize,
    pub n_heads: usize,
    pub hidden_dim: usize,
    // Co-occurrence input is prepended to input_dim.
    cooc_dim: usize,
    // Multi-head Q/K/V weights per head.
    w_q: Vec<Vec<Vec<f64>>>,
    w_k: Vec<Vec<Vec<f64>>>,
    w_v: Vec<Vec<Vec<f64>>>,
    head_dim: usize,
    // Output projection.
    w_out: Vec<Vec<f64>>,
    b_out: Vec<f64>,
    // FFN.
    w_ff1: Vec<Vec<f64>>,
    b_ff1: Vec<f64>,
    w_ff2: Vec<Vec<f64>>,
    b_ff2: Vec<f64>,
}

impl DyGFormerLayer {
    /// Create a new DyGFormer layer.
    pub fn new(
        input_dim: usize,
        cooc_dim: usize,
        n_heads: usize,
        hidden_dim: usize,
        seed: u64,
    ) -> Result<Self> {
        if n_heads == 0 {
            return Err(TensorError::invalid_argument(
                "DyGFormerLayer: n_heads must be > 0".to_string(),
            ));
        }
        let total_input = input_dim + cooc_dim;
        let head_dim = (total_input + n_heads - 1) / n_heads;
        let mut rng = StdRng::seed_from_u64(seed);
        let s = (2.0 / total_input as f64).sqrt();

        let w_q: Vec<Vec<Vec<f64>>> = (0..n_heads)
            .map(|_| rand_mat(head_dim, total_input, s, &mut rng))
            .collect();
        let w_k: Vec<Vec<Vec<f64>>> = (0..n_heads)
            .map(|_| rand_mat(head_dim, total_input, s, &mut rng))
            .collect();
        let w_v: Vec<Vec<Vec<f64>>> = (0..n_heads)
            .map(|_| rand_mat(head_dim, total_input, s, &mut rng))
            .collect();

        let mha_out_dim = n_heads * head_dim;
        let w_out = rand_mat(total_input, mha_out_dim, s, &mut rng);
        let b_out = zero_vec(total_input);

        let w_ff1 = rand_mat(hidden_dim, total_input, s, &mut rng);
        let b_ff1 = zero_vec(hidden_dim);
        let w_ff2 = rand_mat(total_input, hidden_dim, s, &mut rng);
        let b_ff2 = zero_vec(total_input);

        Ok(Self {
            input_dim,
            n_heads,
            hidden_dim,
            cooc_dim,
            w_q,
            w_k,
            w_v,
            head_dim,
            w_out,
            b_out,
            w_ff1,
            b_ff1,
            w_ff2,
            b_ff2,
        })
    }

    /// Forward pass over a sequence of neighbour tokens.
    ///
    /// `tokens`: list of (input_feat, cooc_feat) pairs — one per neighbour.
    /// Returns output vectors parallel to `tokens`.
    pub fn forward(&self, tokens: &[(Vec<f64>, Vec<f64>)]) -> Result<Vec<Vec<f64>>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        // Build combined token matrix.
        let combined: Vec<Vec<f64>> = tokens
            .iter()
            .map(|(inp, cooc)| {
                let mut v = inp.clone();
                v.extend_from_slice(cooc);
                v
            })
            .collect();

        let seq_len = combined.len();
        // Multi-head self-attention.
        let mut head_outputs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(self.n_heads);
        for h in 0..self.n_heads {
            let queries: Vec<Vec<f64>> = combined.iter().map(|t| matvec(&self.w_q[h], t)).collect();
            let keys: Vec<Vec<f64>> = combined.iter().map(|t| matvec(&self.w_k[h], t)).collect();
            let values: Vec<Vec<f64>> = combined.iter().map(|t| matvec(&self.w_v[h], t)).collect();

            let mut head_out: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
            for q in &queries {
                let scores: Vec<f64> = keys
                    .iter()
                    .map(|k| {
                        q.iter()
                            .zip(k.iter())
                            .map(|(&qi, &ki)| qi * ki)
                            .sum::<f64>()
                            / (self.head_dim as f64).sqrt()
                    })
                    .collect();
                let attn = softmax(&scores);
                let mut ctx = vec![0.0; self.head_dim];
                for (a, v) in attn.iter().zip(values.iter()) {
                    for (ci, vi) in ctx.iter_mut().zip(v.iter()) {
                        *ci += a * vi;
                    }
                }
                head_out.push(ctx);
            }
            head_outputs.push(head_out);
        }

        // Concatenate heads → output projection → residual add.
        let mut out: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let mut cat: Vec<f64> = Vec::with_capacity(self.n_heads * self.head_dim);
            for h in 0..self.n_heads {
                cat.extend_from_slice(&head_outputs[h][i]);
            }
            let proj = linear(&self.w_out, &self.b_out, &cat);
            // Add residual from combined[i] (truncated / padded to match).
            let res = vecadd(
                &proj,
                &combined[i]
                    .iter()
                    .cloned()
                    .chain(std::iter::repeat(0.0))
                    .take(proj.len())
                    .collect::<Vec<_>>(),
            );
            // FFN + residual.
            let ff1 = matvec(&self.w_ff1, &res)
                .iter()
                .zip(self.b_ff1.iter())
                .map(|(&x, &b)| relu(x + b))
                .collect::<Vec<_>>();
            let ff2 = linear(&self.w_ff2, &self.b_ff2, &ff1);
            let final_out = vecadd(&res, &ff2);
            out.push(final_out);
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DynamicGraphTransformer
// ─────────────────────────────────────────────────────────────────────────────

/// Stack of DyGFormer layers with a shared NeighborSampler and memory.
#[derive(Debug, Clone)]
pub struct DynamicGraphTransformer {
    pub layers: Vec<DyGFormerLayer>,
    pub sampler: NeighborSampler,
    pub cooc_encoder: CoOccurrenceEncoder,
    pub memory: NodeMemory,
    pub time_encoder: TimeEncoder,
}

impl DynamicGraphTransformer {
    /// Build a DynamicGraphTransformer.
    pub fn new(
        n_nodes: usize,
        memory_dim: usize,
        n_layers: usize,
        n_heads: usize,
        hidden_dim: usize,
        k_neighbours: usize,
        n_windows: usize,
        time_span: f64,
        edges: Vec<TemporalEdge>,
        seed: u64,
    ) -> Result<Self> {
        let cooc_encoder = CoOccurrenceEncoder::new(n_windows, time_span)?;
        let time_dim = 16usize;
        let input_dim = memory_dim + time_dim;
        let cooc_dim = n_windows;
        let layers: Result<Vec<_>> = (0..n_layers)
            .map(|i| {
                DyGFormerLayer::new(
                    input_dim,
                    cooc_dim,
                    n_heads,
                    hidden_dim,
                    seed.wrapping_add(i as u64),
                )
            })
            .collect();
        let layers = layers?;
        Ok(Self {
            layers,
            sampler: NeighborSampler::new(edges, k_neighbours),
            cooc_encoder,
            memory: NodeMemory::new(n_nodes, memory_dim),
            time_encoder: TimeEncoder::new(time_dim, seed.wrapping_add(42)),
        })
    }

    /// Compute DyGFormer embedding for `node` at time `t`.
    pub fn embed(&self, node: usize, t: f64, all_edges: &[TemporalEdge]) -> Result<Vec<f64>> {
        let neighbours = self.sampler.sample(node, t);
        if neighbours.is_empty() {
            let (mem, _) = self.memory.get(node)?;
            return Ok(mem.clone());
        }

        // Build tokens: (memory + time_enc, cooc_pattern).
        let mut tokens: Vec<(Vec<f64>, Vec<f64>)> = Vec::with_capacity(neighbours.len());
        for e in &neighbours {
            let nb = if e.src == node { e.dst } else { e.src };
            let (mem_nb, t_nb) = self.memory.get(nb)?;
            let dt = (t - t_nb).max(0.0);
            let enc = self.time_encoder.encode(dt);
            let mut inp = mem_nb.clone();
            inp.extend_from_slice(&enc);
            let cooc = self.cooc_encoder.encode(node, nb, all_edges);
            tokens.push((inp, cooc));
        }

        let mut out = tokens.clone();
        for layer in &self.layers {
            let layer_out = layer.forward(&out)?;
            out = layer_out
                .into_iter()
                .zip(out)
                .map(|(l, orig)| (l, orig.1))
                .collect();
        }

        // Aggregate by mean.
        if out.is_empty() {
            let (mem, _) = self.memory.get(node)?;
            return Ok(mem.clone());
        }
        let dim = out[0].0.len();
        let mut agg = vec![0.0f64; dim];
        for (v, _) in &out {
            for (a, x) in agg.iter_mut().zip(v.iter()) {
                *a += x;
            }
        }
        let n = out.len() as f64;
        Ok(agg.iter().map(|x| x / n).collect())
    }
}
