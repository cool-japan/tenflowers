//! Spatio-Temporal Graph Convolutional Network (ST-GCN) — Yan et al. 2018.
//!
//! TemporalConv, StGcnLayer, StGcnBlock, StGcnModel.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

use super::types::{linear, matvec, rand_mat, zero_vec, PartitionStrategy};

// ─────────────────────────────────────────────────────────────────────────────
// TemporalConv — 1D causal convolution over time
// ─────────────────────────────────────────────────────────────────────────────

/// 1D causal temporal convolution applied to [C_in × T] feature maps.
///
/// Output shape: [C_out × T] (causal: no future leakage).
#[derive(Debug, Clone)]
pub struct TemporalConv {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    /// Weight tensor [out_channels × in_channels × kernel_size].
    pub weight: Vec<Vec<Vec<f64>>>,
    pub bias: Vec<f64>,
}

impl TemporalConv {
    /// Create a new TemporalConv layer.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let s = (2.0 / (in_channels * kernel_size) as f64).sqrt();
        let weight: Vec<Vec<Vec<f64>>> = (0..out_channels)
            .map(|_| {
                (0..in_channels)
                    .map(|_| {
                        (0..kernel_size)
                            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * s)
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let bias = zero_vec(out_channels);
        Self {
            in_channels,
            out_channels,
            kernel_size,
            weight,
            bias,
        }
    }

    /// Apply causal 1D convolution.
    ///
    /// Input: `x[in_channels][T]`, Output: `y[out_channels][T]`.
    pub fn forward(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if x.len() != self.in_channels {
            return Err(TensorError::invalid_argument(format!(
                "TemporalConv: in_channels mismatch {} vs {}",
                x.len(),
                self.in_channels
            )));
        }
        let t = if x.is_empty() { 0 } else { x[0].len() };
        let mut out: Vec<Vec<f64>> = (0..self.out_channels).map(|_| vec![0.0; t]).collect();

        for oc in 0..self.out_channels {
            for ti in 0..t {
                let mut acc = self.bias[oc];
                for ic in 0..self.in_channels {
                    for k in 0..self.kernel_size {
                        // Causal: only use past steps (including current).
                        if ti + 1 > k {
                            let from = ti - k;
                            acc += self.weight[oc][ic][k] * x[ic][from];
                        }
                        // Else implicit zero-padding.
                    }
                }
                out[oc][ti] = acc;
            }
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StGcnLayer — spatial GCN + temporal conv
// ─────────────────────────────────────────────────────────────────────────────

/// One ST-GCN layer: partitioned spatial GCN followed by temporal convolution.
///
/// Input shape:  `[N × C_in × T]` — N nodes, C_in channels, T time-steps.
/// Output shape: `[N × C_out × T]`.
#[derive(Debug, Clone)]
pub struct StGcnLayer {
    pub n_nodes: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub strategy: PartitionStrategy,
    /// Learnable weight matrices per partition [n_subsets × out_channels × in_channels].
    pub spatial_w: Vec<Vec<Vec<f64>>>,
    pub temporal_conv: TemporalConv,
}

impl StGcnLayer {
    /// Create a new ST-GCN layer.
    pub fn new(
        n_nodes: usize,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        strategy: PartitionStrategy,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let s = (2.0 / (in_channels * n_nodes) as f64).sqrt();
        let n_subsets = strategy.n_subsets();
        let spatial_w: Vec<Vec<Vec<f64>>> = (0..n_subsets)
            .map(|_| rand_mat(out_channels, in_channels, s, &mut rng))
            .collect();
        let temporal_conv = TemporalConv::new(
            out_channels,
            out_channels,
            kernel_size,
            seed.wrapping_add(7),
        );
        Self {
            n_nodes,
            in_channels,
            out_channels,
            strategy,
            spatial_w,
            temporal_conv,
        }
    }

    /// Forward pass.
    ///
    /// `x`: `[N][C_in][T]`, `adj`: `[n_subsets][N][N]` (normalised adjacency).
    /// Returns `[N][C_out][T]`.
    pub fn forward(
        &self,
        x: &[Vec<Vec<f64>>],
        adj: &[Vec<Vec<f64>>],
    ) -> Result<Vec<Vec<Vec<f64>>>> {
        let n = x.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let t = x[0][0].len();
        let n_subsets = adj.len();
        if n_subsets != self.strategy.n_subsets() {
            return Err(TensorError::invalid_argument(format!(
                "StGcnLayer: adj subset count {} != expected {}",
                n_subsets,
                self.strategy.n_subsets()
            )));
        }

        // For each subset s: compute Z_s[n][out_c] = W_s · sum_j(A_s[n,j] * x[j])
        // Then aggregate subsets and apply temporal conv per node.
        let mut spatial_out: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; t]; self.out_channels]; n];

        for s in 0..n_subsets {
            for node in 0..n {
                // Aggregate neighbour features: agg_feat[C_in][T]
                let mut agg: Vec<Vec<f64>> = vec![vec![0.0; t]; self.in_channels];
                for nb in 0..n {
                    let a = adj[s][node][nb];
                    if a.abs() < 1e-12 {
                        continue;
                    }
                    for c in 0..self.in_channels {
                        for ti in 0..t {
                            agg[c][ti] += a * x[nb][c][ti];
                        }
                    }
                }
                // Apply spatial weight: [out_c × in_c] × [in_c] per time step.
                for ti in 0..t {
                    let feat_t: Vec<f64> = agg.iter().map(|ch| ch[ti]).collect();
                    let out_t = matvec(&self.spatial_w[s], &feat_t);
                    for oc in 0..self.out_channels {
                        spatial_out[node][oc][ti] += out_t[oc];
                    }
                }
            }
        }

        // Apply temporal convolution per node.
        let mut out: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n);
        for node in 0..n {
            let tmp = self.temporal_conv.forward(&spatial_out[node])?;
            out.push(tmp);
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StGcnBlock — layer + BN + dropout + residual
// ─────────────────────────────────────────────────────────────────────────────

/// ST-GCN residual block with optional skip connection.
#[derive(Debug, Clone)]
pub struct StGcnBlock {
    pub layer: StGcnLayer,
    pub dropout_rate: f64,
    /// Skip projection (identity or 1×1 conv if channel sizes differ).
    skip_w: Option<Vec<Vec<f64>>>,
}

impl StGcnBlock {
    /// Create a new ST-GCN block.
    pub fn new(
        n_nodes: usize,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        strategy: PartitionStrategy,
        dropout_rate: f64,
        seed: u64,
    ) -> Self {
        let layer = StGcnLayer::new(
            n_nodes,
            in_channels,
            out_channels,
            kernel_size,
            strategy,
            seed,
        );
        let skip_w = if in_channels != out_channels {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(100));
            let s = (2.0 / in_channels as f64).sqrt();
            Some(rand_mat(out_channels, in_channels, s, &mut rng))
        } else {
            None
        };
        Self {
            layer,
            dropout_rate,
            skip_w,
        }
    }

    /// Forward pass with residual connection (dropout applied at training=true).
    pub fn forward(
        &self,
        x: &[Vec<Vec<f64>>],
        adj: &[Vec<Vec<f64>>],
        training: bool,
        seed: u64,
    ) -> Result<Vec<Vec<Vec<f64>>>> {
        let mut z = self.layer.forward(x, adj)?;

        // Dropout.
        if training && self.dropout_rate > 0.0 {
            let mut rng = StdRng::seed_from_u64(seed);
            let keep = 1.0 - self.dropout_rate;
            for node in &mut z {
                for ch in node {
                    for v in ch {
                        if rng.random::<f64>() < self.dropout_rate {
                            *v = 0.0;
                        } else {
                            *v /= keep;
                        }
                    }
                }
            }
        }

        // Residual.
        let n = x.len();
        let t = if n > 0 { x[0][0].len() } else { 0 };
        for node in 0..n {
            for ti in 0..t {
                match &self.skip_w {
                    None => {
                        // Identity skip.
                        for oc in 0..self.layer.out_channels {
                            z[node][oc][ti] += x[node][oc][ti];
                        }
                    }
                    Some(sw) => {
                        let feat: Vec<f64> = x[node].iter().map(|ch| ch[ti]).collect();
                        let skip = matvec(sw, &feat);
                        for oc in 0..self.layer.out_channels {
                            z[node][oc][ti] += skip[oc];
                        }
                    }
                }
            }
        }
        Ok(z)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StGcnModel
// ─────────────────────────────────────────────────────────────────────────────

/// Full ST-GCN model for skeleton-based action recognition.
#[derive(Debug, Clone)]
pub struct StGcnModel {
    pub blocks: Vec<StGcnBlock>,
    /// Adjacency matrices for each subset.
    pub adj: Vec<Vec<Vec<f64>>>,
    /// Classification head [n_classes × final_channels].
    cls_w: Vec<Vec<f64>>,
    cls_b: Vec<f64>,
    pub n_classes: usize,
}

impl StGcnModel {
    /// Build a ST-GCN model.
    ///
    /// `channel_seq`: sequence of (in_c, out_c) pairs for each block.
    pub fn new(
        n_nodes: usize,
        channel_seq: &[(usize, usize)],
        n_classes: usize,
        kernel_size: usize,
        strategy: PartitionStrategy,
        adj: Vec<Vec<Vec<f64>>>,
        seed: u64,
    ) -> Result<Self> {
        if channel_seq.is_empty() {
            return Err(TensorError::invalid_argument(
                "StGcnModel: channel_seq must not be empty".to_string(),
            ));
        }
        let blocks: Vec<StGcnBlock> = channel_seq
            .iter()
            .enumerate()
            .map(|(i, &(in_c, out_c))| {
                StGcnBlock::new(
                    n_nodes,
                    in_c,
                    out_c,
                    kernel_size,
                    strategy,
                    0.1,
                    seed.wrapping_add(i as u64 * 31),
                )
            })
            .collect();
        let final_ch = channel_seq.last().map(|(_, o)| *o).unwrap_or(1);
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(999));
        let s = (2.0 / final_ch as f64).sqrt();
        let cls_w = rand_mat(n_classes, final_ch, s, &mut rng);
        let cls_b = zero_vec(n_classes);
        Ok(Self {
            blocks,
            adj,
            cls_w,
            cls_b,
            n_classes,
        })
    }

    /// Forward pass: returns class logits \[n_classes\].
    ///
    /// Input `x`: `[N][C_in][T]`.
    pub fn forward(&self, x: &[Vec<Vec<f64>>]) -> Result<Vec<f64>> {
        let mut h = x.to_vec();
        for block in &self.blocks {
            h = block.forward(&h, &self.adj, false, 0)?;
        }
        // Global average pool over nodes and time.
        let n = h.len();
        if n == 0 {
            return Ok(vec![0.0; self.n_classes]);
        }
        let c = h[0].len();
        let t = if c > 0 { h[0][0].len() } else { 0 };
        let mut pooled = vec![0.0f64; c];
        let count = (n * t).max(1) as f64;
        for node in &h {
            for (ci, ch) in node.iter().enumerate() {
                let s: f64 = ch.iter().sum();
                pooled[ci] += s / count;
            }
        }
        let logits = linear(&self.cls_w, &self.cls_b, &pooled);
        Ok(logits)
    }
}
