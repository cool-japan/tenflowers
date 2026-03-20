//! §3 Advanced Graph Pooling.

use super::spectral::GraphLaplacian;
use super::utils::{degree_vec, dot, rand_weight};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

/// Differentiable pooling layer (DiffPool).
#[derive(Debug, Clone)]
pub struct DiffPoolLayer {
    pub w_s: Vec<Vec<f64>>,
    pub w_z: Vec<Vec<f64>>,
    pub n_clusters: usize,
    pub embed_dim: usize,
}

impl DiffPoolLayer {
    pub fn new(in_features: usize, n_clusters: usize, embed_dim: usize, seed: u64) -> Self {
        Self {
            w_s: rand_weight(in_features, n_clusters, seed),
            w_z: rand_weight(in_features, embed_dim, seed + 1),
            n_clusters,
            embed_dim,
        }
    }

    pub fn pool(
        &self,
        x: &[Vec<f64>],
        adj: &[Vec<f64>],
        s: &[Vec<f64>],
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n = x.len();
        let k = self.n_clusters;
        if n == 0 {
            return (vec![vec![0.0; self.embed_dim]; k], vec![vec![0.0; k]; k]);
        }

        let s_logits: Vec<Vec<f64>> = x
            .iter()
            .map(|xi| {
                (0..k)
                    .map(|c| {
                        self.w_s
                            .iter()
                            .enumerate()
                            .map(|(f, row)| xi.get(f).copied().unwrap_or(0.0) * row[c])
                            .sum()
                    })
                    .collect()
            })
            .collect();

        let s_use = if s.len() == n && s.first().map(|r| r.len()).unwrap_or(0) == k {
            s.to_vec()
        } else {
            s_logits
                .iter()
                .map(|row| {
                    let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exp: Vec<f64> = row.iter().map(|x| (x - max).exp()).collect();
                    let sum = exp.iter().sum::<f64>().max(1e-12);
                    exp.iter().map(|e| e / sum).collect()
                })
                .collect()
        };

        let z: Vec<Vec<f64>> = x
            .iter()
            .map(|xi| {
                (0..self.embed_dim)
                    .map(|d| {
                        self.w_z
                            .iter()
                            .enumerate()
                            .map(|(f, row)| xi.get(f).copied().unwrap_or(0.0) * row[d])
                            .sum()
                    })
                    .collect()
            })
            .collect();

        let mut x_pool = vec![vec![0.0; self.embed_dim]; k];
        for i in 0..n {
            for c in 0..k {
                for d in 0..self.embed_dim {
                    x_pool[c][d] += s_use[i][c] * z[i][d];
                }
            }
        }

        let mut tmp = vec![vec![0.0; k]; n];
        for i in 0..n {
            for j in 0..n {
                for c in 0..k {
                    tmp[i][c] += adj[i][j] * s_use[j][c];
                }
            }
        }
        let mut a_pool = vec![vec![0.0; k]; k];
        for i in 0..n {
            for c1 in 0..k {
                for c2 in 0..k {
                    a_pool[c1][c2] += s_use[i][c1] * tmp[i][c2];
                }
            }
        }

        (x_pool, a_pool)
    }
}

/// Self-attention graph pooling (SAGPool).
#[derive(Debug, Clone)]
pub struct SagPoolLayer {
    pub score_w: Vec<f64>,
    pub in_features: usize,
}

impl SagPoolLayer {
    pub fn new(in_features: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let score_w = (0..in_features)
            .map(|_| rng.random::<f64>() * 0.1)
            .collect();
        Self {
            score_w,
            in_features,
        }
    }

    pub fn forward(
        &self,
        x: &[Vec<f64>],
        adj: &[Vec<f64>],
        keep_ratio: f64,
    ) -> (Vec<Vec<f64>>, Vec<usize>) {
        let n = x.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let k = ((n as f64 * keep_ratio).ceil() as usize).max(1).min(n);

        let _lap = GraphLaplacian::compute(adj);
        let mut a_hat = adj.to_vec();
        for i in 0..n {
            a_hat[i][i] += 1.0;
        }
        let d_hat = degree_vec(&a_hat);
        let d_inv: Vec<f64> = d_hat.iter().map(|d| 1.0 / d).collect();

        let mut z = vec![vec![0.0; self.in_features]; n];
        for i in 0..n {
            for j in 0..n {
                let w = a_hat[i][j] * d_inv[i];
                for f in 0..self.in_features.min(x[j].len()) {
                    z[i][f] += w * x[j][f];
                }
            }
        }

        let scores: Vec<f64> = z.iter().map(|zi| (dot(zi, &self.score_w)).tanh()).collect();

        let mut idx_score: Vec<(usize, f64)> = scores.iter().copied().enumerate().collect();
        idx_score.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let selected: Vec<usize> = idx_score.iter().take(k).map(|(i, _)| *i).collect();

        let x_out: Vec<Vec<f64>> = selected
            .iter()
            .map(|&i| x[i].iter().map(|xi| xi * scores[i]).collect())
            .collect();

        (x_out, selected)
    }
}

/// ASAP: Adaptive Structure Aware Pooling.
#[derive(Debug, Clone)]
pub struct AsapPooling {
    pub attn_w: Vec<Vec<f64>>,
    pub in_features: usize,
}

impl AsapPooling {
    pub fn new(in_features: usize, seed: u64) -> Self {
        Self {
            attn_w: rand_weight(in_features, in_features, seed),
            in_features,
        }
    }

    pub fn forward(&self, x: &[Vec<f64>], adj: &[Vec<f64>], n_inducing: usize) -> Vec<Vec<f64>> {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }
        let k = n_inducing.min(n);
        let f = self.in_features;

        let wx: Vec<Vec<f64>> = x
            .iter()
            .map(|xi| {
                (0..f)
                    .map(|d| {
                        self.attn_w
                            .iter()
                            .enumerate()
                            .map(|(fd, row)| xi.get(fd).copied().unwrap_or(0.0) * row[d])
                            .sum()
                    })
                    .collect()
            })
            .collect();

        let scores: Vec<f64> = wx
            .iter()
            .enumerate()
            .map(|(i, wxi)| {
                let nbrs: Vec<f64> = (0..n)
                    .filter(|&j| adj[i][j] > 0.0)
                    .map(|j| dot(wxi, &x[j]))
                    .collect();
                if nbrs.is_empty() {
                    0.0
                } else {
                    nbrs.iter().sum::<f64>() / nbrs.len() as f64
                }
            })
            .collect();

        let mut idx_score: Vec<(usize, f64)> = scores.iter().copied().enumerate().collect();
        idx_score.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let inducing: Vec<usize> = idx_score.iter().take(k).map(|(i, _)| *i).collect();

        inducing
            .iter()
            .map(|&ind| {
                let mut agg = vec![0.0; f];
                let mut cnt = 0.0;
                for j in 0..n {
                    let w = if j == ind { 1.0 } else { adj[ind][j] };
                    if w > 0.0 {
                        for (a, xi) in agg.iter_mut().zip(x[j].iter().take(f)) {
                            *a += w * xi;
                        }
                        cnt += w;
                    }
                }
                agg.iter_mut().for_each(|a| *a /= cnt.max(1e-12));
                agg
            })
            .collect()
    }
}

/// Graclus-inspired graph coarsening.
#[derive(Debug, Clone)]
pub struct GraphCoarsening;

impl GraphCoarsening {
    pub fn coarsen(adj: &[Vec<f64>], x: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<usize>) {
        let n = adj.len();
        if n == 0 {
            return (Vec::new(), Vec::new(), Vec::new());
        }
        let f_dim = x.first().map(|r| r.len()).unwrap_or(0);

        let mut edges: Vec<(f64, usize, usize)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if adj[i][j] > 0.0 {
                    edges.push((adj[i][j], i, j));
                }
            }
        }
        edges.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut matched = vec![false; n];
        let mut mapping = vec![usize::MAX; n];
        let mut cluster_id = 0;

        for (_, i, j) in &edges {
            if !matched[*i] && !matched[*j] {
                mapping[*i] = cluster_id;
                mapping[*j] = cluster_id;
                matched[*i] = true;
                matched[*j] = true;
                cluster_id += 1;
            }
        }
        for i in 0..n {
            if !matched[i] {
                mapping[i] = cluster_id;
                cluster_id += 1;
            }
        }

        let m = cluster_id;
        let mut x_coarse = vec![vec![0.0; f_dim]; m];
        let mut counts = vec![0usize; m];
        for i in 0..n {
            let c = mapping[i];
            for d in 0..f_dim {
                x_coarse[c][d] += x[i].get(d).copied().unwrap_or(0.0);
            }
            counts[c] += 1;
        }
        for c in 0..m {
            let cnt = counts[c].max(1) as f64;
            x_coarse[c].iter_mut().for_each(|v| *v /= cnt);
        }

        let mut adj_coarse = vec![vec![0.0; m]; m];
        for i in 0..n {
            for j in 0..n {
                if adj[i][j] > 0.0 {
                    let ci = mapping[i];
                    let cj = mapping[j];
                    if ci != cj {
                        adj_coarse[ci][cj] += adj[i][j];
                        adj_coarse[cj][ci] += adj[i][j];
                    }
                }
            }
        }
        for c1 in 0..m {
            for c2 in 0..m {
                if adj_coarse[c1][c2] > 0.0 {
                    adj_coarse[c1][c2] = adj_coarse[c1][c2].min(1.0);
                }
            }
        }

        (adj_coarse, x_coarse, mapping)
    }
}

/// State at each pooling level for later reconstruction.
#[derive(Debug, Clone)]
pub struct PoolLevel {
    pub selected_indices: Vec<usize>,
    pub n_original: usize,
    pub features: Vec<Vec<f64>>,
}

/// Multi-level hierarchical pooling.
#[derive(Debug, Clone)]
pub struct HierarchicalPool {
    pub levels: usize,
    pub keep_ratio: f64,
}

impl HierarchicalPool {
    pub fn new(levels: usize, keep_ratio: f64) -> Self {
        Self { levels, keep_ratio }
    }

    pub fn forward(
        &self,
        x: &[Vec<f64>],
        adj: &[Vec<f64>],
        seed: u64,
    ) -> (Vec<Vec<f64>>, Vec<PoolLevel>) {
        let mut cur_x = x.to_vec();
        let mut cur_adj = adj.to_vec();
        let mut history: Vec<PoolLevel> = Vec::new();

        for level in 0..self.levels {
            let n = cur_x.len();
            if n <= 1 {
                break;
            }
            let in_f = cur_x.first().map(|r| r.len()).unwrap_or(1);
            let sag = SagPoolLayer::new(in_f, seed + level as u64 * 17);
            let (new_x, sel_idx) = sag.forward(&cur_x, &cur_adj, self.keep_ratio);

            history.push(PoolLevel {
                selected_indices: sel_idx.clone(),
                n_original: n,
                features: cur_x.clone(),
            });

            if new_x.is_empty() {
                break;
            }

            let new_n = sel_idx.len();
            let mut new_adj = vec![vec![0.0; new_n]; new_n];
            for (i, &ni) in sel_idx.iter().enumerate() {
                for (j, &nj) in sel_idx.iter().enumerate() {
                    new_adj[i][j] = cur_adj[ni][nj];
                }
            }
            cur_x = new_x;
            cur_adj = new_adj;
        }

        (cur_x, history)
    }

    pub fn unpool(pooled: &[Vec<f64>], level: &PoolLevel) -> Vec<Vec<f64>> {
        let f = pooled.first().map(|r| r.len()).unwrap_or(0);
        let mut out = vec![vec![0.0; f]; level.n_original];
        for (i, &idx) in level.selected_indices.iter().enumerate() {
            if let Some(pf) = pooled.get(i) {
                out[idx] = pf.clone();
            }
        }
        out
    }
}
