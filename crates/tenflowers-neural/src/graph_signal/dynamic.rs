//! §4 Dynamic Graph Learning.

use super::spectral::{ChebyshevConv, GraphLaplacian};
use super::utils::{degree_vec, dot, rand_weight, sigmoid, vec_norm};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

/// Learn edge weights from node features via MLP([h_i || h_j]).
#[derive(Debug, Clone)]
pub struct EdgeWeightLearner {
    pub w1: Vec<Vec<f64>>,
    pub w2: Vec<f64>,
    pub in_features: usize,
    pub hidden: usize,
}

impl EdgeWeightLearner {
    pub fn new(in_features: usize, hidden: usize, seed: u64) -> Self {
        Self {
            w1: rand_weight(hidden, 2 * in_features, seed),
            w2: {
                let mut rng = StdRng::seed_from_u64(seed + 1);
                (0..hidden).map(|_| rng.random::<f64>() * 0.1).collect()
            },
            in_features,
            hidden,
        }
    }

    fn edge_score(&self, hi: &[f64], hj: &[f64]) -> f64 {
        let cat: Vec<f64> = hi.iter().chain(hj).copied().collect();
        let h1: Vec<f64> = self.w1.iter().map(|row| dot(row, &cat).max(0.0)).collect();
        sigmoid(dot(&h1, &self.w2))
    }

    pub fn forward(&self, x: &[Vec<f64>], edges: &[(usize, usize)]) -> Vec<f64> {
        edges
            .iter()
            .map(|&(i, j)| {
                let hi = x.get(i).map(|v| v.as_slice()).unwrap_or(&[]);
                let hj = x.get(j).map(|v| v.as_slice()).unwrap_or(&[]);
                self.edge_score(hi, hj)
            })
            .collect()
    }
}

/// Learn sparse graph structure via k-NN + cosine similarity.
#[derive(Debug, Clone)]
pub struct GraphStructureLearning {
    pub k: usize,
    pub threshold: f64,
}

impl GraphStructureLearning {
    pub fn new(k: usize, threshold: f64) -> Self {
        Self { k, threshold }
    }

    fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
        let na = vec_norm(a);
        let nb = vec_norm(b);
        if na < 1e-12 || nb < 1e-12 {
            return 0.0;
        }
        dot(a, b) / (na * nb)
    }

    pub fn learn_adj(&self, x: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }
        let k_use = k.min(n.saturating_sub(1));
        let mut adj = vec![vec![0.0; n]; n];

        for i in 0..n {
            let mut sims: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, Self::cosine_sim(&x[i], &x[j])))
                .collect();
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (j, sim) in sims.iter().take(k_use) {
                if *sim >= self.threshold {
                    adj[i][*j] = *sim;
                    adj[*j][i] = *sim;
                }
            }
        }
        adj
    }
}

/// Alternate between learning node features and graph structure.
#[derive(Debug, Clone)]
pub struct IterativeGraphRefinement {
    pub n_iters: usize,
    pub k: usize,
    pub gsl: GraphStructureLearning,
    pub cheb: ChebyshevConv,
}

impl IterativeGraphRefinement {
    pub fn new(in_features: usize, k: usize, n_iters: usize, seed: u64) -> Self {
        Self {
            n_iters,
            k,
            gsl: GraphStructureLearning::new(k, 0.0),
            cheb: ChebyshevConv::new(in_features, in_features, 2, seed),
        }
    }

    pub fn forward(&self, x: &[Vec<f64>], adj_init: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut cur_x = x.to_vec();
        let mut cur_adj = adj_init.to_vec();
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }
        let lambda_max = 2.0;
        let l_tilde = GraphLaplacian::rescale(&GraphLaplacian::compute(&cur_adj), lambda_max);
        let mut l_tilde_cur = l_tilde;

        for _ in 0..self.n_iters {
            cur_x = self.cheb.forward(&cur_x, &l_tilde_cur, 2);
            cur_adj = self.gsl.learn_adj(&cur_x, self.k);
            let lap = GraphLaplacian::compute(&cur_adj);
            l_tilde_cur = GraphLaplacian::rescale(&lap, lambda_max);
        }
        cur_x
    }
}

/// Adaptively combine learned adj + predefined adj.
#[derive(Debug, Clone)]
pub struct AdaptiveGraphConv {
    pub w: Vec<Vec<f64>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl AdaptiveGraphConv {
    pub fn new(in_features: usize, out_features: usize, seed: u64) -> Self {
        Self {
            w: rand_weight(out_features, in_features, seed),
            in_features,
            out_features,
        }
    }

    pub fn forward(
        &self,
        x: &[Vec<f64>],
        adj_learned: &[Vec<f64>],
        adj_given: &[Vec<f64>],
        alpha: f64,
    ) -> Vec<Vec<f64>> {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }
        let adj_mix: Vec<Vec<f64>> = adj_learned
            .iter()
            .zip(adj_given)
            .map(|(rl, rg)| {
                rl.iter()
                    .zip(rg)
                    .map(|(al, ag)| alpha * al + (1.0 - alpha) * ag)
                    .collect()
            })
            .collect();

        let d = degree_vec(&adj_mix);
        let ax: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let f_out = x.first().map(|r| r.len()).unwrap_or(0);
                let mut row = vec![0.0; f_out];
                for j in 0..n {
                    let w = adj_mix[i][j] / d[i];
                    for (rd, xjd) in row.iter_mut().zip(x[j].iter()) {
                        *rd += w * xjd;
                    }
                }
                row
            })
            .collect();

        ax.iter()
            .map(|axi| {
                (0..self.out_features)
                    .map(|o| {
                        self.w
                            .get(o)
                            .map(|row| row.iter().zip(axi).map(|(wi, xi)| wi * xi).sum::<f64>())
                            .unwrap_or(0.0)
                    })
                    .collect()
            })
            .collect()
    }
}

/// Build graph from n-gram co-occurrence in sequences.
#[derive(Debug, Clone)]
pub struct NgramGraphBuilder;

impl NgramGraphBuilder {
    pub fn build(sequences: &[Vec<usize>], n: usize) -> Vec<Vec<f64>> {
        if sequences.is_empty() || n == 0 {
            return Vec::new();
        }
        let vocab_size = sequences
            .iter()
            .flat_map(|s| s.iter())
            .copied()
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        if vocab_size == 0 {
            return Vec::new();
        }

        let mut adj = vec![vec![0.0; vocab_size]; vocab_size];
        for seq in sequences {
            if seq.len() < n {
                continue;
            }
            for window in seq.windows(n) {
                for i in 0..window.len() {
                    for j in (i + 1)..window.len() {
                        let a = window[i];
                        let b = window[j];
                        if a < vocab_size && b < vocab_size {
                            adj[a][b] += 1.0;
                            adj[b][a] += 1.0;
                        }
                    }
                }
            }
        }
        let max_val = adj
            .iter()
            .flat_map(|r| r.iter())
            .cloned()
            .fold(0.0_f64, f64::max);
        if max_val > 0.0 {
            adj.iter_mut()
                .for_each(|row| row.iter_mut().for_each(|v| *v /= max_val));
        }
        adj
    }
}
