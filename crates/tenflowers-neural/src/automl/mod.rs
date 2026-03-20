//! Neural Architecture Search & AutoML (Weight-sharing NAS, HPO, Zero-cost proxies,
//! Automated Feature Engineering, Architecture Encoding).
//! All randomness via `scirs2_core::random`. No `unsafe`. No `unwrap()`.

pub mod extensions;
pub use extensions::*;

pub mod meta_features;
pub use meta_features::*;

#[cfg(test)]
mod tests;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

pub(crate) fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / logits.len() as f64; logits.len()]
    } else {
        exps.iter().map(|e| e / sum).collect()
    }
}

// ── Section 1: Weight-Sharing NAS ─────────────────────────────────────────────

/// Candidate operation for a supernet cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SupernetOp {
    Identity,
    Conv3x3,
    Conv5x5,
    Skip,
}

/// Supernet layer: N candidate operations, each a weight matrix.
#[derive(Debug, Clone)]
pub struct SupernetLayer {
    pub input_dim: usize,
    pub output_dim: usize,
    pub candidates: Vec<SupernetOp>,
    weights_conv3: Vec<Vec<f64>>,
    weights_conv5: Vec<Vec<f64>>,
}

impl SupernetLayer {
    pub fn new(input_dim: usize, output_dim: usize, seed: u64) -> Result<Self> {
        if input_dim == 0 || output_dim == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "SupernetLayer::new".to_string(),
                reason: "dims must be > 0".to_string(),
                context: None,
            });
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (2.0 / input_dim as f64).sqrt();
        let mut make_w = |r: usize, c: usize| -> Vec<Vec<f64>> {
            (0..r)
                .map(|_| {
                    (0..c)
                        .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                        .collect()
                })
                .collect()
        };
        Ok(Self {
            input_dim,
            output_dim,
            candidates: vec![
                SupernetOp::Identity,
                SupernetOp::Conv3x3,
                SupernetOp::Conv5x5,
                SupernetOp::Skip,
            ],
            weights_conv3: make_w(output_dim, input_dim),
            weights_conv5: make_w(output_dim, input_dim),
        })
    }

    pub fn forward(&self, x: &[f64], op_idx: usize) -> Result<Vec<f64>> {
        if x.len() != self.input_dim {
            return Err(TensorError::ShapeMismatch {
                operation: "SupernetLayer::forward".to_string(),
                expected: format!("{}", self.input_dim),
                got: format!("{}", x.len()),
                context: None,
            });
        }
        if op_idx >= self.candidates.len() {
            return Err(TensorError::InvalidArgument {
                operation: "SupernetLayer::forward".to_string(),
                reason: format!("op_idx {} out of range {}", op_idx, self.candidates.len()),
                context: None,
            });
        }
        Ok(match &self.candidates[op_idx] {
            SupernetOp::Identity => {
                let mut v = vec![0.0_f64; self.output_dim];
                let n = x.len().min(self.output_dim);
                v[..n].copy_from_slice(&x[..n]);
                v
            }
            SupernetOp::Conv3x3 => linear_project(x, &self.weights_conv3),
            SupernetOp::Conv5x5 => linear_project(x, &self.weights_conv5),
            SupernetOp::Skip => vec![0.0_f64; self.output_dim],
        })
    }
}

pub(crate) fn linear_project(x: &[f64], w: &[Vec<f64>]) -> Vec<f64> {
    w.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum())
        .collect()
}

/// Single-Path One-Shot supernet: sample one op per layer uniformly.
#[derive(Debug, Clone)]
pub struct SinglePathOneShot {
    pub layers: Vec<SupernetLayer>,
}

impl SinglePathOneShot {
    pub fn new(layers: Vec<SupernetLayer>) -> Self {
        Self { layers }
    }

    pub fn sample_architecture(&self, rng: &mut StdRng) -> Vec<usize> {
        self.layers
            .iter()
            .map(|l| rng.random_range(0..l.candidates.len()))
            .collect()
    }

    pub fn forward_path(&self, x: &[f64], path: &[usize]) -> Result<Vec<f64>> {
        if path.len() != self.layers.len() {
            return Err(TensorError::InvalidArgument {
                operation: "SinglePathOneShot::forward_path".to_string(),
                reason: "path length must match number of layers".to_string(),
                context: None,
            });
        }
        let mut cur = x.to_vec();
        for (layer, &op) in self.layers.iter().zip(path.iter()) {
            cur = layer.forward(&cur, op)?;
        }
        Ok(cur)
    }
}

/// ProxylessNAS: per-layer binary gates (hard or soft).
#[derive(Debug, Clone)]
pub struct ProxylessNas {
    pub layers: Vec<SupernetLayer>,
    pub arch_weights: Vec<Vec<f64>>,
    pub hard_gate: bool,
}

impl ProxylessNas {
    pub fn new(layers: Vec<SupernetLayer>, hard_gate: bool, seed: u64) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        let arch_weights = layers
            .iter()
            .map(|l| {
                (0..l.candidates.len())
                    .map(|_| rng.random::<f64>())
                    .collect()
            })
            .collect();
        Ok(Self {
            layers,
            arch_weights,
            hard_gate,
        })
    }

    pub fn binarize(&self, weights: &[Vec<f64>]) -> Vec<usize> {
        weights
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect()
    }

    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>> {
        let mut cur = x.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            let w = &self.arch_weights[i];
            if self.hard_gate {
                let op = self.binarize(std::slice::from_ref(w))[0];
                cur = layer.forward(&cur, op)?;
            } else {
                let probs = softmax(w);
                let mut out = vec![0.0_f64; layer.output_dim];
                for (op, prob) in probs.iter().enumerate() {
                    let v = layer.forward(&cur, op)?;
                    for (j, vi) in v.iter().enumerate() {
                        out[j] += prob * vi;
                    }
                }
                cur = out;
            }
        }
        Ok(cur)
    }
}

/// DARTS-like NAS: softmax-weighted mixture over candidates.
#[derive(Debug, Clone)]
pub struct GradientBasedNas {
    pub layers: Vec<SupernetLayer>,
    pub alpha: Vec<Vec<f64>>,
}

impl GradientBasedNas {
    pub fn new(layers: Vec<SupernetLayer>, seed: u64) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        let alpha = layers
            .iter()
            .map(|l| {
                (0..l.candidates.len())
                    .map(|_| rng.random::<f64>())
                    .collect()
            })
            .collect();
        Ok(Self { layers, alpha })
    }

    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>> {
        let mut cur = x.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            let probs = softmax(&self.alpha[i]);
            let mut out = vec![0.0_f64; layer.output_dim];
            for (op, &prob) in probs.iter().enumerate() {
                let v = layer.forward(&cur, op)?;
                for (j, vi) in v.iter().enumerate() {
                    out[j] += prob * vi;
                }
            }
            cur = out;
        }
        Ok(cur)
    }
}

/// Decode discrete architecture from continuous weights (argmax or Gumbel-top1).
#[derive(Debug, Clone)]
pub struct ArchitectureDecoder;

impl ArchitectureDecoder {
    pub fn new() -> Self {
        Self
    }

    pub fn argmax_decode(&self, weights: &[Vec<f64>]) -> Vec<usize> {
        weights
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect()
    }

    pub fn gumbel_top1_decode(&self, weights: &[Vec<f64>], rng: &mut StdRng) -> Vec<usize> {
        weights
            .iter()
            .map(|row| {
                let perturbed: Vec<f64> = row
                    .iter()
                    .map(|w| {
                        let u: f64 = rng.random::<f64>().clamp(1e-10, 1.0 - 1e-10);
                        w + (-(-u.ln()).ln())
                    })
                    .collect();
                perturbed
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect()
    }
}

impl Default for ArchitectureDecoder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Section 2: Hyperparameter Optimization ────────────────────────────────────

/// Generic hyperparameter configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub values: Vec<f64>,
}
impl Config {
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }
}

// Minimalist GP: RBF kernel, Cholesky posterior.
#[derive(Debug, Clone)]
struct SimpleGp {
    x_obs: Vec<Vec<f64>>,
    y_obs: Vec<f64>,
    length_scale: f64,
    noise: f64,
    chol: Vec<Vec<f64>>,
    alpha: Vec<f64>,
}

impl SimpleGp {
    fn new(length_scale: f64, noise: f64) -> Self {
        Self {
            x_obs: Vec::new(),
            y_obs: Vec::new(),
            length_scale,
            noise,
            chol: Vec::new(),
            alpha: Vec::new(),
        }
    }

    fn rbf(&self, a: &[f64], b: &[f64]) -> f64 {
        let sq: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum();
        (-sq / (2.0 * self.length_scale.powi(2))).exp()
    }

    fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) {
        let n = x.len();
        self.x_obs = x;
        self.y_obs = y;
        let k: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let v = self.rbf(&self.x_obs[i], &self.x_obs[j]);
                        if i == j {
                            v + self.noise
                        } else {
                            v
                        }
                    })
                    .collect()
            })
            .collect();
        self.chol = cholesky(&k, n);
        let ly = forward_sub(&self.chol, &self.y_obs);
        self.alpha = backward_sub(&self.chol, &ly);
    }

    fn predict(&self, xs: &[f64]) -> (f64, f64) {
        if self.x_obs.is_empty() {
            return (0.0, 1.0);
        }
        let ks: Vec<f64> = self.x_obs.iter().map(|xi| self.rbf(xi, xs)).collect();
        let mean: f64 = ks.iter().zip(self.alpha.iter()).map(|(k, a)| k * a).sum();
        let v = forward_sub(&self.chol, &ks);
        let var = (self.rbf(xs, xs) - v.iter().map(|vi| vi.powi(2)).sum::<f64>()).max(0.0);
        (mean, var)
    }
}

pub(crate) fn cholesky(a: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s: f64 = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            l[i][j] = if i == j {
                s.max(1e-12).sqrt()
            } else if l[j][j].abs() < 1e-15 {
                0.0
            } else {
                s / l[j][j]
            };
        }
    }
    l
}

pub(crate) fn forward_sub(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i][j] * x[j];
        }
        x[i] = if l[i][i].abs() < 1e-15 {
            0.0
        } else {
            s / l[i][i]
        };
    }
    x
}

pub(crate) fn backward_sub(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= l[j][i] * x[j];
        }
        x[i] = if l[i][i].abs() < 1e-15 {
            0.0
        } else {
            s / l[i][i]
        };
    }
    x
}

pub(crate) fn standard_normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

pub(crate) fn standard_normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let p = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - p * (-x * x).exp())
}

pub(crate) fn expected_improvement(mean: f64, std: f64, best_y: f64, xi: f64) -> f64 {
    if std < 1e-10 {
        return 0.0;
    }
    let z = (best_y - mean - xi) / std;
    (best_y - mean - xi) * standard_normal_cdf(z) + std * standard_normal_pdf(z)
}

/// Bayesian Optimizer: GP surrogate + Expected Improvement acquisition.
#[derive(Debug)]
pub struct BayesianOptimizer {
    gp: SimpleGp,
    pub observations_x: Vec<Vec<f64>>,
    pub observations_y: Vec<f64>,
    xi: f64,
    rng: StdRng,
}

impl Clone for BayesianOptimizer {
    fn clone(&self) -> Self {
        Self {
            gp: self.gp.clone(),
            observations_x: self.observations_x.clone(),
            observations_y: self.observations_y.clone(),
            xi: self.xi,
            rng: StdRng::seed_from_u64(0xBA7E_5100),
        }
    }
}

impl BayesianOptimizer {
    pub fn new(length_scale: f64, noise: f64, xi: f64, seed: u64) -> Self {
        Self {
            gp: SimpleGp::new(length_scale, noise),
            observations_x: Vec::new(),
            observations_y: Vec::new(),
            xi,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn observe(&mut self, x: Vec<f64>, y: f64) {
        self.observations_x.push(x);
        self.observations_y.push(y);
        self.gp
            .fit(self.observations_x.clone(), self.observations_y.clone());
    }

    pub fn suggest(&mut self, bounds: &[(f64, f64)]) -> Result<Vec<f64>> {
        if bounds.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "BayesianOptimizer::suggest".to_string(),
                reason: "bounds must not be empty".to_string(),
                context: None,
            });
        }
        let best_y = self
            .observations_y
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let mut best_x = None;
        let mut best_ei = f64::NEG_INFINITY;
        for _ in 0..512_usize {
            let c: Vec<f64> = bounds
                .iter()
                .map(|(lo, hi)| lo + self.rng.random::<f64>() * (hi - lo))
                .collect();
            let (mean, var) = self.gp.predict(&c);
            let ei = if self.observations_y.is_empty() {
                var.sqrt()
            } else {
                expected_improvement(mean, var.sqrt(), best_y, self.xi)
            };
            if ei > best_ei {
                best_ei = ei;
                best_x = Some(c);
            }
        }
        best_x.ok_or_else(|| TensorError::InvalidArgument {
            operation: "BayesianOptimizer::suggest".to_string(),
            reason: "no candidates generated".to_string(),
            context: None,
        })
    }
}

/// Tree-structured Parzen Estimator: model good/bad regions with KDE.
#[derive(Debug, Clone)]
pub struct TpeSampler {
    gamma: f64,
    n_candidates: usize,
    bandwidth: f64,
    seed: u64,
}

impl TpeSampler {
    pub fn new(gamma: f64, seed: u64) -> Self {
        Self {
            gamma: gamma.clamp(0.05, 0.5),
            n_candidates: 256,
            bandwidth: 0.2,
            seed,
        }
    }

    fn kde(samples: &[Vec<f64>], x: &[f64], bw: f64) -> f64 {
        if samples.is_empty() {
            return 1.0;
        }
        let d = x.len() as f64;
        let norm = (2.0 * std::f64::consts::PI).powf(-d / 2.0) / bw.powf(d);
        samples
            .iter()
            .map(|s| {
                let sq: f64 = s
                    .iter()
                    .zip(x.iter())
                    .map(|(si, xi)| (si - xi).powi(2))
                    .sum();
                norm * (-sq / (2.0 * bw * bw)).exp()
            })
            .sum::<f64>()
            / samples.len() as f64
    }

    pub fn sample(&self, history: &[(Vec<f64>, f64)], bounds: &[(f64, f64)]) -> Result<Vec<f64>> {
        if bounds.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "TpeSampler::sample".to_string(),
                reason: "bounds must not be empty".to_string(),
                context: None,
            });
        }
        let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(history.len() as u64));
        if history.is_empty() {
            return Ok(bounds
                .iter()
                .map(|(lo, hi)| lo + rng.random::<f64>() * (hi - lo))
                .collect());
        }
        let mut sorted: Vec<&(Vec<f64>, f64)> = history.iter().collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let n_good = (self.gamma * sorted.len() as f64).ceil() as usize;
        let n_good = n_good.max(1).min(sorted.len() - 1).max(1);
        let good_x: Vec<Vec<f64>> = sorted[..n_good].iter().map(|r| r.0.clone()).collect();
        let bad_x: Vec<Vec<f64>> = sorted[n_good..].iter().map(|r| r.0.clone()).collect();
        let mut best_x: Vec<f64> = bounds
            .iter()
            .map(|(lo, hi)| lo + rng.random::<f64>() * (hi - lo))
            .collect();
        let mut best_ratio = f64::NEG_INFINITY;
        for _ in 0..self.n_candidates {
            let x: Vec<f64> = bounds
                .iter()
                .map(|(lo, hi)| lo + rng.random::<f64>() * (hi - lo))
                .collect();
            let ratio = Self::kde(&good_x, &x, self.bandwidth)
                / Self::kde(&bad_x, &x, self.bandwidth).max(1e-12);
            if ratio > best_ratio {
                best_ratio = ratio;
                best_x = x;
            }
        }
        Ok(best_x)
    }
}

/// One bracket of HyperBand (successive halving).
#[derive(Debug, Clone)]
pub struct HyperBandBracket {
    pub max_iter: usize,
    pub eta: f64,
    pub s: usize,
}

impl HyperBandBracket {
    pub fn new(max_iter: usize, eta: f64, s: usize) -> Self {
        Self { max_iter, eta, s }
    }

    pub fn run_bracket(&self, configs: &[Config], _max_iter: usize) -> Vec<Config> {
        if configs.is_empty() {
            return Vec::new();
        }
        let mut survivors: Vec<&Config> = configs.iter().collect();
        for rung in 0..=self.s {
            let keep = (survivors.len() as f64 / self.eta).ceil() as usize;
            survivors.sort_by(|a, b| {
                let va = a.values.first().cloned().unwrap_or(0.0);
                let vb = b.values.first().cloned().unwrap_or(0.0);
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            });
            if rung < self.s {
                survivors = survivors.into_iter().take(keep.max(1)).collect();
            }
        }
        survivors.iter().map(|c| (*c).clone()).collect()
    }
}

/// HyperBand: manages multiple brackets.
#[derive(Debug, Clone)]
pub struct HyperBand {
    pub max_iter: usize,
    pub eta: f64,
    pub n_brackets: usize,
}

impl HyperBand {
    pub fn new(max_iter: usize, eta: f64, _seed: u64) -> Self {
        let s_max = (max_iter as f64).log(eta).floor() as usize;
        Self {
            max_iter,
            eta,
            n_brackets: s_max + 1,
        }
    }
    pub fn run_bracket_s(&self, s: usize, configs: &[Config]) -> Vec<Config> {
        HyperBandBracket::new(self.max_iter, self.eta, s).run_bracket(configs, self.max_iter)
    }
}

pub(crate) fn dominates(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b.iter()).all(|(ai, bi)| ai <= bi)
        && a.iter().zip(b.iter()).any(|(ai, bi)| ai < bi)
}

pub(crate) fn non_dominated_sort(obj: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let n = obj.len();
    let mut dom_by: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut dom_cnt: Vec<usize> = vec![0; n];
    let mut fronts: Vec<Vec<usize>> = vec![Vec::new()];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if dominates(&obj[i], &obj[j]) {
                dom_by[i].push(j);
            } else if dominates(&obj[j], &obj[i]) {
                dom_cnt[i] += 1;
            }
        }
        if dom_cnt[i] == 0 {
            fronts[0].push(i);
        }
    }
    let mut f = 0;
    while !fronts[f].is_empty() {
        let mut nxt = Vec::new();
        for &i in &fronts[f] {
            for &j in &dom_by[i] {
                dom_cnt[j] = dom_cnt[j].saturating_sub(1);
                if dom_cnt[j] == 0 {
                    nxt.push(j);
                }
            }
        }
        f += 1;
        if nxt.is_empty() {
            break;
        }
        fronts.push(nxt);
    }
    fronts
}

pub(crate) fn crowding_distance(front: &[usize], obj: &[Vec<f64>]) -> Vec<f64> {
    let n = front.len();
    if n == 0 {
        return Vec::new();
    }
    let m = obj[front[0]].len();
    let mut cd = vec![0.0_f64; n];
    for oi in 0..m {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            obj[front[a]][oi]
                .partial_cmp(&obj[front[b]][oi])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        cd[order[0]] = f64::INFINITY;
        cd[order[n - 1]] = f64::INFINITY;
        let range = (obj[front[order[n - 1]]][oi] - obj[front[order[0]]][oi]).max(1e-12);
        for k in 1..(n - 1) {
            cd[order[k]] += (obj[front[order[k + 1]]][oi] - obj[front[order[k - 1]]][oi]) / range;
        }
    }
    cd
}

/// Multi-objective optimizer using NSGA-II selection.
#[derive(Debug)]
pub struct MultiObjectiveOptimizer {
    pub population_size: usize,
    rng: StdRng,
}

impl Clone for MultiObjectiveOptimizer {
    fn clone(&self) -> Self {
        Self {
            population_size: self.population_size,
            rng: StdRng::seed_from_u64(0xB0B1_FACE),
        }
    }
}

impl MultiObjectiveOptimizer {
    pub fn new(population_size: usize, seed: u64) -> Self {
        Self {
            population_size,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn evolve(&mut self, population: &[(Config, Vec<f64>)]) -> Vec<Config> {
        if population.is_empty() {
            return Vec::new();
        }
        let obj: Vec<Vec<f64>> = population.iter().map(|(_, o)| o.clone()).collect();
        let fronts = non_dominated_sort(&obj);
        let mut sel: Vec<usize> = Vec::new();
        for front in &fronts {
            if sel.len() + front.len() <= self.population_size {
                sel.extend_from_slice(front);
            } else {
                let cd = crowding_distance(front, &obj);
                let mut order: Vec<usize> = (0..front.len()).collect();
                order.sort_by(|&a, &b| {
                    cd[b]
                        .partial_cmp(&cd[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let needed = self.population_size - sel.len();
                for &idx in order.iter().take(needed) {
                    sel.push(front[idx]);
                }
                break;
            }
        }
        sel.iter().map(|&i| population[i].0.clone()).collect()
    }

    pub fn pareto_front<'a>(&self, population: &'a [(Config, Vec<f64>)]) -> Vec<&'a Config> {
        if population.is_empty() {
            return Vec::new();
        }
        let obj: Vec<Vec<f64>> = population.iter().map(|(_, o)| o.clone()).collect();
        non_dominated_sort(&obj)
            .first()
            .map(|f| f.iter().map(|&i| &population[i].0).collect())
            .unwrap_or_default()
    }
}

/// Trial metric record for early stopping.
#[derive(Debug, Clone)]
pub struct TrialMetric {
    pub trial_id: usize,
    pub step: usize,
    pub value: f64,
}

/// Median stopping rule: stop trial if its value < median at same step.
#[derive(Debug, Clone)]
pub struct EarlyStoppingRule {
    pub metric_history: Vec<TrialMetric>,
}

impl EarlyStoppingRule {
    pub fn new() -> Self {
        Self {
            metric_history: Vec::new(),
        }
    }

    pub fn record(&mut self, trial_id: usize, step: usize, value: f64) {
        self.metric_history.push(TrialMetric {
            trial_id,
            step,
            value,
        });
    }

    pub fn should_stop(&self, trial_id: usize, step: usize, current: f64) -> bool {
        let vals: Vec<f64> = self
            .metric_history
            .iter()
            .filter(|m| m.trial_id != trial_id && m.step == step)
            .map(|m| m.value)
            .collect();
        if vals.is_empty() {
            return false;
        }
        current < percentile_sorted(&vals, 50.0)
    }
}

impl Default for EarlyStoppingRule {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn percentile_sorted(data: &[f64], p: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut s = data.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s[((p / 100.0) * (s.len() - 1) as f64).round() as usize]
}

// ── Section 3: Zero-Cost Proxies ──────────────────────────────────────────────

/// SynFlow score: sum of absolute weights across all layers.
#[derive(Debug, Clone, Default)]
pub struct SynflowScore;
impl SynflowScore {
    pub fn new() -> Self {
        Self
    }
    pub fn compute(&self, layer_weights: &[Vec<Vec<f64>>]) -> f64 {
        layer_weights
            .iter()
            .flat_map(|l| l.iter().flat_map(|r| r.iter()))
            .map(|w| w.abs())
            .sum()
    }
}

/// GradNorm score: L2 norm of (weight ⊙ gradient) across parameters.
#[derive(Debug, Clone, Default)]
pub struct GradNormScore;
impl GradNormScore {
    pub fn new() -> Self {
        Self
    }
    pub fn compute(&self, weights: &[f64], gradients: &[f64]) -> f64 {
        weights
            .iter()
            .zip(gradients.iter())
            .map(|(w, g)| (w * g).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// NASWOT score: log-determinant of Hamming-kernel matrix over activation patterns.
#[derive(Debug, Clone, Default)]
pub struct NaswotScore;
impl NaswotScore {
    pub fn new() -> Self {
        Self
    }
    pub fn compute(&self, patterns: &[Vec<bool>]) -> f64 {
        let n = patterns.len();
        if n == 0 {
            return 0.0;
        }
        let d = patterns[0].len().max(1);
        let mut k: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let ham = patterns[i]
                    .iter()
                    .zip(patterns[j].iter())
                    .filter(|(a, b)| a != b)
                    .count() as f64;
                k[i][j] = (d as f64 - ham) / d as f64;
                if i == j {
                    k[i][j] += 1e-6;
                }
            }
        }
        let l = cholesky(&k, n);
        l.iter().enumerate().map(|(i, row)| row[i].abs().ln()).sum()
    }
}

/// Zen-NAS score: standard deviation of activation values at random init.
#[derive(Debug, Clone, Default)]
pub struct ZenScore;
impl ZenScore {
    pub fn new() -> Self {
        Self
    }
    pub fn compute(&self, activations: &[Vec<f64>]) -> f64 {
        let all: Vec<f64> = activations.iter().flat_map(|a| a.iter().cloned()).collect();
        let n = all.len() as f64;
        if n < 2.0 {
            return 0.0;
        }
        let mean = all.iter().sum::<f64>() / n;
        (all.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    }
}

/// Jacobian score: log-Frobenius-norm of input-output Jacobian.
#[derive(Debug, Clone, Default)]
pub struct JacobianScore;
impl JacobianScore {
    pub fn new() -> Self {
        Self
    }
    pub fn compute(&self, jacobian: &[Vec<f64>]) -> f64 {
        let frob: f64 = jacobian
            .iter()
            .flat_map(|r| r.iter())
            .map(|v| v.powi(2))
            .sum::<f64>();
        (frob.sqrt() + 1e-12).ln()
    }
}

// ── Section 4: Automated Feature Engineering ──────────────────────────────────

/// Select top-K features by absolute Pearson correlation with target.
#[derive(Debug, Clone)]
pub struct FeatureSelector {
    pub k: usize,
    pub selected: Vec<usize>,
}

impl FeatureSelector {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            selected: Vec::new(),
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<()> {
        if x.is_empty() || y.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "FeatureSelector::fit".to_string(),
                reason: "empty data".to_string(),
                context: None,
            });
        }
        let d = x[0].len();
        if d == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "FeatureSelector::fit".to_string(),
                reason: "no features".to_string(),
                context: None,
            });
        }
        let n = x.len() as f64;
        let ym = y.iter().sum::<f64>() / n;
        let ys = {
            let v = y.iter().map(|yi| (yi - ym).powi(2)).sum::<f64>() / n;
            v.sqrt().max(1e-12)
        };
        let mut scores: Vec<(usize, f64)> = (0..d)
            .map(|j| {
                let col: Vec<f64> = x.iter().map(|r| r[j]).collect();
                let xm = col.iter().sum::<f64>() / n;
                let xs = {
                    let v = col.iter().map(|v| (v - xm).powi(2)).sum::<f64>() / n;
                    v.sqrt().max(1e-12)
                };
                let cov = col
                    .iter()
                    .zip(y.iter())
                    .map(|(xi, yi)| (xi - xm) * (yi - ym))
                    .sum::<f64>()
                    / n;
                (j, (cov / (xs * ys)).abs())
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        self.selected = scores.iter().take(self.k.min(d)).map(|(i, _)| *i).collect();
        self.selected.sort_unstable();
        Ok(())
    }

    pub fn transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if self.selected.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "FeatureSelector::transform".to_string(),
                reason: "not fitted".to_string(),
                context: None,
            });
        }
        Ok(x.iter()
            .map(|row| self.selected.iter().map(|&j| row[j]).collect())
            .collect())
    }
}

/// Degree-2 polynomial feature expansion.
#[derive(Debug, Clone)]
pub struct PolynomialFeatures {
    pub degree: usize,
    pub include_bias: bool,
}

impl PolynomialFeatures {
    pub fn new(degree: usize, include_bias: bool) -> Self {
        Self {
            degree,
            include_bias,
        }
    }

    pub fn transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if x.is_empty() {
            return Ok(Vec::new());
        }
        let d = x[0].len();
        Ok(x.iter()
            .map(|row| {
                let mut out = Vec::new();
                if self.include_bias {
                    out.push(1.0);
                }
                out.extend_from_slice(row);
                if self.degree >= 2 {
                    for i in 0..d {
                        for j in i..d {
                            out.push(row[i] * row[j]);
                        }
                    }
                }
                out
            })
            .collect())
    }
}

/// Detected distribution type.
#[derive(Debug, Clone, PartialEq)]
pub enum DistributionType {
    Normal,
    Uniform,
    HeavyTail,
}

/// Auto-normalizer: detect distribution and apply appropriate scaling.
#[derive(Debug, Clone)]
pub struct AutoNormalizer {
    pub distribution: Option<DistributionType>,
    means: Vec<f64>,
    stds: Vec<f64>,
    mins: Vec<f64>,
    maxs: Vec<f64>,
    medians: Vec<f64>,
    iqrs: Vec<f64>,
}

impl AutoNormalizer {
    pub fn new() -> Self {
        Self {
            distribution: None,
            means: Vec::new(),
            stds: Vec::new(),
            mins: Vec::new(),
            maxs: Vec::new(),
            medians: Vec::new(),
            iqrs: Vec::new(),
        }
    }

    fn detect(col: &[f64]) -> DistributionType {
        let n = col.len() as f64;
        if n < 4.0 {
            return DistributionType::Normal;
        }
        let mean = col.iter().sum::<f64>() / n;
        let var = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        if var < 1e-12 {
            return DistributionType::Uniform;
        }
        let kurt = col
            .iter()
            .map(|v| ((v - mean) / var.sqrt()).powi(4))
            .sum::<f64>()
            / n
            - 3.0;
        if kurt > 1.5 {
            DistributionType::HeavyTail
        } else if kurt < -0.5 {
            DistributionType::Uniform
        } else {
            DistributionType::Normal
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>]) -> Result<()> {
        if x.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "AutoNormalizer::fit".to_string(),
                reason: "empty data".to_string(),
                context: None,
            });
        }
        let d = x[0].len();
        let n = x.len() as f64;
        self.means = vec![0.0; d];
        self.stds = vec![1.0; d];
        self.mins = vec![0.0; d];
        self.maxs = vec![1.0; d];
        self.medians = vec![0.0; d];
        self.iqrs = vec![1.0; d];
        for j in 0..d {
            let col: Vec<f64> = x.iter().map(|r| r[j]).collect();
            self.means[j] = col.iter().sum::<f64>() / n;
            let var = col.iter().map(|v| (v - self.means[j]).powi(2)).sum::<f64>() / n;
            self.stds[j] = var.sqrt().max(1e-12);
            let mut sc = col.clone();
            sc.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.mins[j] = *sc.first().unwrap_or(&0.0);
            self.maxs[j] = *sc.last().unwrap_or(&1.0);
            self.medians[j] = percentile_sorted(&sc, 50.0);
            self.iqrs[j] = (percentile_sorted(&sc, 75.0) - percentile_sorted(&sc, 25.0)).max(1e-12);
            if j == 0 {
                self.distribution = Some(Self::detect(&col));
            }
        }
        Ok(())
    }

    pub fn transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let dist = self
            .distribution
            .as_ref()
            .ok_or_else(|| TensorError::InvalidArgument {
                operation: "AutoNormalizer::transform".to_string(),
                reason: "not fitted".to_string(),
                context: None,
            })?;
        let d = self.means.len();
        Ok(x.iter()
            .map(|row| {
                (0..d.min(row.len()))
                    .map(|j| match dist {
                        DistributionType::Normal => (row[j] - self.means[j]) / self.stds[j],
                        DistributionType::Uniform => {
                            (row[j] - self.mins[j]) / (self.maxs[j] - self.mins[j]).max(1e-12)
                        }
                        DistributionType::HeavyTail => (row[j] - self.medians[j]) / self.iqrs[j],
                    })
                    .collect()
            })
            .collect())
    }
}

impl Default for AutoNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Search for useful pairwise feature interactions by correlation with target.
#[derive(Debug, Clone)]
pub struct FeatureInteractionSearch {
    pub top_k: usize,
    pub selected_pairs: Vec<(usize, usize)>,
}

impl FeatureInteractionSearch {
    pub fn new(top_k: usize) -> Self {
        Self {
            top_k,
            selected_pairs: Vec::new(),
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<()> {
        if x.is_empty() || y.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "FeatureInteractionSearch::fit".to_string(),
                reason: "empty data".to_string(),
                context: None,
            });
        }
        let d = x[0].len();
        let n = x.len() as f64;
        let ym = y.iter().sum::<f64>() / n;
        let ys = {
            let v = y.iter().map(|yi| (yi - ym).powi(2)).sum::<f64>() / n;
            v.sqrt().max(1e-12)
        };
        let mut scores: Vec<((usize, usize), f64)> = Vec::new();
        for i in 0..d {
            for j in i..d {
                let col: Vec<f64> = x.iter().map(|r| r[i] * r[j]).collect();
                let xm = col.iter().sum::<f64>() / n;
                let xs = {
                    let v = col.iter().map(|v| (v - xm).powi(2)).sum::<f64>() / n;
                    v.sqrt().max(1e-12)
                };
                let cov = col
                    .iter()
                    .zip(y.iter())
                    .map(|(xi, yi)| (xi - xm) * (yi - ym))
                    .sum::<f64>()
                    / n;
                scores.push(((i, j), (cov / (xs * ys)).abs()));
            }
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        self.selected_pairs = scores.iter().take(self.top_k).map(|(p, _)| *p).collect();
        Ok(())
    }

    pub fn transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if self.selected_pairs.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "FeatureInteractionSearch::transform".to_string(),
                reason: "not fitted".to_string(),
                context: None,
            });
        }
        Ok(x.iter()
            .map(|row| {
                let mut out = row.clone();
                for &(i, j) in &self.selected_pairs {
                    out.push(
                        row.get(i).cloned().unwrap_or(0.0) * row.get(j).cloned().unwrap_or(0.0),
                    );
                }
                out
            })
            .collect())
    }
}

/// Composite pipeline: FeatureSelector → PolynomialFeatures → AutoNormalizer.
#[derive(Debug, Clone)]
pub struct AutoFeaturePipeline {
    pub selector: FeatureSelector,
    pub poly: PolynomialFeatures,
    pub normalizer: AutoNormalizer,
}

impl AutoFeaturePipeline {
    pub fn new(k_select: usize) -> Self {
        Self {
            selector: FeatureSelector::new(k_select),
            poly: PolynomialFeatures::new(2, false),
            normalizer: AutoNormalizer::new(),
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<()> {
        self.selector.fit(x, y)?;
        let xs = self.selector.transform(x)?;
        let xp = self.poly.transform(&xs)?;
        self.normalizer.fit(&xp)
    }

    pub fn transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let xs = self.selector.transform(x)?;
        let xp = self.poly.transform(&xs)?;
        self.normalizer.transform(&xp)
    }
}

// ── Section 5: Architecture Encoding ─────────────────────────────────────────

/// Graph encoding: adjacency matrix + one-hot op labels.
#[derive(Debug, Clone)]
pub struct GraphEncoding {
    pub n_ops: usize,
    pub max_nodes: usize,
}

impl GraphEncoding {
    pub fn new(n_ops: usize, max_nodes: usize) -> Self {
        Self { n_ops, max_nodes }
    }

    pub fn encode(&self, ops: &[usize], edges: &[(usize, usize)]) -> Vec<f64> {
        let n = self.max_nodes;
        let mut adj = vec![0.0_f64; n * n];
        for &(u, v) in edges {
            if u < n && v < n {
                adj[u * n + v] = 1.0;
            }
        }
        let mut op_hot = vec![0.0_f64; n * self.n_ops];
        for (node, &op) in ops.iter().enumerate().take(n) {
            if op < self.n_ops {
                op_hot[node * self.n_ops + op] = 1.0;
            }
        }
        let mut enc = adj;
        enc.extend_from_slice(&op_hot);
        enc
    }
}

/// Path encoding: binary hash vector of all source→sink DAG paths.
#[derive(Debug, Clone)]
pub struct PathEncoding {
    pub n_ops: usize,
    pub max_nodes: usize,
}

impl PathEncoding {
    pub fn new(n_ops: usize, max_nodes: usize) -> Self {
        Self { n_ops, max_nodes }
    }

    pub fn encode(&self, ops: &[usize], edges: &[(usize, usize)]) -> Vec<f64> {
        let n = self.max_nodes;
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(u, v) in edges {
            if u < n && v < n {
                adj[u].push(v);
            }
        }
        let bucket = self.n_ops.pow(n as u32).max(1);
        let mut enc = vec![0.0_f64; bucket];
        let mut stack: Vec<(usize, Vec<usize>)> = vec![(0, vec![0])];
        while let Some((node, path)) = stack.pop() {
            if node == n - 1 && n > 1 {
                let hash = path.iter().enumerate().fold(0usize, |acc, (i, &s)| {
                    acc + ops.get(s).cloned().unwrap_or(0) * self.n_ops.pow(i as u32)
                });
                enc[hash % bucket] = 1.0;
                continue;
            }
            for &next in &adj[node] {
                if !path.contains(&next) {
                    let mut np = path.clone();
                    np.push(next);
                    stack.push((next, np));
                }
            }
        }
        enc
    }
}

/// Two-hidden-layer MLP predictor: encodes graph → predicted accuracy.
#[derive(Debug, Clone)]
pub struct ArchitecturePredictor {
    input_dim: usize,
    hidden_dim: usize,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
    w3: Vec<f64>,
    b3: f64,
}

impl ArchitecturePredictor {
    pub fn new(input_dim: usize, hidden_dim: usize, seed: u64) -> Result<Self> {
        if input_dim == 0 || hidden_dim == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "ArchitecturePredictor::new".to_string(),
                reason: "dims must be > 0".to_string(),
                context: None,
            });
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let s1 = (2.0 / input_dim as f64).sqrt();
        let s2 = (2.0 / hidden_dim as f64).sqrt();
        let mut mw = |r: usize, c: usize, s: f64| -> Vec<Vec<f64>> {
            (0..r)
                .map(|_| {
                    (0..c)
                        .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * s)
                        .collect()
                })
                .collect()
        };
        Ok(Self {
            input_dim,
            hidden_dim,
            w1: mw(hidden_dim, input_dim, s1),
            b1: vec![0.0; hidden_dim],
            w2: mw(hidden_dim, hidden_dim, s2),
            b2: vec![0.0; hidden_dim],
            w3: (0..hidden_dim)
                .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * s2)
                .collect(),
            b3: 0.0,
        })
    }

    pub fn predict(&self, enc: &[f64]) -> Result<f64> {
        if enc.len() < self.input_dim {
            return Err(TensorError::ShapeMismatch {
                operation: "ArchitecturePredictor::predict".to_string(),
                expected: format!("{}", self.input_dim),
                got: format!("{}", enc.len()),
                context: None,
            });
        }
        let inp = &enc[..self.input_dim];
        let z1: Vec<f64> = self
            .w1
            .iter()
            .zip(self.b1.iter())
            .map(|(row, b)| row.iter().zip(inp.iter()).map(|(w, x)| w * x).sum::<f64>() + b)
            .collect();
        let a1: Vec<f64> = z1.iter().map(|v| v.max(0.0)).collect();
        let z2: Vec<f64> = self
            .w2
            .iter()
            .zip(self.b2.iter())
            .map(|(row, b)| row.iter().zip(a1.iter()).map(|(w, x)| w * x).sum::<f64>() + b)
            .collect();
        let a2: Vec<f64> = z2.iter().map(|v| v.max(0.0)).collect();
        Ok(self
            .w3
            .iter()
            .zip(a2.iter())
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + self.b3)
    }
}

/// Bank of evaluated architecture encodings; query nearest by L2 distance.
#[derive(Debug, Clone, Default)]
pub struct ArchitectureBank {
    pub encodings: Vec<Vec<f64>>,
    pub scores: Vec<f64>,
    pub op_sequences: Vec<Vec<usize>>,
}

impl ArchitectureBank {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, enc: Vec<f64>, ops: Vec<usize>, score: f64) {
        self.encodings.push(enc);
        self.op_sequences.push(ops);
        self.scores.push(score);
    }

    pub fn nearest(&self, query: &[f64]) -> Option<(usize, f64)> {
        if self.encodings.is_empty() {
            return None;
        }
        self.encodings
            .iter()
            .enumerate()
            .map(|(i, enc)| {
                let d: f64 = enc
                    .iter()
                    .zip(query.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (i, d)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }
}

/// Levenshtein edit distance between two operation sequences.
#[derive(Debug, Clone, Default)]
pub struct EditDistance;

impl EditDistance {
    pub fn new() -> Self {
        Self
    }

    pub fn compute(&self, a: &[usize], b: &[usize]) -> usize {
        let (m, n) = (a.len(), b.len());
        let mut dp = vec![vec![0usize; n + 1]; m + 1];
        for i in 0..=m {
            dp[i][0] = i;
        }
        for j in 0..=n {
            dp[0][j] = j;
        }
        for i in 1..=m {
            for j in 1..=n {
                dp[i][j] = if a[i - 1] == b[j - 1] {
                    dp[i - 1][j - 1]
                } else {
                    1 + dp[i - 1][j - 1].min(dp[i - 1][j]).min(dp[i][j - 1])
                };
            }
        }
        dp[m][n]
    }
}
