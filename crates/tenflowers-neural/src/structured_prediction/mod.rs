//! Structured Output Prediction
//!
//! Implements CRF (linear-chain and second-order), Structured SVM,
//! Energy-Based Models, and Belief Propagation for pairwise MRFs.
//!
//! All public items use `Sp` prefix to avoid collisions with
//! `nlp_components::CrfLayer` and `zero_shot_learning::StructuredPrediction`.

pub mod extensions;
pub use extensions::*;

pub mod advanced;
pub use advanced::*;

use std::f64;

// ─────────────────────────────────────────────────────────────────────────────
// §0 Shared utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Tiny linear layer (weights + bias).
#[derive(Debug, Clone)]
pub struct SpLinear {
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
}

impl SpLinear {
    /// Xavier-uniform initialisation: U[-√(6/(fan_in+fan_out)), …]
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let limit = (6.0_f64 / (in_dim + out_dim) as f64).sqrt();
        let mut w = vec![vec![0.0_f64; in_dim]; out_dim];
        // deterministic pseudo-random via LCG for reproducibility without deps
        let mut rng_state: u64 = 0x853c_49e6_748f_ea9b;
        let lcg_next = |s: u64| -> (f64, u64) {
            let ns = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (ns >> 33) as u32;
            let v = (bits as f64) / (u32::MAX as f64) * 2.0 - 1.0;
            (v * limit, ns)
        };
        for row in w.iter_mut() {
            for val in row.iter_mut() {
                let (v, ns) = lcg_next(rng_state);
                rng_state = ns;
                *val = v;
            }
        }
        Self {
            w,
            b: vec![0.0; out_dim],
        }
    }

    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        self.w
            .iter()
            .zip(self.b.iter())
            .map(|(row, &bi)| {
                row.iter()
                    .zip(x.iter())
                    .map(|(wi, xi)| wi * xi)
                    .sum::<f64>()
                    + bi
            })
            .collect()
    }

    pub fn update(&mut self, grad_w: &[Vec<f64>], grad_b: &[f64], lr: f64) {
        for (i, row) in self.w.iter_mut().enumerate() {
            for (j, wij) in row.iter_mut().enumerate() {
                if let Some(gw) = grad_w.get(i).and_then(|r| r.get(j)) {
                    *wij -= lr * gw;
                }
            }
        }
        for (bi, gi) in self.b.iter_mut().zip(grad_b.iter()) {
            *bi -= lr * gi;
        }
    }
}

/// Numerically stable softmax.
pub fn softmax(v: &[f64]) -> Vec<f64> {
    if v.is_empty() {
        return vec![];
    }
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / v.len() as f64; v.len()];
    }
    exps.iter().map(|e| e / sum).collect()
}

/// Numerically stable log-sum-exp.
pub fn log_sum_exp(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    max + v.iter().map(|x| (x - max).exp()).sum::<f64>().ln()
}

// ─────────────────────────────────────────────────────────────────────────────
// §1 Linear-Chain CRF
// ─────────────────────────────────────────────────────────────────────────────

const NEG_INF: f64 = f64::NEG_INFINITY;

/// Configuration for a linear-chain CRF.
#[derive(Debug, Clone)]
pub struct LinearChainCrfConfig {
    pub n_classes: usize,
    pub feature_dim: usize,
}

/// p(y|x) ∝ exp(Σ_t φ(y_t, x_t) + Σ_t ψ(y_t, y_{t-1}) + start + end).
#[derive(Debug, Clone)]
pub struct SpLinearChainCrf {
    pub emission: SpLinear,
    /// transition\[i\]\[j\] = score of y_t=i, y_{t-1}=j
    pub transition: Vec<Vec<f64>>,
    pub start_scores: Vec<f64>,
    pub end_scores: Vec<f64>,
    pub config: LinearChainCrfConfig,
}

impl SpLinearChainCrf {
    pub fn new(config: LinearChainCrfConfig) -> Self {
        let n = config.n_classes;
        let emission = SpLinear::new(config.feature_dim, n);
        let transition = vec![vec![0.0_f64; n]; n];
        let start_scores = vec![0.0_f64; n];
        let end_scores = vec![0.0_f64; n];
        Self {
            emission,
            transition,
            start_scores,
            end_scores,
            config,
        }
    }

    /// Compute emission scores: \[T\]\[n_classes\].
    pub fn emission_scores(&self, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        features.iter().map(|f| self.emission.forward(f)).collect()
    }

    /// Viterbi decoding: argmax label sequence.
    pub fn viterbi(&self, features: &[Vec<f64>]) -> Vec<usize> {
        let t_len = features.len();
        if t_len == 0 {
            return vec![];
        }
        let n = self.config.n_classes;
        let emit = self.emission_scores(features);

        // viterbi[t][c] = best score ending in class c at time t
        let mut viterbi_scores = vec![vec![NEG_INF; n]; t_len];
        let mut backptr = vec![vec![0usize; n]; t_len];

        // t=0: initialise with start_scores + emission
        for c in 0..n {
            viterbi_scores[0][c] = self.start_scores[c] + emit[0][c];
        }

        // t>0: forward max-sum DP
        for t in 1..t_len {
            for c in 0..n {
                let mut best_score = NEG_INF;
                let mut best_prev = 0usize;
                for c_prev in 0..n {
                    let score =
                        viterbi_scores[t - 1][c_prev] + self.transition[c][c_prev] + emit[t][c];
                    if score > best_score {
                        best_score = score;
                        best_prev = c_prev;
                    }
                }
                viterbi_scores[t][c] = best_score;
                backptr[t][c] = best_prev;
            }
        }

        // Add end scores and find best final class
        let mut best_final = 0usize;
        let mut best_final_score = NEG_INF;
        for c in 0..n {
            let score = viterbi_scores[t_len - 1][c] + self.end_scores[c];
            if score > best_final_score {
                best_final_score = score;
                best_final = c;
            }
        }

        // Traceback
        let mut path = vec![0usize; t_len];
        path[t_len - 1] = best_final;
        for t in (1..t_len).rev() {
            path[t - 1] = backptr[t][path[t]];
        }
        path
    }

    /// Forward algorithm: compute log Z (log partition function).
    pub fn forward_algorithm(&self, features: &[Vec<f64>]) -> f64 {
        let t_len = features.len();
        if t_len == 0 {
            return 0.0;
        }
        let n = self.config.n_classes;
        let emit = self.emission_scores(features);

        // alpha[c] = log sum over all sequences ending at c at current time
        let mut alpha = vec![NEG_INF; n];
        for c in 0..n {
            alpha[c] = self.start_scores[c] + emit[0][c];
        }

        for t in 1..t_len {
            let mut alpha_new = vec![NEG_INF; n];
            for c in 0..n {
                let mut vals = Vec::with_capacity(n);
                for c_prev in 0..n {
                    vals.push(alpha[c_prev] + self.transition[c][c_prev] + emit[t][c]);
                }
                alpha_new[c] = log_sum_exp(&vals);
            }
            alpha = alpha_new;
        }

        // Add end scores
        let end_vals: Vec<f64> = (0..n).map(|c| alpha[c] + self.end_scores[c]).collect();
        log_sum_exp(&end_vals)
    }

    /// log p(y|x) = score(y) - log Z
    pub fn log_likelihood(&self, features: &[Vec<f64>], labels: &[usize]) -> f64 {
        if features.is_empty() || labels.is_empty() {
            return 0.0;
        }
        let emit = self.emission_scores(features);
        let t_len = features.len().min(labels.len());

        // Numerator: gold path score
        let mut gold_score = 0.0_f64;
        let l0 = labels[0];
        if l0 < self.config.n_classes {
            gold_score += self.start_scores[l0] + emit[0][l0];
        }
        for t in 1..t_len {
            let lc = labels[t];
            let lp = labels[t - 1];
            if lc < self.config.n_classes && lp < self.config.n_classes {
                gold_score += self.transition[lc][lp] + emit[t][lc];
            }
        }
        let ln_last = labels[t_len - 1];
        if ln_last < self.config.n_classes {
            gold_score += self.end_scores[ln_last];
        }

        let log_z = self.forward_algorithm(features);
        gold_score - log_z
    }

    /// Forward-backward marginals: \[T\]\[n_classes\], p(y_t=c | x).
    pub fn marginals(&self, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let t_len = features.len();
        if t_len == 0 {
            return vec![];
        }
        let n = self.config.n_classes;
        let emit = self.emission_scores(features);

        // Forward pass
        let mut fwd = vec![vec![NEG_INF; n]; t_len];
        for c in 0..n {
            fwd[0][c] = self.start_scores[c] + emit[0][c];
        }
        for t in 1..t_len {
            for c in 0..n {
                let vals: Vec<f64> = (0..n)
                    .map(|cp| fwd[t - 1][cp] + self.transition[c][cp] + emit[t][c])
                    .collect();
                fwd[t][c] = log_sum_exp(&vals);
            }
        }

        // Backward pass
        let mut bwd = vec![vec![NEG_INF; n]; t_len];
        for c in 0..n {
            bwd[t_len - 1][c] = self.end_scores[c];
        }
        for t in (0..t_len - 1).rev() {
            for c in 0..n {
                let vals: Vec<f64> = (0..n)
                    .map(|cn| self.transition[cn][c] + emit[t + 1][cn] + bwd[t + 1][cn])
                    .collect();
                bwd[t][c] = log_sum_exp(&vals);
            }
        }

        // log Z for normalisation
        let log_z_vals: Vec<f64> = (0..n)
            .map(|c| fwd[t_len - 1][c] + self.end_scores[c])
            .collect();
        let log_z = log_sum_exp(&log_z_vals);

        // Beliefs
        let mut beliefs = vec![vec![0.0_f64; n]; t_len];
        for t in 0..t_len {
            let log_b: Vec<f64> = (0..n).map(|c| fwd[t][c] + bwd[t][c] - log_z).collect();
            let max_lb = log_b.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = log_b.iter().map(|v| (v - max_lb).exp()).collect();
            let s: f64 = exps.iter().sum();
            for c in 0..n {
                beliefs[t][c] = if s > 0.0 { exps[c] / s } else { 1.0 / n as f64 };
            }
        }
        beliefs
    }

    /// One gradient step via finite differences; returns -log_likelihood.
    pub fn train_step(&mut self, features: &[Vec<f64>], labels: &[usize], lr: f64) -> f64 {
        let eps = 1e-5_f64;
        let base_ll = self.log_likelihood(features, labels);
        let loss = -base_ll;

        // Gradient w.r.t. emission weights (FD)
        let out_dim = self.config.n_classes;
        let in_dim = self.config.feature_dim;
        let mut grad_w = vec![vec![0.0_f64; in_dim]; out_dim];
        let mut grad_b = vec![0.0_f64; out_dim];

        for i in 0..out_dim {
            for j in 0..in_dim {
                self.emission.w[i][j] += eps;
                let ll_plus = self.log_likelihood(features, labels);
                self.emission.w[i][j] -= eps;
                grad_w[i][j] = -(ll_plus - base_ll) / eps;
            }
            self.emission.b[i] += eps;
            let ll_plus = self.log_likelihood(features, labels);
            self.emission.b[i] -= eps;
            grad_b[i] = -(ll_plus - base_ll) / eps;
        }
        self.emission.update(&grad_w, &grad_b, lr);

        // Gradient w.r.t. transition matrix
        let n = self.config.n_classes;
        for i in 0..n {
            for j in 0..n {
                self.transition[i][j] += eps;
                let ll_plus = self.log_likelihood(features, labels);
                self.transition[i][j] -= eps;
                let g = -(ll_plus - base_ll) / eps;
                self.transition[i][j] -= lr * g;
            }
        }

        loss
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2 Second-Order CRF
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a second-order CRF.
#[derive(Debug, Clone)]
pub struct SecondOrderCrfConfig {
    pub n_classes: usize,
    pub feature_dim: usize,
    pub window_size: usize,
}

/// Second-order CRF with transition triples T2\[c\]\[c'\][c''].
#[derive(Debug, Clone)]
pub struct SecondOrderCrf {
    pub emission: SpLinear,
    /// pairwise\[pair_idx\]\[feature_dim\]: pairwise feature weights
    pub pairwise: Vec<Vec<f64>>,
    /// transition2\[c\]\[c'\][c'']: score for (y_t=c, y_{t-1}=c', y_{t-2}=c'')
    pub transition2: Vec<Vec<Vec<f64>>>,
    pub config: SecondOrderCrfConfig,
}

impl SecondOrderCrf {
    pub fn new(config: SecondOrderCrfConfig) -> Self {
        let n = config.n_classes;
        let emission = SpLinear::new(config.feature_dim, n);
        let n_pairs = n * n;
        let pairwise = vec![vec![0.0_f64; config.feature_dim]; n_pairs];
        // T2[n][n][n]
        let transition2 = vec![vec![vec![0.0_f64; n]; n]; n];
        Self {
            emission,
            pairwise,
            transition2,
            config,
        }
    }

    fn emit(&self, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        features.iter().map(|f| self.emission.forward(f)).collect()
    }

    /// Viterbi with state = (y_t, y_{t-1}) pair.
    pub fn viterbi2(&self, features: &[Vec<f64>]) -> Vec<usize> {
        let t_len = features.len();
        if t_len == 0 {
            return vec![];
        }
        let n = self.config.n_classes;
        let emit = self.emit(features);

        if t_len == 1 {
            // Just pick best emission at t=0
            let mut best = 0usize;
            let mut best_score = f64::NEG_INFINITY;
            for c in 0..n {
                if emit[0][c] > best_score {
                    best_score = emit[0][c];
                    best = c;
                }
            }
            return vec![best];
        }

        // State = (c_cur, c_prev) → index c_cur * n + c_prev
        let n_states = n * n;
        let state_idx = |cc: usize, cp: usize| cc * n + cp;

        // scores[state] at time t
        let mut scores = vec![f64::NEG_INFINITY; n_states];
        let mut backptr = vec![vec![0usize; n_states]; t_len];

        // Init t=1 using t=0 emission
        for c0 in 0..n {
            for c1 in 0..n {
                let s = emit[0][c0] + emit[1][c1] + self.transition2[c1][c0][0]; // c''=0 dummy
                let st = state_idx(c1, c0);
                if s > scores[st] {
                    scores[st] = s;
                }
            }
        }

        // t >= 2
        for t in 2..t_len {
            let mut new_scores = vec![f64::NEG_INFINITY; n_states];
            for c_cur in 0..n {
                for c_prev in 0..n {
                    let st = state_idx(c_cur, c_prev);
                    let mut best_s = f64::NEG_INFINITY;
                    let mut best_from = 0usize;
                    for c_prev2 in 0..n {
                        let from_st = state_idx(c_prev, c_prev2);
                        let s = scores[from_st]
                            + self.transition2[c_cur][c_prev][c_prev2]
                            + emit[t][c_cur];
                        if s > best_s {
                            best_s = s;
                            best_from = from_st;
                        }
                    }
                    new_scores[st] = best_s;
                    backptr[t][st] = best_from;
                }
            }
            scores = new_scores;
        }

        // Find best final state
        let (mut best_state, _) =
            scores
                .iter()
                .enumerate()
                .fold((0, f64::NEG_INFINITY), |(bi, bs), (i, &s)| {
                    if s > bs {
                        (i, s)
                    } else {
                        (bi, bs)
                    }
                });

        let mut path = vec![0usize; t_len];
        // Decode last two positions from best_state
        path[t_len - 1] = best_state / n;
        if t_len >= 2 {
            path[t_len - 2] = best_state % n;
        }

        for t in (2..t_len).rev() {
            best_state = backptr[t][best_state];
            path[t - 2] = best_state % n;
        }

        path
    }

    /// log p(y|x) for a second-order CRF via path score.
    pub fn log_likelihood(&self, features: &[Vec<f64>], labels: &[usize]) -> f64 {
        let t_len = features.len().min(labels.len());
        if t_len == 0 {
            return 0.0;
        }
        let emit = self.emit(features);
        let n = self.config.n_classes;

        let mut gold_score = 0.0_f64;
        for t in 0..t_len {
            let c = labels[t];
            if c < n {
                gold_score += emit[t][c];
            }
            if t >= 2 {
                let c0 = labels[t];
                let c1 = labels[t - 1];
                let c2 = labels[t - 2];
                if c0 < n && c1 < n && c2 < n {
                    gold_score += self.transition2[c0][c1][c2];
                }
            }
        }

        // Rough log Z via brute-force for short sequences (or forward algorithm for 2-order states)
        // Use state = (c_cur, c_prev) forward algorithm
        if t_len < 2 {
            return gold_score; // trivial normalisation = gold score
        }

        let n_states = n * n;
        let state_idx = |cc: usize, cp: usize| cc * n + cp;

        // Init alpha at t=1
        let mut alpha = vec![f64::NEG_INFINITY; n_states];
        for c0 in 0..n {
            for c1 in 0..n {
                let s = emit[0][c0] + emit[1][c1] + self.transition2[c1][c0][0];
                let st = state_idx(c1, c0);
                // log-sum-exp
                let prev = alpha[st];
                alpha[st] = if prev == f64::NEG_INFINITY {
                    s
                } else {
                    let m = prev.max(s);
                    m + ((prev - m).exp() + (s - m).exp()).ln()
                };
            }
        }

        for t in 2..t_len {
            let mut new_alpha = vec![f64::NEG_INFINITY; n_states];
            for c_cur in 0..n {
                for c_prev in 0..n {
                    let st = state_idx(c_cur, c_prev);
                    let mut vals = Vec::with_capacity(n);
                    for c_prev2 in 0..n {
                        let from_st = state_idx(c_prev, c_prev2);
                        vals.push(
                            alpha[from_st]
                                + self.transition2[c_cur][c_prev][c_prev2]
                                + emit[t][c_cur],
                        );
                    }
                    new_alpha[st] = log_sum_exp(&vals);
                }
            }
            alpha = new_alpha;
        }

        let log_z = log_sum_exp(&alpha);
        gold_score - log_z
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3 Structured SVM
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a Structured SVM.
#[derive(Debug, Clone)]
pub struct SsvmConfig {
    pub n_classes: usize,
    pub feature_dim: usize,
    pub c_penalty: f64,
    pub n_epochs: usize,
    pub lr: f64,
}

impl Default for SsvmConfig {
    fn default() -> Self {
        Self {
            n_classes: 2,
            feature_dim: 4,
            c_penalty: 1.0,
            n_epochs: 10,
            lr: 0.01,
        }
    }
}

/// Structured SVM (Tsochantaridis et al. 2005), trained via subgradient.
#[derive(Debug, Clone)]
pub struct StructuredSvm {
    pub w: Vec<f64>,
    pub config: SsvmConfig,
}

/// Trait for structured loss functions.
pub trait StructuredLoss {
    fn loss(&self, y_pred: &[usize], y_true: &[usize]) -> f64;
}

/// Hamming loss: fraction of positions differing.
pub struct HammingLoss;

impl StructuredLoss for HammingLoss {
    fn loss(&self, y_pred: &[usize], y_true: &[usize]) -> f64 {
        let n = y_pred.len().min(y_true.len());
        if n == 0 {
            return 0.0;
        }
        let mismatches = y_pred
            .iter()
            .zip(y_true.iter())
            .filter(|(p, t)| p != t)
            .count();
        mismatches as f64 / n as f64
    }
}

impl StructuredSvm {
    pub fn new(config: SsvmConfig) -> Self {
        let dim = config.n_classes * config.feature_dim;
        Self {
            w: vec![0.0_f64; dim],
            config,
        }
    }

    /// Joint feature map φ(x,y): mean over positions of one_hot(y_t) ⊗ x_t.
    /// Result shape: [n_classes * feature_dim]
    pub fn joint_features(&self, features: &[Vec<f64>], labels: &[usize]) -> Vec<f64> {
        let n = self.config.n_classes;
        let d = self.config.feature_dim;
        let t_len = features.len().min(labels.len());
        if t_len == 0 {
            return vec![0.0; n * d];
        }
        let mut phi = vec![0.0_f64; n * d];
        for t in 0..t_len {
            let c = labels[t];
            if c < n {
                for j in 0..d.min(features[t].len()) {
                    phi[c * d + j] += features[t][j];
                }
            }
        }
        // Mean over T
        for v in phi.iter_mut() {
            *v /= t_len as f64;
        }
        phi
    }

    /// w · φ(x, y)
    pub fn score(&self, joint_feats: &[f64]) -> f64 {
        self.w
            .iter()
            .zip(joint_feats.iter())
            .map(|(wi, phi_i)| wi * phi_i)
            .sum()
    }

    /// Loss-augmented decoding: ŷ = argmax_{y} [w·φ(x,y) + Δ(y, y*)]
    /// Uses position-wise greedy maximisation with Hamming loss augmentation.
    pub fn loss_augmented_decode(
        &self,
        features: &[Vec<f64>],
        y_true: &[usize],
        loss_fn: &dyn StructuredLoss,
    ) -> Vec<usize> {
        let n = self.config.n_classes;
        let d = self.config.feature_dim;
        let t_len = features.len();
        if t_len == 0 {
            return vec![];
        }

        let _ = loss_fn; // trait used for extensibility; currently Hamming decomposes
        let mut best_labels = vec![0usize; t_len];
        for t in 0..t_len {
            let mut best_c = 0usize;
            let mut best_score = f64::NEG_INFINITY;
            for c in 0..n {
                // w · φ contribution from position t with label c
                let mut score_c = 0.0_f64;
                for j in 0..d.min(features[t].len()) {
                    score_c += self.w[c * d + j] * features[t][j] / t_len as f64;
                }
                // Hamming loss augmentation
                let hamming_delta = if t < y_true.len() && c != y_true[t] {
                    1.0 / t_len as f64
                } else {
                    0.0
                };
                score_c += hamming_delta;
                if score_c > best_score {
                    best_score = score_c;
                    best_c = c;
                }
            }
            best_labels[t] = best_c;
        }
        best_labels
    }

    /// Structural hinge loss: max(0, score(ŷ)+Δ(ŷ,y*) - score(y*))
    pub fn hinge_loss(
        &self,
        features: &[Vec<f64>],
        y_true: &[usize],
        loss_fn: &dyn StructuredLoss,
    ) -> f64 {
        let y_hat = self.loss_augmented_decode(features, y_true, loss_fn);
        let phi_hat = self.joint_features(features, &y_hat);
        let phi_true = self.joint_features(features, y_true);
        let delta = loss_fn.loss(&y_hat, y_true);
        let margin = self.score(&phi_hat) + delta - self.score(&phi_true);
        margin.max(0.0)
    }

    /// Subgradient update. Returns hinge loss.
    pub fn train_step(
        &mut self,
        features: &[Vec<f64>],
        y_true: &[usize],
        loss_fn: &dyn StructuredLoss,
    ) -> f64 {
        let y_hat = self.loss_augmented_decode(features, y_true, loss_fn);
        let phi_hat = self.joint_features(features, &y_hat);
        let phi_true = self.joint_features(features, y_true);
        let delta = loss_fn.loss(&y_hat, y_true);
        let score_hat = self.score(&phi_hat) + delta;
        let score_true = self.score(&phi_true);
        let loss = (score_hat - score_true).max(0.0);

        if score_hat > score_true {
            // Subgradient: w -= lr * (phi_hat - phi_true) + lr * lambda * w (L2)
            let lr = self.config.lr;
            let lambda = 1.0 / self.config.c_penalty;
            let dim = self.w.len();
            for i in 0..dim {
                let g = phi_hat.get(i).cloned().unwrap_or(0.0)
                    - phi_true.get(i).cloned().unwrap_or(0.0);
                self.w[i] -= lr * g + lr * lambda * self.w[i];
            }
        }
        loss
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4 Energy-Based Model
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for energy-based structured prediction.
#[derive(Debug, Clone)]
pub struct EnergyNetConfig {
    pub input_dim: usize,
    pub output_dim: usize,
    pub hidden_dim: usize,
    pub n_inference_steps: usize,
    pub inference_lr: f64,
    pub langevin_noise: f64,
}

/// Energy network: E(x, y) → scalar.
/// Architecture: [input_dim + output_dim] → hidden → hidden → 1
#[derive(Debug, Clone)]
pub struct EnergyNetwork {
    pub energy_net: Vec<SpLinear>,
    pub config: EnergyNetConfig,
}

fn relu(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| x.max(0.0)).collect()
}

impl EnergyNetwork {
    pub fn new(config: EnergyNetConfig) -> Self {
        let in_dim = config.input_dim + config.output_dim;
        let h = config.hidden_dim;
        let layers = vec![
            SpLinear::new(in_dim, h),
            SpLinear::new(h, h),
            SpLinear::new(h, 1),
        ];
        Self {
            energy_net: layers,
            config,
        }
    }

    fn forward_layers(&self, inp: &[f64]) -> f64 {
        let mut h = self.energy_net[0].forward(inp);
        h = relu(&h);
        h = self.energy_net[1].forward(&h);
        h = relu(&h);
        let out = self.energy_net[2].forward(&h);
        out.first().cloned().unwrap_or(0.0)
    }

    /// Compute scalar energy E(x, y).
    pub fn energy(&self, x: &[f64], y: &[f64]) -> f64 {
        let mut inp = Vec::with_capacity(x.len() + y.len());
        inp.extend_from_slice(x);
        inp.extend_from_slice(y);
        self.forward_layers(&inp)
    }

    /// Gradient descent on y to minimise E(x, y).
    pub fn predict(&self, x: &[f64], y_init: &[f64]) -> Vec<f64> {
        let eps = 1e-5_f64;
        let steps = self.config.n_inference_steps;
        let lr = self.config.inference_lr;
        let mut y = y_init.to_vec();

        for _ in 0..steps {
            let e0 = self.energy(x, &y);
            // FD gradient w.r.t. y
            let mut grad = vec![0.0_f64; y.len()];
            for i in 0..y.len() {
                y[i] += eps;
                let e1 = self.energy(x, &y);
                y[i] -= eps;
                grad[i] = (e1 - e0) / eps;
            }
            for i in 0..y.len() {
                y[i] -= lr * grad[i];
            }
        }
        y
    }

    /// Contrastive loss: max(0, E(x, y_true) - E(x, y_pred) + margin).
    pub fn contrastive_loss(&self, x: &[f64], y_true: &[f64]) -> f64 {
        let margin = 1.0_f64;
        let y_pred = self.predict(x, y_true);
        let e_true = self.energy(x, y_true);
        let e_pred = self.energy(x, &y_pred);
        (e_true - e_pred + margin).max(0.0)
    }

    /// Train on a batch; returns mean contrastive loss.
    pub fn train_step(&mut self, x_batch: &[Vec<f64>], y_batch: &[Vec<f64>], lr: f64) -> f64 {
        let batch = x_batch.len().min(y_batch.len());
        if batch == 0 {
            return 0.0;
        }
        let eps = 1e-5_f64;
        let mut total_loss = 0.0_f64;

        for b in 0..batch {
            let x = &x_batch[b];
            let y_true = &y_batch[b];
            total_loss += self.contrastive_loss(x, y_true);
        }

        // FD gradient w.r.t. each layer's weights (mean over batch)
        let n_layers = self.energy_net.len();
        for l in 0..n_layers {
            let out_d = self.energy_net[l].w.len();
            let in_d = if out_d > 0 {
                self.energy_net[l].w[0].len()
            } else {
                0
            };
            let mut grad_w = vec![vec![0.0_f64; in_d]; out_d];
            let mut grad_b = vec![0.0_f64; out_d];

            for b in 0..batch {
                let x = &x_batch[b];
                let y_true = &y_batch[b];
                let base = self.contrastive_loss(x, y_true);

                for i in 0..out_d {
                    for j in 0..in_d {
                        self.energy_net[l].w[i][j] += eps;
                        let lp = self.contrastive_loss(x, y_true);
                        self.energy_net[l].w[i][j] -= eps;
                        grad_w[i][j] += (lp - base) / eps / batch as f64;
                    }
                    self.energy_net[l].b[i] += eps;
                    let lp = self.contrastive_loss(x, y_true);
                    self.energy_net[l].b[i] -= eps;
                    grad_b[i] += (lp - base) / eps / batch as f64;
                }
            }

            // Apply update
            for i in 0..out_d {
                for j in 0..in_d {
                    self.energy_net[l].w[i][j] -= lr * grad_w[i][j];
                }
                self.energy_net[l].b[i] -= lr * grad_b[i];
            }
        }

        total_loss / batch as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5 Belief Propagation on Pairwise MRF
// ─────────────────────────────────────────────────────────────────────────────

/// Pairwise MRF factor graph.
#[derive(Debug, Clone)]
pub struct FactorGraph {
    pub n_variables: usize,
    pub n_states: usize,
    pub unary: Vec<Vec<f64>>,
    /// (var_i, var_j, potential\[n_states\]\[n_states\])
    pub pairwise: Vec<(usize, usize, Vec<Vec<f64>>)>,
}

impl FactorGraph {
    pub fn new(n_variables: usize, n_states: usize) -> Self {
        let unary = vec![vec![1.0_f64; n_states]; n_variables];
        Self {
            n_variables,
            n_states,
            unary,
            pairwise: vec![],
        }
    }

    pub fn set_unary(&mut self, var: usize, potentials: Vec<f64>) {
        if var < self.n_variables {
            self.unary[var] = potentials;
        }
    }

    pub fn add_pairwise(&mut self, var_i: usize, var_j: usize, potentials: Vec<Vec<f64>>) {
        self.pairwise.push((var_i, var_j, potentials));
    }
}

/// Loopy belief propagation.
#[derive(Debug, Clone)]
pub struct BeliefPropagation {
    pub n_iterations: usize,
    pub damping: f64,
}

impl BeliefPropagation {
    pub fn new(n_iterations: usize) -> Self {
        Self {
            n_iterations,
            damping: 0.5,
        }
    }

    /// Sum-product BP on pairwise MRF. Returns normalised beliefs \[n_var\]\[n_states\].
    pub fn run(&self, graph: &FactorGraph) -> Vec<Vec<f64>> {
        let nv = graph.n_variables;
        let ns = graph.n_states;
        let ne = graph.pairwise.len();

        if nv == 0 || ns == 0 {
            return vec![];
        }

        // Messages: log_m_ij[e][xj] = log m_{vi→vj}(xj), log_m_ji[e][xi] = log m_{vj→vi}(xi)
        let mut log_m_ij: Vec<Vec<f64>> = vec![vec![0.0; ns]; ne];
        let mut log_m_ji: Vec<Vec<f64>> = vec![vec![0.0; ns]; ne];

        for _iter in 0..self.n_iterations {
            let mut new_log_m_ij = log_m_ij.clone();
            let mut new_log_m_ji = log_m_ji.clone();

            for (e, &(vi, vj, ref pot)) in graph.pairwise.iter().enumerate() {
                let log_unary_i: Vec<f64> = graph.unary[vi]
                    .iter()
                    .map(|&u| u.max(1e-300).ln())
                    .collect();
                let log_unary_j: Vec<f64> = graph.unary[vj]
                    .iter()
                    .map(|&u| u.max(1e-300).ln())
                    .collect();
                let mut log_prod_at_vi = log_unary_i.clone();
                let mut log_prod_at_vj = log_unary_j.clone();

                for (e2, &(vi2, vj2, _)) in graph.pairwise.iter().enumerate() {
                    if e2 == e {
                        continue;
                    }
                    if vj2 == vi {
                        for xi in 0..ns {
                            log_prod_at_vi[xi] += log_m_ij[e2][xi];
                        }
                    }
                    if vi2 == vi {
                        for xi in 0..ns {
                            log_prod_at_vi[xi] += log_m_ji[e2][xi];
                        }
                    }
                    if vj2 == vj {
                        for xj in 0..ns {
                            log_prod_at_vj[xj] += log_m_ij[e2][xj];
                        }
                    }
                    if vi2 == vj {
                        for xj in 0..ns {
                            log_prod_at_vj[xj] += log_m_ji[e2][xj];
                        }
                    }
                }

                // m_{vi→vj}(xj) = LSE_{xi} [ log_pot[xi][xj] + log_prod_at_vi[xi] ]
                for xj in 0..ns {
                    let vals: Vec<f64> = (0..ns)
                        .map(|xi| {
                            let log_pot = pot
                                .get(xi)
                                .and_then(|r| r.get(xj))
                                .cloned()
                                .unwrap_or(0.0_f64)
                                .max(1e-300)
                                .ln();
                            log_pot + log_prod_at_vi[xi]
                        })
                        .collect();
                    let new_val = log_sum_exp(&vals);
                    new_log_m_ij[e][xj] =
                        (1.0 - self.damping) * new_val + self.damping * log_m_ij[e][xj];
                }

                // m_{vj→vi}(xi) = LSE_{xj} [ log_pot[xi][xj] + log_prod_at_vj[xj] ]
                for xi in 0..ns {
                    let vals: Vec<f64> = (0..ns)
                        .map(|xj| {
                            let log_pot = pot
                                .get(xi)
                                .and_then(|r| r.get(xj))
                                .cloned()
                                .unwrap_or(0.0_f64)
                                .max(1e-300)
                                .ln();
                            log_pot + log_prod_at_vj[xj]
                        })
                        .collect();
                    let new_val = log_sum_exp(&vals);
                    new_log_m_ji[e][xi] =
                        (1.0 - self.damping) * new_val + self.damping * log_m_ji[e][xi];
                }
            }

            log_m_ij = new_log_m_ij;
            log_m_ji = new_log_m_ji;
        }

        // Compute beliefs b_i(xi) ∝ ψ_i(xi) * Π_{j∈N(i)} m_{j→i}(xi)
        let mut beliefs = Vec::with_capacity(nv);
        for vi in 0..nv {
            let log_unary: Vec<f64> = graph.unary[vi]
                .iter()
                .map(|&u| u.max(1e-300).ln())
                .collect();
            let mut log_b = log_unary;
            for (e, &(ei, ej, _)) in graph.pairwise.iter().enumerate() {
                if ej == vi {
                    // message from ei→vi
                    for xi in 0..ns {
                        log_b[xi] += log_m_ij[e][xi];
                    }
                }
                if ei == vi {
                    // message from ej→vi
                    for xi in 0..ns {
                        log_b[xi] += log_m_ji[e][xi];
                    }
                }
            }
            // Normalise in log-space
            let lse = log_sum_exp(&log_b);
            let b: Vec<f64> = log_b.iter().map(|lv| (lv - lse).exp()).collect();
            // Ensure sum = 1
            let s: f64 = b.iter().sum();
            let b_norm: Vec<f64> = if s > 0.0 {
                b.iter().map(|v| v / s).collect()
            } else {
                vec![1.0 / ns as f64; ns]
            };
            beliefs.push(b_norm);
        }
        beliefs
    }

    /// Max-product BP → MAP estimate (argmax per variable belief).
    pub fn map_decode(&self, graph: &FactorGraph) -> Vec<usize> {
        let beliefs = self.run(graph);
        beliefs
            .iter()
            .map(|b| {
                b.iter()
                    .enumerate()
                    .fold((0, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
                        if v > bv {
                            (i, v)
                        } else {
                            (bi, bv)
                        }
                    })
                    .0
            })
            .collect()
    }

    /// Bethe free energy approximation.
    /// F_Bethe = -Σ_i H_i + Σ_{ij} I_{ij}
    pub fn bethe_free_energy(&self, graph: &FactorGraph, beliefs: &[Vec<f64>]) -> f64 {
        let nv = graph.n_variables;
        let ns = graph.n_states;
        if beliefs.len() < nv {
            return 0.0;
        }

        // Entropy terms -H_i = Σ_xi b_i log b_i
        let neg_entropy: f64 = beliefs
            .iter()
            .take(nv)
            .map(|b| {
                b.iter()
                    .map(|&bv| if bv > 0.0 { bv * bv.ln() } else { 0.0 })
                    .sum::<f64>()
            })
            .sum();

        // Edge terms I_{ij}: approximate pair beliefs as outer product b_i * b_j
        let edge_sum: f64 = graph
            .pairwise
            .iter()
            .map(|&(vi, vj, ref pot)| {
                let bi = &beliefs[vi];
                let bj = &beliefs[vj];
                let mut sum = 0.0_f64;
                for xi in 0..ns {
                    for xj in 0..ns {
                        let bij = bi[xi] * bj[xj];
                        if bij > 0.0 {
                            let log_pot = pot
                                .get(xi)
                                .and_then(|r| r.get(xj))
                                .cloned()
                                .unwrap_or(1.0_f64)
                                .max(1e-300)
                                .ln();
                            sum -= bij * log_pot;
                            if bi[xi] > 0.0 && bj[xj] > 0.0 {
                                sum += bij * (bij.ln() - bi[xi].ln() - bj[xj].ln());
                            }
                        }
                    }
                }
                sum
            })
            .sum();

        neg_entropy + edge_sum
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6 Sequence Structured Loss Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-entropy over a label sequence.
pub fn sequence_cross_entropy(logits: &[Vec<f64>], labels: &[usize]) -> f64 {
    let t_len = logits.len().min(labels.len());
    if t_len == 0 {
        return 0.0;
    }
    let mut total = 0.0_f64;
    for t in 0..t_len {
        let probs = softmax(&logits[t]);
        let c = labels[t];
        let p = probs.get(c).cloned().unwrap_or(1e-10).max(1e-10);
        total -= p.ln();
    }
    total / t_len as f64
}

/// CTC loss via forward algorithm.
/// `log_probs[T][n_classes+1]` where last class = blank.
/// `target` is the target label sequence (no blanks).
pub fn ctc_loss(log_probs: &[Vec<f64>], target: &[usize]) -> f64 {
    let t_cap = log_probs.len();
    if t_cap == 0 || target.is_empty() {
        return 0.0;
    }

    let blank = if !log_probs[0].is_empty() {
        log_probs[0].len() - 1
    } else {
        0
    };

    // Extend target with blanks: l' = [blank, t0, blank, t1, blank, ..., tN, blank]
    let s_len = 2 * target.len() + 1;
    let mut l_prime = vec![blank; s_len];
    for (i, &t) in target.iter().enumerate() {
        l_prime[2 * i + 1] = t;
    }

    // CTC forward algorithm in log-space
    let neg_inf = f64::NEG_INFINITY;
    let mut alpha = vec![vec![neg_inf; s_len]; t_cap];

    // Initialise t=0
    alpha[0][0] = log_probs[0].get(l_prime[0]).cloned().unwrap_or(neg_inf);
    if s_len > 1 {
        alpha[0][1] = log_probs[0].get(l_prime[1]).cloned().unwrap_or(neg_inf);
    }

    for t in 1..t_cap {
        for s in 0..s_len {
            let lp = log_probs[t].get(l_prime[s]).cloned().unwrap_or(neg_inf);
            let mut acc = alpha[t - 1][s];
            if s > 0 {
                let prev = alpha[t - 1][s - 1];
                acc = lse2(acc, prev);
            }
            if s > 1 && l_prime[s] != l_prime[s - 2] {
                let prev2 = alpha[t - 1][s - 2];
                acc = lse2(acc, prev2);
            }
            alpha[t][s] = if acc == neg_inf { neg_inf } else { acc + lp };
        }
    }

    let s_last = s_len - 1;
    let log_p = if s_last == 0 {
        alpha[t_cap - 1][0]
    } else {
        lse2(alpha[t_cap - 1][s_last], alpha[t_cap - 1][s_last - 1])
    };

    if log_p == neg_inf {
        return 100.0;
    }
    (-log_p).max(0.0)
}

#[inline]
fn lse2(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Label smoothing cross-entropy loss.
pub fn label_smoothing_loss(logits: &[Vec<f64>], labels: &[usize], smoothing: f64) -> f64 {
    let t_len = logits.len().min(labels.len());
    if t_len == 0 {
        return 0.0;
    }
    let mut total = 0.0_f64;
    for t in 0..t_len {
        let log_probs: Vec<f64> = {
            let p = softmax(&logits[t]);
            p.iter().map(|&pv| pv.max(1e-10).ln()).collect()
        };
        let n_classes = log_probs.len();
        if n_classes == 0 {
            continue;
        }
        let c = labels[t];
        let ce_term = log_probs.get(c).cloned().unwrap_or(f64::NEG_INFINITY);

        let h_uniform: f64 = -log_probs.iter().sum::<f64>() / n_classes as f64;

        let loss_t = (1.0 - smoothing) * (-ce_term) + smoothing * h_uniform;
        total += loss_t;
    }
    total / t_len as f64
}

/// ListNet (Cao et al. 2007) ordered-prediction loss: KL(softmax(targets) ‖ softmax(predictions)).
pub fn ordered_prediction_loss(predictions: &[f64], targets: &[f64]) -> f64 {
    if predictions.is_empty() || targets.is_empty() {
        return 0.0;
    }
    let p = softmax(targets);
    let q = softmax(predictions);
    // KL(p || q) = Σ p_i log(p_i / q_i)
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi > 0.0 && qi > 0.0 {
                pi * (pi / qi).ln()
            } else if pi > 0.0 {
                pi * 100.0 // large penalty
            } else {
                0.0
            }
        })
        .sum::<f64>()
        .max(0.0)
}
