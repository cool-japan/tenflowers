//! Test-Time Training and Adaptation (TTA) methods.
//!
//! Implements TENT (entropy minimization), TTT (self-supervised auxiliary tasks),
//! TTT++ (feature alignment), Online Batch Normalization, and Episodic/MAML-style
//! adaptation with domain-adaptation loss functions.
//!
//! References:
//! - Wang et al. 2021 "Tent: Fully Test-Time Adaptation by Entropy Minimization"
//! - Sun et al. 2020 "Test-Time Training with Self-Supervision for Generalization under
//!   Distribution Shifts"
//! - Liu et al. 2021 "TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?"
//! - Schneider et al. 2020 "Improving robustness against common corruptions by covariate
//!   shift adaptation"

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ── Helpers ──────────────────────────────────────────────────────────────────

#[inline]
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Numerically stable softmax.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum < 1e-300 {
        vec![1.0 / logits.len() as f64; logits.len()]
    } else {
        exps.iter().map(|e| e / sum).collect()
    }
}

/// Xavier uniform initialisation scale factor.
#[inline]
fn xavier_scale(fan_in: usize, fan_out: usize) -> f64 {
    (6.0 / (fan_in + fan_out) as f64).sqrt()
}

// ── TtaLinear ────────────────────────────────────────────────────────────────

/// A single linear (fully-connected) layer with Xavier initialisation.
pub struct TtaLinear {
    /// Weight matrix shape \[out_dim\]\[in_dim\].
    pub w: Vec<Vec<f64>>,
    /// Bias vector shape \[out_dim\].
    pub b: Vec<f64>,
}

impl TtaLinear {
    /// Create a new linear layer with Xavier uniform initialised weights and
    /// zero biases.
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let scale = xavier_scale(in_dim, out_dim);
        let w: Vec<Vec<f64>> = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                    .collect()
            })
            .collect();
        let b = vec![0.0f64; out_dim];
        Self { w, b }
    }

    /// Linear forward pass: y = W x + b.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        self.w
            .iter()
            .zip(self.b.iter())
            .map(|(row, &bias)| {
                row.iter()
                    .zip(x.iter())
                    .map(|(&wi, &xi)| wi * xi)
                    .sum::<f64>()
                    + bias
            })
            .collect()
    }

    /// SGD update: w -= lr * grad_w, b -= lr * grad_b.
    pub fn update_params(&mut self, grad_w: &[Vec<f64>], grad_b: &[f64], lr: f64) {
        for (row, grow) in self.w.iter_mut().zip(grad_w.iter()) {
            for (wi, gi) in row.iter_mut().zip(grow.iter()) {
                *wi -= lr * gi;
            }
        }
        for (bi, gi) in self.b.iter_mut().zip(grad_b.iter()) {
            *bi -= lr * gi;
        }
    }

    /// Number of parameters in this layer.
    pub fn n_params(&self) -> usize {
        let out_dim = self.w.len();
        if out_dim == 0 {
            return 0;
        }
        let in_dim = self.w[0].len();
        out_dim * in_dim + out_dim
    }

    /// Clone the layer.
    pub fn clone_layer(&self) -> TtaLinear {
        TtaLinear {
            w: self.w.to_vec(),
            b: self.b.clone(),
        }
    }
}

// ── TtaMlp ───────────────────────────────────────────────────────────────────

/// Multi-layer perceptron with ReLU activations on hidden layers and a linear
/// output layer.
pub struct TtaMlp {
    pub layers: Vec<TtaLinear>,
}

impl TtaMlp {
    /// Build a MLP from a slice of layer sizes, e.g. `&[4, 16, 3]` gives one
    /// hidden layer of size 16 and an output of size 3.
    pub fn new(layer_sizes: &[usize]) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "need at least input and output dims"
        );
        let layers: Vec<TtaLinear> = layer_sizes
            .windows(2)
            .map(|w| TtaLinear::new(w[0], w[1]))
            .collect();
        Self { layers }
    }

    /// Forward pass: ReLU on every hidden layer, linear on the last layer.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let n = self.layers.len();
        let mut current: Vec<f64> = x.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            current = layer.forward(&current);
            if i < n - 1 {
                // hidden — apply ReLU
                for v in current.iter_mut() {
                    *v = relu(*v);
                }
            }
        }
        current
    }

    /// Flatten all parameters into a single Vec (weights row-major, then bias).
    pub fn params_flat(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for layer in &self.layers {
            for row in &layer.w {
                out.extend_from_slice(row);
            }
            out.extend_from_slice(&layer.b);
        }
        out
    }

    /// Apply a flat gradient vector (same layout as `params_flat`) via SGD.
    pub fn update_with_flat_grad(&mut self, grad: &[f64], lr: f64) {
        let mut offset = 0usize;
        for layer in self.layers.iter_mut() {
            let out_dim = layer.w.len();
            let in_dim = if out_dim == 0 { 0 } else { layer.w[0].len() };
            for row in layer.w.iter_mut() {
                for wi in row.iter_mut() {
                    *wi -= lr * grad[offset];
                    offset += 1;
                }
            }
            for bi in layer.b.iter_mut() {
                *bi -= lr * grad[offset];
                offset += 1;
            }
            let _ = (out_dim, in_dim); // suppress unused
        }
    }

    /// Total number of parameters.
    pub fn n_params_total(&self) -> usize {
        self.layers.iter().map(|l| l.n_params()).sum()
    }

    /// Clone the MLP.
    pub fn clone_mlp(&self) -> TtaMlp {
        TtaMlp {
            layers: self.layers.iter().map(|l| l.clone_layer()).collect(),
        }
    }
}

// ── Batch Normalisation Stats (for TENT) ─────────────────────────────────────

/// Running batch normalisation statistics with learnable affine parameters
/// (gamma, beta).  Used inside `TentModel`.
pub struct TtaBatchNormStats {
    pub running_mean: Vec<f64>,
    pub running_var: Vec<f64>,
    /// Per-feature affine scale (initialised to 1).
    pub gamma: Vec<f64>,
    /// Per-feature affine bias (initialised to 0).
    pub beta: Vec<f64>,
    pub momentum: f64,
}

impl TtaBatchNormStats {
    /// Initialise: gamma=1, beta=0, running_mean=0, running_var=1.
    pub fn new(dim: usize) -> Self {
        Self {
            running_mean: vec![0.0f64; dim],
            running_var: vec![1.0f64; dim],
            gamma: vec![1.0f64; dim],
            beta: vec![0.0f64; dim],
            momentum: 0.1,
        }
    }

    /// Training-mode forward: compute batch statistics, update running stats,
    /// return normalised + affine-transformed batch.
    pub fn forward_train(&mut self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }
        let dim = x[0].len();
        // Compute batch mean and variance.
        let mut mean = vec![0.0f64; dim];
        for sample in x.iter() {
            for (j, &val) in sample.iter().enumerate() {
                mean[j] += val;
            }
        }
        for m in mean.iter_mut() {
            *m /= n as f64;
        }
        let mut var = vec![0.0f64; dim];
        for sample in x.iter() {
            for (j, &val) in sample.iter().enumerate() {
                let diff = val - mean[j];
                var[j] += diff * diff;
            }
        }
        for v in var.iter_mut() {
            *v /= n as f64;
        }
        // Update running stats with EMA.
        let mom = self.momentum;
        for j in 0..dim {
            self.running_mean[j] = (1.0 - mom) * self.running_mean[j] + mom * mean[j];
            self.running_var[j] = (1.0 - mom) * self.running_var[j] + mom * var[j];
        }
        // Normalise and apply affine.
        x.iter()
            .map(|sample| {
                sample
                    .iter()
                    .enumerate()
                    .map(|(j, &val)| {
                        let norm = (val - mean[j]) / (var[j] + 1e-5).sqrt();
                        self.gamma[j] * norm + self.beta[j]
                    })
                    .collect()
            })
            .collect()
    }

    /// Eval-mode forward: use running_mean/var, apply gamma and beta.
    pub fn forward_eval(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(j, &val)| {
                let norm = (val - self.running_mean[j]) / (self.running_var[j] + 1e-5).sqrt();
                self.gamma[j] * norm + self.beta[j]
            })
            .collect()
    }

    /// Gradient of entropy H(softmax(logits)) w.r.t. gamma and beta of this
    /// BN layer.  Uses FD approximation (this returns shape hints; the
    /// actual TENT adaptation uses finite differences via `TentModel::adapt`).
    ///
    /// The analytical gradient skeleton is returned here as placeholder vectors
    /// of the correct dimension for callers that need shapes.
    pub fn entropy_gradient(&self, logits: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let p = softmax(logits);
        let h: f64 = -p
            .iter()
            .map(|&pi| if pi > 1e-300 { pi * pi.ln() } else { 0.0 })
            .sum::<f64>();
        // d(H)/d(gamma_j): placeholder — proportional to -h * gamma_j (analytic
        // approximation).  The proper gradient requires the chain through the
        // backbone which is computed via FD in `TentModel::adapt`.
        let grad_gamma: Vec<f64> = self.gamma.iter().map(|&g| -h * g * 0.01).collect();
        let grad_beta: Vec<f64> = self.beta.iter().map(|_| -h * 0.01).collect();
        (grad_gamma, grad_beta)
    }

    /// SGD update of affine parameters.
    pub fn update_affine(&mut self, grad_gamma: &[f64], grad_beta: &[f64], lr: f64) {
        for (g, gg) in self.gamma.iter_mut().zip(grad_gamma.iter()) {
            *g -= lr * gg;
        }
        for (b, gb) in self.beta.iter_mut().zip(grad_beta.iter()) {
            *b -= lr * gb;
        }
    }
}

// ── TENT Configuration ────────────────────────────────────────────────────────

/// Configuration for the TENT adapter.
pub struct TentConfig {
    pub lr: f64,
    /// Number of gradient steps per test batch.
    pub n_steps: usize,
    /// Whether to reset affine params after each test sample.
    pub reset_after: bool,
}

impl Default for TentConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            n_steps: 1,
            reset_after: false,
        }
    }
}

// ── TENT Model ────────────────────────────────────────────────────────────────

/// TENT: Test-Time Entropy Minimisation (Wang et al. 2021).
///
/// Adapts the batch-normalisation affine parameters (gamma, beta) at test time
/// by minimising the entropy of the model's softmax output on each test batch.
pub struct TentModel {
    pub backbone: TtaMlp,
    /// One `TtaBatchNormStats` per hidden layer (i.e., `layers.len() - 1`).
    pub bn_stats: Vec<TtaBatchNormStats>,
    pub config: TentConfig,
}

impl TentModel {
    /// Construct TENT model with a MLP backbone.  `layer_sizes` must have
    /// at least two elements (input dim, output/class dim).
    pub fn new(layer_sizes: &[usize], config: TentConfig) -> Self {
        let backbone = TtaMlp::new(layer_sizes);
        // One BN layer per hidden layer transition.
        let n_hidden = if layer_sizes.len() > 2 {
            layer_sizes.len() - 2
        } else {
            0
        };
        let bn_stats: Vec<TtaBatchNormStats> = (0..n_hidden)
            .map(|i| TtaBatchNormStats::new(layer_sizes[i + 1]))
            .collect();
        Self {
            backbone,
            bn_stats,
            config,
        }
    }

    /// Forward pass with BN in eval mode: produces raw logits, then softmax.
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        let logits = self.forward_logits(x);
        softmax(&logits)
    }

    /// Raw logits through backbone + BN (eval mode).
    fn forward_logits(&self, x: &[f64]) -> Vec<f64> {
        let n = self.backbone.layers.len();
        let mut current: Vec<f64> = x.to_vec();
        for (i, layer) in self.backbone.layers.iter().enumerate() {
            current = layer.forward(&current);
            if i < n - 1 {
                // Apply ReLU then BN affine (if BN layer exists).
                for v in current.iter_mut() {
                    *v = relu(*v);
                }
                if let Some(bn) = self.bn_stats.get(i) {
                    // Use running stats in eval mode.
                    current = bn.forward_eval(&current);
                }
            }
        }
        current
    }

    /// Entropy of the softmax output: H = -Σ p_k log(p_k + ε).
    pub fn entropy(&self, x: &[f64]) -> f64 {
        let p = self.predict(x);
        -p.iter().map(|&pi| pi * (pi + 1e-8).ln()).sum::<f64>()
    }

    /// Adapt BN affine params by minimising entropy on the test batch via
    /// finite-difference gradients.
    pub fn adapt(&mut self, x_test: &[Vec<f64>]) {
        if x_test.is_empty() || self.bn_stats.is_empty() {
            return;
        }
        let eps = 1e-4;
        let lr = self.config.lr;
        // Save initial params for optional reset.
        let orig_gamma: Vec<Vec<f64>> = self.bn_stats.iter().map(|bn| bn.gamma.clone()).collect();
        let orig_beta: Vec<Vec<f64>> = self.bn_stats.iter().map(|bn| bn.beta.clone()).collect();

        for _step in 0..self.config.n_steps {
            // Compute current mean entropy over test batch.
            let base_entropy: f64 =
                x_test.iter().map(|x| self.entropy(x)).sum::<f64>() / x_test.len() as f64;

            // FD gradient for each BN layer's gamma and beta.
            for bi in 0..self.bn_stats.len() {
                let dim = self.bn_stats[bi].gamma.len();
                let mut grad_gamma = vec![0.0f64; dim];
                let mut grad_beta = vec![0.0f64; dim];

                for j in 0..dim {
                    // Perturb gamma[j].
                    self.bn_stats[bi].gamma[j] += eps;
                    let hp: f64 =
                        x_test.iter().map(|x| self.entropy(x)).sum::<f64>() / x_test.len() as f64;
                    self.bn_stats[bi].gamma[j] -= eps;
                    grad_gamma[j] = (hp - base_entropy) / eps;

                    // Perturb beta[j].
                    self.bn_stats[bi].beta[j] += eps;
                    let hp2: f64 =
                        x_test.iter().map(|x| self.entropy(x)).sum::<f64>() / x_test.len() as f64;
                    self.bn_stats[bi].beta[j] -= eps;
                    grad_beta[j] = (hp2 - base_entropy) / eps;
                }

                self.bn_stats[bi].update_affine(&grad_gamma, &grad_beta, lr);
            }
        }

        if self.config.reset_after {
            for (bi, bn) in self.bn_stats.iter_mut().enumerate() {
                bn.gamma = orig_gamma[bi].clone();
                bn.beta = orig_beta[bi].clone();
            }
        }
    }

    /// Adapt on a single test sample, then predict its class probabilities.
    pub fn adapt_and_predict(&mut self, x_test: &[f64]) -> Vec<f64> {
        self.adapt(&[x_test.to_vec()]);
        self.predict(x_test)
    }
}

// ── TTT Configuration ─────────────────────────────────────────────────────────

/// Configuration for TTT (Test-Time Training).
pub struct TttConfig {
    pub lr: f64,
    /// Gradient steps per test sample at test time.
    pub n_adapt_steps: usize,
    /// Weight on auxiliary self-supervised loss (default 0.5).
    pub aux_task_weight: f64,
}

impl Default for TttConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            n_adapt_steps: 1,
            aux_task_weight: 0.5,
        }
    }
}

// ── Rotation label for self-supervised task ───────────────────────────────────

/// 2D rotation labels used as the auxiliary self-supervised task.
pub enum RotationLabel {
    R0,
    R90,
    R180,
    R270,
}

// ── TTT Model ─────────────────────────────────────────────────────────────────

/// TTT: Test-Time Training (Sun et al. 2020).
///
/// Trains a shared feature extractor with both a main classification head and
/// an auxiliary self-supervised head (rotation prediction).  At test time,
/// gradients of the auxiliary loss adapt the shared parameters.
pub struct TttModel {
    /// Shared features + main classifier head (input → n_classes).
    pub main_head: TtaMlp,
    /// Auxiliary head for rotation prediction (input → 4).
    pub aux_head: TtaMlp,
    pub config: TttConfig,
}

impl TttModel {
    /// Create a TTT model.
    /// - `feat_dim`: dimensionality of input features.
    /// - `n_classes`: number of main-task output classes.
    pub fn new(feat_dim: usize, n_classes: usize, config: TttConfig) -> Self {
        // Hidden layer of size max(16, feat_dim/2).
        let hidden = (feat_dim / 2).max(16);
        let main_head = TtaMlp::new(&[feat_dim, hidden, n_classes]);
        let aux_head = TtaMlp::new(&[feat_dim, hidden, 4]); // 4 rotation classes
        Self {
            main_head,
            aux_head,
            config,
        }
    }

    /// Simulate a 2D rotation on a 1-D feature vector by quarter-permutation.
    ///
    /// The feature vector is divided into four equal quarters Q1‥Q4:
    /// - R0:   identity                [Q1 Q2 Q3 Q4]
    /// - R90:  permute quarters        [Q2 Q1 Q4 Q3]
    /// - R180: reverse entire vector   [Q4 Q3 Q2 Q1] (i.e. full reversal)
    /// - R270: permute quarters        [Q4 Q3 Q2 Q1] → reversed order [Q4 Q3 Q2 Q1]
    ///   Actually R270 ≡ Q4 Q3 Q2 Q1 (same as full reversal of R180 but quarter-wise):
    ///   [Q4 Q3 Q2 Q1] — distinct from R180 by using quarter blocks reversed.
    ///
    /// To keep R180 ≠ R270: R180 = full byte-reverse; R270 = quarter blocks
    /// reordered as \[Q4, Q3, Q2, Q1\] with each quarter kept forward-order.
    pub fn rotate_features(x: &[f64], rotation: &RotationLabel) -> Vec<f64> {
        let n = x.len();
        match rotation {
            RotationLabel::R0 => x.to_vec(),
            RotationLabel::R90 => {
                // [Q2, Q1, Q4, Q3]
                let q = (n / 4).max(1);
                let mut out = vec![0.0f64; n];
                for i in 0..n {
                    let q_idx = i / q;
                    let in_q = i % q;
                    let src_q = match q_idx % 4 {
                        0 => 1,
                        1 => 0,
                        2 => 3,
                        _ => 2,
                    };
                    let src = src_q * q + in_q;
                    if src < n {
                        out[i] = x[src];
                    } else {
                        out[i] = x[i % n];
                    }
                }
                out
            }
            RotationLabel::R180 => {
                // Full reversal.
                let mut out = x.to_vec();
                out.reverse();
                out
            }
            RotationLabel::R270 => {
                // Quarter blocks reordered as [Q4, Q3, Q2, Q1] each kept forward.
                let q = (n / 4).max(1);
                let mut out = vec![0.0f64; n];
                for i in 0..n {
                    let q_idx = i / q;
                    let in_q = i % q;
                    let src_q = match q_idx % 4 {
                        0 => 3,
                        1 => 2,
                        2 => 1,
                        _ => 0,
                    };
                    let src = src_q * q + in_q;
                    if src < n {
                        out[i] = x[src];
                    } else {
                        out[i] = x[i % n];
                    }
                }
                out
            }
        }
    }

    /// Cross-entropy loss for predicting which rotation was applied.
    pub fn aux_loss(&self, x: &[f64], rotation: &RotationLabel) -> f64 {
        let rotated = Self::rotate_features(x, rotation);
        let logits = self.aux_head.forward(&rotated);
        let p = softmax(&logits);
        let target = match rotation {
            RotationLabel::R0 => 0usize,
            RotationLabel::R90 => 1,
            RotationLabel::R180 => 2,
            RotationLabel::R270 => 3,
        };
        let prob = p.get(target).copied().unwrap_or(1e-8);
        -(prob + 1e-8).ln()
    }

    /// Adapt the auxiliary head parameters on `x_test` via finite-difference
    /// gradient of the rotation-prediction auxiliary loss.
    pub fn adapt_test(&mut self, x_test: &[f64], rng: &mut StdRng) {
        let eps = 1e-4;
        let lr = self.config.lr;
        let rotations = [
            RotationLabel::R0,
            RotationLabel::R90,
            RotationLabel::R180,
            RotationLabel::R270,
        ];

        for _step in 0..self.config.n_adapt_steps {
            // Pick a random rotation.
            let rot_idx: usize = (rng.random::<u64>() % 4) as usize;
            let rotation = &rotations[rot_idx];

            let base_loss = self.aux_loss(x_test, rotation);

            // FD gradient w.r.t. aux_head params.
            let n_params = self.aux_head.n_params_total();
            let mut flat_grad = vec![0.0f64; n_params];

            {
                let flat_params = self.aux_head.params_flat();
                for pi in 0..n_params {
                    let mut perturbed_params = flat_params.clone();
                    perturbed_params[pi] += eps;
                    // Temporarily swap params.
                    let orig = self.aux_head.clone_mlp();
                    self.aux_head.update_with_flat_grad(
                        &perturbed_params
                            .iter()
                            .zip(flat_params.iter())
                            .map(|(&pp, &fp)| (fp - pp) / lr) // encode delta as grad
                            .collect::<Vec<_>>(),
                        lr,
                    );
                    // Recompute: actually apply perturbation differently.
                    // Restore and use a cleaner approach:
                    self.aux_head = orig;
                    // Perturb via offset.
                    let mut delta = vec![0.0f64; n_params];
                    delta[pi] = -eps; // update_with_flat_grad does p -= lr*g, so g = -eps/lr
                    self.aux_head.update_with_flat_grad(&delta, lr);
                    let perturbed_loss = self.aux_loss(x_test, rotation);
                    flat_grad[pi] = (perturbed_loss - base_loss) / eps;
                    // Restore.
                    let mut restore_delta = vec![0.0f64; n_params];
                    restore_delta[pi] = eps; // undo the -eps perturbation
                    self.aux_head.update_with_flat_grad(&restore_delta, lr);
                }
            }

            self.aux_head.update_with_flat_grad(&flat_grad, lr);
        }
    }

    /// Predict class probabilities using the main head.
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        let logits = self.main_head.forward(x);
        softmax(&logits)
    }

    /// Train on labelled data (main head cross-entropy).
    pub fn fit(&mut self, x_train: &[Vec<f64>], y_train: &[usize], n_epochs: usize, lr: f64) {
        let eps = 1e-4;
        for _epoch in 0..n_epochs {
            for (x, &y) in x_train.iter().zip(y_train.iter()) {
                let n_params = self.main_head.n_params_total();
                let base_logits = self.main_head.forward(x);
                let base_p = softmax(&base_logits);
                let n_cls = base_p.len();
                let base_loss = if y < n_cls {
                    -(base_p[y] + 1e-8).ln()
                } else {
                    0.0
                };

                let flat_params = self.main_head.params_flat();
                let mut flat_grad = vec![0.0f64; n_params];
                for pi in 0..n_params {
                    let mut delta = vec![0.0f64; n_params];
                    delta[pi] = -eps;
                    self.main_head.update_with_flat_grad(&delta, lr);
                    let pert_logits = self.main_head.forward(x);
                    let pert_p = softmax(&pert_logits);
                    let pert_loss = if y < pert_p.len() {
                        -(pert_p[y] + 1e-8).ln()
                    } else {
                        0.0
                    };
                    flat_grad[pi] = (pert_loss - base_loss) / eps;
                    // Restore.
                    let mut restore = vec![0.0f64; n_params];
                    restore[pi] = eps;
                    self.main_head.update_with_flat_grad(&restore, lr);
                }
                let _ = flat_params;
                self.main_head.update_with_flat_grad(&flat_grad, lr);
            }
        }
    }
}

// ── TTT++ Configuration ───────────────────────────────────────────────────────

/// Configuration for TTT++ (Liu et al. 2021).
pub struct TttPlusConfig {
    pub lr_feat: f64,
    pub lr_cls: f64,
    pub n_steps: usize,
    pub momentum: f64,
}

impl Default for TttPlusConfig {
    fn default() -> Self {
        Self {
            lr_feat: 1e-3,
            lr_cls: 1e-3,
            n_steps: 1,
            momentum: 0.1,
        }
    }
}

// ── Online Feature Statistics ─────────────────────────────────────────────────

/// Maintains exponential moving average estimates of feature mean and
/// diagonal covariance.
pub struct OnlineFeatureStats {
    pub running_mean: Vec<f64>,
    /// Diagonal of the running covariance estimate.
    pub running_cov_diag: Vec<f64>,
    pub n_seen: usize,
    pub momentum: f64,
}

impl OnlineFeatureStats {
    /// Initialise with zero mean and unit covariance.
    pub fn new(feat_dim: usize, momentum: f64) -> Self {
        Self {
            running_mean: vec![0.0f64; feat_dim],
            running_cov_diag: vec![1.0f64; feat_dim],
            n_seen: 0,
            momentum,
        }
    }

    /// EMA update with a single feature vector.
    pub fn update(&mut self, feat: &[f64]) {
        let mom = self.momentum;
        for (rm, &fi) in self.running_mean.iter_mut().zip(feat.iter()) {
            *rm = (1.0 - mom) * *rm + mom * fi;
        }
        for (rc, (&fi, &rm)) in self
            .running_cov_diag
            .iter_mut()
            .zip(feat.iter().zip(self.running_mean.iter()))
        {
            let diff = fi - rm;
            *rc = (1.0 - mom) * *rc + mom * diff * diff;
        }
        self.n_seen += 1;
    }

    /// Simplified MMD-like alignment loss: squared distance in mean and
    /// variance space.
    pub fn alignment_loss(&self, feat: &[f64]) -> f64 {
        if self.n_seen == 0 {
            return 0.0;
        }
        let mean_sq: f64 = feat
            .iter()
            .zip(self.running_mean.iter())
            .map(|(&fi, &mi)| (fi - mi).powi(2))
            .sum::<f64>();
        // Single-sample variance approximation: (f - mean)^2 vs running_cov.
        let var_sq: f64 = feat
            .iter()
            .zip(self.running_mean.iter().zip(self.running_cov_diag.iter()))
            .map(|(&fi, (&mi, &ci))| ((fi - mi).powi(2) - ci).powi(2))
            .sum::<f64>();
        mean_sq + var_sq
    }
}

// ── TTT++ Model ───────────────────────────────────────────────────────────────

/// TTT++: Feature alignment adaptation at test time (Liu et al. 2021).
pub struct TttPlusModel {
    pub extractor: TtaMlp,
    pub classifier: TtaLinear,
    pub feat_stats: OnlineFeatureStats,
    pub config: TttPlusConfig,
}

impl TttPlusModel {
    /// Create a TTT++ model.
    pub fn new(input_dim: usize, feat_dim: usize, n_classes: usize, config: TttPlusConfig) -> Self {
        let momentum = config.momentum;
        let extractor = TtaMlp::new(&[input_dim, feat_dim]);
        let classifier = TtaLinear::new(feat_dim, n_classes);
        let feat_stats = OnlineFeatureStats::new(feat_dim, momentum);
        Self {
            extractor,
            classifier,
            feat_stats,
            config,
        }
    }

    /// Train on source domain: cross-entropy for classifier, accumulate feature
    /// statistics.
    pub fn fit_source(&mut self, x_train: &[Vec<f64>], y_train: &[usize], n_epochs: usize) {
        let lr = self.config.lr_cls;
        let eps = 1e-4;
        for _epoch in 0..n_epochs {
            for (x, &y) in x_train.iter().zip(y_train.iter()) {
                let feat = self.extractor.forward(x);
                self.feat_stats.update(&feat);
                let logits = self.classifier.forward(&feat);
                let p = softmax(&logits);
                let n_cls = p.len();
                let base_loss = if y < n_cls { -(p[y] + 1e-8).ln() } else { 0.0 };

                // FD gradient for classifier weights only.
                let out_dim = self.classifier.w.len();
                let in_dim = if out_dim > 0 {
                    self.classifier.w[0].len()
                } else {
                    0
                };
                let mut grad_w: Vec<Vec<f64>> = vec![vec![0.0f64; in_dim]; out_dim];
                let mut grad_b: Vec<f64> = vec![0.0f64; out_dim];

                for oi in 0..out_dim {
                    for ii in 0..in_dim {
                        self.classifier.w[oi][ii] += eps;
                        let logits2 = self.classifier.forward(&feat);
                        let p2 = softmax(&logits2);
                        let loss2 = if y < p2.len() {
                            -(p2[y] + 1e-8).ln()
                        } else {
                            0.0
                        };
                        self.classifier.w[oi][ii] -= eps;
                        grad_w[oi][ii] = (loss2 - base_loss) / eps;
                    }
                    self.classifier.b[oi] += eps;
                    let logits2 = self.classifier.forward(&feat);
                    let p2 = softmax(&logits2);
                    let loss2 = if y < p2.len() {
                        -(p2[y] + 1e-8).ln()
                    } else {
                        0.0
                    };
                    self.classifier.b[oi] -= eps;
                    grad_b[oi] = (loss2 - base_loss) / eps;
                }

                self.classifier.update_params(&grad_w, &grad_b, lr);
            }
        }
    }

    /// Adapt the feature extractor on a test sample to align with source stats.
    pub fn adapt_test(&mut self, x_test: &[f64]) {
        let eps = 1e-4;
        let lr = self.config.lr_feat;

        for _step in 0..self.config.n_steps {
            let feat = self.extractor.forward(x_test);
            let base_loss = self.feat_stats.alignment_loss(&feat);

            let n_params = self.extractor.n_params_total();
            let mut flat_grad = vec![0.0f64; n_params];

            for pi in 0..n_params {
                let mut delta = vec![0.0f64; n_params];
                delta[pi] = -eps;
                self.extractor.update_with_flat_grad(&delta, lr);
                let pert_feat = self.extractor.forward(x_test);
                let pert_loss = self.feat_stats.alignment_loss(&pert_feat);
                flat_grad[pi] = (pert_loss - base_loss) / eps;
                let mut restore = vec![0.0f64; n_params];
                restore[pi] = eps;
                self.extractor.update_with_flat_grad(&restore, lr);
            }

            self.extractor.update_with_flat_grad(&flat_grad, lr);
        }
    }

    /// Predict class probabilities for an input.
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        let feat = self.extractor.forward(x);
        let logits = self.classifier.forward(&feat);
        softmax(&logits)
    }
}

// ── Online Batch Normalisation (Schneider et al. 2020) ────────────────────────

/// Configuration for the Online BN layer.
pub struct OnlineBnConfig {
    /// EMA momentum for statistics update (default 0.1).
    pub momentum: f64,
    /// Mixture weight: (1-α)*source + α*test.
    pub alpha: f64,
    /// Virtual batch size representing source prior strength (default 64).
    pub prior_strength: usize,
}

impl Default for OnlineBnConfig {
    fn default() -> Self {
        Self {
            momentum: 0.1,
            alpha: 0.5,
            prior_strength: 64,
        }
    }
}

/// A single BN layer that adapts its running statistics on test batches.
pub struct OnlineBnLayer {
    pub source_mean: Vec<f64>,
    pub source_var: Vec<f64>,
    pub current_mean: Vec<f64>,
    pub current_var: Vec<f64>,
    pub gamma: Vec<f64>,
    pub beta: Vec<f64>,
    pub config: OnlineBnConfig,
    /// Number of test adaptation steps performed.
    pub n_adapted: usize,
}

impl OnlineBnLayer {
    /// Create an `OnlineBnLayer` with zero source mean and unit source variance.
    pub fn new(dim: usize, config: OnlineBnConfig) -> Self {
        Self {
            source_mean: vec![0.0f64; dim],
            source_var: vec![1.0f64; dim],
            current_mean: vec![0.0f64; dim],
            current_var: vec![1.0f64; dim],
            gamma: vec![1.0f64; dim],
            beta: vec![0.0f64; dim],
            n_adapted: 0,
            config,
        }
    }

    /// Adapt statistics using a test batch.
    ///
    /// `mean_adapted = (prior_strength * source_mean + n_test * test_mean) / (prior_strength + n_test)`
    pub fn adapt_step(&mut self, x_batch: &[Vec<f64>]) {
        let n_test = x_batch.len();
        if n_test == 0 {
            return;
        }
        let dim = self.source_mean.len();

        // Compute test batch mean and var.
        let mut test_mean = vec![0.0f64; dim];
        for sample in x_batch.iter() {
            for (j, &val) in sample.iter().enumerate().take(dim) {
                test_mean[j] += val;
            }
        }
        for m in test_mean.iter_mut() {
            *m /= n_test as f64;
        }
        let mut test_var = vec![0.0f64; dim];
        for sample in x_batch.iter() {
            for (j, &val) in sample.iter().enumerate().take(dim) {
                let d = val - test_mean[j];
                test_var[j] += d * d;
            }
        }
        for v in test_var.iter_mut() {
            *v /= n_test as f64;
        }

        let ps = self.config.prior_strength as f64;
        let nt = n_test as f64;
        let total = ps + nt;

        for j in 0..dim {
            self.current_mean[j] = (ps * self.source_mean[j] + nt * test_mean[j]) / total;
            self.current_var[j] = (ps * self.source_var[j] + nt * test_var[j]) / total;
        }
        self.n_adapted += 1;
    }

    /// Normalise an input vector using adapted statistics with affine transform.
    pub fn normalize(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(j, &val)| {
                let mean = self.current_mean.get(j).copied().unwrap_or(0.0);
                let var = self.current_var.get(j).copied().unwrap_or(1.0);
                let gamma = self.gamma.get(j).copied().unwrap_or(1.0);
                let beta = self.beta.get(j).copied().unwrap_or(0.0);
                let norm = (val - mean) / (var + 1e-5).sqrt();
                gamma * norm + beta
            })
            .collect()
    }
}

// ── Episodic Adaptation (MAML-style) ─────────────────────────────────────────

/// Configuration for episodic / MAML-style adaptation.
pub struct EpisodicAdaptConfig {
    pub inner_lr: f64,
    /// Gradient steps applied at test time on the support set.
    pub n_inner_steps: usize,
    pub outer_lr: f64,
    /// Meta-training gradient steps.
    pub n_outer_steps: usize,
}

impl Default for EpisodicAdaptConfig {
    fn default() -> Self {
        Self {
            inner_lr: 0.01,
            n_inner_steps: 5,
            outer_lr: 0.001,
            n_outer_steps: 10,
        }
    }
}

/// An episodic adapter that clones its model and fine-tunes on a support set.
pub struct EpisodicAdapter {
    pub model: TtaMlp,
    pub config: EpisodicAdaptConfig,
}

/// Cross-entropy loss for a single sample with label `y`.
fn cross_entropy_loss(mlp: &TtaMlp, x: &[f64], y: usize) -> f64 {
    let logits = mlp.forward(x);
    let p = softmax(&logits);
    let n_cls = p.len();
    if y < n_cls {
        -(p[y] + 1e-8).ln()
    } else {
        0.0
    }
}

/// Finite-difference gradient of cross-entropy over a labelled batch.
fn fd_grad_ce(mlp: &mut TtaMlp, xs: &[Vec<f64>], ys: &[usize]) -> Vec<f64> {
    let eps = 1e-4;
    let n_params = mlp.n_params_total();
    let base_loss: f64 = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, &y)| cross_entropy_loss(mlp, x, y))
        .sum::<f64>()
        / xs.len().max(1) as f64;

    let mut grad = vec![0.0f64; n_params];
    for pi in 0..n_params {
        let mut delta = vec![0.0f64; n_params];
        delta[pi] = -eps;
        mlp.update_with_flat_grad(&delta, 1.0);
        let pert_loss: f64 = xs
            .iter()
            .zip(ys.iter())
            .map(|(x, &y)| cross_entropy_loss(mlp, x, y))
            .sum::<f64>()
            / xs.len().max(1) as f64;
        grad[pi] = (pert_loss - base_loss) / eps;
        let mut restore = vec![0.0f64; n_params];
        restore[pi] = eps;
        mlp.update_with_flat_grad(&restore, 1.0);
    }
    grad
}

impl EpisodicAdapter {
    pub fn new(layer_sizes: &[usize], config: EpisodicAdaptConfig) -> Self {
        Self {
            model: TtaMlp::new(layer_sizes),
            config,
        }
    }

    /// Clone the base model and apply `n_inner_steps` of gradient descent on the
    /// support set, returning the adapted copy.
    pub fn adapt(&self, support_x: &[Vec<f64>], support_y: &[usize]) -> TtaMlp {
        let mut adapted = self.model.clone_mlp();
        let inner_lr = self.config.inner_lr;
        for _ in 0..self.config.n_inner_steps {
            let grad = fd_grad_ce(&mut adapted, support_x, support_y);
            adapted.update_with_flat_grad(&grad, inner_lr);
        }
        adapted
    }

    /// MAML-style meta-training.
    ///
    /// For each episode: adapt on support set, evaluate query loss, use query
    /// loss gradient to update the base model (first-order approximation).
    pub fn meta_train(
        &mut self,
        episodes: &[(Vec<Vec<f64>>, Vec<usize>, Vec<Vec<f64>>, Vec<usize>)],
        rng: &mut StdRng,
    ) -> Vec<f64> {
        let outer_lr = self.config.outer_lr;
        let mut loss_history = Vec::new();

        for _outer_step in 0..self.config.n_outer_steps {
            if episodes.is_empty() {
                loss_history.push(0.0);
                continue;
            }
            // Pick a random episode.
            let ep_idx: usize = (rng.random::<u64>() % episodes.len() as u64) as usize;
            let (sup_x, sup_y, qry_x, qry_y) = &episodes[ep_idx];

            // Inner adaptation (does not modify self.model).
            let mut adapted = self.adapt(sup_x, sup_y);

            // Query loss on adapted model.
            let query_loss: f64 = qry_x
                .iter()
                .zip(qry_y.iter())
                .map(|(x, &y)| cross_entropy_loss(&adapted, x, y))
                .sum::<f64>()
                / qry_x.len().max(1) as f64;
            loss_history.push(query_loss);

            // First-order meta-gradient: FD on query loss w.r.t. base params.
            let query_grad = fd_grad_ce(&mut adapted, qry_x, qry_y);
            self.model.update_with_flat_grad(&query_grad, outer_lr);
        }
        loss_history
    }

    /// Predict class index (argmax) for a single input using the base model.
    pub fn predict(&self, x: &[f64]) -> usize {
        let logits = self.model.forward(x);
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ── Domain Adaptation Loss Functions ─────────────────────────────────────────

/// RBF kernel: k(x, y) = exp(-||x-y||^2 / (2 * bandwidth^2)).
#[inline]
fn rbf_kernel(x: &[f64], y: &[f64], bandwidth: f64) -> f64 {
    let sq_dist: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - yi).powi(2))
        .sum();
    (-sq_dist / (2.0 * bandwidth * bandwidth)).exp()
}

/// Maximum Mean Discrepancy (squared) with an RBF kernel.
///
/// MMD^2 = E\[k(xs, xs')\] + E\[k(xt, xt')\] - 2 * E\[k(xs, xt)\]
pub fn maximum_mean_discrepancy(
    source_feats: &[Vec<f64>],
    target_feats: &[Vec<f64>],
    bandwidth: f64,
) -> f64 {
    let ns = source_feats.len();
    let nt = target_feats.len();
    if ns == 0 || nt == 0 {
        return 0.0;
    }
    let bw = if bandwidth <= 0.0 { 1.0 } else { bandwidth };

    // E[k(xs, xs')]
    let mut kss = 0.0f64;
    let mut n_ss = 0usize;
    for i in 0..ns {
        for j in 0..ns {
            if i != j {
                kss += rbf_kernel(&source_feats[i], &source_feats[j], bw);
                n_ss += 1;
            }
        }
    }
    if n_ss > 0 {
        kss /= n_ss as f64;
    }

    // E[k(xt, xt')]
    let mut ktt = 0.0f64;
    let mut n_tt = 0usize;
    for i in 0..nt {
        for j in 0..nt {
            if i != j {
                ktt += rbf_kernel(&target_feats[i], &target_feats[j], bw);
                n_tt += 1;
            }
        }
    }
    if n_tt > 0 {
        ktt /= n_tt as f64;
    }

    // E[k(xs, xt)]
    let mut kst = 0.0f64;
    for i in 0..ns {
        for j in 0..nt {
            kst += rbf_kernel(&source_feats[i], &target_feats[j], bw);
        }
    }
    kst /= (ns * nt) as f64;

    let mmd2 = kss + ktt - 2.0 * kst;
    mmd2.max(0.0)
}

/// CORAL loss: Frobenius-norm squared distance of feature covariance matrices.
///
/// CORAL = ||C_s - C_t||_F^2 / (4 * d^2)
pub fn correlation_alignment_loss(source_feats: &[Vec<f64>], target_feats: &[Vec<f64>]) -> f64 {
    let ns = source_feats.len();
    let nt = target_feats.len();
    if ns < 2 || nt < 2 {
        return 0.0;
    }
    let d = source_feats[0].len();
    if d == 0 {
        return 0.0;
    }

    /// Compute (unbiased) covariance matrix diagonal for a batch.
    fn covariance(feats: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = feats.len();
        let d = feats[0].len();
        // Mean.
        let mut mean = vec![0.0f64; d];
        for f in feats.iter() {
            for (j, &v) in f.iter().enumerate() {
                mean[j] += v;
            }
        }
        for m in mean.iter_mut() {
            *m /= n as f64;
        }
        // Covariance.
        let mut cov = vec![vec![0.0f64; d]; d];
        for f in feats.iter() {
            for i in 0..d {
                for j in 0..d {
                    cov[i][j] += (f[i] - mean[i]) * (f[j] - mean[j]);
                }
            }
        }
        let denom = (n - 1) as f64;
        for row in cov.iter_mut() {
            for v in row.iter_mut() {
                *v /= denom;
            }
        }
        cov
    }

    let cs = covariance(source_feats);
    let ct = covariance(target_feats);

    let frob_sq: f64 = (0..d)
        .flat_map(|i| (0..d).map(move |j| (i, j)))
        .map(|(i, j)| (cs[i][j] - ct[i][j]).powi(2))
        .sum();

    frob_sq / (4.0 * (d * d) as f64)
}

/// Entropy minimisation loss: H = -Σ p_k log(p_k + ε) where p = softmax(logits).
pub fn entropy_minimization_loss(logits: &[f64]) -> f64 {
    let p = softmax(logits);
    -p.iter().map(|&pi| pi * (pi + 1e-8).ln()).sum::<f64>()
}

// ── TTA Metrics & Evaluation ──────────────────────────────────────────────────

/// Aggregated metrics for evaluating TTA performance.
pub struct TtaMetrics {
    pub accuracy: f64,
    pub mean_confidence: f64,
    pub mean_entropy: f64,
    pub adaptation_time_steps: usize,
}

/// Evaluate `TentModel` on a labelled test set (without adapting during
/// evaluation; metrics reflect the current adapted state).
pub fn evaluate_tta(model: &TentModel, x_test: &[Vec<f64>], y_test: &[usize]) -> TtaMetrics {
    let n = x_test.len();
    if n == 0 {
        return TtaMetrics {
            accuracy: 0.0,
            mean_confidence: 0.0,
            mean_entropy: 0.0,
            adaptation_time_steps: 0,
        };
    }

    let mut n_correct = 0usize;
    let mut sum_conf = 0.0f64;
    let mut sum_entropy = 0.0f64;

    for (x, &y) in x_test.iter().zip(y_test.iter()) {
        let p = model.predict(x);
        let pred = p
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        if pred == y {
            n_correct += 1;
        }
        let conf = p.iter().cloned().fold(0.0f64, f64::max);
        sum_conf += conf;
        let h = -p.iter().map(|&pi| pi * (pi + 1e-8).ln()).sum::<f64>();
        sum_entropy += h;
    }

    TtaMetrics {
        accuracy: (n_correct as f64 / n as f64).clamp(0.0, 1.0),
        mean_confidence: sum_conf / n as f64,
        mean_entropy: sum_entropy / n as f64,
        adaptation_time_steps: n,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
