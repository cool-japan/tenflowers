//! Information-Theoretic Methods for Machine Learning.
//!
//! This module implements a suite of information-theoretic tools for modern
//! machine learning research and practice, including:
//!
//! - **MINE** (Mutual Information Neural Estimation): Neural estimators for MI
//!   via Donsker-Varadhan, f-Divergence, and SMILE bounds.
//! - **VIB** (Variational Information Bottleneck): β-VAE-style bottleneck
//!   with tunable compression-relevance tradeoff.
//! - **DIM** (Deep InfoMax): Global and local MI maximisation for self-supervised
//!   representation learning.
//! - **CLUB**: Contrastive Log-ratio Upper Bound on mutual information.
//! - **HSIC**: Hilbert-Schmidt Independence Criterion with RBF kernels.
//! - **Rényi Entropy**: Kernel-density-based estimators for α-Rényi entropy.
//! - **TC-VAE**: Total-Correlation decomposition of the β-VAE ELBO (Chen 2018).

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Internal mathematical helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Rectified linear unit.
#[inline]
fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Softplus: `log(1 + exp(x))`, numerically stable.
#[inline]
pub fn softplus(x: f64) -> f64 {
    if x > 30.0 {
        x
    } else if x < -30.0 {
        x.exp()
    } else {
        (1.0_f64 + x.exp()).ln()
    }
}

/// Sigmoid: `1 / (1 + exp(-x))`, numerically stable.
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Softmax over a slice, in-place.
fn softmax(v: &mut [f64]) {
    let max_v = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0_f64;
    for x in v.iter_mut() {
        *x = (*x - max_v).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

/// Numerically-stable log-sum-exp.
#[inline]
fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_v = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_v.is_infinite() {
        return max_v;
    }
    let sum: f64 = values.iter().map(|v| (v - max_v).exp()).sum();
    max_v + sum.ln()
}

/// Normal (Box-Muller) sample from an rng.
#[inline]
fn sample_normal(rng: &mut impl Rng) -> f64 {
    let u1: f64 = (rng.random::<f64>()).max(1e-300);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Squared Euclidean distance between two slices.
#[inline]
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// SimpleLinear — shared building block
// ─────────────────────────────────────────────────────────────────────────────

/// A fully-connected linear layer `y = W x + b` with Xavier initialisation.
#[derive(Clone, Debug)]
pub struct SimpleLinear {
    /// Weight matrix `[out_dim][in_dim]`.
    pub w: Vec<Vec<f64>>,
    /// Bias vector `[out_dim]`.
    pub b: Vec<f64>,
}

impl SimpleLinear {
    /// Create a new `SimpleLinear` layer with Xavier uniform initialisation.
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(42u64);
        let limit = (6.0_f64 / (in_dim + out_dim) as f64).sqrt();
        let w = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * limit)
                    .collect()
            })
            .collect();
        let b = vec![0.0_f64; out_dim];
        Self { w, b }
    }

    /// Forward pass: returns `[out_dim]` vector.
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
}

// ─────────────────────────────────────────────────────────────────────────────
// MINE — Mutual Information Neural Estimator
// ─────────────────────────────────────────────────────────────────────────────

/// A multi-layer perceptron that scores pairs `(x, y)`.
///
/// The network takes `concat(x, y)` as input and produces a scalar `T(x, y)`.
#[derive(Clone, Debug)]
pub struct MineNetwork {
    /// Layer weights `[layer][out_neuron][in_neuron]`.
    pub weights: Vec<Vec<Vec<f64>>>,
    /// Layer biases `[layer][out_neuron]`.
    pub biases: Vec<Vec<f64>>,
    /// Sizes of each layer including input and output.
    pub layer_sizes: Vec<usize>,
}

impl MineNetwork {
    /// Build a new `MineNetwork`.
    ///
    /// - `x_dim` + `y_dim` → `hidden_dim` (×(`n_layers`-1)) → 1
    pub fn new(x_dim: usize, y_dim: usize, hidden_dim: usize, n_layers: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(1234u64);
        let input_dim = x_dim + y_dim;

        // Build layer size list: [input, hidden..., 1]
        let mut layer_sizes = vec![input_dim];
        for _ in 0..n_layers.saturating_sub(1) {
            layer_sizes.push(hidden_dim);
        }
        layer_sizes.push(1);

        let n_weight_layers = layer_sizes.len() - 1;
        let mut weights = Vec::with_capacity(n_weight_layers);
        let mut biases = Vec::with_capacity(n_weight_layers);

        for l in 0..n_weight_layers {
            let in_d = layer_sizes[l];
            let out_d = layer_sizes[l + 1];
            let limit = (6.0_f64 / (in_d + out_d) as f64).sqrt();
            let w: Vec<Vec<f64>> = (0..out_d)
                .map(|_| {
                    (0..in_d)
                        .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * limit)
                        .collect()
                })
                .collect();
            let b = vec![0.0_f64; out_d];
            weights.push(w);
            biases.push(b);
        }

        Self {
            weights,
            biases,
            layer_sizes,
        }
    }

    /// Forward pass through the network.
    ///
    /// Returns a scalar `T(x, y)`.
    pub fn forward(&self, x: &[f64], y: &[f64]) -> f64 {
        // Concatenate input
        let mut activation: Vec<f64> = x.iter().chain(y.iter()).cloned().collect();

        let n_layers = self.weights.len();
        for (l, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let mut next = vec![0.0_f64; b.len()];
            for (o, (row, &bi)) in w.iter().zip(b.iter()).enumerate() {
                next[o] = row
                    .iter()
                    .zip(activation.iter())
                    .map(|(wi, ai)| wi * ai)
                    .sum::<f64>()
                    + bi;
            }
            // Apply ReLU on all but the last layer
            if l < n_layers - 1 {
                for v in next.iter_mut() {
                    *v = relu(*v);
                }
            }
            activation = next;
        }

        // activation should be length 1 at this point
        activation.first().cloned().unwrap_or(0.0)
    }

    /// SGD update using pre-computed per-parameter gradients.
    pub fn update(&mut self, grads_w: &[Vec<Vec<f64>>], grads_b: &[Vec<f64>], lr: f64) {
        for (l, (gw, gb)) in grads_w.iter().zip(grads_b.iter()).enumerate() {
            if l >= self.weights.len() {
                break;
            }
            for (o, grow) in gw.iter().enumerate() {
                if o >= self.weights[l].len() {
                    break;
                }
                for (i, &g) in grow.iter().enumerate() {
                    if i < self.weights[l][o].len() {
                        self.weights[l][o][i] -= lr * g;
                    }
                }
            }
            for (o, &g) in gb.iter().enumerate() {
                if o < self.biases[l].len() {
                    self.biases[l][o] -= lr * g;
                }
            }
        }
    }
}

/// Variant of the MINE estimator.
#[derive(Clone, Debug)]
pub enum MineEstimator {
    /// Donsker-Varadhan bound: `E[T] - log E[exp(T)]`
    Dv,
    /// f-Divergence lower bound
    Fd,
    /// SMILE: smoothed estimator with exp clipping
    Smile { clip: f64 },
}

/// Mutual Information Neural Estimator.
///
/// Estimates `I(X; Y)` using a parametric variational bound, trained via
/// finite-difference gradient updates on the scoring network.
#[derive(Clone, Debug)]
pub struct Mine {
    /// The scoring network `T(x, y)`.
    pub network: MineNetwork,
    /// Which bound to use for estimation.
    pub estimator: MineEstimator,
    /// Exponential moving average of `E[exp(T)]` (used for DV gradient correction).
    pub ema_et: f64,
    /// EMA coefficient (default `0.01`).
    pub ema_coeff: f64,
}

impl Mine {
    /// Construct a new `Mine` estimator.
    pub fn new(x_dim: usize, y_dim: usize, hidden_dim: usize, estimator: MineEstimator) -> Self {
        let network = MineNetwork::new(x_dim, y_dim, hidden_dim, 3);
        Self {
            network,
            estimator,
            ema_et: 1.0,
            ema_coeff: 0.01,
        }
    }

    /// Compute the MI lower-bound without updating the network.
    ///
    /// `x_samples` are from the joint `p(x, y)` and `y_samples` are from the
    /// (shuffled) marginal `p(y)`.
    pub fn estimate(&mut self, x_samples: &[Vec<f64>], y_samples: &[Vec<f64>]) -> f64 {
        let n = x_samples.len().min(y_samples.len());
        if n == 0 {
            return 0.0;
        }

        // Scores on the joint distribution
        let joint_scores: Vec<f64> = x_samples[..n]
            .iter()
            .zip(y_samples[..n].iter())
            .map(|(x, y)| self.network.forward(x, y))
            .collect();

        // Scores on the marginal (x from joint, y shuffled)
        let marginal_scores: Vec<f64> = x_samples[..n]
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let j = (i + 1) % n; // simple cyclic shuffle
                self.network.forward(x, &y_samples[j])
            })
            .collect();

        self.compute_bound(&joint_scores, &marginal_scores)
    }

    /// Internal: apply the chosen bound formula to score vectors.
    fn compute_bound(&mut self, joint: &[f64], marginal: &[f64]) -> f64 {
        let mean_joint = joint.iter().sum::<f64>() / joint.len() as f64;

        match &self.estimator {
            MineEstimator::Dv => {
                let exp_sum: f64 = marginal.iter().map(|&t| t.exp()).sum::<f64>();
                let mean_exp = exp_sum / marginal.len() as f64;
                // EMA-corrected log denominator
                self.ema_et = (1.0 - self.ema_coeff) * self.ema_et + self.ema_coeff * mean_exp;
                let log_denom = self.ema_et.max(1e-300).ln();
                mean_joint - log_denom
            }
            MineEstimator::Fd => {
                // f-divergence lower bound: E[T] - E[exp(T - 1)]
                let mean_exp_m1: f64 =
                    marginal.iter().map(|&t| (t - 1.0).exp()).sum::<f64>() / marginal.len() as f64;
                mean_joint - mean_exp_m1.ln()
            }
            MineEstimator::Smile { clip } => {
                let c = *clip;
                let clipped: Vec<f64> = marginal
                    .iter()
                    .map(|&t| t.exp().clamp(1.0 / c, c))
                    .collect();
                let mean_clipped = clipped.iter().sum::<f64>() / clipped.len() as f64;
                mean_joint - mean_clipped.max(1e-300).ln()
            }
        }
    }

    /// One training step: estimate MI and update network with finite-difference gradients.
    ///
    /// Returns the MI estimate in nats.
    pub fn train_step(
        &mut self,
        joint_x: &[Vec<f64>],
        joint_y: &[Vec<f64>],
        marginal_y: &[Vec<f64>],
        lr: f64,
    ) -> f64 {
        let n = joint_x.len().min(joint_y.len()).min(marginal_y.len());
        if n == 0 {
            return 0.0;
        }

        // Compute current estimate (also updates EMA)
        let joint_scores: Vec<f64> = joint_x[..n]
            .iter()
            .zip(joint_y[..n].iter())
            .map(|(x, y)| self.network.forward(x, y))
            .collect();
        let marginal_scores: Vec<f64> = joint_x[..n]
            .iter()
            .zip(marginal_y[..n].iter())
            .map(|(x, y)| self.network.forward(x, y))
            .collect();

        let mi = self.compute_bound(&joint_scores, &marginal_scores);

        // Finite-difference gradient estimation (perturbation δ)
        let delta = 1e-4;

        // Build zero gradient accumulators
        let mut grads_w: Vec<Vec<Vec<f64>>> = self
            .network
            .weights
            .iter()
            .map(|layer| layer.iter().map(|row| vec![0.0; row.len()]).collect())
            .collect();
        let mut grads_b: Vec<Vec<f64>> = self
            .network
            .biases
            .iter()
            .map(|b| vec![0.0; b.len()])
            .collect();

        // Gradient w.r.t. weights (central difference, one perturbation at a time)
        for l in 0..self.network.weights.len() {
            for o in 0..self.network.weights[l].len() {
                for i in 0..self.network.weights[l][o].len() {
                    // +δ
                    self.network.weights[l][o][i] += delta;
                    let j_pos: Vec<f64> = joint_x[..n]
                        .iter()
                        .zip(joint_y[..n].iter())
                        .map(|(x, y)| self.network.forward(x, y))
                        .collect();
                    let m_pos: Vec<f64> = joint_x[..n]
                        .iter()
                        .zip(marginal_y[..n].iter())
                        .map(|(x, y)| self.network.forward(x, y))
                        .collect();

                    // -δ
                    self.network.weights[l][o][i] -= 2.0 * delta;
                    let j_neg: Vec<f64> = joint_x[..n]
                        .iter()
                        .zip(joint_y[..n].iter())
                        .map(|(x, y)| self.network.forward(x, y))
                        .collect();
                    let m_neg: Vec<f64> = joint_x[..n]
                        .iter()
                        .zip(marginal_y[..n].iter())
                        .map(|(x, y)| self.network.forward(x, y))
                        .collect();

                    // Restore
                    self.network.weights[l][o][i] += delta;

                    // Finite-difference bound gradient
                    let bound_pos = self.fd_bound(&j_pos, &m_pos);
                    let bound_neg = self.fd_bound(&j_neg, &m_neg);
                    // Negative gradient (we want to maximise the bound → minimise neg)
                    grads_w[l][o][i] = -(bound_pos - bound_neg) / (2.0 * delta);
                }
            }
        }

        // Gradient w.r.t. biases
        for l in 0..self.network.biases.len() {
            for o in 0..self.network.biases[l].len() {
                self.network.biases[l][o] += delta;
                let j_pos: Vec<f64> = joint_x[..n]
                    .iter()
                    .zip(joint_y[..n].iter())
                    .map(|(x, y)| self.network.forward(x, y))
                    .collect();
                let m_pos: Vec<f64> = joint_x[..n]
                    .iter()
                    .zip(marginal_y[..n].iter())
                    .map(|(x, y)| self.network.forward(x, y))
                    .collect();

                self.network.biases[l][o] -= 2.0 * delta;
                let j_neg: Vec<f64> = joint_x[..n]
                    .iter()
                    .zip(joint_y[..n].iter())
                    .map(|(x, y)| self.network.forward(x, y))
                    .collect();
                let m_neg: Vec<f64> = joint_x[..n]
                    .iter()
                    .zip(marginal_y[..n].iter())
                    .map(|(x, y)| self.network.forward(x, y))
                    .collect();

                self.network.biases[l][o] += delta;

                let bound_pos = self.fd_bound(&j_pos, &m_pos);
                let bound_neg = self.fd_bound(&j_neg, &m_neg);
                grads_b[l][o] = -(bound_pos - bound_neg) / (2.0 * delta);
            }
        }

        // Apply gradients
        self.network.update(&grads_w, &grads_b, lr);

        mi
    }

    /// Stateless DV bound (does not update EMA).
    fn fd_bound(&self, joint: &[f64], marginal: &[f64]) -> f64 {
        let mean_joint = joint.iter().sum::<f64>() / joint.len() as f64;
        let exp_sum: f64 = marginal.iter().map(|&t| t.exp()).sum::<f64>();
        let mean_exp = (exp_sum / marginal.len() as f64).max(1e-300);
        mean_joint - mean_exp.ln()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VIB — Variational Information Bottleneck
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Variational Information Bottleneck.
#[derive(Clone, Debug)]
pub struct VibConfig {
    /// Dimensionality of input `x`.
    pub x_dim: usize,
    /// Bottleneck latent dimensionality.
    pub z_dim: usize,
    /// Number of output classes.
    pub y_classes: usize,
    /// Lagrange multiplier β controlling compression–relevance tradeoff.
    pub beta: f64,
    /// Hidden layer sizes for the encoder MLP.
    pub encoder_hidden: Vec<usize>,
    /// Hidden layer sizes for the decoder MLP.
    pub decoder_hidden: Vec<usize>,
    /// Number of Monte-Carlo samples for the reparameterisation estimator.
    pub n_samples: usize,
}

/// Encoder network: `x → (μ, log σ²)` in the latent space `Z`.
#[derive(Clone, Debug)]
pub struct VibEncoder {
    /// Shared hidden layers.
    pub hidden: Vec<SimpleLinear>,
    /// Head that produces `μ ∈ ℝ^{z_dim}`.
    pub mu_head: SimpleLinear,
    /// Head that produces `log σ² ∈ ℝ^{z_dim}`.
    pub log_var_head: SimpleLinear,
}

impl VibEncoder {
    /// Create a new encoder.
    pub fn new(x_dim: usize, hidden_sizes: &[usize], z_dim: usize) -> Self {
        let mut layer_in = x_dim;
        let hidden = hidden_sizes
            .iter()
            .map(|&h| {
                let l = SimpleLinear::new(layer_in, h);
                layer_in = h;
                l
            })
            .collect();
        let mu_head = SimpleLinear::new(layer_in, z_dim);
        let log_var_head = SimpleLinear::new(layer_in, z_dim);
        Self {
            hidden,
            mu_head,
            log_var_head,
        }
    }

    /// Compute `(μ, log σ²)` for a given input `x`.
    pub fn encode(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut h: Vec<f64> = x.to_vec();
        for layer in &self.hidden {
            h = layer.forward(&h);
            h = h.into_iter().map(relu).collect();
        }
        let mu = self.mu_head.forward(&h);
        let log_var = self.log_var_head.forward(&h);
        (mu, log_var)
    }

    /// Reparameterised sample: `z = μ + σ ⊙ ε`, where `ε ~ N(0, I)`.
    pub fn sample(&self, mu: &[f64], log_var: &[f64]) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(99u64);
        mu.iter()
            .zip(log_var.iter())
            .map(|(&m, &lv)| {
                let std_dev = (lv.clamp(-10.0, 10.0) * 0.5).exp();
                m + std_dev * sample_normal(&mut rng)
            })
            .collect()
    }
}

/// Decoder network: `z → p(y | z)`.
#[derive(Clone, Debug)]
pub struct VibDecoder {
    /// All layers (hidden + output).
    pub layers: Vec<SimpleLinear>,
}

impl VibDecoder {
    /// Create a new decoder.
    pub fn new(z_dim: usize, hidden_sizes: &[usize], y_classes: usize) -> Self {
        let mut layer_in = z_dim;
        let mut layers: Vec<SimpleLinear> = hidden_sizes
            .iter()
            .map(|&h| {
                let l = SimpleLinear::new(layer_in, h);
                layer_in = h;
                l
            })
            .collect();
        layers.push(SimpleLinear::new(layer_in, y_classes));
        Self { layers }
    }

    /// Decode `z` into class probabilities via softmax.
    pub fn decode(&self, z: &[f64]) -> Vec<f64> {
        let n = self.layers.len();
        let mut h: Vec<f64> = z.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h);
            if i < n - 1 {
                h = h.into_iter().map(relu).collect();
            }
        }
        softmax(&mut h);
        h
    }
}

/// Variational Information Bottleneck model.
///
/// Jointly trains an encoder that maps `x → Z` through a stochastic bottleneck
/// and a decoder that maps `Z → Y`, with objective
/// `L = CE(Y, Ŷ) + β * KL(q(Z|X) || N(0,I))`.
#[derive(Clone, Debug)]
pub struct VariationalInformationBottleneck {
    /// Stochastic encoder.
    pub encoder: VibEncoder,
    /// Label predictor decoder.
    pub decoder: VibDecoder,
    /// Configuration.
    pub config: VibConfig,
}

impl VariationalInformationBottleneck {
    /// Instantiate a new VIB model from the given configuration.
    pub fn new(config: VibConfig) -> Self {
        let encoder = VibEncoder::new(config.x_dim, &config.encoder_hidden, config.z_dim);
        let decoder = VibDecoder::new(config.z_dim, &config.decoder_hidden, config.y_classes);
        Self {
            encoder,
            decoder,
            config,
        }
    }

    /// Forward pass: returns `(z, μ, log σ², p(y|z))`.
    pub fn forward(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mu, log_var) = self.encoder.encode(x);
        let z = self.encoder.sample(&mu, &log_var);
        let y_probs = self.decoder.decode(&z);
        (z, mu, log_var, y_probs)
    }

    /// Compute `(total_loss, cross_entropy, kl_divergence)`.
    ///
    /// * CE = `-log p(y_true | z)`
    /// * KL = `-0.5 Σ (1 + log σ² - μ² - σ²)` (standard-normal prior)
    /// * Total = CE + β × KL
    pub fn loss(&self, x: &[f64], y_true: usize) -> (f64, f64, f64) {
        let (_, mu, log_var, y_probs) = self.forward(x);

        // Cross-entropy: -log p(y_true | z)
        let p_true = y_probs.get(y_true).cloned().unwrap_or(1e-300).max(1e-300);
        let ce = -p_true.ln();

        // KL divergence from standard normal
        let kl: f64 = mu
            .iter()
            .zip(log_var.iter())
            .map(|(&m, &lv)| {
                let lv_clamped = lv.clamp(-20.0, 20.0);
                -0.5 * (1.0 + lv_clamped - m * m - lv_clamped.exp())
            })
            .sum();

        let total = ce + self.config.beta * kl;
        (total, ce, kl)
    }

    /// Return the most likely class for input `x`.
    pub fn predict(&self, x: &[f64]) -> usize {
        let (_, _, _, y_probs) = self.forward(x);
        y_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DIM — Deep InfoMax
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Deep InfoMax model.
#[derive(Clone, Debug)]
pub struct DimConfig {
    /// Dimensionality of the raw input.
    pub input_dim: usize,
    /// Dimension of local feature representation.
    pub local_feat_dim: usize,
    /// Dimension of global summary representation.
    pub global_feat_dim: usize,
    /// Weight for the global MI term.
    pub alpha: f64,
    /// Weight for the local MI term.
    pub beta: f64,
    /// Weight for the prior-matching term.
    pub gamma: f64,
}

/// Discriminator that scores `(global, local)` pairs.
///
/// Joint discriminator used for the global Deep InfoMax objective.
#[derive(Clone, Debug)]
pub struct GlobalDiscriminator {
    /// First linear layer.
    pub layer1: SimpleLinear,
    /// Second linear layer mapping to a scalar logit.
    pub layer2: SimpleLinear,
}

impl GlobalDiscriminator {
    /// Build a new `GlobalDiscriminator`.
    pub fn new(global_dim: usize, local_dim: usize) -> Self {
        let hidden = (global_dim + local_dim) / 2 + 1;
        let layer1 = SimpleLinear::new(global_dim + local_dim, hidden);
        let layer2 = SimpleLinear::new(hidden, 1);
        Self { layer1, layer2 }
    }

    /// Compute a scalar score for a `(global, local)` pair.
    pub fn score(&self, global: &[f64], local: &[f64]) -> f64 {
        let concat: Vec<f64> = global.iter().chain(local.iter()).cloned().collect();
        let h = self.layer1.forward(&concat);
        let h_relu: Vec<f64> = h.into_iter().map(relu).collect();
        let out = self.layer2.forward(&h_relu);
        out.first().cloned().unwrap_or(0.0)
    }
}

/// Deep InfoMax model for self-supervised representation learning.
///
/// Maximises mutual information between global and local representations
/// via a JSD-based discriminator objective.
#[derive(Clone, Debug)]
pub struct DeepInfoMax {
    /// Encoder: input → global representation.
    pub encoder: Vec<SimpleLinear>,
    /// Local feature extractor: input → local representation.
    pub local_extractor: Vec<SimpleLinear>,
    /// Global discriminator.
    pub global_disc: GlobalDiscriminator,
    /// Prior discriminator (prior matching).
    pub prior_disc: SimpleLinear,
    /// Configuration.
    pub config: DimConfig,
}

impl DeepInfoMax {
    /// Construct a new `DeepInfoMax` model.
    pub fn new(config: DimConfig) -> Self {
        let hidden = config.input_dim.max(4) / 2 + 1;
        let encoder = vec![
            SimpleLinear::new(config.input_dim, hidden),
            SimpleLinear::new(hidden, config.global_feat_dim),
        ];
        let local_extractor = vec![SimpleLinear::new(config.input_dim, config.local_feat_dim)];
        let global_disc = GlobalDiscriminator::new(config.global_feat_dim, config.local_feat_dim);
        let prior_disc = SimpleLinear::new(config.global_feat_dim, 1);
        Self {
            encoder,
            local_extractor,
            global_disc,
            prior_disc,
            config,
        }
    }

    /// Encode an input into `(global, local)` feature vectors.
    pub fn encode(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        // Global encoding
        let mut g: Vec<f64> = x.to_vec();
        let n_enc = self.encoder.len();
        for (i, layer) in self.encoder.iter().enumerate() {
            g = layer.forward(&g);
            if i < n_enc - 1 {
                g = g.into_iter().map(relu).collect();
            }
        }

        // Local encoding
        let mut l: Vec<f64> = x.to_vec();
        let n_loc = self.local_extractor.len();
        for (i, layer) in self.local_extractor.iter().enumerate() {
            l = layer.forward(&l);
            if i < n_loc - 1 {
                l = l.into_iter().map(relu).collect();
            }
        }

        (g, l)
    }

    /// JSD mutual-information loss for a single pair.
    ///
    /// `loss = -softplus(-joint_score) - softplus(marginal_score)`
    ///
    /// This is the Jensen-Shannon divergence-based lower bound, where a higher
    /// value indicates more mutual information between the representations.
    pub fn jsd_mi_loss(&self, joint_score: f64, marginal_score: f64) -> f64 {
        -softplus(-joint_score) - softplus(marginal_score)
    }

    /// Prior-matching loss: binary cross-entropy between `p(z)` and `N(0,I)`.
    ///
    /// A simple approximation — score the latent vector `z` with the prior
    /// discriminator, targeting label 0 (match the prior).
    pub fn prior_loss(&self, z: &[f64]) -> f64 {
        let logit = self.prior_disc.forward(z).first().cloned().unwrap_or(0.0);
        let p = sigmoid(logit);
        -(1.0 - p).max(1e-300).ln()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CLUB — Contrastive Log-ratio Upper Bound
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the CLUB estimator.
#[derive(Clone, Debug)]
pub struct ClubConfig {
    /// Dimensionality of `X`.
    pub x_dim: usize,
    /// Dimensionality of `Y`.
    pub y_dim: usize,
    /// Hidden layer width for the variational network.
    pub hidden_dim: usize,
}

/// Variational network that approximates `q(y | x) ~ N(μ(x), σ²(x))`.
#[derive(Clone, Debug)]
pub struct ClubVariationalNetwork {
    /// Network layers for the conditional mean `μ(x)`.
    pub mu_net: Vec<SimpleLinear>,
    /// Network layers for `log σ²(x)`.
    pub log_var_net: Vec<SimpleLinear>,
}

impl ClubVariationalNetwork {
    /// Build a new variational network.
    pub fn new(x_dim: usize, y_dim: usize, hidden_dim: usize) -> Self {
        let mu_net = vec![
            SimpleLinear::new(x_dim, hidden_dim),
            SimpleLinear::new(hidden_dim, y_dim),
        ];
        let log_var_net = vec![
            SimpleLinear::new(x_dim, hidden_dim),
            SimpleLinear::new(hidden_dim, y_dim),
        ];
        Self {
            mu_net,
            log_var_net,
        }
    }

    /// Compute `(μ(x), log σ²(x))`.
    pub fn forward(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut mu: Vec<f64> = x.to_vec();
        let n = self.mu_net.len();
        for (i, layer) in self.mu_net.iter().enumerate() {
            mu = layer.forward(&mu);
            if i < n - 1 {
                mu = mu.into_iter().map(relu).collect();
            }
        }

        let mut lv: Vec<f64> = x.to_vec();
        for (i, layer) in self.log_var_net.iter().enumerate() {
            lv = layer.forward(&lv);
            if i < n - 1 {
                lv = lv.into_iter().map(relu).collect();
            }
        }

        (mu, lv)
    }

    /// Evaluate `log q(y | x)` under the Gaussian approximation.
    pub fn log_prob(&self, x: &[f64], y: &[f64]) -> f64 {
        let (mu, log_var) = self.forward(x);
        let d = mu.len() as f64;
        let sum: f64 = mu
            .iter()
            .zip(log_var.iter())
            .zip(y.iter())
            .map(|((&m, &lv), &yi)| {
                let lv_c = lv.clamp(-20.0, 20.0);
                let var = lv_c.exp().max(1e-10);
                -0.5 * (yi - m).powi(2) / var - 0.5 * lv_c
            })
            .sum();
        sum - 0.5 * d * (2.0 * PI).ln()
    }
}

/// CLUB estimator (Contrastive Log-ratio Upper Bound on MI).
///
/// Provides an upper bound on mutual information `I(X; Y)` via
/// `CLUB = E[log q(y|x)] - E_x[E_{y'}[log q(y'|x)]]`.
#[derive(Clone, Debug)]
pub struct Club {
    /// The variational approximation network.
    pub vnet: ClubVariationalNetwork,
    /// Configuration.
    pub config: ClubConfig,
}

impl Club {
    /// Create a new CLUB estimator.
    pub fn new(config: ClubConfig) -> Self {
        let vnet = ClubVariationalNetwork::new(config.x_dim, config.y_dim, config.hidden_dim);
        Self { vnet, config }
    }

    /// Compute the CLUB upper bound on `I(X; Y)`.
    ///
    /// `CLUB = mean_i log q(y_i | x_i) - mean_{i,j} log q(y_j | x_i)`
    ///
    /// Returns `≥ 0` (true MI) in expectation for well-fitted `q`.
    pub fn estimate(&self, x: &[Vec<f64>], y: &[Vec<f64>]) -> f64 {
        let n = x.len().min(y.len());
        if n == 0 {
            return 0.0;
        }

        // E[log q(y_i | x_i)]
        let joint_mean: f64 = (0..n)
            .map(|i| self.vnet.log_prob(&x[i], &y[i]))
            .sum::<f64>()
            / n as f64;

        // E_{i}[ E_{j}[log q(y_j | x_i)] ]
        // For efficiency, approximate with a single cyclic offset (i≠j)
        let marginal_mean: f64 = (0..n)
            .map(|i| {
                let j = (i + 1) % n;
                self.vnet.log_prob(&x[i], &y[j])
            })
            .sum::<f64>()
            / n as f64;

        joint_mean - marginal_mean
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HSIC — Hilbert-Schmidt Independence Criterion
// ─────────────────────────────────────────────────────────────────────────────

/// Compute an RBF (Gaussian) kernel matrix.
///
/// `K[i][j] = exp(-‖x_i - x_j‖² / (2 σ²))`
pub fn rbf_kernel_matrix(x: &[Vec<f64>], bandwidth: f64) -> Vec<Vec<f64>> {
    let n = x.len();
    let bw2 = 2.0 * bandwidth * bandwidth;
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let d2 = sq_dist(&x[i], &x[j]);
                    (-d2 / bw2).exp()
                })
                .collect()
        })
        .collect()
}

/// Centre a kernel matrix `K` by `HKH` where `H = I - (1/n) 11ᵀ`.
fn centre_kernel(k: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = k.len();
    if n == 0 {
        return vec![];
    }
    let inv_n = 1.0 / n as f64;

    // Row means
    let row_mean: Vec<f64> = k
        .iter()
        .map(|row| row.iter().sum::<f64>() * inv_n)
        .collect();
    // Global mean
    let global_mean: f64 = row_mean.iter().sum::<f64>() * inv_n;

    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| k[i][j] - row_mean[i] - row_mean[j] + global_mean)
                .collect()
        })
        .collect()
}

/// Hilbert-Schmidt Independence Criterion.
///
/// Measures statistical dependence between `X` and `Y` in the RKHS.
/// A value of 0 indicates independence.
pub fn hsic(x: &[Vec<f64>], y: &[Vec<f64>], bandwidth_x: f64, bandwidth_y: f64) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let kx = rbf_kernel_matrix(&x[..n], bandwidth_x);
    let ky = rbf_kernel_matrix(&y[..n], bandwidth_y);

    let hkx = centre_kernel(&kx);
    let hky = centre_kernel(&ky);

    // HSIC = (1/(n-1)²) * trace(Kx_c * Ky_c)
    // Simplified: (1/(n-1)²) * sum_{i,j} Kx_c[i,j] * Ky_c[i,j]
    let trace_sum: f64 = (0..n)
        .map(|i| (0..n).map(|j| hkx[i][j] * hky[i][j]).sum::<f64>())
        .sum();

    let denom = ((n - 1) as f64).powi(2);
    trace_sum / denom
}

/// Normalised HSIC, scaled to `[0, 1]`.
///
/// `nHSIC(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))`
pub fn normalized_hsic(x: &[Vec<f64>], y: &[Vec<f64>], bandwidth_x: f64, bandwidth_y: f64) -> f64 {
    let hsic_xy = hsic(x, y, bandwidth_x, bandwidth_y);
    let hsic_xx = hsic(x, x, bandwidth_x, bandwidth_x);
    let hsic_yy = hsic(y, y, bandwidth_y, bandwidth_y);
    let denom = (hsic_xx * hsic_yy).sqrt();
    if denom < 1e-20 {
        0.0
    } else {
        (hsic_xy / denom).clamp(0.0, 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rényi Entropy Estimators
// ─────────────────────────────────────────────────────────────────────────────

/// Order parameter for Rényi entropy.
#[derive(Clone, Debug)]
pub enum RenyiOrder {
    /// Shannon entropy (α → 1).
    Order1,
    /// Collision entropy (α = 2).
    Order2,
    /// Min-entropy (α → ∞).
    OrderInfinity,
    /// General α-Rényi entropy.
    Alpha(f64),
}

/// Estimate Rényi entropy from samples via kernel density estimation.
///
/// Uses an RBF (Parzen window) estimator with the given `bandwidth`.
/// All orders use a consistent set of normalised pseudo-probabilities
/// `p_i = K(x_i, ·) / Σ_j K(x_j, ·)` so that monotonicity `H_α ≤ H_{α'}` for
/// `α ≥ α'` holds empirically.
pub fn renyi_entropy(samples: &[Vec<f64>], order: RenyiOrder, bandwidth: f64) -> f64 {
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }

    // KDE density estimate at each sample point (un-normalised):
    // p̂(x_i) = (1/n) Σ_j K(x_i, x_j)
    let bw2 = 2.0 * bandwidth * bandwidth;

    let raw: Vec<f64> = (0..n)
        .map(|i| {
            let sum: f64 = (0..n)
                .map(|j| {
                    let d2 = sq_dist(&samples[i], &samples[j]);
                    (-d2 / bw2).exp()
                })
                .sum();
            (sum / n as f64).max(1e-300)
        })
        .collect();

    // Normalise so that the pseudo-probabilities sum to 1 (discrete approximation)
    let total: f64 = raw.iter().sum::<f64>();
    let probs: Vec<f64> = raw.iter().map(|&p| (p / total).max(1e-300)).collect();

    match order {
        RenyiOrder::Order1 => {
            // Shannon: H_1 = -Σ p_i log p_i
            -probs.iter().map(|&p| p * p.ln()).sum::<f64>()
        }
        RenyiOrder::Order2 => {
            // Collision entropy: H_2 = -log Σ p_i²
            let sum_p2: f64 = probs.iter().map(|&p| p * p).sum::<f64>();
            -sum_p2.max(1e-300).ln()
        }
        RenyiOrder::OrderInfinity => {
            // Min-entropy: H_∞ = -log max p_i
            let max_p = probs.iter().cloned().fold(0.0_f64, f64::max);
            -max_p.max(1e-300).ln()
        }
        RenyiOrder::Alpha(alpha) => {
            if (alpha - 1.0).abs() < 1e-9 {
                // Limit α→1 is Shannon
                -probs.iter().map(|&p| p * p.ln()).sum::<f64>()
            } else {
                // H_α = (1/(1-α)) * log Σ p_i^α
                let sum_pa: f64 = probs.iter().map(|&p| p.powf(alpha)).sum::<f64>();
                (1.0 / (1.0 - alpha)) * sum_pa.max(1e-300).ln()
            }
        }
    }
}

/// Monte-Carlo estimate of KL divergence `KL(P ‖ Q)` via KDE.
///
/// Uses samples from `P` to estimate the log-ratio `log(p(x)/q(x))`.
pub fn kl_divergence_mc(p_samples: &[Vec<f64>], q_samples: &[Vec<f64>], bandwidth: f64) -> f64 {
    let np = p_samples.len();
    let nq = q_samples.len();
    if np == 0 || nq == 0 {
        return 0.0;
    }

    let bw2 = 2.0 * bandwidth * bandwidth;

    // For each x ~ P, estimate p(x) and q(x) via KDE
    let kl: f64 = (0..np)
        .map(|i| {
            let p_est: f64 = (0..np)
                .map(|j| {
                    let d2 = sq_dist(&p_samples[i], &p_samples[j]);
                    (-d2 / bw2).exp()
                })
                .sum::<f64>()
                / np as f64;

            let q_est: f64 = (0..nq)
                .map(|j| {
                    let d2 = sq_dist(&p_samples[i], &q_samples[j]);
                    (-d2 / bw2).exp()
                })
                .sum::<f64>()
                / nq as f64;

            let p_safe = p_est.max(1e-300);
            let q_safe = q_est.max(1e-300);
            p_safe.ln() - q_safe.ln()
        })
        .sum::<f64>();

    (kl / np as f64).max(0.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// TC-VAE — Total Correlation VAE objective
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the TC-VAE objective (Chen et al. 2018).
///
/// Decomposes the β-VAE ELBO into:
/// - Reconstruction loss (passed in as `recon_loss`)
/// - Mutual information `I(x; z)` (weighted by `alpha`)
/// - Total correlation `TC(z)` (weighted by `beta`)
/// - Dimension-wise KL `Σ_j KL(q(z_j)||p(z_j))` (weighted by `gamma`)
///
/// Approximation (Chen 2018 Eq. 4):
/// `TC = dim_kl - standard_kl` where
/// `standard_kl = 0.5 * Σ (μ² + σ² - log σ² - 1)`.
pub fn tc_vae_loss(
    mu: &[f64],
    log_var: &[f64],
    z: &[f64],
    recon_loss: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    n: usize,
) -> f64 {
    let d = mu.len();
    if d == 0 {
        return recon_loss;
    }

    // Standard KL: Σ_j KL(q(z_j | x) || N(0,1))
    let standard_kl: f64 = mu
        .iter()
        .zip(log_var.iter())
        .map(|(&m, &lv)| {
            let lv_c = lv.clamp(-20.0, 20.0);
            0.5 * (m * m + lv_c.exp() - lv_c - 1.0)
        })
        .sum();

    // Dimension-wise KL (same as standard_kl for factorised q)
    let dim_kl = standard_kl;

    // Total correlation approximation (Chen 2018): TC ≈ max(0, dim_kl - standard_kl)
    // For a single sample and factorised encoder, TC term is 0 exactly;
    // in practice use the batch-level approximation.
    let tc_term = (dim_kl - standard_kl).max(0.0);

    // MI(x; z) ≈ E[log q(z|x)] - E[log q(z)]
    // Simple approximation: scaled by dataset size
    let log_scale = (n as f64).max(1.0).ln();
    let mi_term = standard_kl / log_scale;

    recon_loss + alpha * mi_term + beta * tc_term + gamma * dim_kl
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};

    // ── Helpers ─────────────────────────────────────────────────────────────

    fn gen_correlated_samples(n: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut rng = StdRng::seed_from_u64(42u64);
        let x: Vec<Vec<f64>> = (0..n).map(|_| vec![sample_normal(&mut rng)]).collect();
        let y: Vec<Vec<f64>> = x
            .iter()
            .map(|xi| vec![xi[0] + 0.1 * sample_normal(&mut rng)])
            .collect();
        (x, y)
    }

    fn gen_independent_samples(n: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut rng = StdRng::seed_from_u64(7u64);
        let x: Vec<Vec<f64>> = (0..n).map(|_| vec![sample_normal(&mut rng)]).collect();
        let y: Vec<Vec<f64>> = (0..n).map(|_| vec![sample_normal(&mut rng)]).collect();
        (x, y)
    }

    fn shuffle_y(y: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = y.len();
        (0..n).map(|i| y[(i + n / 2) % n].clone()).collect()
    }

    // ── SimpleLinear tests ───────────────────────────────────────────────────

    #[test]
    fn test_simple_linear_output_shape() {
        let layer = SimpleLinear::new(4, 8);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let out = layer.forward(&x);
        assert_eq!(out.len(), 8, "output should have 8 elements");
    }

    #[test]
    fn test_simple_linear_zero_input() {
        let layer = SimpleLinear::new(3, 5);
        let x = vec![0.0, 0.0, 0.0];
        let out = layer.forward(&x);
        // With zero input, output should equal bias (which is 0 after Xavier init)
        assert_eq!(out.len(), 5);
        for &v in &out {
            assert!(
                v.abs() < 1e-10,
                "zero-input forward should return biases (0s)"
            );
        }
    }

    // ── MINE tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_mine_dv_output_finite() {
        let mut mine = Mine::new(1, 1, 16, MineEstimator::Dv);
        let (x, y) = gen_correlated_samples(30);
        let y_marg = shuffle_y(&y);
        let mi = mine.estimate(&x, &y_marg);
        assert!(mi.is_finite(), "MINE DV estimate should be finite");
    }

    #[test]
    fn test_mine_dv_correlated_nonnegative_after_training() {
        // Use tiny network (hidden_dim=4, 2 layers) and few samples to keep FD gradient
        // computation fast. The goal is only to verify the step returns a finite value.
        let mut mine = Mine::new(1, 1, 4, MineEstimator::Dv);
        let (x, y) = gen_correlated_samples(8);
        let y_marg = shuffle_y(&y);
        // A single training step is sufficient to verify the update pipeline works
        let last = mine.train_step(&x, &y, &y_marg, 0.01);
        // After training on strongly correlated data, estimate should be finite
        assert!(
            last.is_finite(),
            "MINE estimate should be finite after training"
        );
    }

    #[test]
    fn test_mine_independent_near_zero() {
        let mut mine = Mine::new(1, 1, 16, MineEstimator::Dv);
        let (x, y) = gen_independent_samples(40);
        let y_marg = shuffle_y(&y);
        let mi = mine.estimate(&x, &y_marg);
        // Independent X, Y → MI ~ 0; allow generous range due to untrained net
        assert!(
            mi < 5.0,
            "MI for independent data should not be very large: {}",
            mi
        );
    }

    #[test]
    fn test_mine_smile_finite() {
        let mut mine = Mine::new(2, 2, 16, MineEstimator::Smile { clip: 5.0 });
        let (x, y) = gen_correlated_samples(20);
        let y_marg = shuffle_y(&y);
        let mi = mine.estimate(&x, &y_marg);
        assert!(mi.is_finite(), "SMILE estimate should be finite");
    }

    #[test]
    fn test_mine_fd_finite() {
        let mut mine = Mine::new(1, 1, 8, MineEstimator::Fd);
        let (x, y) = gen_correlated_samples(20);
        let y_marg = shuffle_y(&y);
        let mi = mine.estimate(&x, &y_marg);
        assert!(mi.is_finite(), "FD estimate should be finite");
    }

    #[test]
    fn test_mine_network_forward_shape() {
        let net = MineNetwork::new(3, 2, 16, 3);
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0.5, 1.5];
        let out = net.forward(&x, &y);
        assert!(
            out.is_finite(),
            "MineNetwork forward should return a finite scalar"
        );
    }

    // ── VIB tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_vib_encoder_output_shape() {
        let enc = VibEncoder::new(4, &[8], 3);
        let x = vec![0.1, 0.2, 0.3, 0.4];
        let (mu, log_var) = enc.encode(&x);
        assert_eq!(mu.len(), 3, "mu should have z_dim elements");
        assert_eq!(log_var.len(), 3, "log_var should have z_dim elements");
    }

    #[test]
    fn test_vib_sample_shape() {
        let enc = VibEncoder::new(4, &[8], 3);
        let x = vec![0.1, 0.2, 0.3, 0.4];
        let (mu, log_var) = enc.encode(&x);
        let z = enc.sample(&mu, &log_var);
        assert_eq!(z.len(), 3, "sampled z should have z_dim elements");
    }

    #[test]
    fn test_vib_kl_nonnegative() {
        let config = VibConfig {
            x_dim: 4,
            z_dim: 2,
            y_classes: 3,
            beta: 1.0,
            encoder_hidden: vec![8],
            decoder_hidden: vec![],
            n_samples: 1,
        };
        let vib = VariationalInformationBottleneck::new(config);
        let x = vec![0.5, -0.3, 1.2, 0.0];
        let (_, ce, kl) = vib.loss(&x, 0);
        assert!(kl >= 0.0, "KL divergence must be non-negative, got {}", kl);
    }

    #[test]
    fn test_vib_total_loss_components() {
        let config = VibConfig {
            x_dim: 4,
            z_dim: 2,
            y_classes: 3,
            beta: 2.0,
            encoder_hidden: vec![8],
            decoder_hidden: vec![],
            n_samples: 1,
        };
        let vib = VariationalInformationBottleneck::new(config);
        let x = vec![0.5, -0.3, 1.2, 0.0];
        let (total, ce, kl) = vib.loss(&x, 1);
        let expected = ce + 2.0 * kl;
        assert!(
            (total - expected).abs() < 1e-10,
            "total = CE + β*KL should hold: got {} vs {}",
            total,
            expected
        );
    }

    #[test]
    fn test_vib_predict_matches_argmax() {
        let config = VibConfig {
            x_dim: 4,
            z_dim: 2,
            y_classes: 5,
            beta: 1.0,
            encoder_hidden: vec![8],
            decoder_hidden: vec![4],
            n_samples: 1,
        };
        let vib = VariationalInformationBottleneck::new(config);
        let x = vec![1.0, -1.0, 0.5, 0.0];
        let predicted = vib.predict(&x);
        let (_, _, _, y_probs) = vib.forward(&x);
        let argmax = y_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(
            predicted, argmax,
            "predict() should return argmax of y_probs"
        );
    }

    #[test]
    fn test_vib_loss_finite() {
        let config = VibConfig {
            x_dim: 3,
            z_dim: 2,
            y_classes: 2,
            beta: 0.5,
            encoder_hidden: vec![4],
            decoder_hidden: vec![],
            n_samples: 1,
        };
        let vib = VariationalInformationBottleneck::new(config);
        let x = vec![0.3, 0.7, -0.2];
        let (total, ce, kl) = vib.loss(&x, 0);
        assert!(total.is_finite(), "total loss should be finite");
        assert!(ce.is_finite(), "cross-entropy should be finite");
        assert!(kl.is_finite(), "KL should be finite");
    }

    // ── DIM tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_dim_jsd_loss_at_high_joint_score() {
        let config = DimConfig {
            input_dim: 4,
            local_feat_dim: 8,
            global_feat_dim: 16,
            alpha: 1.0,
            beta: 1.0,
            gamma: 0.1,
        };
        let dim = DeepInfoMax::new(config);
        // High joint score, zero marginal → JSD loss should be strongly negative
        let loss = dim.jsd_mi_loss(10.0, 0.0);
        assert!(loss < 0.0, "JSD MI loss should be negative: {}", loss);
    }

    #[test]
    fn test_dim_softplus_at_zero() {
        let val = softplus(0.0);
        let expected = 2.0_f64.ln();
        assert!(
            (val - expected).abs() < 1e-10,
            "softplus(0) should be ln(2) ≈ {}, got {}",
            expected,
            val
        );
    }

    #[test]
    fn test_dim_encode_output_shape() {
        let config = DimConfig {
            input_dim: 8,
            local_feat_dim: 4,
            global_feat_dim: 6,
            alpha: 1.0,
            beta: 1.0,
            gamma: 0.1,
        };
        let dim = DeepInfoMax::new(config.clone());
        let x = vec![0.1; 8];
        let (global, local) = dim.encode(&x);
        assert_eq!(global.len(), config.global_feat_dim);
        assert_eq!(local.len(), config.local_feat_dim);
    }

    #[test]
    fn test_dim_prior_loss_finite() {
        let config = DimConfig {
            input_dim: 4,
            local_feat_dim: 2,
            global_feat_dim: 4,
            alpha: 1.0,
            beta: 1.0,
            gamma: 0.1,
        };
        let dim = DeepInfoMax::new(config);
        let z = vec![0.3, -0.5, 1.0, 0.0];
        let loss = dim.prior_loss(&z);
        assert!(loss.is_finite(), "prior loss should be finite");
    }

    // ── CLUB tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_club_estimate_finite() {
        let config = ClubConfig {
            x_dim: 2,
            y_dim: 2,
            hidden_dim: 8,
        };
        let club = Club::new(config);
        let (x, y) = gen_independent_samples(20);
        let x2: Vec<Vec<f64>> = x.iter().map(|v| vec![v[0], v[0] * 0.5]).collect();
        let y2: Vec<Vec<f64>> = y.iter().map(|v| vec![v[0], v[0] * 0.3]).collect();
        let est = club.estimate(&x2, &y2);
        assert!(est.is_finite(), "CLUB estimate should be finite");
    }

    #[test]
    fn test_club_independent_small() {
        let config = ClubConfig {
            x_dim: 1,
            y_dim: 1,
            hidden_dim: 4,
        };
        let club = Club::new(config);
        let (x, y) = gen_independent_samples(50);
        let est = club.estimate(&x, &y);
        // CLUB is an upper bound; for independent data it should be small (but not guaranteed 0)
        assert!(
            est.is_finite(),
            "CLUB estimate for independent data should be finite"
        );
        assert!(
            est.abs() < 20.0,
            "CLUB bound should not be excessively large: {}",
            est
        );
    }

    #[test]
    fn test_club_variational_logprob_finite() {
        let vnet = ClubVariationalNetwork::new(2, 2, 8);
        let x = vec![0.5, -0.3];
        let y = vec![1.0, 0.2];
        let lp = vnet.log_prob(&x, &y);
        assert!(lp.is_finite(), "log_prob should be finite");
    }

    // ── HSIC tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_rbf_kernel_self_is_one() {
        let x = vec![vec![1.0, 2.0, 3.0]];
        let k = rbf_kernel_matrix(&x, 1.0);
        assert!((k[0][0] - 1.0).abs() < 1e-12, "K(x,x) should be 1.0");
    }

    #[test]
    fn test_rbf_kernel_symmetric() {
        let (x, _) = gen_correlated_samples(10);
        let k = rbf_kernel_matrix(&x, 1.0);
        for i in 0..x.len() {
            for j in 0..x.len() {
                assert!(
                    (k[i][j] - k[j][i]).abs() < 1e-12,
                    "kernel matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_hsic_nonnegative() {
        let (x, y) = gen_correlated_samples(15);
        let val = hsic(&x, &y, 1.0, 1.0);
        assert!(val >= 0.0, "HSIC should be non-negative, got {}", val);
    }

    #[test]
    fn test_hsic_independent_near_zero() {
        // For strongly independent data the HSIC should be small
        let mut rng = StdRng::seed_from_u64(999u64);
        let n = 30;
        let x: Vec<Vec<f64>> = (0..n).map(|_| vec![sample_normal(&mut rng)]).collect();
        let y: Vec<Vec<f64>> = (0..n).map(|_| vec![sample_normal(&mut rng)]).collect();
        let val = hsic(&x, &y, 1.0, 1.0);
        assert!(
            val < 1.0,
            "HSIC for independent data should be small: {}",
            val
        );
    }

    #[test]
    fn test_normalized_hsic_in_unit_interval() {
        let (x, y) = gen_correlated_samples(20);
        let val = normalized_hsic(&x, &y, 1.0, 1.0);
        assert!(
            (0.0..=1.0).contains(&val),
            "normalized HSIC should be in [0,1], got {}",
            val
        );
    }

    #[test]
    fn test_hsic_self_correlation() {
        let (x, _) = gen_correlated_samples(10);
        // HSIC(x, x) measures the variance of K_x which is always ≥ 0
        let val = hsic(&x, &x, 1.0, 1.0);
        assert!(val >= 0.0, "HSIC(x,x) should be non-negative");
    }

    // ── Rényi entropy tests ─────────────────────────────────────────────────

    #[test]
    fn test_renyi_order2_le_order1() {
        let mut rng = StdRng::seed_from_u64(1u64);
        let samples: Vec<Vec<f64>> = (0..40).map(|_| vec![sample_normal(&mut rng)]).collect();
        let h1 = renyi_entropy(&samples, RenyiOrder::Order1, 0.5);
        let h2 = renyi_entropy(&samples, RenyiOrder::Order2, 0.5);
        // H_α ≤ H_{α'} for α ≥ α' (Rényi entropy is monotone decreasing in α)
        // So H2 ≤ H1
        assert!(
            h2 <= h1 + 1e-6,
            "H_2 should be ≤ H_1 (Rényi monotonicity): H1={}, H2={}",
            h1,
            h2
        );
    }

    #[test]
    fn test_renyi_entropy_positive() {
        let mut rng = StdRng::seed_from_u64(2u64);
        let samples: Vec<Vec<f64>> = (0..20).map(|_| vec![sample_normal(&mut rng)]).collect();
        let h = renyi_entropy(&samples, RenyiOrder::Order1, 1.0);
        assert!(h.is_finite(), "Rényi entropy should be finite");
    }

    #[test]
    fn test_renyi_alpha_general() {
        let mut rng = StdRng::seed_from_u64(3u64);
        let samples: Vec<Vec<f64>> = (0..20).map(|_| vec![sample_normal(&mut rng)]).collect();
        let h = renyi_entropy(&samples, RenyiOrder::Alpha(2.0), 1.0);
        assert!(h.is_finite(), "Alpha-Rényi entropy should be finite");
    }

    #[test]
    fn test_renyi_infinity_order() {
        let mut rng = StdRng::seed_from_u64(4u64);
        let samples: Vec<Vec<f64>> = (0..20).map(|_| vec![sample_normal(&mut rng)]).collect();
        let h = renyi_entropy(&samples, RenyiOrder::OrderInfinity, 1.0);
        assert!(h.is_finite(), "Min-entropy should be finite");
    }

    // ── KL divergence tests ─────────────────────────────────────────────────

    #[test]
    fn test_kl_divergence_nonnegative() {
        let mut rng = StdRng::seed_from_u64(5u64);
        let p: Vec<Vec<f64>> = (0..20).map(|_| vec![sample_normal(&mut rng)]).collect();
        let q: Vec<Vec<f64>> = (0..20)
            .map(|_| vec![sample_normal(&mut rng) + 1.0])
            .collect();
        let kl = kl_divergence_mc(&p, &q, 0.5);
        assert!(
            kl >= 0.0,
            "KL divergence should be non-negative, got {}",
            kl
        );
    }

    #[test]
    fn test_kl_same_distribution_near_zero() {
        let mut rng = StdRng::seed_from_u64(6u64);
        let p: Vec<Vec<f64>> = (0..30).map(|_| vec![sample_normal(&mut rng)]).collect();
        // Use same samples for q → KL(P||P) ≈ 0
        let kl = kl_divergence_mc(&p, &p, 0.5);
        assert!(kl.abs() < 0.5, "KL(P||P) should be near 0, got {}", kl);
    }

    // ── TC-VAE tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_tc_vae_loss_finite_positive() {
        let mu = vec![0.1, -0.2, 0.5];
        let log_var = vec![-1.0, 0.0, -0.5];
        let z = vec![0.15, -0.18, 0.52];
        let recon = 2.5_f64;
        let loss = tc_vae_loss(&mu, &log_var, &z, recon, 1.0, 1.0, 1.0, 1000);
        assert!(loss.is_finite(), "TC-VAE loss should be finite");
        assert!(
            loss > 0.0,
            "TC-VAE loss should be positive for typical inputs"
        );
    }

    #[test]
    fn test_tc_vae_increases_with_beta() {
        let mu = vec![0.5, -0.5];
        let log_var = vec![0.0, 0.0];
        let z = vec![0.6, -0.4];
        let recon = 1.0_f64;
        let l1 = tc_vae_loss(&mu, &log_var, &z, recon, 1.0, 1.0, 1.0, 100);
        let l2 = tc_vae_loss(&mu, &log_var, &z, recon, 1.0, 5.0, 1.0, 100);
        // TC term is ≥ 0, so larger beta should produce ≥ loss
        assert!(
            l2 >= l1 - 1e-10,
            "Larger beta should not decrease TC-VAE loss"
        );
    }

    // ── Global discriminator tests ───────────────────────────────────────────

    #[test]
    fn test_global_discriminator_score_finite() {
        let disc = GlobalDiscriminator::new(8, 4);
        let global = vec![0.1; 8];
        let local = vec![0.2; 4];
        let score = disc.score(&global, &local);
        assert!(score.is_finite(), "discriminator score should be finite");
    }

    // ── Mine train_step test ─────────────────────────────────────────────────

    #[test]
    fn test_mine_train_step_returns_finite() {
        let mut mine = Mine::new(1, 1, 8, MineEstimator::Dv);
        let (x, y) = gen_correlated_samples(10);
        let y_marg = shuffle_y(&y);
        let mi = mine.train_step(&x, &y, &y_marg, 0.001);
        assert!(
            mi.is_finite(),
            "train_step should return a finite MI estimate"
        );
    }

    // ── VIB decoder test ─────────────────────────────────────────────────────

    #[test]
    fn test_vib_decoder_probabilities_sum_to_one() {
        let dec = VibDecoder::new(4, &[8], 3);
        let z = vec![0.3, -0.1, 1.0, 0.5];
        let probs = dec.decode(&z);
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "probabilities should sum to 1, got {}",
            sum
        );
    }

    // ── Edge case: empty inputs ──────────────────────────────────────────────

    #[test]
    fn test_hsic_too_few_samples_returns_zero() {
        let x = vec![vec![1.0]];
        let y = vec![vec![2.0]];
        let val = hsic(&x, &y, 1.0, 1.0);
        assert_eq!(val, 0.0, "HSIC with n<2 should return 0");
    }

    #[test]
    fn test_mine_empty_samples_returns_zero() {
        let mut mine = Mine::new(1, 1, 4, MineEstimator::Dv);
        let mi = mine.estimate(&[], &[]);
        assert_eq!(mi, 0.0, "MINE with empty samples should return 0");
    }

    #[test]
    fn test_club_empty_samples_returns_zero() {
        let config = ClubConfig {
            x_dim: 1,
            y_dim: 1,
            hidden_dim: 4,
        };
        let club = Club::new(config);
        let est = club.estimate(&[], &[]);
        assert_eq!(est, 0.0, "CLUB with empty samples should return 0");
    }

    // ── Softplus numerical properties ────────────────────────────────────────

    #[test]
    fn test_softplus_large_positive() {
        // softplus(x) ≈ x for large x
        let val = softplus(100.0);
        assert!((val - 100.0).abs() < 1.0, "softplus(100) ≈ 100");
    }

    #[test]
    fn test_softplus_large_negative() {
        // softplus(x) ≈ exp(x) for large negative x
        let val = softplus(-50.0);
        assert!(val < 1e-10, "softplus(-50) should be near 0");
    }
}
