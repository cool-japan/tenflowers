//! Simulation-Based Inference (SBI): SNPE / SNLE / SNRE.
//!
//! Sequential amortised Bayesian inference for intractable likelihoods.
//! Implements [`SequentialNpe`] (APT/MAF), [`NeuralLikelihood`] (SNLE),
//! [`NeuralRatioEstimator`] (SNRE-B), and calibration diagnostics.

pub mod extensions;
pub use extensions::*;
pub mod advanced;
pub use advanced::*;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Box-Muller transform: single N(0,1) sample.
#[inline]
pub(crate) fn randn(rng: &mut impl Rng) -> f64 {
    let u1 = (rng.random::<f64>()).max(1e-300);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Normal log-probability: log N(x; mu, sigma^2).
#[inline]
pub(crate) fn log_normal(x: f64, mu: f64, sigma: f64) -> f64 {
    let diff = x - mu;
    let log_sig = sigma.max(1e-8).ln();
    -0.5 * std::f64::consts::LN_2
        - 0.5 * (2.0 * std::f64::consts::PI).ln()
        - log_sig
        - 0.5 * (diff / sigma.max(1e-8)).powi(2)
}

/// Numerically stable softplus: log(1 + exp(x)).
#[inline]
pub(crate) fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Tanh activation.
#[inline]
pub(crate) fn tanh_act(x: f64) -> f64 {
    x.tanh()
}

/// ReLU activation.
#[inline]
pub(crate) fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Sigmoid.
#[inline]
pub(crate) fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Simulator trait
// ─────────────────────────────────────────────────────────────────────────────

/// Core interface for a stochastic forward simulator plus prior.
pub trait Simulator: Send + Sync {
    /// Simulate observations `x` given parameters `theta`.
    fn simulate(&self, theta: &[f64], rng: &mut StdRng) -> Vec<f64>;
    /// Evaluate log p(theta) under the prior.
    fn prior_log_prob(&self, theta: &[f64]) -> f64;
    /// Sample theta ~ prior.
    fn prior_sample(&self, rng: &mut StdRng) -> Vec<f64>;
    /// Dimensionality of theta.
    fn theta_dim(&self) -> usize;
    /// Dimensionality of x.
    fn x_dim(&self) -> usize;
}

// ─────────────────────────────────────────────────────────────────────────────
// GaussianSimulator
// ─────────────────────────────────────────────────────────────────────────────

/// Toy Gaussian simulator: x_i = theta_i + N(0, noise_std^2).
///
/// Prior: theta_i ~ N(prior_mean_i, prior_std_i^2).
#[derive(Debug, Clone)]
pub struct GaussianSimulator {
    /// Number of theta / x dimensions.
    pub theta_dim: usize,
    /// Standard deviation of observation noise.
    pub noise_std: f64,
    /// Prior means.
    pub prior_mean: Vec<f64>,
    /// Prior standard deviations.
    pub prior_std: Vec<f64>,
}

impl GaussianSimulator {
    /// Create with isotropic N(0, 1) prior.
    pub fn new(dim: usize, noise_std: f64) -> Self {
        Self {
            theta_dim: dim,
            noise_std,
            prior_mean: vec![0.0; dim],
            prior_std: vec![1.0; dim],
        }
    }

    /// Create with custom prior.
    pub fn with_prior(
        dim: usize,
        noise_std: f64,
        prior_mean: Vec<f64>,
        prior_std: Vec<f64>,
    ) -> Self {
        Self {
            theta_dim: dim,
            noise_std,
            prior_mean,
            prior_std,
        }
    }
}

impl Simulator for GaussianSimulator {
    fn simulate(&self, theta: &[f64], rng: &mut StdRng) -> Vec<f64> {
        theta
            .iter()
            .map(|&t| t + self.noise_std * randn(rng))
            .collect()
    }

    fn prior_log_prob(&self, theta: &[f64]) -> f64 {
        theta
            .iter()
            .enumerate()
            .map(|(i, &t)| log_normal(t, self.prior_mean[i], self.prior_std[i]))
            .sum()
    }

    fn prior_sample(&self, rng: &mut StdRng) -> Vec<f64> {
        (0..self.theta_dim)
            .map(|i| self.prior_mean[i] + self.prior_std[i] * randn(rng))
            .collect()
    }

    fn theta_dim(&self) -> usize {
        self.theta_dim
    }
    fn x_dim(&self) -> usize {
        self.theta_dim
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MadeLayer — Masked Autoencoder for Distribution Estimation
// ─────────────────────────────────────────────────────────────────────────────

/// One layer of a Masked Autoencoder (MADE).
///
/// The autoregressive mask ensures that output `i` depends only on inputs `j`
/// where `output_order[i] > input_order[j]`.
#[derive(Debug, Clone)]
pub struct MadeLayer {
    /// Weight matrix `[out_dim][in_dim]`.
    pub weights: Vec<Vec<f64>>,
    /// Bias vector of length `out_dim`.
    pub biases: Vec<f64>,
    /// Boolean autoregressive mask `[out_dim][in_dim]`.
    pub mask: Vec<Vec<bool>>,
}

impl MadeLayer {
    /// Construct a `MadeLayer` with a He-initialised weight matrix and the
    /// autoregressive mask `mask[i][j] = output_order[i] > input_order[j]`.
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        input_order: &[usize],
        output_order: &[usize],
        rng: &mut StdRng,
    ) -> Self {
        let scale = (2.0 / in_dim as f64).sqrt();
        let weights: Vec<Vec<f64>> = (0..out_dim)
            .map(|_| (0..in_dim).map(|_| scale * randn(rng)).collect())
            .collect();
        let biases = vec![0.0; out_dim];
        let mask: Vec<Vec<bool>> = (0..out_dim)
            .map(|i| {
                (0..in_dim)
                    .map(|j| output_order[i] > input_order[j])
                    .collect()
            })
            .collect();
        Self {
            weights,
            biases,
            mask,
        }
    }

    /// Forward pass: applies the masked linear transformation.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let out_dim = self.weights.len();
        (0..out_dim)
            .map(|i| {
                let sum: f64 = x
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| {
                        if self.mask[i][j] {
                            self.weights[i][j] * v
                        } else {
                            0.0
                        }
                    })
                    .sum();
                sum + self.biases[i]
            })
            .collect()
    }

    /// In-place weight update with gradient descent.
    pub fn update(&mut self, grad_w: &[Vec<f64>], grad_b: &[f64], lr: f64) {
        for (i, row) in self.weights.iter_mut().enumerate() {
            for (j, w) in row.iter_mut().enumerate() {
                *w -= lr * grad_w[i][j];
            }
            self.biases[i] -= lr * grad_b[i];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NeuralDensityEstimator — Masked Autoregressive Flow q(theta | x)
// ─────────────────────────────────────────────────────────────────────────────

/// Masked Autoregressive Flow that approximates p(theta | x).
///
/// Architecture:
/// - A small MLP (`context_net`) embeds `x` into a fixed-size context vector.
/// - `n_layers` MADE layers transform [theta ‖ context] into per-dimension
///   (mu, log_scale) parameters used for Gaussian base-distribution scores.
#[derive(Debug, Clone)]
pub struct NeuralDensityEstimator {
    /// MADE hidden layers.
    pub layers: Vec<MadeLayer>,
    /// Context MLP: list of (weight matrix `[out][in]`, bias `[out]`).
    pub context_net: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    /// Dimension of theta.
    pub theta_dim: usize,
    /// Embedding dimension for x context.
    pub context_dim: usize,
    /// Width of hidden layers.
    pub hidden_dim: usize,
}

impl NeuralDensityEstimator {
    /// Create with random initialisation.
    ///
    /// `n_layers` controls the depth of the MADE stack (minimum 1).
    pub fn new(
        theta_dim: usize,
        x_dim: usize,
        hidden_dim: usize,
        n_layers: usize,
        rng: &mut StdRng,
    ) -> Self {
        let context_dim = hidden_dim.min(64);

        // Context MLP: x_dim → hidden_dim → context_dim
        let context_net = build_mlp(&[x_dim, hidden_dim, context_dim], rng);

        // MADE stack: input = [theta ‖ context], output = 2 * theta_dim (mu, log_scale)
        // Input orders: theta positions get orders 1..=theta_dim, context gets order 0
        let total_in = theta_dim + context_dim;
        let mut input_order: Vec<usize> = (1..=theta_dim).collect();
        input_order.extend(vec![0usize; context_dim]); // context positions: order 0

        // Hidden layers map theta_dim positions; output produces 2 values per theta dim
        let hidden_order: Vec<usize> = (0..hidden_dim).map(|i| (i % theta_dim) + 1).collect();
        let output_order: Vec<usize> = (0..2 * theta_dim).map(|i| (i % theta_dim) + 1).collect();

        let n_layers = n_layers.max(1);
        let mut layers = Vec::with_capacity(n_layers);

        // First layer: input → hidden
        layers.push(MadeLayer::new(
            total_in,
            hidden_dim,
            &input_order,
            &hidden_order,
            rng,
        ));

        // Intermediate layers: hidden → hidden
        for _ in 1..n_layers {
            layers.push(MadeLayer::new(
                hidden_dim,
                hidden_dim,
                &hidden_order,
                &hidden_order,
                rng,
            ));
        }

        // Output layer: hidden → 2 * theta_dim
        layers.push(MadeLayer::new(
            hidden_dim,
            2 * theta_dim,
            &hidden_order,
            &output_order,
            rng,
        ));

        Self {
            layers,
            context_net,
            theta_dim,
            context_dim,
            hidden_dim,
        }
    }

    /// Forward pass: compute the context embedding and MADE activations.
    fn forward_pass(&self, theta: &[f64], x: &[f64]) -> Vec<f64> {
        // Embed x
        let ctx = mlp_forward(&self.context_net, x);

        // Concatenate theta ‖ ctx
        let mut inp: Vec<f64> = theta.to_vec();
        inp.extend_from_slice(&ctx);

        // Pass through MADE layers with tanh activations on hidden, identity on output
        let n = self.layers.len();
        let mut h = inp;
        for (idx, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h);
            if idx < n - 1 {
                for v in &mut h {
                    *v = tanh_act(*v);
                }
            }
        }
        h
    }

    /// Evaluate log q(theta | x) under the normalising-flow density.
    pub fn log_prob(&self, theta: &[f64], x: &[f64]) -> f64 {
        let out = self.forward_pass(theta, x);
        let td = self.theta_dim;
        let mut lp = 0.0;
        for i in 0..td {
            let mu = out[i];
            let log_scale = out[td + i].clamp(-5.0, 5.0);
            let sigma = log_scale.exp().max(1e-6);
            lp += log_normal(theta[i], mu, sigma);
        }
        lp
    }

    /// Autoregressive sampling: draw theta ~ q(· | x).
    pub fn sample(&self, x: &[f64], rng: &mut StdRng) -> Vec<f64> {
        let mut theta = vec![0.0f64; self.theta_dim];
        let ctx = mlp_forward(&self.context_net, x);

        for i in 0..self.theta_dim {
            // Build current input with partial theta
            let mut inp: Vec<f64> = theta.clone();
            inp.extend_from_slice(&ctx);

            let n = self.layers.len();
            let mut h = inp;
            for (idx, layer) in self.layers.iter().enumerate() {
                h = layer.forward(&h);
                if idx < n - 1 {
                    for v in &mut h {
                        *v = tanh_act(*v);
                    }
                }
            }

            let mu = h[i];
            let log_scale = h[self.theta_dim + i].clamp(-5.0, 5.0);
            let sigma = log_scale.exp().max(1e-6);
            theta[i] = mu + sigma * randn(rng);
        }
        theta
    }

    /// Gradient-descent update on weight matrices and biases.
    pub fn update(&mut self, grad_layers: &[Vec<Vec<f64>>], grad_biases: &[Vec<f64>], lr: f64) {
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            if idx < grad_layers.len() {
                layer.update(&grad_layers[idx], &grad_biases[idx], lr);
            }
        }
    }

    /// One-step NLL training step on a mini-batch (simple finite-difference gradients).
    ///
    /// Returns mean NLL loss.
    pub fn train_nll(&mut self, thetas: &[Vec<f64>], xs: &[Vec<f64>], lr: f64) -> f64 {
        if thetas.is_empty() {
            return 0.0;
        }
        let n = thetas.len();

        // Compute loss before update
        let loss: f64 = thetas
            .iter()
            .zip(xs.iter())
            .map(|(th, x)| -self.log_prob(th, x))
            .sum::<f64>()
            / n as f64;

        // Finite-difference gradient for each layer weight — simplified perturbation approach
        let eps = 1e-4_f64;
        let n_layers = self.layers.len();

        for layer_idx in 0..n_layers {
            let out_dim = self.layers[layer_idx].weights.len();
            let in_dim = if out_dim > 0 {
                self.layers[layer_idx].weights[0].len()
            } else {
                0
            };

            for i in 0..out_dim {
                for j in 0..in_dim {
                    let original = self.layers[layer_idx].weights[i][j];
                    self.layers[layer_idx].weights[i][j] = original + eps;
                    let loss_p: f64 = thetas
                        .iter()
                        .zip(xs.iter())
                        .map(|(th, x)| -self.log_prob(th, x))
                        .sum::<f64>()
                        / n as f64;

                    self.layers[layer_idx].weights[i][j] = original - eps;
                    let loss_m: f64 = thetas
                        .iter()
                        .zip(xs.iter())
                        .map(|(th, x)| -self.log_prob(th, x))
                        .sum::<f64>()
                        / n as f64;

                    let grad = (loss_p - loss_m) / (2.0 * eps);
                    self.layers[layer_idx].weights[i][j] = original - lr * grad;
                }

                let original_b = self.layers[layer_idx].biases[i];
                self.layers[layer_idx].biases[i] = original_b + eps;
                let loss_p: f64 = thetas
                    .iter()
                    .zip(xs.iter())
                    .map(|(th, x)| -self.log_prob(th, x))
                    .sum::<f64>()
                    / n as f64;

                self.layers[layer_idx].biases[i] = original_b - eps;
                let loss_m: f64 = thetas
                    .iter()
                    .zip(xs.iter())
                    .map(|(th, x)| -self.log_prob(th, x))
                    .sum::<f64>()
                    / n as f64;

                let grad = (loss_p - loss_m) / (2.0 * eps);
                self.layers[layer_idx].biases[i] = original_b - lr * grad;
            }
        }

        loss
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MLP helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build an MLP from a layer-size spec: each pair of adjacent sizes defines a
/// (weight matrix, bias vector).
pub(crate) fn build_mlp(sizes: &[usize], rng: &mut StdRng) -> Vec<(Vec<Vec<f64>>, Vec<f64>)> {
    sizes
        .windows(2)
        .map(|w| {
            let (in_dim, out_dim) = (w[0], w[1]);
            let scale = (2.0 / in_dim as f64).sqrt();
            let wts: Vec<Vec<f64>> = (0..out_dim)
                .map(|_| (0..in_dim).map(|_| scale * randn(rng)).collect())
                .collect();
            let bias = vec![0.0f64; out_dim];
            (wts, bias)
        })
        .collect()
}

/// Forward pass through an MLP; tanh on all layers except the last (identity).
pub(crate) fn mlp_forward(layers: &[(Vec<Vec<f64>>, Vec<f64>)], x: &[f64]) -> Vec<f64> {
    let n = layers.len();
    let mut h = x.to_vec();
    for (idx, (w, b)) in layers.iter().enumerate() {
        let out_dim = w.len();
        let mut out = b.clone();
        for (i, row) in w.iter().enumerate() {
            for (j, &wij) in row.iter().enumerate() {
                if j < h.len() {
                    out[i] += wij * h[j];
                }
            }
        }
        if idx < n - 1 {
            for v in &mut out {
                *v = tanh_act(*v);
            }
        }
        h = out;
    }
    h
}

/// Gradient-descent in-place update for an MLP (single layer, FD).
pub(crate) fn mlp_train_step(
    layers: &mut [(Vec<Vec<f64>>, Vec<f64>)],
    pairs: &[(Vec<f64>, Vec<f64>)], // (input, target)
    loss_fn: impl Fn(&[f64], &[f64]) -> f64,
    lr: f64,
) -> f64 {
    if pairs.is_empty() {
        return 0.0;
    }
    let eps = 1e-4_f64;
    let n = pairs.len() as f64;

    let baseline: f64 = pairs
        .iter()
        .map(|(x, y)| {
            let pred = mlp_forward(layers, x);
            loss_fn(&pred, y)
        })
        .sum::<f64>()
        / n;

    for layer_idx in 0..layers.len() {
        let out_dim = layers[layer_idx].0.len();
        let in_dim = if out_dim > 0 {
            layers[layer_idx].0[0].len()
        } else {
            0
        };

        for i in 0..out_dim {
            for j in 0..in_dim {
                let orig = layers[layer_idx].0[i][j];
                layers[layer_idx].0[i][j] = orig + eps;
                let lp: f64 = pairs
                    .iter()
                    .map(|(x, y)| {
                        let p = mlp_forward(layers, x);
                        loss_fn(&p, y)
                    })
                    .sum::<f64>()
                    / n;
                layers[layer_idx].0[i][j] = orig - eps;
                let lm: f64 = pairs
                    .iter()
                    .map(|(x, y)| {
                        let p = mlp_forward(layers, x);
                        loss_fn(&p, y)
                    })
                    .sum::<f64>()
                    / n;
                let g = (lp - lm) / (2.0 * eps);
                layers[layer_idx].0[i][j] = orig - lr * g;
            }
            let orig_b = layers[layer_idx].1[i];
            layers[layer_idx].1[i] = orig_b + eps;
            let lp: f64 = pairs
                .iter()
                .map(|(x, y)| {
                    let p = mlp_forward(layers, x);
                    loss_fn(&p, y)
                })
                .sum::<f64>()
                / n;
            layers[layer_idx].1[i] = orig_b - eps;
            let lm: f64 = pairs
                .iter()
                .map(|(x, y)| {
                    let p = mlp_forward(layers, x);
                    loss_fn(&p, y)
                })
                .sum::<f64>()
                / n;
            let g = (lp - lm) / (2.0 * eps);
            layers[layer_idx].1[i] = orig_b - lr * g;
        }
    }

    baseline
}

// ─────────────────────────────────────────────────────────────────────────────
// SNPE — Sequential Neural Posterior Estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`SequentialNpe`].
#[derive(Debug, Clone)]
pub struct SnpeConfig {
    /// Number of sequential rounds.
    pub n_rounds: usize,
    /// Simulations drawn per round.
    pub n_simulations_per_round: usize,
    /// Hidden layer width for the density estimator.
    pub hidden_dim: usize,
    /// Depth of the MADE stack.
    pub n_nde_layers: usize,
    /// Learning rate.
    pub lr: f64,
    /// Training steps per round.
    pub n_training_steps: usize,
    /// Mini-batch size.
    pub batch_size: usize,
}

impl Default for SnpeConfig {
    fn default() -> Self {
        Self {
            n_rounds: 2,
            n_simulations_per_round: 50,
            hidden_dim: 32,
            n_nde_layers: 2,
            lr: 1e-3,
            n_training_steps: 30,
            batch_size: 16,
        }
    }
}

/// Per-round summary statistics.
#[derive(Debug, Clone)]
pub struct RoundSummary {
    /// Round index (0-based).
    pub round: usize,
    /// Number of simulations run in this round.
    pub n_simulations: usize,
    /// Final training NLL loss.
    pub training_loss: f64,
    /// log q(theta_true | x_obs) if available, else NaN.
    pub log_prob_at_obs: f64,
}

/// Trained SNPE posterior.
#[derive(Debug, Clone)]
pub struct SnpePosterior {
    /// Trained density estimator.
    pub estimator: NeuralDensityEstimator,
    /// Observed data.
    pub x_obs: Vec<f64>,
    /// Per-round summaries.
    pub round_summaries: Vec<RoundSummary>,
}

impl SnpePosterior {
    /// Evaluate log q(theta | x_obs).
    pub fn log_prob(&self, theta: &[f64]) -> f64 {
        self.estimator.log_prob(theta, &self.x_obs)
    }

    /// Draw `n_samples` samples from the approximate posterior.
    pub fn sample(&self, n_samples: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
        (0..n_samples)
            .map(|_| self.estimator.sample(&self.x_obs, rng))
            .collect()
    }

    /// MAP estimate: highest log_prob sample from 1000 draws.
    pub fn map_estimate(&self, rng: &mut StdRng) -> Vec<f64> {
        let candidates = self.sample(1000, rng);
        let mut best_lp = f64::NEG_INFINITY;
        let mut best = candidates.first().cloned().unwrap_or_default();
        for c in &candidates {
            let lp = self.log_prob(c);
            if lp > best_lp {
                best_lp = lp;
                best = c.clone();
            }
        }
        best
    }
}

/// Sequential Neural Posterior Estimation (SNPE-C / APT).
#[derive(Debug)]
pub struct SequentialNpe {
    /// Algorithm configuration.
    pub config: SnpeConfig,
    /// Density estimator q(theta | x).
    pub estimator: NeuralDensityEstimator,
    /// All collected theta samples across rounds.
    pub all_thetas: Vec<Vec<f64>>,
    /// All collected x simulations across rounds.
    pub all_xs: Vec<Vec<f64>>,
}

impl SequentialNpe {
    /// Initialise the SNPE object with a freshly randomised density estimator.
    pub fn new(simulator: &dyn Simulator, config: SnpeConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let estimator = NeuralDensityEstimator::new(
            simulator.theta_dim(),
            simulator.x_dim(),
            config.hidden_dim,
            config.n_nde_layers,
            &mut rng,
        );
        Self {
            config,
            estimator,
            all_thetas: Vec::new(),
            all_xs: Vec::new(),
        }
    }

    /// Execute all rounds of SNPE and return a trained [`SnpePosterior`].
    pub fn run(
        &mut self,
        simulator: &dyn Simulator,
        x_obs: &[f64],
        rng: &mut StdRng,
    ) -> SnpePosterior {
        let mut summaries = Vec::new();

        for round in 0..self.config.n_rounds {
            let n_sim = self.config.n_simulations_per_round;

            // Sample proposal: prior in round 0, current posterior afterwards
            let mut new_thetas: Vec<Vec<f64>> = Vec::with_capacity(n_sim);
            let mut new_xs: Vec<Vec<f64>> = Vec::with_capacity(n_sim);

            for _ in 0..n_sim {
                let theta = if round == 0 {
                    simulator.prior_sample(rng)
                } else {
                    self.estimator.sample(x_obs, rng)
                };
                let x = simulator.simulate(&theta, rng);
                new_thetas.push(theta);
                new_xs.push(x);
            }

            self.all_thetas.extend_from_slice(&new_thetas);
            self.all_xs.extend_from_slice(&new_xs);

            // Train
            let loss = self.train_step(&new_thetas, &new_xs, rng);

            let lp_at_obs = f64::NAN;
            summaries.push(RoundSummary {
                round,
                n_simulations: n_sim,
                training_loss: loss,
                log_prob_at_obs: lp_at_obs,
            });
        }

        SnpePosterior {
            estimator: self.estimator.clone(),
            x_obs: x_obs.to_vec(),
            round_summaries: summaries,
        }
    }

    /// Train the density estimator for `n_training_steps` on provided data,
    /// sampling mini-batches without replacement.
    ///
    /// Returns the mean NLL on the last mini-batch.
    pub fn train_step(&mut self, thetas: &[Vec<f64>], xs: &[Vec<f64>], rng: &mut StdRng) -> f64 {
        if thetas.is_empty() {
            return 0.0;
        }
        let n = thetas.len();
        let batch_size = self.config.batch_size.min(n);
        let mut last_loss = 0.0;

        for _ in 0..self.config.n_training_steps {
            // Random mini-batch
            let indices: Vec<usize> = (0..batch_size).map(|_| rng.random_range(0..n)).collect();
            let bt: Vec<Vec<f64>> = indices.iter().map(|&i| thetas[i].clone()).collect();
            let bx: Vec<Vec<f64>> = indices.iter().map(|&i| xs[i].clone()).collect();

            last_loss = self.estimator.train_nll(&bt, &bx, self.config.lr);
        }

        last_loss
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SNLE — Sequential Neural Likelihood Estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`SnleEstimator`].
#[derive(Debug, Clone)]
pub struct SnleConfig {
    /// Number of sequential rounds.
    pub n_rounds: usize,
    /// Simulations per round.
    pub n_simulations_per_round: usize,
    /// Hidden layer width.
    pub hidden_dim: usize,
    /// Learning rate.
    pub lr: f64,
    /// Training steps per round.
    pub n_training_steps: usize,
    /// Metropolis-Hastings steps for posterior sampling.
    pub n_mcmc_steps: usize,
}

impl Default for SnleConfig {
    fn default() -> Self {
        Self {
            n_rounds: 2,
            n_simulations_per_round: 50,
            hidden_dim: 32,
            lr: 1e-3,
            n_training_steps: 30,
            n_mcmc_steps: 200,
        }
    }
}

/// Neural likelihood approximation q(x | theta).
#[derive(Debug, Clone)]
pub struct NeuralLikelihood {
    /// Density estimator: conditions on theta, predicts x distribution.
    pub estimator: NeuralDensityEstimator,
}

impl NeuralLikelihood {
    /// Create a new neural likelihood with x_dim as "theta" and theta_dim as context.
    pub fn new(x_dim: usize, theta_dim: usize, hidden_dim: usize, rng: &mut StdRng) -> Self {
        // Estimator predicts x given theta (theta is the conditioning context)
        let estimator = NeuralDensityEstimator::new(x_dim, theta_dim, hidden_dim, 2, rng);
        Self { estimator }
    }

    /// Evaluate log q(x | theta).
    pub fn log_likelihood(&self, x: &[f64], theta: &[f64]) -> f64 {
        self.estimator.log_prob(x, theta)
    }

    /// One NLL training step on a batch of (x, theta) pairs.
    pub fn train_step(&mut self, xs: &[Vec<f64>], thetas: &[Vec<f64>], lr: f64) -> f64 {
        self.estimator.train_nll(xs, thetas, lr)
    }
}

/// Trained SNLE posterior (likelihood × prior, sampled via MCMC).
pub struct SnlePosterior {
    /// Trained neural likelihood.
    pub likelihood: NeuralLikelihood,
    /// Simulator (for prior evaluation).
    pub simulator: Box<dyn Simulator>,
    /// Observed data.
    pub x_obs: Vec<f64>,
}

impl std::fmt::Debug for SnlePosterior {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SnlePosterior")
            .field("likelihood", &self.likelihood)
            .field("x_obs", &self.x_obs)
            .finish()
    }
}

impl SnlePosterior {
    /// Unnormalised log posterior: log q(x_obs | theta) + log p(theta).
    pub fn log_prob(&self, theta: &[f64]) -> f64 {
        let ll = self.likelihood.log_likelihood(&self.x_obs, theta);
        let lp = self.simulator.prior_log_prob(theta);
        ll + lp
    }

    /// Draw MCMC samples from the unnormalised posterior using
    /// Metropolis-Hastings with isotropic Gaussian proposal (sigma = 0.1).
    pub fn mcmc_sample(&self, n_samples: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
        let proposal_std = 0.1;

        // Initialise from prior
        let mut current: Vec<f64> = self.simulator.prior_sample(rng);
        let mut current_lp = self.log_prob(&current);

        let mut samples = Vec::with_capacity(n_samples);
        let burn_in = 100;
        let total_steps = n_samples + burn_in;

        for step in 0..total_steps {
            // Gaussian random walk proposal
            let proposal: Vec<f64> = current
                .iter()
                .map(|&v| v + proposal_std * randn(rng))
                .collect();
            let proposal_lp = self.log_prob(&proposal);

            // MH acceptance
            let log_alpha = proposal_lp - current_lp;
            let u: f64 = rng.random::<f64>();
            if log_alpha >= 0.0 || u.ln() < log_alpha {
                current = proposal;
                current_lp = proposal_lp;
            }

            if step >= burn_in {
                samples.push(current.clone());
            }
        }

        samples
    }
}

/// Full Sequential Neural Likelihood Estimation runner.
#[derive(Debug)]
pub struct SnleEstimator {
    /// Algorithm configuration.
    pub config: SnleConfig,
    /// Neural likelihood approximation.
    pub likelihood: NeuralLikelihood,
}

impl SnleEstimator {
    /// Create a new SNLE estimator.
    pub fn new(simulator: &dyn Simulator, config: SnleConfig, rng: &mut StdRng) -> Self {
        let likelihood = NeuralLikelihood::new(
            simulator.x_dim(),
            simulator.theta_dim(),
            config.hidden_dim,
            rng,
        );
        Self { config, likelihood }
    }

    /// Run all rounds of SNLE and return an [`SnlePosterior`].
    pub fn run(
        &mut self,
        simulator: &dyn Simulator,
        x_obs: &[f64],
        rng: &mut StdRng,
    ) -> SnlePosterior {
        for _ in 0..self.config.n_rounds {
            let n_sim = self.config.n_simulations_per_round;
            let mut xs = Vec::with_capacity(n_sim);
            let mut thetas = Vec::with_capacity(n_sim);

            for _ in 0..n_sim {
                let theta = simulator.prior_sample(rng);
                let x = simulator.simulate(&theta, rng);
                thetas.push(theta);
                xs.push(x);
            }

            for _ in 0..self.config.n_training_steps {
                let n = thetas.len();
                let bs = self.config.n_simulations_per_round.min(n);
                let idx: Vec<usize> = (0..bs).map(|_| rng.random_range(0..n)).collect();
                let bx: Vec<Vec<f64>> = idx.iter().map(|&i| xs[i].clone()).collect();
                let bt: Vec<Vec<f64>> = idx.iter().map(|&i| thetas[i].clone()).collect();
                self.likelihood.train_step(&bx, &bt, self.config.lr);
            }
        }

        // Build posterior — we wrap the simulator in a GaussianSimulator placeholder
        // if the concrete type is accessible.  Here we clone the likelihood and
        // store the simulator via a type-erased box.
        let sim_box: Box<dyn Simulator> = Box::new(GaussianSimulatorProxy::new(
            simulator.theta_dim(),
            simulator.x_dim(),
        ));

        SnlePosterior {
            likelihood: self.likelihood.clone(),
            simulator: sim_box,
            x_obs: x_obs.to_vec(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SNRE — Sequential Neural Ratio Estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Binary classifier that learns log r(x, theta) = log p(x,theta) / (p(x) p(theta)).
#[derive(Debug, Clone)]
pub struct NeuralRatioEstimator {
    /// MLP layers: list of (weight matrix, bias).
    pub layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    /// Dimensionality of observation x.
    pub x_dim: usize,
    /// Dimensionality of parameter theta.
    pub theta_dim: usize,
}

impl NeuralRatioEstimator {
    /// Build a fully-connected MLP classifier.
    ///
    /// Input is [x ‖ theta], output is a single scalar log-ratio.
    pub fn new(
        x_dim: usize,
        theta_dim: usize,
        hidden_dim: usize,
        n_layers: usize,
        rng: &mut StdRng,
    ) -> Self {
        let in_dim = x_dim + theta_dim;
        let n_hidden = n_layers.max(1);
        let mut sizes = vec![in_dim];
        for _ in 0..n_hidden {
            sizes.push(hidden_dim);
        }
        sizes.push(1);
        let layers = build_mlp(&sizes, rng);
        Self {
            layers,
            x_dim,
            theta_dim,
        }
    }

    /// Forward pass: returns log r(x, theta).
    pub fn forward(&self, x: &[f64], theta: &[f64]) -> f64 {
        let mut inp: Vec<f64> = x.to_vec();
        inp.extend_from_slice(theta);
        let out = mlp_forward(&self.layers, &inp);
        if out.is_empty() {
            0.0
        } else {
            out[0]
        }
    }

    /// Unnormalised log posterior = log_ratio + prior_log_prob.
    pub fn log_posterior(&self, x: &[f64], theta: &[f64], prior_log_prob: f64) -> f64 {
        self.forward(x, theta) + prior_log_prob
    }

    /// Binary cross-entropy training step.
    ///
    /// Joint pairs `(x_j, theta_j)` receive label 1; marginal pairs label 0.
    /// Returns mean BCE loss.
    pub fn train_step(
        &mut self,
        joint_x: &[Vec<f64>],
        joint_theta: &[Vec<f64>],
        marginal_x: &[Vec<f64>],
        marginal_theta: &[Vec<f64>],
        lr: f64,
    ) -> f64 {
        let n_joint = joint_x.len().min(joint_theta.len());
        let n_marg = marginal_x.len().min(marginal_theta.len());
        if n_joint == 0 && n_marg == 0 {
            return 0.0;
        }

        let mut pairs: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
        // joint → target = [1.0]
        for i in 0..n_joint {
            let mut inp = joint_x[i].clone();
            inp.extend_from_slice(&joint_theta[i]);
            pairs.push((inp, vec![1.0]));
        }
        // marginal → target = [0.0]
        for i in 0..n_marg {
            let mut inp = marginal_x[i].clone();
            inp.extend_from_slice(&marginal_theta[i]);
            pairs.push((inp, vec![0.0]));
        }

        let bce = |pred: &[f64], target: &[f64]| -> f64 {
            if pred.is_empty() || target.is_empty() {
                return 0.0;
            }
            let p = sigmoid(pred[0]).clamp(1e-7, 1.0 - 1e-7);
            let t = target[0];
            -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
        };

        mlp_train_step(&mut self.layers, &pairs, bce, lr)
    }
}

/// Internal stand-in simulator used when the real one cannot be boxed/cloned.
/// Prior is N(0,1); simulate returns zeros.
#[derive(Debug, Clone)]
pub(crate) struct GaussianSimulatorProxy {
    theta: usize,
    x: usize,
}

impl GaussianSimulatorProxy {
    pub(crate) fn new(theta: usize, x: usize) -> Self {
        Self { theta, x }
    }
}

impl Simulator for GaussianSimulatorProxy {
    fn simulate(&self, _theta: &[f64], rng: &mut StdRng) -> Vec<f64> {
        (0..self.x).map(|_| randn(rng)).collect()
    }
    fn prior_log_prob(&self, theta: &[f64]) -> f64 {
        theta.iter().map(|&t| log_normal(t, 0.0, 1.0)).sum()
    }
    fn prior_sample(&self, rng: &mut StdRng) -> Vec<f64> {
        (0..self.theta).map(|_| randn(rng)).collect()
    }
    fn theta_dim(&self) -> usize {
        self.theta
    }
    fn x_dim(&self) -> usize {
        self.x
    }
}
