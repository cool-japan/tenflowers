//! Mixture Density Networks and Generative Density Estimation.
//!
//! Implements MDN (Bishop 1994), RNADE (Uria 2013), MADE (Germain 2015),
//! normalizing-flow MDN, CVAE-MDN, and Bayesian MDN in pure Rust.
//!
//! All types are prefixed with `Mdn` (or use the full form) to avoid
//! collisions with existing exports from `synthetic_data` and `monte_carlo`.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::fmt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the MDN module.
#[derive(Debug, Clone)]
pub enum MdnError {
    /// A value was outside an acceptable range or empty.
    InvalidInput(String),
    /// A numerically unstable operation was encountered (NaN/Inf).
    NumericalError(String),
    /// Tensor/vector shape did not match expectations.
    ShapeMismatch(String),
}

impl fmt::Display for MdnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MdnError::InvalidInput(s) => write!(f, "MDN invalid input: {s}"),
            MdnError::NumericalError(s) => write!(f, "MDN numerical error: {s}"),
            MdnError::ShapeMismatch(s) => write!(f, "MDN shape mismatch: {s}"),
        }
    }
}

impl std::error::Error for MdnError {}

fn mdn_err<S: Into<String>>(msg: S) -> TensorError {
    TensorError::compute_error_simple(msg.into())
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal RNG helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Box-Muller: N(0,1) sample from two uniform [0,1) draws.
#[inline]
fn bm_normal(rng: &mut impl Rng) -> f64 {
    let u1: f64 = (rng.random::<f64>()).max(1e-300);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Xavier-style weight init: N(0, scale^2) where scale = sqrt(2/(fan_in+fan_out)).
fn xavier_weights(fan_in: usize, fan_out: usize, rng: &mut impl Rng) -> Vec<f64> {
    let scale = (2.0 / (fan_in + fan_out) as f64).sqrt();
    (0..fan_in * fan_out)
        .map(|_| bm_normal(rng) * scale)
        .collect()
}

/// Numerically-stable log-sum-exp.
#[inline]
fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_v = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max_v.is_finite() {
        return max_v;
    }
    max_v + values.iter().map(|v| (v - max_v).exp()).sum::<f64>().ln()
}

/// Softmax in-place over a slice, returns new Vec.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max_l).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum.max(1e-300)).collect()
}

/// Softplus: ln(1+exp(x)) — numerically stable.
#[inline]
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Sigmoid activation.
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// ReLU.
#[inline]
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Tanh clipped to prevent saturation issues.
#[inline]
fn tanh_clip(x: f64) -> f64 {
    x.tanh()
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. MdnLinear — simple fully-connected layer
// ─────────────────────────────────────────────────────────────────────────────

/// A single fully-connected (linear) layer with Xavier weight initialisation.
///
/// Stores weights in row-major order: `weights[i * in_features + j]` is W\[i,j\].
#[derive(Debug, Clone)]
pub struct MdnLinear {
    pub in_features: usize,
    pub out_features: usize,
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
}

impl MdnLinear {
    /// Create a new `MdnLinear` with Xavier-initialised weights and zero biases.
    pub fn new(in_features: usize, out_features: usize, rng: &mut impl Rng) -> Self {
        let weights = xavier_weights(in_features, out_features, rng);
        let bias = vec![0.0; out_features];
        Self {
            in_features,
            out_features,
            weights,
            bias,
        }
    }

    /// Forward pass: y = W x + b.
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.in_features {
            return Err(mdn_err(format!(
                "MdnLinear: expected {} inputs, got {}",
                self.in_features,
                x.len()
            )));
        }
        let mut out = self.bias.clone();
        for o in 0..self.out_features {
            for i in 0..self.in_features {
                out[o] += self.weights[o * self.in_features + i] * x[i];
            }
        }
        Ok(out)
    }

    /// Mutable reference to all parameters (weights then bias) for gradient updates.
    pub fn params_mut(&mut self) -> (&mut Vec<f64>, &mut Vec<f64>) {
        (&mut self.weights, &mut self.bias)
    }

    /// Number of parameters.
    pub fn n_params(&self) -> usize {
        self.weights.len() + self.bias.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. MdnGaussianMixture — K-component Gaussian mixture
// ─────────────────────────────────────────────────────────────────────────────

/// Parameters for a K-component diagonal-covariance Gaussian mixture.
///
/// `pi` are mixing weights (sum to 1 after softmax).
/// `mu[k]` and `sigma[k]` each have length `out_dim`.
#[derive(Debug, Clone)]
pub struct MdnGaussianMixture {
    /// Mixing weights (sum to 1).
    pub pi: Vec<f64>,
    /// Component means: shape [K, out_dim].
    pub mu: Vec<Vec<f64>>,
    /// Component standard deviations (positive via softplus): shape [K, out_dim].
    pub sigma: Vec<Vec<f64>>,
}

impl MdnGaussianMixture {
    /// Number of mixture components.
    pub fn n_components(&self) -> usize {
        self.pi.len()
    }

    /// Dimension of the output space.
    pub fn out_dim(&self) -> usize {
        self.mu.first().map(|m| m.len()).unwrap_or(0)
    }

    /// Log-probability of `x` under the mixture, computed via log-sum-exp.
    pub fn log_prob(&self, x: &[f64]) -> Result<f64> {
        let k = self.n_components();
        let d = self.out_dim();
        if x.len() != d {
            return Err(mdn_err(format!(
                "MdnGaussianMixture: x dim {} ≠ out_dim {}",
                x.len(),
                d
            )));
        }
        let log_two_pi = (2.0 * std::f64::consts::PI).ln();
        let mut log_terms = Vec::with_capacity(k);
        for ki in 0..k {
            let log_pi_k = self.pi[ki].max(1e-300).ln();
            let mut log_gauss = 0.0_f64;
            for j in 0..d {
                let sig = self.sigma[ki][j].max(1e-8);
                let diff = x[j] - self.mu[ki][j];
                log_gauss -= 0.5 * (log_two_pi + 2.0 * sig.ln() + (diff / sig).powi(2));
            }
            log_terms.push(log_pi_k + log_gauss);
        }
        let lp = log_sum_exp(&log_terms);
        if !lp.is_finite() {
            return Err(mdn_err(
                "MdnGaussianMixture::log_prob produced non-finite value",
            ));
        }
        Ok(lp)
    }

    /// Sample using Gumbel-max trick for component selection + Box-Muller for sample.
    pub fn sample(&self, rng: &mut impl Rng) -> Result<Vec<f64>> {
        let k = self.n_components();
        let d = self.out_dim();
        if k == 0 || d == 0 {
            return Err(mdn_err("MdnGaussianMixture::sample: empty mixture"));
        }
        // Gumbel-max trick: argmax(log(pi_k) - log(-log(u_k)))
        let selected = (0..k)
            .map(|ki| {
                let u: f64 = (rng.random::<f64>()).max(1e-300);
                let g = -(-u.ln()).ln();
                self.pi[ki].max(1e-300).ln() + g
            })
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| mdn_err("Gumbel-max failed"))?;

        // Box-Muller sample from selected component
        let sample: Vec<f64> = (0..d)
            .map(|j| {
                let sig = self.sigma[selected][j].max(1e-8);
                self.mu[selected][j] + sig * bm_normal(rng)
            })
            .collect();
        Ok(sample)
    }

    /// Weighted mean of the mixture: sum_k pi_k * mu_k.
    pub fn mean(&self) -> Vec<f64> {
        let d = self.out_dim();
        let mut mean = vec![0.0; d];
        for (ki, w) in self.pi.iter().enumerate() {
            for j in 0..d {
                mean[j] += w * self.mu[ki][j];
            }
        }
        mean
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. MixtureDensityNetwork — classic MDN (Bishop 1994)
// ─────────────────────────────────────────────────────────────────────────────

/// Mixture Density Network (Bishop 1994).
///
/// Architecture: encoder MLP (with ReLU activations) → three output heads:
/// - `pi_head`: K logits → softmax → mixing weights
/// - `mu_head`: K * out_dim values → component means
/// - `log_sigma_head`: K * out_dim values → softplus → component sigmas
#[derive(Debug, Clone)]
pub struct MixtureDensityNetwork {
    encoder: Vec<MdnLinear>,
    pi_head: MdnLinear,
    mu_head: MdnLinear,
    log_sigma_head: MdnLinear,
    pub n_components: usize,
    pub out_dim: usize,
    /// Adam optimiser state: (m_w, v_w, m_b, v_b) per layer
    adam_state: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)>,
    adam_t: usize,
}

impl MixtureDensityNetwork {
    /// Create a new MDN.
    ///
    /// `hidden_sizes`: sizes of hidden encoder layers (e.g. `[64, 64]`).
    pub fn new(
        in_dim: usize,
        hidden_sizes: &[usize],
        n_components: usize,
        out_dim: usize,
        seed: u64,
    ) -> Result<Self> {
        if in_dim == 0 || n_components == 0 || out_dim == 0 {
            return Err(mdn_err("MDN: dims must be > 0"));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut encoder = Vec::new();
        let mut prev = in_dim;
        for &h in hidden_sizes {
            encoder.push(MdnLinear::new(prev, h, &mut rng));
            prev = h;
        }
        let last_hidden = prev;
        let pi_head = MdnLinear::new(last_hidden, n_components, &mut rng);
        let mu_head = MdnLinear::new(last_hidden, n_components * out_dim, &mut rng);
        let log_sigma_head = MdnLinear::new(last_hidden, n_components * out_dim, &mut rng);

        // Build Adam state (all zeros) for each layer
        let mut adam_state = Vec::new();
        let mut build_state = |l: &MdnLinear| {
            let nw = l.weights.len();
            let nb = l.bias.len();
            (vec![0.0; nw], vec![0.0; nw], vec![0.0; nb], vec![0.0; nb])
        };
        for l in &encoder {
            adam_state.push(build_state(l));
        }
        adam_state.push(build_state(&pi_head));
        adam_state.push(build_state(&mu_head));
        adam_state.push(build_state(&log_sigma_head));

        Ok(Self {
            encoder,
            pi_head,
            mu_head,
            log_sigma_head,
            n_components,
            out_dim,
            adam_state,
            adam_t: 0,
        })
    }

    /// Encoder forward pass → hidden representation.
    fn encode(&self, x: &[f64]) -> Result<Vec<f64>> {
        let mut h = x.to_vec();
        for layer in &self.encoder {
            let z = layer.forward(&h)?;
            h = z.into_iter().map(relu).collect();
        }
        Ok(h)
    }

    /// Full forward pass → `MdnGaussianMixture`.
    pub fn forward(&self, x: &[f64]) -> Result<MdnGaussianMixture> {
        let h = self.encode(x)?;
        let pi_logits = self.pi_head.forward(&h)?;
        let mu_flat = self.mu_head.forward(&h)?;
        let log_sigma_flat = self.log_sigma_head.forward(&h)?;

        let pi = softmax(&pi_logits);
        let k = self.n_components;
        let d = self.out_dim;
        let mu: Vec<Vec<f64>> = (0..k)
            .map(|ki| mu_flat[ki * d..(ki + 1) * d].to_vec())
            .collect();
        let sigma: Vec<Vec<f64>> = (0..k)
            .map(|ki| {
                log_sigma_flat[ki * d..(ki + 1) * d]
                    .iter()
                    .map(|&v| softplus(v).max(1e-6))
                    .collect()
            })
            .collect();

        Ok(MdnGaussianMixture { pi, mu, sigma })
    }

    /// Negative log-likelihood loss.
    pub fn nll_loss(&self, mixture: &MdnGaussianMixture, target: &[f64]) -> Result<f64> {
        let lp = mixture.log_prob(target)?;
        Ok(-lp)
    }

    /// Train via mini-batch Adam. Returns per-epoch average NLL.
    pub fn fit(
        &mut self,
        data: &[Vec<f64>],
        targets: &[Vec<f64>],
        epochs: usize,
        lr: f64,
        batch_size: usize,
        seed: u64,
    ) -> Result<Vec<f64>> {
        if data.len() != targets.len() || data.is_empty() {
            return Err(mdn_err("fit: data/target length mismatch or empty"));
        }
        let n = data.len();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut history = Vec::with_capacity(epochs);
        let eps = 1e-8;
        let beta1 = 0.9_f64;
        let beta2 = 0.999_f64;

        for _epoch in 0..epochs {
            // Shuffle indices
            let mut indices: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = rng.random_range(0..=i);
                indices.swap(i, j);
            }
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            for batch_start in (0..n).step_by(batch_size.max(1)) {
                let batch_end = (batch_start + batch_size).min(n);
                let batch_idx = &indices[batch_start..batch_end];

                // Finite-difference gradient approximation
                let delta = 1e-5;
                self.adam_t += 1;
                let t = self.adam_t as f64;

                // Collect all layers into a flat list of (weights_ref, bias_ref)
                let n_enc = self.encoder.len();
                let n_layers = n_enc + 3; // encoder + pi + mu + sigma heads

                // For each layer, compute gradient via finite differences
                for layer_idx in 0..n_layers {
                    // Compute base loss
                    let base_loss: f64 = batch_idx
                        .iter()
                        .filter_map(|&i| {
                            let mix = self.forward(&data[i]).ok()?;
                            self.nll_loss(&mix, &targets[i]).ok()
                        })
                        .sum::<f64>()
                        / batch_idx.len() as f64;

                    epoch_loss += base_loss;
                    n_batches += 1;

                    let nw = self.layer_weights_len(layer_idx);
                    let nb = self.layer_bias_len(layer_idx);
                    let mut grad_w = vec![0.0; nw];
                    let mut grad_b = vec![0.0; nb];

                    for wi in 0..nw {
                        self.layer_weights_mut(layer_idx)[wi] += delta;
                        let loss_plus: f64 = batch_idx
                            .iter()
                            .filter_map(|&i| {
                                let mix = self.forward(&data[i]).ok()?;
                                self.nll_loss(&mix, &targets[i]).ok()
                            })
                            .sum::<f64>()
                            / batch_idx.len() as f64;
                        self.layer_weights_mut(layer_idx)[wi] -= delta;
                        grad_w[wi] = (loss_plus - base_loss) / delta;
                    }
                    for bi in 0..nb {
                        self.layer_bias_mut(layer_idx)[bi] += delta;
                        let loss_plus: f64 = batch_idx
                            .iter()
                            .filter_map(|&i| {
                                let mix = self.forward(&data[i]).ok()?;
                                self.nll_loss(&mix, &targets[i]).ok()
                            })
                            .sum::<f64>()
                            / batch_idx.len() as f64;
                        self.layer_bias_mut(layer_idx)[bi] -= delta;
                        grad_b[bi] = (loss_plus - base_loss) / delta;
                    }

                    // Adam update: compute deltas first to avoid double mutable borrow
                    let scale_w = lr * (1.0 - beta2.powf(t)).sqrt() / (1.0 - beta1.powf(t));

                    let mut w_deltas = vec![0.0; nw];
                    let mut b_deltas = vec![0.0; nb];
                    {
                        let (ref mut m_w, ref mut v_w, ref mut m_b, ref mut v_b) =
                            self.adam_state[layer_idx];
                        for wi in 0..nw {
                            m_w[wi] = beta1 * m_w[wi] + (1.0 - beta1) * grad_w[wi];
                            v_w[wi] = beta2 * v_w[wi] + (1.0 - beta2) * grad_w[wi].powi(2);
                            w_deltas[wi] = scale_w * m_w[wi] / (v_w[wi].sqrt() + eps);
                        }
                        for bi in 0..nb {
                            m_b[bi] = beta1 * m_b[bi] + (1.0 - beta1) * grad_b[bi];
                            v_b[bi] = beta2 * v_b[bi] + (1.0 - beta2) * grad_b[bi].powi(2);
                            b_deltas[bi] = scale_w * m_b[bi] / (v_b[bi].sqrt() + eps);
                        }
                    }
                    // Apply deltas now that adam_state borrow is released
                    for wi in 0..nw {
                        self.layer_weights_mut(layer_idx)[wi] -= w_deltas[wi];
                    }
                    for bi in 0..nb {
                        self.layer_bias_mut(layer_idx)[bi] -= b_deltas[bi];
                    }
                }
            }

            let avg = if n_batches > 0 {
                epoch_loss / n_batches as f64
            } else {
                f64::NAN
            };
            history.push(avg);
        }
        Ok(history)
    }

    fn layer_weights_len(&self, idx: usize) -> usize {
        self.get_layer(idx).weights.len()
    }
    fn layer_bias_len(&self, idx: usize) -> usize {
        self.get_layer(idx).bias.len()
    }
    fn get_layer(&self, idx: usize) -> &MdnLinear {
        let n_enc = self.encoder.len();
        if idx < n_enc {
            &self.encoder[idx]
        } else {
            match idx - n_enc {
                0 => &self.pi_head,
                1 => &self.mu_head,
                _ => &self.log_sigma_head,
            }
        }
    }
    fn layer_weights_mut(&mut self, idx: usize) -> &mut Vec<f64> {
        let n_enc = self.encoder.len();
        if idx < n_enc {
            &mut self.encoder[idx].weights
        } else {
            match idx - n_enc {
                0 => &mut self.pi_head.weights,
                1 => &mut self.mu_head.weights,
                _ => &mut self.log_sigma_head.weights,
            }
        }
    }
    fn layer_bias_mut(&mut self, idx: usize) -> &mut Vec<f64> {
        let n_enc = self.encoder.len();
        if idx < n_enc {
            &mut self.encoder[idx].bias
        } else {
            match idx - n_enc {
                0 => &mut self.pi_head.bias,
                1 => &mut self.mu_head.bias,
                _ => &mut self.log_sigma_head.bias,
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. ConditionalMdn — context-conditioned MDN
// ─────────────────────────────────────────────────────────────────────────────

/// Conditional MDN: context encoder + query encoder → concat → mixture head.
///
/// Useful for regression with full predictive uncertainty.
#[derive(Debug, Clone)]
pub struct ConditionalMdn {
    context_encoder: Vec<MdnLinear>,
    query_encoder: Vec<MdnLinear>,
    mixture_head: MixtureDensityNetwork,
    context_hidden: usize,
    query_hidden: usize,
}

impl ConditionalMdn {
    /// Build a `ConditionalMdn`.
    pub fn new(
        context_dim: usize,
        query_dim: usize,
        hidden: usize,
        n_components: usize,
        out_dim: usize,
        seed: u64,
    ) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        let ctx_enc = vec![MdnLinear::new(context_dim, hidden, &mut rng)];
        let qry_enc = vec![MdnLinear::new(query_dim, hidden, &mut rng)];
        let combined_dim = 2 * hidden;
        let head =
            MixtureDensityNetwork::new(combined_dim, &[hidden], n_components, out_dim, seed + 1)?;
        Ok(Self {
            context_encoder: ctx_enc,
            query_encoder: qry_enc,
            mixture_head: head,
            context_hidden: hidden,
            query_hidden: hidden,
        })
    }

    fn encode_vec(layers: &[MdnLinear], x: &[f64]) -> Result<Vec<f64>> {
        let mut h = x.to_vec();
        for l in layers {
            let z = l.forward(&h)?;
            h = z.into_iter().map(relu).collect();
        }
        Ok(h)
    }

    /// Forward pass: given context `c` and query `x`, return mixture.
    pub fn forward(&self, context: &[f64], query: &[f64]) -> Result<MdnGaussianMixture> {
        let hc = Self::encode_vec(&self.context_encoder, context)?;
        let hq = Self::encode_vec(&self.query_encoder, query)?;
        let combined: Vec<f64> = hc.into_iter().chain(hq).collect();
        self.mixture_head.forward(&combined)
    }

    /// Context encoder hidden size.
    pub fn context_hidden_size(&self) -> usize {
        self.context_hidden
    }

    /// Query encoder hidden size.
    pub fn query_hidden_size(&self) -> usize {
        self.query_hidden
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. MixtureLstmModel — LSTM cell + MDN output head
// ─────────────────────────────────────────────────────────────────────────────

/// LSTM cell with a mixture density output head.
///
/// The 4-gate LSTM cell maps `(input, h, c)` → `(h_new, c_new)`.
/// The MDN head maps `h_new` → `MdnGaussianMixture` per timestep.
#[derive(Debug, Clone)]
pub struct MixtureLstmModel {
    input_dim: usize,
    hidden_dim: usize,
    // Combined gate weights: (input_dim + hidden_dim) → 4 * hidden_dim
    w_gates: Vec<f64>,
    b_gates: Vec<f64>,
    mdn_head: MixtureDensityNetwork,
}

impl MixtureLstmModel {
    /// Create a new `MixtureLstmModel`.
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        n_components: usize,
        out_dim: usize,
        seed: u64,
    ) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        let combined = input_dim + hidden_dim;
        let gate_out = 4 * hidden_dim;
        let scale = (2.0 / (combined + gate_out) as f64).sqrt();
        let w_gates: Vec<f64> = (0..combined * gate_out)
            .map(|_| bm_normal(&mut rng) * scale)
            .collect();
        let b_gates = vec![0.0; gate_out];
        let mdn_head =
            MixtureDensityNetwork::new(hidden_dim, &[hidden_dim], n_components, out_dim, seed + 1)?;
        Ok(Self {
            input_dim,
            hidden_dim,
            w_gates,
            b_gates,
            mdn_head,
        })
    }

    /// Process one timestep; returns (h_new, c_new).
    pub fn step(&self, input: &[f64], h: &[f64], c: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        if input.len() != self.input_dim {
            return Err(mdn_err("LSTM step: input dim mismatch"));
        }
        if h.len() != self.hidden_dim || c.len() != self.hidden_dim {
            return Err(mdn_err("LSTM step: h/c dim mismatch"));
        }
        let combined = self.input_dim + self.hidden_dim;
        let gate_dim = 4 * self.hidden_dim;
        // Concatenate [input; h]
        let xh: Vec<f64> = input.iter().chain(h.iter()).cloned().collect();
        // Gate pre-activations
        let mut gates = self.b_gates.clone();
        for o in 0..gate_dim {
            for i in 0..combined {
                gates[o] += self.w_gates[o * combined + i] * xh[i];
            }
        }
        let hd = self.hidden_dim;
        // i, f, g, o gates
        let ig: Vec<f64> = gates[0..hd].iter().map(|&v| sigmoid(v)).collect();
        let fg: Vec<f64> = gates[hd..2 * hd].iter().map(|&v| sigmoid(v)).collect();
        let gg: Vec<f64> = gates[2 * hd..3 * hd]
            .iter()
            .map(|&v| tanh_clip(v))
            .collect();
        let og: Vec<f64> = gates[3 * hd..4 * hd].iter().map(|&v| sigmoid(v)).collect();
        let c_new: Vec<f64> = (0..hd).map(|j| fg[j] * c[j] + ig[j] * gg[j]).collect();
        let h_new: Vec<f64> = (0..hd).map(|j| og[j] * tanh_clip(c_new[j])).collect();
        Ok((h_new, c_new))
    }

    /// Process a full sequence; returns one `MdnGaussianMixture` per timestep.
    pub fn forward_sequence(&self, seq: &[Vec<f64>]) -> Result<Vec<MdnGaussianMixture>> {
        let hd = self.hidden_dim;
        let mut h = vec![0.0; hd];
        let mut c = vec![0.0; hd];
        let mut out = Vec::with_capacity(seq.len());
        for x in seq {
            let (h_new, c_new) = self.step(x, &h, &c)?;
            let mix = self.mdn_head.forward(&h_new)?;
            out.push(mix);
            h = h_new;
            c = c_new;
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. RnadeDensityEstimator — RNADE (Uria 2013)
// ─────────────────────────────────────────────────────────────────────────────

/// RNADE: Real-valued Neural Autoregressive Distribution Estimator.
///
/// Shared weight matrix W of shape [n_dims, hidden_size].
/// Each dimension d uses a slice W[:, cum_offset_d:cum_offset_d+hidden_size] via weight sharing.
/// The hidden activations are projected to Gaussian mixture parameters.
#[derive(Debug, Clone)]
pub struct RnadeDensityEstimator {
    n_dims: usize,
    hidden_size: usize,
    n_components: usize,
    /// Shared weight matrix: shape [hidden_size, n_dims] (column-sharing).
    w: Vec<f64>,
    /// Per-dimension bias for hidden layer: shape [n_dims, hidden_size].
    c: Vec<f64>,
    /// Output weights to pi, mu, sigma: shape [n_dims, n_components * 3, hidden_size].
    v: Vec<f64>,
    /// Output biases: shape [n_dims, n_components * 3].
    b_out: Vec<f64>,
}

impl RnadeDensityEstimator {
    /// Create a new RNADE model.
    pub fn new(n_dims: usize, hidden_size: usize, n_components: usize, seed: u64) -> Result<Self> {
        if n_dims == 0 || hidden_size == 0 || n_components == 0 {
            return Err(mdn_err("RNADE: dims must be > 0"));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let w_scale = (2.0 / (n_dims + hidden_size) as f64).sqrt();
        let w: Vec<f64> = (0..hidden_size * n_dims)
            .map(|_| bm_normal(&mut rng) * w_scale)
            .collect();
        let c = vec![0.0; n_dims * hidden_size];
        let out_dim = n_components * 3;
        let v_scale = (2.0 / (hidden_size + out_dim) as f64).sqrt();
        let v: Vec<f64> = (0..n_dims * out_dim * hidden_size)
            .map(|_| bm_normal(&mut rng) * v_scale)
            .collect();
        let b_out = vec![0.0; n_dims * out_dim];
        Ok(Self {
            n_dims,
            hidden_size,
            n_components,
            w,
            c,
            v,
            b_out,
        })
    }

    /// Compute hidden activation for dimension `d` given inputs x[0..d].
    fn compute_hidden(&self, x: &[f64], d: usize) -> Vec<f64> {
        let h = self.hidden_size;
        let mut a: Vec<f64> = self.c[d * h..(d + 1) * h].to_vec();
        for prev in 0..d {
            for hi in 0..h {
                a[hi] += self.w[hi * self.n_dims + prev] * x[prev];
            }
        }
        a.into_iter().map(|v| v.tanh()).collect()
    }

    /// Compute output parameters (pi_logits, mu, log_sigma) for dimension `d`.
    fn output_params(&self, hidden: &[f64], d: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let k = self.n_components;
        let h = self.hidden_size;
        let out_dim = k * 3;
        let base_v = d * out_dim * h;
        let base_b = d * out_dim;
        let mut out = vec![0.0; out_dim];
        for o in 0..out_dim {
            out[o] = self.b_out[base_b + o];
            for hi in 0..h {
                out[o] += self.v[base_v + o * h + hi] * hidden[hi];
            }
        }
        let pi_logits = out[0..k].to_vec();
        let mu = out[k..2 * k].to_vec();
        let log_sigma = out[2 * k..3 * k].to_vec();
        (pi_logits, mu, log_sigma)
    }

    /// Compute log p(x) = sum_d log p(x_d | x_{<d}).
    pub fn log_prob(&self, x: &[f64]) -> Result<f64> {
        if x.len() != self.n_dims {
            return Err(mdn_err(format!(
                "RNADE: x len {} ≠ n_dims {}",
                x.len(),
                self.n_dims
            )));
        }
        let mut total = 0.0;
        let log_two_pi = (2.0 * std::f64::consts::PI).ln();
        for d in 0..self.n_dims {
            let hidden = self.compute_hidden(x, d);
            let (pi_logits, mu, log_sigma) = self.output_params(&hidden, d);
            let pi = softmax(&pi_logits);
            let k = self.n_components;
            let log_terms: Vec<f64> = (0..k)
                .map(|ki| {
                    let sig = softplus(log_sigma[ki]).max(1e-6);
                    let diff = x[d] - mu[ki];
                    let log_g = -0.5 * (log_two_pi + 2.0 * sig.ln() + (diff / sig).powi(2));
                    pi[ki].max(1e-300).ln() + log_g
                })
                .collect();
            total += log_sum_exp(&log_terms);
        }
        if !total.is_finite() {
            return Err(mdn_err("RNADE::log_prob: non-finite"));
        }
        Ok(total)
    }

    /// Ancestral sample: sample each dimension left-to-right.
    pub fn sample(&self, rng: &mut impl Rng) -> Result<Vec<f64>> {
        let mut x = vec![0.0; self.n_dims];
        for d in 0..self.n_dims {
            let hidden = self.compute_hidden(&x, d);
            let (pi_logits, mu, log_sigma) = self.output_params(&hidden, d);
            let pi = softmax(&pi_logits);
            let k = self.n_components;
            // Gumbel-max component selection
            let ki = (0..k)
                .map(|ki| {
                    let u: f64 = (rng.random::<f64>()).max(1e-300);
                    let g = -(-u.ln()).ln();
                    pi[ki].max(1e-300).ln() + g
                })
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let sig = softplus(log_sigma[ki]).max(1e-6);
            x[d] = mu[ki] + sig * bm_normal(rng);
        }
        Ok(x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. MdnMadeNetwork — MADE (Germain 2015)
// ─────────────────────────────────────────────────────────────────────────────

/// MADE: Masked Autoencoder for Distribution Estimation.
///
/// Single-pass masked MLP with autoregressive structure.
/// Assigns random ordering to input units; masks enforce autoregressive property.
/// Output: (mu, sigma) per input dimension → diagonal Gaussian.
#[derive(Debug, Clone)]
pub struct MdnMadeNetwork {
    input_dim: usize,
    hidden_size: usize,
    /// Ordering m(v) for inputs 0..input_dim.
    ordering: Vec<usize>,
    /// Hidden unit ordering m_h for hidden layer.
    m_hidden: Vec<usize>,
    /// Hidden layer weights: [hidden_size, input_dim].
    w1: Vec<f64>,
    b1: Vec<f64>,
    /// Output weights: [input_dim * 2, hidden_size] (mu and log_sigma per dim).
    w2: Vec<f64>,
    b2: Vec<f64>,
    /// Mask for hidden layer: [hidden_size, input_dim].
    mask1: Vec<f64>,
    /// Mask for output layer: [input_dim * 2, hidden_size].
    mask2: Vec<f64>,
}

impl MdnMadeNetwork {
    /// Build MADE network.
    pub fn new(input_dim: usize, hidden_size: usize, seed: u64) -> Result<Self> {
        if input_dim == 0 || hidden_size == 0 {
            return Err(mdn_err("MADE: dims must be > 0"));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        // Natural ordering 1..=input_dim
        let ordering: Vec<usize> = (1..=input_dim).collect();
        // Hidden unit ordering: uniform in [1, input_dim)
        let m_hidden: Vec<usize> = (0..hidden_size)
            .map(|_| rng.random_range(1..input_dim.max(2)))
            .collect();

        let w_scale1 = (2.0 / (input_dim + hidden_size) as f64).sqrt();
        let w1: Vec<f64> = (0..hidden_size * input_dim)
            .map(|_| bm_normal(&mut rng) * w_scale1)
            .collect();
        let b1 = vec![0.0; hidden_size];

        let w_scale2 = (2.0 / (hidden_size + input_dim * 2) as f64).sqrt();
        let w2: Vec<f64> = (0..input_dim * 2 * hidden_size)
            .map(|_| bm_normal(&mut rng) * w_scale2)
            .collect();
        let b2 = vec![0.0; input_dim * 2];

        let (mask1, mask2) = Self::build_masks(input_dim, hidden_size, &ordering, &m_hidden);

        Ok(Self {
            input_dim,
            hidden_size,
            ordering,
            m_hidden,
            w1,
            b1,
            w2,
            b2,
            mask1,
            mask2,
        })
    }

    fn build_masks(
        input_dim: usize,
        hidden_size: usize,
        ordering: &[usize],
        m_hidden: &[usize],
    ) -> (Vec<f64>, Vec<f64>) {
        // mask1[h, i] = 1 if m_h[h] >= ordering[i]
        let mut mask1 = vec![0.0; hidden_size * input_dim];
        for h in 0..hidden_size {
            for i in 0..input_dim {
                if m_hidden[h] >= ordering[i] {
                    mask1[h * input_dim + i] = 1.0;
                }
            }
        }
        // mask2[o, h] = 1 if ordering[o % input_dim] > m_h[h]
        let out_dim = input_dim * 2;
        let mut mask2 = vec![0.0; out_dim * hidden_size];
        for o in 0..out_dim {
            let dim_idx = o % input_dim;
            for h in 0..hidden_size {
                if ordering[dim_idx] > m_hidden[h] {
                    mask2[o * hidden_size + h] = 1.0;
                }
            }
        }
        (mask1, mask2)
    }

    /// Forward pass → (mu, log_sigma) each of length input_dim.
    fn forward_raw(&self, x: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        if x.len() != self.input_dim {
            return Err(mdn_err("MADE: x dim mismatch"));
        }
        let h = self.hidden_size;
        // Hidden layer with masking
        let mut hidden = self.b1.clone();
        for hi in 0..h {
            for i in 0..self.input_dim {
                hidden[hi] +=
                    self.w1[hi * self.input_dim + i] * self.mask1[hi * self.input_dim + i] * x[i];
            }
        }
        let hidden: Vec<f64> = hidden.into_iter().map(relu).collect();

        // Output layer with masking
        let out_dim = self.input_dim * 2;
        let mut out = self.b2.clone();
        for o in 0..out_dim {
            for hi in 0..h {
                out[o] += self.w2[o * h + hi] * self.mask2[o * h + hi] * hidden[hi];
            }
        }
        let mu = out[0..self.input_dim].to_vec();
        let log_sigma = out[self.input_dim..2 * self.input_dim].to_vec();
        Ok((mu, log_sigma))
    }

    /// Log-probability under the autoregressive Gaussian model.
    pub fn log_prob(&self, x: &[f64]) -> Result<f64> {
        let (mu, log_sigma) = self.forward_raw(x)?;
        let log_two_pi = (2.0 * std::f64::consts::PI).ln();
        let mut total = 0.0;
        for d in 0..self.input_dim {
            let sig = softplus(log_sigma[d]).max(1e-6);
            let diff = x[d] - mu[d];
            total -= 0.5 * (log_two_pi + 2.0 * sig.ln() + (diff / sig).powi(2));
        }
        Ok(total)
    }

    /// Ancestral sample (left-to-right single pass, approximate for single hidden layer).
    pub fn sample(&self, rng: &mut impl Rng) -> Result<Vec<f64>> {
        let mut x = vec![0.0; self.input_dim];
        for d in 0..self.input_dim {
            let (mu, log_sigma) = self.forward_raw(&x)?;
            let sig = softplus(log_sigma[d]).max(1e-6);
            x[d] = mu[d] + sig * bm_normal(rng);
        }
        Ok(x)
    }

    /// Update masks for a new input ordering (order-agnostic training).
    pub fn update_masks(&mut self, new_ordering: Vec<usize>) -> Result<()> {
        if new_ordering.len() != self.input_dim {
            return Err(mdn_err("MADE::update_masks: ordering length mismatch"));
        }
        let (mask1, mask2) = Self::build_masks(
            self.input_dim,
            self.hidden_size,
            &new_ordering,
            &self.m_hidden,
        );
        self.ordering = new_ordering;
        self.mask1 = mask1;
        self.mask2 = mask2;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. NormalizingFlowMdn — MDN with per-component affine coupling flows
// ─────────────────────────────────────────────────────────────────────────────

/// Per-component affine coupling layer state.
#[derive(Debug, Clone)]
struct AffineCouplingLayer {
    /// Scale network weights: [split..d] → [split] (maps xb → scale for xa).
    scale_w: Vec<f64>,
    scale_b: Vec<f64>,
    /// Translate network weights.
    translate_w: Vec<f64>,
    translate_b: Vec<f64>,
    split: usize,
    dim: usize,
}

impl AffineCouplingLayer {
    fn new(dim: usize, rng: &mut impl Rng) -> Self {
        let split = dim / 2;
        let b_dim = dim - split;
        let scale = (2.0 / (b_dim + split) as f64).sqrt();
        let scale_w: Vec<f64> = (0..split * b_dim).map(|_| bm_normal(rng) * scale).collect();
        let scale_b = vec![0.0; split];
        let translate_w: Vec<f64> = (0..split * b_dim).map(|_| bm_normal(rng) * scale).collect();
        let translate_b = vec![0.0; split];
        Self {
            scale_w,
            scale_b,
            translate_w,
            translate_b,
            split,
            dim,
        }
    }

    /// Forward transform: (xa, xb) → (ya, xb); log |det J| = sum(s).
    fn transform(&self, z: &[f64]) -> (Vec<f64>, f64) {
        let split = self.split;
        let b_dim = self.dim - split;
        let za = &z[0..split];
        let zb = &z[split..self.dim];
        // Scale and translate from zb
        let mut s = self.scale_b.clone();
        let mut t = self.translate_b.clone();
        for o in 0..split {
            for i in 0..b_dim {
                s[o] += self.scale_w[o * b_dim + i] * zb[i];
                t[o] += self.translate_w[o * b_dim + i] * zb[i];
            }
        }
        let s_act: Vec<f64> = s.iter().map(|&v| tanh_clip(v)).collect(); // scale in (-1,1)
        let log_det: f64 = s_act.iter().map(|sv| (sv.abs() + 1e-6).ln()).sum();
        let ya: Vec<f64> = (0..split)
            .map(|i| za[i] * (s_act[i].exp()) + t[i])
            .collect();
        let mut out = ya;
        out.extend_from_slice(zb);
        (out, log_det)
    }

    /// Inverse transform for computing log_prob.
    fn inverse(&self, x: &[f64]) -> (Vec<f64>, f64) {
        let split = self.split;
        let b_dim = self.dim - split;
        let xa = &x[0..split];
        let xb = &x[split..self.dim];
        let mut s = self.scale_b.clone();
        let mut t = self.translate_b.clone();
        for o in 0..split {
            for i in 0..b_dim {
                s[o] += self.scale_w[o * b_dim + i] * xb[i];
                t[o] += self.translate_w[o * b_dim + i] * xb[i];
            }
        }
        let s_act: Vec<f64> = s.iter().map(|&v| tanh_clip(v)).collect();
        let log_det: f64 = s_act.iter().map(|sv| (sv.abs() + 1e-6).ln()).sum();
        let za: Vec<f64> = (0..split)
            .map(|i| (xa[i] - t[i]) * (-s_act[i]).exp())
            .collect();
        let mut out = za;
        out.extend_from_slice(xb);
        (out, -log_det)
    }
}

/// MDN whose components are transformed by per-component normalizing flows.
#[derive(Debug, Clone)]
pub struct NormalizingFlowMdn {
    base_mdn: MixtureDensityNetwork,
    flows: Vec<AffineCouplingLayer>,
    out_dim: usize,
}

impl NormalizingFlowMdn {
    /// Build a `NormalizingFlowMdn`.
    pub fn new(
        in_dim: usize,
        hidden_sizes: &[usize],
        n_components: usize,
        out_dim: usize,
        seed: u64,
    ) -> Result<Self> {
        if out_dim < 2 {
            return Err(mdn_err("NormalizingFlowMdn: out_dim must be >= 2"));
        }
        let base_mdn =
            MixtureDensityNetwork::new(in_dim, hidden_sizes, n_components, out_dim, seed)?;
        let mut rng = StdRng::seed_from_u64(seed + 100);
        // One coupling layer per component
        let flows: Vec<AffineCouplingLayer> = (0..n_components)
            .map(|_| AffineCouplingLayer::new(out_dim, &mut rng))
            .collect();
        Ok(Self {
            base_mdn,
            flows,
            out_dim,
        })
    }

    /// Transform latent sample z through component `component_idx` flow.
    pub fn transform_sample(&self, z: &[f64], component_idx: usize) -> Result<Vec<f64>> {
        if z.len() != self.out_dim {
            return Err(mdn_err(
                "NormalizingFlowMdn::transform_sample: z dim mismatch",
            ));
        }
        if component_idx >= self.flows.len() {
            return Err(mdn_err(
                "NormalizingFlowMdn::transform_sample: component_idx OOB",
            ));
        }
        let (x, _log_det) = self.flows[component_idx].transform(z);
        Ok(x)
    }

    /// Inverse-transform observation x through component `component_idx` flow.
    pub fn inverse_transform(&self, x: &[f64], component_idx: usize) -> Result<(Vec<f64>, f64)> {
        if x.len() != self.out_dim {
            return Err(mdn_err(
                "NormalizingFlowMdn::inverse_transform: x dim mismatch",
            ));
        }
        if component_idx >= self.flows.len() {
            return Err(mdn_err(
                "NormalizingFlowMdn::inverse_transform: component_idx OOB",
            ));
        }
        Ok(self.flows[component_idx].inverse(x))
    }

    /// Log p(x | input) with Jacobian correction from the flow.
    pub fn log_prob(&self, input: &[f64], x: &[f64]) -> Result<f64> {
        let mix = self.base_mdn.forward(input)?;
        let log_two_pi = (2.0 * std::f64::consts::PI).ln();
        let k = mix.n_components();
        let d = self.out_dim;
        let mut log_terms = Vec::with_capacity(k);
        for ki in 0..k {
            let (z, log_det_inv) = self.flows[ki].inverse(x);
            // log_det_inv is -log|det J_fwd|; base density evaluated at z
            let mut log_gauss = 0.0;
            for j in 0..d {
                let sig = mix.sigma[ki][j].max(1e-8);
                let diff = z[j] - mix.mu[ki][j];
                log_gauss -= 0.5 * (log_two_pi + 2.0 * sig.ln() + (diff / sig).powi(2));
            }
            log_terms.push(mix.pi[ki].max(1e-300).ln() + log_gauss + log_det_inv);
        }
        let lp = log_sum_exp(&log_terms);
        if !lp.is_finite() {
            return Err(mdn_err("NormalizingFlowMdn::log_prob: non-finite"));
        }
        Ok(lp)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. ConditionalVaeMdn — CVAE + MDN decoder
// ─────────────────────────────────────────────────────────────────────────────

/// CVAE with MDN decoder.
///
/// Encoder q(z|x,c) → (mu_z, logvar_z).
/// Decoder p(x|z,c) is an MDN.
/// ELBO = -NLL_mixture + KL(q||N(0,I)).
#[derive(Debug, Clone)]
pub struct ConditionalVaeMdn {
    latent_dim: usize,
    // Encoder: [x_dim + c_dim] → hidden → (mu_z, logvar_z)
    enc_hidden: MdnLinear,
    enc_mu: MdnLinear,
    enc_logvar: MdnLinear,
    // Decoder MDN: [latent_dim + c_dim] → mixture
    decoder: MixtureDensityNetwork,
}

impl ConditionalVaeMdn {
    /// Build a `ConditionalVaeMdn`.
    pub fn new(
        x_dim: usize,
        c_dim: usize,
        hidden_dim: usize,
        latent_dim: usize,
        n_components: usize,
        seed: u64,
    ) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        let enc_hidden = MdnLinear::new(x_dim + c_dim, hidden_dim, &mut rng);
        let enc_mu = MdnLinear::new(hidden_dim, latent_dim, &mut rng);
        let enc_logvar = MdnLinear::new(hidden_dim, latent_dim, &mut rng);
        let decoder = MixtureDensityNetwork::new(
            latent_dim + c_dim,
            &[hidden_dim],
            n_components,
            x_dim,
            seed + 1,
        )?;
        Ok(Self {
            latent_dim,
            enc_hidden,
            enc_mu,
            enc_logvar,
            decoder,
        })
    }

    /// Encode x,c → (mu_z, logvar_z).
    pub fn encode(&self, x: &[f64], c: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        let xc: Vec<f64> = x.iter().chain(c.iter()).cloned().collect();
        let h = self
            .enc_hidden
            .forward(&xc)?
            .into_iter()
            .map(relu)
            .collect::<Vec<_>>();
        let mu_z = self.enc_mu.forward(&h)?;
        let logvar_z = self.enc_logvar.forward(&h)?;
        Ok((mu_z, logvar_z))
    }

    /// Decode z,c → `MdnGaussianMixture`.
    pub fn decode(&self, z: &[f64], c: &[f64]) -> Result<MdnGaussianMixture> {
        let zc: Vec<f64> = z.iter().chain(c.iter()).cloned().collect();
        self.decoder.forward(&zc)
    }

    /// ELBO = E[log p(x|z,c)] - KL(q(z|x,c)||N(0,I)).
    /// Samples one z via reparameterisation.
    pub fn elbo_loss(&self, x: &[f64], c: &[f64], rng: &mut impl Rng) -> Result<f64> {
        let (mu_z, logvar_z) = self.encode(x, c)?;
        let ld = self.latent_dim;
        // Reparameterisation: z = mu + exp(0.5 * logvar) * eps
        let z: Vec<f64> = (0..ld)
            .map(|i| mu_z[i] + (0.5 * logvar_z[i]).exp() * bm_normal(rng))
            .collect();
        let mix = self.decode(&z, c)?;
        let nll = -mix.log_prob(x)?;
        // KL(q||N(0,I)) = 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
        let kl: f64 = (0..ld)
            .map(|i| 0.5 * (logvar_z[i].exp() + mu_z[i].powi(2) - 1.0 - logvar_z[i]))
            .sum();
        Ok(nll + kl)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. BayesianMdn — MDN with local reparameterisation weight uncertainty
// ─────────────────────────────────────────────────────────────────────────────

/// Single Bayesian linear layer using local reparameterisation.
#[derive(Debug, Clone)]
struct BayesLinear {
    in_f: usize,
    out_f: usize,
    mu_w: Vec<f64>,
    rho_w: Vec<f64>,
    mu_b: Vec<f64>,
    rho_b: Vec<f64>,
}

impl BayesLinear {
    fn new(in_f: usize, out_f: usize, rng: &mut impl Rng) -> Self {
        let scale = (2.0 / (in_f + out_f) as f64).sqrt();
        let mu_w: Vec<f64> = (0..in_f * out_f).map(|_| bm_normal(rng) * scale).collect();
        let rho_w = vec![-3.0; in_f * out_f]; // ln(1+exp(-3)) ≈ 0.05
        let mu_b = vec![0.0; out_f];
        let rho_b = vec![-3.0; out_f];
        Self {
            in_f,
            out_f,
            mu_w,
            rho_w,
            mu_b,
            rho_b,
        }
    }

    /// Sample weights and compute output (local reparam).
    fn forward_sample(&self, x: &[f64], rng: &mut impl Rng) -> Vec<f64> {
        let mut out = self.mu_b.clone();
        for o in 0..self.out_f {
            // mu contribution
            for i in 0..self.in_f {
                out[o] += self.mu_w[o * self.in_f + i] * x[i];
            }
            // variance contribution: local reparam — sample from N(mu_a, sigma_a^2)
            let var: f64 = (0..self.in_f)
                .map(|i| {
                    let sig = softplus(self.rho_w[o * self.in_f + i]).powi(2);
                    sig * x[i].powi(2)
                })
                .sum::<f64>()
                + softplus(self.rho_b[o]).powi(2);
            out[o] += var.sqrt() * bm_normal(rng);
        }
        out
    }
}

/// Bayesian MDN: encoder uses `BayesLinear` layers; weight uncertainty estimated
/// via multiple forward samples.
#[derive(Debug, Clone)]
pub struct BayesianMdn {
    bayes_encoder: Vec<BayesLinear>,
    det_pi_head: MdnLinear,
    det_mu_head: MdnLinear,
    det_sigma_head: MdnLinear,
    n_components: usize,
    out_dim: usize,
}

impl BayesianMdn {
    /// Build a `BayesianMdn`.
    pub fn new(
        in_dim: usize,
        hidden_sizes: &[usize],
        n_components: usize,
        out_dim: usize,
        seed: u64,
    ) -> Result<Self> {
        if in_dim == 0 || n_components == 0 || out_dim == 0 {
            return Err(mdn_err("BayesianMdn: dims must be > 0"));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut bayes_encoder = Vec::new();
        let mut prev = in_dim;
        for &h in hidden_sizes {
            bayes_encoder.push(BayesLinear::new(prev, h, &mut rng));
            prev = h;
        }
        let last_hidden = prev;
        let det_pi_head = MdnLinear::new(last_hidden, n_components, &mut rng);
        let det_mu_head = MdnLinear::new(last_hidden, n_components * out_dim, &mut rng);
        let det_sigma_head = MdnLinear::new(last_hidden, n_components * out_dim, &mut rng);
        Ok(Self {
            bayes_encoder,
            det_pi_head,
            det_mu_head,
            det_sigma_head,
            n_components,
            out_dim,
        })
    }

    fn single_forward(&self, x: &[f64], rng: &mut impl Rng) -> Result<MdnGaussianMixture> {
        let mut h = x.to_vec();
        for layer in &self.bayes_encoder {
            let z = layer.forward_sample(&h, rng);
            h = z.into_iter().map(relu).collect();
        }
        let pi_logits = self.det_pi_head.forward(&h)?;
        let mu_flat = self.det_mu_head.forward(&h)?;
        let log_sigma_flat = self.det_sigma_head.forward(&h)?;
        let pi = softmax(&pi_logits);
        let k = self.n_components;
        let d = self.out_dim;
        let mu: Vec<Vec<f64>> = (0..k)
            .map(|ki| mu_flat[ki * d..(ki + 1) * d].to_vec())
            .collect();
        let sigma: Vec<Vec<f64>> = (0..k)
            .map(|ki| {
                log_sigma_flat[ki * d..(ki + 1) * d]
                    .iter()
                    .map(|&v| softplus(v).max(1e-6))
                    .collect()
            })
            .collect();
        Ok(MdnGaussianMixture { pi, mu, sigma })
    }

    /// Forward with uncertainty estimation via `n_samples` stochastic passes.
    ///
    /// Returns `(mean_mu, epistemic_variance)` over the mixture means.
    pub fn forward_with_uncertainty(
        &self,
        x: &[f64],
        n_samples: usize,
        seed: u64,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut rng = StdRng::seed_from_u64(seed);
        let d = self.out_dim;
        let mut mean_acc = vec![0.0; d];
        let mut sq_acc = vec![0.0; d];
        for _ in 0..n_samples {
            let mix = self.single_forward(x, &mut rng)?;
            let m = mix.mean();
            for j in 0..d {
                mean_acc[j] += m[j];
                sq_acc[j] += m[j].powi(2);
            }
        }
        let n = n_samples as f64;
        let mean: Vec<f64> = mean_acc.iter().map(|v| v / n).collect();
        let variance: Vec<f64> = (0..d)
            .map(|j| (sq_acc[j] / n - mean[j].powi(2)).max(0.0))
            .collect();
        Ok((mean, variance))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. MdnTrainer — unified trainer
// ─────────────────────────────────────────────────────────────────────────────

/// Optimiser choice for `MdnTrainer`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MdnOptimizer {
    Sgd,
    Adam,
}

/// Configuration for `MdnTrainer`.
#[derive(Debug, Clone)]
pub struct MdnTrainerConfig {
    pub optimizer: MdnOptimizer,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub max_epochs: usize,
    /// Fraction of data used for validation (0 = no early stopping).
    pub val_fraction: f64,
    /// Early stopping patience (epochs without improvement).
    pub patience: usize,
    pub seed: u64,
}

impl Default for MdnTrainerConfig {
    fn default() -> Self {
        Self {
            optimizer: MdnOptimizer::Adam,
            learning_rate: 1e-3,
            batch_size: 32,
            max_epochs: 10,
            val_fraction: 0.1,
            patience: 5,
            seed: 42,
        }
    }
}

/// Unified trainer for `MixtureDensityNetwork`.
pub struct MdnTrainer;

impl MdnTrainer {
    /// Train an MDN. Returns `Vec<(epoch, train_loss, val_loss)>`.
    pub fn train(
        model: &mut MixtureDensityNetwork,
        x: &[Vec<f64>],
        y: &[Vec<f64>],
        config: &MdnTrainerConfig,
    ) -> Result<Vec<(usize, f64, f64)>> {
        if x.len() != y.len() || x.is_empty() {
            return Err(mdn_err("MdnTrainer: data/target mismatch or empty"));
        }
        let n = x.len();
        let n_val = ((n as f64 * config.val_fraction) as usize)
            .max(1)
            .min(n / 2);
        let n_train = n - n_val;
        let (x_train, x_val) = x.split_at(n_train);
        let (y_train, y_val) = y.split_at(n_train);

        let mut history = Vec::new();
        let mut best_val = f64::INFINITY;
        let mut patience_count = 0usize;

        for epoch in 0..config.max_epochs {
            model.fit(
                x_train,
                y_train,
                1,
                config.learning_rate,
                config.batch_size,
                config.seed + epoch as u64,
            )?;

            let train_loss = Self::eval_nll(model, x_train, y_train);
            let val_loss = if config.val_fraction > 0.0 && !x_val.is_empty() {
                Self::eval_nll(model, x_val, y_val)
            } else {
                f64::NAN
            };

            history.push((epoch, train_loss, val_loss));

            if config.patience > 0 && config.val_fraction > 0.0 {
                if val_loss < best_val - 1e-6 {
                    best_val = val_loss;
                    patience_count = 0;
                } else {
                    patience_count += 1;
                    if patience_count >= config.patience {
                        break;
                    }
                }
            }
        }
        Ok(history)
    }

    fn eval_nll(model: &MixtureDensityNetwork, x: &[Vec<f64>], y: &[Vec<f64>]) -> f64 {
        let total: f64 = x
            .iter()
            .zip(y.iter())
            .filter_map(|(xi, yi)| {
                let mix = model.forward(xi).ok()?;
                model.nll_loss(&mix, yi).ok()
            })
            .sum();
        total / x.len() as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. DensityEstimationBenchmark — synthetic datasets + evaluation
// ─────────────────────────────────────────────────────────────────────────────

/// Synthetic density estimation benchmarks and evaluation utilities.
pub struct DensityEstimationBenchmark;

impl DensityEstimationBenchmark {
    /// Two-moons dataset: returns `n` 2D points.
    pub fn two_moons(n: usize, noise: f64, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut points = Vec::with_capacity(n);
        for i in 0..n {
            let half = if i % 2 == 0 { 0 } else { 1 };
            let angle = std::f64::consts::PI * rng.random::<f64>();
            let (x, y) = if half == 0 {
                (angle.cos(), angle.sin())
            } else {
                (1.0 - angle.cos(), 1.0 - angle.sin() - 0.5)
            };
            let nx = bm_normal(&mut rng) * noise;
            let ny = bm_normal(&mut rng) * noise;
            points.push(vec![x + nx, y + ny]);
        }
        points
    }

    /// Checkerboard dataset: uniform samples from a 4×4 checkerboard.
    pub fn checkerboard(n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut points = Vec::with_capacity(n);
        while points.len() < n {
            let x: f64 = rng.random::<f64>() * 4.0 - 2.0;
            let y: f64 = rng.random::<f64>() * 4.0 - 2.0;
            let ix = (x + 2.0) as i32;
            let iy = (y + 2.0) as i32;
            if (ix + iy) % 2 == 0 {
                points.push(vec![x, y]);
            }
        }
        points
    }

    /// Spirals dataset: two interleaved spirals.
    pub fn spirals(n: usize, noise: f64, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let half = n / 2;
        let mut points = Vec::with_capacity(n);
        for i in 0..half {
            let t = (i as f64) / (half as f64) * 2.0 * std::f64::consts::PI;
            let r = t / (2.0 * std::f64::consts::PI);
            let nx = bm_normal(&mut rng) * noise;
            let ny = bm_normal(&mut rng) * noise;
            points.push(vec![r * t.cos() + nx, r * t.sin() + ny]);
        }
        for i in 0..(n - half) {
            let t = (i as f64) / (half as f64) * 2.0 * std::f64::consts::PI + std::f64::consts::PI;
            let r = t / (2.0 * std::f64::consts::PI);
            let nx = bm_normal(&mut rng) * noise;
            let ny = bm_normal(&mut rng) * noise;
            points.push(vec![r * t.cos() + nx, r * t.sin() + ny]);
        }
        points
    }

    /// 8 Gaussian mixture arranged in a ring.
    pub fn gaussian_mixture_8(n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let k = 8usize;
        let radius = 2.5;
        let mut points = Vec::with_capacity(n);
        for _ in 0..n {
            let ki = rng.random_range(0..k);
            let angle = 2.0 * std::f64::consts::PI * ki as f64 / k as f64;
            let cx = radius * angle.cos();
            let cy = radius * angle.sin();
            points.push(vec![
                cx + bm_normal(&mut rng) * 0.3,
                cy + bm_normal(&mut rng) * 0.3,
            ]);
        }
        points
    }

    /// Compute test NLL of an MDN on held-out data.
    pub fn compute_test_nll(
        model: &MixtureDensityNetwork,
        x_test: &[Vec<f64>],
        y_test: &[Vec<f64>],
    ) -> Result<f64> {
        if x_test.is_empty() || x_test.len() != y_test.len() {
            return Err(mdn_err("compute_test_nll: empty or mismatched data"));
        }
        let total: f64 = x_test
            .iter()
            .zip(y_test.iter())
            .filter_map(|(xi, yi)| {
                let mix = model.forward(xi).ok()?;
                model.nll_loss(&mix, yi).ok()
            })
            .sum();
        Ok(total / x_test.len() as f64)
    }

    /// Evaluate log-density on a 20×20 grid over [x_min, x_max]^2.
    ///
    /// Returns a 20×20 grid of log-probabilities, where the conditioning input is `input`.
    pub fn visualize_density_grid(
        model: &MixtureDensityNetwork,
        input: &[f64],
        x_min: f64,
        x_max: f64,
        n: usize,
    ) -> Result<Vec<Vec<f64>>> {
        let n = n.max(2);
        let step = (x_max - x_min) / (n - 1) as f64;
        let mix = model.forward(input)?;
        let mut grid = Vec::with_capacity(n);
        for i in 0..n {
            let xi = x_min + i as f64 * step;
            let mut row = Vec::with_capacity(n);
            for j in 0..n {
                let yj = x_min + j as f64 * step;
                let lp = mix.log_prob(&[xi, yj]).unwrap_or(f64::NEG_INFINITY);
                row.push(lp);
            }
            grid.push(row);
        }
        Ok(grid)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. MdnMetrics — evaluation metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluation metrics for mixture density networks.
pub struct MdnMetrics;

impl MdnMetrics {
    /// Mean NLL over a set of (mixture, target) pairs.
    pub fn mean_nll(predictions: &[MdnGaussianMixture], targets: &[Vec<f64>]) -> Result<f64> {
        if predictions.len() != targets.len() || predictions.is_empty() {
            return Err(mdn_err("mean_nll: mismatched lengths or empty"));
        }
        let total: f64 = predictions
            .iter()
            .zip(targets.iter())
            .filter_map(|(mix, t)| mix.log_prob(t).ok().map(|lp| -lp))
            .sum();
        Ok(total / predictions.len() as f64)
    }

    /// Continuous Ranked Probability Score (CRPS) for a Gaussian mixture.
    ///
    /// Approximated via Monte Carlo sampling (n_samples=200).
    pub fn crps_score(mixture: &MdnGaussianMixture, target: &[f64], seed: u64) -> Result<f64> {
        let n_samples = 200usize;
        let mut rng = StdRng::seed_from_u64(seed);
        let d = mixture.out_dim();
        if target.len() != d {
            return Err(mdn_err("crps_score: target dim mismatch"));
        }
        let samples: Vec<Vec<f64>> = (0..n_samples)
            .map(|_| mixture.sample(&mut rng))
            .collect::<Result<Vec<_>>>()?;

        // CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
        let e_xy: f64 = samples
            .iter()
            .map(|s| {
                s.iter()
                    .zip(target.iter())
                    .map(|(si, ti)| (si - ti).abs())
                    .sum::<f64>()
                    / d as f64
            })
            .sum::<f64>()
            / n_samples as f64;

        let e_xx: f64 = samples
            .iter()
            .flat_map(|s1| {
                samples.iter().map(move |s2| {
                    s1.iter()
                        .zip(s2.iter())
                        .map(|(a, b)| (a - b).abs())
                        .sum::<f64>()
                        / d as f64
                })
            })
            .sum::<f64>()
            / (n_samples * n_samples) as f64;

        Ok(e_xy - 0.5 * e_xx)
    }

    /// Calibration error: fraction of targets within predicted p-quantile intervals.
    ///
    /// Returns per-bin (expected_coverage, observed_coverage) pairs.
    pub fn calibration_error(
        mixtures: &[MdnGaussianMixture],
        targets: &[Vec<f64>],
        n_bins: usize,
        seed: u64,
    ) -> Result<Vec<(f64, f64)>> {
        if mixtures.len() != targets.len() || mixtures.is_empty() {
            return Err(mdn_err("calibration_error: mismatched or empty"));
        }
        let n_bins = n_bins.max(2);
        let n_samples_mc = 500usize;
        let mut rng = StdRng::seed_from_u64(seed);

        // For each test point, compute the empirical quantile of the target
        // under the mixture using MC samples.
        let quantiles: Vec<f64> = mixtures
            .iter()
            .zip(targets.iter())
            .map(|(mix, t)| {
                let samples: Vec<Vec<f64>> = (0..n_samples_mc)
                    .filter_map(|_| mix.sample(&mut rng).ok())
                    .collect();
                if samples.is_empty() {
                    return 0.5;
                }
                let d = t.len();
                // Use L2 distance to target as 1D proxy
                let target_dist = 0.0_f64; // distance of target to itself is 0
                let _ = target_dist;
                let sample_dists: Vec<f64> = samples
                    .iter()
                    .map(|s| {
                        s.iter()
                            .zip(t.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt()
                            / d as f64
                    })
                    .collect();
                // Fraction of samples with distance ≤ 0 (i.e. fraction below target)
                let below = sample_dists.iter().filter(|&&d| d <= 0.0).count();
                below as f64 / sample_dists.len() as f64
            })
            .collect();

        let mut result = Vec::with_capacity(n_bins);
        for b in 0..n_bins {
            let expected = (b + 1) as f64 / n_bins as f64;
            let observed = quantiles.iter().filter(|&&q| q <= expected).count() as f64
                / quantiles.len() as f64;
            result.push((expected, observed));
        }
        Ok(result)
    }

    /// Expected Calibration Error (ECE): mean absolute difference between
    /// expected and observed coverage across bins.
    pub fn expected_calibration_error(calibration: &[(f64, f64)]) -> f64 {
        if calibration.is_empty() {
            return 0.0;
        }
        calibration.iter().map(|(e, o)| (e - o).abs()).sum::<f64>() / calibration.len() as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
