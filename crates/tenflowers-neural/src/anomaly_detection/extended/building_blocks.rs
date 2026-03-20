//! Shared f64 building blocks: `AdLinear` and `AdMlp`.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers (re-exported so sibling modules can use them)
// ─────────────────────────────────────────────────────────────────────────────

/// Box-Muller normal sample using two uniform values in (0, 1).
#[inline]
pub fn box_muller(u1: f64, u2: f64) -> f64 {
    let u1c = u1.max(1e-30);
    (-2.0 * u1c.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// ReLU activation.
#[inline]
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// ReLU derivative.
#[inline]
pub fn relu_d(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Xavier uniform limit for given fan-in and fan-out.
#[inline]
pub fn xavier_limit(fan_in: usize, fan_out: usize) -> f64 {
    (6.0 / (fan_in + fan_out) as f64).sqrt()
}

/// Percentile of a slice (sorts a copy).
pub fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ─────────────────────────────────────────────────────────────────────────────
// AdLinear: f64 linear layer
// ─────────────────────────────────────────────────────────────────────────────

/// A single dense (linear) layer with f64 weights.
///
/// Weight layout: `w[out_i][in_i]`.
#[derive(Clone)]
pub struct AdLinear {
    /// Weight matrix `[out_dim][in_dim]`.
    pub w: Vec<Vec<f64>>,
    /// Bias vector `[out_dim]`.
    pub b: Vec<f64>,
}

impl AdLinear {
    /// Construct with Xavier-uniform initialisation.
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(0xdead_beef_cafe);
        let limit = xavier_limit(in_dim, out_dim);
        let w: Vec<Vec<f64>> = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| {
                        let u: f64 = rng.random();
                        u * 2.0 * limit - limit
                    })
                    .collect()
            })
            .collect();
        let b = vec![0.0_f64; out_dim];
        Self { w, b }
    }

    /// Construct with Xavier-uniform init seeded from `seed`.
    pub fn new_seeded(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let limit = xavier_limit(in_dim, out_dim);
        let w: Vec<Vec<f64>> = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| {
                        let u: f64 = rng.random();
                        u * 2.0 * limit - limit
                    })
                    .collect()
            })
            .collect();
        let b = vec![0.0_f64; out_dim];
        Self { w, b }
    }

    /// Forward pass: `y = W * x + b`.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let out_dim = self.w.len();
        let mut y = vec![0.0_f64; out_dim];
        for (j, (row, bj)) in self.w.iter().zip(self.b.iter()).enumerate() {
            let mut acc = *bj;
            for (wi, xi) in row.iter().zip(x.iter()) {
                acc += wi * xi;
            }
            y[j] = acc;
        }
        y
    }

    /// Apply gradient update: `w -= lr * grad_w`, `b -= lr * grad_b`.
    pub fn update(&mut self, grad_w: &[Vec<f64>], grad_b: &[f64], lr: f64) {
        for (j, (gw_row, gb)) in grad_w.iter().zip(grad_b.iter()).enumerate() {
            for (i, gw) in gw_row.iter().enumerate() {
                self.w[j][i] -= lr * gw;
            }
            self.b[j] -= lr * gb;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AdMlp: f64 multi-layer perceptron
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-layer perceptron with f64 weights.
///
/// Hidden layers use ReLU; the output layer is linear.
#[derive(Clone)]
pub struct AdMlp {
    /// Sequence of linear layers.
    pub layers: Vec<AdLinear>,
}

impl AdMlp {
    /// Construct from `layer_sizes` (including input and output dimensions).
    ///
    /// Each successive pair of sizes defines an `AdLinear` layer.
    /// Hidden layers will use ReLU activation during `forward()`; the last is linear.
    pub fn new(layer_sizes: &[usize]) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "AdMlp needs at least two layer sizes"
        );
        let mut seed_counter = 0xfeed_c0de_u64;
        let layers: Vec<AdLinear> = layer_sizes
            .windows(2)
            .map(|w| {
                seed_counter = seed_counter.wrapping_add(0x9e37_79b9_7f4a_7c15);
                AdLinear::new_seeded(w[0], w[1], seed_counter)
            })
            .collect();
        Self { layers }
    }

    /// Forward pass; ReLU on all hidden layers, linear output.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let n_layers = self.layers.len();
        let mut h = x.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            let z = layer.forward(&h);
            if i < n_layers - 1 {
                h = z.iter().map(|&v| relu(v)).collect();
            } else {
                h = z;
            }
        }
        h
    }

    /// Compute per-layer (grad_w_row, grad_b_scalar) via finite differences.
    ///
    /// Returns `grads[layer][out_neuron][(row_grad_w, [grad_b_j])]`.
    /// For each output neuron `j`:
    /// - `row_grad_w[i]` = ∂L/∂w\[j\]\[i\]
    /// - The second element is a single-element Vec with ∂L/∂b\[j\]
    pub fn gradient_fd(
        &self,
        x: &[f64],
        target: &[f64],
        eps: f64,
    ) -> Vec<Vec<(Vec<f64>, Vec<f64>)>> {
        let mse = |net: &AdMlp| -> f64 {
            let y = net.forward(x);
            y.iter()
                .zip(target.iter())
                .map(|(yi, ti)| (yi - ti).powi(2))
                .sum::<f64>()
                / y.len().max(1) as f64
        };

        let base_loss = mse(self);
        let mut all_grads: Vec<Vec<(Vec<f64>, Vec<f64>)>> = Vec::with_capacity(self.layers.len());

        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            let out_dim = layer.w.len();
            let in_dim = if out_dim > 0 { layer.w[0].len() } else { 0 };

            let mut layer_grads: Vec<(Vec<f64>, Vec<f64>)> = Vec::with_capacity(out_dim);

            for j in 0..out_dim {
                let mut row_grad_w = vec![0.0_f64; in_dim];
                for i in 0..in_dim {
                    let mut net_p = self.clone();
                    net_p.layers[layer_idx].w[j][i] += eps;
                    let loss_p = mse(&net_p);
                    row_grad_w[i] = (loss_p - base_loss) / eps;
                }
                let mut net_p = self.clone();
                net_p.layers[layer_idx].b[j] += eps;
                let loss_p = mse(&net_p);
                let grad_b_j = (loss_p - base_loss) / eps;

                layer_grads.push((row_grad_w, vec![grad_b_j]));
            }

            all_grads.push(layer_grads);
        }

        all_grads
    }

    /// Apply gradients returned by `gradient_fd`.
    pub fn apply_gradients(&mut self, grads: &[Vec<(Vec<f64>, Vec<f64>)>], lr: f64) {
        for (layer_idx, layer_grads) in grads.iter().enumerate() {
            if layer_idx >= self.layers.len() {
                break;
            }
            for (j, (row_gw, gb)) in layer_grads.iter().enumerate() {
                if j >= self.layers[layer_idx].w.len() {
                    break;
                }
                for (i, gw) in row_gw.iter().enumerate() {
                    if i < self.layers[layer_idx].w[j].len() {
                        self.layers[layer_idx].w[j][i] -= lr * gw;
                    }
                }
                if let Some(&gb_val) = gb.first() {
                    self.layers[layer_idx].b[j] -= lr * gb_val;
                }
            }
        }
    }

    /// Output dimension of the MLP.
    pub fn output_dim(&self) -> usize {
        self.layers.last().map(|l| l.w.len()).unwrap_or(0)
    }
}
