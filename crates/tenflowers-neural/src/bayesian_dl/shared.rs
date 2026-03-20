//! BdlLinear and BdlMlp — shared neural network types for bayesian_dl.

use super::helpers::relu;
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// BdlLinear
// ─────────────────────────────────────────────────────────────────────────────

/// Single fully-connected layer with f64 weights.
/// Weights are laid out as `w[out_i][in_j]`.
#[derive(Clone, Debug)]
pub struct BdlLinear {
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl BdlLinear {
    /// Xavier-initialised linear layer.
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let limit = (6.0 / (in_dim + out_dim) as f64).sqrt();
        let seed = (in_dim as u64).wrapping_mul(6364136223846793005)
            ^ (out_dim as u64).wrapping_add(1442695040888963407);
        let mut rng = StdRng::seed_from_u64(seed);
        let w = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| {
                        let u: f64 = rng.random();
                        u * 2.0 * limit - limit
                    })
                    .collect()
            })
            .collect();
        Self {
            w,
            b: vec![0.0; out_dim],
            in_dim,
            out_dim,
        }
    }

    /// Linear forward: `W x + b`.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        (0..self.out_dim)
            .map(|o| {
                let dot: f64 = self.w[o].iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
                dot + self.b[o]
            })
            .collect()
    }

    /// Flatten all parameters: weights row-major then biases.
    pub fn params_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_params());
        for row in &self.w {
            out.extend_from_slice(row);
        }
        out.extend_from_slice(&self.b);
        out
    }

    /// Restore parameters from flat vector.
    pub fn set_params(&mut self, flat: &[f64]) {
        let mut idx = 0;
        for o in 0..self.out_dim {
            for i in 0..self.in_dim {
                if idx < flat.len() {
                    self.w[o][i] = flat[idx];
                }
                idx += 1;
            }
        }
        for b_i in 0..self.out_dim {
            if idx < flat.len() {
                self.b[b_i] = flat[idx];
            }
            idx += 1;
        }
    }

    /// Total parameter count: `in_dim * out_dim + out_dim`.
    pub fn n_params(&self) -> usize {
        self.in_dim * self.out_dim + self.out_dim
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BdlMlp
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-layer perceptron with ReLU hidden activations and linear output.
#[derive(Clone, Debug)]
pub struct BdlMlp {
    pub layers: Vec<BdlLinear>,
    pub layer_sizes: Vec<usize>,
}

impl BdlMlp {
    /// Build MLP from a slice of layer sizes (e.g. `&[2, 16, 1]`).
    pub fn new(layer_sizes: &[usize]) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "Need at least input and output size"
        );
        let layers = (0..layer_sizes.len() - 1)
            .map(|i| BdlLinear::new(layer_sizes[i], layer_sizes[i + 1]))
            .collect();
        Self {
            layers,
            layer_sizes: layer_sizes.to_vec(),
        }
    }

    /// Forward pass: ReLU on hidden layers, linear output.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let n = self.layers.len();
        let mut h = x.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h);
            if i < n - 1 {
                for v in h.iter_mut() {
                    *v = relu(*v);
                }
            }
        }
        h
    }

    /// Flatten all layer parameters.
    pub fn params_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_params());
        for layer in &self.layers {
            out.extend(layer.params_flat());
        }
        out
    }

    /// Restore all layer parameters from a flat vector.
    pub fn set_params(&mut self, flat: &[f64]) {
        let mut offset = 0;
        for layer in self.layers.iter_mut() {
            let np = layer.n_params();
            let end = (offset + np).min(flat.len());
            layer.set_params(&flat[offset..end]);
            offset += np;
        }
    }

    /// Total parameter count.
    pub fn n_params(&self) -> usize {
        self.layers.iter().map(|l| l.n_params()).sum()
    }

    /// MSE loss over a single (x, y) pair.
    pub fn loss(&self, x: &[f64], y: &[f64]) -> f64 {
        let pred = self.forward(x);
        let n = pred.len().max(1);
        pred.iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / n as f64
    }

    /// Finite-difference gradient of MSE loss w.r.t. parameters.
    pub fn gradient_fd(&self, x: &[f64], y: &[f64], eps: f64) -> Vec<f64> {
        let params = self.params_flat();
        let n = params.len();
        let mut grad = vec![0.0; n];
        let mut model_copy = self.clone();
        let loss_0 = self.loss(x, y);
        for i in 0..n {
            let mut p_plus = params.clone();
            p_plus[i] += eps;
            model_copy.set_params(&p_plus);
            let loss_plus = model_copy.loss(x, y);
            grad[i] = (loss_plus - loss_0) / eps;
        }
        model_copy.set_params(&params);
        grad
    }

    /// Batch MSE loss across multiple (x, y) pairs.
    pub fn batch_loss(&self, x_data: &[Vec<f64>], y_data: &[Vec<f64>]) -> f64 {
        if x_data.is_empty() {
            return 0.0;
        }
        let total: f64 = x_data
            .iter()
            .zip(y_data.iter())
            .map(|(x, y)| self.loss(x, y))
            .sum();
        total / x_data.len() as f64
    }

    /// Batch gradient via finite differences.
    pub fn batch_gradient_fd(
        &self,
        x_data: &[Vec<f64>],
        y_data: &[Vec<f64>],
        eps: f64,
    ) -> Vec<f64> {
        let params = self.params_flat();
        let n = params.len();
        let mut grad = vec![0.0; n];
        let mut model_copy = self.clone();
        let loss_0 = self.batch_loss(x_data, y_data);
        for i in 0..n {
            let mut p_plus = params.clone();
            p_plus[i] += eps;
            model_copy.set_params(&p_plus);
            let loss_plus = model_copy.batch_loss(x_data, y_data);
            grad[i] = (loss_plus - loss_0) / eps;
        }
        model_copy.set_params(&params);
        grad
    }
}
