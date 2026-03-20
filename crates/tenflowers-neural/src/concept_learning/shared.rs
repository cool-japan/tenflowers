//! Shared building blocks: ClLinear and ClMlp.

use super::helpers::{dot_f64, relu_f64, xavier_matrix};
use scirs2_core::random::{rngs::StdRng, SeedableRng};

/// A single fully-connected linear layer `y = W x + b`.
///
/// Named with `Cl` prefix to avoid collision with other `Linear` types in the
/// workspace.
#[derive(Debug, Clone)]
pub struct ClLinear {
    /// Weight matrix `[out][in]`.
    pub w: Vec<Vec<f64>>,
    /// Bias vector `[out]`.
    pub b: Vec<f64>,
}

impl ClLinear {
    /// Construct a Xavier-initialised layer.
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(0x00c0_0000);
        Self {
            w: xavier_matrix(out_dim, in_dim, &mut rng),
            b: vec![0.0; out_dim],
        }
    }

    /// Forward pass: returns `[out_dim]` vector.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        self.w
            .iter()
            .zip(self.b.iter())
            .map(|(row, &bias)| dot_f64(row, x) + bias)
            .collect()
    }

    /// SGD weight update.
    pub fn update(&mut self, grad_w: &[Vec<f64>], grad_b: &[f64], lr: f64) {
        for (row, gw) in self.w.iter_mut().zip(grad_w.iter()) {
            for (w_ij, &g) in row.iter_mut().zip(gw.iter()) {
                *w_ij -= lr * g;
            }
        }
        for (b_i, &g) in self.b.iter_mut().zip(grad_b.iter()) {
            *b_i -= lr * g;
        }
    }
}

/// A multi-layer perceptron built from [`ClLinear`] layers.
///
/// Hidden layers use ReLU; the final layer is linear.
#[derive(Debug, Clone)]
pub struct ClMlp {
    pub layers: Vec<ClLinear>,
}

impl ClMlp {
    /// Build an MLP with the given sequence of layer widths (including input and output).
    pub fn new(layer_sizes: &[usize]) -> Self {
        assert!(layer_sizes.len() >= 2, "need at least input+output dims");
        let layers = layer_sizes
            .windows(2)
            .map(|w| ClLinear::new(w[0], w[1]))
            .collect();
        Self { layers }
    }

    /// Forward pass through all layers; ReLU on hidden, linear on output.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let n = self.layers.len();
        let mut h: Vec<f64> = x.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            let out = layer.forward(&h);
            if i + 1 < n {
                h = out.into_iter().map(relu_f64).collect();
            } else {
                h = out;
            }
        }
        h
    }

    /// Finite-difference gradients.  Returns `[layer][(grad_w_matrix, grad_b)]`.
    pub fn gradient_fd(
        &self,
        x: &[f64],
        target: &[f64],
        eps: f64,
    ) -> Vec<Vec<(Vec<Vec<f64>>, Vec<f64>)>> {
        let loss_fn = |model: &ClMlp, inp: &[f64]| -> f64 {
            let pred = model.forward(inp);
            pred.iter()
                .zip(target.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                * 0.5
        };

        let base_loss = loss_fn(self, x);
        let mut grads: Vec<Vec<(Vec<Vec<f64>>, Vec<f64>)>> = Vec::new();

        for l in 0..self.layers.len() {
            let out_dim = self.layers[l].w.len();
            let in_dim = if out_dim == 0 {
                0
            } else {
                self.layers[l].w[0].len()
            };
            let mut layer_grads: Vec<(Vec<Vec<f64>>, Vec<f64>)> = Vec::new();

            let mut gw = vec![vec![0.0_f64; in_dim]; out_dim];
            let mut gb = vec![0.0_f64; out_dim];

            for i in 0..out_dim {
                for j in 0..in_dim {
                    let mut model_p = self.clone();
                    model_p.layers[l].w[i][j] += eps;
                    let loss_p = loss_fn(&model_p, x);
                    gw[i][j] = (loss_p - base_loss) / eps;
                }
                let mut model_p = self.clone();
                model_p.layers[l].b[i] += eps;
                let loss_p = loss_fn(&model_p, x);
                gb[i] = (loss_p - base_loss) / eps;
            }

            layer_grads.push((gw, gb));
            grads.push(layer_grads);
        }
        grads
    }

    /// Apply gradients returned by `gradient_fd`.
    pub fn apply_gradients(&mut self, grads: &[Vec<(Vec<Vec<f64>>, Vec<f64>)>], lr: f64) {
        for (l, layer_grads) in grads.iter().enumerate() {
            if l >= self.layers.len() {
                break;
            }
            for (gw, gb) in layer_grads.iter() {
                self.layers[l].update(gw, gb, lr);
            }
        }
    }
}
