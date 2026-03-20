//! Shared MLP and CNF dynamics backbone.
//!
//! Contains `CnfMlp`, `CnfDynamics`, and `ContinuousNormalizingFlow`.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::f64::consts::PI;

// ────────────────────────────────────────────────────────────────────────────
// Shared MLP for dynamics / velocity networks
// ────────────────────────────────────────────────────────────────────────────

/// Multi-layer perceptron used as backbone for CNF dynamics and flow matching velocity fields.
///
/// Weights layout: `weights[layer][out_neuron][in_neuron]`
/// Activation: tanh (smoother than ReLU — avoids discontinuous Jacobians in ODE integration).
#[derive(Clone)]
pub struct CnfMlp {
    /// Layer weights: `weights[l][j][i]` = weight from neuron `i` (layer `l`) to neuron `j` (layer `l+1`).
    pub weights: Vec<Vec<Vec<f64>>>,
    /// Layer biases: `biases[l][j]`.
    pub biases: Vec<Vec<f64>>,
}

impl CnfMlp {
    /// Construct a new MLP with Xavier-initialised weights.
    ///
    /// `layer_sizes` specifies the width of each layer including input and output.
    /// E.g. `[4, 64, 64, 4]` gives a 2-hidden-layer network mapping R^4 → R^4.
    pub fn new(layer_sizes: &[usize]) -> Self {
        assert!(layer_sizes.len() >= 2, "need at least input + output layer");
        let n_layers = layer_sizes.len() - 1;
        let mut weights = Vec::with_capacity(n_layers);
        let mut biases = Vec::with_capacity(n_layers);
        let mut rng = StdRng::seed_from_u64(0xabcdef01_u64);

        for l in 0..n_layers {
            let fan_in = layer_sizes[l];
            let fan_out = layer_sizes[l + 1];
            // Xavier uniform: U[-limit, limit], limit = sqrt(6 / (fan_in + fan_out))
            let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
            let layer_w: Vec<Vec<f64>> = (0..fan_out)
                .map(|_| {
                    (0..fan_in)
                        .map(|_| {
                            let u: f64 = rng.random();
                            u * 2.0 * limit - limit
                        })
                        .collect()
                })
                .collect();
            let layer_b: Vec<f64> = vec![0.0; fan_out];
            weights.push(layer_w);
            biases.push(layer_b);
        }

        CnfMlp { weights, biases }
    }

    /// Forward pass: applies affine transform + tanh for hidden layers, linear output.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let n_layers = self.weights.len();
        let mut h: Vec<f64> = x.to_vec();

        for (l, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let out_dim = w.len();
            let mut z = vec![0.0_f64; out_dim];
            for j in 0..out_dim {
                let mut acc = b[j];
                for (i, hi) in h.iter().enumerate() {
                    acc += w[j][i] * hi;
                }
                z[j] = acc;
            }
            // Apply tanh activation on all layers except the last
            if l < n_layers - 1 {
                for v in z.iter_mut() {
                    *v = v.tanh();
                }
            }
            h = z;
        }
        h
    }

    /// Diagonal of the Jacobian via central finite differences.
    ///
    /// Returns `∂f_i/∂x_i` for each dimension `i`.
    pub fn jacobian_diagonal_approx(&self, x: &[f64]) -> Vec<f64> {
        let eps = 1e-5;
        let d = x.len();
        let mut diag = vec![0.0_f64; d];
        for i in 0..d {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[i] += eps;
            xm[i] -= eps;
            let fp = self.forward(&xp);
            let fm = self.forward(&xm);
            if i < fp.len() {
                diag[i] = (fp[i] - fm[i]) / (2.0 * eps);
            }
        }
        diag
    }

    /// SGD weight update: `θ ← θ - lr * grad`.
    pub fn update(&mut self, grad_w: &[Vec<Vec<f64>>], grad_b: &[Vec<f64>], lr: f64) {
        for (l, (gw, gb)) in grad_w.iter().zip(grad_b.iter()).enumerate() {
            if l >= self.weights.len() {
                break;
            }
            for j in 0..self.weights[l].len().min(gw.len()) {
                for i in 0..self.weights[l][j].len().min(gw[j].len()) {
                    self.weights[l][j][i] -= lr * gw[j][i];
                }
            }
            for j in 0..self.biases[l].len().min(gb.len()) {
                self.biases[l][j] -= lr * gb[j];
            }
        }
    }

    /// Return the number of layers (including the output layer).
    pub fn n_layers(&self) -> usize {
        self.weights.len()
    }

    /// Output dimension of the MLP.
    pub fn out_dim(&self) -> usize {
        self.weights.last().map(|w| w.len()).unwrap_or(0)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// CNF Dynamics — f(z, t)
// ────────────────────────────────────────────────────────────────────────────

/// ODE dynamics `dz/dt = f(z, t; θ)` backed by a `CnfMlp`.
///
/// If `include_time = true`, the input to the MLP is the concatenation `[z; t]`
/// giving a time-conditioned vector field.
#[derive(Clone)]
pub struct CnfDynamics {
    /// The neural network parameterising the vector field.
    pub mlp: CnfMlp,
    /// Dimensionality of the state vector `z`.
    pub z_dim: usize,
    /// Whether to include time `t` as an extra input coordinate.
    pub include_time: bool,
}

impl CnfDynamics {
    /// Create CNF dynamics with the given architecture.
    ///
    /// Network input: `z_dim` (or `z_dim + 1` if `include_time`), output: `z_dim`.
    pub fn new(z_dim: usize, hidden_dim: usize, n_layers: usize, include_time: bool) -> Self {
        let in_dim = if include_time { z_dim + 1 } else { z_dim };
        let mut sizes = vec![in_dim];
        for _ in 0..n_layers {
            sizes.push(hidden_dim);
        }
        sizes.push(z_dim);
        CnfDynamics {
            mlp: CnfMlp::new(&sizes),
            z_dim,
            include_time,
        }
    }

    /// Evaluate the vector field `f(z, t)`.
    pub fn forward(&self, z: &[f64], t: f64) -> Vec<f64> {
        if self.include_time {
            let mut inp = z.to_vec();
            inp.push(t);
            self.mlp.forward(&inp)
        } else {
            self.mlp.forward(z)
        }
    }

    /// Hutchinson trace estimator: `E_ε[ε^T (∂f/∂z) ε]` with `ε ~ Rademacher(±1)`.
    ///
    /// For each sample, finite-difference JVP: `(∂f/∂z)ε ≈ (f(z+δε) - f(z-δε)) / (2δ)`.
    /// Then inner-product with `ε` gives an unbiased trace estimate.
    pub fn trace_jac_approx(&self, z: &[f64], t: f64, n_samples: usize, rng: &mut StdRng) -> f64 {
        let eps = 1e-4;
        let mut trace_est = 0.0_f64;
        let n = n_samples.max(1);

        for _ in 0..n {
            // Sample Rademacher vector ε ∈ {-1, +1}^d
            let epsilon: Vec<f64> = (0..self.z_dim)
                .map(|_| if rng.random::<f64>() < 0.5 { 1.0 } else { -1.0 })
                .collect();

            // z + eps * ε and z - eps * ε
            let z_plus: Vec<f64> = z
                .iter()
                .zip(epsilon.iter())
                .map(|(zi, ei)| zi + eps * ei)
                .collect();
            let z_minus: Vec<f64> = z
                .iter()
                .zip(epsilon.iter())
                .map(|(zi, ei)| zi - eps * ei)
                .collect();

            let f_plus = self.forward(&z_plus, t);
            let f_minus = self.forward(&z_minus, t);

            // JVP: Jε ≈ (f(z + eps*ε) - f(z - eps*ε)) / (2*eps)
            // Trace estimate: ε^T Jε
            let sample_est: f64 = epsilon
                .iter()
                .zip(f_plus.iter())
                .zip(f_minus.iter())
                .map(|((ei, fp_i), fm_i)| ei * (fp_i - fm_i) / (2.0 * eps))
                .sum();
            trace_est += sample_est;
        }
        trace_est / n as f64
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Continuous Normalizing Flow
// ────────────────────────────────────────────────────────────────────────────

/// Continuous Normalizing Flow model.
///
/// Transforms a base distribution `p_0 = N(base_mean, diag(base_std^2))` into
/// a target distribution via the ODE `dz/dt = f(z, t; θ)`.
///
/// Log-likelihood computation:
/// ```text
/// log p(x) = log p_0(z_0) + ∫_0^T tr(∂f/∂z) dt
/// ```
pub struct ContinuousNormalizingFlow {
    /// ODE dynamics network.
    pub dynamics: CnfDynamics,
    /// Mean of the base Gaussian distribution.
    pub base_mean: Vec<f64>,
    /// Standard deviation of the base Gaussian distribution (diagonal).
    pub base_std: Vec<f64>,
}

impl ContinuousNormalizingFlow {
    /// Create a CNF with standard-normal base distribution.
    pub fn new(z_dim: usize, hidden_dim: usize, n_layers: usize) -> Self {
        ContinuousNormalizingFlow {
            dynamics: CnfDynamics::new(z_dim, hidden_dim, n_layers, true),
            base_mean: vec![0.0; z_dim],
            base_std: vec![1.0; z_dim],
        }
    }

    /// Euler-integrate the augmented ODE `(dz/dt, d_logdet/dt)` from `t_start` to `t_end`.
    ///
    /// Returns `(z_T, log_det_jacobian)` where `log_det_jacobian = ∫ tr(∂f/∂z) dt`.
    pub fn integrate_forward(
        &self,
        z0: &[f64],
        n_steps: usize,
        t_start: f64,
        t_end: f64,
    ) -> (Vec<f64>, f64) {
        let n = n_steps.max(1);
        let dt = (t_end - t_start) / n as f64;
        let mut z = z0.to_vec();
        let mut log_det = 0.0_f64;
        let mut rng = StdRng::seed_from_u64(0xdeadbeef_u64);

        for step in 0..n {
            let t = t_start + step as f64 * dt;
            let dz = self.dynamics.forward(&z, t);
            let tr = self.dynamics.trace_jac_approx(&z, t, 1, &mut rng);
            // Euler step
            for (zi, dzi) in z.iter_mut().zip(dz.iter()) {
                *zi += dt * dzi;
            }
            log_det += dt * tr;
        }
        (z, log_det)
    }

    /// Euler-integrate backwards from `T=1` to `T=0` (inverse direction).
    ///
    /// Returns `(z_0, log_det)` where `log_det = -∫_T^0 tr(∂f/∂z) dt`.
    pub fn integrate_backward(&self, x: &[f64], n_steps: usize) -> (Vec<f64>, f64) {
        let n = n_steps.max(1);
        let dt = 1.0 / n as f64;
        let mut z = x.to_vec();
        let mut log_det = 0.0_f64;
        let mut rng = StdRng::seed_from_u64(0xcafe1234_u64);

        // Integrate from t=1 down to t=0 in steps of -dt
        for step in 0..n {
            let t = 1.0 - step as f64 * dt;
            let dz = self.dynamics.forward(&z, t);
            let tr = self.dynamics.trace_jac_approx(&z, t, 1, &mut rng);
            // Euler step backwards: dz/dt in reverse time is -f
            for (zi, dzi) in z.iter_mut().zip(dz.iter()) {
                *zi -= dt * dzi;
            }
            log_det += dt * tr;
        }
        (z, log_det)
    }

    /// Compute log p(x) = log p_0(z_0) + log_det via backward integration.
    pub fn log_prob(&self, x: &[f64], n_steps: usize) -> f64 {
        let (z0, log_det) = self.integrate_backward(x, n_steps);
        let log_p0 = self.log_base_prob(&z0);
        log_p0 + log_det
    }

    /// Log-probability under the base Gaussian `N(base_mean, diag(base_std^2))`.
    pub(crate) fn log_base_prob(&self, z: &[f64]) -> f64 {
        let d = z.len().min(self.base_mean.len()).min(self.base_std.len());
        let mut lp = 0.0_f64;
        for i in 0..d {
            let sigma = self.base_std[i].max(1e-15);
            let diff = z[i] - self.base_mean[i];
            lp -= 0.5 * (diff * diff / (sigma * sigma) + (2.0 * PI * sigma * sigma).ln());
        }
        lp
    }

    /// Sample a point from the model by sampling from the base distribution and integrating forward.
    pub fn sample(&self, n_steps: usize, rng: &mut StdRng) -> Vec<f64> {
        let d = self.dynamics.z_dim;
        // Sample z_0 ~ N(base_mean, diag(base_std^2)) via Box-Muller
        let z0: Vec<f64> = (0..d)
            .map(|i| {
                let u1: f64 = rng.random::<f64>().max(1e-15);
                let u2: f64 = rng.random::<f64>();
                let g = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
                self.base_mean[i] + self.base_std[i] * g
            })
            .collect();
        let (x, _log_det) = self.integrate_forward(&z0, n_steps, 0.0, 1.0);
        x
    }

    /// One training step: compute mean negative-log-likelihood over `x_batch`, update via FD gradient.
    ///
    /// Returns the mean NLL loss.
    pub fn train_step(&mut self, x_batch: &[Vec<f64>], n_steps: usize, lr: f64) -> f64 {
        if x_batch.is_empty() {
            return 0.0;
        }
        let batch_size = x_batch.len();

        // Compute baseline loss
        let base_loss: f64 = x_batch
            .iter()
            .map(|x| -self.log_prob(x, n_steps))
            .sum::<f64>()
            / batch_size as f64;

        // Finite-difference gradient w.r.t. each MLP parameter and update
        let fd_eps = 1e-4;
        let n_layers = self.dynamics.mlp.n_layers();

        let mut rng = StdRng::seed_from_u64(0x98765432_u64);
        let mut grad_w: Vec<Vec<Vec<f64>>> = self
            .dynamics
            .mlp
            .weights
            .iter()
            .map(|lw| lw.iter().map(|row| vec![0.0; row.len()]).collect())
            .collect();
        let mut grad_b: Vec<Vec<f64>> = self
            .dynamics
            .mlp
            .biases
            .iter()
            .map(|lb| vec![0.0; lb.len()])
            .collect();

        // Stochastic FD: random subset of parameters
        for l in 0..n_layers {
            for j in 0..self.dynamics.mlp.weights[l].len() {
                for i in 0..self.dynamics.mlp.weights[l][j].len() {
                    if rng.random::<f64>() < 0.05 {
                        // Perturb weight
                        self.dynamics.mlp.weights[l][j][i] += fd_eps;
                        let perturbed_loss: f64 = x_batch
                            .iter()
                            .map(|x| -self.log_prob(x, n_steps))
                            .sum::<f64>()
                            / batch_size as f64;
                        self.dynamics.mlp.weights[l][j][i] -= fd_eps;
                        grad_w[l][j][i] = (perturbed_loss - base_loss) / fd_eps;
                    }
                }
            }
            for j in 0..self.dynamics.mlp.biases[l].len() {
                if rng.random::<f64>() < 0.05 {
                    self.dynamics.mlp.biases[l][j] += fd_eps;
                    let perturbed_loss: f64 = x_batch
                        .iter()
                        .map(|x| -self.log_prob(x, n_steps))
                        .sum::<f64>()
                        / batch_size as f64;
                    self.dynamics.mlp.biases[l][j] -= fd_eps;
                    grad_b[l][j] = (perturbed_loss - base_loss) / fd_eps;
                }
            }
        }

        self.dynamics.mlp.update(&grad_w, &grad_b, lr);
        base_loss
    }
}
