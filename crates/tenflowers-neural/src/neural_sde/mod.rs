//! Neural Stochastic Differential Equations, Neural CDEs, and Path Signatures
//!
//! Implements:
//! - Latent SDE (Kidger et al. 2021): dZ = f(t,Z)dt + g(t,Z)dW
//! - Neural CDE (Kidger et al. 2020): dZ = f(Z) dX, X = natural cubic spline
//! - Path Signature (iterated integrals via Chen's identity)
//! - Signature Kernel (truncated inner product in tensor algebra)
//!
//! All computation is 100% pure Rust, f64, no unwrap().

pub mod extensions;
pub use extensions::*;
pub mod advanced;
pub use advanced::*;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// Box-Muller standard normal sampler

#[inline]
pub(crate) fn sample_standard_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-10);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// Softplus activation

#[inline]
pub(crate) fn softplus(x: f64) -> f64 {
    // Numerically stable softplus with a floor to ensure strict positivity.
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln().max(1e-7)
    }
}

// NsdeMlp: multi-layer perceptron for Neural SDE components

/// Multi-layer perceptron used internally by Neural SDE / CDE components.
///
/// Architecture: ReLU on all hidden layers, linear (no activation) on final layer.
#[derive(Debug, Clone)]
pub struct NsdeMlp {
    /// Layer weights: `[layer][out_dim][in_dim]`
    pub weights: Vec<Vec<Vec<f64>>>,
    /// Layer biases: `[layer][out_dim]`
    pub biases: Vec<Vec<f64>>,
}

impl NsdeMlp {
    /// Construct an MLP with Xavier-uniform initialization.
    /// `layer_sizes` = `[in, h1, ..., out]`.
    pub fn new(layer_sizes: &[usize]) -> Self {
        assert!(layer_sizes.len() >= 2, "need at least input + output");
        let mut rng = StdRng::seed_from_u64(42);
        let n_layers = layer_sizes.len() - 1;
        let mut weights = Vec::with_capacity(n_layers);
        let mut biases = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let fan_in = layer_sizes[l];
            let fan_out = layer_sizes[l + 1];
            let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
            let w: Vec<Vec<f64>> = (0..fan_out)
                .map(|_| {
                    (0..fan_in)
                        .map(|_| 2.0 * limit * rng.random::<f64>() - limit)
                        .collect()
                })
                .collect();
            weights.push(w);
            biases.push(vec![0.0; fan_out]);
        }
        Self { weights, biases }
    }

    /// Forward pass. Hidden layers use ReLU; final layer is linear.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut activation: Vec<f64> = x.to_vec();
        let n_layers = self.weights.len();
        for (l, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let mut next = Vec::with_capacity(w.len());
            for (row, bi) in w.iter().zip(b.iter()) {
                let v = row
                    .iter()
                    .zip(activation.iter())
                    .map(|(wi, ai)| wi * ai)
                    .sum::<f64>()
                    + bi;
                next.push(if l < n_layers - 1 { v.max(0.0) } else { v });
            }
            activation = next;
        }
        activation
    }

    /// Output dimension.
    pub fn output_dim(&self) -> usize {
        self.weights.last().map(|w| w.len()).unwrap_or(0)
    }

    /// Input dimension.
    pub fn input_dim(&self) -> usize {
        self.weights
            .first()
            .map(|w| w.first().map(|r| r.len()).unwrap_or(0))
            .unwrap_or(0)
    }

    /// SGD weight update given gradients.
    pub fn update(&mut self, grad_w: &[Vec<Vec<f64>>], grad_b: &[Vec<f64>], lr: f64) {
        for (l, (gw, gb)) in grad_w.iter().zip(grad_b.iter()).enumerate() {
            if l >= self.weights.len() {
                break;
            }
            for (i, grow) in gw.iter().enumerate() {
                if i >= self.weights[l].len() {
                    break;
                }
                for (j, &g) in grow.iter().enumerate() {
                    if j >= self.weights[l][i].len() {
                        break;
                    }
                    self.weights[l][i][j] -= lr * g;
                }
            }
            for (i, &g) in gb.iter().enumerate() {
                if i >= self.biases[l].len() {
                    break;
                }
                self.biases[l][i] -= lr * g;
            }
        }
    }
}

// SDE Drift Net

/// Parameterizes the drift function f(t, Z) → R^{z_dim}.
///
/// Time `t` is concatenated to the state vector before the MLP.
#[derive(Debug, Clone)]
pub struct SdeDriftNet {
    /// Underlying MLP.
    pub mlp: NsdeMlp,
    /// Dimension of the time input (always 1).
    pub t_dim: usize,
    /// Latent dimension.
    pub z_dim: usize,
}

impl SdeDriftNet {
    /// Build a drift network with `n_layers` hidden layers of `hidden_dim` units.
    pub fn new(z_dim: usize, hidden_dim: usize, n_layers: usize) -> Self {
        let input_dim = 1 + z_dim; // [t, z]
        let mut sizes = vec![input_dim];
        for _ in 0..n_layers {
            sizes.push(hidden_dim);
        }
        sizes.push(z_dim);
        Self {
            mlp: NsdeMlp::new(&sizes),
            t_dim: 1,
            z_dim,
        }
    }

    /// Evaluate f(t, z) → R^{z_dim}.
    pub fn forward(&self, t: f64, z: &[f64]) -> Vec<f64> {
        let mut inp = Vec::with_capacity(1 + z.len());
        inp.push(t);
        inp.extend_from_slice(z);
        self.mlp.forward(&inp)
    }
}

// SDE Diffusion Net

/// Parameterizes the diagonal diffusion g(t, Z) → R^{z_dim} (positive).
///
/// Softplus is applied to the output to enforce positivity.
#[derive(Debug, Clone)]
pub struct SdeDiffusionNet {
    /// Underlying MLP.
    pub mlp: NsdeMlp,
    /// Latent dimension.
    pub z_dim: usize,
}

impl SdeDiffusionNet {
    /// Build a diffusion network.
    pub fn new(z_dim: usize, hidden_dim: usize, n_layers: usize) -> Self {
        let input_dim = 1 + z_dim;
        let mut sizes = vec![input_dim];
        for _ in 0..n_layers {
            sizes.push(hidden_dim);
        }
        sizes.push(z_dim);
        Self {
            mlp: NsdeMlp::new(&sizes),
            z_dim,
        }
    }

    /// Evaluate g(t, z) → R^{z_dim}, all values positive via softplus.
    pub fn forward(&self, t: f64, z: &[f64]) -> Vec<f64> {
        let mut inp = Vec::with_capacity(1 + z.len());
        inp.push(t);
        inp.extend_from_slice(z);
        self.mlp
            .forward(&inp)
            .iter()
            .map(|&v| softplus(v))
            .collect()
    }
}

// SDE Encoder

/// Maps an observation x_0 → (z0_mean, z0_log_var).
#[derive(Debug, Clone)]
pub struct SdeEncoder {
    /// Underlying MLP mapping x → 2*z_dim.
    pub mlp: NsdeMlp,
    /// Observation dimension.
    pub x_dim: usize,
    /// Latent dimension.
    pub z_dim: usize,
}

impl SdeEncoder {
    /// Build the encoder.
    pub fn new(x_dim: usize, z_dim: usize, hidden_dim: usize) -> Self {
        let sizes = [x_dim, hidden_dim, hidden_dim, 2 * z_dim];
        Self {
            mlp: NsdeMlp::new(&sizes),
            x_dim,
            z_dim,
        }
    }

    /// Encode x → (z0_mean, z0_log_var), each of length z_dim.
    pub fn encode(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let out = self.mlp.forward(x);
        let mean: Vec<f64> = out[..self.z_dim].to_vec();
        let log_var: Vec<f64> = out[self.z_dim..].to_vec();
        (mean, log_var)
    }
}

// SDE Decoder

/// Maps a latent state z → observation space x.
#[derive(Debug, Clone)]
pub struct SdeDecoder {
    /// Underlying MLP.
    pub mlp: NsdeMlp,
    /// Latent dimension.
    pub z_dim: usize,
    /// Observation dimension.
    pub x_dim: usize,
}

impl SdeDecoder {
    /// Build the decoder.
    pub fn new(z_dim: usize, x_dim: usize, hidden_dim: usize) -> Self {
        let sizes = [z_dim, hidden_dim, x_dim];
        Self {
            mlp: NsdeMlp::new(&sizes),
            z_dim,
            x_dim,
        }
    }

    /// Decode z → x.
    pub fn decode(&self, z: &[f64]) -> Vec<f64> {
        self.mlp.forward(z)
    }
}

// LatentSde configuration and model

/// Configuration for the Latent SDE model.
#[derive(Debug, Clone)]
pub struct LatentSdeConfig {
    /// Latent dimension.
    pub z_dim: usize,
    /// Observation dimension.
    pub x_dim: usize,
    /// Hidden layer width.
    pub hidden_dim: usize,
    /// Number of Euler-Maruyama steps.
    pub n_steps: usize,
    /// Step size dt.
    pub dt: f64,
    /// KL divergence coefficient (beta in beta-VAE style).
    pub kl_coeff: f64,
}

/// Latent Stochastic Differential Equation model (Kidger et al. 2021).
///
/// Generative model: dZ = f_prior(t, Z) dt + g(t, Z) dW
/// Inference model:  dZ = f_posterior(t, Z) dt + g(t, Z) dW
/// ELBO = E[log p(x|z)] - kl_coeff * KL(posterior || prior)
#[derive(Debug, Clone)]
pub struct LatentSde {
    /// Encoder: x_0 → (z0_mean, z0_log_var)
    pub encoder: SdeEncoder,
    /// Posterior drift network.
    pub posterior_drift: SdeDriftNet,
    /// Prior drift network.
    pub prior_drift: SdeDriftNet,
    /// Shared diffusion network.
    pub diffusion: SdeDiffusionNet,
    /// Decoder: z_t → x_t
    pub decoder: SdeDecoder,
    /// Hyperparameters.
    pub config: LatentSdeConfig,
}

impl LatentSde {
    /// Construct a new LatentSde with default network sizes (1 hidden layer each).
    pub fn new(config: LatentSdeConfig) -> Self {
        let hidden = config.hidden_dim;
        let z = config.z_dim;
        let x = config.x_dim;
        Self {
            encoder: SdeEncoder::new(x, z, hidden),
            posterior_drift: SdeDriftNet::new(z, hidden, 1),
            prior_drift: SdeDriftNet::new(z, hidden, 1),
            diffusion: SdeDiffusionNet::new(z, hidden, 1),
            decoder: SdeDecoder::new(z, x, hidden),
            config,
        }
    }

    /// Sample initial latent state z_0 via reparameterization.
    ///
    /// z_0 = mean + eps * exp(0.5 * log_var), eps ~ N(0, I)
    pub fn sample_z0(&self, x0: &[f64], rng: &mut StdRng) -> Vec<f64> {
        let (mean, log_var) = self.encoder.encode(x0);
        mean.iter()
            .zip(log_var.iter())
            .map(|(&m, &lv)| {
                let eps = sample_standard_normal(rng);
                m + eps * (0.5 * lv).exp()
            })
            .collect()
    }

    /// Simulate trajectory via Euler-Maruyama discretization.
    ///
    /// Returns `[n_steps+1][z_dim]` array of latent states.
    pub fn euler_maruyama(
        &self,
        z0: &[f64],
        use_posterior: bool,
        rng: &mut StdRng,
    ) -> Vec<Vec<f64>> {
        let n = self.config.n_steps;
        let dt = self.config.dt;
        let sqrt_dt = dt.sqrt();
        let mut traj = Vec::with_capacity(n + 1);
        let mut z = z0.to_vec();
        traj.push(z.clone());
        for step in 0..n {
            let t = step as f64 * dt;
            let drift = if use_posterior {
                self.posterior_drift.forward(t, &z)
            } else {
                self.prior_drift.forward(t, &z)
            };
            let diffusion = self.diffusion.forward(t, &z);
            let mut z_new = Vec::with_capacity(z.len());
            for i in 0..z.len() {
                let eps = sample_standard_normal(rng);
                let di = drift.get(i).copied().unwrap_or(0.0);
                let gi = diffusion.get(i).copied().unwrap_or(1e-6);
                z_new.push(z[i] + di * dt + gi * sqrt_dt * eps);
            }
            z = z_new;
            traj.push(z.clone());
        }
        traj
    }

    /// Decode entire latent trajectory to observation space.
    ///
    /// Returns `[n_steps+1][x_dim]`.
    pub fn reconstruct(&self, z_trajectory: &[Vec<f64>]) -> Vec<Vec<f64>> {
        z_trajectory
            .iter()
            .map(|z| self.decoder.decode(z))
            .collect()
    }

    /// Compute the ELBO loss for an observed sequence.
    ///
    /// Returns `(elbo, recon_loss, kl_loss)`.
    ///
    /// - `recon_loss` = mean squared error between decoded z and x_obs
    /// - `kl_loss` ≈ 0.5 * Σ_t ||f_post(t,Z) - f_prior(t,Z)||^2 / g^2 * dt (Girsanov)
    /// - `elbo` = -recon_loss - kl_coeff * kl_loss
    pub fn elbo(&self, x_obs: &[Vec<f64>], rng: &mut StdRng) -> (f64, f64, f64) {
        let x0 = match x_obs.first() {
            Some(v) => v.as_slice(),
            None => return (0.0, 0.0, 0.0),
        };
        let z0 = self.sample_z0(x0, rng);
        let z_traj = self.euler_maruyama(&z0, true, rng);
        let x_pred = self.reconstruct(&z_traj);

        // Reconstruction loss (MSE)
        let mut recon_sum = 0.0;
        let mut recon_count = 0usize;
        for (xp, xo) in x_pred.iter().zip(x_obs.iter()) {
            for (&p, &o) in xp.iter().zip(xo.iter()) {
                let diff = p - o;
                recon_sum += diff * diff;
                recon_count += 1;
            }
        }
        let recon_loss = if recon_count > 0 {
            recon_sum / recon_count as f64
        } else {
            0.0
        };

        // KL loss via Girsanov discretization
        let dt = self.config.dt;
        let mut kl_sum = 0.0;
        for (step, z) in z_traj.iter().enumerate() {
            let t = step as f64 * dt;
            let f_post = self.posterior_drift.forward(t, z);
            let f_prior = self.prior_drift.forward(t, z);
            let g = self.diffusion.forward(t, z);
            for i in 0..z.len() {
                let fp = f_post.get(i).copied().unwrap_or(0.0);
                let fpr = f_prior.get(i).copied().unwrap_or(0.0);
                let gi = g.get(i).copied().unwrap_or(1.0).max(1e-8);
                let diff = fp - fpr;
                kl_sum += 0.5 * diff * diff / (gi * gi) * dt;
            }
        }
        let kl_loss = kl_sum / z_traj.len().max(1) as f64;
        let elbo = -recon_loss - self.config.kl_coeff * kl_loss;
        (elbo, recon_loss, kl_loss)
    }

    /// Sample z_0 from encoder, run posterior Euler-Maruyama, decode all steps.
    pub fn forward(&self, x0: &[f64], rng: &mut StdRng) -> Vec<Vec<f64>> {
        let z0 = self.sample_z0(x0, rng);
        let z_traj = self.euler_maruyama(&z0, true, rng);
        self.reconstruct(&z_traj)
    }
}

// Natural Cubic Spline

/// Natural cubic spline interpolant for time series data.
///
/// Used internally by `NeuralCde` to provide a differentiable path X(t).
#[derive(Debug, Clone)]
pub struct NaturalCubicSpline {
    /// Knot times (strictly increasing).
    pub times: Vec<f64>,
    /// Per-segment cubic coefficients: `coeffs[i] = [a, b, c, d]` per dimension.
    /// Length `n_intervals * x_dim * 4` stored as `[n_intervals][4 * x_dim]`.
    pub coeffs: Vec<Vec<f64>>,
    /// Observation dimension.
    pub x_dim: usize,
}

impl NaturalCubicSpline {
    /// Fit a natural cubic spline to (times, values).
    ///
    /// Falls back to linear interpolation when fewer than 3 knots are given.
    pub fn new(times: &[f64], values: &[Vec<f64>]) -> Self {
        let n = times.len();
        assert!(n >= 2, "need at least 2 knots");
        assert_eq!(n, values.len(), "times and values must have equal length");
        let x_dim = values[0].len();

        if n == 2 {
            // Linear segment
            let h = times[1] - times[0];
            let h_safe = if h.abs() < 1e-12 { 1e-12 } else { h };
            let mut seg = Vec::with_capacity(4 * x_dim);
            for d in 0..x_dim {
                let a = values[0][d];
                let slope = (values[1][d] - values[0][d]) / h_safe;
                // a + slope*(t-t0) + 0*(t-t0)^2 + 0*(t-t0)^3
                seg.push(a);
                seg.push(slope);
                seg.push(0.0);
                seg.push(0.0);
            }
            return Self {
                times: times.to_vec(),
                coeffs: vec![seg],
                x_dim,
            };
        }

        // Natural cubic spline: solve tridiagonal system for second derivatives M[i].
        // Natural BCs: M[0] = M[n-1] = 0.
        let n_segs = n - 1;
        let h: Vec<f64> = (0..n_segs)
            .map(|i| (times[i + 1] - times[i]).max(1e-12))
            .collect();
        let mut all_coeffs: Vec<Vec<f64>> = vec![Vec::new(); n_segs];
        for d in 0..x_dim {
            let y: Vec<f64> = values.iter().map(|v| v[d]).collect();
            let mut diag = vec![0.0f64; n];
            let mut upper = vec![0.0f64; n];
            let mut lower = vec![0.0f64; n];
            let mut rhs = vec![0.0f64; n];
            diag[0] = 1.0;
            diag[n - 1] = 1.0;
            for i in 1..(n - 1) {
                lower[i] = h[i - 1];
                diag[i] = 2.0 * (h[i - 1] + h[i]);
                upper[i] = h[i];
                rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
            }
            // Thomas algorithm
            let mut c_prime = vec![0.0f64; n];
            let mut d_prime = vec![0.0f64; n];
            c_prime[0] = if diag[0].abs() < 1e-14 {
                0.0
            } else {
                upper[0] / diag[0]
            };
            d_prime[0] = if diag[0].abs() < 1e-14 {
                0.0
            } else {
                rhs[0] / diag[0]
            };
            for i in 1..n {
                let denom = diag[i] - lower[i] * c_prime[i - 1];
                let ds = if denom.abs() < 1e-14 { 1e-14 } else { denom };
                c_prime[i] = if i < n - 1 { upper[i] / ds } else { 0.0 };
                d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / ds;
            }
            let mut m = vec![0.0f64; n];
            m[n - 1] = d_prime[n - 1];
            for i in (0..(n - 1)).rev() {
                m[i] = d_prime[i] - c_prime[i] * m[i + 1];
            }
            for i in 0..n_segs {
                let hi = h[i];
                let ai = y[i];
                let bi = (y[i + 1] - y[i]) / hi - hi * (2.0 * m[i] + m[i + 1]) / 6.0;
                let ci = m[i] / 2.0;
                let di = (m[i + 1] - m[i]) / (6.0 * hi);
                all_coeffs[i].extend_from_slice(&[ai, bi, ci, di]);
            }
        }

        Self {
            times: times.to_vec(),
            coeffs: all_coeffs,
            x_dim,
        }
    }

    /// Clamp time to valid range [t_min, t_max].
    pub fn clamp_time(&self, t: f64) -> f64 {
        let t_min = self.times.first().copied().unwrap_or(0.0);
        let t_max = self.times.last().copied().unwrap_or(1.0);
        t.clamp(t_min, t_max)
    }

    /// Find the segment index for time t (returns n_segs-1 at the right endpoint).
    fn segment_index(&self, t: f64) -> usize {
        let n = self.times.len();
        if n < 2 {
            return 0;
        }
        let mut lo = 0usize;
        let mut hi = n - 2;
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if t < self.times[mid] {
                hi = mid - 1;
            } else {
                lo = mid;
            }
        }
        lo
    }

    /// Evaluate X(t) via the cubic spline.
    pub fn evaluate(&self, t: f64) -> Vec<f64> {
        let tc = self.clamp_time(t);
        let i = self.segment_index(tc);
        let dt = tc - self.times[i];
        let mut out = Vec::with_capacity(self.x_dim);
        for d in 0..self.x_dim {
            let base = 4 * d;
            let a = self.coeffs[i].get(base).copied().unwrap_or(0.0);
            let b = self.coeffs[i].get(base + 1).copied().unwrap_or(0.0);
            let c = self.coeffs[i].get(base + 2).copied().unwrap_or(0.0);
            let dd = self.coeffs[i].get(base + 3).copied().unwrap_or(0.0);
            out.push(a + b * dt + c * dt * dt + dd * dt * dt * dt);
        }
        out
    }

    /// Evaluate dX/dt(t) (derivative of the spline).
    pub fn derivative(&self, t: f64) -> Vec<f64> {
        let tc = self.clamp_time(t);
        let i = self.segment_index(tc);
        let dt = tc - self.times[i];
        let mut out = Vec::with_capacity(self.x_dim);
        for d in 0..self.x_dim {
            let base = 4 * d;
            let b = self.coeffs[i].get(base + 1).copied().unwrap_or(0.0);
            let c = self.coeffs[i].get(base + 2).copied().unwrap_or(0.0);
            let dd = self.coeffs[i].get(base + 3).copied().unwrap_or(0.0);
            out.push(b + 2.0 * c * dt + 3.0 * dd * dt * dt);
        }
        out
    }
}

// Neural CDE

/// Configuration for the Neural CDE model.
#[derive(Debug, Clone)]
pub struct NeuralCdeConfig {
    /// Latent state dimension.
    pub z_dim: usize,
    /// Observation dimension.
    pub x_dim: usize,
    /// Hidden layer width.
    pub hidden_dim: usize,
    /// RK4 integration steps per time interval.
    pub n_steps: usize,
}

/// Vector field f(Z; θ): R^{z_dim} → R^{z_dim × x_dim}.
///
/// Used in the CDE: dZ = f(Z) dX.
#[derive(Debug, Clone)]
pub struct CdeVectorField {
    /// MLP mapping z_dim → z_dim * x_dim.
    pub mlp: NsdeMlp,
    /// Latent dimension.
    pub z_dim: usize,
    /// Observation dimension.
    pub x_dim: usize,
}

impl CdeVectorField {
    /// Build the vector field network.
    pub fn new(z_dim: usize, x_dim: usize, hidden_dim: usize) -> Self {
        let out_dim = z_dim * x_dim;
        let sizes = [z_dim, hidden_dim, hidden_dim, out_dim];
        Self {
            mlp: NsdeMlp::new(&sizes),
            z_dim,
            x_dim,
        }
    }

    /// Compute the flat [z_dim * x_dim] matrix output f(z).
    pub fn forward(&self, z: &[f64]) -> Vec<f64> {
        self.mlp.forward(z)
    }

    /// Compute f(z) * dx: mat-vec product [z_dim×x_dim] × \[x_dim\] → \[z_dim\].
    pub fn apply(&self, z: &[f64], dx: &[f64]) -> Vec<f64> {
        let mat = self.forward(z);
        let mut out = vec![0.0; self.z_dim];
        for zi in 0..self.z_dim {
            let mut acc = 0.0;
            for xi in 0..self.x_dim {
                let m = mat.get(zi * self.x_dim + xi).copied().unwrap_or(0.0);
                let dx_xi = dx.get(xi).copied().unwrap_or(0.0);
                acc += m * dx_xi;
            }
            out[zi] = acc;
        }
        out
    }
}

/// Neural Controlled Differential Equation (Kidger et al. 2020).
///
/// Model: dZ = f(Z; θ) dX, where X is the natural cubic spline of observations.
#[derive(Debug, Clone)]
pub struct NeuralCde {
    /// Vector field f(Z) → R^{z_dim × x_dim}.
    pub vector_field: CdeVectorField,
    /// Maps initial observation x_0 → initial state z_0.
    pub initial_net: NsdeMlp,
    /// Maps final state z_T → output.
    pub readout_net: NsdeMlp,
    /// Hyperparameters.
    pub config: NeuralCdeConfig,
}

impl NeuralCde {
    /// Construct a new NeuralCde.
    pub fn new(config: NeuralCdeConfig) -> Self {
        let z = config.z_dim;
        let x = config.x_dim;
        let h = config.hidden_dim;
        Self {
            vector_field: CdeVectorField::new(z, x, h),
            initial_net: NsdeMlp::new(&[x, h, z]),
            readout_net: NsdeMlp::new(&[z, h, z]),
            config,
        }
    }

    /// Integrate the CDE using RK4 over the full spline domain.
    ///
    /// Returns final latent state z_T.
    pub fn integrate(&self, spline: &NaturalCubicSpline, z0: &[f64]) -> Vec<f64> {
        let t_start = spline.times.first().copied().unwrap_or(0.0);
        let t_end = spline.times.last().copied().unwrap_or(1.0);
        let total_steps = self.config.n_steps.max(1);
        let dt = (t_end - t_start) / total_steps as f64;

        let mut z = z0.to_vec();
        for step in 0..total_steps {
            let t = t_start + step as f64 * dt;
            let half_dt = 0.5 * dt;
            let k1 = self.vector_field.apply(&z, &spline.derivative(t));
            let z2: Vec<f64> = z
                .iter()
                .zip(k1.iter())
                .map(|(&zi, &ki)| zi + half_dt * ki)
                .collect();
            let k2 = self
                .vector_field
                .apply(&z2, &spline.derivative(t + half_dt));
            let z3: Vec<f64> = z
                .iter()
                .zip(k2.iter())
                .map(|(&zi, &ki)| zi + half_dt * ki)
                .collect();
            let k3 = self
                .vector_field
                .apply(&z3, &spline.derivative(t + half_dt));
            let z4: Vec<f64> = z
                .iter()
                .zip(k3.iter())
                .map(|(&zi, &ki)| zi + dt * ki)
                .collect();
            let k4 = self.vector_field.apply(&z4, &spline.derivative(t + dt));
            z = (0..z.len())
                .map(|i| {
                    z[i] + dt / 6.0
                        * (k1.get(i).copied().unwrap_or(0.0)
                            + 2.0 * k2.get(i).copied().unwrap_or(0.0)
                            + 2.0 * k3.get(i).copied().unwrap_or(0.0)
                            + k4.get(i).copied().unwrap_or(0.0))
                })
                .collect();
        }
        z
    }

    /// Full forward pass: fit spline, compute z0, integrate, apply readout.
    pub fn forward(&self, times: &[f64], values: &[Vec<f64>]) -> Vec<f64> {
        let spline = NaturalCubicSpline::new(times, values);
        let x0 = match values.first() {
            Some(v) => v.as_slice(),
            None => return vec![0.0; self.config.z_dim],
        };
        let z0 = self.initial_net.forward(x0);
        let z_t = self.integrate(&spline, &z0);
        self.readout_net.forward(&z_t)
    }

    /// Classify a time series: forward → softmax argmax.
    pub fn classify(&self, times: &[f64], values: &[Vec<f64>], n_classes: usize) -> usize {
        let logits = self.forward(times, values);
        let n = n_classes.min(logits.len());
        if n == 0 {
            return 0;
        }
        let max_logit = logits[..n]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits[..n].iter().map(|&v| (v - max_logit).exp()).collect();
        let sum: f64 = exps.iter().sum::<f64>().max(1e-12);
        exps.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let av = *a / sum;
                let bv = *b / sum;
                av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// Path Signature

/// Configuration for path signature computation.
#[derive(Debug, Clone)]
pub struct SignatureConfig {
    /// Truncation order M (compute levels 0..=M).
    pub truncation_order: usize,
    /// Ambient path dimension.
    pub x_dim: usize,
}

/// Computes the truncated signature of a path via Chen's identity.
#[derive(Debug, Clone)]
pub struct PathSignature {
    /// Configuration.
    pub config: SignatureConfig,
}

impl PathSignature {
    /// Construct a PathSignature computer.
    pub fn new(x_dim: usize, truncation_order: usize) -> Self {
        Self {
            config: SignatureConfig {
                truncation_order,
                x_dim,
            },
        }
    }

    /// Total number of terms in the truncated signature.
    ///
    /// = Σ_{k=0}^{M} d^k
    pub fn signature_dim(&self) -> usize {
        let d = self.config.x_dim;
        let m = self.config.truncation_order;
        if d == 1 {
            return m + 1;
        }
        // (d^{M+1} - 1) / (d - 1)
        let mut total = 1usize;
        let mut power = 1usize;
        for _ in 0..m {
            power = power.saturating_mul(d);
            total = total.saturating_add(power);
        }
        total
    }

    /// Compute the truncated signature: levels 0..=M concatenated.
    pub fn compute(&self, path: &[Vec<f64>]) -> Vec<f64> {
        let d = self.config.x_dim;
        let m = self.config.truncation_order;
        let n = path.len();
        if n == 0 {
            return vec![0.0; self.signature_dim()];
        }
        // Increments delta[t][i] = path[t+1][i] - path[t][i]
        let increments: Vec<Vec<f64>> = (0..n.saturating_sub(1))
            .map(|t| {
                (0..d)
                    .map(|i| {
                        path.get(t + 1)
                            .and_then(|v| v.get(i))
                            .copied()
                            .unwrap_or(0.0)
                            - path.get(t).and_then(|v| v.get(i)).copied().unwrap_or(0.0)
                    })
                    .collect()
            })
            .collect();
        let mut level_sigs: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        level_sigs.push(vec![1.0]); // level 0
        if m == 0 {
            return vec![1.0];
        }
        // Level 1: sum increments
        let mut level1 = vec![0.0; d];
        for delta in &increments {
            for i in 0..d {
                level1[i] += delta.get(i).copied().unwrap_or(0.0);
            }
        }
        level_sigs.push(level1);

        // Higher levels via Chen's recurrence:
        // S^{i1..ik}_{0,T} = Σ_t S^{i1..i(k-1)}_{0,t} * ΔX^{ik}_t
        for k in 2..=m {
            let prev_dim = level_sigs[k - 1].len(); // d^{k-1}
            let curr_dim = prev_dim * d;
            let mut curr_level = vec![0.0; curr_dim];
            let mut running_prev = vec![0.0; prev_dim];
            let prev_dim_lev1 = d;
            let mut running_all: Vec<Vec<f64>> = vec![vec![0.0; prev_dim]];
            {
                let mut all_running: Vec<Vec<Vec<f64>>> = Vec::new();
                let mut run1 = vec![vec![0.0; prev_dim_lev1]];
                for delta in &increments {
                    let default_r = vec![0.0; prev_dim_lev1];
                    let prev_r = run1.last().unwrap_or(&default_r);
                    let mut next_r = prev_r.clone();
                    for i in 0..d {
                        next_r[i] += delta.get(i).copied().unwrap_or(0.0);
                    }
                    run1.push(next_r);
                }
                all_running.push(run1); // index 0 = level 1 running

                // Levels 2..k-1
                for lev in 2..k {
                    let prev_lev_idx = lev - 2;
                    let prev_dim_here = d.pow((lev - 1) as u32);
                    let curr_dim_here = prev_dim_here * d;
                    let mut run_lev = vec![vec![0.0; curr_dim_here]];
                    let n_steps = increments.len();
                    for t in 0..n_steps {
                        let prev_run_t = all_running[prev_lev_idx]
                            .get(t)
                            .cloned()
                            .unwrap_or_default();
                        let prev_run_here = run_lev
                            .last()
                            .cloned()
                            .unwrap_or_else(|| vec![0.0; curr_dim_here]);
                        let mut next_r = prev_run_here.clone();
                        let delta = &increments[t];
                        for prev_idx in 0..prev_dim_here {
                            for i_last in 0..d {
                                let curr_idx = prev_idx * d + i_last;
                                next_r[curr_idx] +=
                                    prev_run_t.get(prev_idx).copied().unwrap_or(0.0)
                                        * delta.get(i_last).copied().unwrap_or(0.0);
                            }
                        }
                        run_lev.push(next_r);
                    }
                    all_running.push(run_lev);
                }

                if k == 2 {
                    running_all = all_running.into_iter().next().unwrap_or_default();
                } else {
                    running_all = all_running.last().cloned().unwrap_or_default();
                }
                running_prev = running_all.last().cloned().unwrap_or_default();
            }

            let n_steps = increments.len();
            for t in 0..n_steps {
                let prev_run_t = running_all
                    .get(t)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; prev_dim]);
                let delta = &increments[t];
                for prev_idx in 0..prev_dim {
                    for i_last in 0..d {
                        let curr_idx = prev_idx * d + i_last;
                        curr_level[curr_idx] += prev_run_t.get(prev_idx).copied().unwrap_or(0.0)
                            * delta.get(i_last).copied().unwrap_or(0.0);
                    }
                }
            }

            let _ = running_prev; // suppress warning
            level_sigs.push(curr_level);
        }

        // Concatenate all levels
        level_sigs.into_iter().flatten().collect()
    }

    /// Compute the log-signature (Lie algebra projection for order ≤ 2).
    ///
    /// For M=1: returns level-1 terms (d elements).
    /// For M=2: returns level-1 terms + antisymmetric bracket terms d*(d-1)/2.
    /// Total dimension for M=2: d + d*(d-1)/2.
    pub fn compute_log_signature(&self, path: &[Vec<f64>]) -> Vec<f64> {
        let d = self.config.x_dim;
        let m = self.config.truncation_order;
        let sig = self.compute(path);

        // Level 1 elements (offset 1 to skip level-0 scalar)
        let level1: Vec<f64> = sig.iter().skip(1).take(d).cloned().collect();

        if m < 2 {
            return level1;
        }

        // Level 2: log-sig bracket [e_i, e_j] = S^{ij} - S^{ji} for i < j
        let level2_offset = 1 + d;
        let mut brackets = Vec::new();
        for i in 0..d {
            for j in (i + 1)..d {
                let s_ij = sig.get(level2_offset + i * d + j).copied().unwrap_or(0.0);
                let s_ji = sig.get(level2_offset + j * d + i).copied().unwrap_or(0.0);
                brackets.push(s_ij - s_ji);
            }
        }

        let mut result = level1;
        result.extend(brackets);
        result
    }
}
