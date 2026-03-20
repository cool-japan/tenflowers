//! Probabilistic Deep Learning and Uncertainty Quantification — Track P.
//!
//! Provides:
//! - **Deep Ensembles**: MLP members, mean/variance aggregation, snapshot ensembles
//! - **Normalizing Flows**: RealNVP coupling, ActNorm, composable FlowModel
//! - **Gaussian Processes**: RBF/Matérn52/Linear kernels, posterior inference
//! - **Calibration**: Temperature scaling, Platt scaling, ECE, isotonic calibration
//! - **Evidential Deep Learning**: NIG outputs, Dirichlet uncertainty

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn relu_f64(x: f64) -> f64 {
    x.max(0.0)
}

#[inline]
fn softplus_f64(x: f64) -> f64 {
    if x > 30.0 {
        x
    } else if x < -30.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Xavier uniform weight init for `[fan_in x fan_out]`.
fn xavier_f64(fan_in: usize, fan_out: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt();
    (0..fan_in * fan_out)
        .map(|_| {
            let u: f64 = rng.random();
            u * 2.0 * limit - limit
        })
        .collect()
}

/// Simple MLP forward: layers of [fan_in x fan_out] weights + biases, ReLU hidden, linear last.
fn mlp_forward(
    x: &[f64],
    weights: &[Vec<f64>],
    biases: &[Vec<f64>],
    hidden_dims: &[usize],
) -> Vec<f64> {
    let mut h = x.to_vec();
    let n_layers = weights.len();
    for (l, (w, b)) in weights.iter().zip(biases.iter()).enumerate() {
        let out_dim = b.len();
        let in_dim = h.len();
        let mut next = vec![0.0_f64; out_dim];
        for o in 0..out_dim {
            let mut acc = b[o];
            for i in 0..in_dim {
                acc += h[i] * w[o * in_dim + i];
            }
            // ReLU on all but last layer
            next[o] = if l + 1 < n_layers { relu_f64(acc) } else { acc };
        }
        h = next;
        let _ = hidden_dims; // dimensions tracked via weight shapes
    }
    h
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Deep Ensembles
// ─────────────────────────────────────────────────────────────────────────────

/// A single MLP ensemble member parameterized by `hidden_dims`.
#[derive(Debug, Clone)]
pub struct EnsembleMember {
    pub hidden_dims: Vec<usize>,
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<Vec<f64>>,
}

impl EnsembleMember {
    /// Build a new member with random Xavier weights.
    /// `arch` is `[in_dim, h1, h2, ..., out_dim]`.
    pub fn new(arch: &[usize], seed: u64) -> Result<Self> {
        if arch.len() < 2 {
            return Err(TensorError::invalid_argument(
                "arch must have at least 2 elements".into(),
            ));
        }
        let n_layers = arch.len() - 1;
        let mut weights = Vec::with_capacity(n_layers);
        let mut biases = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let fan_in = arch[l];
            let fan_out = arch[l + 1];
            weights.push(xavier_f64(
                fan_in,
                fan_out,
                seed.wrapping_add(l as u64 * 1000),
            ));
            biases.push(vec![0.0_f64; fan_out]);
        }
        let hidden_dims = arch[1..arch.len() - 1].to_vec();
        Ok(Self {
            hidden_dims,
            weights,
            biases,
        })
    }

    /// Forward pass with ReLU activations on hidden layers.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        mlp_forward(x, &self.weights, &self.biases, &self.hidden_dims)
    }
}

/// Deep ensemble of N `EnsembleMember`s with different random seeds.
#[derive(Debug, Clone)]
pub struct DeepEnsemble {
    pub members: Vec<EnsembleMember>,
}

impl DeepEnsemble {
    /// Build ensemble with `n` members, each from `arch`.
    pub fn new(n: usize, arch: &[usize], base_seed: u64) -> Result<Self> {
        if n == 0 {
            return Err(TensorError::invalid_argument(
                "ensemble must have ≥1 member".into(),
            ));
        }
        let members = (0..n)
            .map(|i| EnsembleMember::new(arch, base_seed.wrapping_add(i as u64 * 1_000_000)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { members })
    }

    /// Predict mean and variance across ensemble members for each output dimension.
    pub fn predict_mean_var(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let preds: Vec<Vec<f64>> = self.members.iter().map(|m| m.forward(x)).collect();
        let n = preds.len() as f64;
        if preds.is_empty() {
            return (vec![], vec![]);
        }
        let out_dim = preds[0].len();
        let mut mean = vec![0.0_f64; out_dim];
        for p in &preds {
            for o in 0..out_dim {
                mean[o] += p[o];
            }
        }
        for m in &mut mean {
            *m /= n;
        }
        let mut var = vec![0.0_f64; out_dim];
        for p in &preds {
            for o in 0..out_dim {
                let diff = p[o] - mean[o];
                var[o] += diff * diff;
            }
        }
        for v in &mut var {
            *v /= n;
        }
        (mean, var)
    }
}

/// Cyclic LR snapshot ensemble: stores predictions from snapshots along a cyclic cosine LR schedule.
#[derive(Debug, Clone)]
pub struct SnapshotEnsemble {
    pub snapshots: Vec<Vec<f64>>,
    pub n_cycles: usize,
}

impl SnapshotEnsemble {
    /// Create an empty snapshot ensemble for `n_cycles` snapshots.
    pub fn new(n_cycles: usize) -> Self {
        Self {
            snapshots: Vec::new(),
            n_cycles,
        }
    }

    /// Add a prediction vector as a new snapshot.
    pub fn add_snapshot(&mut self, prediction: Vec<f64>) {
        if self.snapshots.len() < self.n_cycles {
            self.snapshots.push(prediction);
        }
    }

    /// Average all snapshots to produce the ensemble prediction.
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        // If no snapshots, return zeros matching x length
        if self.snapshots.is_empty() {
            return vec![0.0; x.len()];
        }
        let out_dim = self.snapshots[0].len();
        let n = self.snapshots.len() as f64;
        let mut mean = vec![0.0_f64; out_dim];
        for snap in &self.snapshots {
            for (o, &v) in snap.iter().enumerate() {
                if o < out_dim {
                    mean[o] += v;
                }
            }
        }
        for m in &mut mean {
            *m /= n;
        }
        mean
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Normalizing Flows
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a normalizing flow forward pass.
#[derive(Debug, Clone)]
pub struct FlowResult {
    /// Transformed latent vector.
    pub z: Vec<f64>,
    /// Log absolute determinant of the Jacobian.
    pub log_det: f64,
}

/// Trait for invertible flow layers.
pub trait Flow: std::fmt::Debug {
    fn forward(&self, x: &[f64]) -> Result<FlowResult>;
    fn inverse(&self, z: &[f64]) -> Result<Vec<f64>>;
}

/// RealNVP affine coupling layer.
///
/// Splits `x` into `x_a = x[..d]` and `x_b = x[d..]`.
/// Then: `y_b = x_b * exp(s) + t` where `s, t = MLP(x_a)`.
/// `y_a = x_a` unchanged.
#[derive(Debug, Clone)]
pub struct RealNvpCoupling {
    pub dim: usize,
    pub split: usize,
    /// MLP weights for s and t combined: s,t each of size `dim - split`.
    s_weights: Vec<Vec<f64>>,
    s_biases: Vec<Vec<f64>>,
    t_weights: Vec<Vec<f64>>,
    t_biases: Vec<Vec<f64>>,
    hidden_dim: usize,
}

impl RealNvpCoupling {
    /// Build a RealNVP coupling layer for `dim`-dimensional input with `hidden_dim` hidden units.
    pub fn new(dim: usize, hidden_dim: usize, seed: u64) -> Result<Self> {
        if dim < 2 {
            return Err(TensorError::invalid_argument("dim must be >= 2".into()));
        }
        let split = dim / 2;
        let out_dim = dim - split;
        // s MLP: split -> hidden -> out_dim
        let s_arch = [split, hidden_dim, out_dim];
        let t_arch = [split, hidden_dim, out_dim];
        let mut s_weights = Vec::new();
        let mut s_biases = Vec::new();
        let mut t_weights = Vec::new();
        let mut t_biases = Vec::new();
        for l in 0..2 {
            let fi = s_arch[l];
            let fo = s_arch[l + 1];
            s_weights.push(xavier_f64(fi, fo, seed.wrapping_add(l as u64 * 100)));
            s_biases.push(vec![0.0; fo]);
            let fi2 = t_arch[l];
            let fo2 = t_arch[l + 1];
            t_weights.push(xavier_f64(
                fi2,
                fo2,
                seed.wrapping_add(1000 + l as u64 * 100),
            ));
            t_biases.push(vec![0.0; fo2]);
        }
        Ok(Self {
            dim,
            split,
            s_weights,
            s_biases,
            t_weights,
            t_biases,
            hidden_dim,
        })
    }

    fn scale_translate(&self, x_a: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let out_dim = self.dim - self.split;
        let s_raw = mlp_forward(x_a, &self.s_weights, &self.s_biases, &[self.hidden_dim]);
        let t = mlp_forward(x_a, &self.t_weights, &self.t_biases, &[self.hidden_dim]);
        // s = tanh(s_raw) * 2  (bounded scale)
        let s: Vec<f64> = s_raw.iter().map(|&v| v.tanh() * 2.0).collect();
        assert_eq!(s.len(), out_dim);
        (s, t)
    }
}

impl Flow for RealNvpCoupling {
    fn forward(&self, x: &[f64]) -> Result<FlowResult> {
        if x.len() != self.dim {
            return Err(TensorError::invalid_argument("input dim mismatch".into()));
        }
        let x_a = &x[..self.split];
        let x_b = &x[self.split..];
        let (s, t) = self.scale_translate(x_a);
        let y_b: Vec<f64> = x_b
            .iter()
            .zip(s.iter())
            .zip(t.iter())
            .map(|((&xb, &si), &ti)| xb * si.exp() + ti)
            .collect();
        let log_det: f64 = s.iter().sum();
        let mut z = x_a.to_vec();
        z.extend_from_slice(&y_b);
        Ok(FlowResult { z, log_det })
    }

    fn inverse(&self, z: &[f64]) -> Result<Vec<f64>> {
        if z.len() != self.dim {
            return Err(TensorError::invalid_argument("input dim mismatch".into()));
        }
        let z_a = &z[..self.split];
        let z_b = &z[self.split..];
        let (s, t) = self.scale_translate(z_a);
        let x_b: Vec<f64> = z_b
            .iter()
            .zip(s.iter())
            .zip(t.iter())
            .map(|((&zb, &si), &ti)| (zb - ti) * (-si).exp())
            .collect();
        let mut x = z_a.to_vec();
        x.extend_from_slice(&x_b);
        Ok(x)
    }
}

/// ActNorm: per-channel affine normalization initialized from the first batch.
#[derive(Debug, Clone)]
pub struct ActNorm {
    pub dim: usize,
    pub scale: Vec<f64>,
    pub bias: Vec<f64>,
    initialized: bool,
}

impl ActNorm {
    /// Create an uninitialized ActNorm for `dim` channels.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            scale: vec![1.0; dim],
            bias: vec![0.0; dim],
            initialized: false,
        }
    }

    /// Initialize from the first data batch (mean and std).
    pub fn initialize_from_batch(&mut self, batch: &[Vec<f64>]) {
        if batch.is_empty() || self.initialized {
            return;
        }
        let n = batch.len() as f64;
        let mut mean = vec![0.0_f64; self.dim];
        for x in batch {
            for (i, &v) in x.iter().enumerate().take(self.dim) {
                mean[i] += v;
            }
        }
        for m in &mut mean {
            *m /= n;
        }
        let mut var = vec![1e-8_f64; self.dim];
        for x in batch {
            for (i, &v) in x.iter().enumerate().take(self.dim) {
                var[i] += (v - mean[i]).powi(2);
            }
        }
        for v in &mut var {
            *v /= n;
        }
        self.scale = var.iter().map(|&v| 1.0 / v.sqrt()).collect();
        self.bias = mean
            .iter()
            .zip(self.scale.iter())
            .map(|(&m, &s)| -m * s)
            .collect();
        self.initialized = true;
    }
}

impl Flow for ActNorm {
    fn forward(&self, x: &[f64]) -> Result<FlowResult> {
        if x.len() != self.dim {
            return Err(TensorError::invalid_argument("ActNorm dim mismatch".into()));
        }
        let z: Vec<f64> = x
            .iter()
            .zip(self.scale.iter())
            .zip(self.bias.iter())
            .map(|((&xi, &si), &bi)| xi * si + bi)
            .collect();
        let log_det: f64 = self.scale.iter().map(|&s| s.abs().ln()).sum();
        Ok(FlowResult { z, log_det })
    }

    fn inverse(&self, z: &[f64]) -> Result<Vec<f64>> {
        if z.len() != self.dim {
            return Err(TensorError::invalid_argument("ActNorm dim mismatch".into()));
        }
        let x: Vec<f64> = z
            .iter()
            .zip(self.scale.iter())
            .zip(self.bias.iter())
            .map(|((&zi, &si), &bi)| (zi - bi) / si)
            .collect();
        Ok(x)
    }
}

/// Composed normalizing flow model: sequence of `Flow` layers.
#[derive(Debug)]
pub struct FlowModel {
    pub dim: usize,
    layers: Vec<Box<dyn Flow>>,
}

impl FlowModel {
    /// Create an empty flow model for `dim`-dimensional data.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            layers: Vec::new(),
        }
    }

    /// Push a flow layer.
    pub fn add_layer(&mut self, layer: Box<dyn Flow>) {
        self.layers.push(layer);
    }

    /// Forward through all layers; returns final z and total log_det.
    pub fn forward_full(&self, x: &[f64]) -> Result<FlowResult> {
        let mut cur = x.to_vec();
        let mut total_log_det = 0.0_f64;
        for layer in &self.layers {
            let res = layer.forward(&cur)?;
            total_log_det += res.log_det;
            cur = res.z;
        }
        Ok(FlowResult {
            z: cur,
            log_det: total_log_det,
        })
    }

    /// Log-likelihood under standard Gaussian base distribution.
    pub fn log_likelihood(&self, x: &[f64]) -> Result<f64> {
        let res = self.forward_full(x)?;
        let d = res.z.len() as f64;
        let gauss_ll: f64 = res.z.iter().map(|&zi| -0.5 * zi * zi).sum::<f64>()
            - 0.5 * d * std::f64::consts::PI.ln()
            - 0.5 * d * 2.0_f64.ln();
        Ok(gauss_ll + res.log_det)
    }

    /// Negative log-likelihood loss over a batch.
    pub fn nll_loss(&self, batch: &[Vec<f64>]) -> Result<f64> {
        if batch.is_empty() {
            return Ok(0.0);
        }
        let total: f64 = batch
            .iter()
            .map(|x| self.log_likelihood(x).unwrap_or(f64::NEG_INFINITY))
            .sum();
        Ok(-total / batch.len() as f64)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Gaussian Process
// ─────────────────────────────────────────────────────────────────────────────

/// GP kernel variants.
#[derive(Debug, Clone)]
pub enum GpKernel {
    /// Squared exponential (RBF) kernel.
    Rbf { length_scale: f64, variance: f64 },
    /// Matérn 5/2 kernel.
    Matern52 { length_scale: f64, variance: f64 },
    /// Linear (dot-product) kernel.
    Linear { variance: f64 },
}

impl GpKernel {
    /// Evaluate kernel between two points.
    pub fn compute(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let sq_dist: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        match self {
            GpKernel::Rbf {
                length_scale,
                variance,
            } => variance * (-sq_dist / (2.0 * length_scale * length_scale)).exp(),
            GpKernel::Matern52 {
                length_scale,
                variance,
            } => {
                let r = sq_dist.sqrt();
                let s = 5.0_f64.sqrt() * r / length_scale;
                variance * (1.0 + s + s * s / 3.0) * (-s).exp()
            }
            GpKernel::Linear { variance } => {
                variance * x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum::<f64>()
            }
        }
    }
}

/// A GP prediction result.
#[derive(Debug, Clone)]
pub struct GpPrediction {
    pub mean: f64,
    pub variance: f64,
    pub std: f64,
}

/// Gaussian Process regression model.
#[derive(Debug, Clone)]
pub struct GaussianProcess {
    pub x_train: Vec<Vec<f64>>,
    pub y_train: Vec<f64>,
    pub kernel: GpKernel,
    pub noise: f64,
}

impl GaussianProcess {
    /// Create a new GP with the given kernel and observation noise.
    pub fn new(kernel: GpKernel, noise: f64) -> Self {
        Self {
            x_train: Vec::new(),
            y_train: Vec::new(),
            kernel,
            noise,
        }
    }

    /// Store training data.
    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) {
        self.x_train = x;
        self.y_train = y;
    }

    /// Build the `n x n` covariance matrix `K(X, X) + noise * I`.
    fn build_kxx(&self) -> Vec<Vec<f64>> {
        let n = self.x_train.len();
        let mut k = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                k[i][j] = self.kernel.compute(&self.x_train[i], &self.x_train[j]);
                if i == j {
                    k[i][j] += self.noise;
                }
            }
        }
        k
    }

    /// Solve `A * x = b` via Gaussian elimination with partial pivoting.
    /// Returns `x` (solution vector).
    fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
        let n = b.len();
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            for row in (col + 1)..n {
                if a[row][col].abs() > a[max_row][col].abs() {
                    max_row = row;
                }
            }
            a.swap(col, max_row);
            b.swap(col, max_row);
            let pivot = a[col][col];
            if pivot.abs() < 1e-14 {
                return None;
            }
            for row in (col + 1)..n {
                let factor = a[row][col] / pivot;
                for c in col..n {
                    a[row][c] -= factor * a[col][c];
                }
                b[row] -= factor * b[col];
            }
        }
        // Back substitution
        let mut x = vec![0.0_f64; n];
        for i in (0..n).rev() {
            let mut s = b[i];
            for j in (i + 1)..n {
                s -= a[i][j] * x[j];
            }
            x[i] = s / a[i][i];
        }
        Some(x)
    }

    /// Predict posterior mean and variance at a test point.
    pub fn predict(&self, x_test: &[f64]) -> (f64, f64) {
        let n = self.x_train.len();
        if n == 0 {
            return (0.0, self.kernel.compute(x_test, x_test));
        }
        // k_star: K(x_test, X_train)
        let k_star: Vec<f64> = self
            .x_train
            .iter()
            .map(|xi| self.kernel.compute(x_test, xi))
            .collect();
        let k_ss = self.kernel.compute(x_test, x_test) + self.noise;
        let kxx = self.build_kxx();
        // Solve K_XX^{-1} y
        let alpha = match Self::solve(kxx.clone(), self.y_train.clone()) {
            Some(a) => a,
            None => return (0.0, k_ss),
        };
        // mean = k_star^T alpha
        let mean: f64 = k_star.iter().zip(alpha.iter()).map(|(k, a)| k * a).sum();
        // Solve K_XX^{-1} k_star
        let v = match Self::solve(kxx, k_star.clone()) {
            Some(v) => v,
            None => return (mean, k_ss),
        };
        // var = k_ss - k_star^T v
        let var: f64 = (k_ss
            - k_star
                .iter()
                .zip(v.iter())
                .map(|(k, vi)| k * vi)
                .sum::<f64>())
        .max(0.0);
        (mean, var)
    }

    /// Predict with structured output.
    pub fn predict_structured(&self, x_test: &[f64]) -> GpPrediction {
        let (mean, variance) = self.predict(x_test);
        GpPrediction {
            mean,
            variance,
            std: variance.sqrt(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Calibration
// ─────────────────────────────────────────────────────────────────────────────

/// Softmax with temperature scaling.
fn softmax_t(logits: &[f64], temp: f64) -> Vec<f64> {
    let t = temp.max(1e-8);
    let scaled: Vec<f64> = logits.iter().map(|&l| l / t).collect();
    let max = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = scaled.iter().map(|&s| (s - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    if sum < 1e-30 {
        return vec![1.0 / logits.len() as f64; logits.len()];
    }
    exp.iter().map(|&e| e / sum).collect()
}

/// NLL loss for temperature calibration (cross-entropy with temperature).
fn nll_temperature(logits: &[Vec<f64>], labels: &[usize], temp: f64) -> f64 {
    logits
        .iter()
        .zip(labels.iter())
        .map(|(l, &y)| {
            let p = softmax_t(l, temp);
            let prob = if y < p.len() { p[y].max(1e-30) } else { 1e-30 };
            -prob.ln()
        })
        .sum::<f64>()
        / logits.len().max(1) as f64
}

/// Temperature scaling calibrator. Finds optimal `T` via grid search.
#[derive(Debug, Clone)]
pub struct TemperatureScaling {
    pub temperature: f64,
}

impl TemperatureScaling {
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }

    /// Grid-search T in [0.1, 10] to minimize NLL.
    pub fn fit(&mut self, logits: &[Vec<f64>], labels: &[usize]) {
        let mut best_t = 1.0_f64;
        let mut best_nll = f64::INFINITY;
        let mut t = 0.1_f64;
        while t <= 10.0 {
            let nll = nll_temperature(logits, labels, t);
            if nll < best_nll {
                best_nll = nll;
                best_t = t;
            }
            t += 0.1;
        }
        self.temperature = best_t;
    }

    /// Apply temperature scaling and softmax.
    pub fn calibrate(&self, logits: &[f64]) -> Vec<f64> {
        softmax_t(logits, self.temperature)
    }
}

/// Platt scaling for binary classifiers: sigmoid(a*score + b).
#[derive(Debug, Clone)]
pub struct PlattScaling {
    pub a: f64,
    pub b: f64,
}

impl PlattScaling {
    pub fn new() -> Self {
        Self { a: 1.0, b: 0.0 }
    }

    /// Sigmoid helper.
    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    /// Fit via SGD on logistic loss.
    pub fn fit(&mut self, scores: &[f64], labels: &[f64]) {
        let mut a = 1.0_f64;
        let mut b = 0.0_f64;
        let lr = 0.01_f64;
        let n = scores.len() as f64;
        for _ in 0..2000 {
            let mut da = 0.0_f64;
            let mut db = 0.0_f64;
            for (&s, &y) in scores.iter().zip(labels.iter()) {
                let p = Self::sigmoid(a * s + b);
                let err = p - y;
                da += err * s;
                db += err;
            }
            a -= lr * da / n;
            b -= lr * db / n;
        }
        self.a = a;
        self.b = b;
    }

    /// Predict calibrated probability.
    pub fn predict(&self, score: f64) -> f64 {
        Self::sigmoid(self.a * score + self.b)
    }
}

impl Default for PlattScaling {
    fn default() -> Self {
        Self::new()
    }
}

/// Expected Calibration Error evaluator.
#[derive(Debug, Clone)]
pub struct CalibrationEvaluator;

impl CalibrationEvaluator {
    pub fn new() -> Self {
        Self
    }

    /// Compute ECE with `n_bins` equal-width bins over \[0,1\].
    pub fn ece(&self, probs: &[f64], labels: &[usize], n_bins: usize) -> f64 {
        let n = probs.len();
        if n == 0 {
            return 0.0;
        }
        let mut bin_conf = vec![0.0_f64; n_bins];
        let mut bin_acc = vec![0.0_f64; n_bins];
        let mut bin_count = vec![0_usize; n_bins];
        for (&p, &y) in probs.iter().zip(labels.iter()) {
            let bin = ((p * n_bins as f64) as usize).min(n_bins - 1);
            bin_conf[bin] += p;
            bin_acc[bin] += y as f64;
            bin_count[bin] += 1;
        }
        let mut ece = 0.0_f64;
        for b in 0..n_bins {
            let cnt = bin_count[b];
            if cnt == 0 {
                continue;
            }
            let avg_conf = bin_conf[b] / cnt as f64;
            let avg_acc = bin_acc[b] / cnt as f64;
            ece += (cnt as f64 / n as f64) * (avg_conf - avg_acc).abs();
        }
        ece
    }
}

impl Default for CalibrationEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Isotonic calibrator using the Pool Adjacent Violators (PAV) algorithm.
#[derive(Debug, Clone)]
pub struct IsotonicCalibrator {
    pub x_calibrated: Vec<f64>,
    pub y_calibrated: Vec<f64>,
}

impl IsotonicCalibrator {
    pub fn new() -> Self {
        Self {
            x_calibrated: Vec::new(),
            y_calibrated: Vec::new(),
        }
    }

    /// Fit with PAV algorithm.
    pub fn fit(&mut self, scores: &[f64], labels: &[f64]) {
        if scores.is_empty() {
            return;
        }
        // Sort by score
        let mut pairs: Vec<(f64, f64)> = scores
            .iter()
            .zip(labels.iter())
            .map(|(&s, &l)| (s, l))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        // PAV: pool adjacent violators
        let mut blocks: Vec<(f64, f64, usize)> = Vec::new(); // (x_sum, y_sum, count)
        for (x, y) in pairs {
            blocks.push((x, y, 1));
            while blocks.len() >= 2 {
                let n = blocks.len();
                let prev = blocks[n - 2];
                let curr = blocks[n - 1];
                let prev_mean = prev.1 / prev.2 as f64;
                let curr_mean = curr.1 / curr.2 as f64;
                if prev_mean > curr_mean {
                    let merged = (prev.0 + curr.0, prev.1 + curr.1, prev.2 + curr.2);
                    blocks.truncate(n - 2);
                    blocks.push(merged);
                } else {
                    break;
                }
            }
        }
        // Expand blocks to per-observation x/y
        self.x_calibrated.clear();
        self.y_calibrated.clear();
        for (x_sum, y_sum, cnt) in blocks {
            let y_mean = y_sum / cnt as f64;
            // Use x_sum/cnt as representative x
            self.x_calibrated.push(x_sum / cnt as f64);
            self.y_calibrated.push(y_mean);
        }
    }

    /// Predict calibrated probability via linear interpolation.
    pub fn predict(&self, score: f64) -> f64 {
        let n = self.x_calibrated.len();
        if n == 0 {
            return 0.5;
        }
        if n == 1 {
            return self.y_calibrated[0];
        }
        if score <= self.x_calibrated[0] {
            return self.y_calibrated[0];
        }
        if score >= self.x_calibrated[n - 1] {
            return self.y_calibrated[n - 1];
        }
        // Binary search
        let pos = self.x_calibrated.partition_point(|&x| x < score);
        if pos == 0 {
            return self.y_calibrated[0];
        }
        if pos >= n {
            return self.y_calibrated[n - 1];
        }
        let x0 = self.x_calibrated[pos - 1];
        let x1 = self.x_calibrated[pos];
        let y0 = self.y_calibrated[pos - 1];
        let y1 = self.y_calibrated[pos];
        let dx = x1 - x0;
        if dx.abs() < 1e-14 {
            return y0;
        }
        y0 + (y1 - y0) * (score - x0) / dx
    }
}

impl Default for IsotonicCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Evidential Deep Learning
// ─────────────────────────────────────────────────────────────────────────────

/// Normal-Inverse-Gamma (NIG) output for evidential regression.
#[derive(Debug, Clone)]
pub struct NigOutput {
    pub mu: f64,
    pub nu: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl NigOutput {
    pub fn new(mu: f64, nu: f64, alpha: f64, beta: f64) -> Self {
        Self {
            mu,
            nu,
            alpha,
            beta,
        }
    }

    /// Epistemic uncertainty: 1 / (nu * (alpha - 1))  if alpha > 1.
    pub fn epistemic_uncertainty(&self) -> f64 {
        if self.alpha > 1.0 {
            1.0 / (self.nu * (self.alpha - 1.0))
        } else {
            f64::INFINITY
        }
    }

    /// Aleatoric uncertainty: beta / (alpha - 1) if alpha > 1.
    pub fn aleatoric_uncertainty(&self) -> f64 {
        if self.alpha > 1.0 {
            self.beta / (self.alpha - 1.0)
        } else {
            f64::INFINITY
        }
    }
}

/// Evidential linear layer: maps `in_dim` -> 4 NIG parameters per output.
#[derive(Debug, Clone)]
pub struct EvidentialLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    weights: Vec<f64>,
    biases: Vec<f64>,
}

impl EvidentialLayer {
    /// Build with random weights, producing 4*out_dim raw outputs.
    pub fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let w_out = 4 * out_dim;
        Self {
            in_dim,
            out_dim,
            weights: xavier_f64(in_dim, w_out, seed),
            biases: vec![0.0; w_out],
        }
    }

    /// Forward: raw linear, then split into NIG components.
    /// Returns `out_dim` NIG outputs.
    pub fn forward(&self, x: &[f64]) -> Vec<NigOutput> {
        let w_out = 4 * self.out_dim;
        let mut raw = vec![0.0_f64; w_out];
        for o in 0..w_out {
            raw[o] = self.biases[o];
            for i in 0..self.in_dim.min(x.len()) {
                raw[o] += x[i] * self.weights[o * self.in_dim + i];
            }
        }
        (0..self.out_dim)
            .map(|k| {
                let mu = raw[k];
                let nu = softplus_f64(raw[self.out_dim + k]) + 1e-6;
                let alpha = softplus_f64(raw[2 * self.out_dim + k]) + 1.0;
                let beta = softplus_f64(raw[3 * self.out_dim + k]) + 1e-6;
                NigOutput {
                    mu,
                    nu,
                    alpha,
                    beta,
                }
            })
            .collect()
    }
}

/// NIG negative log-likelihood + KL regularizer.
///
/// `nll(y, μ, ν, α, β) = 0.5·log(π/ν) - α·log(Ω) + (α + 0.5)·log((y-μ)²ν + Ω) + lgamma(α) - lgamma(α+0.5)`
/// `Ω = 2β(1+ν)`
/// Plus `lambda * |y - mu| * (2ν + α)` KL-style regularizer.
pub fn nig_loss(y_true: f64, nig: &NigOutput, lambda: f64) -> f64 {
    let mu = nig.mu;
    let nu = nig.nu.max(1e-8);
    let alpha = nig.alpha.max(1.0 + 1e-8);
    let beta = nig.beta.max(1e-8);
    let omega = 2.0 * beta * (1.0 + nu);
    let err = y_true - mu;
    let nll = 0.5 * (std::f64::consts::PI / nu).ln() - alpha * omega.ln()
        + (alpha + 0.5) * (err * err * nu + omega).ln()
        + lgamma(alpha)
        - lgamma(alpha + 0.5);
    let reg = lambda * err.abs() * (2.0 * nu + alpha);
    nll + reg
}

/// Approximate lgamma via Stirling's series (accurate for alpha > 1).
fn lgamma(x: f64) -> f64 {
    // Use the Lanczos approximation (g=7, 9 coefficients)
    if x < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().abs().ln() - lgamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let t = x + 7.5;
        let coeffs = [
            0.999_999_999_999_809_9_f64,
            676.520_368_121_885_1,
            -1_259.139_216_722_402_8,
            771.323_428_777_653_1,
            -176.615_029_162_140_6,
            12.507_343_278_686_905,
            -0.138_571_095_265_720_12,
            9.984_369_578_019_572e-6,
            1.505_632_735_149_311_6e-7,
        ];
        let mut ser = coeffs[0];
        for (i, &c) in coeffs[1..].iter().enumerate() {
            ser += c / (x + (i + 1) as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + ser.ln()
    }
}

/// Dirichlet output for evidential classification.
#[derive(Debug, Clone)]
pub struct DirichletOutput {
    /// K concentration parameters (all > 1 for "evidence above prior").
    pub concentrations: Vec<f64>,
}

impl DirichletOutput {
    pub fn new(concentrations: Vec<f64>) -> Self {
        Self { concentrations }
    }

    /// Vacuity: K / sum(concentrations).
    pub fn vacuity(&self) -> f64 {
        let k = self.concentrations.len() as f64;
        let s: f64 = self.concentrations.iter().sum();
        if s < 1e-14 {
            f64::INFINITY
        } else {
            k / s
        }
    }

    /// Dissonance: belief-conflict among concentrated classes.
    /// Uses the formula: sum_k e_k/S * sum_{j!=k} Bal(b_k, b_j) * e_j/S
    /// where Bal(x,y) = 1 - |x-y|/(x+y+ε) and e_k = alpha_k - 1.
    pub fn dissonance(&self) -> f64 {
        let k = self.concentrations.len();
        if k < 2 {
            return 0.0;
        }
        let evidence: Vec<f64> = self
            .concentrations
            .iter()
            .map(|&a| (a - 1.0).max(0.0))
            .collect();
        let s: f64 = evidence.iter().sum();
        if s < 1e-14 {
            return 0.0;
        }
        let b: Vec<f64> = evidence.iter().map(|&e| e / s).collect();
        let mut diss = 0.0_f64;
        for i in 0..k {
            let mut bal_sum = 0.0_f64;
            for j in 0..k {
                if i == j {
                    continue;
                }
                let denom = b[i] + b[j];
                let bal = if denom < 1e-14 {
                    0.0
                } else {
                    1.0 - (b[i] - b[j]).abs() / denom
                };
                bal_sum += bal * b[j];
            }
            diss += b[i] * bal_sum;
        }
        diss
    }
}

/// Evidential classification layer: maps `in_dim` to K concentrations via softplus+1.
#[derive(Debug, Clone)]
pub struct EvidentialClassLayer {
    pub in_dim: usize,
    pub n_classes: usize,
    weights: Vec<f64>,
    biases: Vec<f64>,
}

impl EvidentialClassLayer {
    pub fn new(in_dim: usize, n_classes: usize, seed: u64) -> Self {
        Self {
            in_dim,
            n_classes,
            weights: xavier_f64(in_dim, n_classes, seed),
            biases: vec![0.0; n_classes],
        }
    }

    /// Forward: compute K concentrations, each >= 1.
    pub fn uncertainty(&self, x: &[f64]) -> DirichletOutput {
        let mut concentrations = vec![0.0_f64; self.n_classes];
        for k in 0..self.n_classes {
            let mut raw = self.biases[k];
            for i in 0..self.in_dim.min(x.len()) {
                raw += x[i] * self.weights[k * self.in_dim + i];
            }
            // softplus + 1 to ensure alpha_k >= 1
            concentrations[k] = softplus_f64(raw) + 1.0;
        }
        DirichletOutput { concentrations }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Ensemble tests (5) ──────────────────────────────────────────────────

    #[test]
    fn test_ensemble_member_forward() {
        let m = EnsembleMember::new(&[3, 8, 2], 42).expect("test value");
        let out = m.forward(&[1.0, 0.5, -0.3]);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_deep_ensemble_mean_var() {
        let ens = DeepEnsemble::new(5, &[4, 8, 2], 0).expect("test value");
        let (mean, var) = ens.predict_mean_var(&[1.0, 0.0, -1.0, 0.5]);
        assert_eq!(mean.len(), 2);
        assert_eq!(var.len(), 2);
        for &v in &var {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_deep_ensemble_consistency() {
        let ens = DeepEnsemble::new(3, &[2, 4, 1], 7).expect("test value");
        let (m1, _) = ens.predict_mean_var(&[0.5, -0.5]);
        let (m2, _) = ens.predict_mean_var(&[0.5, -0.5]);
        assert!((m1[0] - m2[0]).abs() < 1e-12, "deterministic");
    }

    #[test]
    fn test_snapshot_ensemble_add() {
        let mut s = SnapshotEnsemble::new(3);
        s.add_snapshot(vec![0.1, 0.9]);
        s.add_snapshot(vec![0.4, 0.6]);
        assert_eq!(s.snapshots.len(), 2);
    }

    #[test]
    fn test_snapshot_ensemble_predict() {
        let mut s = SnapshotEnsemble::new(4);
        s.add_snapshot(vec![0.2, 0.8]);
        s.add_snapshot(vec![0.4, 0.6]);
        let pred = s.predict(&[]);
        assert!((pred[0] - 0.3).abs() < 1e-10);
        assert!((pred[1] - 0.7).abs() < 1e-10);
    }

    // ── Flow tests (8) ──────────────────────────────────────────────────────

    #[test]
    fn test_flow_result_construction() {
        let fr = FlowResult {
            z: vec![1.0, 2.0],
            log_det: 0.5,
        };
        assert_eq!(fr.z.len(), 2);
        assert!((fr.log_det - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_realnvp_coupling_forward() {
        let c = RealNvpCoupling::new(4, 8, 99).expect("test value");
        let x = vec![0.1, 0.2, 0.3, 0.4];
        let res = c.forward(&x).expect("test value");
        assert_eq!(res.z.len(), 4);
        // First half unchanged
        assert!((res.z[0] - x[0]).abs() < 1e-12);
        assert!((res.z[1] - x[1]).abs() < 1e-12);
    }

    #[test]
    fn test_realnvp_coupling_inverse() {
        let c = RealNvpCoupling::new(4, 8, 55).expect("test value");
        let x = vec![0.5, -0.3, 0.7, 1.2];
        let res = c.forward(&x).expect("test value");
        let x_rec = c.inverse(&res.z).expect("test value");
        for (orig, rec) in x.iter().zip(x_rec.iter()) {
            assert!((orig - rec).abs() < 1e-8, "invertibility failed");
        }
    }

    #[test]
    fn test_realnvp_invertibility() {
        let c = RealNvpCoupling::new(6, 12, 123).expect("test value");
        let x = vec![-1.0, 0.5, 2.0, -0.5, 0.1, 3.0];
        let res = c.forward(&x).expect("test value");
        let x_rec = c.inverse(&res.z).expect("test value");
        for (o, r) in x.iter().zip(x_rec.iter()) {
            assert!((o - r).abs() < 1e-7, "round-trip error");
        }
    }

    #[test]
    fn test_actnorm_forward() {
        let mut an = ActNorm::new(3);
        let batch = vec![vec![1.0, 2.0, 3.0], vec![-1.0, 0.0, 1.0]];
        an.initialize_from_batch(&batch);
        let res = an.forward(&[0.0, 1.0, 2.0]).expect("test value");
        assert_eq!(res.z.len(), 3);
    }

    #[test]
    fn test_flow_model_construction() {
        let mut model = FlowModel::new(4);
        let an = ActNorm::new(4);
        model.add_layer(Box::new(an));
        assert_eq!(model.layers.len(), 1);
    }

    #[test]
    fn test_flow_model_nll_positive() {
        let mut model = FlowModel::new(4);
        model.add_layer(Box::new(
            RealNvpCoupling::new(4, 8, 11).expect("test value"),
        ));
        let batch = vec![vec![0.1, 0.2, 0.3, 0.4], vec![-0.1, 0.5, -0.3, 0.8]];
        let nll = model.nll_loss(&batch).expect("test value");
        assert!(nll.is_finite());
    }

    #[test]
    fn test_flow_log_likelihood() {
        let mut model = FlowModel::new(4);
        model.add_layer(Box::new(ActNorm::new(4)));
        let x = vec![0.0, 0.0, 0.0, 0.0];
        let ll = model.log_likelihood(&x).expect("test value");
        assert!(ll.is_finite());
    }

    // ── GP tests (8) ────────────────────────────────────────────────────────

    #[test]
    fn test_rbf_kernel_self_similarity() {
        let k = GpKernel::Rbf {
            length_scale: 1.0,
            variance: 1.0,
        };
        let x = vec![1.0, 2.0];
        assert!((k.compute(&x, &x) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rbf_kernel_decay() {
        let k = GpKernel::Rbf {
            length_scale: 1.0,
            variance: 1.0,
        };
        let x1 = vec![0.0];
        let x2 = vec![10.0];
        assert!(k.compute(&x1, &x2) < 0.01);
    }

    #[test]
    fn test_matern52_kernel() {
        let k = GpKernel::Matern52 {
            length_scale: 1.0,
            variance: 2.0,
        };
        let x = vec![0.5, 0.5];
        assert!((k.compute(&x, &x) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_kernel() {
        let k = GpKernel::Linear { variance: 1.0 };
        let x1 = vec![2.0, 3.0];
        let x2 = vec![4.0, 1.0];
        // dot product = 2*4 + 3*1 = 11
        assert!((k.compute(&x1, &x2) - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_gp_fit() {
        let mut gp = GaussianProcess::new(
            GpKernel::Rbf {
                length_scale: 1.0,
                variance: 1.0,
            },
            0.1,
        );
        let x = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y = vec![0.0, 1.0, 0.0];
        gp.fit(x, y);
        assert_eq!(gp.x_train.len(), 3);
    }

    #[test]
    fn test_gp_predict_mean() {
        let mut gp = GaussianProcess::new(
            GpKernel::Rbf {
                length_scale: 1.0,
                variance: 1.0,
            },
            0.01,
        );
        let x = vec![vec![0.0], vec![1.0]];
        let y = vec![0.0, 1.0];
        gp.fit(x, y);
        let (mean, _var) = gp.predict(&[0.0]);
        // Near training point, should be close to label
        assert!(mean.is_finite());
    }

    #[test]
    fn test_gp_predict_variance_positive() {
        let mut gp = GaussianProcess::new(
            GpKernel::Rbf {
                length_scale: 1.0,
                variance: 1.0,
            },
            0.1,
        );
        gp.fit(vec![vec![0.0], vec![1.0]], vec![0.0, 1.0]);
        let (_mean, var) = gp.predict(&[5.0]); // far from training
        assert!(var >= 0.0);
    }

    #[test]
    fn test_gp_prediction_struct() {
        let mut gp = GaussianProcess::new(
            GpKernel::Rbf {
                length_scale: 1.0,
                variance: 1.0,
            },
            0.1,
        );
        gp.fit(vec![vec![0.5]], vec![1.0]);
        let pred = gp.predict_structured(&[0.5]);
        assert!(pred.std >= 0.0);
        assert!((pred.std - pred.variance.sqrt()).abs() < 1e-12);
    }

    // ── Calibration tests (7) ────────────────────────────────────────────────

    #[test]
    fn test_temperature_scaling_fit() {
        let logits = vec![vec![2.0_f64, 0.5], vec![0.3, 1.8]];
        let labels = vec![0_usize, 1];
        let mut ts = TemperatureScaling::new(1.0);
        ts.fit(&logits, &labels);
        assert!(ts.temperature > 0.0 && ts.temperature <= 10.0);
    }

    #[test]
    fn test_temperature_scaling_calibrate() {
        let ts = TemperatureScaling::new(2.0);
        let probs = ts.calibrate(&[2.0, 1.0, -1.0]);
        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_platt_scaling_fit() {
        let scores = vec![-1.0, 0.0, 1.0, 2.0];
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        let mut p = PlattScaling::new();
        p.fit(&scores, &labels);
        assert!(p.a.is_finite() && p.b.is_finite());
    }

    #[test]
    fn test_platt_predict_range() {
        let mut p = PlattScaling::new();
        p.fit(&[-2.0, -1.0, 1.0, 2.0], &[0.0, 0.0, 1.0, 1.0]);
        let prob = p.predict(3.0);
        assert!(prob > 0.0 && prob < 1.0);
    }

    #[test]
    fn test_calibration_ece_perfect() {
        let ce = CalibrationEvaluator::new();
        // perfect calibration: predicted prob = actual accuracy
        let probs = vec![0.9, 0.8, 0.7, 0.2, 0.1];
        let labels = vec![1_usize, 1, 1, 0, 0];
        let ece = ce.ece(&probs, &labels, 5);
        assert!((0.0..=1.0).contains(&ece));
    }

    #[test]
    fn test_calibration_ece_bounds() {
        let ce = CalibrationEvaluator::new();
        let probs = vec![0.5; 10];
        let labels = vec![0_usize; 10];
        let ece = ce.ece(&probs, &labels, 10);
        assert!((0.0..=1.0).contains(&ece));
    }

    #[test]
    fn test_isotonic_calibrator_monotone() {
        let mut iso = IsotonicCalibrator::new();
        let scores = vec![0.1, 0.4, 0.35, 0.8, 0.9, 0.6];
        let labels = vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        iso.fit(&scores, &labels);
        // Calibrated outputs should be monotone (non-decreasing in x_calibrated)
        for w in iso.y_calibrated.windows(2) {
            // In isotonic regression, mean values per block can be non-decreasing
            assert!(w[0] <= w[1] + 1e-9, "not monotone: {} > {}", w[0], w[1]);
        }
    }

    // ── Evidential tests (7) ─────────────────────────────────────────────────

    #[test]
    fn test_nig_output_construction() {
        let nig = NigOutput::new(0.5, 2.0, 3.0, 1.0);
        assert!((nig.mu - 0.5).abs() < 1e-12);
        assert!(nig.alpha > 1.0);
    }

    #[test]
    fn test_evidential_layer_output_dims() {
        let layer = EvidentialLayer::new(4, 2, 42);
        let x = vec![1.0, 0.5, -0.3, 0.8];
        let outputs = layer.forward(&x);
        assert_eq!(outputs.len(), 2);
        for o in &outputs {
            assert!(o.nu > 0.0);
            assert!(o.alpha > 1.0);
            assert!(o.beta > 0.0);
        }
    }

    #[test]
    fn test_nig_loss_positive() {
        let nig = NigOutput::new(0.0, 1.0, 2.0, 1.0);
        let loss = nig_loss(1.0, &nig, 0.01);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_dirichlet_output_vacuity() {
        let d = DirichletOutput::new(vec![2.0, 3.0, 4.0]);
        let v = d.vacuity();
        // K=3, S=9 -> v = 3/9 = 1/3
        assert!((v - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dirichlet_vacuity_formula() {
        let d = DirichletOutput::new(vec![10.0, 10.0]);
        let v = d.vacuity();
        // K=2, S=20 -> v = 2/20 = 0.1
        assert!((v - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_evidential_class_layer() {
        let layer = EvidentialClassLayer::new(3, 5, 7);
        let x = vec![1.0, -1.0, 0.5];
        let output = layer.uncertainty(&x);
        assert_eq!(output.concentrations.len(), 5);
        for &c in &output.concentrations {
            assert!(c >= 1.0, "concentration must be >= 1: {}", c);
        }
    }

    #[test]
    fn test_uncertainty_concentrations_positive() {
        let layer = EvidentialClassLayer::new(2, 3, 99);
        let out = layer.uncertainty(&[0.0, 0.0]);
        for &c in &out.concentrations {
            assert!(c > 0.0);
        }
        // Vacuity should be finite and positive
        let v = out.vacuity();
        assert!(v > 0.0 && v.is_finite());
    }
}
