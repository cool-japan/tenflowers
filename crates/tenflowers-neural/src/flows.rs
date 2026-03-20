//! Normalizing Flows — Track X.
//!
//! Implements invertible density models based on the RealNVP/Glow family of
//! normalizing flows.  Every component is backed by plain `Vec<f32>` weight
//! buffers so there is no dependency on `Tensor` or an autograd engine, making
//! the module self-contained and easy to compose.
//!
//! # Mathematical background
//!
//! A normalizing flow is a sequence of invertible maps
//! ```text
//! z = f_N(f_{N-1}(... f_1(x)))
//! ```
//! that transforms a complex data distribution p(x) into a simple base
//! distribution p(z) (here an isotropic Gaussian).  The change-of-variables
//! formula gives the exact log-likelihood:
//! ```text
//! log p(x) = log p(z) + Σ_k log |det J_k|
//! ```
//! where J_k is the Jacobian of the k-th layer's forward transform.
//!
//! # Implemented layers
//!
//! | Layer | Reference |
//! |-------|-----------|
//! | [`AffineCouplingLayer`] | RealNVP (Dinh et al., 2017) |
//! | [`PermutationLayer`] | standard shuffling step |
//! | [`ActNorm`] | Glow (Kingma & Dhariwal, 2018) |
//!
//! # Quick start
//!
//! ```rust,ignore
//! use tenflowers_neural::flows::{AffineCouplingLayer, PermutationLayer, ActNorm, NormalizingFlow};
//!
//! let mut flow = NormalizingFlow::new(4);
//! flow.add_layer(ActNorm::new(4));
//! flow.add_layer(AffineCouplingLayer::new(4, 16)?);
//! flow.add_layer(PermutationLayer::reverse(4));
//! flow.add_layer(AffineCouplingLayer::new(4, 16)?);
//!
//! let x = vec![1.0_f32, -0.5, 0.3, 2.1];
//! let (z, log_det) = flow.forward(&x)?;
//! let x_rec = flow.inverse(&z)?;
//! let ll = flow.log_likelihood(&x)?;
//! ```

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Activation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Elementwise ReLU: `max(0, x)`.
#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Elementwise tanh clamped for numerical safety.
#[inline]
fn tanh_safe(x: f32) -> f32 {
    // clamp to avoid inf in gradient computations
    x.clamp(-15.0, 15.0).tanh()
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight initialisation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Xavier/Glorot uniform initialisation for a weight matrix of shape
/// `[fan_in, fan_out]`.  Scale = `sqrt(6 / (fan_in + fan_out))`.
fn xavier_uniform(fan_in: usize, fan_out: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt() as f32;
    let n = fan_in * fan_out;
    (0..n)
        .map(|_| {
            let u: f32 = rng.random();
            u * 2.0 * limit - limit
        })
        .collect()
}

/// Zero-initialise a bias vector.
fn zeros(n: usize) -> Vec<f32> {
    vec![0.0_f32; n]
}

// ─────────────────────────────────────────────────────────────────────────────
// Dense-layer primitives (two-layer MLP used inside coupling layers)
// ─────────────────────────────────────────────────────────────────────────────

/// Linear + ReLU: `output[j] = relu(Σ_i input[i] * W[i*out_dim+j] + b[j])`.
///
/// Weights are stored row-major: `W[i * out_dim + j]` maps input `i` → output `j`.
fn linear_relu(input: &[f32], weight: &[f32], bias: &[f32], out_dim: usize) -> Result<Vec<f32>> {
    let in_dim = input.len();
    let expected = in_dim * out_dim;
    if weight.len() != expected {
        return Err(TensorError::invalid_argument_op(
            "linear_relu",
            &format!(
                "weight length mismatch: expected {expected}, got {}",
                weight.len()
            ),
        ));
    }
    if bias.len() != out_dim {
        return Err(TensorError::invalid_argument_op(
            "linear_relu",
            &format!(
                "bias length mismatch: expected {out_dim}, got {}",
                bias.len()
            ),
        ));
    }
    let mut out = bias.to_vec();
    for i in 0..in_dim {
        let xi = input[i];
        for j in 0..out_dim {
            out[j] += xi * weight[i * out_dim + j];
        }
    }
    for v in out.iter_mut() {
        *v = relu(*v);
    }
    Ok(out)
}

/// Linear (no activation): `output[j] = Σ_i input[i] * W[i*out_dim+j] + b[j]`.
fn linear(input: &[f32], weight: &[f32], bias: &[f32], out_dim: usize) -> Result<Vec<f32>> {
    let in_dim = input.len();
    let expected = in_dim * out_dim;
    if weight.len() != expected {
        return Err(TensorError::invalid_argument_op(
            "linear",
            &format!(
                "weight length mismatch: expected {expected}, got {}",
                weight.len()
            ),
        ));
    }
    if bias.len() != out_dim {
        return Err(TensorError::invalid_argument_op(
            "linear",
            &format!(
                "bias length mismatch: expected {out_dim}, got {}",
                bias.len()
            ),
        ));
    }
    let mut out = bias.to_vec();
    for i in 0..in_dim {
        let xi = input[i];
        for j in 0..out_dim {
            out[j] += xi * weight[i * out_dim + j];
        }
    }
    Ok(out)
}

/// Linear + tanh (used for scale network to produce bounded log-scales).
fn linear_tanh(input: &[f32], weight: &[f32], bias: &[f32], out_dim: usize) -> Result<Vec<f32>> {
    let mut out = linear(input, weight, bias, out_dim)?;
    for v in out.iter_mut() {
        *v = tanh_safe(*v);
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// FlowLayer trait
// ─────────────────────────────────────────────────────────────────────────────

/// Trait for invertible transformations (flow layers).
///
/// Each layer must support a forward pass that returns `(z, log_det_jacobian)`
/// and an inverse pass.  Both directions must be consistent:
/// `inverse(forward(x).0) ≈ x`.
pub trait FlowLayer: Send + Sync {
    /// Forward transform: x → z.
    ///
    /// Returns `(z, log_det_jacobian)` where `log_det_jacobian` is the
    /// log-absolute-determinant of the Jacobian ∂z/∂x.
    fn forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)>;

    /// Inverse transform: z → x.
    fn inverse(&self, z: &[f32]) -> Result<Vec<f32>>;

    /// Dimensionality of the input/output vectors (they must be equal).
    fn dim(&self) -> usize;

    /// Human-readable layer name (used in diagnostics).
    fn name(&self) -> &str;
}

// ─────────────────────────────────────────────────────────────────────────────
// AffineCouplingLayer  (RealNVP — Dinh et al. 2017)
// ─────────────────────────────────────────────────────────────────────────────

/// Affine coupling layer (RealNVP, Dinh et al. 2017).
///
/// The input `x ∈ ℝ^d` is split at `split_idx`:
/// ```text
/// x1 = x[0 .. split_idx]      (conditioner — passed through unchanged)
/// x2 = x[split_idx .. d]      (transformed)
///
/// s = s_net(x1)  ∈ ℝ^{d - split_idx}
/// t = t_net(x1)  ∈ ℝ^{d - split_idx}
///
/// z1 = x1
/// z2 = x2 ⊙ exp(s) + t
///
/// log|det J| = Σ s_i
/// ```
/// The inverse is simply `x2 = (z2 - t) ⊙ exp(-s)`.
///
/// The scale (`s`) network uses a tanh output to keep `exp(s)` bounded.
pub struct AffineCouplingLayer {
    /// Total input/output dimension.
    pub dim: usize,
    /// Index at which the input is split.
    pub split_idx: usize,
    /// Hidden dimension of the scale and translation networks.
    pub hidden_dim: usize,

    // Scale network: x1[split_idx] → hidden[hidden_dim] → s[dim-split_idx]
    s_w1: Vec<f32>, // [split_idx  × hidden_dim]
    s_b1: Vec<f32>, // [hidden_dim]
    s_w2: Vec<f32>, // [hidden_dim × (dim-split_idx)]
    s_b2: Vec<f32>, // [dim-split_idx]

    // Translation network: x1[split_idx] → hidden[hidden_dim] → t[dim-split_idx]
    t_w1: Vec<f32>, // [split_idx  × hidden_dim]
    t_b1: Vec<f32>, // [hidden_dim]
    t_w2: Vec<f32>, // [hidden_dim × (dim-split_idx)]
    t_b2: Vec<f32>, // [dim-split_idx]
}

impl AffineCouplingLayer {
    /// Create a new coupling layer with `split_idx = dim / 2`.
    ///
    /// Weights are Xavier-initialised from a deterministic seed derived from
    /// the layer dimensions.
    pub fn new(dim: usize, hidden_dim: usize) -> Result<Self> {
        Self::new_with_split(dim, hidden_dim, dim / 2)
    }

    /// Create a new coupling layer with an explicit split index.
    pub fn new_with_split(dim: usize, hidden_dim: usize, split_idx: usize) -> Result<Self> {
        if dim < 2 {
            return Err(TensorError::invalid_argument_op(
                "AffineCouplingLayer::new",
                "dim must be at least 2",
            ));
        }
        if split_idx == 0 || split_idx >= dim {
            return Err(TensorError::invalid_argument_op(
                "AffineCouplingLayer::new",
                &format!("split_idx must be in [1, dim-1]; got split_idx={split_idx}, dim={dim}"),
            ));
        }
        if hidden_dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "AffineCouplingLayer::new",
                "hidden_dim must be at least 1",
            ));
        }

        let out_dim = dim - split_idx;

        // Use different seed offsets so the four weight matrices are distinct.
        let s_w1 = xavier_uniform(split_idx, hidden_dim, 0x1a2b3c4d_u64);
        let s_b1 = zeros(hidden_dim);
        let s_w2 = xavier_uniform(hidden_dim, out_dim, 0x5e6f7a8b_u64);
        let s_b2 = zeros(out_dim);

        let t_w1 = xavier_uniform(split_idx, hidden_dim, 0x9c0d1e2f_u64);
        let t_b1 = zeros(hidden_dim);
        let t_w2 = xavier_uniform(hidden_dim, out_dim, 0x3f4a5b6c_u64);
        let t_b2 = zeros(out_dim);

        Ok(Self {
            dim,
            split_idx,
            hidden_dim,
            s_w1,
            s_b1,
            s_w2,
            s_b2,
            t_w1,
            t_b1,
            t_w2,
            t_b2,
        })
    }

    /// Evaluate the scale network on `x1`, returning `s ∈ ℝ^{dim-split_idx}`.
    ///
    /// Architecture: linear + ReLU → linear + tanh (bounded log-scale).
    fn scale_net(&self, x1: &[f32]) -> Result<Vec<f32>> {
        let h = linear_relu(x1, &self.s_w1, &self.s_b1, self.hidden_dim)?;
        linear_tanh(&h, &self.s_w2, &self.s_b2, self.dim - self.split_idx)
    }

    /// Evaluate the translation network on `x1`, returning `t ∈ ℝ^{dim-split_idx}`.
    ///
    /// Architecture: linear + ReLU → linear (unbounded translation).
    fn translate_net(&self, x1: &[f32]) -> Result<Vec<f32>> {
        let h = linear_relu(x1, &self.t_w1, &self.t_b1, self.hidden_dim)?;
        linear(&h, &self.t_w2, &self.t_b2, self.dim - self.split_idx)
    }
}

impl FlowLayer for AffineCouplingLayer {
    fn forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)> {
        if x.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "AffineCouplingLayer::forward",
                &format!(
                    "input length mismatch: expected {}, got {}",
                    self.dim,
                    x.len()
                ),
            ));
        }

        let x1 = &x[..self.split_idx];
        let x2 = &x[self.split_idx..];

        let s = self.scale_net(x1)?;
        let t = self.translate_net(x1)?;

        let out_dim = self.dim - self.split_idx;
        debug_assert_eq!(s.len(), out_dim);
        debug_assert_eq!(t.len(), out_dim);

        let mut z = Vec::with_capacity(self.dim);
        z.extend_from_slice(x1); // z1 = x1
        for i in 0..out_dim {
            z.push(x2[i] * s[i].exp() + t[i]); // z2 = x2 ⊙ exp(s) + t
        }

        let log_det: f32 = s.iter().sum(); // Σ s_i
        Ok((z, log_det))
    }

    fn inverse(&self, z: &[f32]) -> Result<Vec<f32>> {
        if z.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "AffineCouplingLayer::inverse",
                &format!(
                    "input length mismatch: expected {}, got {}",
                    self.dim,
                    z.len()
                ),
            ));
        }

        let z1 = &z[..self.split_idx];
        let z2 = &z[self.split_idx..];

        // x1 = z1 (identity part)
        let s = self.scale_net(z1)?;
        let t = self.translate_net(z1)?;

        let out_dim = self.dim - self.split_idx;

        let mut x = Vec::with_capacity(self.dim);
        x.extend_from_slice(z1); // x1 = z1
        for i in 0..out_dim {
            x.push((z2[i] - t[i]) * (-s[i]).exp()); // x2 = (z2 - t) ⊙ exp(-s)
        }
        Ok(x)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &str {
        "AffineCouplingLayer"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PermutationLayer
// ─────────────────────────────────────────────────────────────────────────────

/// Permutation layer — reorders dimensions.
///
/// Since a permutation is an orthogonal transform its Jacobian determinant is
/// ±1, so `log|det J| = 0` always.
///
/// Two convenient constructors are provided:
/// - [`PermutationLayer::reverse`]: reverses the order of all dimensions
///   (this is its own inverse).
/// - [`PermutationLayer::fixed`]: arbitrary user-supplied permutation.
pub struct PermutationLayer {
    /// Total input/output dimension.
    pub dim: usize,
    /// Forward permutation: `z[i] = x[permutation[i]]`.
    permutation: Vec<usize>,
    /// Inverse permutation: `x[i] = z[inv_permutation[i]]`.
    inv_permutation: Vec<usize>,
}

impl PermutationLayer {
    /// Build a layer that reverses all dimensions.
    ///
    /// `z[i] = x[d - 1 - i]`
    ///
    /// This is its own inverse (reversing twice is the identity), so applying
    /// it between two coupling layers is the cheapest way to alternate which
    /// half of the input is conditioned on.
    pub fn reverse(dim: usize) -> Self {
        let permutation: Vec<usize> = (0..dim).map(|i| dim - 1 - i).collect();
        // Reversal is its own inverse.
        let inv_permutation = permutation.clone();
        Self {
            dim,
            permutation,
            inv_permutation,
        }
    }

    /// Build a layer from a user-supplied permutation.
    ///
    /// `permutation` must be a valid permutation of `0..dim`, i.e. each index
    /// in `0..dim` appears exactly once.  Returns an error otherwise.
    pub fn fixed(permutation: Vec<usize>) -> Result<Self> {
        let dim = permutation.len();
        if dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "PermutationLayer::fixed",
                "permutation must be non-empty",
            ));
        }

        // Validate: every value in 0..dim appears exactly once.
        let mut seen = vec![false; dim];
        for &idx in &permutation {
            if idx >= dim {
                return Err(TensorError::invalid_argument_op(
                    "PermutationLayer::fixed",
                    &format!("permutation index {idx} out of range for dim {dim}"),
                ));
            }
            if seen[idx] {
                return Err(TensorError::invalid_argument_op(
                    "PermutationLayer::fixed",
                    &format!("permutation index {idx} appears more than once"),
                ));
            }
            seen[idx] = true;
        }

        // Build inverse permutation: inv[p[i]] = i.
        let mut inv_permutation = vec![0usize; dim];
        for (i, &p) in permutation.iter().enumerate() {
            inv_permutation[p] = i;
        }

        Ok(Self {
            dim,
            permutation,
            inv_permutation,
        })
    }
}

impl FlowLayer for PermutationLayer {
    /// Apply the permutation: `z[i] = x[permutation[i]]`.
    fn forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)> {
        if x.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "PermutationLayer::forward",
                &format!(
                    "input length mismatch: expected {}, got {}",
                    self.dim,
                    x.len()
                ),
            ));
        }
        let z: Vec<f32> = self.permutation.iter().map(|&i| x[i]).collect();
        Ok((z, 0.0)) // log|det J| = 0 for any permutation
    }

    /// Apply the inverse permutation: `x[i] = z[inv_permutation[i]]`.
    fn inverse(&self, z: &[f32]) -> Result<Vec<f32>> {
        if z.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "PermutationLayer::inverse",
                &format!(
                    "input length mismatch: expected {}, got {}",
                    self.dim,
                    z.len()
                ),
            ));
        }
        let x: Vec<f32> = self.inv_permutation.iter().map(|&i| z[i]).collect();
        Ok(x)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &str {
        "PermutationLayer"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ActNorm  (Glow — Kingma & Dhariwal, 2018)
// ─────────────────────────────────────────────────────────────────────────────

/// Activation-normalisation layer (Glow, Kingma & Dhariwal, 2018).
///
/// Acts as a per-channel affine transform with *learnable* (or data-initialised)
/// scale and bias:
/// ```text
/// z[i] = (x[i] - bias[i]) / scale[i]
/// log|det J| = -Σ log(scale[i])   [= -d * log(scale) for isotropic]
/// ```
/// where `scale[i] = exp(log_scale[i])` is stored in log-space so the
/// Jacobian contribution is simply `- Σ log_scale[i]`.
///
/// Call [`ActNorm::initialize_from_data`] on the first mini-batch to
/// data-dependently set bias ← mean and log_scale ← log(std), so that after
/// initialisation the activations have zero mean and unit variance.
pub struct ActNorm {
    /// Total input/output dimension.
    pub dim: usize,
    /// Per-channel log-scale (learnable). `scale[i] = exp(log_scale[i])`.
    scale: Vec<f32>, // stores log_scale internally
    /// Per-channel bias (learnable).
    bias: Vec<f32>,
    /// Whether [`initialize_from_data`] has been called.
    initialized: bool,
}

impl ActNorm {
    /// Create a new `ActNorm` layer with identity initialisation
    /// (scale = 1, bias = 0, so `log_scale = 0`).
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            scale: zeros(dim), // log_scale = 0  ⟹  scale = 1
            bias: zeros(dim),  // bias = 0
            initialized: false,
        }
    }

    /// Data-dependent initialisation from the first mini-batch.
    ///
    /// After this call:
    /// - `bias[i]  ← mean of x[i] over the n samples`
    /// - `scale[i] ← log(std of x[i] over the n samples + ε)` (log-scale)
    ///
    /// so that the layer's forward pass produces activations with
    /// approximately zero mean and unit variance on this batch.
    ///
    /// `x` is a flat array of `n * dim` values in row-major order
    /// (sample-major: `x[s * dim + i]` is dimension `i` of sample `s`).
    pub fn initialize_from_data(&mut self, x: &[f32], n: usize) -> Result<()> {
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "ActNorm::initialize_from_data",
                "n (number of samples) must be at least 1",
            ));
        }
        if x.len() != n * self.dim {
            return Err(TensorError::invalid_argument_op(
                "ActNorm::initialize_from_data",
                &format!(
                    "x length mismatch: expected {} (n={n} * dim={}), got {}",
                    n * self.dim,
                    self.dim,
                    x.len()
                ),
            ));
        }

        let n_f = n as f32;

        // Compute per-dimension mean.
        let mut mean = zeros(self.dim);
        for s in 0..n {
            for i in 0..self.dim {
                mean[i] += x[s * self.dim + i];
            }
        }
        for m in mean.iter_mut() {
            *m /= n_f;
        }

        // Compute per-dimension variance.
        let mut var = zeros(self.dim);
        for s in 0..n {
            for i in 0..self.dim {
                let diff = x[s * self.dim + i] - mean[i];
                var[i] += diff * diff;
            }
        }
        for v in var.iter_mut() {
            *v /= n_f;
        }

        // Set bias = mean, log_scale = log(std + ε).
        let eps = 1e-6_f32;
        for i in 0..self.dim {
            self.bias[i] = mean[i];
            self.scale[i] = (var[i].sqrt() + eps).ln();
        }
        self.initialized = true;
        Ok(())
    }

    /// Retrieve the log-scale vector (for inspection / gradient updates).
    pub fn log_scales(&self) -> &[f32] {
        &self.scale
    }

    /// Retrieve the bias vector (for inspection / gradient updates).
    pub fn biases(&self) -> &[f32] {
        &self.bias
    }
}

impl FlowLayer for ActNorm {
    /// Forward: `z[i] = (x[i] - bias[i]) * exp(-log_scale[i])`.
    ///
    /// `log|det J| = -Σ log_scale[i]`.
    fn forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)> {
        if x.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "ActNorm::forward",
                &format!(
                    "input length mismatch: expected {}, got {}",
                    self.dim,
                    x.len()
                ),
            ));
        }
        let mut z = Vec::with_capacity(self.dim);
        let mut log_det = 0.0_f32;
        for i in 0..self.dim {
            let log_s = self.scale[i];
            z.push((x[i] - self.bias[i]) * (-log_s).exp());
            log_det -= log_s; // contribution: -log_scale[i]
        }
        Ok((z, log_det))
    }

    /// Inverse: `x[i] = z[i] * exp(log_scale[i]) + bias[i]`.
    fn inverse(&self, z: &[f32]) -> Result<Vec<f32>> {
        if z.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "ActNorm::inverse",
                &format!(
                    "input length mismatch: expected {}, got {}",
                    self.dim,
                    z.len()
                ),
            ));
        }
        let mut x = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let log_s = self.scale[i];
            x.push(z[i] * log_s.exp() + self.bias[i]);
        }
        Ok(x)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &str {
        "ActNorm"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NormalizingFlow  (composition of flow layers)
// ─────────────────────────────────────────────────────────────────────────────

/// Composition of flow layers: `z = f_N(... f_2(f_1(x)))`.
///
/// All layers must have the same [`FlowLayer::dim`]; an error is returned at
/// [`add_layer`](NormalizingFlow::add_layer) time if the dimensions mismatch.
///
/// The base distribution is an isotropic standard Gaussian, so
/// ```text
/// log p(z) = -0.5 · Σ z_i² − 0.5 · d · log(2π)
/// log p(x) = log p(z) + Σ_k log|det J_k|
/// ```
pub struct NormalizingFlow {
    layers: Vec<Box<dyn FlowLayer>>,
    /// Common input/output dimension.
    pub dim: usize,
}

impl NormalizingFlow {
    /// Create a new (empty) flow for data in ℝ^`dim`.
    pub fn new(dim: usize) -> Self {
        Self {
            layers: Vec::new(),
            dim,
        }
    }

    /// Append a layer to the end of the flow.
    ///
    /// Returns an error if `layer.dim() != self.dim`.
    pub fn add_layer<L: FlowLayer + 'static>(&mut self, layer: L) -> Result<()> {
        if layer.dim() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "NormalizingFlow::add_layer",
                &format!(
                    "layer dim {} does not match flow dim {}",
                    layer.dim(),
                    self.dim
                ),
            ));
        }
        self.layers.push(Box::new(layer));
        Ok(())
    }

    /// Forward pass through all layers: `x → z`.
    ///
    /// Returns `(z, total_log_det_jacobian)`.
    pub fn forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)> {
        if x.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "NormalizingFlow::forward",
                &format!(
                    "input length mismatch: expected {}, got {}",
                    self.dim,
                    x.len()
                ),
            ));
        }
        let mut current = x.to_vec();
        let mut total_log_det = 0.0_f32;
        for layer in &self.layers {
            let (next, log_det) = layer.forward(&current)?;
            current = next;
            total_log_det += log_det;
        }
        Ok((current, total_log_det))
    }

    /// Inverse pass through all layers in reverse: `z → x`.
    pub fn inverse(&self, z: &[f32]) -> Result<Vec<f32>> {
        if z.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "NormalizingFlow::inverse",
                &format!(
                    "input length mismatch: expected {}, got {}",
                    self.dim,
                    z.len()
                ),
            ));
        }
        let mut current = z.to_vec();
        for layer in self.layers.iter().rev() {
            current = layer.inverse(&current)?;
        }
        Ok(current)
    }

    /// Exact log-likelihood of a single point `x` under the flow.
    ///
    /// ```text
    /// log p(x) = log p_Z(z) + total_log_det_jacobian
    ///          = -0.5 · ‖z‖² − 0.5 · d · log(2π) + total_log_det
    /// ```
    pub fn log_likelihood(&self, x: &[f32]) -> Result<f32> {
        let (z, log_det) = self.forward(x)?;
        let d = self.dim as f32;
        let log_pz: f32 =
            -0.5 * z.iter().map(|&zi| zi * zi).sum::<f32>() - 0.5 * d * std::f32::consts::TAU.ln(); // 0.5*d*log(2π)
        Ok(log_pz + log_det)
    }

    /// Mean negative log-likelihood loss over a batch.
    ///
    /// `batch` is a flat array of `n * dim` values (sample-major layout).
    /// Returns `NLL = -(1/n) Σ_s log p(x_s)`.
    pub fn nll_loss(&self, batch: &[f32], n: usize) -> Result<f32> {
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "NormalizingFlow::nll_loss",
                "batch must contain at least one sample (n >= 1)",
            ));
        }
        if batch.len() != n * self.dim {
            return Err(TensorError::invalid_argument_op(
                "NormalizingFlow::nll_loss",
                &format!(
                    "batch length mismatch: expected {} (n={n} * dim={}), got {}",
                    n * self.dim,
                    self.dim,
                    batch.len()
                ),
            ));
        }
        let mut total_nll = 0.0_f32;
        for s in 0..n {
            let x = &batch[s * self.dim..(s + 1) * self.dim];
            let ll = self.log_likelihood(x)?;
            total_nll -= ll;
        }
        Ok(total_nll / n as f32)
    }

    /// Number of layers in the flow.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Maximum absolute difference between two equal-length slices.
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// Build a simple 4-dimensional normalizing flow for reuse in tests.
    fn build_simple_flow() -> Result<NormalizingFlow> {
        let mut flow = NormalizingFlow::new(4);
        flow.add_layer(ActNorm::new(4))?;
        flow.add_layer(AffineCouplingLayer::new(4, 8)?)?;
        flow.add_layer(PermutationLayer::reverse(4))?;
        flow.add_layer(AffineCouplingLayer::new(4, 8)?)?;
        Ok(flow)
    }

    // ── AffineCouplingLayer tests ─────────────────────────────────────────────

    #[test]
    fn test_affine_coupling_forward_inverse_roundtrip() -> Result<()> {
        let layer = AffineCouplingLayer::new(6, 8)?;
        let x = vec![0.5_f32, -1.2, 0.3, 2.1, -0.7, 0.9];
        let (z, _log_det) = layer.forward(&x)?;
        let x_rec = layer.inverse(&z)?;
        assert_eq!(x_rec.len(), x.len());
        assert!(
            max_abs_diff(&x, &x_rec) < 1e-4,
            "roundtrip error {:.2e}",
            max_abs_diff(&x, &x_rec)
        );
        Ok(())
    }

    #[test]
    fn test_affine_coupling_identity_weights_log_det() -> Result<()> {
        // When all scale-network weights are zero the output of the scale net
        // (after tanh) is tanh(0) = 0, so exp(s) = 1 and log_det = 0.
        let mut layer = AffineCouplingLayer::new(4, 4)?;
        // Zero out scale-network weights → scale net always outputs zeros.
        layer.s_w1 = zeros(layer.split_idx * layer.hidden_dim);
        layer.s_w2 = zeros(layer.hidden_dim * (layer.dim - layer.split_idx));
        layer.s_b1 = zeros(layer.hidden_dim);
        layer.s_b2 = zeros(layer.dim - layer.split_idx);

        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let (_z, log_det) = layer.forward(&x)?;
        // With s = 0, log_det = Σ s_i = 0.
        assert!(log_det.abs() < 1e-6, "expected log_det ≈ 0, got {log_det}");
        Ok(())
    }

    #[test]
    fn test_affine_coupling_x1_passthrough() -> Result<()> {
        let layer = AffineCouplingLayer::new(6, 8)?;
        let x = vec![3.0_f32, -1.5, 0.7, 0.2, -0.9, 1.1];
        let (z, _) = layer.forward(&x)?;
        // z1 = x1 (identity for first split_idx elements)
        for i in 0..layer.split_idx {
            assert!(
                (z[i] - x[i]).abs() < 1e-6,
                "z[{i}] should equal x[{i}]: got {} vs {}",
                z[i],
                x[i]
            );
        }
        Ok(())
    }

    #[test]
    fn test_affine_coupling_dim_mismatch_error() {
        let layer = AffineCouplingLayer::new(4, 8).expect("construct layer");
        let bad_x = vec![1.0_f32; 5]; // wrong length
        assert!(
            layer.forward(&bad_x).is_err(),
            "expected error on dim mismatch"
        );
        assert!(
            layer.inverse(&bad_x).is_err(),
            "expected error on dim mismatch in inverse"
        );
    }

    #[test]
    fn test_affine_coupling_construction_errors() {
        // dim < 2
        assert!(AffineCouplingLayer::new(1, 4).is_err());
        // split_idx = 0
        assert!(AffineCouplingLayer::new_with_split(4, 4, 0).is_err());
        // split_idx >= dim
        assert!(AffineCouplingLayer::new_with_split(4, 4, 4).is_err());
        // hidden_dim = 0
        assert!(AffineCouplingLayer::new(4, 0).is_err());
    }

    #[test]
    fn test_affine_coupling_with_explicit_split() -> Result<()> {
        let layer = AffineCouplingLayer::new_with_split(6, 8, 2)?;
        assert_eq!(layer.split_idx, 2);
        let x = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let (z, _log_det) = layer.forward(&x)?;
        let x_rec = layer.inverse(&z)?;
        assert!(
            max_abs_diff(&x, &x_rec) < 1e-4,
            "roundtrip error {:.2e}",
            max_abs_diff(&x, &x_rec)
        );
        Ok(())
    }

    // ── PermutationLayer tests ────────────────────────────────────────────────

    #[test]
    fn test_permutation_reverse_is_own_inverse() -> Result<()> {
        let layer = PermutationLayer::reverse(5);
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let (z, log_det) = layer.forward(&x)?;
        let x_rec = layer.inverse(&z)?;
        assert!(
            max_abs_diff(&x, &x_rec) < 1e-7,
            "reversal should be self-inverse"
        );
        assert_eq!(log_det, 0.0, "permutation log_det must be 0");
        Ok(())
    }

    #[test]
    fn test_permutation_reverse_values() -> Result<()> {
        let layer = PermutationLayer::reverse(4);
        let x = vec![10.0_f32, 20.0, 30.0, 40.0];
        let (z, _) = layer.forward(&x)?;
        assert_eq!(z, vec![40.0_f32, 30.0, 20.0, 10.0]);
        Ok(())
    }

    #[test]
    fn test_permutation_fixed_known() -> Result<()> {
        // Permutation: [2, 0, 1] → z[0]=x[2], z[1]=x[0], z[2]=x[1]
        let layer = PermutationLayer::fixed(vec![2, 0, 1])?;
        let x = vec![10.0_f32, 20.0, 30.0];
        let (z, log_det) = layer.forward(&x)?;
        assert_eq!(z, vec![30.0_f32, 10.0, 20.0]);
        assert_eq!(log_det, 0.0);

        // Roundtrip.
        let x_rec = layer.inverse(&z)?;
        assert!(
            max_abs_diff(&x, &x_rec) < 1e-7,
            "fixed permutation roundtrip failed"
        );
        Ok(())
    }

    #[test]
    fn test_permutation_fixed_invalid() {
        // Duplicate index.
        assert!(PermutationLayer::fixed(vec![0, 0, 2]).is_err());
        // Out-of-range index.
        assert!(PermutationLayer::fixed(vec![0, 1, 5]).is_err());
        // Empty.
        assert!(PermutationLayer::fixed(vec![]).is_err());
    }

    #[test]
    fn test_permutation_dim_mismatch_error() {
        let layer = PermutationLayer::reverse(4);
        assert!(layer.forward(&[1.0, 2.0, 3.0]).is_err());
        assert!(layer.inverse(&[1.0, 2.0, 3.0]).is_err());
    }

    // ── ActNorm tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_actnorm_identity_init_roundtrip() -> Result<()> {
        let layer = ActNorm::new(4); // log_scale=0, bias=0 → identity
        let x = vec![1.5_f32, -0.3, 2.7, -1.1];
        let (z, _log_det) = layer.forward(&x)?;
        let x_rec = layer.inverse(&z)?;
        assert!(
            max_abs_diff(&x, &x_rec) < 1e-6,
            "ActNorm identity roundtrip failed"
        );
        Ok(())
    }

    #[test]
    fn test_actnorm_initialize_from_data_normalises() -> Result<()> {
        // Construct a batch where dim 0 has mean 5 and std 2, dim 1 has mean -3 and std 1.
        let n = 200_usize;
        let dim = 2_usize;
        let mut rng = StdRng::seed_from_u64(0xdeadbeef);

        let mut batch = Vec::with_capacity(n * dim);
        for _ in 0..n {
            let u: f32 = rng.random::<f32>() * 2.0 - 1.0; // in [-1, 1]
            batch.push(5.0 + 2.0 * u); // dim 0: mean≈5, std≈2·(1/√3)≈1.15
            let v: f32 = rng.random::<f32>() * 2.0 - 1.0;
            batch.push(-3.0 + v); // dim 1: mean≈-3, std≈0.58
        }

        let mut layer = ActNorm::new(dim);
        layer.initialize_from_data(&batch, n)?;
        assert!(layer.initialized);

        // After forward pass the mean of each dimension should be ≈ 0.
        let mut sum = vec![0.0_f32; dim];
        for s in 0..n {
            let x = &batch[s * dim..(s + 1) * dim];
            let (z, _) = layer.forward(x)?;
            for i in 0..dim {
                sum[i] += z[i];
            }
        }
        for i in 0..dim {
            let mean = sum[i] / n as f32;
            assert!(
                mean.abs() < 0.1,
                "dim {i}: mean after ActNorm init should be ≈0, got {mean:.4}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_actnorm_initialize_errors() {
        let mut layer = ActNorm::new(4);
        // n=0.
        assert!(layer.initialize_from_data(&[], 0).is_err());
        // Wrong length.
        assert!(layer.initialize_from_data(&[1.0, 2.0, 3.0], 2).is_err());
    }

    #[test]
    fn test_actnorm_log_det_correct() -> Result<()> {
        let dim = 3_usize;
        let mut layer = ActNorm::new(dim);
        // Manually set log_scale to known values.
        layer.scale = vec![0.5_f32, -0.3, 1.2]; // log_scale values
        layer.bias = vec![0.0; dim];

        let x = vec![1.0_f32, 2.0, 3.0];
        let (_z, log_det) = layer.forward(&x)?;
        // Expected: -(0.5 + (-0.3) + 1.2) = -1.4
        let expected = -(0.5_f32 + (-0.3) + 1.2);
        assert!(
            (log_det - expected).abs() < 1e-5,
            "log_det expected {expected:.6}, got {log_det:.6}"
        );
        Ok(())
    }

    #[test]
    fn test_actnorm_dim_mismatch_error() {
        let layer = ActNorm::new(4);
        assert!(layer.forward(&[1.0, 2.0, 3.0]).is_err());
        assert!(layer.inverse(&[1.0, 2.0, 3.0]).is_err());
    }

    // ── NormalizingFlow tests ──────────────────────────────────────────────────

    #[test]
    fn test_flow_forward_inverse_roundtrip() -> Result<()> {
        let flow = build_simple_flow()?;
        let x = vec![1.0_f32, -0.5, 0.3, 2.1];
        let (z, _log_det) = flow.forward(&x)?;
        let x_rec = flow.inverse(&z)?;
        assert_eq!(x_rec.len(), x.len());
        assert!(
            max_abs_diff(&x, &x_rec) < 1e-4,
            "flow roundtrip error {:.2e}",
            max_abs_diff(&x, &x_rec)
        );
        Ok(())
    }

    #[test]
    fn test_flow_log_likelihood_finite() -> Result<()> {
        let flow = build_simple_flow()?;
        let x = vec![0.3_f32, -0.1, 0.7, -0.5];
        let ll = flow.log_likelihood(&x)?;
        assert!(ll.is_finite(), "log_likelihood must be finite, got {ll}");
        Ok(())
    }

    #[test]
    fn test_flow_nll_loss_finite() -> Result<()> {
        let flow = build_simple_flow()?;
        let batch: Vec<f32> = (0..20).map(|i| (i as f32 - 10.0) * 0.1).collect();
        let nll = flow.nll_loss(&batch, 5)?; // 5 samples × dim=4
        assert!(nll.is_finite(), "nll_loss must be finite, got {nll}");
        Ok(())
    }

    #[test]
    fn test_flow_num_layers() -> Result<()> {
        let flow = build_simple_flow()?;
        assert_eq!(flow.num_layers(), 4);
        Ok(())
    }

    #[test]
    fn test_flow_empty_forward_inverse() -> Result<()> {
        let flow = NormalizingFlow::new(3);
        let x = vec![1.0_f32, 2.0, 3.0];
        let (z, log_det) = flow.forward(&x)?;
        assert_eq!(z, x);
        assert_eq!(log_det, 0.0);
        let x_rec = flow.inverse(&z)?;
        assert_eq!(x_rec, x);
        Ok(())
    }

    #[test]
    fn test_flow_add_layer_dim_mismatch() {
        let mut flow = NormalizingFlow::new(4);
        let wrong_layer = PermutationLayer::reverse(6); // dim=6 ≠ 4
        assert!(
            flow.add_layer(wrong_layer).is_err(),
            "should reject mismatched layer"
        );
    }

    #[test]
    fn test_flow_coupling_plus_permutation() -> Result<()> {
        let mut flow = NormalizingFlow::new(4);
        flow.add_layer(AffineCouplingLayer::new(4, 8)?)?;
        flow.add_layer(PermutationLayer::reverse(4))?;
        flow.add_layer(AffineCouplingLayer::new(4, 8)?)?;

        let x = vec![0.5_f32, -1.0, 1.5, -2.0];
        let (z, _log_det) = flow.forward(&x)?;
        let x_rec = flow.inverse(&z)?;
        assert!(
            max_abs_diff(&x, &x_rec) < 1e-4,
            "coupling+perm roundtrip error {:.2e}",
            max_abs_diff(&x, &x_rec)
        );
        Ok(())
    }

    #[test]
    fn test_flow_input_dim_mismatch() {
        let flow = build_simple_flow().expect("build flow");
        assert!(flow.forward(&[1.0, 2.0]).is_err());
        assert!(flow.inverse(&[1.0, 2.0]).is_err());
        assert!(flow.log_likelihood(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_flow_nll_loss_errors() -> Result<()> {
        let flow = build_simple_flow()?;
        // n=0.
        assert!(flow.nll_loss(&[], 0).is_err());
        // Wrong batch length.
        assert!(flow.nll_loss(&[1.0, 2.0, 3.0], 2).is_err());
        Ok(())
    }

    #[test]
    fn test_flow_log_likelihood_monotone_near_mode() -> Result<()> {
        // A point closer to the mean of the base distribution should have
        // higher log-likelihood than one farther away (all else equal) when
        // the flow is identity-like (ActNorm with zero initialisation).
        let mut flow = NormalizingFlow::new(2);
        flow.add_layer(ActNorm::new(2))?; // identity init
        let x_near = vec![0.0_f32, 0.0]; // near the Gaussian mode
        let x_far = vec![5.0_f32, 5.0]; // far from the Gaussian mode
        let ll_near = flow.log_likelihood(&x_near)?;
        let ll_far = flow.log_likelihood(&x_far)?;
        assert!(
            ll_near > ll_far,
            "near mode should have higher ll: {ll_near:.4} vs {ll_far:.4}"
        );
        Ok(())
    }
}
