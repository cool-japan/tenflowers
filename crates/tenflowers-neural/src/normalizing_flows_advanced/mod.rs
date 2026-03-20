//! Advanced Normalizing Flows — discrete-flow models not covered by `flows.rs`
//! (RealNVP/ActNorm) or `continuous_normalizing_flows/` (FFJORD/FlowMatching).
//!
//! # Implemented models
//!
//! | Type | Reference |
//! |------|-----------|
//! | [`NfaInvertible1x1Conv`] | Glow — LU-decomposed invertible linear (Kingma & Dhariwal 2018) |
//! | [`NfaGlowStep`] | One Glow step: ActNorm → Inv1x1 → AffineCoupling |
//! | [`NfaGlowModel`] | Multi-scale Glow: squeeze → K steps → split |
//! | [`NfaMaskedAutoregressive`] | MAF: MADE-based autoregressive flow (Papamakarios 2017) |
//! | [`NfaInverseAutoregressive`] | IAF: inverse of MAF — fast sampling (Kingma 2016) |
//! | [`NfaNeuralSpline`] | Neural Spline Flows — rational-quadratic splines (Durkan 2019) |
//! | [`NfaRadialFlow`] | Radial flows (Rezende & Mohamed 2015) |
//! | [`NfaHouseholderFlow`] | Householder orthogonal reflections (Tomczak 2016) |
//! | [`NfaFlowVAE`] | VAE with normalizing-flow posterior |
//! | [`NfaMetrics`] | BPD, NLL, KL diagnostics |
//!
//! All computation is `f32`-based, 100% pure Rust, no C/Fortran dependencies.
//! Every public API returns `Result<_, TensorError>` — no `unwrap()` usage.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Internal math helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Sigmoid activation: 1 / (1 + exp(-x)).
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Softplus: log(1 + exp(x)).
#[inline]
fn softplus(x: f32) -> f32 {
    (1.0 + x.exp()).ln()
}

/// Elementwise ReLU.
#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Xavier uniform init for weight matrix [fan_in × fan_out].
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

/// Zero vector of length n.
#[inline]
fn zeros(n: usize) -> Vec<f32> {
    vec![0.0_f32; n]
}

/// Log-normal pdf: -(x-mu)²/(2σ²) - log σ - 0.5*log(2π).
#[inline]
fn log_normal(x: f32, mu: f32, sigma: f32) -> f32 {
    let diff = x - mu;
    -diff * diff / (2.0 * sigma * sigma) - sigma.ln() - 0.5 * (2.0 * std::f32::consts::PI).ln()
}

/// Standard normal log-pdf.
#[inline]
fn log_standard_normal(z: f32) -> f32 {
    -0.5 * z * z - 0.5 * (2.0 * std::f32::consts::PI).ln()
}

/// Dense linear layer: out[j] = Σ_i in[i]*W[i*out+j] + b[j].
fn linear_fwd(input: &[f32], w: &[f32], b: &[f32], out_dim: usize) -> Result<Vec<f32>> {
    let in_dim = input.len();
    if w.len() != in_dim * out_dim {
        return Err(TensorError::invalid_argument_op(
            "linear_fwd",
            &format!(
                "weight size mismatch: expected {}, got {}",
                in_dim * out_dim,
                w.len()
            ),
        ));
    }
    if b.len() != out_dim {
        return Err(TensorError::invalid_argument_op(
            "linear_fwd",
            &format!("bias size mismatch: expected {out_dim}, got {}", b.len()),
        ));
    }
    let mut out = b.to_vec();
    for i in 0..in_dim {
        let xi = input[i];
        for j in 0..out_dim {
            out[j] += xi * w[i * out_dim + j];
        }
    }
    Ok(out)
}

/// Linear + ReLU.
fn linear_relu_fwd(input: &[f32], w: &[f32], b: &[f32], out_dim: usize) -> Result<Vec<f32>> {
    let mut out = linear_fwd(input, w, b, out_dim)?;
    for v in out.iter_mut() {
        *v = relu(*v);
    }
    Ok(out)
}

/// Softmax over a slice.
fn softmax(v: &[f32]) -> Vec<f32> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let denom = if sum.abs() < 1e-12 { 1.0 } else { sum };
    exps.iter().map(|e| e / denom).collect()
}

/// Box-Muller standard normal sample.
fn sample_standard_normal(rng: &mut StdRng) -> f32 {
    let u1: f32 = rng.random::<f32>().max(1e-10_f32);
    let u2: f32 = rng.random::<f32>();
    (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * std::f32::consts::PI * u2).cos()
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  NfaInvertible1x1Conv — LU-decomposed invertible linear
// ─────────────────────────────────────────────────────────────────────────────

/// Invertible 1×1 convolution via LU decomposition.
///
/// Stores the weight matrix W and its LU factorization PA=LU.
/// Forward: z = W·x in O(d²), log|det W| = Σ log|U_{ii}|.
/// Inverse: x = W⁻¹·z via two triangular solves in O(d²).
pub struct NfaInvertible1x1Conv {
    /// Dimension d.
    pub dim: usize,
    /// Weight matrix W (row-major, d×d).
    w_mat: Vec<f32>,
    /// Row permutation from partial pivoting: perm[i] = original row at position i.
    perm: Vec<usize>,
    /// LU factored form of P·W = L·U stored in one array (L below diag, U on/above).
    lu: Vec<f32>,
    /// log|det W| (precomputed).
    log_det_val: f32,
}

impl NfaInvertible1x1Conv {
    /// Construct with a random invertible W.
    pub fn new(dim: usize, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "NfaInvertible1x1Conv::new",
                "dim must be > 0",
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        // Random Gaussian matrix: well-conditioned with high probability.
        let w: Vec<f32> = (0..dim * dim)
            .map(|_| sample_standard_normal(&mut rng))
            .collect();
        Self::from_w(w, dim)
    }

    /// Build from an arbitrary weight matrix.
    fn from_w(w: Vec<f32>, dim: usize) -> Result<Self> {
        // Compute LU factorization of W with partial pivoting: P·W = L·U.
        let mut lu = w.clone();
        let mut perm: Vec<usize> = (0..dim).collect();

        for k in 0..dim {
            // Find pivot in column k from row k downward.
            let mut max_val = lu[k * dim + k].abs();
            let mut max_row = k;
            for r in (k + 1)..dim {
                let v = lu[r * dim + k].abs();
                if v > max_val {
                    max_val = v;
                    max_row = r;
                }
            }
            if max_val < 1e-9 {
                lu[k * dim + k] += 1e-6;
            }
            if max_row != k {
                perm.swap(k, max_row);
                for j in 0..dim {
                    lu.swap(k * dim + j, max_row * dim + j);
                }
            }
            let pivot = lu[k * dim + k];
            let pivot_safe = pivot + if pivot.abs() < 1e-15 { 1e-15 } else { 0.0 };
            for r in (k + 1)..dim {
                let factor = lu[r * dim + k] / pivot_safe;
                lu[r * dim + k] = factor;
                for j in (k + 1)..dim {
                    let sub = factor * lu[k * dim + j];
                    lu[r * dim + j] -= sub;
                }
            }
        }

        // log|det W| = Σ log|U_{kk}| (sign from permutation is irrelevant for log-abs-det).
        let log_det_val: f32 = (0..dim)
            .map(|k| lu[k * dim + k].abs().max(1e-12).ln())
            .sum();

        Ok(Self {
            dim,
            w_mat: w,
            perm,
            lu,
            log_det_val,
        })
    }

    /// Forward: z = W·x (plain matrix-vector multiply), log_det precomputed.
    pub fn forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)> {
        let d = self.dim;
        if x.len() != d {
            return Err(TensorError::invalid_argument_op(
                "NfaInvertible1x1Conv::forward",
                "input length mismatch",
            ));
        }
        let mut z = zeros(d);
        for i in 0..d {
            let mut s = 0.0_f32;
            for j in 0..d {
                s += self.w_mat[i * d + j] * x[j];
            }
            z[i] = s;
        }
        Ok((z, self.log_det_val))
    }

    /// Inverse: x = W⁻¹·z via LU solve.
    ///
    /// Solves P·W·x = P·z using the stored LU factorization.
    /// i.e., L·(U·x) = P·z → forward substitution, then back substitution.
    pub fn inverse(&self, z: &[f32]) -> Result<Vec<f32>> {
        let d = self.dim;
        if z.len() != d {
            return Err(TensorError::invalid_argument_op(
                "NfaInvertible1x1Conv::inverse",
                "input length mismatch",
            ));
        }
        // Apply P: b[i] = z[perm[i]]
        let b: Vec<f32> = self.perm.iter().map(|&pi| z[pi]).collect();

        // Forward substitution: L·y = b (L is unit lower triangular)
        let mut y = zeros(d);
        for i in 0..d {
            let mut s = b[i];
            for j in 0..i {
                s -= self.lu[i * d + j] * y[j]; // lu[i,j] for j<i is the L factor
            }
            y[i] = s;
        }

        // Back substitution: U·x = y (U is upper triangular, stored in lu on/above diagonal)
        let mut x = zeros(d);
        for i in (0..d).rev() {
            let mut s = y[i];
            for j in (i + 1)..d {
                s -= self.lu[i * d + j] * x[j];
            }
            let diag = self.lu[i * d + i];
            if diag.abs() < 1e-12 {
                return Err(TensorError::invalid_argument_op(
                    "NfaInvertible1x1Conv::inverse",
                    "near-singular U diagonal in LU solve",
                ));
            }
            x[i] = s / diag;
        }
        Ok(x)
    }

    /// log|det W| (precomputed).
    pub fn log_det(&self) -> f32 {
        self.log_det_val
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §1b  NfaActNorm — per-channel bias & scale (data-dependent init)
// ─────────────────────────────────────────────────────────────────────────────

/// ActNorm layer: z_i = (x_i - b_i) / exp(s_i).
/// log|det J| = -Σ s_i.
pub struct NfaActNorm {
    /// Dimension.
    pub dim: usize,
    /// Log-scale parameters (trainable, data-initialized).
    pub log_scale: Vec<f32>,
    /// Bias parameters (trainable, data-initialized).
    pub bias: Vec<f32>,
    initialized: bool,
}

impl NfaActNorm {
    /// Create with zero bias and zero log-scale (scale = 1).
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            log_scale: zeros(dim),
            bias: zeros(dim),
            initialized: false,
        }
    }

    /// Initialize from a batch of samples (mean/std initialization).
    pub fn initialize_from_batch(&mut self, batch: &[Vec<f32>]) -> Result<()> {
        if batch.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "NfaActNorm::initialize_from_batch",
                "empty batch",
            ));
        }
        let n = batch.len() as f32;
        let mut mean = zeros(self.dim);
        for sample in batch {
            if sample.len() != self.dim {
                return Err(TensorError::invalid_argument_op(
                    "NfaActNorm::initialize_from_batch",
                    "sample dimension mismatch",
                ));
            }
            for (i, &v) in sample.iter().enumerate() {
                mean[i] += v / n;
            }
        }
        let mut var = zeros(self.dim);
        for sample in batch {
            for (i, &v) in sample.iter().enumerate() {
                let d = v - mean[i];
                var[i] += d * d / n;
            }
        }
        for i in 0..self.dim {
            self.bias[i] = mean[i];
            self.log_scale[i] = var[i].sqrt().max(1e-5).ln();
        }
        self.initialized = true;
        Ok(())
    }

    /// Forward: z = (x - b) * exp(-s).
    pub fn forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)> {
        if x.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "NfaActNorm::forward",
                "input length mismatch",
            ));
        }
        let mut z = Vec::with_capacity(self.dim);
        let mut log_det = 0.0_f32;
        for i in 0..self.dim {
            let s = self.log_scale[i];
            z.push((x[i] - self.bias[i]) * (-s).exp());
            log_det -= s;
        }
        Ok((z, log_det))
    }

    /// Inverse: x = z * exp(s) + b.
    pub fn inverse(&self, z: &[f32]) -> Result<Vec<f32>> {
        if z.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "NfaActNorm::inverse",
                "input length mismatch",
            ));
        }
        let x: Vec<f32> = (0..self.dim)
            .map(|i| z[i] * self.log_scale[i].exp() + self.bias[i])
            .collect();
        Ok(x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §1c  Affine coupling (shared with flows.rs concept, distinct Nfa prefix)
// ─────────────────────────────────────────────────────────────────────────────

/// Two-hidden-layer coupling network returning (s, t) ∈ ℝ^{dim_out} each.
struct NfaCouplingNet {
    in_dim: usize,
    out_dim: usize,
    hidden: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    ws: Vec<f32>,
    bs: Vec<f32>,
    wt: Vec<f32>,
    bt: Vec<f32>,
}

impl NfaCouplingNet {
    fn new(in_dim: usize, out_dim: usize, hidden: usize, seed: u64) -> Self {
        Self {
            in_dim,
            out_dim,
            hidden,
            w1: xavier_uniform(in_dim, hidden, seed),
            b1: zeros(hidden),
            w2: xavier_uniform(hidden, hidden, seed.wrapping_add(1)),
            b2: zeros(hidden),
            ws: xavier_uniform(hidden, out_dim, seed.wrapping_add(2)),
            bs: zeros(out_dim),
            wt: xavier_uniform(hidden, out_dim, seed.wrapping_add(3)),
            bt: zeros(out_dim),
        }
    }

    fn forward(&self, x: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let h1 = linear_relu_fwd(x, &self.w1, &self.b1, self.hidden)?;
        let h2 = linear_relu_fwd(&h1, &self.w2, &self.b2, self.hidden)?;
        let mut s = linear_fwd(&h2, &self.ws, &self.bs, self.out_dim)?;
        let t = linear_fwd(&h2, &self.wt, &self.bt, self.out_dim)?;
        // Clamp s to keep exp(s) numerically stable
        for v in s.iter_mut() {
            *v = v.tanh() * 2.0;
        }
        Ok((s, t))
    }
}

/// NfaAffineCoupling: mask-based affine coupling step.
/// x1 = x[0..split], x2 = x[split..dim].
/// z1 = x1, z2 = x2 ⊙ exp(s) + t, s,t = net(x1).
pub struct NfaAffineCoupling {
    /// Total dimension.
    pub dim: usize,
    /// Split index.
    pub split: usize,
    net: NfaCouplingNet,
}

impl NfaAffineCoupling {
    /// Create with given split index.
    pub fn new(dim: usize, hidden: usize, seed: u64) -> Result<Self> {
        if dim < 2 {
            return Err(TensorError::invalid_argument_op(
                "NfaAffineCoupling::new",
                "dim must be >= 2",
            ));
        }
        let split = dim / 2;
        let net = NfaCouplingNet::new(split, dim - split, hidden, seed);
        Ok(Self { dim, split, net })
    }

    /// Forward: z = (x1, x2 ⊙ exp(s(x1)) + t(x1)).
    pub fn forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)> {
        if x.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "NfaAffineCoupling::forward",
                "input length mismatch",
            ));
        }
        let x1 = &x[..self.split];
        let x2 = &x[self.split..];
        let (s, t) = self.net.forward(x1)?;
        let mut z = x1.to_vec();
        let mut log_det = 0.0_f32;
        for idx in 0..x2.len() {
            let xi: f32 = x2[idx];
            let si: f32 = s[idx];
            let ti: f32 = t[idx];
            z.push(xi * si.exp() + ti);
            log_det += si;
        }
        Ok((z, log_det))
    }

    /// Inverse.
    pub fn inverse(&self, z: &[f32]) -> Result<Vec<f32>> {
        if z.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "NfaAffineCoupling::inverse",
                "input length mismatch",
            ));
        }
        let z1 = &z[..self.split];
        let z2 = &z[self.split..];
        let (s, t) = self.net.forward(z1)?;
        let mut x = z1.to_vec();
        for idx in 0..z2.len() {
            let zi: f32 = z2[idx];
            let si: f32 = s[idx];
            let ti: f32 = t[idx];
            x.push((zi - ti) * (-si).exp());
        }
        Ok(x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  NfaGlowStep / NfaGlowModel
// ─────────────────────────────────────────────────────────────────────────────

/// One Glow step: ActNorm → Inv1x1Conv → AffineCoupling.
pub struct NfaGlowStep {
    /// Dimension.
    pub dim: usize,
    /// ActNorm layer.
    pub act_norm: NfaActNorm,
    /// Invertible 1×1 convolution.
    pub inv1x1: NfaInvertible1x1Conv,
    /// Affine coupling layer.
    pub coupling: NfaAffineCoupling,
}

impl NfaGlowStep {
    /// Create a new Glow step.
    pub fn new(dim: usize, hidden: usize, seed: u64) -> Result<Self> {
        Ok(Self {
            dim,
            act_norm: NfaActNorm::new(dim),
            inv1x1: NfaInvertible1x1Conv::new(dim, seed)?,
            coupling: NfaAffineCoupling::new(dim, hidden, seed.wrapping_add(1000))?,
        })
    }

    /// Forward: x → (z, log_det).
    pub fn forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)> {
        let (h1, ld1) = self.act_norm.forward(x)?;
        let (h2, ld2) = self.inv1x1.forward(&h1)?;
        let (z, ld3) = self.coupling.forward(&h2)?;
        Ok((z, ld1 + ld2 + ld3))
    }

    /// Inverse: z → x.
    pub fn inverse(&self, z: &[f32]) -> Result<Vec<f32>> {
        let h2 = self.coupling.inverse(z)?;
        let h1 = self.inv1x1.inverse(&h2)?;
        self.act_norm.inverse(&h1)
    }
}

/// Multi-scale Glow model.
///
/// Architecture per scale level: squeeze → K NfaGlowSteps → split-off half.
/// For simplicity (flat vectors, not image tensors) the squeeze/split are
/// implemented as interleaved index selection.
pub struct NfaGlowModel {
    /// Input dimension (must be divisible by 2^num_levels).
    pub dim: usize,
    /// Number of scale levels.
    pub num_levels: usize,
    /// Steps per level.
    pub steps_per_level: usize,
    /// Flow steps organized as \[level\]\[step\].
    pub steps: Vec<Vec<NfaGlowStep>>,
}

impl NfaGlowModel {
    /// Build a new Glow model.
    pub fn new(
        dim: usize,
        num_levels: usize,
        steps_per_level: usize,
        hidden: usize,
        seed: u64,
    ) -> Result<Self> {
        if dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "NfaGlowModel::new",
                "dim must be > 0",
            ));
        }
        if num_levels == 0 {
            return Err(TensorError::invalid_argument_op(
                "NfaGlowModel::new",
                "num_levels must be > 0",
            ));
        }
        let factor = 1usize << num_levels;
        if dim % factor != 0 {
            return Err(TensorError::invalid_argument_op(
                "NfaGlowModel::new",
                &format!("dim ({dim}) must be divisible by 2^num_levels ({factor})"),
            ));
        }
        let mut steps = Vec::with_capacity(num_levels);
        let mut cur_dim = dim;
        for level in 0..num_levels {
            // Steps at this level operate on cur_dim (full dimension before split).
            let level_dim = cur_dim;
            let mut level_steps = Vec::with_capacity(steps_per_level);
            for step in 0..steps_per_level {
                let s = NfaGlowStep::new(
                    level_dim,
                    hidden,
                    seed.wrapping_add((level * 1000 + step * 100) as u64),
                )?;
                level_steps.push(s);
            }
            steps.push(level_steps);
            // After split (except last level), the continuation has half the size.
            if level < num_levels - 1 {
                cur_dim /= 2;
            }
        }
        Ok(Self {
            dim,
            num_levels,
            steps_per_level,
            steps,
        })
    }

    /// Forward: x → (z_concat, total_log_det).
    /// Returns concatenation of all split-off latent codes plus final code.
    pub fn forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)> {
        if x.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "NfaGlowModel::forward",
                "input length mismatch",
            ));
        }
        let mut h = x.to_vec();
        let mut total_ld = 0.0_f32;
        let mut latent_parts: Vec<Vec<f32>> = Vec::new();

        for (level, level_steps) in self.steps.iter().enumerate() {
            // Run flow steps
            for step in level_steps {
                let (hz, ld) = step.forward(&h)?;
                h = hz;
                total_ld += ld;
            }
            // Split: at each level except last, split off the second half
            if level < self.num_levels - 1 {
                let mid = h.len() / 2;
                let latent = h.split_off(mid);
                latent_parts.push(latent);
            }
        }
        // h now holds the final latent
        latent_parts.push(h);
        let z: Vec<f32> = latent_parts.into_iter().flatten().collect();
        Ok((z, total_ld))
    }

    /// Inverse: z_concat → x.
    pub fn inverse(&self, z: &[f32]) -> Result<Vec<f32>> {
        if z.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "NfaGlowModel::inverse",
                "z length must equal dim",
            ));
        }
        // Reconstruct level sizes
        let mut level_sizes: Vec<usize> = Vec::with_capacity(self.num_levels);
        let mut cur = self.dim;
        for level in 0..self.num_levels {
            if level < self.num_levels - 1 {
                level_sizes.push(cur / 2);
                cur /= 2;
            } else {
                level_sizes.push(cur);
            }
        }
        // Parse z into parts (last level first)
        let mut parts: Vec<Vec<f32>> = Vec::new();
        let mut offset = 0;
        // Split-off parts are stored in order [level0_latent, level1_latent, ..., final]
        // But sizes from split: level0 gives dim/2, level1 gives dim/4, etc.
        let mut sz = self.dim;
        let mut split_sizes: Vec<usize> = Vec::new();
        for level in 0..self.num_levels {
            if level < self.num_levels - 1 {
                split_sizes.push(sz / 2);
                sz /= 2;
            } else {
                split_sizes.push(sz);
            }
        }
        for &s in &split_sizes {
            parts.push(z[offset..offset + s].to_vec());
            offset += s;
        }

        // Reconstruct from last level to first
        let mut h = parts.pop().ok_or_else(|| {
            TensorError::invalid_argument_op("NfaGlowModel::inverse", "empty parts")
        })?;

        for level in (0..self.num_levels).rev() {
            // Re-join with split-off part (except last level which has no join)
            if level < self.num_levels - 1 {
                let latent = parts.pop().ok_or_else(|| {
                    TensorError::invalid_argument_op("NfaGlowModel::inverse", "missing latent part")
                })?;
                // h was the continuation, latent was the split-off second half
                let mut joined = h;
                joined.extend_from_slice(&latent);
                h = joined;
            }
            // Reverse flow steps
            for step in self.steps[level].iter().rev() {
                h = step.inverse(&h)?;
            }
        }
        Ok(h)
    }

    /// Log-likelihood of x under isotropic Gaussian base.
    pub fn log_likelihood(&self, x: &[f32]) -> Result<f32> {
        let (z, log_det) = self.forward(x)?;
        let log_pz: f32 = z.iter().map(|&zi| log_standard_normal(zi)).sum();
        Ok(log_pz + log_det)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  NfaMaskedAutoregressive — MAF (Papamakarios 2017)
// ─────────────────────────────────────────────────────────────────────────────

/// MADE network for masked autoregressive transforms.
/// Implements connectivity masks ensuring s_i, t_i = f(x_{1..i-1}).
struct NfaMade {
    dim: usize,
    hidden: usize,
    /// Masks: w1_mask[i*hidden+h] = 1 if x_i can affect h_h.
    w1_mask: Vec<f32>,
    w2s_mask: Vec<f32>,
    w2t_mask: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2s: Vec<f32>,
    b2s: Vec<f32>,
    w2t: Vec<f32>,
    b2t: Vec<f32>,
}

impl NfaMade {
    fn new(dim: usize, hidden: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        // Assign orders m_h ∈ {1..dim-1} to hidden units
        let h_orders: Vec<usize> = (0..hidden).map(|h| 1 + (h % (dim - 1))).collect();
        // Input order: x_i has order i+1 (1-indexed)
        // Mask w1: h unit h gets input i iff m_h >= i+1 → NO autoregressive: we want m_h >= m_i
        // Standard MADE: W1[i,h] = 1 if m_h >= m_i; W2[h,k] = 1 if m_k > m_h
        let mut w1_mask = vec![0.0_f32; dim * hidden];
        for i in 0..dim {
            let mi = i + 1; // 1-indexed order
            for h in 0..hidden {
                if h_orders[h] >= mi {
                    w1_mask[i * hidden + h] = 1.0;
                }
            }
        }
        // Output masks for s and t: output k has order k+1
        let mut w2s_mask = vec![0.0_f32; hidden * dim];
        let mut w2t_mask = vec![0.0_f32; hidden * dim];
        for h in 0..hidden {
            for k in 0..dim {
                let mk = k + 1;
                if mk > h_orders[h] {
                    w2s_mask[h * dim + k] = 1.0;
                    w2t_mask[h * dim + k] = 1.0;
                }
            }
        }
        // Initialize weights
        let limit = (6.0 / (dim + hidden) as f64).sqrt() as f32;
        let randf = |rng: &mut StdRng| {
            let u: f32 = rng.random();
            u * 2.0 * limit - limit
        };
        let w1: Vec<f32> = (0..dim * hidden).map(|_| randf(&mut rng)).collect();
        let b1 = zeros(hidden);
        let w2s: Vec<f32> = (0..hidden * dim).map(|_| randf(&mut rng)).collect();
        let b2s = zeros(dim);
        let w2t: Vec<f32> = (0..hidden * dim).map(|_| randf(&mut rng)).collect();
        let b2t = zeros(dim);
        Self {
            dim,
            hidden,
            w1_mask,
            w2s_mask,
            w2t_mask,
            w1,
            b1,
            w2s,
            b2s,
            w2t,
            b2t,
        }
    }

    /// Masked forward: returns (s, t) of shape [dim] each.
    fn forward(&self, x: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let d = self.dim;
        let h = self.hidden;
        // h1 = relu(W1 ⊙ M1 · x + b1)
        let mut h1 = self.b1.clone();
        for i in 0..d {
            let xi = x[i];
            for hj in 0..h {
                h1[hj] += xi * self.w1[i * h + hj] * self.w1_mask[i * h + hj];
            }
        }
        for v in h1.iter_mut() {
            *v = relu(*v);
        }
        // s = (W2s ⊙ M2) · h1 + b2s  (tanh to bound)
        let mut s = self.b2s.clone();
        let mut t = self.b2t.clone();
        for hj in 0..h {
            let hv = h1[hj];
            for k in 0..d {
                s[k] += hv * self.w2s[hj * d + k] * self.w2s_mask[hj * d + k];
                t[k] += hv * self.w2t[hj * d + k] * self.w2t_mask[hj * d + k];
            }
        }
        for v in s.iter_mut() {
            *v = v.tanh() * 2.0;
        }
        Ok((s, t))
    }
}

/// Masked Autoregressive Flow (MAF).
///
/// z_i = (x_i - t_i(x_{<i})) * exp(-s_i(x_{<i}))
/// log|det J| = -Σ s_i
///
/// Forward: sequential O(d) passes through MADE.
/// Inverse: one forward pass through MADE (parallel).
pub struct NfaMaskedAutoregressive {
    /// Dimension.
    pub dim: usize,
    /// Number of MADE steps (flow depth).
    pub num_steps: usize,
    made_layers: Vec<NfaMade>,
    /// Permutations between steps (for expressivity).
    perms: Vec<Vec<usize>>,
}

impl NfaMaskedAutoregressive {
    /// Create a MAF with given depth.
    pub fn new(dim: usize, hidden: usize, num_steps: usize, seed: u64) -> Result<Self> {
        if dim < 2 {
            return Err(TensorError::invalid_argument_op(
                "NfaMaskedAutoregressive::new",
                "dim must be >= 2",
            ));
        }
        if num_steps == 0 {
            return Err(TensorError::invalid_argument_op(
                "NfaMaskedAutoregressive::new",
                "num_steps must be > 0",
            ));
        }
        let mut made_layers = Vec::with_capacity(num_steps);
        let mut perms = Vec::with_capacity(num_steps);
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(99));
        for step in 0..num_steps {
            made_layers.push(NfaMade::new(
                dim,
                hidden,
                seed.wrapping_add(step as u64 * 71),
            ));
            // Random permutation
            let mut perm: Vec<usize> = (0..dim).collect();
            for i in (1..dim).rev() {
                let j = (rng.random::<f32>() * (i + 1) as f32) as usize % (i + 1);
                perm.swap(i, j);
            }
            perms.push(perm);
        }
        Ok(Self {
            dim,
            num_steps,
            made_layers,
            perms,
        })
    }

    /// Forward: x → (z, log_det).  Sequential per dimension.
    pub fn forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)> {
        if x.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "NfaMaskedAutoregressive::forward",
                "input length mismatch",
            ));
        }
        let mut h = x.to_vec();
        let mut total_ld = 0.0_f32;
        for (made, perm) in self.made_layers.iter().zip(self.perms.iter()) {
            let (s, t) = made.forward(&h)?;
            let mut z = zeros(self.dim);
            let mut ld = 0.0_f32;
            for i in 0..self.dim {
                z[i] = (h[i] - t[i]) * (-s[i]).exp();
                ld -= s[i];
            }
            // Apply permutation
            let mut permuted = zeros(self.dim);
            for (i, &pi) in perm.iter().enumerate() {
                permuted[pi] = z[i];
            }
            h = permuted;
            total_ld += ld;
        }
        Ok((h, total_ld))
    }

    /// Inverse: z → x.  Parallel per MADE step.
    pub fn inverse(&self, z: &[f32]) -> Result<Vec<f32>> {
        if z.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "NfaMaskedAutoregressive::inverse",
                "input length mismatch",
            ));
        }
        let mut h = z.to_vec();
        // Reverse through steps
        for (made, perm) in self.made_layers.iter().zip(self.perms.iter()).rev() {
            // Undo permutation
            let mut unpermuted = zeros(self.dim);
            for (i, &pi) in perm.iter().enumerate() {
                unpermuted[i] = h[pi];
            }
            // MADE inverse: given z, compute x_i = z_i * exp(s_i(x_{<i})) + t_i(x_{<i})
            // This requires sequential computation
            let mut x = zeros(self.dim);
            for i in 0..self.dim {
                // Use current x to compute s_i, t_i (autoregressive)
                let (s, t) = made.forward(&x)?;
                x[i] = unpermuted[i] * s[i].exp() + t[i];
            }
            h = x;
        }
        Ok(h)
    }

    /// Log-likelihood under standard normal base.
    pub fn log_likelihood(&self, x: &[f32]) -> Result<f32> {
        let (z, log_det) = self.forward(x)?;
        let log_pz: f32 = z.iter().map(|&zi| log_standard_normal(zi)).sum();
        Ok(log_pz + log_det)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  NfaInverseAutoregressive — IAF (Kingma 2016)
// ─────────────────────────────────────────────────────────────────────────────

/// Inverse Autoregressive Flow.
///
/// IAF is the inverse of MAF:
///   z_i = x_i * exp(s_i(z_{<i})) + t_i(z_{<i})
///
/// Fast sampling (one MADE pass), slow density evaluation (sequential).
/// Ideal as variational posterior: q(z|x) uses IAF to enrich Gaussian base.
pub struct NfaInverseAutoregressive {
    /// Dimension.
    pub dim: usize,
    /// Number of IAF steps.
    pub num_steps: usize,
    made_layers: Vec<NfaMade>,
    perms: Vec<Vec<usize>>,
}

impl NfaInverseAutoregressive {
    /// Create IAF.
    pub fn new(dim: usize, hidden: usize, num_steps: usize, seed: u64) -> Result<Self> {
        if dim < 2 {
            return Err(TensorError::invalid_argument_op(
                "NfaInverseAutoregressive::new",
                "dim must be >= 2",
            ));
        }
        let mut made_layers = Vec::with_capacity(num_steps);
        let mut perms = Vec::with_capacity(num_steps);
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(17));
        for step in 0..num_steps {
            made_layers.push(NfaMade::new(
                dim,
                hidden,
                seed.wrapping_add(step as u64 * 53),
            ));
            let mut perm: Vec<usize> = (0..dim).collect();
            for i in (1..dim).rev() {
                let j = (rng.random::<f32>() * (i + 1) as f32) as usize % (i + 1);
                perm.swap(i, j);
            }
            perms.push(perm);
        }
        Ok(Self {
            dim,
            num_steps,
            made_layers,
            perms,
        })
    }

    /// Forward (sampling direction): x ~ N(0,I) → z.  One MADE pass per step.
    pub fn forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)> {
        if x.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "NfaInverseAutoregressive::forward",
                "input length mismatch",
            ));
        }
        let mut h = x.to_vec();
        let mut total_ld = 0.0_f32;
        for (made, perm) in self.made_layers.iter().zip(self.perms.iter()) {
            let (s, t) = made.forward(&h)?;
            let mut z = zeros(self.dim);
            let mut ld = 0.0_f32;
            for i in 0..self.dim {
                z[i] = h[i] * s[i].exp() + t[i];
                ld += s[i];
            }
            let mut permuted = zeros(self.dim);
            for (i, &pi) in perm.iter().enumerate() {
                permuted[pi] = z[i];
            }
            h = permuted;
            total_ld += ld;
        }
        Ok((h, total_ld))
    }

    /// Inverse (density evaluation direction): z → x.  Sequential per step.
    pub fn inverse(&self, z: &[f32]) -> Result<Vec<f32>> {
        if z.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "NfaInverseAutoregressive::inverse",
                "input length mismatch",
            ));
        }
        let mut h = z.to_vec();
        for (made, perm) in self.made_layers.iter().zip(self.perms.iter()).rev() {
            // Undo permutation
            let mut unpermuted = zeros(self.dim);
            for (i, &pi) in perm.iter().enumerate() {
                unpermuted[i] = h[pi];
            }
            // Sequential: x_i = (z_i - t_i(z_{<i})) * exp(-s_i(z_{<i}))
            let mut x = zeros(self.dim);
            for i in 0..self.dim {
                let (s, t) = made.forward(&x)?;
                x[i] = (unpermuted[i] - t[i]) * (-s[i]).exp();
            }
            h = x;
        }
        Ok(h)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  NfaNeuralSpline — Rational-Quadratic Spline Flows (Durkan 2019)
// ─────────────────────────────────────────────────────────────────────────────

/// Rational-Quadratic (RQ) monotone spline for one dimension.
/// K bins parameterized by widths, heights, and derivatives.
struct RqSpline {
    /// Number of bins K.
    k: usize,
    /// Bin widths W_k (positive, sum = domain_width).
    widths: Vec<f32>,
    /// Bin heights H_k (positive, sum = domain_height).
    heights: Vec<f32>,
    /// Knot derivatives d_k (positive), length K+1.
    derivs: Vec<f32>,
    /// Left boundary of spline domain.
    left: f32,
    /// Right boundary of spline domain.
    right: f32,
    /// Bottom boundary.
    bottom: f32,
    /// Top boundary.
    top: f32,
}

impl RqSpline {
    /// Build from raw network output of length 3K+1.
    fn from_params(params: &[f32], k: usize, left: f32, right: f32) -> Result<Self> {
        if params.len() < 3 * k + 1 {
            return Err(TensorError::invalid_argument_op(
                "RqSpline::from_params",
                "params length mismatch",
            ));
        }
        // widths and heights from softmax, derivatives from softplus
        let w_raw = &params[0..k];
        let h_raw = &params[k..2 * k];
        let d_raw = &params[2 * k..3 * k + 1];

        let domain = right - left;
        let w_sm = softmax(w_raw);
        let widths: Vec<f32> = w_sm.iter().map(|&w| w * domain).collect();

        let h_sm = softmax(h_raw);
        let heights: Vec<f32> = h_sm.iter().map(|&h| h * domain).collect();

        let derivs: Vec<f32> = d_raw.iter().map(|&d| softplus(d) + 1e-5).collect();

        Ok(Self {
            k,
            widths,
            heights,
            derivs,
            left,
            right,
            bottom: left,
            top: right,
        })
    }

    /// Cumulative widths/heights for bin lookup.
    fn cum_widths(&self) -> Vec<f32> {
        let mut c = vec![self.left];
        for &w in &self.widths {
            c.push(c.last().copied().unwrap_or(self.left) + w);
        }
        c
    }

    fn cum_heights(&self) -> Vec<f32> {
        let mut c = vec![self.bottom];
        for &h in &self.heights {
            c.push(c.last().copied().unwrap_or(self.bottom) + h);
        }
        c
    }

    /// Find bin index k* such that cum_widths[k*] <= x < cum_widths[k*+1].
    fn find_bin(&self, x: f32, cum: &[f32]) -> usize {
        let mut lo = 0;
        let mut hi = self.k;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if cum[mid + 1] <= x {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo.min(self.k - 1)
    }

    /// Forward spline transform (x within domain → y).
    fn forward_scalar(&self, x: f32) -> Result<(f32, f32)> {
        let x_clamped = x.clamp(self.left + 1e-6, self.right - 1e-6);
        let cum_w = self.cum_widths();
        let cum_h = self.cum_heights();
        let k = self.find_bin(x_clamped, &cum_w);
        let w_k = self.widths[k];
        let h_k = self.heights[k];
        let d_k = self.derivs[k];
        let d_k1 = self.derivs[k + 1];
        let x_w = cum_w[k];
        let y_h = cum_h[k];

        if w_k < 1e-12 {
            return Ok((y_h, 0.0));
        }

        let xi = (x_clamped - x_w) / w_k; // ∈ [0,1]
        let s_k = h_k / w_k;

        // RQ formula (Durkan 2019 eq 4)
        let num = h_k * (s_k * xi * xi + d_k * xi * (1.0 - xi));
        let denom = s_k + (d_k + d_k1 - 2.0 * s_k) * xi * (1.0 - xi);
        let denom_safe = denom.abs().max(1e-8);

        let y = y_h + num / denom_safe;

        // Derivative (log det)
        let deriv_num =
            s_k * s_k * (d_k1 * xi * xi + 2.0 * s_k * xi * (1.0 - xi) + d_k * (1.0 - xi).powi(2));
        let log_det = (deriv_num / (denom_safe * denom_safe))
            .abs()
            .max(1e-12)
            .ln();

        Ok((y, log_det))
    }

    /// Inverse spline (y → x) via Newton's method.
    fn inverse_scalar(&self, y: f32) -> Result<f32> {
        let y_clamped = y.clamp(self.bottom + 1e-6, self.top - 1e-6);
        let cum_w = self.cum_widths();
        let cum_h = self.cum_heights();
        let k = self.find_bin(y_clamped, &cum_h);
        let w_k = self.widths[k];
        let h_k = self.heights[k];
        let d_k = self.derivs[k];
        let d_k1 = self.derivs[k + 1];
        let x_w = cum_w[k];
        let y_h = cum_h[k];

        if h_k < 1e-12 {
            return Ok(x_w);
        }

        let s_k = h_k / w_k;
        let dy = y_clamped - y_h;

        // Solve quadratic: a·xi² + b·xi + c = 0
        let a = h_k * (s_k - d_k) + dy * (d_k + d_k1 - 2.0 * s_k);
        let b = h_k * d_k - dy * (d_k + d_k1 - 2.0 * s_k);
        let c = -s_k * dy;

        let xi = if a.abs() < 1e-8 {
            // Linear case
            (-c / b.max(1e-12)).clamp(0.0, 1.0)
        } else {
            let disc = b * b - 4.0 * a * c;
            let disc_safe = disc.max(0.0).sqrt();
            ((-b + disc_safe) / (2.0 * a)).clamp(0.0, 1.0)
        };

        Ok(x_w + xi * w_k)
    }
}

/// Coupling layer using rational-quadratic splines.
///
/// Applies the spline transform to the second half of the input,
/// conditioned on the first half through a network.
pub struct NfaNeuralSpline {
    /// Total dimension.
    pub dim: usize,
    /// Number of spline bins.
    pub num_bins: usize,
    split: usize,
    /// Network parameters: in=[split], out=[num_bins*3+1]*(dim-split)
    net_w1: Vec<f32>,
    net_b1: Vec<f32>,
    net_w2: Vec<f32>,
    net_b2: Vec<f32>,
    hidden: usize,
    param_per_dim: usize,
}

impl NfaNeuralSpline {
    /// Create a neural spline coupling layer.
    pub fn new(dim: usize, num_bins: usize, hidden: usize, seed: u64) -> Result<Self> {
        if dim < 2 {
            return Err(TensorError::invalid_argument_op(
                "NfaNeuralSpline::new",
                "dim must be >= 2",
            ));
        }
        if num_bins < 2 {
            return Err(TensorError::invalid_argument_op(
                "NfaNeuralSpline::new",
                "num_bins must be >= 2",
            ));
        }
        let split = dim / 2;
        let out_dim = dim - split;
        let param_per_dim = 3 * num_bins + 1;
        let total_out = out_dim * param_per_dim;
        Ok(Self {
            dim,
            num_bins,
            split,
            net_w1: xavier_uniform(split, hidden, seed),
            net_b1: zeros(hidden),
            net_w2: xavier_uniform(hidden, total_out, seed.wrapping_add(1)),
            net_b2: zeros(total_out),
            hidden,
            param_per_dim,
        })
    }

    fn compute_params(&self, x1: &[f32]) -> Result<Vec<Vec<f32>>> {
        let h1 = linear_relu_fwd(x1, &self.net_w1, &self.net_b1, self.hidden)?;
        let flat = linear_fwd(
            &h1,
            &self.net_w2,
            &self.net_b2,
            (self.dim - self.split) * self.param_per_dim,
        )?;
        let mut result = Vec::with_capacity(self.dim - self.split);
        for i in 0..(self.dim - self.split) {
            let start = i * self.param_per_dim;
            result.push(flat[start..start + self.param_per_dim].to_vec());
        }
        Ok(result)
    }

    /// Forward: (x1, x2) → (x1, spline(x2 | x1)), log_det.
    pub fn forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)> {
        if x.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "NfaNeuralSpline::forward",
                "input length mismatch",
            ));
        }
        let x1 = &x[..self.split];
        let x2 = &x[self.split..];
        let params = self.compute_params(x1)?;
        let mut z = x1.to_vec();
        let mut total_ld = 0.0_f32;
        for (i, &xi) in x2.iter().enumerate() {
            let spline = RqSpline::from_params(&params[i], self.num_bins, -5.0, 5.0)?;
            let (y, ld) = spline.forward_scalar(xi)?;
            z.push(y);
            total_ld += ld;
        }
        Ok((z, total_ld))
    }

    /// Inverse.
    pub fn inverse(&self, z: &[f32]) -> Result<Vec<f32>> {
        if z.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "NfaNeuralSpline::inverse",
                "input length mismatch",
            ));
        }
        let z1 = &z[..self.split];
        let z2 = &z[self.split..];
        let params = self.compute_params(z1)?;
        let mut x = z1.to_vec();
        for (i, &zi) in z2.iter().enumerate() {
            let spline = RqSpline::from_params(&params[i], self.num_bins, -5.0, 5.0)?;
            let xi = spline.inverse_scalar(zi)?;
            x.push(xi);
        }
        Ok(x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  NfaRadialFlow — Rezende & Mohamed 2015
// ─────────────────────────────────────────────────────────────────────────────

/// Radial flow: f(z) = z + β·h(α, r)·(z - z₀), r = ||z - z₀||.
///
/// h(α, r) = 1/(α + r).
/// Volume-changing: log|det J| = (d-1)·log|1 + β·h| + log|1 + β·h + β·h'·r|.
pub struct NfaRadialFlow {
    /// Dimension d.
    pub dim: usize,
    /// Reference point z₀.
    pub z0: Vec<f32>,
    /// log α (to ensure α > 0).
    pub log_alpha: f32,
    /// Raw β̂ (β̂ is unconstrained; actual β = -α + softplus(β̂) to ensure β > -α).
    pub beta_hat: f32,
}

impl NfaRadialFlow {
    /// Create with given parameters.
    pub fn new(dim: usize, z0: Vec<f32>, log_alpha: f32, beta_hat: f32) -> Result<Self> {
        if z0.len() != dim {
            return Err(TensorError::invalid_argument_op(
                "NfaRadialFlow::new",
                "z0 length must equal dim",
            ));
        }
        Ok(Self {
            dim,
            z0,
            log_alpha,
            beta_hat,
        })
    }

    /// Create with zero-initialized parameters.
    pub fn zeros_init(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let z0: Vec<f32> = (0..dim)
            .map(|_| sample_standard_normal(&mut rng) * 0.1)
            .collect();
        Self {
            dim,
            z0,
            log_alpha: 0.0,
            beta_hat: 0.0,
        }
    }

    fn alpha(&self) -> f32 {
        self.log_alpha.exp()
    }

    fn beta(&self) -> f32 {
        let alpha = self.alpha();
        -alpha + softplus(self.beta_hat)
    }

    /// Forward: z' = f(z), (z', log_det).
    pub fn forward(&self, z: &[f32]) -> Result<(Vec<f32>, f32)> {
        if z.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "NfaRadialFlow::forward",
                "input length mismatch",
            ));
        }
        let alpha = self.alpha();
        let beta = self.beta();
        let diff: Vec<f32> = z
            .iter()
            .zip(self.z0.iter())
            .map(|(&zi, &z0i)| zi - z0i)
            .collect();
        let r = diff.iter().map(|&d| d * d).sum::<f32>().sqrt().max(1e-8);
        let h = 1.0 / (alpha + r);
        let bh = beta * h;
        let scale = 1.0 + bh;

        let z_out: Vec<f32> = z
            .iter()
            .zip(diff.iter())
            .map(|(&zi, &di)| zi + bh * di)
            .collect();

        let d = self.dim as f32;
        let h_prime = -h * h;
        let bh_prime = beta * h_prime;
        // log|det J| = (d-1)*log|1+bh| + log|1+bh + bh'*r|
        let log_det =
            (d - 1.0) * scale.abs().max(1e-12).ln() + (scale + bh_prime * r).abs().max(1e-12).ln();

        Ok((z_out, log_det))
    }

    /// Inverse via fixed-point iteration.
    pub fn inverse(&self, z_out: &[f32]) -> Result<Vec<f32>> {
        if z_out.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "NfaRadialFlow::inverse",
                "input length mismatch",
            ));
        }
        // Fixed-point: z = z_out - β·h(r)·(z - z₀)
        // Iterate: z_{t+1} = z_out - β·h(||z_t - z₀||)·(z_t - z₀)
        let alpha = self.alpha();
        let beta = self.beta();
        let mut z = z_out.to_vec();
        for _ in 0..50 {
            let diff: Vec<f32> = z
                .iter()
                .zip(self.z0.iter())
                .map(|(&zi, &z0i)| zi - z0i)
                .collect();
            let r = diff.iter().map(|&d| d * d).sum::<f32>().sqrt().max(1e-8);
            let h = 1.0 / (alpha + r);
            let new_z: Vec<f32> = z_out
                .iter()
                .zip(diff.iter())
                .map(|(&zo, &di)| zo - beta * h * di)
                .collect();
            let diff_norm: f32 = new_z
                .iter()
                .zip(z.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            z = new_z;
            if diff_norm < 1e-6 {
                break;
            }
        }
        Ok(z)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  NfaHouseholderFlow — Tomczak 2016
// ─────────────────────────────────────────────────────────────────────────────

/// Householder reflection: H = I - 2·v·vᵀ / ||v||².
/// det(H) = -1, so log|det| = 0 (volume-preserving).
/// Stack K reflections for a rich orthogonal transform.
pub struct NfaHouseholderFlow {
    /// Dimension.
    pub dim: usize,
    /// Number of Householder reflections.
    pub num_reflections: usize,
    /// Reflection vectors v_k, shape [num_reflections × dim].
    pub vectors: Vec<Vec<f32>>,
}

impl NfaHouseholderFlow {
    /// Create with random reflection vectors.
    pub fn new(dim: usize, num_reflections: usize, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "NfaHouseholderFlow::new",
                "dim must be > 0",
            ));
        }
        if num_reflections == 0 {
            return Err(TensorError::invalid_argument_op(
                "NfaHouseholderFlow::new",
                "num_reflections must be > 0",
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let vectors: Vec<Vec<f32>> = (0..num_reflections)
            .map(|_| {
                let v: Vec<f32> = (0..dim).map(|_| sample_standard_normal(&mut rng)).collect();
                // Normalize
                let norm = v.iter().map(|&vi| vi * vi).sum::<f32>().sqrt().max(1e-8);
                v.into_iter().map(|vi| vi / norm).collect()
            })
            .collect();
        Ok(Self {
            dim,
            num_reflections,
            vectors,
        })
    }

    /// Apply one Householder reflection: H·x = x - 2·(vᵀx)·v / ||v||².
    fn reflect(x: &[f32], v: &[f32]) -> Vec<f32> {
        let norm_sq = v.iter().map(|&vi| vi * vi).sum::<f32>().max(1e-12);
        let dot: f32 = x.iter().zip(v.iter()).map(|(&xi, &vi)| xi * vi).sum();
        let scale = 2.0 * dot / norm_sq;
        x.iter()
            .zip(v.iter())
            .map(|(&xi, &vi)| xi - scale * vi)
            .collect()
    }

    /// Forward: apply reflections in order.  log_det = 0.
    pub fn forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)> {
        if x.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "NfaHouseholderFlow::forward",
                "input length mismatch",
            ));
        }
        let mut h = x.to_vec();
        for v in &self.vectors {
            h = Self::reflect(&h, v);
        }
        Ok((h, 0.0))
    }

    /// Inverse: apply reflections in reverse order (Hᵀ = H for Householder).
    pub fn inverse(&self, z: &[f32]) -> Result<Vec<f32>> {
        if z.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "NfaHouseholderFlow::inverse",
                "input length mismatch",
            ));
        }
        let mut h = z.to_vec();
        for v in self.vectors.iter().rev() {
            h = Self::reflect(&h, v);
        }
        Ok(h)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  NfaFlowVAE — VAE with Normalizing Flow posterior
// ─────────────────────────────────────────────────────────────────────────────

/// VAE with normalizing flow as the approximate posterior.
///
/// Encoder: q₀(z|x) = N(μ(x), σ²(x)).
/// Flow: z_K = f_K(... f_1(z_0)).
/// ELBO: E[log p(x|z_K)] - KL(q_K || p(z)).
///   = E[log p(x|z_K)] - E[log q_K(z_K)] + E[log p(z_K)]
///   = E[log p(x|z_K)] - E[log q_0(z_0)] - Σ E[log|det J_k|] + E[log p(z_K)]
pub struct NfaFlowVAE {
    /// Input dimension.
    pub input_dim: usize,
    /// Latent dimension.
    pub latent_dim: usize,
    /// Encoder: input → hidden.
    enc_w1: Vec<f32>,
    enc_b1: Vec<f32>,
    enc_wmu: Vec<f32>,
    enc_bmu: Vec<f32>,
    enc_wlv: Vec<f32>,
    enc_blv: Vec<f32>,
    enc_hidden: usize,
    /// Decoder: latent → hidden → input.
    dec_w1: Vec<f32>,
    dec_b1: Vec<f32>,
    dec_w2: Vec<f32>,
    dec_b2: Vec<f32>,
    dec_hidden: usize,
    /// Normalizing flow on latent space.
    pub flow: NfaHouseholderFlow,
}

impl NfaFlowVAE {
    /// Create a new FlowVAE.
    pub fn new(
        input_dim: usize,
        latent_dim: usize,
        enc_hidden: usize,
        dec_hidden: usize,
        num_flow_steps: usize,
        seed: u64,
    ) -> Result<Self> {
        if latent_dim < 2 {
            return Err(TensorError::invalid_argument_op(
                "NfaFlowVAE::new",
                "latent_dim must be >= 2",
            ));
        }
        Ok(Self {
            input_dim,
            latent_dim,
            enc_w1: xavier_uniform(input_dim, enc_hidden, seed),
            enc_b1: zeros(enc_hidden),
            enc_wmu: xavier_uniform(enc_hidden, latent_dim, seed.wrapping_add(1)),
            enc_bmu: zeros(latent_dim),
            enc_wlv: xavier_uniform(enc_hidden, latent_dim, seed.wrapping_add(2)),
            enc_blv: zeros(latent_dim),
            enc_hidden,
            dec_w1: xavier_uniform(latent_dim, dec_hidden, seed.wrapping_add(3)),
            dec_b1: zeros(dec_hidden),
            dec_w2: xavier_uniform(dec_hidden, input_dim, seed.wrapping_add(4)),
            dec_b2: zeros(input_dim),
            dec_hidden,
            flow: NfaHouseholderFlow::new(latent_dim, num_flow_steps, seed.wrapping_add(5))?,
        })
    }

    /// Encode: x → (mu, log_var).
    pub fn encode(&self, x: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let h = linear_relu_fwd(x, &self.enc_w1, &self.enc_b1, self.enc_hidden)?;
        let mu = linear_fwd(&h, &self.enc_wmu, &self.enc_bmu, self.latent_dim)?;
        let log_var = linear_fwd(&h, &self.enc_wlv, &self.enc_blv, self.latent_dim)?;
        Ok((mu, log_var))
    }

    /// Reparameterize: z₀ = μ + ε·exp(0.5·log_var).
    pub fn reparameterize(&self, mu: &[f32], log_var: &[f32], rng: &mut StdRng) -> Vec<f32> {
        mu.iter()
            .zip(log_var.iter())
            .map(|(&m, &lv)| m + sample_standard_normal(rng) * (0.5 * lv).exp())
            .collect()
    }

    /// Decode: z_K → x_recon.
    pub fn decode(&self, z: &[f32]) -> Result<Vec<f32>> {
        let h = linear_relu_fwd(z, &self.dec_w1, &self.dec_b1, self.dec_hidden)?;
        let out = linear_fwd(&h, &self.dec_w2, &self.dec_b2, self.input_dim)?;
        // Sigmoid for reconstruction in [0,1]
        Ok(out.iter().map(|&v| sigmoid(v)).collect())
    }

    /// Compute ELBO for a single sample.
    /// Returns (recon_loss, kl_loss, flow_log_det).
    pub fn elbo(&self, x: &[f32], rng: &mut StdRng) -> Result<(f32, f32, f32)> {
        let (mu, log_var) = self.encode(x)?;
        let z0 = self.reparameterize(&mu, &log_var, rng);
        // Apply flow
        let (z_k, flow_ld) = self.flow.forward(&z0)?;
        // Decode
        let x_recon = self.decode(&z_k)?;
        // Reconstruction loss: BCE
        let recon: f32 = x
            .iter()
            .zip(x_recon.iter())
            .map(|(&xi, &ri)| {
                let ri_safe = ri.clamp(1e-7, 1.0 - 1e-7);
                -(xi * ri_safe.ln() + (1.0 - xi) * (1.0 - ri_safe).ln())
            })
            .sum();
        // KL divergence: KL(q_0 || p(z)) = 0.5 * Σ(mu² + σ² - log σ² - 1)
        let kl: f32 = mu
            .iter()
            .zip(log_var.iter())
            .map(|(&m, &lv)| 0.5 * (m * m + lv.exp() - lv - 1.0))
            .sum();
        // Flow adjusts the KL (lower bound tightened)
        Ok((recon, kl - flow_ld, flow_ld))
    }

    /// Sample from the model.
    pub fn sample(&self, rng: &mut StdRng) -> Result<Vec<f32>> {
        // Sample z_K ~ p(z) = N(0,I)
        let z_prior: Vec<f32> = (0..self.latent_dim)
            .map(|_| sample_standard_normal(rng))
            .collect();
        // Apply flow inverse to get z_0 in posterior space, then decode
        let z_prior_inv = self.flow.inverse(&z_prior)?;
        self.decode(&z_prior_inv)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  NfaMetrics — flow evaluation metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluation metrics for normalizing flows.
pub struct NfaMetrics;

impl NfaMetrics {
    /// Bits per dimension: BPD = -NLL / (d · log 2).
    pub fn bits_per_dim(nll: f32, dim: usize) -> f32 {
        -nll / (dim as f32 * 2.0_f32.ln())
    }

    /// Negative log-likelihood over a test batch (mean).
    pub fn nll_batch<F>(forward_fn: F, test_set: &[Vec<f32>]) -> Result<f32>
    where
        F: Fn(&[f32]) -> Result<f32>,
    {
        if test_set.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "NfaMetrics::nll_batch",
                "empty test set",
            ));
        }
        let total: f32 = test_set
            .iter()
            .map(|x| forward_fn(x))
            .collect::<Result<Vec<f32>>>()?
            .iter()
            .sum();
        Ok(-total / test_set.len() as f32)
    }

    /// Approximate KL divergence KL(q || p) via Monte Carlo samples.
    /// KL = E_q[log q(z)] - E_q[log p(z)].
    /// We estimate this using log_det returned from the flow.
    pub fn approximate_kl(
        log_q_z0_samples: &[f32],
        flow_log_dets: &[f32],
        log_p_zk_samples: &[f32],
    ) -> Result<f32> {
        let n = log_q_z0_samples.len();
        if n == 0 || flow_log_dets.len() != n || log_p_zk_samples.len() != n {
            return Err(TensorError::invalid_argument_op(
                "NfaMetrics::approximate_kl",
                "length mismatch or empty input",
            ));
        }
        // KL = E[log q_0(z_0) - log|det J| - log p(z_K)]
        let kl: f32 = (0..n)
            .map(|i| log_q_z0_samples[i] - flow_log_dets[i] - log_p_zk_samples[i])
            .sum::<f32>()
            / n as f32;
        Ok(kl)
    }

    /// Evaluate a Glow model on a test set.
    pub fn evaluate_glow(model: &NfaGlowModel, test_set: &[Vec<f32>]) -> Result<NfaReport> {
        if test_set.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "NfaMetrics::evaluate_glow",
                "empty test set",
            ));
        }
        let dim = model.dim;
        let mut total_ll = 0.0_f32;
        for x in test_set {
            total_ll += model.log_likelihood(x)?;
        }
        let mean_nll = -total_ll / test_set.len() as f32;
        let bpd = Self::bits_per_dim(-mean_nll, dim);
        Ok(NfaReport {
            model_name: "NfaGlowModel".to_string(),
            num_samples: test_set.len(),
            dim,
            mean_nll,
            bpd,
        })
    }

    /// Evaluate a MAF model on a test set.
    pub fn evaluate_maf(
        model: &NfaMaskedAutoregressive,
        test_set: &[Vec<f32>],
    ) -> Result<NfaReport> {
        if test_set.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "NfaMetrics::evaluate_maf",
                "empty test set",
            ));
        }
        let dim = model.dim;
        let mut total_ll = 0.0_f32;
        for x in test_set {
            total_ll += model.log_likelihood(x)?;
        }
        let mean_nll = -total_ll / test_set.len() as f32;
        let bpd = Self::bits_per_dim(-mean_nll, dim);
        Ok(NfaReport {
            model_name: "NfaMaskedAutoregressive".to_string(),
            num_samples: test_set.len(),
            dim,
            mean_nll,
            bpd,
        })
    }
}

/// Summary report from flow evaluation.
#[derive(Debug, Clone)]
pub struct NfaReport {
    /// Model name.
    pub model_name: String,
    /// Number of test samples.
    pub num_samples: usize,
    /// Data dimensionality.
    pub dim: usize,
    /// Mean negative log-likelihood.
    pub mean_nll: f32,
    /// Bits per dimension.
    pub bpd: f32,
}

impl NfaReport {
    /// Returns true if the NLL is finite (basic sanity check).
    pub fn is_valid(&self) -> bool {
        self.mean_nll.is_finite() && self.bpd.is_finite()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  NfaFlowComposite — compose multiple flow steps
// ─────────────────────────────────────────────────────────────────────────────

/// Trait for any flow layer with forward/inverse/dim.
pub trait NfaFlowLayer: Send + Sync {
    /// Forward: x → (z, log_det).
    fn nfa_forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)>;
    /// Inverse: z → x.
    fn nfa_inverse(&self, z: &[f32]) -> Result<Vec<f32>>;
    /// Dimensionality.
    fn nfa_dim(&self) -> usize;
}

impl NfaFlowLayer for NfaActNorm {
    fn nfa_forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)> {
        self.forward(x)
    }
    fn nfa_inverse(&self, z: &[f32]) -> Result<Vec<f32>> {
        self.inverse(z)
    }
    fn nfa_dim(&self) -> usize {
        self.dim
    }
}

impl NfaFlowLayer for NfaInvertible1x1Conv {
    fn nfa_forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)> {
        self.forward(x)
    }
    fn nfa_inverse(&self, z: &[f32]) -> Result<Vec<f32>> {
        self.inverse(z)
    }
    fn nfa_dim(&self) -> usize {
        self.dim
    }
}

impl NfaFlowLayer for NfaAffineCoupling {
    fn nfa_forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)> {
        self.forward(x)
    }
    fn nfa_inverse(&self, z: &[f32]) -> Result<Vec<f32>> {
        self.inverse(z)
    }
    fn nfa_dim(&self) -> usize {
        self.dim
    }
}

impl NfaFlowLayer for NfaHouseholderFlow {
    fn nfa_forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)> {
        self.forward(x)
    }
    fn nfa_inverse(&self, z: &[f32]) -> Result<Vec<f32>> {
        self.inverse(z)
    }
    fn nfa_dim(&self) -> usize {
        self.dim
    }
}

impl NfaFlowLayer for NfaRadialFlow {
    fn nfa_forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)> {
        self.forward(x)
    }
    fn nfa_inverse(&self, z: &[f32]) -> Result<Vec<f32>> {
        self.inverse(z)
    }
    fn nfa_dim(&self) -> usize {
        self.dim
    }
}

impl NfaFlowLayer for NfaNeuralSpline {
    fn nfa_forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)> {
        self.forward(x)
    }
    fn nfa_inverse(&self, z: &[f32]) -> Result<Vec<f32>> {
        self.inverse(z)
    }
    fn nfa_dim(&self) -> usize {
        self.dim
    }
}

/// Composite flow: chain of heterogeneous flow layers.
pub struct NfaFlowComposite {
    /// Shared dimension.
    pub dim: usize,
    layers: Vec<Box<dyn NfaFlowLayer>>,
}

impl NfaFlowComposite {
    /// Create empty composite.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            layers: Vec::new(),
        }
    }

    /// Add a flow layer.
    pub fn add<L: NfaFlowLayer + 'static>(&mut self, layer: L) -> Result<()> {
        if layer.nfa_dim() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "NfaFlowComposite::add",
                "layer dim mismatch",
            ));
        }
        self.layers.push(Box::new(layer));
        Ok(())
    }

    /// Forward through all layers.
    pub fn forward(&self, x: &[f32]) -> Result<(Vec<f32>, f32)> {
        let mut h = x.to_vec();
        let mut total_ld = 0.0_f32;
        for layer in &self.layers {
            let (z, ld) = layer.nfa_forward(&h)?;
            h = z;
            total_ld += ld;
        }
        Ok((h, total_ld))
    }

    /// Inverse through all layers (reversed).
    pub fn inverse(&self, z: &[f32]) -> Result<Vec<f32>> {
        let mut h = z.to_vec();
        for layer in self.layers.iter().rev() {
            h = layer.nfa_inverse(&h)?;
        }
        Ok(h)
    }

    /// Log-likelihood under N(0,I) base.
    pub fn log_likelihood(&self, x: &[f32]) -> Result<f32> {
        let (z, ld) = self.forward(x)?;
        let log_pz: f32 = z.iter().map(|&zi| log_standard_normal(zi)).sum();
        Ok(log_pz + ld)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
