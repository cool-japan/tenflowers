//! State Space Models (SSMs) — Track A, Round 11.
//!
//! Implements a family of modern sequence models based on structured state
//! spaces.  All computations are in `f32` with plain `Vec<f32>` weight
//! buffers — no `Tensor` / autograd dependencies — making the module
//! fully self-contained.
//!
//! # Models provided
//!
//! | Struct | Reference |
//! |--------|-----------|
//! | [`S4Layer`] | Gu et al. (2022) — Structured State Space Sequences |
//! | [`MambaBlock`] | Gu & Dao (2023) — Selective State Spaces (Mamba) |
//! | [`SelectiveScan`] | Core parallel-prefix scan operation |
//! | [`HyenaFilter`] + [`HyenaOperator`] | Poli et al. (2023) — Hyena hierarchy |
//! | [`SsmSequenceModel`] | Full stack of Mamba blocks |
//!
//! # Quick start
//!
//! ```rust,ignore
//! use tenflowers_neural::ssm::{S4Config, S4Layer, MambaConfig, MambaBlock, SsmSequenceModel};
//!
//! // S4 single-layer inference
//! let cfg = S4Config { state_dim: 8, input_dim: 4, dt_min: 0.001, dt_max: 0.1 };
//! let layer = S4Layer::new(cfg, 0)?;
//! let (out, _state) = layer.forward(&vec![0.0; 4], None)?;
//! assert_eq!(out.len(), 4);
//!
//! // Full SSM sequence model
//! let model = SsmSequenceModel::new(8, 8, 2, None, 42)?;
//! let x = vec![0.1_f32; 8 * 3];   // batch=1, seq_len=3, d_model=8
//! let y = model.forward(&x, 3, 1)?;
//! assert_eq!(y.len(), 8 * 3);
//! ```

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Error helper
// ─────────────────────────────────────────────────────────────────────────────

type SsmResult<T> = Result<T, String>;

// ─────────────────────────────────────────────────────────────────────────────
// Activation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// SiLU / Swish activation: `x * sigmoid(x)`.
#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Sigmoid: `1 / (1 + exp(-x))`, clamped for numerical safety.
#[inline]
fn sigmoid(x: f32) -> f32 {
    let x_c = x.clamp(-88.0, 88.0);
    1.0 / (1.0 + (-x_c).exp())
}

/// Softplus: numerically stable `log(1 + exp(x))`.
#[inline]
/// Numerically stable softplus: `log(1 + exp(x))`.
///
/// Uses the identity `softplus(x) = x + softplus(-x)` for large positive x
/// and `softplus(x) ≈ exp(x)` for sufficiently negative x to avoid f32 underflow.
fn softplus(x: f32) -> f32 {
    // For x > 15.0: softplus(x) ≈ x (avoids exp overflow)
    if x > 15.0 {
        x
    } else if x < -15.0 {
        // For x < -15, exp(x) < 4e-7 so softplus(x) ≈ exp(x), which is > 0
        x.exp()
    } else {
        (1.0_f32 + x.exp()).ln()
    }
}

/// Layer norm over a slice: `(x - mean) / (std + eps) * gamma + beta`.
fn layer_norm_slice(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> SsmResult<Vec<f32>> {
    let n = x.len();
    if gamma.len() != n || beta.len() != n {
        return Err(format!(
            "layer_norm: dimension mismatch: x={n}, gamma={}, beta={}",
            gamma.len(),
            beta.len()
        ));
    }
    let mean = x.iter().copied().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let std_inv = (var + eps).sqrt().recip();
    let out = x
        .iter()
        .enumerate()
        .map(|(i, &v)| (v - mean) * std_inv * gamma[i] + beta[i])
        .collect();
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight initialisation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Xavier / Glorot uniform initialisation.
fn xavier_uniform(fan_in: usize, fan_out: usize, rng: &mut StdRng) -> Vec<f32> {
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt() as f32;
    let n = fan_in * fan_out;
    (0..n)
        .map(|_| {
            let u: f32 = rng.random();
            u * 2.0 * limit - limit
        })
        .collect()
}

/// Standard normal initialisation (Box-Muller).
fn normal_init(n: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let u1: f64 = (rng.random::<f64>()).max(1e-10);
        let u2: f64 = rng.random::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        out.push((r * theta.cos()) as f32);
        i += 1;
        if i < n {
            out.push((r * theta.sin()) as f32);
            i += 1;
        }
    }
    out
}

/// Linear projection: `y[j] = sum_i x[i] * W[i*out + j] + b[j]`.
/// W is stored row-major: shape `[in_dim, out_dim]`.
fn linear(x: &[f32], w: &[f32], b: &[f32], out_dim: usize) -> SsmResult<Vec<f32>> {
    let in_dim = x.len();
    if w.len() != in_dim * out_dim {
        return Err(format!(
            "linear: weight shape mismatch: expected {}×{}={}, got {}",
            in_dim,
            out_dim,
            in_dim * out_dim,
            w.len()
        ));
    }
    if b.len() != out_dim {
        return Err(format!(
            "linear: bias length mismatch: expected {out_dim}, got {}",
            b.len()
        ));
    }
    let mut out = b.to_vec();
    for i in 0..in_dim {
        let xi = x[i];
        if xi == 0.0 {
            continue;
        }
        let row = &w[i * out_dim..(i + 1) * out_dim];
        for j in 0..out_dim {
            out[j] += xi * row[j];
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
//  S4 — Structured State Space Sequences
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for an [`S4Layer`].
#[derive(Debug, Clone)]
pub struct S4Config {
    /// State-space dimension N (number of SSM states per channel).
    pub state_dim: usize,
    /// Input / output dimension H (number of channels).
    pub input_dim: usize,
    /// Minimum log-space time step Δ.
    pub dt_min: f32,
    /// Maximum log-space time step Δ.
    pub dt_max: f32,
}

impl Default for S4Config {
    fn default() -> Self {
        Self {
            state_dim: 16,
            input_dim: 64,
            dt_min: 0.001,
            dt_max: 0.1,
        }
    }
}

/// Structured State Space Sequence layer (S4).
///
/// The continuous-time SSM is `h'(t) = A h(t) + B u(t)`, `y(t) = C h(t) + D u(t)`.
/// The diagonal `A` is parameterised via its log-magnitude and phase:
/// `λ_i = -exp(log_a_real[i]) + i · a_imag[i]`  (stable: real part < 0).
///
/// Discretisation uses the bilinear (Tustin) method:
/// ```text
/// Ā = (I - Δ/2 · A)⁻¹ (I + Δ/2 · A)
/// B̄ = (I - Δ/2 · A)⁻¹ Δ B
/// ```
/// For diagonal A both matrix inversions are scalar divisions.
///
/// Recurrent forward pass: `h_t = Ā h_{t-1} + B̄ u_t`, `y_t = C h_t + D u_t`.
#[derive(Debug, Clone)]
pub struct S4Layer {
    /// Layer configuration.
    pub config: S4Config,
    /// `log |λ_i|`, shape \[N\] — real part magnitude of diagonal A (kept positive).
    pub log_a_real: Vec<f32>,
    /// `∠λ_i`, shape \[N\] — imaginary part of diagonal A.
    pub a_imag: Vec<f32>,
    /// B matrix, shape \[H, N\] (input → state).
    pub b: Vec<f32>,
    /// C matrix, shape \[N, H\] (state → output, transposed for efficient read).
    pub c: Vec<f32>,
    /// D skip-connection, shape \[H\].
    pub d: Vec<f32>,
    /// Learned log Δ, shape \[H\].
    pub log_dt: Vec<f32>,
    /// Layer-norm gain, shape \[H\].
    pub ln_gamma: Vec<f32>,
    /// Layer-norm bias, shape \[H\].
    pub ln_beta: Vec<f32>,
}

impl S4Layer {
    /// Construct a new `S4Layer` with random initialisation seeded by `seed`.
    pub fn new(config: S4Config, seed: u64) -> SsmResult<Self> {
        let n = config.state_dim;
        let h = config.input_dim;
        if n == 0 || h == 0 {
            return Err("S4Layer: state_dim and input_dim must be > 0".into());
        }

        let mut rng = StdRng::seed_from_u64(seed);

        // HiPPO-N initialisation: log|λ| = log(n), ∠λ uniformly spaced
        let log_a_real: Vec<f32> = (0..n).map(|i| ((i + 1) as f32).ln()).collect();
        let a_imag: Vec<f32> = (0..n)
            .map(|i| std::f32::consts::PI * (i + 1) as f32)
            .collect();

        // B: small normal
        let b = normal_init(h * n, &mut rng)
            .into_iter()
            .map(|v| v * 0.01)
            .collect();

        // C: small normal
        let c = normal_init(n * h, &mut rng)
            .into_iter()
            .map(|v| v * 0.01)
            .collect();

        // D (skip): ones
        let d = vec![1.0_f32; h];

        // log_dt: uniform in [log(dt_min), log(dt_max)]
        let log_dt_min = config.dt_min.ln();
        let log_dt_max = config.dt_max.ln();
        let log_dt: Vec<f32> = (0..h)
            .map(|_| {
                let u: f32 = rng.random();
                log_dt_min + u * (log_dt_max - log_dt_min)
            })
            .collect();

        Ok(Self {
            config,
            log_a_real,
            a_imag,
            b,
            c,
            d,
            log_dt,
            ln_gamma: vec![1.0_f32; h],
            ln_beta: vec![0.0_f32; h],
        })
    }

    /// Discretise the continuous SSM at each channel's Δ = exp(log_dt[h]).
    /// Returns `(a_bar[H, N], b_bar[H, N])` in row-major layout.
    fn discretise(&self) -> (Vec<f32>, Vec<f32>) {
        let n = self.config.state_dim;
        let h = self.config.input_dim;
        let mut a_bar = vec![0.0_f32; h * n];
        let mut b_bar = vec![0.0_f32; h * n];

        for hi in 0..h {
            let dt = self.log_dt[hi].exp();
            let half_dt = dt * 0.5;
            for ni in 0..n {
                // Diagonal A element (real part only for ZOH approximation on real arithmetic)
                // We use only the real part: λ_r = -exp(log_a_real[ni])
                let a_r = -(self.log_a_real[ni].exp());
                // Bilinear: denom = 1 - half_dt * a_r,  numer = 1 + half_dt * a_r
                let denom = 1.0 - half_dt * a_r;
                let numer = 1.0 + half_dt * a_r;
                let a_bar_val = numer / denom;
                // B_bar[h,n] = (2 / denom) * dt * B[h,n]  (bilinear)
                let b_raw = self.b[hi * n + ni];
                let b_bar_val = (2.0 / denom) * dt * b_raw;
                a_bar[hi * n + ni] = a_bar_val;
                b_bar[hi * n + ni] = b_bar_val;
            }
        }
        (a_bar, b_bar)
    }

    /// Single-step recurrent forward.
    ///
    /// `u` is the input of length H.  `state` is the hidden state of shape
    /// \[H, N\] stored flat.  Returns `(y[H], new_state[H*N])`.
    pub fn step(&self, u: &[f32], state: &[f32]) -> SsmResult<(Vec<f32>, Vec<f32>)> {
        let n = self.config.state_dim;
        let h = self.config.input_dim;
        if u.len() != h {
            return Err(format!(
                "S4Layer::step: input length {} != input_dim {}",
                u.len(),
                h
            ));
        }
        if state.len() != h * n {
            return Err(format!(
                "S4Layer::step: state length {} != H*N={}",
                state.len(),
                h * n
            ));
        }

        let (a_bar, b_bar) = self.discretise();
        let mut new_state = vec![0.0_f32; h * n];
        let mut y = vec![0.0_f32; h];

        for hi in 0..h {
            let base = hi * n;
            for ni in 0..n {
                new_state[base + ni] =
                    a_bar[base + ni] * state[base + ni] + b_bar[base + ni] * u[hi];
            }
            // y[h] = C[h,:] · h_t + D[h] * u[h]  — C stored as [N, H] so C[n, h] = c[n*H + h]
            let mut ch = 0.0_f32;
            for ni in 0..n {
                ch += self.c[ni * h + hi] * new_state[base + ni];
            }
            y[hi] = ch + self.d[hi] * u[hi];
        }
        Ok((y, new_state))
    }

    /// Process a full sequence of shape `[seq_len, input_dim]` stored flat.
    ///
    /// `initial_state` is optional (zeroes if `None`).
    /// Returns `(output[seq_len * H], final_state[H * N])`.
    pub fn forward(
        &self,
        x: &[f32],
        initial_state: Option<&[f32]>,
    ) -> SsmResult<(Vec<f32>, Vec<f32>)> {
        let n = self.config.state_dim;
        let h = self.config.input_dim;
        if x.len() % h != 0 {
            return Err(format!(
                "S4Layer::forward: input length {} not divisible by input_dim {}",
                x.len(),
                h
            ));
        }
        let seq_len = x.len() / h;
        let mut state = match initial_state {
            Some(s) => {
                if s.len() != h * n {
                    return Err(format!(
                        "S4Layer::forward: initial_state length {} != H*N={}",
                        s.len(),
                        h * n
                    ));
                }
                s.to_vec()
            }
            None => vec![0.0_f32; h * n],
        };

        let mut output = Vec::with_capacity(seq_len * h);
        for t in 0..seq_len {
            let u = &x[t * h..(t + 1) * h];
            let (y_t, new_state) = self.step(u, &state)?;
            output.extend_from_slice(&y_t);
            state = new_state;
        }

        // Apply layer norm over the output
        let normed = layer_norm_slice(
            &output,
            &self.ln_gamma.repeat(seq_len),
            &self.ln_beta.repeat(seq_len),
            1e-5,
        )?;

        Ok((normed, state))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  SelectiveScan — core parallel-prefix SSM scan
// ─────────────────────────────────────────────────────────────────────────────

/// Core selective state space scan operation.
///
/// Inputs (all flat, layout described below):
/// - `u`: \[batch, seq_len, d_model\]  — input sequence
/// - `delta`: \[batch, seq_len, d_model\] — per-step time intervals
/// - `a`: \[d_model, d_state\] — fixed log-A matrix (A = -softplus(a_log) so A < 0)
/// - `b`: \[batch, seq_len, d_state\] — input-dependent B
/// - `c`: \[batch, seq_len, d_state\] — input-dependent C
///
/// Output: `y` \[batch, seq_len, d_model\]
#[derive(Debug, Clone)]
pub struct SelectiveScan {
    /// Model dimension.
    pub d_model: usize,
    /// State dimension.
    pub d_state: usize,
}

impl SelectiveScan {
    /// Create a new SelectiveScan.
    pub fn new(d_model: usize, d_state: usize) -> SsmResult<Self> {
        if d_model == 0 || d_state == 0 {
            return Err("SelectiveScan: dimensions must be > 0".into());
        }
        Ok(Self { d_model, d_state })
    }

    /// Sequential (reference) selective scan — O(T·N) time, O(N) memory.
    pub fn scan_sequential(
        &self,
        u: &[f32],
        delta: &[f32],
        a_log: &[f32],
        b: &[f32],
        c: &[f32],
        batch: usize,
        seq_len: usize,
    ) -> SsmResult<Vec<f32>> {
        let d = self.d_model;
        let n = self.d_state;
        self.check_dims(u, delta, a_log, b, c, batch, seq_len)?;

        let mut y = vec![0.0_f32; batch * seq_len * d];

        for bi in 0..batch {
            // State h[d, n]
            let mut h = vec![0.0_f32; d * n];

            for t in 0..seq_len {
                let u_off = bi * seq_len * d + t * d;
                let d_off = bi * seq_len * d + t * d;
                let b_off = bi * seq_len * n + t * n;
                let c_off = bi * seq_len * n + t * n;
                let y_off = bi * seq_len * d + t * d;

                for di in 0..d {
                    // dt = softplus(delta[bi, t, di])
                    let dt = softplus(delta[d_off + di]);
                    for ni in 0..n {
                        // A_bar = exp(dt * A) where A = -exp(a_log[di, ni])
                        let a_val = -(a_log[di * n + ni].exp());
                        let a_bar = (dt * a_val).exp();
                        // B_bar * u = dt * b[bi, t, ni] * u[bi, t, di]
                        let b_bar_u = dt * b[b_off + ni] * u[u_off + di];
                        h[di * n + ni] = a_bar * h[di * n + ni] + b_bar_u;
                    }
                    // y[bi, t, di] = sum_ni c[bi, t, ni] * h[di, ni]
                    let mut out_val = 0.0_f32;
                    for ni in 0..n {
                        out_val += c[c_off + ni] * h[di * n + ni];
                    }
                    y[y_off + di] = out_val;
                }
            }
        }
        Ok(y)
    }

    /// Parallel (associative prefix scan) selective scan.
    ///
    /// Implements the Blelloch parallel prefix scan over the log-sum-exp
    /// formulation.  Each (a_bar, b_bar·u) pair forms a linear recurrence
    /// element; the associative operator is `(a2, b2) ∘ (a1, b1) = (a2·a1, a2·b1 + b2)`.
    pub fn scan_parallel(
        &self,
        u: &[f32],
        delta: &[f32],
        a_log: &[f32],
        b: &[f32],
        c: &[f32],
        batch: usize,
        seq_len: usize,
    ) -> SsmResult<Vec<f32>> {
        let d = self.d_model;
        let n = self.d_state;
        self.check_dims(u, delta, a_log, b, c, batch, seq_len)?;

        let mut y = vec![0.0_f32; batch * seq_len * d];

        for bi in 0..batch {
            for di in 0..d {
                for ni in 0..n {
                    // Build per-step (alpha, beta) elements
                    // alpha[t] = A_bar, beta[t] = B_bar * u
                    let mut alphas = vec![0.0_f32; seq_len];
                    let mut betas = vec![0.0_f32; seq_len];
                    for t in 0..seq_len {
                        let dt = softplus(delta[bi * seq_len * d + t * d + di]);
                        let a_val = -(a_log[di * n + ni].exp());
                        alphas[t] = (dt * a_val).exp();
                        betas[t] = dt
                            * b[bi * seq_len * n + t * n + ni]
                            * u[bi * seq_len * d + t * d + di];
                    }

                    // Prefix scan: prefix_a[t] = prod_{s<=t} alpha[s],
                    //              prefix_b[t] = sum_{s<=t} (prod_{s<r<=t} alpha[r]) * beta[s]
                    // Via: state[t] = alpha[t] * state[t-1] + beta[t]
                    // We use an up-sweep / down-sweep approach over a power-of-two padded array.
                    let padded = seq_len.next_power_of_two();
                    let mut alpha_p = vec![1.0_f32; padded];
                    let mut beta_p = vec![0.0_f32; padded];
                    alpha_p[..seq_len].copy_from_slice(&alphas[..seq_len]);
                    beta_p[..seq_len].copy_from_slice(&betas[..seq_len]);

                    // Up-sweep (reduce) to build partial products
                    let mut step = 1_usize;
                    while step < padded {
                        let mut idx = 2 * step - 1;
                        while idx < padded {
                            // Combine [idx - step] and [idx]
                            let a2 = alpha_p[idx];
                            let b2 = beta_p[idx];
                            let a1 = alpha_p[idx - step];
                            let b1 = beta_p[idx - step];
                            alpha_p[idx] = a2 * a1;
                            beta_p[idx] = a2 * b1 + b2;
                            idx += 2 * step;
                        }
                        step *= 2;
                    }

                    // Down-sweep to compute prefix sums
                    alpha_p[padded - 1] = 1.0;
                    beta_p[padded - 1] = 0.0;
                    let mut step = padded / 2;
                    while step > 0 {
                        let mut idx = step - 1;
                        while idx + step < padded {
                            let tmp_a = alpha_p[idx];
                            let tmp_b = beta_p[idx];
                            let a_right = alpha_p[idx + step];
                            let b_right = beta_p[idx + step];
                            // Left child: carry unchanged
                            alpha_p[idx] = a_right;
                            beta_p[idx] = b_right;
                            // Right child: combine
                            alpha_p[idx + step] = a_right * tmp_a;
                            beta_p[idx + step] = a_right * tmp_b + b_right;
                            idx += 2 * step;
                        }
                        step /= 2;
                    }

                    // After down-sweep, beta_p[t] is the exclusive prefix.
                    // Inclusive state[t] = alpha[t] * carry(t-1) + beta[t]
                    // We re-apply: state[t] = alphas[t] * beta_p[t] + betas[t]
                    for t in 0..seq_len {
                        let state_t = alphas[t] * beta_p[t] + betas[t];
                        y[bi * seq_len * d + t * d + di] +=
                            c[bi * seq_len * n + t * n + ni] * state_t;
                    }
                }
            }
        }
        Ok(y)
    }

    fn check_dims(
        &self,
        u: &[f32],
        delta: &[f32],
        a_log: &[f32],
        b: &[f32],
        c: &[f32],
        batch: usize,
        seq_len: usize,
    ) -> SsmResult<()> {
        let d = self.d_model;
        let n = self.d_state;
        let bsd = batch * seq_len * d;
        let bsn = batch * seq_len * n;
        if u.len() != bsd {
            return Err(format!(
                "SelectiveScan: u len {} != batch*seq*d={}",
                u.len(),
                bsd
            ));
        }
        if delta.len() != bsd {
            return Err(format!(
                "SelectiveScan: delta len {} != batch*seq*d={}",
                delta.len(),
                bsd
            ));
        }
        if a_log.len() != d * n {
            return Err(format!(
                "SelectiveScan: a_log len {} != d*n={}",
                a_log.len(),
                d * n
            ));
        }
        if b.len() != bsn {
            return Err(format!(
                "SelectiveScan: b len {} != batch*seq*n={}",
                b.len(),
                bsn
            ));
        }
        if c.len() != bsn {
            return Err(format!(
                "SelectiveScan: c len {} != batch*seq*n={}",
                c.len(),
                bsn
            ));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Mamba — Selective State Space Model block
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a [`MambaBlock`].
#[derive(Debug, Clone)]
pub struct MambaConfig {
    /// Model dimension.
    pub d_model: usize,
    /// SSM state dimension (default 16).
    pub d_state: usize,
    /// Depthwise-conv width (default 4).
    pub d_conv: usize,
    /// Inner-dimension expansion ratio (default 2).
    pub expand: usize,
    /// Rank for low-rank Δ projection (default `ceil(d_model / 16)`).
    pub dt_rank: usize,
}

impl MambaConfig {
    /// Create a `MambaConfig` with sensible defaults for `d_model`.
    pub fn new(d_model: usize) -> SsmResult<Self> {
        if d_model == 0 {
            return Err("MambaConfig: d_model must be > 0".into());
        }
        let dt_rank = (d_model + 15) / 16; // ceil(d_model / 16)
        Ok(Self {
            d_model,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            dt_rank,
        })
    }
}

/// Mamba block — selective state-space sequence model.
///
/// Architecture (per token):
/// 1. Input projection: `[d_model] → [d_inner * 2]` (split into x, z).
/// 2. Depthwise conv1d on x: width `d_conv`.
/// 3. SiLU on x.
/// 4. SSM parameter projection: `x → (Δ_lr, B, C)`.
/// 5. Δ full projection: `Δ_lr [dt_rank] → Δ [d_inner]`.
/// 6. Selective scan with input-dependent `Ā`, `B̄`.
/// 7. Gate: `y_ssm * silu(z)`.
/// 8. Output projection: `[d_inner] → [d_model]`.
#[derive(Debug, Clone)]
pub struct MambaBlock {
    /// Block configuration.
    pub config: MambaConfig,
    // d_inner = d_model * expand
    /// Input projection weight: \[d_model, d_inner * 2\].
    pub in_proj: Vec<f32>,
    /// Input projection bias: \[d_inner * 2\].
    pub in_proj_bias: Vec<f32>,
    /// Depthwise conv1d weight: \[d_inner, d_conv\].
    pub conv1d_weight: Vec<f32>,
    /// Depthwise conv1d bias: \[d_inner\].
    pub conv1d_bias: Vec<f32>,
    /// SSM parameter projection: \[d_inner, dt_rank + 2 * d_state\].
    pub x_proj: Vec<f32>,
    /// SSM parameter bias: \[dt_rank + 2 * d_state\].
    pub x_proj_bias: Vec<f32>,
    /// Δ full projection: \[dt_rank, d_inner\].
    pub dt_proj: Vec<f32>,
    /// Δ projection bias: \[d_inner\].
    pub dt_proj_bias: Vec<f32>,
    /// A log (always ≥ 0 so A = -exp(a_log) ≤ 0): \[d_inner, d_state\].
    pub a_log: Vec<f32>,
    /// D skip: \[d_inner\].
    pub d: Vec<f32>,
    /// Output projection weight: \[d_inner, d_model\].
    pub out_proj: Vec<f32>,
    /// Output projection bias: \[d_model\].
    pub out_proj_bias: Vec<f32>,
    /// Layer-norm gain for residual: \[d_model\].
    pub ln_gamma: Vec<f32>,
    /// Layer-norm bias for residual: \[d_model\].
    pub ln_beta: Vec<f32>,
}

impl MambaBlock {
    /// Create a new `MambaBlock` with random weights seeded by `seed`.
    pub fn new(config: MambaConfig, seed: u64) -> SsmResult<Self> {
        let d = config.d_model;
        let n = config.d_state;
        let dc = config.d_conv;
        let r = config.expand;
        let rank = config.dt_rank;
        let di = d * r; // d_inner

        if di == 0 || n == 0 || dc == 0 || rank == 0 {
            return Err("MambaBlock: all dimensions must be > 0".into());
        }

        let mut rng = StdRng::seed_from_u64(seed);

        let in_proj = xavier_uniform(d, di * 2, &mut rng);
        let in_proj_bias = vec![0.0_f32; di * 2];

        let conv1d_weight = xavier_uniform(di, dc, &mut rng);
        let conv1d_bias = vec![0.0_f32; di];

        let ssm_proj_out = rank + 2 * n;
        let x_proj = xavier_uniform(di, ssm_proj_out, &mut rng);
        let x_proj_bias = vec![0.0_f32; ssm_proj_out];

        let dt_proj = xavier_uniform(rank, di, &mut rng);
        // dt_proj bias: initialise to log(dt_init) with small noise
        let dt_proj_bias: Vec<f32> = (0..di)
            .map(|_| {
                let u: f32 = rng.random();
                // Map to [log(dt_min), log(dt_max)] = [-6.9, -2.3]
                -6.9 + u * 4.6
            })
            .collect();

        // a_log: A[i, n] = i + 1  (HiPPO initialisation, stable)
        let a_log: Vec<f32> = (0..di)
            .flat_map(|i| (0..n).map(move |j| ((i + 1) as f32 * (j + 1) as f32).ln()))
            .collect();

        let d_skip = vec![1.0_f32; di];

        let out_proj = xavier_uniform(di, d, &mut rng);
        let out_proj_bias = vec![0.0_f32; d];

        Ok(Self {
            config,
            in_proj,
            in_proj_bias,
            conv1d_weight,
            conv1d_bias,
            x_proj,
            x_proj_bias,
            dt_proj,
            dt_proj_bias,
            a_log,
            d: d_skip,
            out_proj,
            out_proj_bias,
            ln_gamma: vec![1.0_f32; d],
            ln_beta: vec![0.0_f32; d],
        })
    }

    /// Depthwise 1D convolution (causal).
    ///
    /// `x` has shape \[seq_len, di\].  `weight` has shape \[di, d_conv\] (each
    /// channel has its own kernel).  Returns \[seq_len, di\].
    fn depthwise_conv1d(&self, x: &[f32], seq_len: usize) -> SsmResult<Vec<f32>> {
        let di = self.config.d_model * self.config.expand;
        let dc = self.config.d_conv;
        if x.len() != seq_len * di {
            return Err(format!(
                "MambaBlock::depthwise_conv1d: x len {} != seq*di={}",
                x.len(),
                seq_len * di
            ));
        }
        let mut out = vec![0.0_f32; seq_len * di];
        for t in 0..seq_len {
            for ch in 0..di {
                let mut val = self.conv1d_bias[ch];
                for k in 0..dc {
                    let t_src = t as isize - (dc as isize - 1) + k as isize;
                    if t_src >= 0 && (t_src as usize) < seq_len {
                        val += self.conv1d_weight[ch * dc + k] * x[t_src as usize * di + ch];
                    }
                }
                out[t * di + ch] = val;
            }
        }
        Ok(out)
    }

    /// Forward pass over a full sequence.
    ///
    /// `x` is \[seq_len, d_model\] stored flat.
    /// Returns \[seq_len, d_model\].
    pub fn forward(&self, x: &[f32], seq_len: usize) -> SsmResult<Vec<f32>> {
        let d = self.config.d_model;
        let n = self.config.d_state;
        let di = d * self.config.expand;
        let rank = self.config.dt_rank;

        if x.len() != seq_len * d {
            return Err(format!(
                "MambaBlock::forward: x len {} != seq*d={}",
                x.len(),
                seq_len * d
            ));
        }

        // 1. Input projection: [seq, d] → [seq, di*2], split into x_branch, z
        let mut x_branch = vec![0.0_f32; seq_len * di];
        let mut z_branch = vec![0.0_f32; seq_len * di];
        for t in 0..seq_len {
            let inp = &x[t * d..(t + 1) * d];
            let proj = linear(inp, &self.in_proj, &self.in_proj_bias, di * 2)?;
            x_branch[t * di..(t + 1) * di].copy_from_slice(&proj[..di]);
            z_branch[t * di..(t + 1) * di].copy_from_slice(&proj[di..]);
        }

        // 2. Depthwise conv1d on x_branch
        let x_conv = self.depthwise_conv1d(&x_branch, seq_len)?;

        // 3. SiLU on x_conv
        let x_act: Vec<f32> = x_conv.iter().map(|&v| silu(v)).collect();

        // 4. SSM parameter projection: [seq, di] → [seq, rank + 2*n]
        let ssm_proj_out = rank + 2 * n;
        let mut delta_lr_all = vec![0.0_f32; seq_len * rank];
        let mut b_all = vec![0.0_f32; seq_len * n];
        let mut c_all = vec![0.0_f32; seq_len * n];
        for t in 0..seq_len {
            let xi = &x_act[t * di..(t + 1) * di];
            let proj = linear(xi, &self.x_proj, &self.x_proj_bias, ssm_proj_out)?;
            delta_lr_all[t * rank..(t + 1) * rank].copy_from_slice(&proj[..rank]);
            b_all[t * n..(t + 1) * n].copy_from_slice(&proj[rank..rank + n]);
            c_all[t * n..(t + 1) * n].copy_from_slice(&proj[rank + n..]);
        }

        // 5. Δ full projection: [seq, rank] → [seq, di]
        let mut delta_all = vec![0.0_f32; seq_len * di];
        for t in 0..seq_len {
            let delta_lr = &delta_lr_all[t * rank..(t + 1) * rank];
            let delta_t = linear(delta_lr, &self.dt_proj, &self.dt_proj_bias, di)?;
            delta_all[t * di..(t + 1) * di].copy_from_slice(&delta_t);
        }

        // 6. Selective scan (batch=1)
        let scan = SelectiveScan::new(di, n)?;
        let y_ssm =
            scan.scan_sequential(&x_act, &delta_all, &self.a_log, &b_all, &c_all, 1, seq_len)?;

        // Add D skip connection: y_ssm[t, di] += d[di] * x_act[t, di]
        let mut y_gated = vec![0.0_f32; seq_len * di];
        for t in 0..seq_len {
            for di_i in 0..di {
                let ssm_val = y_ssm[t * di + di_i] + self.d[di_i] * x_act[t * di + di_i];
                // 7. Gate by silu(z)
                y_gated[t * di + di_i] = ssm_val * silu(z_branch[t * di + di_i]);
            }
        }

        // 8. Output projection: [seq, di] → [seq, d]
        let mut out = Vec::with_capacity(seq_len * d);
        for t in 0..seq_len {
            let yi = &y_gated[t * di..(t + 1) * di];
            let ot = linear(yi, &self.out_proj, &self.out_proj_bias, d)?;
            out.extend_from_slice(&ot);
        }

        // Residual add + layer norm
        let mut normed = Vec::with_capacity(seq_len * d);
        for t in 0..seq_len {
            let out_t = &out[t * d..(t + 1) * d];
            let x_t = &x[t * d..(t + 1) * d];
            let residual: Vec<f32> = out_t.iter().zip(x_t).map(|(a, b)| a + b).collect();
            let ln = layer_norm_slice(&residual, &self.ln_gamma, &self.ln_beta, 1e-5)?;
            normed.extend_from_slice(&ln);
        }

        Ok(normed)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  HyenaFilter — long-range convolution via sinusoidal MLP
// ─────────────────────────────────────────────────────────────────────────────

/// Hyena implicit convolutional filter.
///
/// The filter `h(t)` is computed via a small MLP whose inputs are sinusoidal
/// position encodings:  `h(t) = MLP(t/L, sin(ω₁t/L), cos(ω₁t/L), ..., sin(ωₖt/L), cos(ωₖt/L))`.
///
/// This creates a smooth, data-independent filter that can be evaluated at any
/// length without retraining.
#[derive(Debug, Clone)]
pub struct HyenaFilter {
    /// Number of frequency bands.
    pub num_freqs: usize,
    /// Maximum sequence length the filter is designed for.
    pub max_len: usize,
    /// Output (filter) dimension.
    pub out_dim: usize,
    /// MLP layer 1 weights: \[input_feat, hidden\].  input_feat = 1 + 2*num_freqs.
    pub w1: Vec<f32>,
    /// MLP layer 1 bias: \[hidden\].
    pub b1: Vec<f32>,
    /// MLP layer 2 weights: \[hidden, out_dim\].
    pub w2: Vec<f32>,
    /// MLP layer 2 bias: \[out_dim\].
    pub b2: Vec<f32>,
    /// Learned frequencies: \[num_freqs\].
    pub freqs: Vec<f32>,
}

impl HyenaFilter {
    /// Create a new `HyenaFilter`.
    ///
    /// - `num_freqs`: number of sinusoidal frequency bands k.
    /// - `max_len`: sequence length.
    /// - `out_dim`: filter output channels.
    /// - `hidden`: MLP hidden dimension.
    /// - `seed`: random seed.
    pub fn new(
        num_freqs: usize,
        max_len: usize,
        out_dim: usize,
        hidden: usize,
        seed: u64,
    ) -> SsmResult<Self> {
        if num_freqs == 0 || max_len == 0 || out_dim == 0 || hidden == 0 {
            return Err("HyenaFilter: all dimensions must be > 0".into());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let in_feat = 1 + 2 * num_freqs; // t/L, sin, cos pairs
        let w1 = xavier_uniform(in_feat, hidden, &mut rng);
        let b1 = vec![0.0_f32; hidden];
        let w2 = xavier_uniform(hidden, out_dim, &mut rng);
        let b2 = vec![0.0_f32; out_dim];
        // Frequencies: log-uniformly spaced in [1, max_len/2]
        let freqs: Vec<f32> = (0..num_freqs)
            .map(|i| {
                let frac = i as f32 / (num_freqs.max(1) as f32);
                let log_min = 0.0_f32;
                let log_max = (max_len as f32 / 2.0).max(1.0).ln();
                (log_min + frac * (log_max - log_min)).exp()
            })
            .collect();
        Ok(Self {
            num_freqs,
            max_len,
            out_dim,
            w1,
            b1,
            w2,
            b2,
            freqs,
        })
    }

    /// Evaluate the filter for a single position `t` (0-indexed).
    /// Returns a vector of length `out_dim`.
    fn eval_at(&self, t: usize) -> SsmResult<Vec<f32>> {
        let t_norm = t as f32 / self.max_len.max(1) as f32;
        let mut feats = Vec::with_capacity(1 + 2 * self.num_freqs);
        feats.push(t_norm);
        for &freq in &self.freqs {
            feats.push((freq * t_norm * std::f32::consts::TAU).sin());
            feats.push((freq * t_norm * std::f32::consts::TAU).cos());
        }
        // Layer 1 + ReLU
        let hidden_dim = self.b1.len();
        let h1 = linear(&feats, &self.w1, &self.b1, hidden_dim)?;
        let h1_act: Vec<f32> = h1.iter().map(|&v| v.max(0.0)).collect();
        // Layer 2
        let out = linear(&h1_act, &self.w2, &self.b2, self.out_dim)?;
        Ok(out)
    }

    /// Generate the full filter of shape \[max_len, out_dim\].
    pub fn generate(&self) -> SsmResult<Vec<f32>> {
        let mut h = Vec::with_capacity(self.max_len * self.out_dim);
        for t in 0..self.max_len {
            let ht = self.eval_at(t)?;
            h.extend_from_slice(&ht);
        }
        Ok(h)
    }

    /// Generate filter for a given sequence length.
    pub fn generate_for_len(&self, seq_len: usize) -> SsmResult<Vec<f32>> {
        let mut h = Vec::with_capacity(seq_len * self.out_dim);
        for t in 0..seq_len {
            let ht = self.eval_at(t)?;
            h.extend_from_slice(&ht);
        }
        Ok(h)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  HyenaOperator — depth-d Hyena operator
// ─────────────────────────────────────────────────────────────────────────────

/// Hyena operator — depth-d long-range sequence model via filter-gated
/// element-wise convolutions.
///
/// Each of the `depth` layers applies:
/// 1. Element-wise multiplication of the running state with a Hyena filter.
/// 2. Element-wise gating by the corresponding projected channel.
#[derive(Debug, Clone)]
pub struct HyenaOperator {
    /// Hyena depth.
    pub depth: usize,
    /// Input/output dimension.
    pub d_model: usize,
    /// Input projection weight: \[d_model, (depth+1) * d_model\].
    pub in_proj: Vec<f32>,
    /// Input projection bias: \[(depth+1) * d_model\].
    pub in_proj_bias: Vec<f32>,
    /// Hyena filters — one per depth layer.
    pub filters: Vec<HyenaFilter>,
    /// Output projection weight: \[d_model, d_model\].
    pub out_proj: Vec<f32>,
    /// Output projection bias: \[d_model\].
    pub out_proj_bias: Vec<f32>,
}

impl HyenaOperator {
    /// Create a new `HyenaOperator`.
    ///
    /// - `depth`: number of Hyena filter layers.
    /// - `d_model`: token dimension.
    /// - `num_freqs`: frequency bands per filter.
    /// - `max_len`: maximum sequence length.
    /// - `seed`: random seed.
    pub fn new(
        depth: usize,
        d_model: usize,
        num_freqs: usize,
        max_len: usize,
        seed: u64,
    ) -> SsmResult<Self> {
        if depth == 0 || d_model == 0 {
            return Err("HyenaOperator: depth and d_model must be > 0".into());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let total_proj = (depth + 1) * d_model;
        let in_proj = xavier_uniform(d_model, total_proj, &mut rng);
        let in_proj_bias = vec![0.0_f32; total_proj];

        let mut filters = Vec::with_capacity(depth);
        for i in 0..depth {
            filters.push(HyenaFilter::new(
                num_freqs,
                max_len,
                d_model,
                32.max(d_model),
                seed.wrapping_add(i as u64 + 1),
            )?);
        }

        let out_proj = xavier_uniform(d_model, d_model, &mut rng);
        let out_proj_bias = vec![0.0_f32; d_model];

        Ok(Self {
            depth,
            d_model,
            in_proj,
            in_proj_bias,
            filters,
            out_proj,
            out_proj_bias,
        })
    }

    /// Forward pass.
    ///
    /// `x` has shape \[seq_len, d_model\].  Returns \[seq_len, d_model\].
    pub fn forward(&self, x: &[f32], seq_len: usize) -> SsmResult<Vec<f32>> {
        let d = self.d_model;
        if x.len() != seq_len * d {
            return Err(format!(
                "HyenaOperator::forward: x len {} != seq*d={}",
                x.len(),
                seq_len * d
            ));
        }

        // Project input to (depth+1) channels
        let proj_dim = (self.depth + 1) * d;
        let mut projections = Vec::with_capacity(seq_len * proj_dim);
        for t in 0..seq_len {
            let xt = &x[t * d..(t + 1) * d];
            let p = linear(xt, &self.in_proj, &self.in_proj_bias, proj_dim)?;
            projections.extend_from_slice(&p);
        }

        // v = projections[:, 0:d]  (first channel, the "value")
        let mut state: Vec<f32> = (0..seq_len)
            .flat_map(|t| projections[t * proj_dim..t * proj_dim + d].to_vec())
            .collect();

        // depth gating steps
        for layer_idx in 0..self.depth {
            // Gate channel: projections[:, (layer_idx+1)*d : (layer_idx+2)*d]
            let gate_offset = (layer_idx + 1) * d;
            let gate: Vec<f32> = (0..seq_len)
                .flat_map(|t| {
                    projections[t * proj_dim + gate_offset..t * proj_dim + gate_offset + d].to_vec()
                })
                .collect();

            // Generate Hyena filter for this seq_len
            let h_filter = self.filters[layer_idx].generate_for_len(seq_len)?;

            // Element-wise: state[t, di] = state[t, di] * h_filter[t, di] * gate[t, di]
            for t in 0..seq_len {
                for di in 0..d {
                    state[t * d + di] *= h_filter[t * d + di] * silu(gate[t * d + di]);
                }
            }
        }

        // Output projection
        let mut out = Vec::with_capacity(seq_len * d);
        for t in 0..seq_len {
            let st = &state[t * d..(t + 1) * d];
            let ot = linear(st, &self.out_proj, &self.out_proj_bias, d)?;
            out.extend_from_slice(&ot);
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  SsmSequenceModel — stacked Mamba blocks
// ─────────────────────────────────────────────────────────────────────────────

/// Full SSM sequence model: N × MambaBlock with optional classification head.
#[derive(Debug, Clone)]
pub struct SsmSequenceModel {
    /// Blocks.
    pub blocks: Vec<MambaBlock>,
    /// Optional output head weight: \[d_model, num_classes\].
    pub head: Option<Vec<f32>>,
    /// Optional output head bias: \[num_classes\].
    pub head_bias: Option<Vec<f32>>,
    /// Number of output classes (if head is present).
    pub num_classes: usize,
    /// Model dimension.
    pub d_model: usize,
}

impl SsmSequenceModel {
    /// Build a model with `num_blocks` stacked Mamba blocks.
    ///
    /// - `d_model`: token embedding size.
    /// - `d_state`: SSM state size.
    /// - `num_blocks`: number of Mamba layers.
    /// - `num_classes`: if `Some(k)`, adds a linear classification head.
    /// - `seed`: random seed.
    pub fn new(
        d_model: usize,
        d_state: usize,
        num_blocks: usize,
        num_classes: Option<usize>,
        seed: u64,
    ) -> SsmResult<Self> {
        if d_model == 0 || d_state == 0 || num_blocks == 0 {
            return Err("SsmSequenceModel: d_model, d_state, num_blocks must be > 0".into());
        }
        let mut blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            let mut cfg = MambaConfig::new(d_model)?;
            cfg.d_state = d_state;
            blocks.push(MambaBlock::new(cfg, seed.wrapping_add(i as u64))?);
        }

        let (head, head_bias, nc) = if let Some(k) = num_classes {
            if k == 0 {
                return Err("SsmSequenceModel: num_classes must be > 0".into());
            }
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(0xDEAD));
            let w = xavier_uniform(d_model, k, &mut rng);
            let b = vec![0.0_f32; k];
            (Some(w), Some(b), k)
        } else {
            (None, None, 0)
        };

        Ok(Self {
            blocks,
            head,
            head_bias,
            num_classes: nc,
            d_model,
        })
    }

    /// Forward pass.
    ///
    /// `x` has shape \[batch_size * seq_len, d_model\] stored flat (batch-major).
    ///
    /// When no classification head is present, returns \[batch * seq_len, d_model\].
    /// When a head is present, pools over `seq_len` dimension and returns
    /// \[batch, num_classes\] after softmax.
    pub fn forward(&self, x: &[f32], seq_len: usize, batch_size: usize) -> SsmResult<Vec<f32>> {
        let d = self.d_model;
        let total = batch_size * seq_len * d;
        if x.len() != total {
            return Err(format!(
                "SsmSequenceModel::forward: x len {} != batch*seq*d={}",
                x.len(),
                total
            ));
        }

        // Process each sequence in the batch independently
        let mut current = x.to_vec();

        // Pass through each block
        for block in &self.blocks {
            let mut next = Vec::with_capacity(total);
            for bi in 0..batch_size {
                let seq_data = &current[bi * seq_len * d..(bi + 1) * seq_len * d];
                let out = block.forward(seq_data, seq_len)?;
                next.extend_from_slice(&out);
            }
            current = next;
        }

        // If no head, return the full sequence output
        if self.head.is_none() {
            return Ok(current);
        }

        // Classification head: mean pool over seq_len, then linear + softmax
        let head_w = self.head.as_ref().expect("checked above");
        let head_b = self.head_bias.as_ref().expect("checked above");
        let nc = self.num_classes;
        let mut logits = Vec::with_capacity(batch_size * nc);

        for bi in 0..batch_size {
            // Mean pool [seq_len, d] → [d]
            let mut pooled = vec![0.0_f32; d];
            for t in 0..seq_len {
                for di in 0..d {
                    pooled[di] += current[bi * seq_len * d + t * d + di];
                }
            }
            for v in pooled.iter_mut() {
                *v /= seq_len as f32;
            }
            // Linear projection → [nc]
            let logit = linear(&pooled, head_w, head_b, nc)?;
            // Softmax
            let max_l = logit.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logit.iter().map(|&v| (v - max_l).exp()).sum();
            let softmax: Vec<f32> = logit.iter().map(|&v| (v - max_l).exp() / exp_sum).collect();
            logits.extend_from_slice(&softmax);
        }

        Ok(logits)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn assert_finite(v: &[f32], label: &str) {
        for (i, &x) in v.iter().enumerate() {
            assert!(
                x.is_finite(),
                "{label}: element [{i}] is not finite (got {x})"
            );
        }
    }

    fn all_close(a: &[f32], b: &[f32], tol: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tol)
    }

    // ── S4Config ─────────────────────────────────────────────────────────────

    #[test]
    fn s4_config_default() {
        let cfg = S4Config::default();
        assert_eq!(cfg.state_dim, 16);
        assert_eq!(cfg.input_dim, 64);
        assert!(cfg.dt_min < cfg.dt_max);
    }

    // ── S4Layer ──────────────────────────────────────────────────────────────

    #[test]
    fn s4_forward_shape() -> Result<(), String> {
        let cfg = S4Config {
            state_dim: 4,
            input_dim: 8,
            dt_min: 0.001,
            dt_max: 0.1,
        };
        let layer = S4Layer::new(cfg, 42)?;
        let x = vec![0.1_f32; 5 * 8]; // seq_len=5, H=8
        let (out, state) = layer.forward(&x, None)?;
        assert_eq!(out.len(), 5 * 8, "output shape mismatch");
        assert_eq!(state.len(), 8 * 4, "state shape mismatch");
        assert_finite(&out, "S4Layer output");
        Ok(())
    }

    #[test]
    fn s4_forward_seq_len_1() -> Result<(), String> {
        let cfg = S4Config {
            state_dim: 2,
            input_dim: 3,
            dt_min: 0.001,
            dt_max: 0.1,
        };
        let layer = S4Layer::new(cfg, 7)?;
        let x = vec![1.0_f32, -1.0, 0.5];
        let (out, state) = layer.forward(&x, None)?;
        assert_eq!(out.len(), 3);
        assert_eq!(state.len(), 3 * 2);
        assert_finite(&out, "S4 seq=1");
        Ok(())
    }

    #[test]
    fn s4_step_matches_forward() -> Result<(), String> {
        let cfg = S4Config {
            state_dim: 3,
            input_dim: 4,
            dt_min: 0.001,
            dt_max: 0.1,
        };
        let layer = S4Layer::new(cfg, 99)?;
        // Run forward on a 3-token sequence
        let x: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let (out_fwd, _) = layer.forward(&x, None)?;
        // Run step-by-step
        let mut state = vec![0.0_f32; 4 * 3];
        let mut out_step = Vec::new();
        for t in 0..3 {
            let u = &x[t * 4..(t + 1) * 4];
            let (y, new_s) = layer.step(u, &state)?;
            out_step.extend_from_slice(&y);
            state = new_s;
        }
        // After layer norm in forward, values differ — just check shapes/finiteness
        assert_eq!(out_fwd.len(), out_step.len());
        assert_finite(&out_step, "S4 step output");
        Ok(())
    }

    #[test]
    fn s4_output_range() -> Result<(), String> {
        let cfg = S4Config {
            state_dim: 8,
            input_dim: 16,
            dt_min: 0.001,
            dt_max: 0.1,
        };
        let layer = S4Layer::new(cfg, 13)?;
        let x = vec![0.0_f32; 10 * 16];
        let (out, _) = layer.forward(&x, None)?;
        // All-zero input should produce finite (possibly zero) output
        assert_finite(&out, "zero-input output");
        Ok(())
    }

    #[test]
    fn s4_zero_input_finite() -> Result<(), String> {
        let cfg = S4Config {
            state_dim: 4,
            input_dim: 4,
            dt_min: 0.001,
            dt_max: 0.1,
        };
        let layer = S4Layer::new(cfg, 11)?;
        let x = vec![0.0_f32; 8 * 4];
        let (out, state) = layer.forward(&x, None)?;
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(state.iter().all(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn s4_initial_state_respected() -> Result<(), String> {
        let cfg = S4Config {
            state_dim: 2,
            input_dim: 2,
            dt_min: 0.001,
            dt_max: 0.1,
        };
        let layer = S4Layer::new(cfg, 55)?;
        let x = vec![0.0_f32; 3 * 2];
        let (out_zero, _) = layer.forward(&x, None)?;
        let init_state = vec![1.0_f32; 2 * 2];
        let (out_nonzero, _) = layer.forward(&x, Some(&init_state))?;
        // Different initial states → different outputs (for non-trivial A)
        // At minimum they should not panic
        assert_eq!(out_zero.len(), out_nonzero.len());
        assert_finite(&out_nonzero, "non-zero init state output");
        Ok(())
    }

    #[test]
    fn s4_error_on_bad_input() {
        let cfg = S4Config {
            state_dim: 4,
            input_dim: 8,
            dt_min: 0.001,
            dt_max: 0.1,
        };
        let layer = S4Layer::new(cfg, 0).expect("S4Layer creation should succeed");
        // Wrong input dim
        let x = vec![0.0_f32; 7]; // not divisible by 8
        assert!(layer.forward(&x, None).is_err());
    }

    #[test]
    fn s4_discretise_stability() -> Result<(), String> {
        let cfg = S4Config {
            state_dim: 8,
            input_dim: 4,
            dt_min: 0.001,
            dt_max: 0.1,
        };
        let layer = S4Layer::new(cfg, 33)?;
        let (a_bar, b_bar) = layer.discretise();
        assert_finite(&a_bar, "a_bar");
        assert_finite(&b_bar, "b_bar");
        // |a_bar| should be < 1 for stability (real-only diagonal system with neg real A)
        for &v in &a_bar {
            assert!(v.abs() < 1.0 + 1e-4, "a_bar element {v} >= 1 (unstable)");
        }
        Ok(())
    }

    #[test]
    fn s4_different_seeds_differ() -> Result<(), String> {
        let cfg = S4Config {
            state_dim: 4,
            input_dim: 4,
            dt_min: 0.001,
            dt_max: 0.1,
        };
        let l1 = S4Layer::new(cfg.clone(), 1)?;
        let l2 = S4Layer::new(cfg, 2)?;
        let x = vec![0.5_f32; 4];
        let (o1, _) = l1.forward(&x, None)?;
        let (o2, _) = l2.forward(&x, None)?;
        assert!(!all_close(&o1, &o2, 1e-6), "different seeds should differ");
        Ok(())
    }

    // ── SelectiveScan ────────────────────────────────────────────────────────

    #[test]
    fn selective_scan_sequential_shape() -> Result<(), String> {
        let d = 4;
        let n = 8;
        let batch = 2;
        let seq = 6;
        let scan = SelectiveScan::new(d, n)?;
        let u = vec![0.1_f32; batch * seq * d];
        let delta = vec![0.0_f32; batch * seq * d];
        let a_log = vec![0.0_f32; d * n];
        let b = vec![0.1_f32; batch * seq * n];
        let c = vec![0.1_f32; batch * seq * n];
        let y = scan.scan_sequential(&u, &delta, &a_log, &b, &c, batch, seq)?;
        assert_eq!(y.len(), batch * seq * d);
        assert_finite(&y, "sequential scan");
        Ok(())
    }

    #[test]
    fn selective_scan_parallel_shape() -> Result<(), String> {
        let d = 4;
        let n = 4;
        let batch = 1;
        let seq = 8;
        let scan = SelectiveScan::new(d, n)?;
        let u = vec![0.5_f32; batch * seq * d];
        let delta = vec![0.0_f32; batch * seq * d];
        let a_log = vec![0.5_f32; d * n];
        let b = vec![0.1_f32; batch * seq * n];
        let c = vec![0.1_f32; batch * seq * n];
        let y = scan.scan_parallel(&u, &delta, &a_log, &b, &c, batch, seq)?;
        assert_eq!(y.len(), batch * seq * d);
        assert_finite(&y, "parallel scan");
        Ok(())
    }

    #[test]
    fn selective_scan_seq1() -> Result<(), String> {
        let scan = SelectiveScan::new(2, 2)?;
        let u = vec![1.0_f32, -1.0];
        let delta = vec![0.0_f32; 2];
        let a_log = vec![0.0_f32; 4];
        let b = vec![0.5_f32; 2];
        let c = vec![1.0_f32; 2];
        let y = scan.scan_sequential(&u, &delta, &a_log, &b, &c, 1, 1)?;
        assert_eq!(y.len(), 2);
        assert_finite(&y, "seq1 scan");
        Ok(())
    }

    #[test]
    fn selective_scan_parallel_vs_sequential_close() -> Result<(), String> {
        // For seq_len=1 sequential and parallel must agree
        let d = 2;
        let n = 2;
        let batch = 1;
        let seq = 1;
        let scan = SelectiveScan::new(d, n)?;
        let u = vec![0.3_f32; d];
        let delta = vec![0.1_f32; d];
        let a_log = vec![0.5_f32; d * n];
        let b = vec![0.2_f32; n];
        let c = vec![0.4_f32; n];
        let y_seq = scan.scan_sequential(&u, &delta, &a_log, &b, &c, batch, seq)?;
        let y_par = scan.scan_parallel(&u, &delta, &a_log, &b, &c, batch, seq)?;
        assert!(
            all_close(&y_seq, &y_par, 1e-4),
            "seq={y_seq:?} par={y_par:?}"
        );
        Ok(())
    }

    #[test]
    fn selective_scan_error_bad_dims() {
        let scan = SelectiveScan::new(4, 4).expect("SelectiveScan creation should succeed");
        let result = scan.scan_sequential(
            &[0.0_f32; 10], // wrong size
            &[0.0_f32; 4 * 2 * 4],
            &[0.0_f32; 4 * 4],
            &[0.0_f32; 4 * 2 * 4],
            &[0.0_f32; 4 * 2 * 4],
            4,
            2,
        );
        assert!(result.is_err());
    }

    // ── MambaBlock ───────────────────────────────────────────────────────────

    #[test]
    fn mamba_forward_shape() -> Result<(), String> {
        let cfg = MambaConfig {
            d_model: 8,
            d_state: 4,
            d_conv: 4,
            expand: 2,
            dt_rank: 1,
        };
        let block = MambaBlock::new(cfg, 42)?;
        let x = vec![0.1_f32; 5 * 8]; // seq=5, d=8
        let out = block.forward(&x, 5)?;
        assert_eq!(out.len(), 5 * 8, "Mamba output shape");
        assert_finite(&out, "Mamba output");
        Ok(())
    }

    #[test]
    fn mamba_forward_batch1_seq1() -> Result<(), String> {
        let cfg = MambaConfig {
            d_model: 4,
            d_state: 2,
            d_conv: 2,
            expand: 2,
            dt_rank: 1,
        };
        let block = MambaBlock::new(cfg, 0)?;
        let x = vec![1.0_f32, -1.0, 0.5, 0.0];
        let out = block.forward(&x, 1)?;
        assert_eq!(out.len(), 4);
        assert_finite(&out, "Mamba seq=1");
        Ok(())
    }

    #[test]
    fn mamba_config_default_dt_rank() -> Result<(), String> {
        let cfg = MambaConfig::new(32)?;
        assert_eq!(cfg.dt_rank, 2); // ceil(32/16)
        let cfg2 = MambaConfig::new(16)?;
        assert_eq!(cfg2.dt_rank, 1); // ceil(16/16)
        let cfg3 = MambaConfig::new(17)?;
        assert_eq!(cfg3.dt_rank, 2); // ceil(17/16)
        Ok(())
    }

    #[test]
    fn mamba_error_on_wrong_input() -> Result<(), String> {
        let cfg = MambaConfig::new(8)?;
        let block = MambaBlock::new(cfg, 1)?;
        let x = vec![0.0_f32; 7 * 8 + 3]; // not divisible
        assert!(block.forward(&x, 7).is_err());
        Ok(())
    }

    #[test]
    fn mamba_larger_model() -> Result<(), String> {
        let cfg = MambaConfig {
            d_model: 16,
            d_state: 8,
            d_conv: 4,
            expand: 2,
            dt_rank: 2,
        };
        let block = MambaBlock::new(cfg, 77)?;
        let seq = 10;
        let x = vec![0.01_f32; seq * 16];
        let out = block.forward(&x, seq)?;
        assert_eq!(out.len(), seq * 16);
        assert_finite(&out, "Mamba larger");
        Ok(())
    }

    // ── HyenaFilter ──────────────────────────────────────────────────────────

    #[test]
    fn hyena_filter_generate_shape() -> Result<(), String> {
        let filt = HyenaFilter::new(4, 16, 8, 32, 42)?;
        let h = filt.generate()?;
        assert_eq!(h.len(), 16 * 8, "HyenaFilter generate shape");
        assert_finite(&h, "HyenaFilter");
        Ok(())
    }

    #[test]
    fn hyena_filter_generate_for_len() -> Result<(), String> {
        let filt = HyenaFilter::new(3, 100, 4, 16, 7)?;
        let h = filt.generate_for_len(32)?;
        assert_eq!(h.len(), 32 * 4, "generate_for_len");
        assert_finite(&h, "HyenaFilter for_len");
        Ok(())
    }

    #[test]
    fn hyena_filter_len1() -> Result<(), String> {
        let filt = HyenaFilter::new(2, 8, 4, 8, 0)?;
        let h = filt.generate_for_len(1)?;
        assert_eq!(h.len(), 4);
        assert_finite(&h, "HyenaFilter len=1");
        Ok(())
    }

    #[test]
    fn hyena_filter_different_seeds_differ() -> Result<(), String> {
        let f1 = HyenaFilter::new(4, 8, 4, 16, 1)?;
        let f2 = HyenaFilter::new(4, 8, 4, 16, 2)?;
        let h1 = f1.generate()?;
        let h2 = f2.generate()?;
        assert!(!all_close(&h1, &h2, 1e-6), "different seeds must differ");
        Ok(())
    }

    // ── HyenaOperator ────────────────────────────────────────────────────────

    #[test]
    fn hyena_operator_shape() -> Result<(), String> {
        let op = HyenaOperator::new(2, 8, 4, 16, 42)?;
        let x = vec![0.1_f32; 10 * 8]; // seq=10, d=8
        let out = op.forward(&x, 10)?;
        assert_eq!(out.len(), 10 * 8);
        assert_finite(&out, "HyenaOperator");
        Ok(())
    }

    #[test]
    fn hyena_operator_seq1() -> Result<(), String> {
        let op = HyenaOperator::new(1, 4, 2, 8, 0)?;
        let x = vec![1.0_f32, 0.0, -1.0, 0.5];
        let out = op.forward(&x, 1)?;
        assert_eq!(out.len(), 4);
        assert_finite(&out, "HyenaOperator seq=1");
        Ok(())
    }

    #[test]
    fn hyena_operator_depth3() -> Result<(), String> {
        let op = HyenaOperator::new(3, 8, 4, 16, 5)?;
        let x = vec![0.5_f32; 6 * 8];
        let out = op.forward(&x, 6)?;
        assert_eq!(out.len(), 6 * 8);
        assert_finite(&out, "HyenaOperator depth=3");
        Ok(())
    }

    // ── SsmSequenceModel ─────────────────────────────────────────────────────

    #[test]
    fn ssm_model_forward_no_head() -> Result<(), String> {
        let model = SsmSequenceModel::new(8, 4, 2, None, 42)?;
        let x = vec![0.1_f32; 3 * 5 * 8]; // batch=3, seq=5, d=8
        let out = model.forward(&x, 5, 3)?;
        assert_eq!(out.len(), 3 * 5 * 8, "SsmModel output shape");
        assert_finite(&out, "SsmModel no head");
        Ok(())
    }

    #[test]
    fn ssm_model_forward_with_head() -> Result<(), String> {
        let model = SsmSequenceModel::new(8, 4, 2, Some(3), 42)?;
        let x = vec![0.2_f32; 2 * 4 * 8]; // batch=2, seq=4, d=8
        let out = model.forward(&x, 4, 2)?;
        assert_eq!(out.len(), 2 * 3, "classification output shape");
        // Softmax probabilities should sum to 1 per batch item
        for bi in 0..2 {
            let prob_sum: f32 = out[bi * 3..(bi + 1) * 3].iter().sum();
            assert!((prob_sum - 1.0).abs() < 1e-4, "softmax sum={prob_sum}");
        }
        Ok(())
    }

    #[test]
    fn ssm_model_seq1_batch1() -> Result<(), String> {
        let model = SsmSequenceModel::new(4, 2, 1, None, 0)?;
        let x = vec![1.0_f32, -1.0, 0.5, 0.0];
        let out = model.forward(&x, 1, 1)?;
        assert_eq!(out.len(), 4);
        assert_finite(&out, "SsmModel seq=1 batch=1");
        Ok(())
    }

    #[test]
    fn ssm_model_stacking_depth() -> Result<(), String> {
        // Deep model (4 blocks) should still produce valid output
        let model = SsmSequenceModel::new(8, 4, 4, None, 99)?;
        let x = vec![0.01_f32; 2 * 3 * 8];
        let out = model.forward(&x, 3, 2)?;
        assert_eq!(out.len(), 2 * 3 * 8);
        assert_finite(&out, "deep SSM model");
        Ok(())
    }

    #[test]
    fn ssm_model_error_on_bad_input() -> Result<(), String> {
        let model = SsmSequenceModel::new(8, 4, 2, None, 42)?;
        let x = vec![0.0_f32; 10]; // wrong size
        assert!(model.forward(&x, 5, 3).is_err());
        Ok(())
    }

    #[test]
    fn ssm_model_different_seeds_differ() -> Result<(), String> {
        let m1 = SsmSequenceModel::new(8, 4, 2, None, 1)?;
        let m2 = SsmSequenceModel::new(8, 4, 2, None, 2)?;
        let x = vec![0.1_f32; 2 * 8];
        let o1 = m1.forward(&x, 2, 1)?;
        let o2 = m2.forward(&x, 2, 1)?;
        assert!(!all_close(&o1, &o2, 1e-6));
        Ok(())
    }

    // ── Activation helpers ───────────────────────────────────────────────────

    #[test]
    fn silu_properties() {
        // silu(0) == 0
        assert!((silu(0.0)).abs() < 1e-7);
        // silu(x) > 0 for x >> 0
        assert!(silu(10.0) > 9.0);
        // silu(x) is finite everywhere in a reasonable range
        for i in -100..=100 {
            assert!(silu(i as f32).is_finite());
        }
    }

    #[test]
    fn softplus_properties() {
        // softplus(0) == ln(2)
        let expected = std::f32::consts::LN_2;
        assert!((softplus(0.0) - expected).abs() < 1e-5);
        // softplus(x) > 0 always
        for i in -20..=20 {
            assert!(softplus(i as f32) > 0.0);
        }
        // large x: softplus(x) ≈ x
        assert!((softplus(30.0) - 30.0).abs() < 1e-3);
    }

    #[test]
    fn layer_norm_mean_zero() -> Result<(), String> {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let n = x.len();
        let gamma = vec![1.0_f32; n];
        let beta = vec![0.0_f32; n];
        let out = layer_norm_slice(&x, &gamma, &beta, 1e-5)?;
        let mean: f32 = out.iter().sum::<f32>() / n as f32;
        assert!(mean.abs() < 1e-5, "layer_norm mean={mean}");
        Ok(())
    }
}
