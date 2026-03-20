//! Advanced Structured State Space Models (S4, Mamba, Griffin/Hawk, CT-RNN, hybrids).

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

type Res<T> = Result<T, String>;

// ── Activation helpers ────────────────────────────────────────────────────────

#[inline]
fn sigmoid64(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-500.0, 500.0)).exp())
}
#[inline]
fn silu64(x: f64) -> f64 {
    x * sigmoid64(x)
}
#[inline]
fn tanh64(x: f64) -> f64 {
    x.tanh()
}
#[inline]
fn relu64(x: f64) -> f64 {
    x.max(0.0)
}
#[inline]
fn softplus64(x: f64) -> f64 {
    if x > 30.0 {
        x
    } else if x < -30.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

// ── Weight init ───────────────────────────────────────────────────────────────

fn xavier_init64(fan_in: usize, fan_out: usize, rng: &mut StdRng) -> Vec<f64> {
    let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
    (0..fan_in * fan_out)
        .map(|_| {
            let u: f64 = rng.random();
            u * 2.0 * limit - limit
        })
        .collect()
}

fn normal_init64(n: usize, rng: &mut StdRng) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    let mut i = 0usize;
    while i < n {
        let u1: f64 = (rng.random::<f64>()).max(1e-12);
        let u2: f64 = rng.random::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        out.push(r * theta.cos());
        i += 1;
        if i < n {
            out.push(r * theta.sin());
            i += 1;
        }
    }
    out
}

fn linear64(x: &[f64], w: &[f64], b: &[f64], out_dim: usize) -> Res<Vec<f64>> {
    let in_dim = x.len();
    if w.len() != in_dim * out_dim {
        return Err(format!("linear64: w {} != {}×{}", w.len(), in_dim, out_dim));
    }
    if b.len() != out_dim {
        return Err(format!("linear64: b {} != {}", b.len(), out_dim));
    }
    let mut out = b.to_vec();
    for i in 0..in_dim {
        let xi = x[i];
        for j in 0..out_dim {
            out[j] += xi * w[i * out_dim + j];
        }
    }
    Ok(out)
}

fn layer_norm64(x: &[f64], eps: f64) -> Vec<f64> {
    let n = x.len();
    let mean = x.iter().copied().sum::<f64>() / n as f64;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let inv_std = 1.0 / (var + eps).sqrt();
    x.iter().map(|v| (v - mean) * inv_std).collect()
}

// ── 1. S4 — Structured State Space Sequence Model ─────────────────────────────

/// HiPPO-LegS matrix: `A_nk = -sqrt((2n+1)(2k+1))` for n>k, `-(n+1)` on diagonal. `B_n = sqrt(2n+1)`.
#[derive(Debug, Clone)]
pub struct HiPPoMatrix;
impl HiPPoMatrix {
    pub fn compute(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut a = vec![vec![0.0f64; n]; n];
        let mut b = vec![0.0f64; n];
        for row in 0..n {
            let nr = row as f64;
            a[row][row] = -(nr + 1.0);
            for col in 0..row {
                a[row][col] = -((2.0 * nr + 1.0) * (2.0 * col as f64 + 1.0)).sqrt();
            }
            b[row] = (2.0 * nr + 1.0_f64).sqrt();
        }
        (a, b)
    }
}

/// ZOH and bilinear discretisation of a diagonal SSM.
#[derive(Debug, Clone)]
pub struct S4Discretization;
impl S4Discretization {
    /// ZOH: `Ā_i = exp(A_i·dt)`, `B̄_i = (Ā_i - 1)/A_i · B_i`.
    pub fn discretize(a_diag: &[f64], b: &[f64], dt: f64) -> Res<(Vec<f64>, Vec<f64>)> {
        if a_diag.len() != b.len() {
            return Err("S4Discretization: len mismatch".into());
        }
        let (mut a_bar, mut b_bar) = (Vec::new(), Vec::new());
        for i in 0..a_diag.len() {
            let ea = (a_diag[i] * dt).exp();
            a_bar.push(ea);
            b_bar.push(if a_diag[i].abs() < 1e-12 {
                dt * b[i]
            } else {
                (ea - 1.0) / a_diag[i] * b[i]
            });
        }
        Ok((a_bar, b_bar))
    }
    /// Bilinear (Tustin): `Ā_i = (1 + dt/2·A_i)/(1 - dt/2·A_i)`.
    pub fn discretize_bilinear(a_diag: &[f64], b: &[f64], dt: f64) -> Res<(Vec<f64>, Vec<f64>)> {
        if a_diag.len() != b.len() {
            return Err("S4Discretization::bilinear: len mismatch".into());
        }
        let half = dt * 0.5;
        let (mut a_bar, mut b_bar) = (Vec::new(), Vec::new());
        for i in 0..a_diag.len() {
            let denom = 1.0 - half * a_diag[i];
            if denom.abs() < 1e-12 {
                return Err(format!("bilinear: singular denom at {i}"));
            }
            a_bar.push((1.0 + half * a_diag[i]) / denom);
            b_bar.push(dt / denom * b[i]);
        }
        Ok((a_bar, b_bar))
    }
}

/// Convolution kernel: `K[t] = sum_i C[i] · A[i]^t · B[i]`.
#[derive(Debug, Clone)]
pub struct S4Kernel;
impl S4Kernel {
    pub fn compute_kernel(a_diag: &[f64], b: &[f64], c: &[f64], seq_len: usize) -> Res<Vec<f64>> {
        let n = a_diag.len();
        if b.len() != n || c.len() != n {
            return Err("S4Kernel: dim mismatch".into());
        }
        let mut kernel = vec![0.0f64; seq_len];
        for t in 0..seq_len {
            kernel[t] = (0..n).map(|i| c[i] * a_diag[i].powi(t as i32) * b[i]).sum();
        }
        Ok(kernel)
    }
}

/// S4 layer: causal convolution + skip connection.
#[derive(Debug, Clone)]
pub struct S4Layer {
    pub state_dim: usize,
    pub input_dim: usize,
    pub log_a: Vec<f64>,
    pub b_proj: Vec<f64>,
    pub c_proj: Vec<f64>,
    pub d_skip: Vec<f64>,
    pub log_dt: Vec<f64>,
}
impl S4Layer {
    pub fn new(state_dim: usize, input_dim: usize, seed: u64) -> Res<Self> {
        if state_dim == 0 || input_dim == 0 {
            return Err("S4Layer: dims > 0".into());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let log_a = (0..state_dim).map(|i| -((i + 1) as f64).ln()).collect();
        let b_proj = xavier_init64(input_dim, state_dim, &mut rng);
        let c_proj = xavier_init64(state_dim, input_dim, &mut rng);
        let d_skip = vec![1.0f64; input_dim];
        let log_dt = (0..input_dim)
            .map(|_| {
                let u: f64 = rng.random();
                -4.0 + u * 3.0
            })
            .collect();
        Ok(Self {
            state_dim,
            input_dim,
            log_a,
            b_proj,
            c_proj,
            d_skip,
            log_dt,
        })
    }
    /// Causal convolution: `y[t] = sum_{s<=t} kernel[t-s]*u[s] + D*u[t]`.
    pub fn forward(&self, u: &[f64], kernel: &[f64]) -> Res<Vec<f64>> {
        if kernel.is_empty() {
            return Err("S4Layer: empty kernel".into());
        }
        let mut y = vec![0.0f64; u.len()];
        for t in 0..u.len() {
            let mut acc = 0.0;
            for s in 0..=t {
                acc += kernel.get(t - s).copied().unwrap_or(0.0) * u[s];
            }
            y[t] = acc + self.d_skip[0] * u[t];
        }
        Ok(y)
    }
    /// Full sequence forward [seq_len × input_dim] (flat).
    pub fn forward_sequence(&self, x: &[f64]) -> Res<Vec<f64>> {
        let h = self.input_dim;
        let n = self.state_dim;
        if x.len() % h != 0 {
            return Err(format!("S4Layer: {} % {h} != 0", x.len()));
        }
        let seq_len = x.len() / h;
        let mut out = vec![0.0f64; seq_len * h];
        for ch in 0..h {
            let dt = self.log_dt[ch].exp();
            let a_diag: Vec<f64> = self.log_a.clone();
            let b_ch: Vec<f64> = (0..n).map(|i| self.b_proj[ch * n + i]).collect();
            let c_ch: Vec<f64> = (0..n).map(|i| self.c_proj[i * h + ch]).collect();
            let (a_bar, b_bar) = S4Discretization::discretize(&a_diag, &b_ch, dt)?;
            let kernel = S4Kernel::compute_kernel(&a_bar, &b_bar, &c_ch, seq_len)?;
            let u_ch: Vec<f64> = (0..seq_len).map(|t| x[t * h + ch]).collect();
            let y_ch = self.forward(&u_ch, &kernel)?;
            for t in 0..seq_len {
                out[t * h + ch] = y_ch[t];
            }
        }
        Ok(out)
    }
}

/// Stack of S4Layers with input/output projections.
#[derive(Debug, Clone)]
pub struct S4Model {
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub state_dim: usize,
    pub in_proj: Vec<f64>,
    pub in_bias: Vec<f64>,
    pub out_proj: Vec<f64>,
    pub out_bias: Vec<f64>,
    pub layers: Vec<S4Layer>,
}
impl S4Model {
    pub fn new(
        num_layers: usize,
        hidden_dim: usize,
        state_dim: usize,
        in_dim: usize,
        out_dim: usize,
        seed: u64,
    ) -> Res<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        let in_proj = xavier_init64(in_dim, hidden_dim, &mut rng);
        let out_proj = xavier_init64(hidden_dim, out_dim, &mut rng);
        let mut layers = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            layers.push(S4Layer::new(state_dim, hidden_dim, seed + l as u64 + 1)?);
        }
        Ok(Self {
            num_layers,
            hidden_dim,
            state_dim,
            in_proj,
            in_bias: vec![0.0; hidden_dim],
            out_proj,
            out_bias: vec![0.0; out_dim],
            layers,
        })
    }
    pub fn forward(&self, x: &[Vec<f64>]) -> Res<Vec<Vec<f64>>> {
        let seq_len = x.len();
        if seq_len == 0 {
            return Ok(vec![]);
        }
        let h = self.hidden_dim;
        let out_dim = self.out_bias.len();
        let mut hidden: Vec<f64> = Vec::with_capacity(seq_len * h);
        for t in 0..seq_len {
            hidden.extend(linear64(&x[t], &self.in_proj, &self.in_bias, h)?);
        }
        for layer in &self.layers {
            hidden = layer.forward_sequence(&hidden)?;
        }
        let mut out = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let normed = layer_norm64(&hidden[t * h..(t + 1) * h], 1e-5);
            out.push(linear64(&normed, &self.out_proj, &self.out_bias, out_dim)?);
        }
        Ok(out)
    }
}

// ── 2. Mamba Architecture ─────────────────────────────────────────────────────

/// Parallel prefix scan: `y[t] = a[t]*y[t-1] + b[t]`.
#[derive(Debug, Clone)]
pub struct ParallelScan;
impl ParallelScan {
    pub fn scan(a: &[f64], b: &[f64]) -> Res<Vec<f64>> {
        if a.len() != b.len() {
            return Err("ParallelScan: len mismatch".into());
        }
        let mut y = vec![0.0f64; a.len()];
        let mut acc = 0.0f64;
        for i in 0..a.len() {
            acc = a[i] * acc + b[i];
            y[i] = acc;
        }
        Ok(y)
    }
}

/// Causal SSM step: `h_new = A*h + B*x`, `y = C·h_new`.
#[derive(Debug, Clone)]
pub struct SelectiveScanCausal {
    pub state_dim: usize,
}
impl SelectiveScanCausal {
    pub fn new(state_dim: usize) -> Res<Self> {
        if state_dim == 0 {
            return Err("SelectiveScanCausal: state_dim > 0".into());
        }
        Ok(Self { state_dim })
    }
    pub fn step(&self, x: f64, h: &[f64], a: &[f64], b: &[f64], c: &[f64]) -> Res<(f64, Vec<f64>)> {
        let n = self.state_dim;
        if h.len() != n || a.len() != n || b.len() != n || c.len() != n {
            return Err(format!("SelectiveScanCausal::step: dim mismatch n={n}"));
        }
        let new_h: Vec<f64> = (0..n).map(|i| a[i] * h[i] + b[i] * x).collect();
        let y = (0..n).map(|i| c[i] * new_h[i]).sum();
        Ok((y, new_h))
    }
}

/// Mamba's input-dependent SSM: dt, B, C from linear projections of x.
#[derive(Debug, Clone)]
pub struct SelectiveStateSpace {
    pub d_model: usize,
    pub d_state: usize,
    pub w_dt: Vec<f64>,
    pub b_dt: Vec<f64>,
    pub w_b: Vec<f64>,
    pub b_b: Vec<f64>,
    pub w_c: Vec<f64>,
    pub b_c: Vec<f64>,
    pub log_a: Vec<f64>,
}
impl SelectiveStateSpace {
    pub fn new(d_model: usize, d_state: usize, seed: u64) -> Res<Self> {
        if d_model == 0 || d_state == 0 {
            return Err("SelectiveStateSpace: dims > 0".into());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            d_model,
            d_state,
            w_dt: xavier_init64(d_model, 1, &mut rng),
            b_dt: vec![0.0; 1],
            w_b: xavier_init64(d_model, d_state, &mut rng),
            b_b: vec![0.0; d_state],
            w_c: xavier_init64(d_model, d_state, &mut rng),
            b_c: vec![0.0; d_state],
            log_a: (0..d_state)
                .map(|i| -((i + 1) as f64).ln().max(0.1))
                .collect(),
        })
    }
    pub fn forward(&self, x: &[f64], a: &[f64]) -> Res<Vec<f64>> {
        let (d, n) = (self.d_model, self.d_state);
        if x.len() % d != 0 {
            return Err(format!("SelectiveStateSpace: {} % {d} != 0", x.len()));
        }
        let seq_len = x.len() / d;
        let scan = SelectiveScanCausal::new(n)?;
        let mut h = vec![0.0f64; n];
        let mut out = vec![0.0f64; seq_len * d];
        for t in 0..seq_len {
            let xt = &x[t * d..(t + 1) * d];
            let dt_raw = linear64(xt, &self.w_dt, &self.b_dt, 1)?;
            let dt = softplus64(dt_raw[0]).max(1e-4);
            let b_vec = linear64(xt, &self.w_b, &self.b_b, n)?;
            let c_vec = linear64(xt, &self.w_c, &self.b_c, n)?;
            let a_disc: Vec<f64> = (0..n)
                .map(|i| {
                    let ai = if i < a.len() { a[i] } else { -1.0 };
                    (ai * dt).exp()
                })
                .collect();
            let b_disc: Vec<f64> = (0..n)
                .map(|i| {
                    let ai = if i < a.len() { a[i] } else { -1.0 };
                    let ea = (ai * dt).exp();
                    if ai.abs() < 1e-12 {
                        dt * b_vec[i]
                    } else {
                        (ea - 1.0) / ai * b_vec[i]
                    }
                })
                .collect();
            let x_mean: f64 = xt.iter().sum::<f64>() / d as f64;
            let (y_scalar, new_h) = scan.step(x_mean, &h, &a_disc, &b_disc, &c_vec)?;
            h = new_h;
            for ch in 0..d {
                out[t * d + ch] = xt[ch] + y_scalar;
            }
        }
        Ok(out)
    }
}

/// Full Mamba block: expand → split(z,x) → SSM(x)*silu(z) → contract.
#[derive(Debug, Clone)]
pub struct MambaBlock {
    pub d_model: usize,
    pub expand: usize,
    pub w_in: Vec<f64>,
    pub b_in: Vec<f64>,
    pub w_out: Vec<f64>,
    pub b_out: Vec<f64>,
    pub ssm: SelectiveStateSpace,
}
impl MambaBlock {
    pub fn new(d_model: usize, d_state: usize, expand: usize, seed: u64) -> Res<Self> {
        if d_model == 0 || d_state == 0 || expand == 0 {
            return Err("MambaBlock: dims > 0".into());
        }
        let d_inner = d_model * expand;
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            d_model,
            expand,
            w_in: xavier_init64(d_model, d_inner * 2, &mut rng),
            b_in: vec![0.0; d_inner * 2],
            w_out: xavier_init64(d_inner, d_model, &mut rng),
            b_out: vec![0.0; d_model],
            ssm: SelectiveStateSpace::new(d_inner, d_state, seed + 1)?,
        })
    }
    pub fn forward(&self, x: &[f64]) -> Res<Vec<f64>> {
        let d = self.d_model;
        if x.len() != d {
            return Err(format!("MambaBlock: expected {d}, got {}", x.len()));
        }
        let d_inner = d * self.expand;
        let xn = layer_norm64(x, 1e-5);
        let expanded = linear64(&xn, &self.w_in, &self.b_in, d_inner * 2)?;
        let x_branch = expanded[..d_inner].to_vec();
        let z_branch = &expanded[d_inner..];
        let a_neg: Vec<f64> = self.ssm.log_a.iter().map(|v| -v.abs()).collect();
        let ssm_out = self.ssm.forward(&x_branch, &a_neg)?;
        let gated: Vec<f64> = ssm_out
            .iter()
            .zip(z_branch.iter())
            .map(|(&s, &z)| s * silu64(z))
            .collect();
        let out = linear64(&gated, &self.w_out, &self.b_out, d)?;
        Ok(x.iter().zip(out.iter()).map(|(&a, &b)| a + b).collect())
    }
}

/// Stack of MambaBlocks + embedding + LM head.
#[derive(Debug, Clone)]
pub struct MambaModel {
    pub vocab_size: usize,
    pub d_model: usize,
    pub embedding: Vec<f64>,
    pub blocks: Vec<MambaBlock>,
    pub lm_head: Vec<f64>,
    pub lm_bias: Vec<f64>,
}
impl MambaModel {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        d_state: usize,
        num_layers: usize,
        seed: u64,
    ) -> Res<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        let embedding = normal_init64(vocab_size * d_model, &mut rng)
            .into_iter()
            .map(|v| v * 0.02)
            .collect();
        let mut blocks = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            blocks.push(MambaBlock::new(d_model, d_state, 2, seed + l as u64 + 1)?);
        }
        let lm_head = xavier_init64(d_model, vocab_size, &mut rng);
        Ok(Self {
            vocab_size,
            d_model,
            embedding,
            blocks,
            lm_head,
            lm_bias: vec![0.0; vocab_size],
        })
    }
    pub fn forward(&self, token_ids: &[usize]) -> Res<Vec<Vec<f64>>> {
        let (d, v) = (self.d_model, self.vocab_size);
        let mut h: Vec<Vec<f64>> = token_ids
            .iter()
            .map(|&tid| {
                let t = tid.min(v.saturating_sub(1));
                self.embedding[t * d..(t + 1) * d].to_vec()
            })
            .collect();
        for block in &self.blocks {
            let mut h_new = Vec::with_capacity(h.len());
            for ht in &h {
                h_new.push(block.forward(ht)?);
            }
            h = h_new;
        }
        h.iter()
            .map(|ht| {
                let normed = layer_norm64(ht, 1e-5);
                linear64(&normed, &self.lm_head, &self.lm_bias, v)
            })
            .collect()
    }
}

// ── 3. Linear Recurrences (Hawk/Griffin) ─────────────────────────────────────

/// Griffin gated linear recurrence: `h_t = sigmoid(a_t)*h_{t-1} + sigmoid(b_t)*x_t`.
#[derive(Debug, Clone)]
pub struct LinearRecurrenceLayer {
    pub dim: usize,
}
impl LinearRecurrenceLayer {
    pub fn new(dim: usize) -> Res<Self> {
        if dim == 0 {
            return Err("LinearRecurrenceLayer: dim > 0".into());
        }
        Ok(Self { dim })
    }
    pub fn forward(&self, x: &[f64], a_gate: &[f64], b_gate: &[f64]) -> Res<Vec<f64>> {
        let d = self.dim;
        let n = x.len();
        if n % d != 0 || a_gate.len() != n || b_gate.len() != n {
            return Err(format!("LinearRecurrenceLayer: shape mismatch x={n}"));
        }
        let seq_len = n / d;
        let mut h = vec![0.0f64; d];
        let mut out = vec![0.0f64; n];
        for t in 0..seq_len {
            for i in 0..d {
                let idx = t * d + i;
                h[i] = sigmoid64(a_gate[idx]) * h[i] + sigmoid64(b_gate[idx]) * x[idx];
                out[idx] = h[i];
            }
        }
        Ok(out)
    }
}

/// HAWK block: linear recurrence + SwiGLU MLP.
#[derive(Debug, Clone)]
pub struct HawkBlock {
    pub dim: usize,
    pub recurrence: LinearRecurrenceLayer,
    pub mlp_w1: Vec<f64>,
    pub mlp_b1: Vec<f64>,
    pub mlp_w2: Vec<f64>,
    pub mlp_b2: Vec<f64>,
    pub gate_w: Vec<f64>,
    pub gate_b: Vec<f64>,
}
impl HawkBlock {
    pub fn new(dim: usize, seed: u64) -> Res<Self> {
        if dim == 0 {
            return Err("HawkBlock: dim > 0".into());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            dim,
            recurrence: LinearRecurrenceLayer::new(dim)?,
            mlp_w1: xavier_init64(dim, dim * 4, &mut rng),
            mlp_b1: vec![0.0; dim * 4],
            mlp_w2: xavier_init64(dim * 4, dim, &mut rng),
            mlp_b2: vec![0.0; dim],
            gate_w: xavier_init64(dim, dim * 4, &mut rng),
            gate_b: vec![0.0; dim * 4],
        })
    }
    pub fn forward(&self, x: &[f64]) -> Res<Vec<f64>> {
        let d = self.dim;
        if x.len() != d {
            return Err(format!("HawkBlock: expected {d}"));
        }
        let a_gate = vec![0.5f64; d];
        let b_gate = vec![0.5f64; d];
        let rec = self.recurrence.forward(x, &a_gate, &b_gate)?;
        let h1 = linear64(&rec, &self.mlp_w1, &self.mlp_b1, d * 4)?;
        let g = linear64(&rec, &self.gate_w, &self.gate_b, d * 4)?;
        let gated: Vec<f64> = h1
            .iter()
            .zip(g.iter())
            .map(|(&h, &gv)| h * silu64(gv))
            .collect();
        let out = linear64(&gated, &self.mlp_w2, &self.mlp_b2, d)?;
        Ok(x.iter().zip(out.iter()).map(|(&a, &b)| a + b).collect())
    }
}

/// Recurrent Gemma: local attention + linear recurrence alternating.
#[derive(Debug, Clone)]
pub struct RecurrentGemmaBlock {
    pub dim: usize,
    pub num_heads: usize,
    pub recurrence: LinearRecurrenceLayer,
    w_q: Vec<f64>,
    w_k: Vec<f64>,
    w_v: Vec<f64>,
    w_o: Vec<f64>,
    ffn_w1: Vec<f64>,
    ffn_b1: Vec<f64>,
    ffn_w2: Vec<f64>,
    ffn_b2: Vec<f64>,
}
impl RecurrentGemmaBlock {
    pub fn new(dim: usize, num_heads: usize, seed: u64) -> Res<Self> {
        if dim == 0 || num_heads == 0 || dim % num_heads != 0 {
            return Err(format!(
                "RecurrentGemmaBlock: dim={dim} not divisible by {num_heads}"
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            dim,
            num_heads,
            recurrence: LinearRecurrenceLayer::new(dim)?,
            w_q: xavier_init64(dim, dim, &mut rng),
            w_k: xavier_init64(dim, dim, &mut rng),
            w_v: xavier_init64(dim, dim, &mut rng),
            w_o: xavier_init64(dim, dim, &mut rng),
            ffn_w1: xavier_init64(dim, dim * 4, &mut rng),
            ffn_b1: vec![0.0; dim * 4],
            ffn_w2: xavier_init64(dim * 4, dim, &mut rng),
            ffn_b2: vec![0.0; dim],
        })
    }
    fn local_attn(&self, x: &[Vec<f64>], w: usize) -> Res<Vec<Vec<f64>>> {
        let (d, scale) = (self.dim, 1.0 / (self.dim as f64).sqrt());
        let bz = vec![0.0f64; d];
        let mut out = vec![vec![0.0f64; d]; x.len()];
        for t in 0..x.len() {
            let start = (t + 1).saturating_sub(w);
            let q = linear64(&x[t], &self.w_q, &bz, d)?;
            let (mut scores, mut vs) = (Vec::new(), Vec::new());
            for s in start..=t {
                let k = linear64(&x[s], &self.w_k, &bz, d)?;
                scores.push(
                    q.iter()
                        .zip(k.iter())
                        .map(|(&qi, &ki)| qi * ki)
                        .sum::<f64>()
                        * scale,
                );
                vs.push(linear64(&x[s], &self.w_v, &bz, d)?);
            }
            let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_s: Vec<f64> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum_e = exp_s.iter().sum::<f64>().max(1e-12);
            let attn: Vec<f64> = exp_s.iter().map(|e| e / sum_e).collect();
            let mut ctx = vec![0.0f64; d];
            for (a, v) in attn.iter().zip(vs.iter()) {
                for i in 0..d {
                    ctx[i] += a * v[i];
                }
            }
            out[t] = linear64(&ctx, &self.w_o, &bz, d)?;
        }
        Ok(out)
    }
    pub fn forward(&self, x: &[Vec<f64>], local_window: usize) -> Res<Vec<Vec<f64>>> {
        let d = self.dim;
        let attn_out = self.local_attn(x, local_window)?;
        let flat: Vec<f64> = attn_out.iter().flat_map(|v| v.iter().copied()).collect();
        let a_gate = vec![0.0f64; flat.len()];
        let b_gate = vec![1.0f64; flat.len()];
        let rec_flat = self.recurrence.forward(&flat, &a_gate, &b_gate)?;
        let mut out = Vec::with_capacity(x.len());
        for t in 0..x.len() {
            let rt = &rec_flat[t * d..(t + 1) * d];
            let res: Vec<f64> = x[t].iter().zip(rt.iter()).map(|(&a, &b)| a + b).collect();
            let normed = layer_norm64(&res, 1e-5);
            let ff1: Vec<f64> = linear64(&normed, &self.ffn_w1, &self.ffn_b1, d * 4)?
                .into_iter()
                .map(relu64)
                .collect();
            let ff2 = linear64(&ff1, &self.ffn_w2, &self.ffn_b2, d)?;
            out.push(res.iter().zip(ff2.iter()).map(|(&a, &b)| a + b).collect());
        }
        Ok(out)
    }
}

/// Minimal GRU: `h_t = z*h_{t-1} + (1-z)*tanh(x_t + r*h_{t-1})`.
#[derive(Debug, Clone)]
pub struct GatedRecurrentUnit {
    pub hidden_dim: usize,
    pub input_dim: usize,
    pub w_z: Vec<f64>,
    pub b_z: Vec<f64>,
    pub w_r: Vec<f64>,
    pub b_r: Vec<f64>,
    pub w_h: Vec<f64>,
    pub b_h: Vec<f64>,
}
impl GatedRecurrentUnit {
    pub fn new(input_dim: usize, hidden_dim: usize, seed: u64) -> Res<Self> {
        if input_dim == 0 || hidden_dim == 0 {
            return Err("GRU: dims > 0".into());
        }
        let combined = input_dim + hidden_dim;
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            hidden_dim,
            input_dim,
            w_z: xavier_init64(combined, hidden_dim, &mut rng),
            b_z: vec![0.0; hidden_dim],
            w_r: xavier_init64(combined, hidden_dim, &mut rng),
            b_r: vec![0.0; hidden_dim],
            w_h: xavier_init64(combined, hidden_dim, &mut rng),
            b_h: vec![0.0; hidden_dim],
        })
    }
    pub fn step(&self, x: &[f64], h: &[f64]) -> Res<Vec<f64>> {
        let (hi, hh) = (self.input_dim, self.hidden_dim);
        if x.len() != hi || h.len() != hh {
            return Err(format!(
                "GRU::step: x={} h={} (expected {hi} {hh})",
                x.len(),
                h.len()
            ));
        }
        let xh: Vec<f64> = x.iter().chain(h.iter()).copied().collect();
        let z: Vec<f64> = linear64(&xh, &self.w_z, &self.b_z, hh)?
            .into_iter()
            .map(sigmoid64)
            .collect();
        let r: Vec<f64> = linear64(&xh, &self.w_r, &self.b_r, hh)?
            .into_iter()
            .map(sigmoid64)
            .collect();
        let xrh: Vec<f64> = x
            .iter()
            .copied()
            .chain(h.iter().zip(r.iter()).map(|(&hi, &ri)| ri * hi))
            .collect();
        let h_cand: Vec<f64> = linear64(&xrh, &self.w_h, &self.b_h, hh)?
            .into_iter()
            .map(tanh64)
            .collect();
        Ok(z.iter()
            .zip(h.iter())
            .zip(h_cand.iter())
            .map(|((&zi, &hi), &hci)| zi * hi + (1.0 - zi) * hci)
            .collect())
    }
}

/// Linearized self-attention via ELU+1 feature maps — O(N).
#[derive(Debug, Clone)]
pub struct LinearizedSelfAttention {
    pub d_model: usize,
    pub num_heads: usize,
    pub w_q: Vec<f64>,
    pub w_k: Vec<f64>,
    pub w_v: Vec<f64>,
    pub w_o: Vec<f64>,
}
impl LinearizedSelfAttention {
    pub fn new(d_model: usize, num_heads: usize, seed: u64) -> Res<Self> {
        if d_model == 0 || num_heads == 0 || d_model % num_heads != 0 {
            return Err(format!(
                "LinearizedSelfAttention: d_model={d_model} % num_heads={num_heads} != 0"
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            d_model,
            num_heads,
            w_q: xavier_init64(d_model, d_model, &mut rng),
            w_k: xavier_init64(d_model, d_model, &mut rng),
            w_v: xavier_init64(d_model, d_model, &mut rng),
            w_o: xavier_init64(d_model, d_model, &mut rng),
        })
    }
    #[inline]
    fn phi(x: f64) -> f64 {
        if x >= 0.0 {
            x + 1.0
        } else {
            x.exp()
        }
    }
    pub fn forward(&self, x: &[f64]) -> Res<Vec<f64>> {
        let d = self.d_model;
        if x.len() % d != 0 {
            return Err(format!("LinearizedSelfAttention: {} % {d} != 0", x.len()));
        }
        let seq_len = x.len() / d;
        let bz = vec![0.0f64; d];
        let qs_phi: Vec<Vec<f64>> = (0..seq_len)
            .map(|t| {
                linear64(&x[t * d..(t + 1) * d], &self.w_q, &bz, d)
                    .map(|q| q.into_iter().map(Self::phi).collect())
            })
            .collect::<Res<_>>()?;
        let ks_phi: Vec<Vec<f64>> = (0..seq_len)
            .map(|t| {
                linear64(&x[t * d..(t + 1) * d], &self.w_k, &bz, d)
                    .map(|k| k.into_iter().map(Self::phi).collect())
            })
            .collect::<Res<_>>()?;
        let vs: Vec<Vec<f64>> = (0..seq_len)
            .map(|t| linear64(&x[t * d..(t + 1) * d], &self.w_v, &bz, d))
            .collect::<Res<_>>()?;
        let mut kv_sum = vec![vec![0.0f64; d]; d];
        let mut k_sum = vec![0.0f64; d];
        let mut out_flat = vec![0.0f64; x.len()];
        for t in 0..seq_len {
            for i in 0..d {
                for j in 0..d {
                    kv_sum[i][j] += ks_phi[t][i] * vs[t][j];
                }
                k_sum[i] += ks_phi[t][i];
            }
            let q = &qs_phi[t];
            let denom = q
                .iter()
                .zip(k_sum.iter())
                .map(|(&qi, &ki)| qi * ki)
                .sum::<f64>()
                .max(1e-6);
            let mut ctx = vec![0.0f64; d];
            for i in 0..d {
                for j in 0..d {
                    ctx[j] += q[i] * kv_sum[i][j];
                }
            }
            let yt = linear64(
                &ctx.iter().map(|&v| v / denom).collect::<Vec<_>>(),
                &self.w_o,
                &bz,
                d,
            )?;
            out_flat[t * d..(t + 1) * d].copy_from_slice(&yt);
        }
        Ok(out_flat)
    }
}

// ── 4. Continuous-Time Models ─────────────────────────────────────────────────

/// CT-RNN: `dx/dt = -x/τ + tanh(W_x x + W_u u + b)`.  Euler step.
#[derive(Debug, Clone)]
pub struct ContinuousTimeRnn {
    pub state_dim: usize,
    pub input_dim: usize,
    pub tau: f64,
    pub w_x: Vec<f64>,
    pub w_u: Vec<f64>,
    pub b: Vec<f64>,
}
impl ContinuousTimeRnn {
    pub fn new(state_dim: usize, input_dim: usize, tau: f64, seed: u64) -> Res<Self> {
        if state_dim == 0 || input_dim == 0 {
            return Err("ContinuousTimeRnn: dims > 0".into());
        }
        if tau <= 0.0 {
            return Err("ContinuousTimeRnn: tau > 0".into());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            state_dim,
            input_dim,
            tau,
            w_x: xavier_init64(state_dim, state_dim, &mut rng),
            w_u: xavier_init64(input_dim, state_dim, &mut rng),
            b: vec![0.0; state_dim],
        })
    }
    pub fn step(&self, x: &[f64], u: &[f64], dt: f64) -> Res<Vec<f64>> {
        let (ns, ni) = (self.state_dim, self.input_dim);
        if x.len() != ns || u.len() != ni {
            return Err(format!(
                "ContinuousTimeRnn::step: x={} u={}",
                x.len(),
                u.len()
            ));
        }
        let bz = vec![0.0f64; ns];
        let wx_x = linear64(x, &self.w_x, &bz, ns)?;
        let wu_u = linear64(u, &self.w_u, &bz, ns)?;
        let f_x: Vec<f64> = wx_x
            .iter()
            .zip(wu_u.iter())
            .zip(self.b.iter())
            .map(|((&wx, &wu), &bi)| tanh64(wx + wu + bi))
            .collect();
        Ok(x.iter()
            .zip(f_x.iter())
            .map(|(&xi, &fi)| xi + dt * (-xi / self.tau + fi))
            .collect())
    }
}

/// Neural CDE: `dh/dt = f(h) · dX/dt`.  Euler integration.
#[derive(Debug, Clone)]
pub struct NeuralCde {
    pub state_dim: usize,
    pub input_dim: usize,
    pub w_f: Vec<f64>,
    pub b_f: Vec<f64>,
}
impl NeuralCde {
    pub fn new(state_dim: usize, input_dim: usize, seed: u64) -> Res<Self> {
        if state_dim == 0 || input_dim == 0 {
            return Err("NeuralCde: dims > 0".into());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let out_dim = state_dim * input_dim;
        Ok(Self {
            state_dim,
            input_dim,
            w_f: xavier_init64(state_dim, out_dim, &mut rng),
            b_f: vec![0.0; out_dim],
        })
    }
    pub fn integrate(&self, h0: &[f64], x_path: &[Vec<f64>], dt: f64) -> Res<Vec<f64>> {
        let (ns, ni) = (self.state_dim, self.input_dim);
        if h0.len() != ns {
            return Err(format!("NeuralCde: h0 len {} != {ns}", h0.len()));
        }
        if x_path.is_empty() {
            return Ok(h0.to_vec());
        }
        let mut h = h0.to_vec();
        for t in 0..x_path.len().saturating_sub(1) {
            let dx: Vec<f64> = x_path[t + 1]
                .iter()
                .zip(x_path[t].iter())
                .map(|(&n, &c)| (n - c) / dt)
                .collect();
            let f_flat: Vec<f64> = linear64(&h, &self.w_f, &self.b_f, ns * ni)?
                .into_iter()
                .map(tanh64)
                .collect();
            let mut dh = vec![0.0f64; ns];
            for i in 0..ns {
                for j in 0..ni {
                    dh[i] += f_flat[i * ni + j] * dx[j];
                }
            }
            h = h
                .iter()
                .zip(dh.iter())
                .map(|(&hi, &dhi)| hi + dt * dhi)
                .collect();
        }
        Ok(h)
    }
}

/// LTC neuron: `τ(x) = τ_0 / (1+exp(-Wx+b))`, `dh/dt = -h/τ + I`.
#[derive(Debug, Clone)]
pub struct LiquidNeuralNetwork {
    pub state_dim: usize,
    pub input_dim: usize,
    pub tau_0: f64,
    pub w_tau: Vec<f64>,
    pub b_tau: Vec<f64>,
    pub w_in: Vec<f64>,
    pub b_in: Vec<f64>,
}
impl LiquidNeuralNetwork {
    pub fn new(state_dim: usize, input_dim: usize, tau_0: f64, seed: u64) -> Res<Self> {
        if state_dim == 0 || input_dim == 0 {
            return Err("LiquidNeuralNetwork: dims > 0".into());
        }
        if tau_0 <= 0.0 {
            return Err("LiquidNeuralNetwork: tau_0 > 0".into());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            state_dim,
            input_dim,
            tau_0,
            w_tau: xavier_init64(input_dim, state_dim, &mut rng),
            b_tau: vec![0.0; state_dim],
            w_in: xavier_init64(input_dim, state_dim, &mut rng),
            b_in: vec![0.0; state_dim],
        })
    }
    pub fn step(&self, x: &[f64], i_ext: &[f64], h: &[f64], dt: f64) -> Res<Vec<f64>> {
        let (ns, ni) = (self.state_dim, self.input_dim);
        if x.len() != ni || i_ext.len() != ns || h.len() != ns {
            return Err(format!(
                "LiquidNeuralNetwork::step: x={} I={} h={}",
                x.len(),
                i_ext.len(),
                h.len()
            ));
        }
        let tau: Vec<f64> = linear64(x, &self.w_tau, &self.b_tau, ns)?
            .into_iter()
            .map(|v| self.tau_0 / (1.0 + (-v).exp()).max(1e-10))
            .collect();
        let i_in = linear64(x, &self.w_in, &self.b_in, ns)?;
        Ok(h.iter()
            .zip(tau.iter())
            .zip(i_in.iter())
            .zip(i_ext.iter())
            .map(|(((&hi, &ti), &ii), &ie)| hi + dt * (-hi / ti + ii + ie))
            .collect())
    }
}

/// Wired CT-RNN with sparse connectivity mask.
#[derive(Debug, Clone)]
pub struct WiredCtRnn {
    pub state_dim: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub tau: f64,
    w_rec: Vec<f64>,
    w_in: Vec<f64>,
    w_out: Vec<f64>,
    b: Vec<f64>,
}
impl WiredCtRnn {
    pub fn new(
        state_dim: usize,
        input_dim: usize,
        output_dim: usize,
        tau: f64,
        seed: u64,
    ) -> Res<Self> {
        if state_dim == 0 || input_dim == 0 || output_dim == 0 {
            return Err("WiredCtRnn: dims > 0".into());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            state_dim,
            input_dim,
            output_dim,
            tau,
            w_rec: xavier_init64(state_dim, state_dim, &mut rng),
            w_in: xavier_init64(input_dim, state_dim, &mut rng),
            w_out: xavier_init64(state_dim, output_dim, &mut rng),
            b: vec![0.0; state_dim],
        })
    }
    pub fn forward(&self, x: &[Vec<f64>], mask: &[Vec<f64>]) -> Res<Vec<Vec<f64>>> {
        let (ns, ni, no, dt) = (self.state_dim, self.input_dim, self.output_dim, 0.1f64);
        if mask.len() != ns || mask.iter().any(|r| r.len() != ns) {
            return Err(format!("WiredCtRnn: mask must be [{ns}][{ns}]"));
        }
        let bz_s = vec![0.0f64; ns];
        let bz_o = vec![0.0f64; no];
        let mut h = vec![0.0f64; ns];
        let mut out = Vec::with_capacity(x.len());
        for xt in x {
            if xt.len() != ni {
                return Err(format!("WiredCtRnn: input len {} != {ni}", xt.len()));
            }
            let mut wx = linear64(&h, &self.w_rec, &bz_s, ns)?;
            for i in 0..ns {
                wx[i] *= mask[i % mask.len()][i % ns];
            }
            let wu = linear64(xt, &self.w_in, &bz_s, ns)?;
            let f_h: Vec<f64> = wx
                .iter()
                .zip(wu.iter())
                .zip(self.b.iter())
                .map(|((&a, &b), &c)| tanh64(a + b + c))
                .collect();
            h = h
                .iter()
                .zip(f_h.iter())
                .map(|(&hi, &fi)| hi + dt * (-hi / self.tau + fi))
                .collect();
            out.push(linear64(&h, &self.w_out, &bz_o, no)?);
        }
        Ok(out)
    }
}

/// LIF spiking neurons: `V_new = V + dt*(-V/τ + I)`, spike when `V > θ`, reset.
#[derive(Debug, Clone)]
pub struct SpikingNeuralNetwork {
    pub num_neurons: usize,
    pub tau: f64,
    pub threshold: f64,
    pub v_reset: f64,
}
impl SpikingNeuralNetwork {
    pub fn new(num_neurons: usize, tau: f64, threshold: f64) -> Res<Self> {
        if num_neurons == 0 {
            return Err("SpikingNeuralNetwork: num_neurons > 0".into());
        }
        if tau <= 0.0 {
            return Err("SpikingNeuralNetwork: tau > 0".into());
        }
        Ok(Self {
            num_neurons,
            tau,
            threshold,
            v_reset: 0.0,
        })
    }
    pub fn step(&self, v: &[f64], i_ext: &[f64], dt: f64) -> Res<(Vec<f64>, Vec<bool>)> {
        let n = self.num_neurons;
        if v.len() != n || i_ext.len() != n {
            return Err(format!("SpikingNeuralNetwork::step: {n}"));
        }
        let mut new_v = Vec::with_capacity(n);
        let mut spikes = Vec::with_capacity(n);
        for i in 0..n {
            let v_new = v[i] + dt * (-v[i] / self.tau + i_ext[i]);
            if v_new > self.threshold {
                spikes.push(true);
                new_v.push(self.v_reset);
            } else {
                spikes.push(false);
                new_v.push(v_new);
            }
        }
        Ok((new_v, spikes))
    }
}

// ── 5. Hybrid Architectures ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TransformerBlock {
    d_model: usize,
    w_q: Vec<f64>,
    w_k: Vec<f64>,
    w_v: Vec<f64>,
    w_o: Vec<f64>,
    ffn_w1: Vec<f64>,
    ffn_b1: Vec<f64>,
    ffn_w2: Vec<f64>,
    ffn_b2: Vec<f64>,
}
impl TransformerBlock {
    fn new(d_model: usize, seed: u64) -> Res<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            d_model,
            w_q: xavier_init64(d_model, d_model, &mut rng),
            w_k: xavier_init64(d_model, d_model, &mut rng),
            w_v: xavier_init64(d_model, d_model, &mut rng),
            w_o: xavier_init64(d_model, d_model, &mut rng),
            ffn_w1: xavier_init64(d_model, d_model * 4, &mut rng),
            ffn_b1: vec![0.0; d_model * 4],
            ffn_w2: xavier_init64(d_model * 4, d_model, &mut rng),
            ffn_b2: vec![0.0; d_model],
        })
    }
    fn forward(&self, x: &[Vec<f64>]) -> Res<Vec<Vec<f64>>> {
        let (d, scale) = (self.d_model, 1.0 / (self.d_model as f64).sqrt());
        let bz = vec![0.0f64; d];
        let mut out = Vec::with_capacity(x.len());
        for t in 0..x.len() {
            let q = linear64(&x[t], &self.w_q, &bz, d)?;
            let mut scores = vec![0.0f64; t + 1];
            for s in 0..=t {
                let k = linear64(&x[s], &self.w_k, &bz, d)?;
                scores[s] = q
                    .iter()
                    .zip(k.iter())
                    .map(|(&qi, &ki)| qi * ki)
                    .sum::<f64>()
                    * scale;
            }
            let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_s: Vec<f64> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum_e = exp_s.iter().sum::<f64>().max(1e-12);
            let attn: Vec<f64> = exp_s.iter().map(|e| e / sum_e).collect();
            let mut ctx = vec![0.0f64; d];
            for s in 0..=t {
                let v = linear64(&x[s], &self.w_v, &bz, d)?;
                for i in 0..d {
                    ctx[i] += attn[s] * v[i];
                }
            }
            let attn_out = linear64(&ctx, &self.w_o, &bz, d)?;
            let res1: Vec<f64> = x[t]
                .iter()
                .zip(attn_out.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            let normed = layer_norm64(&res1, 1e-5);
            let ff1: Vec<f64> = linear64(&normed, &self.ffn_w1, &self.ffn_b1, d * 4)?
                .into_iter()
                .map(relu64)
                .collect();
            let ff2 = linear64(&ff1, &self.ffn_w2, &self.ffn_b2, d)?;
            out.push(res1.iter().zip(ff2.iter()).map(|(&a, &b)| a + b).collect());
        }
        Ok(out)
    }
}

/// Interleave Transformer and Mamba (SSM) blocks.
#[derive(Debug, Clone)]
pub struct TransformerSsmHybrid {
    pub d_model: usize,
    pub num_transformer: usize,
    pub num_ssm_per_transformer: usize,
    transformer_blocks: Vec<TransformerBlock>,
    mamba_blocks: Vec<MambaBlock>,
}
impl TransformerSsmHybrid {
    pub fn new(
        d_model: usize,
        num_transformer: usize,
        num_ssm: usize,
        d_state: usize,
        seed: u64,
    ) -> Res<Self> {
        if d_model == 0 {
            return Err("TransformerSsmHybrid: d_model > 0".into());
        }
        let transformer_blocks: Res<Vec<_>> = (0..num_transformer)
            .map(|i| TransformerBlock::new(d_model, seed + i as u64))
            .collect();
        let mamba_blocks: Res<Vec<_>> = (0..num_ssm)
            .map(|i| MambaBlock::new(d_model, d_state, 2, seed + 100 + i as u64))
            .collect();
        Ok(Self {
            d_model,
            num_transformer,
            num_ssm_per_transformer: num_ssm,
            transformer_blocks: transformer_blocks?,
            mamba_blocks: mamba_blocks?,
        })
    }
    pub fn forward(&self, x: &[Vec<f64>]) -> Res<Vec<Vec<f64>>> {
        let mut h = x.to_vec();
        let max_layers = self.transformer_blocks.len().max(self.mamba_blocks.len());
        for i in 0..max_layers {
            if i < self.transformer_blocks.len() {
                h = self.transformer_blocks[i].forward(&h)?;
            }
            if i < self.mamba_blocks.len() {
                let mut h_new = Vec::with_capacity(h.len());
                for ht in &h {
                    h_new.push(self.mamba_blocks[i].forward(ht)?);
                }
                h = h_new;
            }
        }
        Ok(h)
    }
}

/// Jamba: attention + Mamba + MoE FFN.
#[derive(Debug, Clone)]
pub struct JambaBlock {
    pub d_model: usize,
    attn: TransformerBlock,
    mamba: MambaBlock,
    expert_w1: Vec<Vec<f64>>,
    expert_b1: Vec<Vec<f64>>,
    expert_w2: Vec<Vec<f64>>,
    expert_b2: Vec<Vec<f64>>,
    router_w: Vec<f64>,
}
impl JambaBlock {
    pub fn new(d_model: usize, d_state: usize, num_experts: usize, seed: u64) -> Res<Self> {
        if d_model == 0 || num_experts == 0 {
            return Err("JambaBlock: dims > 0".into());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let attn = TransformerBlock::new(d_model, seed)?;
        let mamba = MambaBlock::new(d_model, d_state, 2, seed + 1)?;
        let (mut ew1, mut eb1, mut ew2, mut eb2) = (vec![], vec![], vec![], vec![]);
        for _ in 0..num_experts {
            ew1.push(xavier_init64(d_model, d_model * 4, &mut rng));
            eb1.push(vec![0.0; d_model * 4]);
            ew2.push(xavier_init64(d_model * 4, d_model, &mut rng));
            eb2.push(vec![0.0; d_model]);
        }
        Ok(Self {
            d_model,
            attn,
            mamba,
            expert_w1: ew1,
            expert_b1: eb1,
            expert_w2: ew2,
            expert_b2: eb2,
            router_w: xavier_init64(d_model, num_experts, &mut rng),
        })
    }
    pub fn forward(&self, x: &[Vec<f64>]) -> Res<Vec<Vec<f64>>> {
        let (ne, d) = (self.expert_w1.len(), self.d_model);
        let attn_out = self.attn.forward(x)?;
        let mut mamba_out = Vec::with_capacity(x.len());
        for xt in &attn_out {
            mamba_out.push(self.mamba.forward(xt)?);
        }
        let bz_ne = vec![0.0f64; ne];
        let mut out = Vec::with_capacity(x.len());
        for xt in &mamba_out {
            let normed = layer_norm64(xt, 1e-5);
            let r = linear64(&normed, &self.router_w, &bz_ne, ne)?;
            let max_r = r.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_r: Vec<f64> = r.iter().map(|&v| (v - max_r).exp()).collect();
            let sum_r = exp_r.iter().sum::<f64>().max(1e-12);
            let rw: Vec<f64> = exp_r.iter().map(|v| v / sum_r).collect();
            let mut idx: Vec<usize> = (0..ne).collect();
            idx.sort_by(|&a, &b| {
                rw[b]
                    .partial_cmp(&rw[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut y = xt.clone();
            for &ei in &idx[..2.min(ne)] {
                let h1: Vec<f64> =
                    linear64(&normed, &self.expert_w1[ei], &self.expert_b1[ei], d * 4)?
                        .into_iter()
                        .map(relu64)
                        .collect();
                let h2 = linear64(&h1, &self.expert_w2[ei], &self.expert_b2[ei], d)?;
                for i in 0..d {
                    y[i] += rw[ei] * h2[i];
                }
            }
            out.push(y);
        }
        Ok(out)
    }
}

/// Zamba: shared global attention + sequential Mamba blocks.
#[derive(Debug, Clone)]
pub struct ZambaBlock {
    pub d_model: usize,
    pub num_mamba: usize,
    global_attn: TransformerBlock,
    mamba_blocks: Vec<MambaBlock>,
    combine_w: Vec<f64>,
    combine_b: Vec<f64>,
}
impl ZambaBlock {
    pub fn new(d_model: usize, d_state: usize, num_mamba: usize, seed: u64) -> Res<Self> {
        if d_model == 0 || num_mamba == 0 {
            return Err("ZambaBlock: dims > 0".into());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mamba_blocks: Res<Vec<_>> = (0..num_mamba)
            .map(|i| MambaBlock::new(d_model, d_state, 2, seed + i as u64 + 1))
            .collect();
        Ok(Self {
            d_model,
            num_mamba,
            global_attn: TransformerBlock::new(d_model, seed)?,
            mamba_blocks: mamba_blocks?,
            combine_w: xavier_init64(d_model * 2, d_model, &mut rng),
            combine_b: vec![0.0; d_model],
        })
    }
    pub fn forward(&self, x: &[Vec<f64>]) -> Res<Vec<Vec<f64>>> {
        let d = self.d_model;
        let attn_out = self.global_attn.forward(x)?;
        let mut mamba_h = x.to_vec();
        for mb in &self.mamba_blocks {
            let mut h_new = Vec::with_capacity(x.len());
            for ht in &mamba_h {
                h_new.push(mb.forward(ht)?);
            }
            mamba_h = h_new;
        }
        let mut out = Vec::with_capacity(x.len());
        for t in 0..x.len() {
            let combined: Vec<f64> = attn_out[t]
                .iter()
                .chain(mamba_h[t].iter())
                .copied()
                .collect();
            out.push(linear64(&combined, &self.combine_w, &self.combine_b, d)?);
        }
        Ok(out)
    }
}

/// Falcon-Mamba: pre-norm + Mamba + pre-norm + FFN with SiLU.
#[derive(Debug, Clone)]
pub struct FalconMambaBlock {
    pub d_model: usize,
    mamba: MambaBlock,
    ffn_w1: Vec<f64>,
    ffn_b1: Vec<f64>,
    ffn_w2: Vec<f64>,
    ffn_b2: Vec<f64>,
}
impl FalconMambaBlock {
    pub fn new(d_model: usize, d_state: usize, seed: u64) -> Res<Self> {
        if d_model == 0 {
            return Err("FalconMambaBlock: d_model > 0".into());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            d_model,
            mamba: MambaBlock::new(d_model, d_state, 2, seed)?,
            ffn_w1: xavier_init64(d_model, d_model * 4, &mut rng),
            ffn_b1: vec![0.0; d_model * 4],
            ffn_w2: xavier_init64(d_model * 4, d_model, &mut rng),
            ffn_b2: vec![0.0; d_model],
        })
    }
    pub fn forward(&self, x: &[Vec<f64>]) -> Res<Vec<Vec<f64>>> {
        let d = self.d_model;
        let mut out = Vec::with_capacity(x.len());
        for xt in x {
            let mamba_out = self.mamba.forward(&layer_norm64(xt, 1e-5))?;
            let res1: Vec<f64> = xt
                .iter()
                .zip(mamba_out.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            let ff1: Vec<f64> = linear64(
                &layer_norm64(&res1, 1e-5),
                &self.ffn_w1,
                &self.ffn_b1,
                d * 4,
            )?
            .into_iter()
            .map(silu64)
            .collect();
            let ff2 = linear64(&ff1, &self.ffn_w2, &self.ffn_b2, d)?;
            out.push(res1.iter().zip(ff2.iter()).map(|(&a, &b)| a + b).collect());
        }
        Ok(out)
    }
}

/// Benchmark SSM recurrence vs Transformer attention complexity.
#[derive(Debug, Clone)]
pub struct SsmEvaluator;
impl SsmEvaluator {
    /// Returns `(ops_ratio, memory_ratio)` for SSM vs Transformer; < 1.0 means SSM is cheaper.
    pub fn compare_recurrence_vs_attention(seq_len: usize) -> (f64, f64) {
        let d_model = 64usize;
        let d_state = 16usize;
        let attn_ops = (seq_len * seq_len * d_model) as f64;
        let ssm_ops = (seq_len * d_model * d_state) as f64;
        let attn_mem = (seq_len * seq_len) as f64;
        let ssm_mem = (seq_len * d_state) as f64;
        let ops_ratio = if attn_ops > 0.0 {
            ssm_ops / attn_ops
        } else {
            1.0
        };
        let mem_ratio = if attn_mem > 0.0 {
            ssm_mem / attn_mem
        } else {
            1.0
        };
        (ops_ratio, mem_ratio)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // S4
    #[test]
    fn test_hippo_matrix_shape() {
        let (a, b) = HiPPoMatrix::compute(8);
        assert_eq!(a.len(), 8);
        assert_eq!(b.len(), 8);
        for row in &a {
            assert_eq!(row.len(), 8);
        }
    }
    #[test]
    fn test_hippo_matrix_values() {
        let (a, b) = HiPPoMatrix::compute(4);
        assert!((a[0][0] - (-1.0)).abs() < 1e-10);
        assert!((a[1][1] - (-2.0)).abs() < 1e-10);
        assert!((b[0] - 1.0).abs() < 1e-10);
        assert!((b[1] - 3.0_f64.sqrt()).abs() < 1e-10);
    }
    #[test]
    fn test_s4_discretization() {
        let a_diag = vec![-1.0, -2.0, -3.0];
        let b = vec![1.0; 3];
        let (a_bar, b_bar) = S4Discretization::discretize(&a_diag, &b, 0.01).expect("test: operation should succeed");
        assert_eq!(a_bar.len(), 3);
        assert_eq!(b_bar.len(), 3);
        for i in 0..3 {
            assert!((a_bar[i] - (a_diag[i] * 0.01).exp()).abs() < 1e-10);
        }
    }
    #[test]
    fn test_s4_discretization_bilinear() {
        let (a_bar, b_bar) =
            S4Discretization::discretize_bilinear(&[-1.0, -2.0], &[1.0, 1.0], 0.1).expect("test: operation should succeed");
        assert_eq!(a_bar.len(), 2);
        assert_eq!(b_bar.len(), 2);
        for v in a_bar.iter().chain(b_bar.iter()) {
            assert!(v.is_finite());
        }
    }
    #[test]
    fn test_s4_kernel_length() {
        let k = S4Kernel::compute_kernel(&[-0.5, -1.0], &[1.0, 0.5], &[1.0, 1.0], 16).expect("test: operation should succeed");
        assert_eq!(k.len(), 16);
    }
    #[test]
    fn test_s4_layer_causal() {
        let layer = S4Layer::new(4, 1, 42).expect("test: operation should succeed");
        let kernel = vec![1.0, 0.5, 0.25, 0.125, 0.0625];
        let y = layer.forward(&[1.0, 0.0, 0.0, 0.0, 0.0], &kernel).expect("test: operation should succeed");
        assert_eq!(y.len(), 5);
        assert!((y[0] - 2.0).abs() < 1e-10); // kernel[0]*1 + D*1 = 1+1 = 2
        assert!((y[1] - 0.5).abs() < 1e-10); // kernel[1]*1 + D*0 = 0.5
    }
    #[test]
    fn test_s4_model_shape() {
        let model = S4Model::new(2, 8, 4, 4, 8, 42).expect("test: operation should succeed");
        let x: Vec<Vec<f64>> = (0..5).map(|_| vec![0.1f64; 4]).collect();
        let y = model.forward(&x).expect("test: operation should succeed");
        assert_eq!(y.len(), 5);
        assert_eq!(y[0].len(), 8);
    }

    // Mamba
    #[test]
    fn test_selective_ssm_forward() {
        let ssm = SelectiveStateSpace::new(4, 8, 42).expect("test: operation should succeed");
        let y = ssm.forward(&[0.1f64; 12], &[-1.0f64; 8]).expect("test: operation should succeed");
        assert_eq!(y.len(), 12);
        for v in &y {
            assert!(v.is_finite());
        }
    }
    #[test]
    fn test_mamba_block_shape() {
        let block = MambaBlock::new(8, 4, 2, 42).expect("test: operation should succeed");
        let y = block.forward(&[0.5f64; 8]).expect("test: operation should succeed");
        assert_eq!(y.len(), 8);
    }
    #[test]
    fn test_mamba_model_logits_shape() {
        let model = MambaModel::new(16, 8, 4, 2, 42).expect("test: operation should succeed");
        let logits = model.forward(&[0, 3, 7, 2]).expect("test: operation should succeed");
        assert_eq!(logits.len(), 4);
        assert_eq!(logits[0].len(), 16);
    }
    #[test]
    fn test_selective_scan_step() {
        let scan = SelectiveScanCausal::new(4).expect("test: operation should succeed");
        let (y, new_h) = scan
            .step(1.0, &[0.0; 4], &[0.9; 4], &[0.1; 4], &[1.0; 4])
            .expect("test: operation should succeed");
        assert!(y.is_finite());
        assert_eq!(new_h.len(), 4);
        for &v in &new_h {
            assert!((v - 0.1).abs() < 1e-10);
        }
    }
    #[test]
    fn test_parallel_scan() {
        let y = ParallelScan::scan(&[0.9, 0.9, 0.9, 0.9], &[1.0, 1.0, 1.0, 1.0]).expect("test: operation should succeed");
        assert_eq!(y.len(), 4);
        assert!((y[0] - 1.0).abs() < 1e-10);
        assert!((y[1] - 1.9).abs() < 1e-10);
    }

    // Recurrence
    #[test]
    fn test_linear_recurrence_shape() {
        let layer = LinearRecurrenceLayer::new(4).expect("test: operation should succeed");
        let y = layer
            .forward(&[0.5f64; 12], &[0.0f64; 12], &[1.0f64; 12])
            .expect("test: operation should succeed");
        assert_eq!(y.len(), 12);
    }
    #[test]
    fn test_linear_recurrence_gating() {
        let layer = LinearRecurrenceLayer::new(1).expect("test: operation should succeed");
        let y = layer
            .forward(&[2.0, 3.0, 4.0], &[-1000.0; 3], &[1000.0; 3])
            .expect("test: operation should succeed");
        assert!((y[0] - 2.0).abs() < 1e-3);
        assert!((y[1] - 3.0).abs() < 1e-3);
    }
    #[test]
    fn test_hawk_block() {
        let block = HawkBlock::new(8, 42).expect("test: operation should succeed");
        let y = block.forward(&[0.1f64; 8]).expect("test: operation should succeed");
        assert_eq!(y.len(), 8);
        for v in &y {
            assert!(v.is_finite());
        }
    }
    #[test]
    fn test_recurrent_gemma() {
        let block = RecurrentGemmaBlock::new(8, 2, 42).expect("test: operation should succeed");
        let x: Vec<Vec<f64>> = (0..4).map(|_| vec![0.1f64; 8]).collect();
        let y = block.forward(&x, 3).expect("test: operation should succeed");
        assert_eq!(y.len(), 4);
        assert_eq!(y[0].len(), 8);
    }
    #[test]
    fn test_gated_recurrent_unit() {
        let gru = GatedRecurrentUnit::new(4, 8, 42).expect("test: operation should succeed");
        let new_h = gru.step(&[0.1f64; 4], &[0.0f64; 8]).expect("test: operation should succeed");
        assert_eq!(new_h.len(), 8);
        for v in &new_h {
            assert!(v.is_finite());
            assert!(v.abs() <= 1.0 + 1e-6);
        }
    }
    #[test]
    fn test_linearized_attention() {
        let attn = LinearizedSelfAttention::new(8, 2, 42).expect("test: operation should succeed");
        let y = attn
            .forward(&(0..32).map(|i| i as f64 * 0.01).collect::<Vec<_>>())
            .expect("test: operation should succeed");
        assert_eq!(y.len(), 32);
        for v in &y {
            assert!(v.is_finite());
        }
    }

    // Continuous-time
    #[test]
    fn test_ct_rnn_step() {
        let rnn = ContinuousTimeRnn::new(4, 2, 1.0, 42).expect("test: operation should succeed");
        let new_x = rnn.step(&[0.0f64; 4], &[0.1f64; 2], 0.01).expect("test: operation should succeed");
        assert_eq!(new_x.len(), 4);
        for v in &new_x {
            assert!(v.is_finite());
        }
    }
    #[test]
    fn test_ct_rnn_decay() {
        let rnn = ContinuousTimeRnn::new(2, 1, 1.0, 42).expect("test: operation should succeed");
        let mut x = vec![1.0f64, 1.0];
        for _ in 0..100 {
            x = rnn.step(&x, &[0.0f64], 0.1).expect("test: operation should succeed");
        }
        let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm < 2.0, "CT-RNN should stay bounded: norm={norm}");
    }
    #[test]
    fn test_neural_cde_shape() {
        let ncde = NeuralCde::new(4, 2, 42).expect("test: operation should succeed");
        let path: Vec<Vec<f64>> = (0..6)
            .map(|t| vec![t as f64 * 0.1, t as f64 * 0.05])
            .collect();
        let h = ncde.integrate(&[0.0f64; 4], &path, 0.1).expect("test: operation should succeed");
        assert_eq!(h.len(), 4);
        for v in &h {
            assert!(v.is_finite());
        }
    }
    #[test]
    fn test_liquid_neural_network() {
        let ltc = LiquidNeuralNetwork::new(4, 2, 1.0, 42).expect("test: operation should succeed");
        let new_h = ltc
            .step(&[0.5f64; 2], &[0.0f64; 4], &[0.0f64; 4], 0.01)
            .expect("test: operation should succeed");
        assert_eq!(new_h.len(), 4);
        for v in &new_h {
            assert!(v.is_finite());
        }
    }
    #[test]
    fn test_wired_ct_rnn() {
        let rnn = WiredCtRnn::new(4, 2, 2, 1.0, 42).expect("test: operation should succeed");
        let x: Vec<Vec<f64>> = (0..3).map(|_| vec![0.1f64; 2]).collect();
        let mask: Vec<Vec<f64>> = (0..4).map(|_| vec![1.0f64; 4]).collect();
        let y = rnn.forward(&x, &mask).expect("test: operation should succeed");
        assert_eq!(y.len(), 3);
        assert_eq!(y[0].len(), 2);
    }
    #[test]
    fn test_spiking_threshold() {
        let snn = SpikingNeuralNetwork::new(4, 10.0, 1.0).expect("test: operation should succeed");
        let (new_v, spikes) = snn.step(&[0.0f64; 4], &[100.0f64; 4], 0.1).expect("test: operation should succeed");
        for &s in &spikes {
            assert!(s, "should spike");
        }
        for &v in &new_v {
            assert!((v - snn.v_reset).abs() < 1e-10);
        }
    }
    #[test]
    fn test_lif_reset() {
        let snn = SpikingNeuralNetwork::new(2, 5.0, 1.0).expect("test: operation should succeed");
        let (new_v, spikes) = snn.step(&[0.5f64; 2], &[0.0f64; 2], 0.1).expect("test: operation should succeed");
        for &s in &spikes {
            assert!(!s, "no spike");
        }
        for (&nv, _) in new_v.iter().zip([0.5f64; 2].iter()) {
            assert!(nv < snn.threshold);
        }
    }

    // Hybrid
    #[test]
    fn test_transformer_ssm_hybrid() {
        let hybrid = TransformerSsmHybrid::new(8, 1, 1, 4, 42).expect("test: operation should succeed");
        let x: Vec<Vec<f64>> = (0..4).map(|_| vec![0.1f64; 8]).collect();
        let y = hybrid.forward(&x).expect("test: operation should succeed");
        assert_eq!(y.len(), 4);
        assert_eq!(y[0].len(), 8);
    }
    #[test]
    fn test_jamba_block_shape() {
        let jamba = JambaBlock::new(8, 4, 2, 42).expect("test: operation should succeed");
        let x: Vec<Vec<f64>> = (0..3).map(|_| vec![0.1f64; 8]).collect();
        let y = jamba.forward(&x).expect("test: operation should succeed");
        assert_eq!(y.len(), 3);
        assert_eq!(y[0].len(), 8);
    }
    #[test]
    fn test_zamba_block_shape() {
        let zamba = ZambaBlock::new(8, 4, 2, 42).expect("test: operation should succeed");
        let x: Vec<Vec<f64>> = (0..4).map(|_| vec![0.1f64; 8]).collect();
        let y = zamba.forward(&x).expect("test: operation should succeed");
        assert_eq!(y.len(), 4);
        assert_eq!(y[0].len(), 8);
    }
    #[test]
    fn test_ssm_evaluator_ratio() {
        let (ops, mem) = SsmEvaluator::compare_recurrence_vs_attention(128);
        assert!(ops < 1.0, "SSM ops_ratio={ops} < 1 for seq_len=128");
        assert!(mem < 1.0, "SSM mem_ratio={mem} < 1 for seq_len=128");
    }

    // Additional coverage
    #[test]
    fn test_parallel_scan_single() {
        let y = ParallelScan::scan(&[0.5], &[2.0]).expect("test: operation should succeed");
        assert!((y[0] - 2.0).abs() < 1e-10);
    }
    #[test]
    fn test_selective_scan_accumulation() {
        let scan = SelectiveScanCausal::new(2).expect("test: operation should succeed");
        let mut h = vec![0.0f64; 2];
        for _ in 0..5 {
            let (_, nh) = scan.step(1.0, &h, &[1.0; 2], &[1.0; 2], &[1.0; 2]).expect("test: operation should succeed");
            h = nh;
        }
        assert!(h.iter().all(|&v| v > 0.0), "accumulating state > 0");
    }
    #[test]
    fn test_mamba_model_token_clamping() {
        let model = MambaModel::new(4, 4, 2, 1, 7).expect("test: operation should succeed");
        let logits = model.forward(&[0, 100, 999]).expect("test: operation should succeed");
        assert_eq!(logits.len(), 3);
    }
    #[test]
    fn test_liquid_network_multiple_steps() {
        let ltc = LiquidNeuralNetwork::new(3, 2, 0.5, 13).expect("test: operation should succeed");
        let mut h = vec![0.0f64; 3];
        for _ in 0..10 {
            h = ltc.step(&[0.1, -0.1], &[0.0; 3], &h, 0.05).expect("test: operation should succeed");
        }
        for v in &h {
            assert!(v.is_finite());
        }
    }
    #[test]
    fn test_snn_subthreshold() {
        let snn = SpikingNeuralNetwork::new(3, 20.0, 2.0).expect("test: operation should succeed");
        let (new_v, spikes) = snn.step(&[0.0; 3], &[0.1; 3], 0.1).expect("test: operation should succeed");
        for &s in &spikes {
            assert!(!s);
        }
        for &v in &new_v {
            assert!(v < snn.threshold);
        }
    }
    #[test]
    fn test_falcon_mamba_block() {
        let falcon = FalconMambaBlock::new(8, 4, 42).expect("test: operation should succeed");
        let x: Vec<Vec<f64>> = (0..3).map(|_| vec![0.1f64; 8]).collect();
        let y = falcon.forward(&x).expect("test: operation should succeed");
        assert_eq!(y.len(), 3);
        assert_eq!(y[0].len(), 8);
    }
    #[test]
    fn test_ssm_evaluator_scaling() {
        let (_, m_small) = SsmEvaluator::compare_recurrence_vs_attention(64);
        let (_, m_large) = SsmEvaluator::compare_recurrence_vs_attention(1024);
        assert!(m_large < m_small, "SSM advantage grows with seq_len");
    }
}
