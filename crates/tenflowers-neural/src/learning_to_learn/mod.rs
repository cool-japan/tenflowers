//! Learning-to-Learn / Meta-Optimizer Module
//!
//! Implements LSTM-based meta-optimizers (Ravi & Larochelle 2017,
//! Andrychowicz et al. 2016), SNAIL (Mishra 2018), gradient preprocessors,
//! warm-start, learned LR scheduling, and meta-training utilities.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::fmt;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors from learning-to-learn operations.
#[derive(Debug)]
pub enum L2lError {
    /// Invalid configuration parameter.
    InvalidConfig { field: String, reason: String },
    /// Gradient computation error.
    GradientError { context: String },
    /// Optimization loop failed.
    OptimizationFailed { reason: String },
}

impl fmt::Display for L2lError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            L2lError::InvalidConfig { field, reason } => {
                write!(f, "invalid config field '{}': {}", field, reason)
            }
            L2lError::GradientError { context } => {
                write!(f, "gradient error: {}", context)
            }
            L2lError::OptimizationFailed { reason } => {
                write!(f, "optimization failed: {}", reason)
            }
        }
    }
}

impl std::error::Error for L2lError {}

type L2lResult<T> = Result<T, L2lError>;

// ─── L2lTensor ────────────────────────────────────────────────────────────────

/// Flat-vector parameter tensor with associated gradient storage.
#[derive(Debug, Clone)]
pub struct L2lTensor {
    /// Parameter values.
    pub data: Vec<f64>,
    /// Gradient values (same shape as data).
    pub grad: Vec<f64>,
}

impl L2lTensor {
    /// Create a new L2lTensor with given data; gradients initialised to zero.
    pub fn new(data: Vec<f64>) -> Self {
        let n = data.len();
        Self {
            data,
            grad: vec![0.0; n],
        }
    }

    /// Create a zero tensor of the given size.
    pub fn zeros(size: usize) -> Self {
        Self {
            data: vec![0.0; size],
            grad: vec![0.0; size],
        }
    }

    /// Create a tensor of the same size filled with zeros.
    pub fn zeros_like(&self) -> Self {
        Self::zeros(self.data.len())
    }

    /// Create a tensor with random values drawn from N(0,1) via Box-Muller.
    pub fn from_random(size: usize, rng: &mut StdRng) -> Self {
        let data = (0..size)
            .map(|_| {
                let u1: f64 = rng.random::<f64>().max(1e-15);
                let u2: f64 = rng.random::<f64>();
                let mag = (-2.0 * u1.ln()).sqrt();
                mag * (2.0 * std::f64::consts::PI * u2).cos()
            })
            .collect();
        Self::new(data)
    }

    /// L2 norm of data.
    pub fn l2_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Dot product with another tensor (data only).
    pub fn dot(&self, other: &L2lTensor) -> f64 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// AXPY: self.data += alpha * other.data
    pub fn axpy(&mut self, alpha: f64, other: &L2lTensor) {
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += alpha * b;
        }
    }

    /// Clip gradient norm to max_norm in place.
    pub fn clip_norm(&mut self, max_norm: f64) {
        let norm: f64 = self.grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if norm > max_norm && norm > 1e-12 {
            let scale = max_norm / norm;
            for g in self.grad.iter_mut() {
                *g *= scale;
            }
        }
    }
}

// ─── Xavier init helper ───────────────────────────────────────────────────────

/// Xavier uniform initializer for a weight matrix [rows×cols].
fn xavier_init(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<f64> {
    let limit = (6.0_f64 / (rows + cols) as f64).sqrt();
    (0..rows * cols)
        .map(|_| rng.random_range(-limit..limit))
        .collect()
}

/// Sigmoid activation.
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Tanh activation (std).
#[inline]
fn tanh_act(x: f64) -> f64 {
    x.tanh()
}

// ─── L2lLstmCell ──────────────────────────────────────────────────────────────

/// LSTM cell for the meta-optimizer network.
///
/// 4-gate formulation (i, f, g, o) with:
/// - input dimension = input_size
/// - hidden dimension = hidden_size
/// - output update_vector of size hidden_size
#[derive(Debug, Clone)]
pub struct L2lLstmCell {
    /// Input size.
    pub input_size: usize,
    /// Hidden size.
    pub hidden_size: usize,
    // Gate weight matrices: [hidden_size × (input_size + hidden_size)] for i,f,g,o
    wi: Vec<f64>,
    wf: Vec<f64>,
    wg: Vec<f64>,
    wo: Vec<f64>,
    // Gate biases: [hidden_size]
    bi: Vec<f64>,
    bf: Vec<f64>,
    bg: Vec<f64>,
    bo: Vec<f64>,
    // Output projection: [update_dim × hidden_size]
    w_out: Vec<f64>,
    b_out: Vec<f64>,
    /// Dimension of the produced update vector.
    pub update_dim: usize,
}

impl L2lLstmCell {
    /// Create a new LSTM cell. Xavier initialised.
    pub fn new(input_size: usize, hidden_size: usize, update_dim: usize, rng: &mut StdRng) -> Self {
        let xh = input_size + hidden_size;
        let wi = xavier_init(hidden_size, xh, rng);
        let wf = xavier_init(hidden_size, xh, rng);
        let wg = xavier_init(hidden_size, xh, rng);
        let wo = xavier_init(hidden_size, xh, rng);
        let bi = vec![0.0; hidden_size];
        let mut bf = vec![0.0; hidden_size]; // forget gate bias = 1 (Jozefowicz 2015)
        for b in bf.iter_mut() {
            *b = 1.0;
        }
        let bg = vec![0.0; hidden_size];
        let bo = vec![0.0; hidden_size];
        let w_out = xavier_init(update_dim, hidden_size, rng);
        let b_out = vec![0.0; update_dim];
        Self {
            input_size,
            hidden_size,
            wi,
            wf,
            wg,
            wo,
            bi,
            bf,
            bg,
            bo,
            w_out,
            b_out,
            update_dim,
        }
    }

    /// Apply one LSTM step.
    /// Returns (h_new, c_new, update_vector).
    pub fn step(
        &self,
        input: &[f64],
        h: &[f64],
        c: &[f64],
    ) -> L2lResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        if input.len() != self.input_size {
            return Err(L2lError::GradientError {
                context: format!("input len {} != {}", input.len(), self.input_size),
            });
        }
        // Concatenate input and h
        let xh_dim = self.input_size + self.hidden_size;
        let mut xh = Vec::with_capacity(xh_dim);
        xh.extend_from_slice(input);
        xh.extend_from_slice(h);

        // Helper: compute gate = activation(W * xh + b)
        let gate = |w: &[f64], b: &[f64], act: fn(f64) -> f64| -> Vec<f64> {
            (0..self.hidden_size)
                .map(|i| {
                    let row_start = i * xh_dim;
                    let pre: f64 = w[row_start..row_start + xh_dim]
                        .iter()
                        .zip(xh.iter())
                        .map(|(wi, xi)| wi * xi)
                        .sum::<f64>()
                        + b[i];
                    act(pre)
                })
                .collect()
        };

        let i_gate = gate(&self.wi, &self.bi, sigmoid);
        let f_gate = gate(&self.wf, &self.bf, sigmoid);
        let g_gate = gate(&self.wg, &self.bg, tanh_act);
        let o_gate = gate(&self.wo, &self.bo, sigmoid);

        // c_new = f * c + i * g
        let c_new: Vec<f64> = (0..self.hidden_size)
            .map(|j| f_gate[j] * c[j] + i_gate[j] * g_gate[j])
            .collect();

        // h_new = o * tanh(c_new)
        let h_new: Vec<f64> = (0..self.hidden_size)
            .map(|j| o_gate[j] * tanh_act(c_new[j]))
            .collect();

        // update_vector = W_out * h_new + b_out
        let update: Vec<f64> = (0..self.update_dim)
            .map(|i| {
                let row_start = i * self.hidden_size;
                self.w_out[row_start..row_start + self.hidden_size]
                    .iter()
                    .zip(h_new.iter())
                    .map(|(w, hv)| w * hv)
                    .sum::<f64>()
                    + self.b_out[i]
            })
            .collect();

        Ok((h_new, c_new, update))
    }

    /// Return all learnable parameters flattened (for meta-gradient updates).
    pub fn parameters(&self) -> Vec<f64> {
        let mut p = Vec::new();
        p.extend_from_slice(&self.wi);
        p.extend_from_slice(&self.wf);
        p.extend_from_slice(&self.wg);
        p.extend_from_slice(&self.wo);
        p.extend_from_slice(&self.bi);
        p.extend_from_slice(&self.bf);
        p.extend_from_slice(&self.bg);
        p.extend_from_slice(&self.bo);
        p.extend_from_slice(&self.w_out);
        p.extend_from_slice(&self.b_out);
        p
    }

    /// Apply a flat parameter update (additive).
    pub fn apply_update(&mut self, delta: &[f64]) {
        let mut offset = 0;
        macro_rules! update_slice {
            ($field:expr) => {
                for v in $field.iter_mut() {
                    if offset < delta.len() {
                        *v += delta[offset];
                        offset += 1;
                    }
                }
            };
        }
        update_slice!(self.wi);
        update_slice!(self.wf);
        update_slice!(self.wg);
        update_slice!(self.wo);
        update_slice!(self.bi);
        update_slice!(self.bf);
        update_slice!(self.bg);
        update_slice!(self.bo);
        update_slice!(self.w_out);
        update_slice!(self.b_out);
    }
}

// ─── GradientPreprocessor ─────────────────────────────────────────────────────

/// Andrychowicz-style gradient preprocessing.
///
/// For each gradient element g produces a 2-element feature vector:
/// - log(|g| + ε) / log(10) clipped to [-1, +1]
/// - sign(g)
#[derive(Debug, Clone)]
pub struct GradientPreprocessor {
    /// Small epsilon for log stability.
    pub epsilon: f64,
    /// Clip bound for log feature.
    pub clip_val: f64,
}

impl GradientPreprocessor {
    /// Create preprocessor with default epsilon=1e-8, clip=1.0.
    pub fn new() -> Self {
        Self {
            epsilon: 1e-8,
            clip_val: 1.0,
        }
    }

    /// Preprocess gradient vector → 2D features per element, length = 2*n.
    pub fn preprocess(&self, grad: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(2 * grad.len());
        let log10 = 10.0_f64.ln();
        for &g in grad {
            let log_feat =
                ((g.abs() + self.epsilon).ln() / log10).clamp(-self.clip_val, self.clip_val);
            let sign_feat = if g > 0.0 {
                1.0
            } else if g < 0.0 {
                -1.0
            } else {
                0.0
            };
            out.push(log_feat);
            out.push(sign_feat);
        }
        out
    }

    /// Recover update scale: multiply element-wise by |grad| + epsilon.
    pub fn unpreprocess(&self, update: &[f64], grad: &[f64]) -> Vec<f64> {
        update
            .iter()
            .zip(grad.iter())
            .map(|(u, g)| u * (g.abs() + self.epsilon))
            .collect()
    }
}

impl Default for GradientPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

// ─── LstmMetaOptimizer ────────────────────────────────────────────────────────

/// Task descriptor for meta-learning.
#[derive(Debug, Clone)]
pub struct L2lTask {
    /// Support-set inputs [n_support × input_dim].
    pub support_x: Vec<Vec<f64>>,
    /// Support-set targets.
    pub support_y: Vec<f64>,
    /// Query-set inputs [n_query × input_dim].
    pub query_x: Vec<Vec<f64>>,
    /// Query-set targets.
    pub query_y: Vec<f64>,
}

/// Hidden state for the LSTM meta-optimizer (one cell per parameter).
#[derive(Debug, Clone)]
pub struct L2lHiddenState {
    pub h: Vec<f64>,
    pub c: Vec<f64>,
}

impl L2lHiddenState {
    pub fn zeros(hidden_size: usize) -> Self {
        Self {
            h: vec![0.0; hidden_size],
            c: vec![0.0; hidden_size],
        }
    }
}

/// Ravi & Larochelle (2017) LSTM meta-optimizer.
///
/// One weight-shared `L2lLstmCell` drives updates for all parameters.
/// Coordinate-wise preprocessing: input = [g_feat1, g_feat2, loss, step].
#[derive(Debug, Clone)]
pub struct LstmMetaOptimizer {
    /// Shared LSTM cell.
    pub cell: L2lLstmCell,
    /// Gradient preprocessor.
    pub preprocessor: GradientPreprocessor,
    /// Meta-learning rate for outer loop.
    pub meta_lr: f64,
    /// Inner loop step count.
    pub inner_steps: usize,
}

impl LstmMetaOptimizer {
    /// Create a new LSTM meta-optimizer.
    /// input_size = 4 (2 gradient features + loss + step).
    pub fn new(hidden_size: usize, inner_steps: usize, meta_lr: f64, rng: &mut StdRng) -> Self {
        let input_size = 4; // [log|g|, sign(g), loss_norm, step_norm]
        let cell = L2lLstmCell::new(input_size, hidden_size, 1, rng);
        Self {
            cell,
            preprocessor: GradientPreprocessor::new(),
            meta_lr,
            inner_steps,
        }
    }

    /// Compute update for a single parameter.
    /// Returns (delta_theta, new_h, new_c).
    pub fn compute_update(
        &self,
        grad_scalar: f64,
        loss: f64,
        step: f64,
        h: &[f64],
        c: &[f64],
    ) -> L2lResult<(f64, Vec<f64>, Vec<f64>)> {
        let log10 = 10.0_f64.ln();
        let log_g = ((grad_scalar.abs() + 1e-8).ln() / log10).clamp(-1.0, 1.0);
        let sign_g = if grad_scalar > 0.0 {
            1.0
        } else if grad_scalar < 0.0 {
            -1.0
        } else {
            0.0
        };
        let input = [
            log_g,
            sign_g,
            loss.tanh(),
            step / (self.inner_steps as f64 + 1.0),
        ];
        let (h_new, c_new, update) = self.cell.step(&input, h, c)?;
        Ok((update[0], h_new, c_new))
    }

    /// Run inner adaptation on a task's support set.
    /// Returns final adapted parameters and final query loss.
    pub fn adapt_on_task(&self, task: &L2lTask, init_params: &[f64]) -> L2lResult<(Vec<f64>, f64)> {
        let n_params = init_params.len();
        let hidden_size = self.cell.hidden_size;
        let mut params = init_params.to_vec();
        let mut hidden_states: Vec<L2lHiddenState> = (0..n_params)
            .map(|_| L2lHiddenState::zeros(hidden_size))
            .collect();

        for step in 0..self.inner_steps {
            // Compute MSE loss and per-parameter gradient on support set
            let (loss, grad) = compute_linear_mse_grad(&params, &task.support_x, &task.support_y);
            let step_f = step as f64;
            let loss_norm = loss / (task.support_y.len() as f64 + 1e-8);

            for (p_idx, (param, g)) in params.iter_mut().zip(grad.iter()).enumerate() {
                let hs = &hidden_states[p_idx];
                let (delta, h_new, c_new) =
                    self.compute_update(*g, loss_norm, step_f, &hs.h, &hs.c)?;
                *param += delta;
                hidden_states[p_idx] = L2lHiddenState { h: h_new, c: c_new };
            }
        }

        let (query_loss, _) = compute_linear_mse_grad(&params, &task.query_x, &task.query_y);
        Ok((params, query_loss / (task.query_y.len() as f64 + 1e-8)))
    }

    /// Meta-train the LSTM cell over a set of tasks using FOMAML-style updates.
    pub fn meta_train(
        &mut self,
        tasks: &[L2lTask],
        n_outer: usize,
        n_params: usize,
        rng: &mut StdRng,
    ) -> L2lResult<Vec<f64>> {
        if tasks.is_empty() {
            return Err(L2lError::InvalidConfig {
                field: "tasks".into(),
                reason: "no tasks provided".into(),
            });
        }

        let mut meta_losses = Vec::with_capacity(n_outer);

        // Adam state for meta-optimizer cell parameters
        let param_count = self.cell.parameters().len();
        let mut m = vec![0.0f64; param_count];
        let mut v = vec![0.0f64; param_count];
        let beta1 = 0.9_f64;
        let beta2 = 0.999_f64;
        let eps_adam = 1e-8_f64;

        for outer in 0..n_outer {
            let task_idx = rng.random_range(0..tasks.len());
            let task = &tasks[task_idx];

            let init_params: Vec<f64> = (0..n_params)
                .map(|_| rng.random::<f64>() * 0.1 - 0.05)
                .collect();

            let (_adapted, query_loss) = self.adapt_on_task(task, &init_params)?;
            meta_losses.push(query_loss);

            // Approximate meta-gradient as finite differences on cell params
            let cell_params = self.cell.parameters();
            let mut meta_grad = vec![0.0f64; cell_params.len()];
            let fd_eps = 1e-4;

            // Sample a small subset of cell params to finite-diff (for efficiency)
            let n_sample = cell_params.len().clamp(1, 20);
            let step_size = cell_params.len() / n_sample;
            for k in 0..n_sample {
                let idx = k * step_size;
                let mut cell_plus = self.cell.clone();
                let mut delta_p = vec![0.0f64; cell_params.len()];
                delta_p[idx] = fd_eps;
                cell_plus.apply_update(&delta_p);

                let mut opt_plus = self.clone();
                opt_plus.cell = cell_plus;
                let (_, loss_plus) = opt_plus.adapt_on_task(task, &init_params)?;

                meta_grad[idx] = (loss_plus - query_loss) / fd_eps;
            }

            // Adam update
            let t = (outer + 1) as f64;
            for (j, (mj, vj)) in m.iter_mut().zip(v.iter_mut()).enumerate() {
                *mj = beta1 * *mj + (1.0 - beta1) * meta_grad[j];
                *vj = beta2 * *vj + (1.0 - beta2) * meta_grad[j] * meta_grad[j];
                let m_hat = *mj / (1.0 - beta1.powf(t));
                let v_hat = *vj / (1.0 - beta2.powf(t));
                meta_grad[j] = self.meta_lr * m_hat / (v_hat.sqrt() + eps_adam);
            }

            // Negate for gradient descent
            let update: Vec<f64> = meta_grad.iter().map(|g| -g).collect();
            self.cell.apply_update(&update);
        }

        Ok(meta_losses)
    }
}

// ─── OptimizerNetwork ─────────────────────────────────────────────────────────

/// Andrychowicz (2016) "Learning to learn by gradient descent by gradient descent" network.
///
/// Input = concat(∇L_t, ∇L_{t-1}) after preprocessing.
/// Per-parameter hidden states.
#[derive(Debug, Clone)]
pub struct OptimizerNetwork {
    /// Shared LSTM cell; input_size = 4 (two preprocessed grads, 2 features each).
    pub cell: L2lLstmCell,
    /// Preprocessor for gradients.
    pub preprocessor: GradientPreprocessor,
    /// Per-parameter hidden states.
    hidden_states: Vec<L2lHiddenState>,
    /// Previous gradients (one per parameter).
    prev_grad: Vec<f64>,
    /// Number of parameters tracked.
    pub n_params: usize,
}

impl OptimizerNetwork {
    /// Create a new optimizer network for `n_params` parameters.
    pub fn new(n_params: usize, hidden_size: usize, rng: &mut StdRng) -> Self {
        // input_size = 4: two gradient features for current + two for previous
        let cell = L2lLstmCell::new(4, hidden_size, 1, rng);
        let hidden_states = (0..n_params)
            .map(|_| L2lHiddenState::zeros(hidden_size))
            .collect();
        Self {
            cell,
            preprocessor: GradientPreprocessor::new(),
            hidden_states,
            prev_grad: vec![0.0; n_params],
            n_params,
        }
    }

    /// Reset per-parameter hidden states for a new task.
    pub fn reset_state(&mut self) {
        let hidden_size = self.cell.hidden_size;
        self.hidden_states = (0..self.n_params)
            .map(|_| L2lHiddenState::zeros(hidden_size))
            .collect();
        self.prev_grad = vec![0.0; self.n_params];
    }

    /// Compute update direction for all parameters.
    /// Returns Vec of per-parameter update scalars.
    pub fn compute_updates(&mut self, grad: &[f64]) -> L2lResult<Vec<f64>> {
        if grad.len() != self.n_params {
            return Err(L2lError::GradientError {
                context: format!("grad len {} != n_params {}", grad.len(), self.n_params),
            });
        }

        let mut updates = Vec::with_capacity(self.n_params);

        for i in 0..self.n_params {
            let g_curr = grad[i];
            let g_prev = self.prev_grad[i];

            let log10 = 10.0_f64.ln();
            let log_g = ((g_curr.abs() + 1e-8).ln() / log10).clamp(-1.0, 1.0);
            let sign_g = if g_curr > 0.0 {
                1.0
            } else if g_curr < 0.0 {
                -1.0
            } else {
                0.0
            };
            let log_gp = ((g_prev.abs() + 1e-8).ln() / log10).clamp(-1.0, 1.0);
            let sign_gp = if g_prev > 0.0 {
                1.0
            } else if g_prev < 0.0 {
                -1.0
            } else {
                0.0
            };

            let input = [log_g, sign_g, log_gp, sign_gp];
            let hs = &self.hidden_states[i];
            let (h_new, c_new, update_vec) = self.cell.step(&input, &hs.h, &hs.c)?;

            self.hidden_states[i] = L2lHiddenState { h: h_new, c: c_new };
            updates.push(update_vec[0]);
        }

        self.prev_grad = grad.to_vec();
        Ok(updates)
    }
}

// ─── SNAIL ────────────────────────────────────────────────────────────────────

/// Temporal convolutional block for SNAIL.
///
/// Causal dilated conv1d with kernel=2, dilation=2^layer_idx.
/// Input shape: [T, D_in], output shape: [T, D_out].
#[derive(Debug, Clone)]
pub struct L2lTcBlock {
    /// Input feature dim.
    pub d_in: usize,
    /// Output feature dim.
    pub d_out: usize,
    /// Dilation factor.
    pub dilation: usize,
    // Weight [D_out × 2 × D_in] (kernel_size=2)
    weight: Vec<f64>,
    bias: Vec<f64>,
}

impl L2lTcBlock {
    /// Create a causal TC block. layer_idx determines dilation = 2^layer_idx.
    pub fn new(d_in: usize, d_out: usize, layer_idx: usize, rng: &mut StdRng) -> Self {
        let dilation = 1 << layer_idx; // 2^layer_idx
        let weight = xavier_init(d_out, 2 * d_in, rng);
        let bias = vec![0.0; d_out];
        Self {
            d_in,
            d_out,
            dilation,
            weight,
            bias,
        }
    }

    /// Forward pass over a sequence.
    /// Input: &[[f64; D_in]] length T.
    /// Output: Vec<`Vec<f64>>` length T, width D_out.
    pub fn forward(&self, x: &[Vec<f64>]) -> L2lResult<Vec<Vec<f64>>> {
        if x.is_empty() {
            return Err(L2lError::GradientError {
                context: "empty input sequence".into(),
            });
        }
        let t_len = x.len();
        let mut out = Vec::with_capacity(t_len);

        for t in 0..t_len {
            // Causal: look back by dilation steps; pad with zeros if before start
            let prev_t = t.checked_sub(self.dilation);

            let mut y = vec![0.0f64; self.d_out];
            for o in 0..self.d_out {
                let row_curr = o * 2 * self.d_in; // kernel pos 0 (current t)
                let row_prev = o * 2 * self.d_in + self.d_in; // kernel pos 1 (prev t)

                // Current time step
                let curr_in = &x[t];
                for (k, &xi) in curr_in.iter().enumerate() {
                    y[o] += self.weight[row_curr + k] * xi;
                }
                // Previous (dilated) step
                if let Some(pt) = prev_t {
                    let prev_in = &x[pt];
                    for (k, &xi) in prev_in.iter().enumerate() {
                        y[o] += self.weight[row_prev + k] * xi;
                    }
                }
                y[o] += self.bias[o];
                // ReLU activation
                if y[o] < 0.0 {
                    y[o] = 0.0;
                }
            }
            out.push(y);
        }
        Ok(out)
    }
}

/// Attention block for SNAIL.
///
/// Single-head key-value memory attention where query = last timestep.
#[derive(Debug, Clone)]
pub struct L2lAttnBlock {
    /// Input / output dimension.
    pub dim: usize,
    /// Key/value projection weight [dim × dim].
    w_key: Vec<f64>,
    w_val: Vec<f64>,
    w_query: Vec<f64>,
    /// Projection for combined output [dim × 2*dim] (concat attn + skip).
    w_out: Vec<f64>,
    b_out: Vec<f64>,
}

impl L2lAttnBlock {
    /// Create an attention block.
    pub fn new(dim: usize, rng: &mut StdRng) -> Self {
        Self {
            dim,
            w_key: xavier_init(dim, dim, rng),
            w_val: xavier_init(dim, dim, rng),
            w_query: xavier_init(dim, dim, rng),
            w_out: xavier_init(dim, 2 * dim, rng),
            b_out: vec![0.0; dim],
        }
    }

    /// Linear projection: [out_dim × in_dim] * x.
    fn proj(w: &[f64], x: &[f64], out_dim: usize, in_dim: usize) -> Vec<f64> {
        (0..out_dim)
            .map(|i| {
                w[i * in_dim..(i + 1) * in_dim]
                    .iter()
                    .zip(x.iter())
                    .map(|(wi, xi)| wi * xi)
                    .sum::<f64>()
            })
            .collect()
    }

    /// Forward: causal attention over sequence.
    /// Input: [T, dim], Output: [T, dim].
    pub fn forward(&self, x: &[Vec<f64>]) -> L2lResult<Vec<Vec<f64>>> {
        if x.is_empty() {
            return Err(L2lError::GradientError {
                context: "empty input to attention block".into(),
            });
        }
        let t_len = x.len();
        let scale = (self.dim as f64).sqrt().max(1e-8);
        let mut out = Vec::with_capacity(t_len);

        for t in 0..t_len {
            // Query from current timestep
            let q = Self::proj(&self.w_query, &x[t], self.dim, self.dim);

            // Keys and values from all past (causal) positions including t
            let attn_len = t + 1;
            let mut scores = Vec::with_capacity(attn_len);
            let mut values = Vec::with_capacity(attn_len);

            for s in 0..attn_len {
                let k = Self::proj(&self.w_key, &x[s], self.dim, self.dim);
                let v = Self::proj(&self.w_val, &x[s], self.dim, self.dim);
                let score: f64 =
                    q.iter().zip(k.iter()).map(|(qi, ki)| qi * ki).sum::<f64>() / scale;
                scores.push(score);
                values.push(v);
            }

            // Softmax over scores
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exps.iter().sum::<f64>().max(1e-30);
            let weights: Vec<f64> = exps.iter().map(|e| e / sum_exp).collect();

            // Weighted sum of values
            let mut attn_out = vec![0.0f64; self.dim];
            for (w, v) in weights.iter().zip(values.iter()) {
                for (a, vi) in attn_out.iter_mut().zip(v.iter()) {
                    *a += w * vi;
                }
            }

            // Concat attn_out and x[t], project to dim
            let mut concat = Vec::with_capacity(2 * self.dim);
            concat.extend_from_slice(&attn_out);
            concat.extend_from_slice(&x[t]);

            let mut y = Self::proj(&self.w_out, &concat, self.dim, 2 * self.dim);
            for (yi, bi) in y.iter_mut().zip(self.b_out.iter()) {
                *yi += bi;
            }
            out.push(y);
        }
        Ok(out)
    }
}

/// SNAIL (Mishra 2018): Simple Neural AttentIve Learner.
///
/// Alternating L2lTcBlock (dilated causal conv) and L2lAttnBlock blocks.
#[derive(Debug, Clone)]
pub struct SnailModel {
    /// Input dimension.
    pub input_dim: usize,
    /// Channel dimension throughout.
    pub channel_dim: usize,
    /// TC blocks.
    tc_blocks: Vec<L2lTcBlock>,
    /// Attention blocks.
    attn_blocks: Vec<L2lAttnBlock>,
    /// Output projection [output_dim × channel_dim].
    w_out: Vec<f64>,
    b_out: Vec<f64>,
    /// Output dimension.
    pub output_dim: usize,
}

impl SnailModel {
    /// Create a SNAIL model.
    /// n_tc: number of TC/attn pairs, channel_dim: hidden width.
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        channel_dim: usize,
        n_pairs: usize,
        rng: &mut StdRng,
    ) -> L2lResult<Self> {
        if n_pairs == 0 {
            return Err(L2lError::InvalidConfig {
                field: "n_pairs".into(),
                reason: "must be >= 1".into(),
            });
        }
        let mut tc_blocks = Vec::with_capacity(n_pairs);
        let mut attn_blocks = Vec::with_capacity(n_pairs);

        // First TC block takes input_dim, rest take channel_dim
        tc_blocks.push(L2lTcBlock::new(input_dim, channel_dim, 0, rng));
        attn_blocks.push(L2lAttnBlock::new(channel_dim, rng));

        for pair in 1..n_pairs {
            tc_blocks.push(L2lTcBlock::new(channel_dim, channel_dim, pair, rng));
            attn_blocks.push(L2lAttnBlock::new(channel_dim, rng));
        }

        let w_out = xavier_init(output_dim, channel_dim, rng);
        let b_out = vec![0.0; output_dim];

        Ok(Self {
            input_dim,
            channel_dim,
            tc_blocks,
            attn_blocks,
            w_out,
            b_out,
            output_dim,
        })
    }

    /// Forward pass over context + query.
    /// context_xy: [(x, y)] support examples, query_x: single query.
    /// Returns prediction vector of length output_dim.
    pub fn forward_sequence(
        &self,
        context_xy: &[(Vec<f64>, f64)],
        query_x: &[f64],
    ) -> L2lResult<Vec<f64>> {
        // Build input sequence: each element = concat(x, y) for support, concat(x, 0) for query
        let seq_len = context_xy.len() + 1;
        let feat_dim = query_x.len() + 1; // x + y scalar
        let mut seq = Vec::with_capacity(seq_len);

        for (x, y) in context_xy {
            let mut feat = x.clone();
            feat.push(*y);
            seq.push(feat);
        }
        // Query: append 0 as dummy label
        let mut q_feat = query_x.to_vec();
        q_feat.push(0.0);
        seq.push(q_feat);

        // Pad/project input to channel_dim via first TC block (expects input_dim)
        // Ensure feat_dim matches input_dim; truncate or zero-pad
        let input_dim = self.input_dim;
        let aligned: Vec<Vec<f64>> = seq
            .iter()
            .map(|f| {
                let mut v = f.clone();
                v.resize(input_dim, 0.0);
                v
            })
            .collect();

        // Pass through alternating TC + Attention blocks
        let mut h = aligned;
        for (tc, attn) in self.tc_blocks.iter().zip(self.attn_blocks.iter()) {
            h = tc.forward(&h)?;
            h = attn.forward(&h)?;
        }

        // Take last timestep and project to output_dim
        let last = h.last().ok_or_else(|| L2lError::OptimizationFailed {
            reason: "empty sequence after SNAIL forward".into(),
        })?;

        let pred: Vec<f64> = (0..self.output_dim)
            .map(|i| {
                self.w_out[i * self.channel_dim..(i + 1) * self.channel_dim]
                    .iter()
                    .zip(last.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
                    + self.b_out[i]
            })
            .collect();

        Ok(pred)
    }
}

// ─── MetaDataset ──────────────────────────────────────────────────────────────

/// Task type in the meta-dataset.
#[derive(Debug, Clone, Copy)]
pub enum MetaTaskType {
    SineRegression,
    LinearClassification,
}

/// A sampled meta-learning task.
#[derive(Debug, Clone)]
pub struct MetaSampledTask {
    pub support_x: Vec<Vec<f64>>,
    pub support_y: Vec<f64>,
    pub query_x: Vec<Vec<f64>>,
    pub query_y: Vec<f64>,
    pub task_type: MetaTaskType,
}

/// Distribution over tasks for meta-learning experiments.
#[derive(Debug, Clone)]
pub struct MetaDataset {
    /// Task type to sample from.
    pub task_type: MetaTaskType,
    /// Input dimension.
    pub input_dim: usize,
}

impl MetaDataset {
    /// Create a sine regression task distribution.
    pub fn sine_regression(input_dim: usize) -> Self {
        Self {
            task_type: MetaTaskType::SineRegression,
            input_dim,
        }
    }

    /// Create a linear classification task distribution.
    pub fn linear_classification(input_dim: usize) -> Self {
        Self {
            task_type: MetaTaskType::LinearClassification,
            input_dim,
        }
    }

    /// Sample one task.
    pub fn sample_task(
        &self,
        n_support: usize,
        n_query: usize,
        rng: &mut StdRng,
    ) -> MetaSampledTask {
        match self.task_type {
            MetaTaskType::SineRegression => {
                let amp = rng.random_range(0.1_f64..5.0_f64);
                let phase = rng.random_range(0.0_f64..std::f64::consts::PI);

                let mut sample_pts = |n: usize| -> (Vec<Vec<f64>>, Vec<f64>) {
                    let xs: Vec<Vec<f64>> = (0..n)
                        .map(|_| vec![rng.random_range(-5.0_f64..5.0_f64)])
                        .collect();
                    let ys: Vec<f64> = xs.iter().map(|x| amp * (x[0] + phase).sin()).collect();
                    (xs, ys)
                };

                let (support_x, support_y) = sample_pts(n_support);
                let (query_x, query_y) = sample_pts(n_query);
                MetaSampledTask {
                    support_x,
                    support_y,
                    query_x,
                    query_y,
                    task_type: self.task_type,
                }
            }
            MetaTaskType::LinearClassification => {
                let d = self.input_dim;
                // Random hyperplane weight vector
                let w: Vec<f64> = (0..d).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
                let w_norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-8);
                let w_hat: Vec<f64> = w.iter().map(|x| x / w_norm).collect();

                let mut sample_pts = |n: usize| -> (Vec<Vec<f64>>, Vec<f64>) {
                    let xs: Vec<Vec<f64>> = (0..n)
                        .map(|_| (0..d).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect())
                        .collect();
                    let ys: Vec<f64> = xs
                        .iter()
                        .map(|x| {
                            let dot: f64 = x.iter().zip(w_hat.iter()).map(|(a, b)| a * b).sum();
                            if dot >= 0.0 {
                                1.0
                            } else {
                                -1.0
                            }
                        })
                        .collect();
                    (xs, ys)
                };

                let (support_x, support_y) = sample_pts(n_support);
                let (query_x, query_y) = sample_pts(n_query);
                MetaSampledTask {
                    support_x,
                    support_y,
                    query_x,
                    query_y,
                    task_type: self.task_type,
                }
            }
        }
    }

    /// Sample a batch of tasks.
    pub fn batch_tasks(
        &self,
        n: usize,
        n_support: usize,
        n_query: usize,
        rng: &mut StdRng,
    ) -> Vec<MetaSampledTask> {
        (0..n)
            .map(|_| self.sample_task(n_support, n_query, rng))
            .collect()
    }
}

// ─── MetaLearningTrainer ──────────────────────────────────────────────────────

/// Unified meta-training loop.
#[derive(Debug, Clone)]
pub struct MetaLearningTrainer {
    /// Number of inner adaptation steps.
    pub inner_steps: usize,
    /// Inner loop learning rate (for baseline / FOMAML reference).
    pub inner_lr: f64,
    /// Meta learning rate.
    pub meta_lr: f64,
    /// Adam state for meta-parameters.
    adam_m: Vec<f64>,
    adam_v: Vec<f64>,
    adam_t: usize,
    /// Beta1, Beta2 for Adam.
    beta1: f64,
    beta2: f64,
}

impl MetaLearningTrainer {
    /// Create a new trainer.
    pub fn new(inner_steps: usize, inner_lr: f64, meta_lr: f64) -> Self {
        Self {
            inner_steps,
            inner_lr,
            meta_lr,
            adam_m: Vec::new(),
            adam_v: Vec::new(),
            adam_t: 0,
            beta1: 0.9,
            beta2: 0.999,
        }
    }

    /// Compute meta-loss across tasks using a simple linear model.
    /// Returns mean query loss after inner_steps gradient steps.
    pub fn meta_loss(&self, tasks: &[L2lTask], init_params: &[f64]) -> L2lResult<f64> {
        if tasks.is_empty() {
            return Err(L2lError::InvalidConfig {
                field: "tasks".into(),
                reason: "empty task set".into(),
            });
        }
        let mut total_loss = 0.0f64;
        for task in tasks {
            let mut params = init_params.to_vec();
            for _ in 0..self.inner_steps {
                let (_, grad) = compute_linear_mse_grad(&params, &task.support_x, &task.support_y);
                for (p, g) in params.iter_mut().zip(grad.iter()) {
                    *p -= self.inner_lr * g;
                }
            }
            let (q_loss, _) = compute_linear_mse_grad(&params, &task.query_x, &task.query_y);
            total_loss += q_loss / (task.query_y.len() as f64 + 1e-8);
        }
        Ok(total_loss / tasks.len() as f64)
    }

    /// Update meta-parameters using Adam given a computed meta_loss gradient.
    /// Uses finite-differences over init_params.
    pub fn update_meta_params(
        &mut self,
        tasks: &[L2lTask],
        init_params: &mut [f64],
    ) -> L2lResult<f64> {
        let n = init_params.len();
        if self.adam_m.len() != n {
            self.adam_m = vec![0.0; n];
            self.adam_v = vec![0.0; n];
        }

        let base_loss = self.meta_loss(tasks, init_params)?;
        let fd_eps = 1e-4;
        let mut grad = vec![0.0f64; n];

        for i in 0..n {
            init_params[i] += fd_eps;
            let loss_plus = self.meta_loss(tasks, init_params)?;
            init_params[i] -= fd_eps;
            grad[i] = (loss_plus - base_loss) / fd_eps;
        }

        self.adam_t += 1;
        let t = self.adam_t as f64;
        let eps = 1e-8;

        for i in 0..n {
            self.adam_m[i] = self.beta1 * self.adam_m[i] + (1.0 - self.beta1) * grad[i];
            self.adam_v[i] = self.beta2 * self.adam_v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
            let m_hat = self.adam_m[i] / (1.0 - self.beta1.powf(t));
            let v_hat = self.adam_v[i] / (1.0 - self.beta2.powf(t));
            init_params[i] -= self.meta_lr * m_hat / (v_hat.sqrt() + eps);
        }

        Ok(base_loss)
    }

    /// Evaluate: return mean final loss on tasks.
    pub fn evaluate(&self, tasks: &[L2lTask], init_params: &[f64]) -> L2lResult<f64> {
        self.meta_loss(tasks, init_params)
    }
}

// ─── WarmStartOptimizer ───────────────────────────────────────────────────────

/// Warm-starting from meta-learned parameters.
#[derive(Debug, Clone)]
pub struct WarmStartOptimizer {
    /// Learning rate for fine-tuning.
    pub lr: f64,
    /// Number of fine-tune steps.
    pub finetune_steps: usize,
}

impl WarmStartOptimizer {
    /// Create a new warm-start optimizer.
    pub fn new(lr: f64, finetune_steps: usize) -> Self {
        Self { lr, finetune_steps }
    }

    /// Initialise task parameters from meta-learned parameters (copy).
    pub fn initialize_from_meta(&self, meta_params: &[f64], _task_params: &[f64]) -> Vec<f64> {
        meta_params.to_vec()
    }

    /// Fine-tune warm-started parameters on a task's support set.
    /// Returns (warm_params_final, warm_losses).
    pub fn finetune_warm(
        &self,
        meta_params: &[f64],
        task: &L2lTask,
    ) -> L2lResult<(Vec<f64>, Vec<f64>)> {
        let mut params = self.initialize_from_meta(meta_params, meta_params);
        let mut losses = Vec::with_capacity(self.finetune_steps);

        for _ in 0..self.finetune_steps {
            let (loss, grad) = compute_linear_mse_grad(&params, &task.support_x, &task.support_y);
            losses.push(loss / (task.support_y.len() as f64 + 1e-8));
            for (p, g) in params.iter_mut().zip(grad.iter()) {
                *p -= self.lr * g;
            }
        }
        Ok((params, losses))
    }

    /// Cold-start fine-tuning (zero init) for comparison.
    pub fn finetune_cold(
        &self,
        n_params: usize,
        task: &L2lTask,
    ) -> L2lResult<(Vec<f64>, Vec<f64>)> {
        let zero_params = vec![0.0f64; n_params];
        let mut params = zero_params;
        let mut losses = Vec::with_capacity(self.finetune_steps);

        for _ in 0..self.finetune_steps {
            let (loss, grad) = compute_linear_mse_grad(&params, &task.support_x, &task.support_y);
            losses.push(loss / (task.support_y.len() as f64 + 1e-8));
            for (p, g) in params.iter_mut().zip(grad.iter()) {
                *p -= self.lr * g;
            }
        }
        Ok((params, losses))
    }

    /// Compare warm vs cold final losses. Returns (warm_final, cold_final).
    pub fn compare_convergence(
        &self,
        meta_params: &[f64],
        task: &L2lTask,
    ) -> L2lResult<(f64, f64)> {
        let (_, warm_losses) = self.finetune_warm(meta_params, task)?;
        let (_, cold_losses) = self.finetune_cold(meta_params.len(), task)?;
        let warm_final = warm_losses.last().copied().unwrap_or(f64::MAX);
        let cold_final = cold_losses.last().copied().unwrap_or(f64::MAX);
        Ok((warm_final, cold_final))
    }
}

// ─── L2lScheduler ─────────────────────────────────────────────────────────────

/// Learned learning rate schedule using an LSTM.
///
/// Input = (step_norm, loss_norm, grad_norm_norm), output = log_lr.
#[derive(Debug, Clone)]
pub struct L2lScheduler {
    /// LSTM cell for predicting log_lr.
    cell: L2lLstmCell,
    /// Hidden state.
    h: Vec<f64>,
    c: Vec<f64>,
    /// Base learning rate scale.
    pub base_lr: f64,
    /// Total steps (for normalisation).
    pub total_steps: usize,
    /// Running max grad norm (for normalisation).
    max_grad_norm: f64,
    /// Running max loss (for normalisation).
    max_loss: f64,
}

impl L2lScheduler {
    /// Create a new learned LR scheduler.
    pub fn new(hidden_size: usize, total_steps: usize, base_lr: f64, rng: &mut StdRng) -> Self {
        let cell = L2lLstmCell::new(3, hidden_size, 1, rng);
        let h = vec![0.0; hidden_size];
        let c = vec![0.0; hidden_size];
        Self {
            cell,
            h,
            c,
            base_lr,
            total_steps,
            max_grad_norm: 1.0,
            max_loss: 1.0,
        }
    }

    /// Predict next learning rate given current step, loss, grad_norm.
    pub fn predict_lr(&mut self, step: usize, loss: f64, grad_norm: f64) -> L2lResult<f64> {
        // Update running maxima
        if loss.abs() > self.max_loss {
            self.max_loss = loss.abs().max(1e-8);
        }
        if grad_norm > self.max_grad_norm {
            self.max_grad_norm = grad_norm.max(1e-8);
        }

        let step_norm = step as f64 / (self.total_steps as f64 + 1.0);
        let loss_norm = loss / self.max_loss;
        let grad_norm_norm = grad_norm / self.max_grad_norm;

        let input = [step_norm, loss_norm, grad_norm_norm];
        let (h_new, c_new, output) = self.cell.step(&input, &self.h, &self.c)?;
        self.h = h_new;
        self.c = c_new;

        // log_lr in [-10, 0] → lr in [e^-10, 1] * base_lr
        let log_lr = output[0].clamp(-10.0, 0.0);
        Ok(self.base_lr * log_lr.exp())
    }

    /// Reset LSTM state for a new task.
    pub fn reset(&mut self) {
        let hs = self.cell.hidden_size;
        self.h = vec![0.0; hs];
        self.c = vec![0.0; hs];
        self.max_grad_norm = 1.0;
        self.max_loss = 1.0;
    }

    /// Meta-train scheduler to minimise final loss on tasks.
    pub fn meta_train_scheduler(
        &mut self,
        tasks: &[L2lTask],
        n_outer: usize,
        n_params: usize,
        rng: &mut StdRng,
    ) -> L2lResult<Vec<f64>> {
        if tasks.is_empty() {
            return Err(L2lError::InvalidConfig {
                field: "tasks".into(),
                reason: "empty".into(),
            });
        }
        let mut losses = Vec::with_capacity(n_outer);
        let param_count = self.cell.parameters().len();
        let mut m = vec![0.0f64; param_count];
        let mut v = vec![0.0f64; param_count];
        let beta1 = 0.9_f64;
        let beta2 = 0.999_f64;

        for outer in 0..n_outer {
            let task_idx = rng.random_range(0..tasks.len());
            let task = &tasks[task_idx];

            self.reset();
            let mut params: Vec<f64> = (0..n_params)
                .map(|_| rng.random::<f64>() * 0.1 - 0.05)
                .collect();

            let mut total_loss = 0.0f64;
            for step in 0..self.total_steps.min(10) {
                let (loss, grad) =
                    compute_linear_mse_grad(&params, &task.support_x, &task.support_y);
                let gn: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
                let lr = self.predict_lr(step, loss, gn)?;
                for (p, g) in params.iter_mut().zip(grad.iter()) {
                    *p -= lr * g;
                }
                total_loss += loss;
            }
            let mean_loss = total_loss / (self.total_steps.min(10) as f64 + 1e-8);
            losses.push(mean_loss);

            // Simple FD meta-gradient on one cell param
            let cell_params = self.cell.parameters();
            let mut meta_grad = vec![0.0f64; cell_params.len()];
            let fd_eps = 1e-4;
            let probe_idx = outer % cell_params.len();

            let mut cell_plus = self.cell.clone();
            let mut delta_p = vec![0.0f64; cell_params.len()];
            delta_p[probe_idx] = fd_eps;
            cell_plus.apply_update(&delta_p);

            let mut sched_plus = self.clone();
            sched_plus.cell = cell_plus;
            sched_plus.reset();
            let mut params2: Vec<f64> = params
                .iter()
                .map(|_| rng.random::<f64>() * 0.1 - 0.05)
                .collect();
            let mut total_loss2 = 0.0f64;
            for step in 0..self.total_steps.min(10) {
                let (loss2, grad2) =
                    compute_linear_mse_grad(&params2, &task.support_x, &task.support_y);
                let gn2: f64 = grad2.iter().map(|g| g * g).sum::<f64>().sqrt();
                let lr2 = sched_plus.predict_lr(step, loss2, gn2)?;
                for (p, g) in params2.iter_mut().zip(grad2.iter()) {
                    *p -= lr2 * g;
                }
                total_loss2 += loss2;
            }
            let mean_loss2 = total_loss2 / (self.total_steps.min(10) as f64 + 1e-8);
            meta_grad[probe_idx] = (mean_loss2 - mean_loss) / fd_eps;

            // Adam update
            let t = (outer + 1) as f64;
            let eps = 1e-8;
            for i in 0..param_count {
                m[i] = beta1 * m[i] + (1.0 - beta1) * meta_grad[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * meta_grad[i] * meta_grad[i];
                let m_hat = m[i] / (1.0 - beta1.powf(t));
                let v_hat = v[i] / (1.0 - beta2.powf(t));
                meta_grad[i] = 0.001 * m_hat / (v_hat.sqrt() + eps);
            }
            let update: Vec<f64> = meta_grad.iter().map(|g| -g).collect();
            self.cell.apply_update(&update);
        }
        Ok(losses)
    }
}

// ─── L2lMetrics ───────────────────────────────────────────────────────────────

/// Meta-learning evaluation metrics.
#[derive(Debug, Clone)]
pub struct L2lMetrics;

impl L2lMetrics {
    /// Meta-generalisation gap: difference of mean final loss on train vs test tasks.
    pub fn meta_generalization_gap(train_losses: &[f64], test_losses: &[f64]) -> f64 {
        let train_mean = if train_losses.is_empty() {
            0.0
        } else {
            train_losses.iter().sum::<f64>() / train_losses.len() as f64
        };
        let test_mean = if test_losses.is_empty() {
            0.0
        } else {
            test_losses.iter().sum::<f64>() / test_losses.len() as f64
        };
        (test_mean - train_mean).abs()
    }

    /// Area under learning curve (trapezoid rule; lower is better if losses are high).
    pub fn learning_curve_auc(losses: &[f64]) -> f64 {
        if losses.len() < 2 {
            return losses.first().copied().unwrap_or(0.0);
        }
        // Normalise x from 0 to 1
        let n = losses.len() as f64;
        let dx = 1.0 / (n - 1.0);
        let mut auc = 0.0;
        for i in 0..losses.len() - 1 {
            auc += 0.5 * (losses[i] + losses[i + 1]) * dx;
        }
        auc
    }

    /// Mean final loss over tasks.
    pub fn final_performance(final_losses: &[f64]) -> f64 {
        if final_losses.is_empty() {
            return 0.0;
        }
        final_losses.iter().sum::<f64>() / final_losses.len() as f64
    }

    /// Adaptation speed: mean loss after k inner steps across tasks.
    /// `loss_curves`: one `Vec<f64>` per task, each of length >= k.
    pub fn adaptation_speed_k(loss_curves: &[Vec<f64>], k: usize) -> f64 {
        if loss_curves.is_empty() {
            return 0.0;
        }
        let mut total = 0.0f64;
        let mut count = 0usize;
        for curve in loss_curves {
            let idx = k.min(curve.len().saturating_sub(1));
            if !curve.is_empty() {
                total += curve[idx];
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            total / count as f64
        }
    }
}

// ─── Shared utility ───────────────────────────────────────────────────────────

/// Compute MSE loss and gradient for a linear model on a dataset.
/// params = [w_0, ..., w_{d-1}, bias]
/// Returns (total_loss, gradient).
pub fn compute_linear_mse_grad(params: &[f64], xs: &[Vec<f64>], ys: &[f64]) -> (f64, Vec<f64>) {
    let n = xs.len().max(1);
    let d = if params.is_empty() {
        0
    } else {
        params.len() - 1
    };
    let mut total_loss = 0.0f64;
    let mut grad = vec![0.0f64; params.len()];

    for (x, y) in xs.iter().zip(ys.iter()) {
        // y_pred = x . w + bias
        let pred: f64 = x
            .iter()
            .enumerate()
            .map(|(i, xi)| if i < d { params[i] * xi } else { 0.0 })
            .sum::<f64>()
            + if params.len() > d { params[d] } else { 0.0 };

        let err = pred - y;
        total_loss += err * err;

        for (i, xi) in x.iter().enumerate() {
            if i < d {
                grad[i] += 2.0 * err * xi / n as f64;
            }
        }
        if params.len() > d {
            grad[d] += 2.0 * err / n as f64;
        }
    }
    (total_loss, grad)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
