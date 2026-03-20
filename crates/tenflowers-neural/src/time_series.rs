//! Time Series Forecasting — Round 12 Track B.
//!
//! Implements two state-of-the-art deep forecasting architectures:
//!
//! | Model | Reference |
//! |-------|-----------|
//! | [`NBeats`] | Oreshkin et al. (2019) — N-BEATS: Neural Basis Expansion Analysis |
//! | [`TemporalFusionTransformer`] | Lim et al. (2021) — TFT |
//!
//! All weights are stored as plain `Vec<f32>` buffers; no `Tensor` / autograd
//! required.  No `unwrap()`, no `unsafe`, and a single file under 2000 lines.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use tenflowers_neural::time_series::{
//!     NBeats, NBeatsConfig, BasisType, TimeSeriesMetrics,
//! };
//!
//! let cfg = NBeatsConfig {
//!     input_size: 24, forecast_size: 6,
//!     stacks: vec![BasisType::Trend { degree: 3 }, BasisType::Generic],
//!     num_blocks_per_stack: 2, hidden_dim: 32, basis_dim: 8,
//! };
//! let model = NBeats::new(cfg, 42)?;
//! let x = vec![1.0_f32; 24];
//! let forecast = model.forward(&x)?;
//! assert_eq!(forecast.len(), 6);
//! ```

use std::f32::consts::PI;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

type TsResult<T> = Result<T, String>;

// ─────────────────────────────────────────────────────────────────────────────
// Activation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// ReLU: max(0, x).
#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// ELU: if x > 0 then x else alpha*(exp(x)-1).  Using alpha=1.
#[inline]
fn elu(x: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        x.exp() - 1.0
    }
}

/// Sigmoid: numerically stable.
#[inline]
fn sigmoid(x: f32) -> f32 {
    let c = x.clamp(-88.0, 88.0);
    1.0 / (1.0 + (-c).exp())
}

/// Softmax over a slice; returns a new Vec.
fn softmax(v: &[f32]) -> Vec<f32> {
    if v.is_empty() {
        return Vec::new();
    }
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / v.len() as f32; v.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight-initialisation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Kaiming uniform initialisation for a weight matrix of `size` elements.
fn kaiming_uniform(size: usize, fan_in: usize, rng: &mut StdRng) -> Vec<f32> {
    let bound = (2.0_f32 / fan_in.max(1) as f32).sqrt();
    (0..size)
        .map(|_| {
            let u: f32 = rng.random();
            2.0 * bound * u - bound
        })
        .collect()
}

/// Xavier uniform initialisation for a weight matrix.
fn xavier_uniform(size: usize, fan_in: usize, fan_out: usize, rng: &mut StdRng) -> Vec<f32> {
    let bound = (6.0_f32 / (fan_in + fan_out).max(1) as f32).sqrt();
    (0..size)
        .map(|_| {
            let u: f32 = rng.random();
            2.0 * bound * u - bound
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix-vector multiplication helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Dense layer forward: y = W*x + b.
/// `w` has shape [out × in] stored row-major, `x` length in, `b` length out.
fn linear(w: &[f32], b: &[f32], x: &[f32]) -> TsResult<Vec<f32>> {
    let in_dim = x.len();
    let out_dim = b.len();
    if w.len() != out_dim * in_dim {
        return Err(format!(
            "linear: w.len()={} != out_dim*in_dim={}*{}={}",
            w.len(),
            out_dim,
            in_dim,
            out_dim * in_dim,
        ));
    }
    let mut y = vec![0.0_f32; out_dim];
    for o in 0..out_dim {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        y[o] = b[o]
            + row
                .iter()
                .zip(x.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>();
    }
    Ok(y)
}

/// Layer normalisation: (x − mean) / (std + eps) * gamma + beta.
fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32]) -> TsResult<Vec<f32>> {
    let n = x.len();
    if gamma.len() != n || beta.len() != n {
        return Err(format!(
            "layer_norm: dim mismatch x={n}, gamma={}, beta={}",
            gamma.len(),
            beta.len()
        ));
    }
    let mean = x.iter().copied().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let std_inv = (var + 1e-5).sqrt().recip();
    Ok(x.iter()
        .enumerate()
        .map(|(i, &v)| (v - mean) * std_inv * gamma[i] + beta[i])
        .collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// ══════════════════════════  N-BEATS  ═══════════════════════════════════════
// ─────────────────────────────────────────────────────────────────────────────

// ── Basis types ──────────────────────────────────────────────────────────────

/// Basis expansion type for an N-BEATS block.
///
/// * `Generic` — learnable (identity) basis.
/// * `Trend` — polynomial basis up to `degree`.
/// * `Seasonality` — Fourier basis with `num_harmonics` harmonics.
#[derive(Debug, Clone)]
pub enum BasisType {
    /// Learnable data-driven basis (linear / identity).
    Generic,
    /// Polynomial basis `[1, t, t², …, t^degree]`.
    Trend {
        /// Maximum polynomial degree.
        degree: usize,
    },
    /// Fourier basis `[1, cos(2πt/H), sin(2πt/H), cos(4πt/H), …]`.
    Seasonality {
        /// Number of harmonics (excluding the constant term).
        num_harmonics: usize,
    },
}

impl BasisType {
    /// Basis dimensionality for a given `BasisType`.
    pub fn basis_dim(&self, default_basis_dim: usize) -> usize {
        match self {
            BasisType::Generic => default_basis_dim,
            BasisType::Trend { degree } => degree + 1,
            BasisType::Seasonality { num_harmonics } => 2 * num_harmonics + 1,
        }
    }
}

// ── N-BEATS block ─────────────────────────────────────────────────────────────

/// Single N-BEATS block: FC stack + basis expansion.
///
/// Architecture:
/// 1. 4-layer FC stack with ReLU: `x → h₁ → h₂ → h₃ → h₄`
/// 2. Two theta projections `θ_b`, `θ_f` (for backcast / forecast).
/// 3. Basis expansion via [`BasisType`] specific matrix.
#[derive(Debug, Clone)]
pub struct NBeatsBlock {
    /// Look-back window length.
    pub input_size: usize,
    /// Number of forecast steps.
    pub forecast_size: usize,
    /// Hidden dimension of the FC stack.
    pub hidden_dim: usize,
    /// Basis expansion type.
    pub basis_type: BasisType,
    /// FC layer weights; shape per layer: `[hidden_dim × in_dim]` row-major.
    pub fc_weights: Vec<Vec<f32>>,
    /// FC layer biases; length `hidden_dim` each.
    pub fc_biases: Vec<Vec<f32>>,
    /// Theta backcast weight matrix `[basis_dim × hidden_dim]` row-major.
    pub theta_b: Vec<f32>,
    /// Theta forecast weight matrix `[basis_dim × hidden_dim]` row-major.
    pub theta_f: Vec<f32>,
}

impl NBeatsBlock {
    /// Create a new block with Kaiming-uniform random initialisation.
    pub fn new(
        input_size: usize,
        forecast_size: usize,
        hidden_dim: usize,
        basis_type: BasisType,
        basis_dim: usize,
        rng: &mut StdRng,
    ) -> TsResult<Self> {
        if input_size == 0 || forecast_size == 0 || hidden_dim == 0 || basis_dim == 0 {
            return Err("NBeatsBlock::new: all dimensions must be > 0".to_string());
        }
        // 4-layer FC stack: in→h, h→h, h→h, h→h
        let mut fc_weights = Vec::with_capacity(4);
        let mut fc_biases = Vec::with_capacity(4);
        for layer_idx in 0..4_usize {
            let in_dim = if layer_idx == 0 {
                input_size
            } else {
                hidden_dim
            };
            let w = kaiming_uniform(hidden_dim * in_dim, in_dim, rng);
            let b = vec![0.0_f32; hidden_dim];
            fc_weights.push(w);
            fc_biases.push(b);
        }
        let eff_basis_dim = basis_type.basis_dim(basis_dim);
        let theta_b = xavier_uniform(eff_basis_dim * hidden_dim, hidden_dim, eff_basis_dim, rng);
        let theta_f = xavier_uniform(eff_basis_dim * hidden_dim, hidden_dim, eff_basis_dim, rng);
        Ok(Self {
            input_size,
            forecast_size,
            hidden_dim,
            basis_type,
            fc_weights,
            fc_biases,
            theta_b,
            theta_f,
        })
    }

    /// Build a polynomial basis matrix of shape `[size × (degree+1)]`.
    ///
    /// Row `t` contains `[1, t/T, (t/T)², …, (t/T)^degree]` where `T = size`.
    pub fn build_trend_basis(size: usize, degree: usize) -> Vec<Vec<f32>> {
        let t_max = size.max(1) as f32;
        (0..size)
            .map(|t| {
                let x = t as f32 / t_max;
                (0..=degree).map(|d| x.powi(d as i32)).collect()
            })
            .collect()
    }

    /// Build a Fourier basis matrix of shape `[size × (2*num_harmonics+1)]`.
    ///
    /// Row `t` contains `[1, cos(2π·1·t/T), sin(2π·1·t/T), …,
    ///   cos(2π·H·t/T), sin(2π·H·t/T)]` where `T = size`.
    pub fn build_seasonality_basis(size: usize, num_harmonics: usize) -> Vec<Vec<f32>> {
        let t_max = size.max(1) as f32;
        (0..size)
            .map(|t| {
                let t_f = t as f32;
                let mut row = Vec::with_capacity(2 * num_harmonics + 1);
                row.push(1.0_f32); // constant term
                for h in 1..=num_harmonics {
                    let angle = 2.0 * PI * h as f32 * t_f / t_max;
                    row.push(angle.cos());
                    row.push(angle.sin());
                }
                row
            })
            .collect()
    }

    /// Forward pass: returns `(backcast, forecast)`.
    ///
    /// * `x` — input slice of length `input_size`.
    pub fn forward(&self, x: &[f32]) -> TsResult<(Vec<f32>, Vec<f32>)> {
        if x.len() != self.input_size {
            return Err(format!(
                "NBeatsBlock::forward: x.len()={} != input_size={}",
                x.len(),
                self.input_size
            ));
        }
        // FC stack
        let mut h = x.to_vec();
        for (w, b) in self.fc_weights.iter().zip(self.fc_biases.iter()) {
            let pre = linear(w, b, &h)?;
            h = pre.into_iter().map(relu).collect();
        }
        // Effective basis dimension
        let eff_basis_dim = self.basis_type.basis_dim(
            // For Generic, theta_b.len() / hidden_dim gives the actual basis_dim
            self.theta_b.len() / self.hidden_dim,
        );
        // Theta projections → coefficients [basis_dim]
        let theta_b_bias = vec![0.0_f32; eff_basis_dim];
        let theta_f_bias = vec![0.0_f32; eff_basis_dim];
        let coeff_b = linear(&self.theta_b, &theta_b_bias, &h)?;
        let coeff_f = linear(&self.theta_f, &theta_f_bias, &h)?;
        // Basis expansion
        let backcast = self.expand_basis(&coeff_b, self.input_size)?;
        let forecast = self.expand_basis(&coeff_f, self.forecast_size)?;
        Ok((backcast, forecast))
    }

    /// Expand `coefficients` over a basis of `size` time steps.
    fn expand_basis(&self, coefficients: &[f32], size: usize) -> TsResult<Vec<f32>> {
        match &self.basis_type {
            BasisType::Generic => {
                // Identity (linear) basis: just use coefficients directly,
                // truncated / padded to `size`.
                let mut out = vec![0.0_f32; size];
                let copy_len = size.min(coefficients.len());
                out[..copy_len].copy_from_slice(&coefficients[..copy_len]);
                Ok(out)
            }
            BasisType::Trend { degree } => {
                let basis = Self::build_trend_basis(size, *degree);
                // out[t] = Σ_d coeff[d] * basis[t][d]
                let out = basis
                    .iter()
                    .map(|row| {
                        row.iter()
                            .zip(coefficients.iter())
                            .map(|(&b, &c)| b * c)
                            .sum::<f32>()
                    })
                    .collect();
                Ok(out)
            }
            BasisType::Seasonality { num_harmonics } => {
                let basis = Self::build_seasonality_basis(size, *num_harmonics);
                let out = basis
                    .iter()
                    .map(|row| {
                        row.iter()
                            .zip(coefficients.iter())
                            .map(|(&b, &c)| b * c)
                            .sum::<f32>()
                    })
                    .collect();
                Ok(out)
            }
        }
    }
}

// ── N-BEATS stack ─────────────────────────────────────────────────────────────

/// Stack of same-type N-BEATS blocks with residual accumulation.
#[derive(Debug, Clone)]
pub struct NBeatsStack {
    /// Constituent blocks.
    pub blocks: Vec<NBeatsBlock>,
    /// Shared basis type for the stack.
    pub stack_type: BasisType,
}

impl NBeatsStack {
    /// Create a stack of `num_blocks` blocks.
    pub fn new(
        input_size: usize,
        forecast_size: usize,
        hidden_dim: usize,
        basis_type: BasisType,
        basis_dim: usize,
        num_blocks: usize,
        rng: &mut StdRng,
    ) -> TsResult<Self> {
        if num_blocks == 0 {
            return Err("NBeatsStack::new: num_blocks must be > 0".to_string());
        }
        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            blocks.push(NBeatsBlock::new(
                input_size,
                forecast_size,
                hidden_dim,
                basis_type.clone(),
                basis_dim,
                rng,
            )?);
        }
        Ok(Self {
            blocks,
            stack_type: basis_type,
        })
    }

    /// Forward: residual subtraction on input, accumulate forecasts.
    ///
    /// Returns the summed forecast of length `forecast_size`.
    pub fn forward(&self, x: &[f32]) -> TsResult<Vec<f32>> {
        if self.blocks.is_empty() {
            return Err("NBeatsStack: no blocks".to_string());
        }
        let forecast_size = self.blocks[0].forecast_size;
        let mut residual = x.to_vec();
        let mut total_forecast = vec![0.0_f32; forecast_size];
        for block in &self.blocks {
            let (backcast, forecast) = block.forward(&residual)?;
            // subtract backcast from residual
            for (r, bc) in residual.iter_mut().zip(backcast.iter()) {
                *r -= bc;
            }
            for (tf, f) in total_forecast.iter_mut().zip(forecast.iter()) {
                *tf += f;
            }
        }
        Ok(total_forecast)
    }
}

// ── N-BEATS full model ────────────────────────────────────────────────────────

/// Configuration for the full N-BEATS model.
#[derive(Debug, Clone)]
pub struct NBeatsConfig {
    /// Length of the look-back window (past observations).
    pub input_size: usize,
    /// Number of future time steps to forecast.
    pub forecast_size: usize,
    /// Ordered list of basis types (one per stack).
    pub stacks: Vec<BasisType>,
    /// Number of blocks per stack.
    pub num_blocks_per_stack: usize,
    /// Hidden dimension of each block's FC stack.
    pub hidden_dim: usize,
    /// Basis dimension for `Generic` stacks.
    pub basis_dim: usize,
}

/// Full N-BEATS model (Oreshkin et al., 2019).
///
/// Organises stacks in series; each stack's residual feeds the next.
#[derive(Debug, Clone)]
pub struct NBeats {
    /// Model configuration.
    pub config: NBeatsConfig,
    /// Ordered stacks.
    pub stacks: Vec<NBeatsStack>,
    /// Per-stack forecast outputs from last forward call (decomposition cache).
    stack_forecasts: Vec<Vec<f32>>,
    /// Per-stack backcast outputs from last forward call.
    stack_backcasts: Vec<Vec<f32>>,
}

impl NBeats {
    /// Construct a new N-BEATS model with random initialisation.
    pub fn new(config: NBeatsConfig, seed: u64) -> TsResult<Self> {
        if config.stacks.is_empty() {
            return Err("NBeats::new: stacks must not be empty".to_string());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut stacks = Vec::with_capacity(config.stacks.len());
        for basis_type in &config.stacks {
            stacks.push(NBeatsStack::new(
                config.input_size,
                config.forecast_size,
                config.hidden_dim,
                basis_type.clone(),
                config.basis_dim,
                config.num_blocks_per_stack,
                &mut rng,
            )?);
        }
        let n_stacks = config.stacks.len();
        Ok(Self {
            config,
            stacks,
            stack_forecasts: vec![Vec::new(); n_stacks],
            stack_backcasts: vec![Vec::new(); n_stacks],
        })
    }

    /// Full forward pass; returns forecast of length `forecast_size`.
    pub fn forward(&mut self, x: &[f32]) -> TsResult<Vec<f32>> {
        if x.len() != self.config.input_size {
            return Err(format!(
                "NBeats::forward: x.len()={} != input_size={}",
                x.len(),
                self.config.input_size
            ));
        }
        let mut residual = x.to_vec();
        let mut total_forecast = vec![0.0_f32; self.config.forecast_size];
        for (stack_idx, stack) in self.stacks.iter().enumerate() {
            // Collect per-block details for decomposition
            let mut stack_forecast = vec![0.0_f32; self.config.forecast_size];
            let mut stack_backcast_sum = vec![0.0_f32; self.config.input_size];
            let mut local_residual = residual.clone();
            for block in &stack.blocks {
                let (backcast, forecast) = block.forward(&local_residual)?;
                for (r, bc) in local_residual.iter_mut().zip(backcast.iter()) {
                    *r -= bc;
                }
                for (sf, f) in stack_forecast.iter_mut().zip(forecast.iter()) {
                    *sf += f;
                }
                for (sb, bc) in stack_backcast_sum.iter_mut().zip(backcast.iter()) {
                    *sb += bc;
                }
            }
            // Update residual for next stack
            residual = local_residual;
            // Accumulate global forecast
            for (tf, sf) in total_forecast.iter_mut().zip(stack_forecast.iter()) {
                *tf += sf;
            }
            self.stack_forecasts[stack_idx] = stack_forecast;
            self.stack_backcasts[stack_idx] = stack_backcast_sum;
        }
        Ok(total_forecast)
    }

    /// Return per-stack backcast decomposition from the last `forward` call.
    pub fn backcast_decomposition(&self) -> &[Vec<f32>] {
        &self.stack_backcasts
    }

    /// Return per-stack forecast decomposition from the last `forward` call.
    pub fn forecast_decomposition(&self) -> &[Vec<f32>] {
        &self.stack_forecasts
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ════════════════  Temporal Fusion Transformer (TFT)  ════════════════════════
// ─────────────────────────────────────────────────────────────────────────────

// ── Gated Residual Network ────────────────────────────────────────────────────

/// Gated Residual Network (GRN) sublayer used throughout TFT.
///
/// Forward: `GRN(x) = LayerNorm(x_skip + GLU(h))`
/// where `h = W2·ELU(W1·x + b1) + b2` and `GLU = h * sigmoid(Wg·x + bg)`.
#[derive(Debug, Clone)]
pub struct GRN {
    // Primary pathway
    /// W1 weight: [hidden_dim × input_dim] row-major.
    pub w1: Vec<f32>,
    /// b1 bias: \[hidden_dim\].
    pub b1: Vec<f32>,
    /// W2 weight: [hidden_dim × hidden_dim] row-major.
    pub w2: Vec<f32>,
    /// b2 bias: \[hidden_dim\].
    pub b2: Vec<f32>,
    // Gate pathway
    /// Gate weight: [hidden_dim × input_dim] row-major.
    pub wg: Vec<f32>,
    /// Gate bias: \[hidden_dim\].
    pub bg: Vec<f32>,
    // Skip connection (only when input_dim != hidden_dim)
    /// Optional projection: [hidden_dim × input_dim].
    pub skip_w: Option<Vec<f32>>,
    // Layer norm params
    /// Layer norm scale: \[hidden_dim\].
    pub ln_gamma: Vec<f32>,
    /// Layer norm shift: \[hidden_dim\].
    pub ln_beta: Vec<f32>,
    /// Input dimension (for skip projection check).
    input_dim: usize,
    /// Hidden / output dimension.
    hidden_dim: usize,
}

impl GRN {
    /// Create a new GRN with Kaiming-uniform weights.
    pub fn new(input_dim: usize, hidden_dim: usize, rng: &mut StdRng) -> TsResult<Self> {
        if input_dim == 0 || hidden_dim == 0 {
            return Err("GRN::new: dimensions must be > 0".to_string());
        }
        let w1 = kaiming_uniform(hidden_dim * input_dim, input_dim, rng);
        let b1 = vec![0.0_f32; hidden_dim];
        let w2 = kaiming_uniform(hidden_dim * hidden_dim, hidden_dim, rng);
        let b2 = vec![0.0_f32; hidden_dim];
        let wg = kaiming_uniform(hidden_dim * input_dim, input_dim, rng);
        let bg = vec![0.0_f32; hidden_dim];
        let skip_w = if input_dim != hidden_dim {
            Some(kaiming_uniform(hidden_dim * input_dim, input_dim, rng))
        } else {
            None
        };
        Ok(Self {
            w1,
            b1,
            w2,
            b2,
            wg,
            bg,
            skip_w,
            ln_gamma: vec![1.0_f32; hidden_dim],
            ln_beta: vec![0.0_f32; hidden_dim],
            input_dim,
            hidden_dim,
        })
    }

    /// Forward pass.
    ///
    /// * `x` — input of length `input_dim`.
    /// * `context` — optional context signal of length `input_dim` added to W1·x.
    pub fn forward(&self, x: &[f32], context: Option<&[f32]>) -> TsResult<Vec<f32>> {
        if x.len() != self.input_dim {
            return Err(format!(
                "GRN::forward: x.len()={} != input_dim={}",
                x.len(),
                self.input_dim
            ));
        }
        // ----- Primary path -----
        let mut h1 = linear(&self.w1, &self.b1, x)?;
        // Add context if supplied
        if let Some(ctx) = context {
            if ctx.len() != self.input_dim {
                return Err(format!(
                    "GRN::forward: context.len()={} != input_dim={}",
                    ctx.len(),
                    self.input_dim
                ));
            }
            // project context with w1 (shared) and add
            let ctx_proj = linear(&self.w1, &vec![0.0_f32; self.hidden_dim], ctx)?;
            for (h, c) in h1.iter_mut().zip(ctx_proj.iter()) {
                *h += c;
            }
        }
        let h1_elu: Vec<f32> = h1.into_iter().map(elu).collect();
        let h2 = linear(&self.w2, &self.b2, &h1_elu)?;

        // ----- Gate path -----
        let gate_pre = linear(&self.wg, &self.bg, x)?;
        let gate: Vec<f32> = gate_pre.into_iter().map(sigmoid).collect();

        // Gated output
        let gated: Vec<f32> = h2.iter().zip(gate.iter()).map(|(&h, &g)| h * g).collect();

        // ----- Skip connection -----
        let skip: Vec<f32> = if let Some(sw) = &self.skip_w {
            linear(sw, &vec![0.0_f32; self.hidden_dim], x)?
        } else {
            x.to_vec()
        };

        // Add + LayerNorm
        let added: Vec<f32> = gated
            .iter()
            .zip(skip.iter())
            .map(|(&g, &s)| g + s)
            .collect();
        layer_norm(&added, &self.ln_gamma, &self.ln_beta)
    }
}

// ── Variable Selection Network ────────────────────────────────────────────────

/// Variable Selection Network (VSN) — learns soft variable importance weights.
///
/// Given `num_vars` input vectors each of dimension `hidden_dim`, it returns:
/// * A weighted combination of per-variable GRN outputs.
/// * The softmax importance weights (for interpretability).
#[derive(Debug, Clone)]
pub struct VSN {
    /// Number of input variables.
    pub num_vars: usize,
    /// Hidden dimension (same for all).
    pub hidden_dim: usize,
    /// One GRN per variable, input_dim = hidden_dim.
    pub grn_weights: Vec<GRN>,
    /// Attention GRN: flattened concat → hidden_dim → softmax logits (num_vars).
    pub attention_grn: GRN,
    /// Projection from hidden_dim → num_vars for final logits.
    attn_proj_w: Vec<f32>,
    attn_proj_b: Vec<f32>,
}

impl VSN {
    /// Create a new VSN.
    ///
    /// Each variable GRN has `input_dim = hidden_dim`.
    /// The attention GRN receives the concatenated input `num_vars * hidden_dim`
    /// and maps it to `hidden_dim`.
    pub fn new(num_vars: usize, hidden_dim: usize, rng: &mut StdRng) -> TsResult<Self> {
        if num_vars == 0 || hidden_dim == 0 {
            return Err("VSN::new: num_vars and hidden_dim must be > 0".to_string());
        }
        let mut grn_weights = Vec::with_capacity(num_vars);
        for _ in 0..num_vars {
            grn_weights.push(GRN::new(hidden_dim, hidden_dim, rng)?);
        }
        // Attention GRN: input is flattened concat [num_vars * hidden_dim]
        let attention_grn = GRN::new(num_vars * hidden_dim, hidden_dim, rng)?;
        // Final logit projection hidden_dim → num_vars
        let attn_proj_w = xavier_uniform(num_vars * hidden_dim, hidden_dim, num_vars, rng);
        let attn_proj_b = vec![0.0_f32; num_vars];
        Ok(Self {
            num_vars,
            hidden_dim,
            grn_weights,
            attention_grn,
            attn_proj_w,
            attn_proj_b,
        })
    }

    /// Forward pass.
    ///
    /// * `inputs` — slice of `num_vars` vectors each of length `hidden_dim`.
    ///
    /// Returns `(selected: Vec<f32>, weights: Vec<f32>)` where:
    /// * `selected` has length `hidden_dim` (softmax-weighted sum).
    /// * `weights` has length `num_vars` and sums to 1.
    pub fn forward(&self, inputs: &[Vec<f32>]) -> TsResult<(Vec<f32>, Vec<f32>)> {
        if inputs.len() != self.num_vars {
            return Err(format!(
                "VSN::forward: inputs.len()={} != num_vars={}",
                inputs.len(),
                self.num_vars
            ));
        }
        for (i, inp) in inputs.iter().enumerate() {
            if inp.len() != self.hidden_dim {
                return Err(format!(
                    "VSN::forward: inputs[{}].len()={} != hidden_dim={}",
                    i,
                    inp.len(),
                    self.hidden_dim
                ));
            }
        }
        // Per-variable GRN outputs
        let mut var_outputs: Vec<Vec<f32>> = Vec::with_capacity(self.num_vars);
        for (grn, inp) in self.grn_weights.iter().zip(inputs.iter()) {
            var_outputs.push(grn.forward(inp, None)?);
        }
        // Concatenate all inputs for attention GRN
        let concat: Vec<f32> = inputs.iter().flat_map(|v| v.iter().copied()).collect();
        let attn_hidden = self.attention_grn.forward(&concat, None)?;
        // Project to logits [num_vars]
        let logits = linear(&self.attn_proj_w, &self.attn_proj_b, &attn_hidden)?;
        let weights = softmax(&logits);
        // Weighted sum of per-variable outputs
        let mut selected = vec![0.0_f32; self.hidden_dim];
        for (v_out, &w) in var_outputs.iter().zip(weights.iter()) {
            for (s, &v) in selected.iter_mut().zip(v_out.iter()) {
                *s += w * v;
            }
        }
        Ok((selected, weights))
    }
}

// ── TFT configuration and model ───────────────────────────────────────────────

/// Configuration for the Temporal Fusion Transformer.
#[derive(Debug, Clone)]
pub struct TftConfig {
    /// Number of past time-step features per step.
    pub past_input_size: usize,
    /// Number of future covariate features per step.
    pub future_input_size: usize,
    /// Hidden dimension throughout the model.
    pub hidden_dim: usize,
    /// Number of self-attention heads.
    pub num_heads: usize,
    /// Number of LSTM encoder layers (simplified: linear gating approximation).
    pub num_lstm_layers: usize,
    /// Dropout rate (currently stored but not applied in inference-only impl).
    pub dropout: f32,
    /// Forecast horizon (steps ahead).
    pub forecast_horizon: usize,
    /// Target quantiles, e.g. `[0.1, 0.5, 0.9]`.
    pub quantiles: Vec<f32>,
}

/// Simplified Temporal Fusion Transformer (Lim et al., 2021).
///
/// Supports:
/// * Past variable selection (VSN).
/// * Future variable selection (VSN).
/// * Simplified LSTM encoder (single-layer linear gate approximation).
/// * Multi-head self-attention with interpretable weights.
/// * Point-wise feed-forward.
/// * Quantile output projection.
#[derive(Debug, Clone)]
pub struct TemporalFusionTransformer {
    /// Model configuration.
    pub config: TftConfig,
    /// VSN for past inputs.
    pub past_vsn: VSN,
    /// VSN for future inputs.
    pub future_vsn: VSN,
    /// LSTM encoder weights (simplified: [4 * hidden_dim × hidden_dim] for gates).
    pub lstm_weights: Vec<f32>,
    /// LSTM encoder bias: [4 * hidden_dim].
    lstm_bias: Vec<f32>,
    /// Attention Q weight: [hidden_dim × hidden_dim].
    pub attn_q: Vec<f32>,
    /// Attention K weight: [hidden_dim × hidden_dim].
    pub attn_k: Vec<f32>,
    /// Attention V weight: [hidden_dim × hidden_dim].
    pub attn_v: Vec<f32>,
    /// Attention output projection: [hidden_dim × hidden_dim].
    pub attn_out: Vec<f32>,
    /// Point-wise FFN W1: [hidden_dim × hidden_dim].
    pub ffn_w1: Vec<f32>,
    /// Point-wise FFN W2: [hidden_dim × hidden_dim].
    pub ffn_w2: Vec<f32>,
    /// Output projection: [len(quantiles) × hidden_dim] per forecast step.
    pub output_w: Vec<f32>,
    /// Output projection bias: [len(quantiles)].
    output_b: Vec<f32>,
    /// Post-attention layer norm params.
    attn_ln_gamma: Vec<f32>,
    attn_ln_beta: Vec<f32>,
    /// Post-FFN layer norm params.
    ffn_ln_gamma: Vec<f32>,
    ffn_ln_beta: Vec<f32>,
}

impl TemporalFusionTransformer {
    /// Construct a new TFT with random initialisation.
    pub fn new(config: TftConfig, seed: u64) -> TsResult<Self> {
        if config.past_input_size == 0
            || config.future_input_size == 0
            || config.hidden_dim == 0
            || config.num_heads == 0
            || config.forecast_horizon == 0
            || config.quantiles.is_empty()
        {
            return Err("TFT::new: all dimensions must be > 0, quantiles non-empty".to_string());
        }
        if config.hidden_dim % config.num_heads != 0 {
            return Err(format!(
                "TFT::new: hidden_dim={} must be divisible by num_heads={}",
                config.hidden_dim, config.num_heads
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let hd = config.hidden_dim;
        let nq = config.quantiles.len();

        let past_vsn = VSN::new(config.past_input_size, hd, &mut rng)?;
        let future_vsn = VSN::new(config.future_input_size, hd, &mut rng)?;

        // LSTM: simplified single-layer; gate matrix [4*hd × hd]
        let lstm_weights = kaiming_uniform(4 * hd * hd, hd, &mut rng);
        let lstm_bias = vec![0.0_f32; 4 * hd];

        let attn_q = xavier_uniform(hd * hd, hd, hd, &mut rng);
        let attn_k = xavier_uniform(hd * hd, hd, hd, &mut rng);
        let attn_v = xavier_uniform(hd * hd, hd, hd, &mut rng);
        let attn_out = xavier_uniform(hd * hd, hd, hd, &mut rng);
        let ffn_w1 = kaiming_uniform(hd * hd, hd, &mut rng);
        let ffn_w2 = kaiming_uniform(hd * hd, hd, &mut rng);
        let output_w = xavier_uniform(nq * hd, hd, nq, &mut rng);
        let output_b = vec![0.0_f32; nq];

        Ok(Self {
            config,
            past_vsn,
            future_vsn,
            lstm_weights,
            lstm_bias,
            attn_q,
            attn_k,
            attn_v,
            attn_out,
            ffn_w1,
            ffn_w2,
            output_w,
            output_b,
            attn_ln_gamma: vec![1.0_f32; hd],
            attn_ln_beta: vec![0.0_f32; hd],
            ffn_ln_gamma: vec![1.0_f32; hd],
            ffn_ln_beta: vec![0.0_f32; hd],
        })
    }

    /// Simplified LSTM step: returns new hidden state of length `hidden_dim`.
    ///
    /// Approximates LSTM gates using a single linear layer (no cell state),
    /// producing `tanh(sigmoid(Wh + b) ⊙ h_prev + …)` style output.
    fn lstm_step(&self, h_prev: &[f32], x: &[f32]) -> TsResult<Vec<f32>> {
        let hd = self.config.hidden_dim;
        if h_prev.len() != hd || x.len() != hd {
            return Err(format!(
                "lstm_step: h_prev.len()={}, x.len()={}, expected {}",
                h_prev.len(),
                x.len(),
                hd
            ));
        }
        // Combined input: [x + h_prev] → [4*hd]
        let combined: Vec<f32> = x.iter().zip(h_prev.iter()).map(|(&a, &b)| a + b).collect();
        let gates = linear(&self.lstm_weights, &self.lstm_bias, &combined)?;
        // i, f, g, o gates
        let i_gate: Vec<f32> = gates[..hd].iter().map(|&v| sigmoid(v)).collect();
        let f_gate: Vec<f32> = gates[hd..2 * hd].iter().map(|&v| sigmoid(v)).collect();
        let g_gate: Vec<f32> = gates[2 * hd..3 * hd].iter().map(|&v| v.tanh()).collect();
        let o_gate: Vec<f32> = gates[3 * hd..].iter().map(|&v| sigmoid(v)).collect();
        // c = f ⊙ h_prev (reuse h as approx cell) + i ⊙ g
        let c: Vec<f32> = f_gate
            .iter()
            .zip(h_prev.iter())
            .zip(i_gate.iter())
            .zip(g_gate.iter())
            .map(|(((f, h), i), g)| f * h + i * g)
            .collect();
        // h = o ⊙ tanh(c)
        let h_new: Vec<f32> = o_gate
            .iter()
            .zip(c.iter())
            .map(|(&o, &ci)| o * ci.tanh())
            .collect();
        Ok(h_new)
    }

    /// Scaled dot-product attention over a sequence.
    ///
    /// `seq` is laid out as `[seq_len × hidden_dim]` row-major.
    /// Returns attended output of same shape.
    fn multi_head_attention(&self, seq: &[f32], seq_len: usize) -> TsResult<Vec<f32>> {
        let hd = self.config.hidden_dim;
        let nh = self.config.num_heads;
        let head_dim = hd / nh;
        if seq.len() != seq_len * hd {
            return Err(format!(
                "mha: seq.len()={} != seq_len*hd={}",
                seq.len(),
                seq_len * hd
            ));
        }
        let b_q = vec![0.0_f32; hd];
        let b_k = vec![0.0_f32; hd];
        let b_v = vec![0.0_f32; hd];
        let b_o = vec![0.0_f32; hd];
        // Project all tokens
        let mut q_all = vec![0.0_f32; seq_len * hd];
        let mut k_all = vec![0.0_f32; seq_len * hd];
        let mut v_all = vec![0.0_f32; seq_len * hd];
        for t in 0..seq_len {
            let tok = &seq[t * hd..(t + 1) * hd];
            let q = linear(&self.attn_q, &b_q, tok)?;
            let k = linear(&self.attn_k, &b_k, tok)?;
            let v = linear(&self.attn_v, &b_v, tok)?;
            q_all[t * hd..(t + 1) * hd].copy_from_slice(&q);
            k_all[t * hd..(t + 1) * hd].copy_from_slice(&k);
            v_all[t * hd..(t + 1) * hd].copy_from_slice(&v);
        }
        let scale = (head_dim as f32).sqrt().recip();
        let mut out_all = vec![0.0_f32; seq_len * hd];
        // Per head
        for h in 0..nh {
            let h_start = h * head_dim;
            for t in 0..seq_len {
                // attention scores for this query token
                let q_t = &q_all[t * hd + h_start..t * hd + h_start + head_dim];
                let scores: Vec<f32> = (0..seq_len)
                    .map(|s| {
                        let k_s = &k_all[s * hd + h_start..s * hd + h_start + head_dim];
                        q_t.iter()
                            .zip(k_s.iter())
                            .map(|(&qi, &ki)| qi * ki)
                            .sum::<f32>()
                            * scale
                    })
                    .collect();
                let attn_w = softmax(&scores);
                // weighted sum of values
                let mut head_out = vec![0.0_f32; head_dim];
                for (s, &aw) in attn_w.iter().enumerate() {
                    let v_s = &v_all[s * hd + h_start..s * hd + h_start + head_dim];
                    for (ho, &vs) in head_out.iter_mut().zip(v_s.iter()) {
                        *ho += aw * vs;
                    }
                }
                // Write to output slot
                out_all[t * hd + h_start..t * hd + h_start + head_dim].copy_from_slice(&head_out);
            }
        }
        // Output projection + residual + layer norm
        let mut result = vec![0.0_f32; seq_len * hd];
        for t in 0..seq_len {
            let attn_t = &out_all[t * hd..(t + 1) * hd];
            let proj = linear(&self.attn_out, &b_o, attn_t)?;
            let tok = &seq[t * hd..(t + 1) * hd];
            let added: Vec<f32> = proj.iter().zip(tok.iter()).map(|(&p, &s)| p + s).collect();
            let normed = layer_norm(&added, &self.attn_ln_gamma, &self.attn_ln_beta)?;
            result[t * hd..(t + 1) * hd].copy_from_slice(&normed);
        }
        Ok(result)
    }

    /// Point-wise FFN with residual + layer norm.
    fn ffn_block(&self, x: &[f32]) -> TsResult<Vec<f32>> {
        let hd = self.config.hidden_dim;
        let b1 = vec![0.0_f32; hd];
        let b2 = vec![0.0_f32; hd];
        let h = linear(&self.ffn_w1, &b1, x)?;
        let h_relu: Vec<f32> = h.into_iter().map(relu).collect();
        let out = linear(&self.ffn_w2, &b2, &h_relu)?;
        let added: Vec<f32> = out.iter().zip(x.iter()).map(|(&o, &xi)| o + xi).collect();
        layer_norm(&added, &self.ffn_ln_gamma, &self.ffn_ln_beta)
    }

    /// Full forward pass.
    ///
    /// * `past` — past observations flattened `[past_seq_len × past_input_size]`.
    ///   The number of past steps is inferred as `past.len() / past_input_size`.
    /// * `future` — future covariates flattened `[forecast_horizon × future_input_size]`.
    ///
    /// Returns quantile forecasts flattened `[forecast_horizon × len(quantiles)]`.
    pub fn forward(&self, past: &[f32], future: &[f32]) -> TsResult<Vec<f32>> {
        let hd = self.config.hidden_dim;
        let horizon = self.config.forecast_horizon;
        let nq = self.config.quantiles.len();
        let past_feat = self.config.past_input_size;
        let fut_feat = self.config.future_input_size;

        if past.len() % past_feat != 0 {
            return Err(format!(
                "TFT::forward: past.len()={} not divisible by past_input_size={}",
                past.len(),
                past_feat
            ));
        }
        let past_len = past.len() / past_feat;
        if future.len() != horizon * fut_feat {
            return Err(format!(
                "TFT::forward: future.len()={} != horizon*fut_feat={}",
                future.len(),
                horizon * fut_feat
            ));
        }

        // ---- Step 1: Variable selection on past time steps ----
        // Each past step: split into individual features, embed via VSN
        let mut encoder_seq = vec![0.0_f32; past_len * hd];
        for t in 0..past_len {
            let step_features: Vec<Vec<f32>> = (0..past_feat)
                .map(|f| {
                    // Embed scalar feature as a hd-dim vector (repeat broadcast)
                    let val = past[t * past_feat + f];
                    vec![val; hd]
                })
                .collect();
            let (selected, _weights) = self.past_vsn.forward(&step_features)?;
            encoder_seq[t * hd..(t + 1) * hd].copy_from_slice(&selected);
        }

        // ---- Step 2: LSTM encoder ----
        let mut h = vec![0.0_f32; hd];
        let mut encoded_seq = vec![0.0_f32; past_len * hd];
        for t in 0..past_len {
            let x_t = &encoder_seq[t * hd..(t + 1) * hd];
            h = self.lstm_step(&h, x_t)?;
            encoded_seq[t * hd..(t + 1) * hd].copy_from_slice(&h);
        }

        // ---- Step 3: Variable selection on future time steps ----
        let mut decoder_seq = vec![0.0_f32; horizon * hd];
        for t in 0..horizon {
            let step_features: Vec<Vec<f32>> = (0..fut_feat)
                .map(|f| {
                    let val = future[t * fut_feat + f];
                    vec![val; hd]
                })
                .collect();
            let (selected, _weights) = self.future_vsn.forward(&step_features)?;
            decoder_seq[t * hd..(t + 1) * hd].copy_from_slice(&selected);
        }

        // ---- Step 4: Concatenate encoder + decoder sequence ----
        let total_len = past_len + horizon;
        let mut full_seq = vec![0.0_f32; total_len * hd];
        full_seq[..past_len * hd].copy_from_slice(&encoded_seq);
        full_seq[past_len * hd..].copy_from_slice(&decoder_seq);

        // ---- Step 5: Self-attention ----
        let attended = self.multi_head_attention(&full_seq, total_len)?;

        // ---- Step 6: FFN block on each token ----
        let mut ffn_out = vec![0.0_f32; total_len * hd];
        for t in 0..total_len {
            let tok = &attended[t * hd..(t + 1) * hd];
            let out = self.ffn_block(tok)?;
            ffn_out[t * hd..(t + 1) * hd].copy_from_slice(&out);
        }

        // ---- Step 7: Output projection on forecast tokens ----
        let mut forecasts = vec![0.0_f32; horizon * nq];
        for t in 0..horizon {
            let tok = &ffn_out[(past_len + t) * hd..(past_len + t + 1) * hd];
            let q_preds = linear(&self.output_w, &self.output_b, tok)?;
            forecasts[t * nq..(t + 1) * nq].copy_from_slice(&q_preds);
        }
        Ok(forecasts)
    }

    /// Pinball (quantile) loss for a single quantile.
    ///
    /// `preds` and `targets` must be the same length.
    /// `quantile` ∈ (0, 1).
    ///
    /// Loss = mean( q*(y - ŷ) if y > ŷ else (1-q)*(ŷ - y) )
    pub fn quantile_loss(preds: &[f32], targets: &[f32], quantile: f32) -> TsResult<f32> {
        if preds.len() != targets.len() {
            return Err(format!(
                "quantile_loss: preds.len()={} != targets.len()={}",
                preds.len(),
                targets.len()
            ));
        }
        if preds.is_empty() {
            return Err("quantile_loss: empty input".to_string());
        }
        let q = quantile.clamp(1e-6, 1.0 - 1e-6);
        let total: f32 = preds
            .iter()
            .zip(targets.iter())
            .map(|(&p, &y)| {
                let err = y - p;
                if err > 0.0 {
                    q * err
                } else {
                    (1.0 - q) * (-err)
                }
            })
            .sum();
        Ok(total / preds.len() as f32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ═══════════════════════  Time Series Metrics  ═══════════════════════════════
// ─────────────────────────────────────────────────────────────────────────────

/// Collection of evaluation metrics for time series forecasting.
pub struct TimeSeriesMetrics;

impl TimeSeriesMetrics {
    /// Mean Absolute Error.
    pub fn mae(preds: &[f32], targets: &[f32]) -> TsResult<f32> {
        Self::check_lengths(preds, targets)?;
        Ok(preds
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| (p - t).abs())
            .sum::<f32>()
            / preds.len() as f32)
    }

    /// Mean Squared Error.
    pub fn mse(preds: &[f32], targets: &[f32]) -> TsResult<f32> {
        Self::check_lengths(preds, targets)?;
        Ok(preds
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| (p - t).powi(2))
            .sum::<f32>()
            / preds.len() as f32)
    }

    /// Mean Absolute Percentage Error.
    ///
    /// Returns the mean of `|y_hat - y| / max(|y|, eps) * 100`.
    pub fn mape(preds: &[f32], targets: &[f32]) -> TsResult<f32> {
        Self::check_lengths(preds, targets)?;
        let eps = 1e-8_f32;
        Ok(preds
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| (p - t).abs() / t.abs().max(eps) * 100.0)
            .sum::<f32>()
            / preds.len() as f32)
    }

    /// Symmetric Mean Absolute Percentage Error.
    ///
    /// `sMAPE = mean( 2*|y_hat - y| / (|y| + |y_hat| + eps) * 100 )`
    pub fn smape(preds: &[f32], targets: &[f32]) -> TsResult<f32> {
        Self::check_lengths(preds, targets)?;
        let eps = 1e-8_f32;
        Ok(preds
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| 2.0 * (p - t).abs() / (t.abs() + p.abs() + eps) * 100.0)
            .sum::<f32>()
            / preds.len() as f32)
    }

    /// Mean Absolute Scaled Error.
    ///
    /// `MASE = MAE / (mean(|in_sample_errors|) + eps)`
    ///
    /// `in_sample_errors` is typically the naive-forecast errors on training data:
    /// `y[t] - y[t-1]` for a simple seasonal naïve baseline.
    pub fn mase(preds: &[f32], targets: &[f32], in_sample_errors: &[f32]) -> TsResult<f32> {
        Self::check_lengths(preds, targets)?;
        if in_sample_errors.is_empty() {
            return Err("mase: in_sample_errors must not be empty".to_string());
        }
        let mae = Self::mae(preds, targets)?;
        let scale = in_sample_errors.iter().map(|e| e.abs()).sum::<f32>()
            / in_sample_errors.len() as f32
            + 1e-8;
        Ok(mae / scale)
    }

    /// Weighted Quantile Loss.
    ///
    /// * `preds_per_quantile` — one vector of predictions per quantile.
    /// * `targets` — ground-truth values.
    /// * `quantiles` — the quantile levels.
    ///
    /// `WQL = mean_q [ 2 * QL(q) / (sum |targets| + eps) ]`
    pub fn wql(
        preds_per_quantile: &[Vec<f32>],
        targets: &[f32],
        quantiles: &[f32],
    ) -> TsResult<f32> {
        if preds_per_quantile.len() != quantiles.len() {
            return Err(format!(
                "wql: preds_per_quantile.len()={} != quantiles.len()={}",
                preds_per_quantile.len(),
                quantiles.len()
            ));
        }
        if preds_per_quantile.is_empty() || targets.is_empty() {
            return Err("wql: empty inputs".to_string());
        }
        let target_sum = targets.iter().map(|t| t.abs()).sum::<f32>() + 1e-8;
        let mut total = 0.0_f32;
        for (preds, &q) in preds_per_quantile.iter().zip(quantiles.iter()) {
            let ql = TemporalFusionTransformer::quantile_loss(preds, targets, q)?;
            total += 2.0 * ql * preds.len() as f32 / target_sum;
        }
        Ok(total / quantiles.len() as f32)
    }

    /// Validate that two slices have equal, nonzero length.
    fn check_lengths(a: &[f32], b: &[f32]) -> TsResult<()> {
        if a.len() != b.len() {
            return Err(format!(
                "metric: length mismatch {} vs {}",
                a.len(),
                b.len()
            ));
        }
        if a.is_empty() {
            return Err("metric: empty input".to_string());
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_nbeats(
        input: usize,
        forecast: usize,
        hidden: usize,
        basis_dim: usize,
        stacks: Vec<BasisType>,
    ) -> NBeats {
        let cfg = NBeatsConfig {
            input_size: input,
            forecast_size: forecast,
            stacks,
            num_blocks_per_stack: 2,
            hidden_dim: hidden,
            basis_dim,
        };
        NBeats::new(cfg, 7).expect("NBeats::new should succeed")
    }

    // ── NBeatsBlock tests ─────────────────────────────────────────────────────

    #[test]
    fn test_nbeats_block_generic_shape() {
        let mut rng = StdRng::seed_from_u64(1);
        let block =
            NBeatsBlock::new(12, 4, 32, BasisType::Generic, 8, &mut rng).expect("block::new");
        let x = vec![0.5_f32; 12];
        let (bc, fc) = block.forward(&x).expect("block::forward");
        assert_eq!(bc.len(), 12, "backcast length");
        assert_eq!(fc.len(), 4, "forecast length");
    }

    #[test]
    fn test_nbeats_block_trend_shape() {
        let mut rng = StdRng::seed_from_u64(2);
        let block = NBeatsBlock::new(16, 5, 32, BasisType::Trend { degree: 3 }, 4, &mut rng)
            .expect("block::new");
        let x = vec![1.0_f32; 16];
        let (bc, fc) = block.forward(&x).expect("block::forward");
        assert_eq!(bc.len(), 16);
        assert_eq!(fc.len(), 5);
    }

    #[test]
    fn test_nbeats_block_seasonality_shape() {
        let mut rng = StdRng::seed_from_u64(3);
        let block = NBeatsBlock::new(
            24,
            6,
            32,
            BasisType::Seasonality { num_harmonics: 4 },
            9,
            &mut rng,
        )
        .expect("block::new");
        let x = vec![0.1_f32; 24];
        let (bc, fc) = block.forward(&x).expect("block::forward");
        assert_eq!(bc.len(), 24);
        assert_eq!(fc.len(), 6);
    }

    #[test]
    fn test_nbeats_block_wrong_input_errors() {
        let mut rng = StdRng::seed_from_u64(4);
        let block =
            NBeatsBlock::new(8, 2, 16, BasisType::Generic, 4, &mut rng).expect("block::new");
        let x = vec![0.0_f32; 5]; // wrong length
        assert!(block.forward(&x).is_err());
    }

    // ── Trend basis tests ─────────────────────────────────────────────────────

    #[test]
    fn test_trend_basis_first_col_all_ones() {
        let basis = NBeatsBlock::build_trend_basis(10, 3);
        assert_eq!(basis.len(), 10, "should have 10 rows");
        for (t, row) in basis.iter().enumerate() {
            assert_eq!(row.len(), 4, "degree 3 → 4 columns");
            let diff = (row[0] - 1.0_f32).abs();
            assert!(
                diff < 1e-6,
                "row {t}: first column should be 1.0, got {}",
                row[0]
            );
        }
    }

    #[test]
    fn test_trend_basis_linear_growth() {
        let basis = NBeatsBlock::build_trend_basis(5, 1);
        // Column 1 should contain t/T for t=0..4, T=5
        for (t, row) in basis.iter().enumerate() {
            let expected = t as f32 / 5.0;
            let diff = (row[1] - expected).abs();
            assert!(
                diff < 1e-5,
                "row {t}: linear term expected {expected}, got {}",
                row[1]
            );
        }
    }

    #[test]
    fn test_trend_basis_quadratic() {
        let basis = NBeatsBlock::build_trend_basis(5, 2);
        for (t, row) in basis.iter().enumerate() {
            let x = t as f32 / 5.0;
            let diff = (row[2] - x * x).abs();
            assert!(diff < 1e-5, "row {t}: quadratic mismatch");
        }
    }

    // ── Seasonality basis tests ───────────────────────────────────────────────

    #[test]
    fn test_seasonality_basis_shape() {
        let harmonics = 3;
        let basis = NBeatsBlock::build_seasonality_basis(12, harmonics);
        assert_eq!(basis.len(), 12);
        for row in &basis {
            assert_eq!(row.len(), 2 * harmonics + 1, "Fourier columns");
        }
    }

    #[test]
    fn test_seasonality_basis_constant_term() {
        let basis = NBeatsBlock::build_seasonality_basis(8, 2);
        for (t, row) in basis.iter().enumerate() {
            let diff = (row[0] - 1.0_f32).abs();
            assert!(
                diff < 1e-6,
                "row {t}: constant term should be 1.0, got {}",
                row[0]
            );
        }
    }

    #[test]
    fn test_seasonality_basis_first_cosine() {
        // row t: cos(2*pi*1*t/T)
        let size = 8_usize;
        let basis = NBeatsBlock::build_seasonality_basis(size, 1);
        for (t, row) in basis.iter().enumerate() {
            let expected = (2.0 * PI * t as f32 / size as f32).cos();
            let diff = (row[1] - expected).abs();
            assert!(diff < 1e-5, "row {t}: cos mismatch");
        }
    }

    // ── NBeatsStack tests ─────────────────────────────────────────────────────

    #[test]
    fn test_nbeats_stack_output_shape() {
        let mut rng = StdRng::seed_from_u64(5);
        let stack =
            NBeatsStack::new(12, 4, 32, BasisType::Generic, 8, 3, &mut rng).expect("stack::new");
        let x = vec![0.5_f32; 12];
        let fc = stack.forward(&x).expect("stack::forward");
        assert_eq!(fc.len(), 4);
    }

    #[test]
    fn test_nbeats_stack_trend_output_shape() {
        let mut rng = StdRng::seed_from_u64(6);
        let stack = NBeatsStack::new(24, 8, 32, BasisType::Trend { degree: 3 }, 4, 2, &mut rng)
            .expect("stack::new");
        let x = vec![1.0_f32; 24];
        let fc = stack.forward(&x).expect("stack::forward");
        assert_eq!(fc.len(), 8);
    }

    // ── NBeats full model tests ───────────────────────────────────────────────

    #[test]
    fn test_nbeats_full_model_output_shape() {
        let mut model = make_nbeats(
            24,
            6,
            32,
            8,
            vec![
                BasisType::Trend { degree: 3 },
                BasisType::Seasonality { num_harmonics: 4 },
                BasisType::Generic,
            ],
        );
        let x = vec![1.0_f32; 24];
        let fc = model.forward(&x).expect("NBeats::forward");
        assert_eq!(fc.len(), 6);
    }

    #[test]
    fn test_nbeats_full_model_finite_output() {
        let mut model = make_nbeats(12, 3, 16, 4, vec![BasisType::Generic]);
        let x: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let fc = model.forward(&x).expect("NBeats::forward");
        for &v in &fc {
            assert!(v.is_finite(), "forecast value is not finite: {v}");
        }
    }

    #[test]
    fn test_nbeats_decomposition_available() {
        let mut model = make_nbeats(
            12,
            3,
            16,
            4,
            vec![BasisType::Trend { degree: 2 }, BasisType::Generic],
        );
        let x = vec![0.5_f32; 12];
        let _ = model.forward(&x).expect("NBeats::forward");
        let bc = model.backcast_decomposition();
        let fc = model.forecast_decomposition();
        assert_eq!(
            bc.len(),
            2,
            "should have 2 stacks in backcast decomposition"
        );
        assert_eq!(
            fc.len(),
            2,
            "should have 2 stacks in forecast decomposition"
        );
    }

    #[test]
    fn test_nbeats_wrong_input_errors() {
        let mut model = make_nbeats(12, 3, 16, 4, vec![BasisType::Generic]);
        let x = vec![0.0_f32; 5]; // wrong length
        assert!(model.forward(&x).is_err());
    }

    // ── GRN tests ─────────────────────────────────────────────────────────────

    #[test]
    fn test_grn_output_shape() {
        let mut rng = StdRng::seed_from_u64(10);
        let grn = GRN::new(16, 16, &mut rng).expect("grn::new");
        let x = vec![0.5_f32; 16];
        let out = grn.forward(&x, None).expect("grn::forward");
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_grn_output_finite() {
        let mut rng = StdRng::seed_from_u64(11);
        let grn = GRN::new(8, 8, &mut rng).expect("grn::new");
        let x: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let out = grn.forward(&x, None).expect("grn::forward");
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_grn_with_context() {
        let mut rng = StdRng::seed_from_u64(12);
        let grn = GRN::new(8, 8, &mut rng).expect("grn::new");
        let x = vec![1.0_f32; 8];
        let ctx = vec![0.5_f32; 8];
        let out_ctx = grn.forward(&x, Some(&ctx)).expect("grn with context");
        let out_no_ctx = grn.forward(&x, None).expect("grn no context");
        // Outputs should differ when context is provided
        assert_ne!(out_ctx, out_no_ctx, "context should change output");
    }

    #[test]
    fn test_grn_projection_shape() {
        // input_dim != hidden_dim → skip_w should be Some
        let mut rng = StdRng::seed_from_u64(13);
        let grn = GRN::new(4, 16, &mut rng).expect("grn::new");
        let x = vec![0.1_f32; 4];
        let out = grn.forward(&x, None).expect("grn::forward");
        assert_eq!(out.len(), 16);
    }

    // ── VSN tests ─────────────────────────────────────────────────────────────

    #[test]
    fn test_vsn_output_shape() {
        let mut rng = StdRng::seed_from_u64(20);
        let vsn = VSN::new(4, 16, &mut rng).expect("vsn::new");
        let inputs: Vec<Vec<f32>> = (0..4).map(|_| vec![0.5_f32; 16]).collect();
        let (selected, weights) = vsn.forward(&inputs).expect("vsn::forward");
        assert_eq!(selected.len(), 16);
        assert_eq!(weights.len(), 4);
    }

    #[test]
    fn test_vsn_weights_sum_to_one() {
        let mut rng = StdRng::seed_from_u64(21);
        let vsn = VSN::new(5, 8, &mut rng).expect("vsn::new");
        let inputs: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32 * 0.1; 8]).collect();
        let (_selected, weights) = vsn.forward(&inputs).expect("vsn::forward");
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "weights should sum to 1, got {sum}"
        );
    }

    #[test]
    fn test_vsn_all_weights_non_negative() {
        let mut rng = StdRng::seed_from_u64(22);
        let vsn = VSN::new(3, 8, &mut rng).expect("vsn::new");
        let inputs: Vec<Vec<f32>> = (0..3).map(|_| vec![1.0_f32; 8]).collect();
        let (_selected, weights) = vsn.forward(&inputs).expect("vsn::forward");
        for &w in &weights {
            assert!(w >= 0.0, "all weights should be non-negative");
        }
    }

    // ── TFT tests ─────────────────────────────────────────────────────────────

    fn make_tft(
        past_f: usize,
        fut_f: usize,
        hd: usize,
        horizon: usize,
    ) -> TemporalFusionTransformer {
        let cfg = TftConfig {
            past_input_size: past_f,
            future_input_size: fut_f,
            hidden_dim: hd,
            num_heads: 2,
            num_lstm_layers: 1,
            dropout: 0.0,
            forecast_horizon: horizon,
            quantiles: vec![0.1, 0.5, 0.9],
        };
        TemporalFusionTransformer::new(cfg, 42).expect("TFT::new")
    }

    #[test]
    fn test_tft_output_shape() {
        let tft = make_tft(3, 2, 8, 4);
        let past = vec![0.5_f32; 6 * 3]; // 6 past steps × 3 features
        let future = vec![0.1_f32; 4 * 2]; // 4 horizon steps × 2 features
        let out = tft.forward(&past, &future).expect("TFT::forward");
        // horizon × n_quantiles = 4 × 3 = 12
        assert_eq!(out.len(), 12, "TFT output should be horizon×quantiles");
    }

    #[test]
    fn test_tft_output_finite() {
        let tft = make_tft(2, 1, 8, 3);
        let past: Vec<f32> = (0..10).map(|i| i as f32 * 0.05).collect(); // 5 steps × 2 feats
        let future = vec![0.2_f32; 3]; // 3 steps × 1 feat
        let out = tft.forward(&past, &future).expect("TFT::forward");
        for &v in &out {
            assert!(v.is_finite(), "TFT output should be finite, got {v}");
        }
    }

    #[test]
    fn test_tft_wrong_past_length_errors() {
        let tft = make_tft(3, 2, 8, 4);
        let past = vec![0.5_f32; 7]; // not divisible by 3
        let future = vec![0.1_f32; 4 * 2];
        assert!(tft.forward(&past, &future).is_err());
    }

    #[test]
    fn test_tft_wrong_future_length_errors() {
        let tft = make_tft(3, 2, 8, 4);
        let past = vec![0.5_f32; 6 * 3];
        let future = vec![0.1_f32; 5]; // wrong length
        assert!(tft.forward(&past, &future).is_err());
    }

    // ── Quantile loss tests ───────────────────────────────────────────────────

    #[test]
    fn test_quantile_loss_zero_for_perfect_prediction() {
        let preds = vec![1.0_f32, 2.0, 3.0];
        let targets = vec![1.0_f32, 2.0, 3.0];
        let loss =
            TemporalFusionTransformer::quantile_loss(&preds, &targets, 0.5).expect("quantile_loss");
        assert!(
            loss.abs() < 1e-6,
            "perfect prediction should have zero loss, got {loss}"
        );
    }

    #[test]
    fn test_quantile_loss_symmetry_at_half() {
        // At q=0.5 loss is symmetric: over-predict by 1 and under-predict by 1 give same loss
        let preds_over = vec![2.0_f32]; // predict 2, target 1 → over by 1
        let preds_under = vec![0.0_f32]; // predict 0, target 1 → under by 1
        let targets = vec![1.0_f32];
        let loss_over = TemporalFusionTransformer::quantile_loss(&preds_over, &targets, 0.5)
            .expect("quantile_loss failed");
        let loss_under = TemporalFusionTransformer::quantile_loss(&preds_under, &targets, 0.5)
            .expect("quantile_loss failed");
        assert!(
            (loss_over - loss_under).abs() < 1e-5,
            "q=0.5 loss should be symmetric: over={loss_over}, under={loss_under}"
        );
    }

    #[test]
    fn test_quantile_loss_asymmetry() {
        // At q=0.9 over-predicting should cost less than under-predicting
        let preds_over = vec![2.0_f32]; // over by 1
        let preds_under = vec![0.0_f32]; // under by 1
        let targets = vec![1.0_f32];
        let loss_over = TemporalFusionTransformer::quantile_loss(&preds_over, &targets, 0.9)
            .expect("quantile_loss failed");
        let loss_under = TemporalFusionTransformer::quantile_loss(&preds_under, &targets, 0.9)
            .expect("quantile_loss failed");
        // q=0.9: over cost = (1-0.9)*1 = 0.1; under cost = 0.9*1 = 0.9
        assert!(
            loss_under > loss_over,
            "at q=0.9 under-predicting should cost more: under={loss_under}, over={loss_over}"
        );
    }

    // ── TimeSeriesMetrics tests ───────────────────────────────────────────────

    #[test]
    fn test_mae_correct() {
        let preds = vec![1.0_f32, 3.0, 5.0];
        let targets = vec![2.0_f32, 2.0, 4.0];
        let mae = TimeSeriesMetrics::mae(&preds, &targets).expect("mae");
        // |1-2| + |3-2| + |5-4| = 3, /3 = 1.0
        assert!((mae - 1.0).abs() < 1e-5, "MAE should be 1.0, got {mae}");
    }

    #[test]
    fn test_mse_correct() {
        let preds = vec![0.0_f32, 2.0];
        let targets = vec![1.0_f32, 0.0];
        let mse = TimeSeriesMetrics::mse(&preds, &targets).expect("mse");
        // (1 + 4) / 2 = 2.5
        assert!((mse - 2.5).abs() < 1e-5, "MSE should be 2.5, got {mse}");
    }

    #[test]
    fn test_mape_50_percent() {
        // prediction is 1.5x target → 50% MAPE
        let targets = vec![2.0_f32, 4.0];
        let preds: Vec<f32> = targets.iter().map(|&t| t * 1.5).collect();
        let mape = TimeSeriesMetrics::mape(&preds, &targets).expect("mape");
        assert!((mape - 50.0).abs() < 1e-3, "MAPE should be 50%, got {mape}");
    }

    #[test]
    fn test_smape_finite() {
        let preds = vec![1.0_f32, 2.0, 3.0];
        let targets = vec![1.5_f32, 1.5, 2.5];
        let smape = TimeSeriesMetrics::smape(&preds, &targets).expect("smape");
        assert!(smape.is_finite(), "sMAPE should be finite, got {smape}");
        assert!(smape >= 0.0, "sMAPE should be non-negative");
    }

    #[test]
    fn test_mase_finite() {
        let preds = vec![1.0_f32, 2.0, 3.0];
        let targets = vec![1.1_f32, 1.9, 3.1];
        let in_sample = vec![0.5_f32, 0.6, 0.4];
        let mase = TimeSeriesMetrics::mase(&preds, &targets, &in_sample).expect("mase");
        assert!(
            mase.is_finite() && mase >= 0.0,
            "MASE should be finite and ≥0, got {mase}"
        );
    }

    #[test]
    fn test_wql_finite() {
        let quantiles = vec![0.1_f32, 0.5, 0.9];
        let targets = vec![1.0_f32, 2.0, 3.0];
        let preds_per_q: Vec<Vec<f32>> =
            quantiles.iter().map(|_| vec![1.5_f32, 2.5, 3.5]).collect();
        let wql = TimeSeriesMetrics::wql(&preds_per_q, &targets, &quantiles).expect("wql");
        assert!(
            wql.is_finite() && wql >= 0.0,
            "WQL should be finite and ≥0, got {wql}"
        );
    }

    #[test]
    fn test_mae_mse_zero_perfect() {
        let preds = vec![1.0_f32, 2.0, 3.0];
        let targets = preds.clone();
        let mae = TimeSeriesMetrics::mae(&preds, &targets).expect("mae");
        let mse = TimeSeriesMetrics::mse(&preds, &targets).expect("mse");
        assert!(mae.abs() < 1e-6);
        assert!(mse.abs() < 1e-6);
    }

    #[test]
    fn test_metrics_length_mismatch_errors() {
        let a = vec![1.0_f32; 3];
        let b = vec![1.0_f32; 4];
        assert!(TimeSeriesMetrics::mae(&a, &b).is_err());
        assert!(TimeSeriesMetrics::mse(&a, &b).is_err());
        assert!(TimeSeriesMetrics::mape(&a, &b).is_err());
        assert!(TimeSeriesMetrics::smape(&a, &b).is_err());
    }
}
