//! Neural Turing Machine (NTM) and Differentiable Neural Computer (DNC) — Track C, Round 12.
//!
//! Implements differentiable external memory architectures following:
//!
//! - **NTM**: Graves et al. (2014) "Neural Turing Machines"
//! - **DNC**: Graves et al. (2016) "Hybrid Computing Using a Neural Network with Dynamic External Memory"
//!
//! All computations use `f32` with plain `Vec<f32>` weight buffers — no `Tensor` / autograd
//! dependencies — making the module fully self-contained.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use tenflowers_neural::memory_networks::{NtmConfig, NeuralTuringMachine};
//!
//! let config = NtmConfig {
//!     input_dim: 8,
//!     output_dim: 8,
//!     controller_dim: 32,
//!     memory_size: 16,
//!     memory_dim: 8,
//!     num_read_heads: 1,
//!     num_write_heads: 1,
//!     shift_range: 1,
//! };
//! let ntm = NeuralTuringMachine::new(config)?;
//! let state = ntm.init_state();
//! let input = vec![0.0_f32; 8];
//! let (output, next_state) = ntm.step(&input, &state)?;
//! assert_eq!(output.len(), 8);
//! ```

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

type MemResult<T> = Result<T, String>;

// ─────────────────────────────────────────────────────────────────────────────
// Numerical helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically stable softmax over a slice.
fn softmax(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let denom = if sum > 0.0 { sum } else { 1.0 };
    exps.iter().map(|v| v / denom).collect()
}

/// Sigmoid: `1 / (1 + exp(-x))`, clamped for numerical safety.
#[inline]
fn sigmoid(x: f32) -> f32 {
    let xc = x.clamp(-88.0, 88.0);
    1.0 / (1.0 + (-xc).exp())
}

/// Tanh, using the standard library.
#[inline]
fn tanh(x: f32) -> f32 {
    x.tanh()
}

/// L2 norm of a slice.
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Dot product of two equal-length slices.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Xavier / Glorot uniform initialisation.
fn xavier_uniform(fan_in: usize, fan_out: usize, rng: &mut StdRng) -> Vec<f32> {
    let limit = (6.0_f64 / (fan_in + fan_out).max(1) as f64).sqrt() as f32;
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

/// Linear projection: `y[j] = Σ_i x[i] * W[i*out_dim + j] + b[j]`.
/// W is stored row-major: shape `[in_dim × out_dim]`.
fn linear(x: &[f32], w: &[f32], b: &[f32], out_dim: usize) -> MemResult<Vec<f32>> {
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
            "linear: bias mismatch: expected {out_dim}, got {}",
            b.len()
        ));
    }
    let mut out = b.to_vec();
    for i in 0..in_dim {
        let xi = x[i];
        if xi == 0.0 {
            continue;
        }
        for j in 0..out_dim {
            out[j] += xi * w[i * out_dim + j];
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoryBank
// ─────────────────────────────────────────────────────────────────────────────

/// External memory matrix of shape `[memory_size × memory_dim]`, stored row-major.
///
/// Provides differentiable read/write operations following the NTM formulation of
/// Graves et al. (2014).
#[derive(Debug, Clone)]
pub struct MemoryBank {
    /// Flat row-major storage: `memory[i * memory_dim + j]` = element (i, j).
    pub memory: Vec<f32>,
    /// N — number of memory slots (rows).
    pub memory_size: usize,
    /// M — dimension of each slot (columns).
    pub memory_dim: usize,
}

impl MemoryBank {
    /// Create a new `MemoryBank` initialised with small random values (seed=42).
    pub fn new(n: usize, m: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let scale = 0.01_f32;
        let memory: Vec<f32> = (0..n * m)
            .map(|_| {
                let u: f32 = rng.random();
                (u * 2.0 - 1.0) * scale
            })
            .collect();
        Self {
            memory,
            memory_size: n,
            memory_dim: m,
        }
    }

    /// Read from memory using attention weights: `r = Σ_i w_i * M_i`.
    ///
    /// `weights` must have length `memory_size`; returns a vector of length `memory_dim`.
    pub fn read(&self, weights: &[f32]) -> MemResult<Vec<f32>> {
        if weights.len() != self.memory_size {
            return Err(format!(
                "MemoryBank::read: weights length {} != memory_size {}",
                weights.len(),
                self.memory_size
            ));
        }
        let mut r = vec![0.0_f32; self.memory_dim];
        for i in 0..self.memory_size {
            let wi = weights[i];
            let row = self.get_row(i);
            for j in 0..self.memory_dim {
                r[j] += wi * row[j];
            }
        }
        Ok(r)
    }

    /// NTM write: `M_i ← M_i * (1 - w_i * e) + w_i * a`.
    ///
    /// - `weights`: attention weights, length `memory_size`.
    /// - `erase_vec`: erase vector, length `memory_dim`, values in \[0,1\].
    /// - `add_vec`: add vector, length `memory_dim`.
    pub fn write(&mut self, weights: &[f32], erase_vec: &[f32], add_vec: &[f32]) -> MemResult<()> {
        if weights.len() != self.memory_size {
            return Err(format!(
                "MemoryBank::write: weights length {} != memory_size {}",
                weights.len(),
                self.memory_size
            ));
        }
        if erase_vec.len() != self.memory_dim {
            return Err(format!(
                "MemoryBank::write: erase_vec length {} != memory_dim {}",
                erase_vec.len(),
                self.memory_dim
            ));
        }
        if add_vec.len() != self.memory_dim {
            return Err(format!(
                "MemoryBank::write: add_vec length {} != memory_dim {}",
                add_vec.len(),
                self.memory_dim
            ));
        }
        for i in 0..self.memory_size {
            let wi = weights[i];
            let base = i * self.memory_dim;
            for j in 0..self.memory_dim {
                // Erase then add
                let m_ij = self.memory[base + j];
                self.memory[base + j] = m_ij * (1.0 - wi * erase_vec[j]) + wi * add_vec[j];
            }
        }
        Ok(())
    }

    /// Borrow the i-th row as a slice.
    pub fn get_row(&self, i: usize) -> &[f32] {
        &self.memory[i * self.memory_dim..(i + 1) * self.memory_dim]
    }

    /// Cosine similarity between `key` (length `memory_dim`) and row `i`.
    /// Returns a value in `[-1, 1]`. Returns 0 if either vector is zero.
    pub fn cosine_similarity(&self, key: &[f32], row: usize) -> f32 {
        let row_data = self.get_row(row);
        let key_norm = l2_norm(key);
        let row_norm = l2_norm(row_data);
        if key_norm < 1e-9 || row_norm < 1e-9 {
            return 0.0;
        }
        let num = dot(key, row_data);
        num / (key_norm * row_norm)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NtmAddressing
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the NTM addressing mechanism.
#[derive(Debug, Clone)]
pub struct NtmAddressingConfig {
    /// N — number of memory slots.
    pub memory_size: usize,
    /// M — dimension of each memory slot.
    pub memory_dim: usize,
    /// `s` — maximum circular shift magnitude.  Shift range produces
    /// `2 * shift_range + 1` possible shift positions.
    pub shift_range: usize,
}

/// Mutable addressing state (one per head).
#[derive(Debug, Clone)]
pub struct NtmAddressingState {
    /// Current attention weights, shape `[memory_size]`, sums to 1.
    pub weights: Vec<f32>,
}

impl NtmAddressingState {
    /// Uniform initialisation: each weight = 1/N.
    pub fn uniform(memory_size: usize) -> Self {
        let w = 1.0 / memory_size.max(1) as f32;
        Self {
            weights: vec![w; memory_size],
        }
    }
}

/// Pure-function NTM addressing pipeline (Graves 2014, §3.3).
///
/// Steps applied in order:
/// 1. **Content addressing** — softmax(β · cosine(key, M_i))
/// 2. **Interpolation** — linear blend with previous weights via gate `g`
/// 3. **Convolutional shift** — circular 1-D convolution with shift distribution
/// 4. **Sharpening** — power-then-normalise with sharpness `γ ≥ 1`
pub struct NtmAddressing;

impl NtmAddressing {
    /// Full addressing pipeline.
    ///
    /// # Parameters
    /// - `key`: content key, length `config.memory_dim`
    /// - `beta`: key strength (scalar, ≥ 0)
    /// - `g`: interpolation gate ∈ (0, 1)
    /// - `shift_logits`: raw logits for shift distribution, length `2*shift_range+1`
    /// - `gamma`: sharpening exponent (≥ 1, enforced via `max(1.0, ·)`)
    /// - `memory`: current memory bank
    /// - `prev_weights`: previous step weights, length `memory_size`
    pub fn address(
        key: &[f32],
        beta: f32,
        g: f32,
        shift_logits: &[f32],
        gamma: f32,
        memory: &MemoryBank,
        prev_weights: &[f32],
    ) -> MemResult<Vec<f32>> {
        let n = memory.memory_size;
        if prev_weights.len() != n {
            return Err(format!(
                "NtmAddressing::address: prev_weights length {} != memory_size {n}",
                prev_weights.len()
            ));
        }

        // Step 1: Content addressing
        let content_logits: Vec<f32> = (0..n)
            .map(|i| {
                let cs = memory.cosine_similarity(key, i);
                beta * cs
            })
            .collect();
        let w_c = softmax(&content_logits);

        // Step 2: Interpolation gate (clamp g to avoid exact 0 or 1)
        let g_clamped = g.clamp(1e-6, 1.0 - 1e-6);
        let w_g: Vec<f32> = (0..n)
            .map(|i| g_clamped * w_c[i] + (1.0 - g_clamped) * prev_weights[i])
            .collect();

        // Step 3: Convolutional shift
        let shift_dist = softmax(shift_logits);
        let w_tilde = Self::convolve_circular(&w_g, &shift_dist, n);

        // Step 4: Sharpening
        let gamma_eff = gamma.max(1.0);
        let powered: Vec<f32> = w_tilde
            .iter()
            .map(|&v| v.max(0.0).powf(gamma_eff))
            .collect();
        let sum_pow: f32 = powered.iter().sum();
        let denom = if sum_pow > 1e-12 { sum_pow } else { 1.0 };
        let w: Vec<f32> = powered.iter().map(|v| v / denom).collect();

        Ok(w)
    }

    /// Circular 1-D convolution:
    /// `w̃(i) = Σ_j w_g(j) * s((i - j) mod N)`
    ///
    /// `shift_dist` length must equal the number of shift positions
    /// (`2*shift_range+1`).  The center element `shift_dist[shift_range]`
    /// corresponds to shift 0.
    pub fn convolve_circular(weights: &[f32], shift_dist: &[f32], size: usize) -> Vec<f32> {
        if size == 0 || shift_dist.is_empty() {
            return vec![0.0; size];
        }
        let s = shift_dist.len();
        let half = (s / 2) as isize; // shift_range
        let mut out = vec![0.0_f32; size];
        for i in 0..size {
            let mut val = 0.0_f32;
            for (k, &sk) in shift_dist.iter().enumerate() {
                let shift = k as isize - half;
                // j such that i - j ≡ shift (mod N), i.e. j = i - shift
                let j = ((i as isize - shift).rem_euclid(size as isize)) as usize;
                val += weights[j] * sk;
            }
            out[i] = val;
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NTM Read Head
// ─────────────────────────────────────────────────────────────────────────────

/// NTM read head.  Projects the controller output to addressing parameters and
/// performs the read operation.
///
/// Weight matrices stored row-major with shapes:
/// - `key_w`:   `[controller_dim × memory_dim]`
/// - `beta_w`:  `[controller_dim × 1]`
/// - `g_w`:     `[controller_dim × 1]`
/// - `shift_w`: `[controller_dim × (2*shift_range+1)]`
/// - `gamma_w`: `[controller_dim × 1]`
///
/// All bias vectors are zero-initialised.
#[derive(Debug, Clone)]
pub struct NtmReadHead {
    /// Weight: controller → key, shape `[controller_dim × memory_dim]`.
    pub key_w: Vec<f32>,
    /// Weight: controller → β (key strength), shape `[controller_dim × 1]`.
    pub beta_w: Vec<f32>,
    /// Weight: controller → g (interpolation gate), shape `[controller_dim × 1]`.
    pub g_w: Vec<f32>,
    /// Weight: controller → shift logits, shape `[controller_dim × (2*shift_range+1)]`.
    pub shift_w: Vec<f32>,
    /// Weight: controller → γ (sharpening), shape `[controller_dim × 1]`.
    pub gamma_w: Vec<f32>,
}

impl NtmReadHead {
    /// Create a new read head with Xavier-uniform initialisation.
    pub fn new(controller_dim: usize, config: &NtmAddressingConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let shift_dim = 2 * config.shift_range + 1;
        Self {
            key_w: xavier_uniform(controller_dim, config.memory_dim, &mut rng),
            beta_w: xavier_uniform(controller_dim, 1, &mut rng),
            g_w: xavier_uniform(controller_dim, 1, &mut rng),
            shift_w: xavier_uniform(controller_dim, shift_dim, &mut rng),
            gamma_w: xavier_uniform(controller_dim, 1, &mut rng),
        }
    }

    /// Emit addressing parameters from `controller_out`, run the full addressing
    /// pipeline, and return the read vector and updated weights.
    ///
    /// Returns `(read_vector: Vec<f32>, new_weights: Vec<f32>)`.
    pub fn emit(
        &self,
        controller_out: &[f32],
        memory: &MemoryBank,
        prev_w: &[f32],
        config: &NtmAddressingConfig,
    ) -> MemResult<(Vec<f32>, Vec<f32>)> {
        let cd = controller_out.len();
        let shift_dim = 2 * config.shift_range + 1;

        // Project to key (no bias)
        let zero_key = vec![0.0_f32; config.memory_dim];
        let key = linear(controller_out, &self.key_w, &zero_key, config.memory_dim)
            .map_err(|e| format!("ReadHead key projection: {e}"))?;

        // Project to β (key strength); apply softplus for positivity
        let zero1 = vec![0.0_f32; 1];
        let beta_raw = linear(controller_out, &self.beta_w, &zero1, 1)
            .map_err(|e| format!("ReadHead beta projection: {e}"))?;
        let beta = softplus_scalar(beta_raw[0]);

        // Project to g (interpolation gate); apply sigmoid
        let g_raw = linear(controller_out, &self.g_w, &zero1, 1)
            .map_err(|e| format!("ReadHead g projection: {e}"))?;
        let g = sigmoid(g_raw[0]);

        // Project to shift logits
        let zero_shift = vec![0.0_f32; shift_dim];
        let shift_logits = linear(controller_out, &self.shift_w, &zero_shift, shift_dim)
            .map_err(|e| format!("ReadHead shift projection: {e}"))?;

        // Project to γ (sharpening); apply softplus then shift to ensure ≥ 1
        let gamma_raw = linear(controller_out, &self.gamma_w, &zero1, 1)
            .map_err(|e| format!("ReadHead gamma projection: {e}"))?;
        let gamma = 1.0 + softplus_scalar(gamma_raw[0]);

        // Run addressing pipeline
        let new_w = NtmAddressing::address(&key, beta, g, &shift_logits, gamma, memory, prev_w)?;

        // Read from memory
        let read_vec = memory.read(&new_w)?;

        Ok((read_vec, new_w))
    }
}

/// Numerically stable softplus scalar: `log(1 + exp(x))`.
#[inline]
fn softplus_scalar(x: f32) -> f32 {
    if x > 15.0 {
        x
    } else if x < -15.0 {
        x.exp()
    } else {
        (1.0_f32 + x.exp()).ln()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NTM Write Head
// ─────────────────────────────────────────────────────────────────────────────

/// NTM write head.  Extends [`NtmReadHead`] with erase and add vector projections.
///
/// Additional weight matrices:
/// - `erase_w`: `[controller_dim × memory_dim]`
/// - `add_w`:   `[controller_dim × memory_dim]`
#[derive(Debug, Clone)]
pub struct NtmWriteHead {
    /// Weight: controller → key, shape `[controller_dim × memory_dim]`.
    pub key_w: Vec<f32>,
    /// Weight: controller → β, shape `[controller_dim × 1]`.
    pub beta_w: Vec<f32>,
    /// Weight: controller → g, shape `[controller_dim × 1]`.
    pub g_w: Vec<f32>,
    /// Weight: controller → shift logits, shape `[controller_dim × (2*shift_range+1)]`.
    pub shift_w: Vec<f32>,
    /// Weight: controller → γ, shape `[controller_dim × 1]`.
    pub gamma_w: Vec<f32>,
    /// Weight: controller → erase vector (sigmoid), shape `[controller_dim × memory_dim]`.
    pub erase_w: Vec<f32>,
    /// Weight: controller → add vector, shape `[controller_dim × memory_dim]`.
    pub add_w: Vec<f32>,
}

impl NtmWriteHead {
    /// Create a new write head with Xavier-uniform initialisation.
    pub fn new(controller_dim: usize, config: &NtmAddressingConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let shift_dim = 2 * config.shift_range + 1;
        Self {
            key_w: xavier_uniform(controller_dim, config.memory_dim, &mut rng),
            beta_w: xavier_uniform(controller_dim, 1, &mut rng),
            g_w: xavier_uniform(controller_dim, 1, &mut rng),
            shift_w: xavier_uniform(controller_dim, shift_dim, &mut rng),
            gamma_w: xavier_uniform(controller_dim, 1, &mut rng),
            erase_w: xavier_uniform(controller_dim, config.memory_dim, &mut rng),
            add_w: xavier_uniform(controller_dim, config.memory_dim, &mut rng),
        }
    }

    /// Emit addressing parameters, perform the write operation on `memory`, and
    /// return the updated attention weights.
    pub fn emit_and_write(
        &self,
        controller_out: &[f32],
        memory: &mut MemoryBank,
        prev_w: &[f32],
        config: &NtmAddressingConfig,
    ) -> MemResult<Vec<f32>> {
        let shift_dim = 2 * config.shift_range + 1;
        let zero_key = vec![0.0_f32; config.memory_dim];
        let zero1 = vec![0.0_f32; 1];
        let zero_shift = vec![0.0_f32; shift_dim];

        // Project addressing parameters
        let key = linear(controller_out, &self.key_w, &zero_key, config.memory_dim)
            .map_err(|e| format!("WriteHead key: {e}"))?;
        let beta_raw = linear(controller_out, &self.beta_w, &zero1, 1)
            .map_err(|e| format!("WriteHead beta: {e}"))?;
        let beta = softplus_scalar(beta_raw[0]);
        let g_raw = linear(controller_out, &self.g_w, &zero1, 1)
            .map_err(|e| format!("WriteHead g: {e}"))?;
        let g = sigmoid(g_raw[0]);
        let shift_logits = linear(controller_out, &self.shift_w, &zero_shift, shift_dim)
            .map_err(|e| format!("WriteHead shift: {e}"))?;
        let gamma_raw = linear(controller_out, &self.gamma_w, &zero1, 1)
            .map_err(|e| format!("WriteHead gamma: {e}"))?;
        let gamma = 1.0 + softplus_scalar(gamma_raw[0]);

        // Compute new weights via addressing pipeline
        let new_w = NtmAddressing::address(&key, beta, g, &shift_logits, gamma, memory, prev_w)?;

        // Project erase and add vectors
        let erase_raw = linear(controller_out, &self.erase_w, &zero_key, config.memory_dim)
            .map_err(|e| format!("WriteHead erase: {e}"))?;
        let erase_vec: Vec<f32> = erase_raw.iter().map(|&v| sigmoid(v)).collect();

        let add_vec = linear(controller_out, &self.add_w, &zero_key, config.memory_dim)
            .map_err(|e| format!("WriteHead add: {e}"))?;

        // Perform memory write
        memory.write(&new_w, &erase_vec, &add_vec)?;

        Ok(new_w)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NTM Controller (LSTM)
// ─────────────────────────────────────────────────────────────────────────────

/// Simple LSTM controller for the NTM.
///
/// The augmented input to the LSTM is `[input ∥ read_vectors]` (concatenation
/// of the actual input with all read vectors from the previous step), so
/// `input_dim` here refers to the *augmented* input dimension:
/// `original_input_dim + num_read_heads * memory_dim`.
///
/// LSTM gate weight matrices are stored row-major with shape
/// `[hidden_dim × (input_dim + hidden_dim)]`.
#[derive(Debug, Clone)]
pub struct NtmController {
    /// Augmented input dimension = original_input_dim + num_read_heads * memory_dim.
    pub input_dim: usize,
    /// Hidden state dimension.
    pub hidden_dim: usize,
    /// Input gate weights, shape `[hidden_dim × (input_dim + hidden_dim)]`.
    pub w_i: Vec<f32>,
    /// Input gate bias, length `hidden_dim`.
    pub b_i: Vec<f32>,
    /// Forget gate weights.
    pub w_f: Vec<f32>,
    /// Forget gate bias.
    pub b_f: Vec<f32>,
    /// Output gate weights.
    pub w_o: Vec<f32>,
    /// Output gate bias.
    pub b_o: Vec<f32>,
    /// Cell gate weights.
    pub w_g: Vec<f32>,
    /// Cell gate bias.
    pub b_g: Vec<f32>,
}

impl NtmController {
    /// Create a new LSTM controller with small normal-initialised weights.
    pub fn new(input_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let combined = input_dim + hidden_dim;
        let scale = 0.01_f32;

        let make_w = |rng: &mut StdRng| -> Vec<f32> {
            normal_init(hidden_dim * combined, rng)
                .iter()
                .map(|v| v * scale)
                .collect()
        };
        let make_b = |val: f32| vec![val; hidden_dim];

        Self {
            input_dim,
            hidden_dim,
            w_i: make_w(&mut rng),
            b_i: make_b(0.0),
            w_f: make_w(&mut rng),
            b_f: make_b(1.0), // forget gate biased towards 1 (standard LSTM trick)
            w_o: make_w(&mut rng),
            b_o: make_b(0.0),
            w_g: make_w(&mut rng),
            b_g: make_b(0.0),
        }
    }

    /// One LSTM step.
    ///
    /// `input` length must equal `self.input_dim`.
    /// `h_prev` and `c_prev` lengths must equal `self.hidden_dim`.
    ///
    /// Returns `(h_new, c_new)`.
    pub fn step(
        &self,
        input: &[f32],
        h_prev: &[f32],
        c_prev: &[f32],
    ) -> MemResult<(Vec<f32>, Vec<f32>)> {
        if input.len() != self.input_dim {
            return Err(format!(
                "NtmController::step: input length {} != input_dim {}",
                input.len(),
                self.input_dim
            ));
        }
        if h_prev.len() != self.hidden_dim {
            return Err(format!(
                "NtmController::step: h_prev length {} != hidden_dim {}",
                h_prev.len(),
                self.hidden_dim
            ));
        }
        if c_prev.len() != self.hidden_dim {
            return Err(format!(
                "NtmController::step: c_prev length {} != hidden_dim {}",
                c_prev.len(),
                self.hidden_dim
            ));
        }

        // Concatenate [input ∥ h_prev]
        let combined: usize = self.input_dim + self.hidden_dim;
        let mut xh = Vec::with_capacity(combined);
        xh.extend_from_slice(input);
        xh.extend_from_slice(h_prev);

        // Gate computations
        let i_gate = linear(&xh, &self.w_i, &self.b_i, self.hidden_dim)?;
        let f_gate = linear(&xh, &self.w_f, &self.b_f, self.hidden_dim)?;
        let o_gate = linear(&xh, &self.w_o, &self.b_o, self.hidden_dim)?;
        let g_gate = linear(&xh, &self.w_g, &self.b_g, self.hidden_dim)?;

        // Apply non-linearities and compute new cell / hidden states
        let mut c_new = Vec::with_capacity(self.hidden_dim);
        let mut h_new = Vec::with_capacity(self.hidden_dim);
        for k in 0..self.hidden_dim {
            let i_k = sigmoid(i_gate[k]);
            let f_k = sigmoid(f_gate[k]);
            let o_k = sigmoid(o_gate[k]);
            let g_k = tanh(g_gate[k]);
            let c_k = f_k * c_prev[k] + i_k * g_k;
            let h_k = o_k * tanh(c_k);
            c_new.push(c_k);
            h_new.push(h_k);
        }

        Ok((h_new, c_new))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NeuralTuringMachine
// ─────────────────────────────────────────────────────────────────────────────

/// Full NTM configuration.
#[derive(Debug, Clone)]
pub struct NtmConfig {
    /// Dimension of the external input vector.
    pub input_dim: usize,
    /// Dimension of the output vector.
    pub output_dim: usize,
    /// Hidden state dimension of the LSTM controller.
    pub controller_dim: usize,
    /// N — number of memory slots.
    pub memory_size: usize,
    /// M — dimension of each memory slot.
    pub memory_dim: usize,
    /// Number of read heads (R).
    pub num_read_heads: usize,
    /// Number of write heads (W).
    pub num_write_heads: usize,
    /// Maximum shift magnitude for circular convolution addressing.
    pub shift_range: usize,
}

/// Full NTM recurrent state.
#[derive(Debug, Clone)]
pub struct NtmState {
    /// Controller hidden state, length `controller_dim`.
    pub h: Vec<f32>,
    /// Controller cell state, length `controller_dim`.
    pub c: Vec<f32>,
    /// External memory bank.
    pub memory: MemoryBank,
    /// Attention weights per read head, shape `[num_read_heads × memory_size]`.
    pub read_weights: Vec<Vec<f32>>,
    /// Attention weights per write head, shape `[num_write_heads × memory_size]`.
    pub write_weights: Vec<Vec<f32>>,
    /// Read vectors from the current step, shape `[num_read_heads × memory_dim]`.
    pub read_vectors: Vec<Vec<f32>>,
}

/// A complete Neural Turing Machine.
#[derive(Debug, Clone)]
pub struct NeuralTuringMachine {
    /// Model configuration.
    pub config: NtmConfig,
    /// LSTM controller.
    pub controller: NtmController,
    /// Read heads.
    pub read_heads: Vec<NtmReadHead>,
    /// Write heads.
    pub write_heads: Vec<NtmWriteHead>,
    /// Output projection weight, shape `[(controller_dim + R*memory_dim) × output_dim]`.
    pub output_w: Vec<f32>,
    /// Output projection bias, length `output_dim`.
    pub output_b: Vec<f32>,
}

impl NeuralTuringMachine {
    /// Construct a new NTM with random initialisations.
    pub fn new(config: NtmConfig) -> MemResult<Self> {
        if config.input_dim == 0 {
            return Err("NeuralTuringMachine::new: input_dim must be > 0".to_string());
        }
        if config.output_dim == 0 {
            return Err("NeuralTuringMachine::new: output_dim must be > 0".to_string());
        }
        if config.controller_dim == 0 {
            return Err("NeuralTuringMachine::new: controller_dim must be > 0".to_string());
        }
        if config.memory_size == 0 || config.memory_dim == 0 {
            return Err(
                "NeuralTuringMachine::new: memory_size and memory_dim must be > 0".to_string(),
            );
        }
        if config.num_read_heads == 0 {
            return Err("NeuralTuringMachine::new: num_read_heads must be > 0".to_string());
        }
        if config.num_write_heads == 0 {
            return Err("NeuralTuringMachine::new: num_write_heads must be > 0".to_string());
        }

        let addr_config = NtmAddressingConfig {
            memory_size: config.memory_size,
            memory_dim: config.memory_dim,
            shift_range: config.shift_range,
        };

        // Controller input = external input + R read vectors concatenated
        let augmented_input_dim = config.input_dim + config.num_read_heads * config.memory_dim;
        let controller = NtmController::new(augmented_input_dim, config.controller_dim, 1000);

        let read_heads: Vec<NtmReadHead> = (0..config.num_read_heads)
            .map(|i| NtmReadHead::new(config.controller_dim, &addr_config, 2000 + i as u64))
            .collect();

        let write_heads: Vec<NtmWriteHead> = (0..config.num_write_heads)
            .map(|i| NtmWriteHead::new(config.controller_dim, &addr_config, 3000 + i as u64))
            .collect();

        // Output dim = controller_dim + R * memory_dim
        let out_in_dim = config.controller_dim + config.num_read_heads * config.memory_dim;
        let mut rng = StdRng::seed_from_u64(4000);
        let output_w = xavier_uniform(out_in_dim, config.output_dim, &mut rng);
        let output_b = vec![0.0_f32; config.output_dim];

        Ok(Self {
            config,
            controller,
            read_heads,
            write_heads,
            output_w,
            output_b,
        })
    }

    /// Create an initial NTM state with uniform attention and zeroed controller.
    pub fn init_state(&self) -> NtmState {
        let cfg = &self.config;
        let h = vec![0.0_f32; cfg.controller_dim];
        let c = vec![0.0_f32; cfg.controller_dim];
        let memory = MemoryBank::new(cfg.memory_size, cfg.memory_dim);
        let uniform_w = || {
            let v = 1.0 / cfg.memory_size.max(1) as f32;
            vec![v; cfg.memory_size]
        };
        let read_weights: Vec<Vec<f32>> = (0..cfg.num_read_heads).map(|_| uniform_w()).collect();
        let write_weights: Vec<Vec<f32>> = (0..cfg.num_write_heads).map(|_| uniform_w()).collect();
        let read_vectors: Vec<Vec<f32>> = (0..cfg.num_read_heads)
            .map(|_| vec![0.0_f32; cfg.memory_dim])
            .collect();
        NtmState {
            h,
            c,
            memory,
            read_weights,
            write_weights,
            read_vectors,
        }
    }

    /// Perform one NTM step: given `input` and the current `state`, produce an
    /// output vector and a new state.
    pub fn step(&self, input: &[f32], state: &NtmState) -> MemResult<(Vec<f32>, NtmState)> {
        let cfg = &self.config;

        if input.len() != cfg.input_dim {
            return Err(format!(
                "NeuralTuringMachine::step: input length {} != input_dim {}",
                input.len(),
                cfg.input_dim
            ));
        }

        let addr_config = NtmAddressingConfig {
            memory_size: cfg.memory_size,
            memory_dim: cfg.memory_dim,
            shift_range: cfg.shift_range,
        };

        // Augmented input = [input ∥ prev_read_vectors...]
        let mut aug_input = Vec::with_capacity(cfg.input_dim + cfg.num_read_heads * cfg.memory_dim);
        aug_input.extend_from_slice(input);
        for rv in &state.read_vectors {
            aug_input.extend_from_slice(rv);
        }

        // Controller step
        let (h_new, c_new) = self.controller.step(&aug_input, &state.h, &state.c)?;

        // Clone memory for mutation
        let mut memory_new = state.memory.clone();

        // Write heads first (order: write then read, standard NTM)
        let mut new_write_weights = Vec::with_capacity(cfg.num_write_heads);
        for (wh_idx, write_head) in self.write_heads.iter().enumerate() {
            let prev_ww = &state.write_weights[wh_idx];
            let ww = write_head.emit_and_write(&h_new, &mut memory_new, prev_ww, &addr_config)?;
            new_write_weights.push(ww);
        }

        // Read heads
        let mut new_read_weights = Vec::with_capacity(cfg.num_read_heads);
        let mut new_read_vectors = Vec::with_capacity(cfg.num_read_heads);
        for (rh_idx, read_head) in self.read_heads.iter().enumerate() {
            let prev_rw = &state.read_weights[rh_idx];
            let (rv, rw) = read_head.emit(&h_new, &memory_new, prev_rw, &addr_config)?;
            new_read_weights.push(rw);
            new_read_vectors.push(rv);
        }

        // Output projection: [h ∥ read_vectors...]
        let mut out_input =
            Vec::with_capacity(cfg.controller_dim + cfg.num_read_heads * cfg.memory_dim);
        out_input.extend_from_slice(&h_new);
        for rv in &new_read_vectors {
            out_input.extend_from_slice(rv);
        }
        let out_in_dim = out_input.len();
        if self.output_w.len() != out_in_dim * cfg.output_dim {
            return Err(format!(
                "NeuralTuringMachine::step: output_w shape mismatch: {}×{} != {}",
                out_in_dim,
                cfg.output_dim,
                self.output_w.len()
            ));
        }
        let output = linear(&out_input, &self.output_w, &self.output_b, cfg.output_dim)?;

        let new_state = NtmState {
            h: h_new,
            c: c_new,
            memory: memory_new,
            read_weights: new_read_weights,
            write_weights: new_write_weights,
            read_vectors: new_read_vectors,
        };

        Ok((output, new_state))
    }

    /// Process a full input sequence, returning one output vector per time step.
    pub fn forward_sequence(&self, inputs: &[Vec<f32>]) -> MemResult<Vec<Vec<f32>>> {
        let mut state = self.init_state();
        let mut outputs = Vec::with_capacity(inputs.len());
        for x in inputs {
            let (out, next_state) = self.step(x, &state)?;
            outputs.push(out);
            state = next_state;
        }
        Ok(outputs)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Differentiable Neural Computer (DNC)
// ─────────────────────────────────────────────────────────────────────────────

/// Full DNC recurrent state (Graves et al. 2016).
///
/// Memory layout:
/// - `memory`:        `[N × M]` flat row-major
/// - `usage`:         `[N]` — slot usage in [0, 1]
/// - `temporal_link`: `[N × N]` — temporal adjacency matrix (diagonal should be 0)
/// - `precedence`:    `[N]` — write-order precedence weighting
/// - `read_weights`:  `[R × N]` — per-head read attention
/// - `write_weights`: `[N]` — write attention
#[derive(Debug, Clone)]
pub struct DncState {
    /// External memory, shape `[N × M]`.
    pub memory: Vec<f32>,
    /// Usage vector, shape `[N]`.
    pub usage: Vec<f32>,
    /// Temporal link matrix, shape `[N × N]`.
    pub temporal_link: Vec<f32>,
    /// Precedence weights, shape `[N]`.
    pub precedence: Vec<f32>,
    /// Read attention weights, shape `[R × N]`.
    pub read_weights: Vec<Vec<f32>>,
    /// Write attention weights, shape `[N]`.
    pub write_weights: Vec<f32>,
}

/// DNC memory dimensions.
#[derive(Debug, Clone, Copy)]
pub struct DncConfig {
    /// N — number of memory slots.
    pub memory_size: usize,
    /// M — slot dimension.
    pub memory_dim: usize,
    /// R — number of read heads.
    pub num_read_heads: usize,
}

impl DncState {
    /// Create an initial DNC state.
    pub fn new(config: DncConfig) -> Self {
        let n = config.memory_size;
        let m = config.memory_dim;
        let r = config.num_read_heads;
        let uniform_rw = 1.0 / n.max(1) as f32;
        Self {
            memory: vec![0.0_f32; n * m],
            usage: vec![0.0_f32; n],
            temporal_link: vec![0.0_f32; n * n],
            precedence: vec![0.0_f32; n],
            read_weights: vec![vec![uniform_rw; n]; r],
            write_weights: vec![uniform_rw; n],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DNC helper: content-based addressing
// ─────────────────────────────────────────────────────────────────────────────

/// Content-based lookup in flat row-major memory `mem` of shape `[n × m]`.
fn dnc_content_weights(mem: &[f32], n: usize, m: usize, key: &[f32], strength: f32) -> Vec<f32> {
    let logits: Vec<f32> = (0..n)
        .map(|i| {
            let row = &mem[i * m..(i + 1) * m];
            let kn = l2_norm(key);
            let rn = l2_norm(row);
            if kn < 1e-9 || rn < 1e-9 {
                0.0
            } else {
                strength * dot(key, row) / (kn * rn)
            }
        })
        .collect();
    softmax(&logits)
}

// ─────────────────────────────────────────────────────────────────────────────
// DNC allocation weighting
// ─────────────────────────────────────────────────────────────────────────────

/// Compute allocation weights from usage vector.
///
/// Implements the DNC allocation: slots are ranked by usage (ascending),
/// and allocation weights are assigned such that less-used slots receive
/// proportionally more weight.
///
/// Reference: Graves et al. 2016, eq. 4.
pub fn allocation_weighting(usage: &[f32]) -> Vec<f32> {
    let n = usage.len();
    if n == 0 {
        return Vec::new();
    }
    // Sort indices by usage ascending
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        usage[a]
            .partial_cmp(&usage[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut alloc = vec![0.0_f32; n];
    let mut running_prod = 1.0_f32;
    for (rank, &i) in indices.iter().enumerate() {
        // a(i) = (1 - u_i) * Π_{j: φ(j) < φ(i)} u_j
        alloc[i] = (1.0 - usage[i]) * running_prod;
        running_prod *= usage[i];
        // running_prod can underflow for large N; clamp for stability
        if running_prod < 1e-30 {
            running_prod = 0.0;
        }
        let _ = rank; // used implicitly via running_prod
    }
    // Normalise so allocation weights sum to ≤ 1
    let sum: f32 = alloc.iter().sum();
    if sum > 1e-12 {
        alloc.iter_mut().for_each(|v| *v /= sum);
    }
    alloc
}

// ─────────────────────────────────────────────────────────────────────────────
// DNC core operations
// ─────────────────────────────────────────────────────────────────────────────

/// Perform one DNC write step.
///
/// # Parameters
/// - `state`: current DNC state (consumed and replaced)
/// - `write_key`: content key for write addressing, length `M`
/// - `write_strength`: key strength β
/// - `erase`: erase vector, length `M` (values in \[0,1\] after sigmoid)
/// - `add`: add vector, length `M`
/// - `write_gate`: scalar gate ∈ (0,1) controlling overall write magnitude
/// - `alloc_gate`: scalar ∈ (0,1) mixing content-based vs allocation weighting
///
/// Returns the updated `DncState`.
pub fn dnc_write(
    state: &DncState,
    config: DncConfig,
    write_key: &[f32],
    write_strength: f32,
    erase: &[f32],
    add: &[f32],
    write_gate: f32,
    alloc_gate: f32,
) -> MemResult<DncState> {
    let n = config.memory_size;
    let m = config.memory_dim;

    if write_key.len() != m {
        return Err(format!(
            "dnc_write: write_key length {} != memory_dim {m}",
            write_key.len()
        ));
    }
    if erase.len() != m {
        return Err(format!(
            "dnc_write: erase length {} != memory_dim {m}",
            erase.len()
        ));
    }
    if add.len() != m {
        return Err(format!(
            "dnc_write: add length {} != memory_dim {m}",
            add.len()
        ));
    }

    // Content write weights
    let c_w = dnc_content_weights(&state.memory, n, m, write_key, write_strength);

    // Allocation weights
    let a_w = allocation_weighting(&state.usage);

    // Interpolated write weights
    let g_w = write_gate.clamp(0.0, 1.0);
    let g_a = alloc_gate.clamp(0.0, 1.0);
    // w_write = g_w * (g_a * a_w + (1 - g_a) * c_w)
    let write_w: Vec<f32> = (0..n)
        .map(|i| g_w * (g_a * a_w[i] + (1.0 - g_a) * c_w[i]))
        .collect();

    // Erase then add memory update
    let erase_sig: Vec<f32> = erase.iter().map(|&v| sigmoid(v)).collect();
    let mut new_memory = state.memory.clone();
    for i in 0..n {
        let wi = write_w[i];
        let base = i * m;
        for j in 0..m {
            let m_ij = new_memory[base + j];
            new_memory[base + j] = m_ij * (1.0 - wi * erase_sig[j]) + wi * add[j];
        }
    }

    // Usage update: u_new = (u_prev + w_write - u_prev * w_write)
    let new_usage: Vec<f32> = state
        .usage
        .iter()
        .zip(write_w.iter())
        .map(|(&u, &w)| (u + w - u * w).clamp(0.0, 1.0))
        .collect();

    // Temporal link matrix update:
    // L_new(i,j) = (1 - w_write_i - w_write_j) * L_old(i,j) + w_write_i * p_prev(j)
    let mut new_link = state.temporal_link.clone();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                new_link[i * n + j] = 0.0; // diagonal always 0
            } else {
                let old_l = state.temporal_link[i * n + j];
                new_link[i * n + j] =
                    (1.0 - write_w[i] - write_w[j]) * old_l + write_w[i] * state.precedence[j];
            }
        }
    }

    // Precedence update: p_new = (1 - Σ w_write) * p_prev + w_write
    let w_sum: f32 = write_w.iter().sum();
    let new_precedence: Vec<f32> = state
        .precedence
        .iter()
        .zip(write_w.iter())
        .map(|(&p, &w)| (1.0 - w_sum) * p + w)
        .collect();

    Ok(DncState {
        memory: new_memory,
        usage: new_usage,
        temporal_link: new_link,
        precedence: new_precedence,
        read_weights: state.read_weights.clone(),
        write_weights: write_w,
    })
}

/// DNC read modes per head.
#[derive(Debug, Clone)]
pub struct DncReadModes {
    /// Weight for content-based lookup (scalar per head).
    pub content_lookup: f32,
    /// Weight for temporal forward traversal (scalar per head).
    pub forward: f32,
    /// Weight for temporal backward traversal (scalar per head).
    pub backward: f32,
}

impl DncReadModes {
    /// Content-only mode (no temporal traversal).
    pub fn content_only() -> Self {
        Self {
            content_lookup: 1.0,
            forward: 0.0,
            backward: 0.0,
        }
    }
}

/// Perform one DNC read step and return concatenated read vectors.
///
/// # Parameters
/// - `state`: current DNC state
/// - `read_keys`: slice of read keys, one per head, each length `M`
/// - `read_strengths`: key strength per head, length `R`
/// - `read_modes`: read mode weights per head, length `R`
///
/// Returns a flat vector of length `R × M` (read vectors concatenated).
pub fn dnc_read(
    state: &DncState,
    config: DncConfig,
    read_keys: &[Vec<f32>],
    read_strengths: &[f32],
    read_modes: &[DncReadModes],
) -> MemResult<Vec<f32>> {
    let n = config.memory_size;
    let m = config.memory_dim;
    let r = config.num_read_heads;

    if read_keys.len() != r {
        return Err(format!(
            "dnc_read: read_keys length {} != num_read_heads {r}",
            read_keys.len()
        ));
    }
    if read_strengths.len() != r {
        return Err(format!(
            "dnc_read: read_strengths length {} != num_read_heads {r}",
            read_strengths.len()
        ));
    }
    if read_modes.len() != r {
        return Err(format!(
            "dnc_read: read_modes length {} != num_read_heads {r}",
            read_modes.len()
        ));
    }

    let mut all_reads = Vec::with_capacity(r * m);

    for head in 0..r {
        // Content weights
        let c_w = dnc_content_weights(&state.memory, n, m, &read_keys[head], read_strengths[head]);

        // Forward temporal weights: f_w = L^T * r_prev
        let prev_rw = &state.read_weights[head];
        let mut fwd_w = vec![0.0_f32; n];
        for i in 0..n {
            for j in 0..n {
                // L^T[i,j] = L[j,i]
                fwd_w[i] += state.temporal_link[j * n + i] * prev_rw[j];
            }
        }

        // Backward temporal weights: b_w = L * r_prev
        let mut bwd_w = vec![0.0_f32; n];
        for i in 0..n {
            for j in 0..n {
                bwd_w[i] += state.temporal_link[i * n + j] * prev_rw[j];
            }
        }

        // Normalise fwd and bwd
        let fwd_sum: f32 = fwd_w.iter().sum();
        let bwd_sum: f32 = bwd_w.iter().sum();
        if fwd_sum > 1e-12 {
            fwd_w.iter_mut().for_each(|v| *v /= fwd_sum);
        }
        if bwd_sum > 1e-12 {
            bwd_w.iter_mut().for_each(|v| *v /= bwd_sum);
        }

        // Mix modes: normalise mode coefficients first (softmax-like)
        let mode = &read_modes[head];
        let mode_sum = mode.content_lookup + mode.forward + mode.backward;
        let denom = if mode_sum > 1e-12 { mode_sum } else { 1.0 };
        let wc = mode.content_lookup / denom;
        let wf = mode.forward / denom;
        let wb = mode.backward / denom;

        let read_w: Vec<f32> = (0..n)
            .map(|i| wc * c_w[i] + wf * fwd_w[i] + wb * bwd_w[i])
            .collect();

        // Read from memory
        for j in 0..m {
            let val: f32 = (0..n).map(|i| read_w[i] * state.memory[i * m + j]).sum();
            all_reads.push(val);
        }
    }

    Ok(all_reads)
}

/// Update the usage vector when free gates are applied.
///
/// `u_new[i] = (u_prev[i] - f_t[i] * r_prev[i]) * (1 - w_write[i])`
///
/// where `f_t[i]` is the free gate for slot i (in \[0,1\]) and `r_prev` is the
/// previous read weight for a single head (simplified single-head version).
///
/// # Parameters
/// - `u_prev`: previous usage vector, length `N`
/// - `w_prev`: previous read weight for a head, length `N`
/// - `f_t`:    free gate per slot, length `N`, values in \[0,1\]
///
/// Returns the updated usage vector.
pub fn usage_update(u_prev: &[f32], w_prev: &[f32], f_t: &[f32]) -> MemResult<Vec<f32>> {
    let n = u_prev.len();
    if w_prev.len() != n {
        return Err(format!(
            "usage_update: w_prev length {} != u_prev length {n}",
            w_prev.len()
        ));
    }
    if f_t.len() != n {
        return Err(format!(
            "usage_update: f_t length {} != u_prev length {n}",
            f_t.len()
        ));
    }
    // Memory retention ψ = Π_h (1 - f_h * rw_h) (single head here)
    let psi: Vec<f32> = (0..n)
        .map(|i| 1.0 - f_t[i].clamp(0.0, 1.0) * w_prev[i])
        .collect();
    // u_new = (u_prev + w_write - u_prev * w_write) * ψ
    // (In the simplified one-head version the write term is already factored
    //  into u_prev; we apply only retention here.)
    let new_usage: Vec<f32> = (0..n)
        .map(|i| (u_prev[i] * psi[i]).clamp(0.0, 1.0))
        .collect();
    Ok(new_usage)
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── MemoryBank ─────────────────────────────────────────────────────────

    #[test]
    fn test_memory_bank_read_weighted_sum() {
        // 3 slots of dim 2; memory = identity-like for easy verification
        let mut mb = MemoryBank::new(3, 2);
        mb.memory = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
        let weights = vec![1.0, 0.0, 0.0];
        let r = mb.read(&weights).expect("operation should succeed");
        assert!((r[0] - 1.0).abs() < 1e-6, "r[0]={}", r[0]);
        assert!((r[1] - 0.0).abs() < 1e-6, "r[1]={}", r[1]);
    }

    #[test]
    fn test_memory_bank_read_interpolated() {
        let mut mb = MemoryBank::new(2, 2);
        mb.memory = vec![2.0, 0.0, 0.0, 4.0];
        let weights = vec![0.5, 0.5];
        let r = mb.read(&weights).expect("operation should succeed");
        assert!((r[0] - 1.0).abs() < 1e-6);
        assert!((r[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_memory_bank_write_erase_add() {
        // Single slot, known values
        let mut mb = MemoryBank::new(1, 2);
        mb.memory = vec![1.0, 1.0];
        // w=1, e=[1,1] fully erase, then add [0.5, 0.3]
        mb.write(&[1.0], &[1.0, 1.0], &[0.5, 0.3])
            .expect("operation should succeed");
        assert!((mb.memory[0] - 0.5).abs() < 1e-6);
        assert!((mb.memory[1] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_memory_bank_write_partial_erase() {
        // w=0.5, e=[1,1], add=[0,0]: M_new = M*(1-0.5) = 0.5*M
        let mut mb = MemoryBank::new(1, 2);
        mb.memory = vec![2.0, 4.0];
        mb.write(&[0.5], &[1.0, 1.0], &[0.0, 0.0])
            .expect("operation should succeed");
        assert!((mb.memory[0] - 1.0).abs() < 1e-6, "got {}", mb.memory[0]);
        assert!((mb.memory[1] - 2.0).abs() < 1e-6, "got {}", mb.memory[1]);
    }

    #[test]
    fn test_memory_bank_get_row() {
        let mut mb = MemoryBank::new(2, 3);
        mb.memory = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let row0 = mb.get_row(0);
        assert_eq!(row0, &[1.0, 2.0, 3.0]);
        let row1 = mb.get_row(1);
        assert_eq!(row1, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let mut mb = MemoryBank::new(1, 3);
        mb.memory = vec![1.0, 0.0, 0.0];
        let key = vec![1.0, 0.0, 0.0];
        let cs = mb.cosine_similarity(&key, 0);
        assert!((cs - 1.0).abs() < 1e-5, "cs={cs}");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let mut mb = MemoryBank::new(1, 2);
        mb.memory = vec![1.0, 0.0];
        let key = vec![0.0, 1.0];
        let cs = mb.cosine_similarity(&key, 0);
        assert!(cs.abs() < 1e-5, "cs={cs}");
    }

    // ── Content addressing ────────────────────────────────────────────────

    #[test]
    fn test_content_addressing_highest_weight_at_similar_row() {
        // Memory has 4 slots; row 2 is identical to the key → should get highest weight
        let mut mb = MemoryBank::new(4, 3);
        mb.memory = vec![
            0.0, 0.0, 1.0, // row 0
            0.0, 1.0, 0.0, // row 1
            1.0, 0.0, 0.0, // row 2 — matches key
            0.5, 0.5, 0.0, // row 3
        ];
        let key = vec![1.0, 0.0, 0.0];
        let beta = 10.0_f32;
        let g = 0.999;
        let shift_logits = vec![0.0, 10.0, 0.0]; // identity shift (centre dominant)
        let gamma = 1.0;
        let prev_w = vec![0.25_f32; 4];
        let w = NtmAddressing::address(&key, beta, g, &shift_logits, gamma, &mb, &prev_w)
            .expect("operation should succeed");
        let max_idx = w
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("computation failed");
        assert_eq!(max_idx, 2, "weights={w:?}");
    }

    #[test]
    fn test_content_weights_sum_to_one() {
        let mb = MemoryBank::new(8, 4);
        let key = vec![0.3, 0.7, 0.1, 0.9];
        let beta = 3.0;
        let g = 0.5;
        let shift_logits = vec![0.0, 1.0, 0.0];
        let gamma = 1.0;
        let prev_w = vec![0.125_f32; 8];
        let w = NtmAddressing::address(&key, beta, g, &shift_logits, gamma, &mb, &prev_w)
            .expect("operation should succeed");
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
    }

    // ── Circular convolution ──────────────────────────────────────────────

    #[test]
    fn test_convolve_circular_identity_shift() {
        // shift_dist with all weight on centre (shift=0) should be identity
        let weights = vec![0.1, 0.5, 0.3, 0.1];
        let shift_dist = vec![0.0, 1.0, 0.0]; // centre = index 1 = shift 0
        let out = NtmAddressing::convolve_circular(&weights, &shift_dist, 4);
        for (o, w) in out.iter().zip(weights.iter()) {
            assert!((o - w).abs() < 1e-6, "out={o}, expected={w}");
        }
    }

    #[test]
    fn test_convolve_circular_shift_right() {
        // shift_dist with all weight at index 2 (shift=+1 relative to centre at 1)
        // means w̃(i) = Σ_j w_g(j) * s(i-j): with s at position +1, w̃(i) = w_g(i-1)
        let weights = vec![1.0, 0.0, 0.0, 0.0];
        let shift_dist = vec![0.0, 0.0, 1.0]; // shift +1 from centre
        let out = NtmAddressing::convolve_circular(&weights, &shift_dist, 4);
        // w̃(0) = w_g(0 - (-1)) = w_g(1) = 0 (shift = index2 - centre = 1)
        // shift_dist[2] => k=2, shift = 2 - 1 = +1 => j = i - 1
        // w̃(1) = w_g(0) = 1.0
        assert!((out[1] - 1.0).abs() < 1e-6, "out={out:?}");
    }

    // ── Sharpening ────────────────────────────────────────────────────────

    #[test]
    fn test_sharpening_makes_weights_peaked() {
        // Start with a slightly unequal distribution; sharpening γ>1 should
        // increase the maximum weight relative to the average.
        let mut mb = MemoryBank::new(4, 2);
        mb.memory = vec![1.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0];
        let key = vec![1.0, 0.0];
        let prev_w = vec![0.25_f32; 4];
        let shift_logits = vec![0.0, 1.0, 0.0];

        let w_low = NtmAddressing::address(&key, 2.0, 0.99, &shift_logits, 1.0, &mb, &prev_w)
            .expect("operation should succeed");
        let w_high = NtmAddressing::address(&key, 2.0, 0.99, &shift_logits, 5.0, &mb, &prev_w)
            .expect("operation should succeed");

        let max_low = w_low.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max_high = w_high.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_high >= max_low,
            "sharpening should increase maximum weight: max_low={max_low}, max_high={max_high}"
        );
    }

    // ── NtmController ─────────────────────────────────────────────────────

    #[test]
    fn test_ntm_controller_output_shape() {
        let controller = NtmController::new(8, 16, 42);
        let input = vec![0.1_f32; 8];
        let h_prev = vec![0.0_f32; 16];
        let c_prev = vec![0.0_f32; 16];
        let (h, c) = controller
            .step(&input, &h_prev, &c_prev)
            .expect("operation should succeed");
        assert_eq!(h.len(), 16);
        assert_eq!(c.len(), 16);
    }

    #[test]
    fn test_ntm_controller_deterministic() {
        let controller = NtmController::new(4, 8, 99);
        let input = vec![0.5_f32; 4];
        let h0 = vec![0.0_f32; 8];
        let c0 = vec![0.0_f32; 8];
        let (h1a, _) = controller
            .step(&input, &h0, &c0)
            .expect("operation should succeed");
        let (h1b, _) = controller
            .step(&input, &h0, &c0)
            .expect("operation should succeed");
        assert_eq!(h1a, h1b);
    }

    // ── NTM init_state ────────────────────────────────────────────────────

    #[test]
    fn test_ntm_init_state_weights_sum_to_one() {
        let config = NtmConfig {
            input_dim: 4,
            output_dim: 4,
            controller_dim: 16,
            memory_size: 8,
            memory_dim: 4,
            num_read_heads: 1,
            num_write_heads: 1,
            shift_range: 1,
        };
        let ntm = NeuralTuringMachine::new(config).expect("operation should succeed");
        let state = ntm.init_state();
        for rw in &state.read_weights {
            let s: f32 = rw.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "read weight sum={s}");
        }
        for ww in &state.write_weights {
            let s: f32 = ww.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "write weight sum={s}");
        }
    }

    #[test]
    fn test_ntm_init_state_memory_shape() {
        let config = NtmConfig {
            input_dim: 4,
            output_dim: 4,
            controller_dim: 16,
            memory_size: 10,
            memory_dim: 6,
            num_read_heads: 2,
            num_write_heads: 1,
            shift_range: 1,
        };
        let ntm = NeuralTuringMachine::new(config.clone()).expect("operation should succeed");
        let state = ntm.init_state();
        assert_eq!(
            state.memory.memory.len(),
            config.memory_size * config.memory_dim
        );
        assert_eq!(state.memory.memory_size, config.memory_size);
        assert_eq!(state.memory.memory_dim, config.memory_dim);
    }

    // ── NTM step ─────────────────────────────────────────────────────────

    #[test]
    fn test_ntm_step_output_shape() {
        let config = NtmConfig {
            input_dim: 8,
            output_dim: 8,
            controller_dim: 32,
            memory_size: 16,
            memory_dim: 8,
            num_read_heads: 1,
            num_write_heads: 1,
            shift_range: 1,
        };
        let ntm = NeuralTuringMachine::new(config.clone()).expect("operation should succeed");
        let state = ntm.init_state();
        let input = vec![0.1_f32; 8];
        let (output, _next) = ntm.step(&input, &state).expect("operation should succeed");
        assert_eq!(output.len(), config.output_dim);
    }

    #[test]
    fn test_ntm_step_state_shapes() {
        let config = NtmConfig {
            input_dim: 4,
            output_dim: 4,
            controller_dim: 16,
            memory_size: 8,
            memory_dim: 4,
            num_read_heads: 2,
            num_write_heads: 1,
            shift_range: 1,
        };
        let ntm = NeuralTuringMachine::new(config.clone()).expect("operation should succeed");
        let state = ntm.init_state();
        let input = vec![0.0_f32; 4];
        let (_, next) = ntm.step(&input, &state).expect("operation should succeed");
        assert_eq!(next.h.len(), config.controller_dim);
        assert_eq!(next.c.len(), config.controller_dim);
        assert_eq!(next.read_weights.len(), config.num_read_heads);
        assert_eq!(next.write_weights.len(), config.num_write_heads);
        assert_eq!(next.read_vectors.len(), config.num_read_heads);
    }

    // ── NTM forward_sequence ─────────────────────────────────────────────

    #[test]
    fn test_ntm_forward_sequence_length() {
        let config = NtmConfig {
            input_dim: 4,
            output_dim: 4,
            controller_dim: 16,
            memory_size: 8,
            memory_dim: 4,
            num_read_heads: 1,
            num_write_heads: 1,
            shift_range: 1,
        };
        let ntm = NeuralTuringMachine::new(config.clone()).expect("operation should succeed");
        let inputs: Vec<Vec<f32>> = (0..7).map(|_| vec![0.0_f32; 4]).collect();
        let outputs = ntm
            .forward_sequence(&inputs)
            .expect("operation should succeed");
        assert_eq!(outputs.len(), 7, "output seq length should match input");
    }

    #[test]
    fn test_ntm_forward_sequence_output_dim() {
        let config = NtmConfig {
            input_dim: 6,
            output_dim: 3,
            controller_dim: 16,
            memory_size: 8,
            memory_dim: 4,
            num_read_heads: 1,
            num_write_heads: 1,
            shift_range: 1,
        };
        let ntm = NeuralTuringMachine::new(config.clone()).expect("operation should succeed");
        let inputs: Vec<Vec<f32>> = (0..5).map(|_| vec![0.0_f32; 6]).collect();
        let outputs = ntm
            .forward_sequence(&inputs)
            .expect("operation should succeed");
        for o in &outputs {
            assert_eq!(o.len(), config.output_dim);
        }
    }

    // ── NtmReadHead ───────────────────────────────────────────────────────

    #[test]
    fn test_read_head_emit_correct_dim() {
        let config = NtmAddressingConfig {
            memory_size: 8,
            memory_dim: 4,
            shift_range: 1,
        };
        let head = NtmReadHead::new(16, &config, 7);
        let mb = MemoryBank::new(8, 4);
        let controller_out = vec![0.1_f32; 16];
        let prev_w = vec![0.125_f32; 8];
        let (rv, rw) = head
            .emit(&controller_out, &mb, &prev_w, &config)
            .expect("operation should succeed");
        assert_eq!(rv.len(), 4, "read vector dim");
        assert_eq!(rw.len(), 8, "weight dim");
    }

    #[test]
    fn test_read_head_weights_sum_to_one() {
        let config = NtmAddressingConfig {
            memory_size: 6,
            memory_dim: 3,
            shift_range: 1,
        };
        let head = NtmReadHead::new(12, &config, 77);
        let mb = MemoryBank::new(6, 3);
        let controller_out = vec![0.5_f32; 12];
        let prev_w = vec![1.0 / 6.0_f32; 6];
        let (_, rw) = head
            .emit(&controller_out, &mb, &prev_w, &config)
            .expect("operation should succeed");
        let s: f32 = rw.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "sum={s}");
    }

    // ── DNC usage_update ──────────────────────────────────────────────────

    #[test]
    fn test_dnc_usage_update_free_gate_one_reduces_usage() {
        let u_prev = vec![0.8_f32, 0.5, 0.3];
        let w_prev = vec![0.6_f32, 0.3, 0.1];
        // free gates = 1 for all slots
        let f_t = vec![1.0_f32; 3];
        let u_new = usage_update(&u_prev, &w_prev, &f_t).expect("operation should succeed");
        for (i, (&old, &new)) in u_prev.iter().zip(u_new.iter()).enumerate() {
            assert!(
                new <= old + 1e-5,
                "slot {i}: usage should not increase with free gate=1, old={old}, new={new}"
            );
        }
    }

    #[test]
    fn test_dnc_usage_update_free_gate_zero_no_change() {
        let u_prev = vec![0.7_f32, 0.4, 0.9];
        let w_prev = vec![0.5_f32, 0.3, 0.2];
        let f_t = vec![0.0_f32; 3];
        let u_new = usage_update(&u_prev, &w_prev, &f_t).expect("operation should succeed");
        for (old, new) in u_prev.iter().zip(u_new.iter()) {
            assert!(
                (old - new).abs() < 1e-5,
                "with free gate=0, usage should not change"
            );
        }
    }

    // ── DNC temporal link diagonal is zero ────────────────────────────────

    #[test]
    fn test_dnc_temporal_link_diagonal_zero() {
        let config = DncConfig {
            memory_size: 4,
            memory_dim: 3,
            num_read_heads: 1,
        };
        let state = DncState::new(config);
        let n = config.memory_size;
        for i in 0..n {
            assert_eq!(
                state.temporal_link[i * n + i],
                0.0,
                "diagonal must be 0 at ({i},{i})"
            );
        }
        // After a write step, diagonal should still be 0
        let write_key = vec![0.3_f32; 3];
        let erase = vec![0.5_f32; 3];
        let add = vec![0.1_f32; 3];
        let new_state = dnc_write(&state, config, &write_key, 1.0, &erase, &add, 0.8, 0.5)
            .expect("operation should succeed");
        for i in 0..n {
            assert!(
                new_state.temporal_link[i * n + i].abs() < 1e-6,
                "diagonal at ({i},{i}) should be 0, got {}",
                new_state.temporal_link[i * n + i]
            );
        }
    }

    // ── DNC write + read roundtrip ────────────────────────────────────────

    #[test]
    fn test_dnc_write_read_roundtrip() {
        let config = DncConfig {
            memory_size: 4,
            memory_dim: 3,
            num_read_heads: 1,
        };
        let state = DncState::new(config);
        let write_key = vec![1.0_f32, 0.0, 0.0];
        let erase = vec![10.0_f32; 3]; // strong erase (sigmoid → ~1)
        let add = vec![7.0_f32, 3.0, 2.0]; // values to write

        // Write with alloc_gate=1 (allocation-based) and strong write gate
        let new_state = dnc_write(&state, config, &write_key, 5.0, &erase, &add, 1.0, 1.0)
            .expect("operation should succeed");

        // Read back with content-only mode using same key
        let read_keys = vec![write_key.clone()];
        let read_strengths = vec![5.0_f32];
        let modes = vec![DncReadModes::content_only()];
        let reads = dnc_read(&new_state, config, &read_keys, &read_strengths, &modes)
            .expect("operation should succeed");
        assert_eq!(reads.len(), 3, "read vector length");
    }

    #[test]
    fn test_dnc_read_output_length() {
        let config = DncConfig {
            memory_size: 6,
            memory_dim: 4,
            num_read_heads: 2,
        };
        let state = DncState::new(config);
        let read_keys = vec![vec![0.1_f32; 4]; 2];
        let read_strengths = vec![1.0_f32; 2];
        let modes = vec![DncReadModes::content_only(); 2];
        let reads = dnc_read(&state, config, &read_keys, &read_strengths, &modes)
            .expect("operation should succeed");
        assert_eq!(reads.len(), config.num_read_heads * config.memory_dim);
    }

    // ── Softmax helper ────────────────────────────────────────────────────

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0_f32, 2.0, 3.0, 0.5];
        let s = softmax(&logits);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_max_entry_has_highest_prob() {
        let logits = vec![0.0_f32, 5.0, 1.0, 2.0];
        let s = softmax(&logits);
        let max_idx = s
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("computation failed");
        assert_eq!(max_idx, 1);
    }
}
