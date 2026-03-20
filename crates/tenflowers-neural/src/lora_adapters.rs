//! # LoRA Adapters — Parameter-Efficient Fine-Tuning (PEFT) Toolkit
//!
//! This module provides a comprehensive, production-grade suite of PEFT methods
//! operating directly on raw `f32` weight slices (no high-level Tensor abstraction
//! required), making them easy to integrate into any existing model pipeline.
//!
//! ## Included Methods
//!
//! | Type | Description |
//! |------|-------------|
//! | [`LoraLayer`] | Standard LoRA: A (in×r) + B (r×out) low-rank adapter |
//! | [`DoraLayer`] | Weight Decomposition LoRA: magnitude + direction fine-tuning |
//! | [`AdaLoraLayer`] | Adaptive LoRA: SVD-based importance-driven rank pruning |
//! | [`Ia3Layer`] | IA³: learned rescaling vectors for keys, values, and FF |
//! | [`PrefixTuning`] | Prepends learned prefix tokens to key/value matrices |
//! | [`PromptTuning`] | Prepends soft prompt embeddings to the input sequence |
//! | [`BitFit`] | Bias-only fine-tuning — all other weights frozen |
//! | [`LoraPlus`] | LoRA+ with differential learning rates for A and B |
//! | [`LoftqInit`] | LoftQ: INT-r quantized base + residual LoRA init |
//! | [`PeftManager`] | Collection manager: apply, merge, save, and load adapters |

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Dense matrix-vector product `y = A x` where A is (rows × cols) stored
/// row-major and `x` has length `cols`.  Returns a `Vec<f32>` of length `rows`.
#[inline]
fn matvec(a: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    let mut y = vec![0.0_f32; rows];
    for r in 0..rows {
        let mut acc = 0.0_f32;
        for c in 0..cols {
            acc += a[r * cols + c] * x[c];
        }
        y[r] = acc;
    }
    y
}

/// Dense matrix-matrix product `C = A B` where A is (m × k), B is (k × n).
/// Both stored row-major; returns row-major (m × n).
#[inline]
fn matmul_flat(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// L2 norm of a slice.
#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Box-Muller single sample using StdRng.
#[inline]
fn sample_normal(rng: &mut StdRng) -> f32 {
    let u1: f32 = (rng.random::<f32>() + 1e-10).ln().abs().sqrt();
    let u2: f32 = rng.random::<f32>();
    u1 * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Thin SVD via power iteration.
/// Returns `(U, S, Vt)` where U is (rows × rank), S is (rank,), Vt is (rank × cols).
fn truncated_svd(
    a: &[f32],
    rows: usize,
    cols: usize,
    rank: usize,
    iters: usize,
    rng: &mut StdRng,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let actual_rank = rank.min(rows).min(cols);
    let mut u_mat: Vec<Vec<f32>> = Vec::with_capacity(actual_rank);
    let mut s_vec: Vec<f32> = Vec::with_capacity(actual_rank);
    let mut vt_mat: Vec<Vec<f32>> = Vec::with_capacity(actual_rank);

    // Working copy of A for deflation
    let mut residual = a.to_vec();

    for _ in 0..actual_rank {
        // Random starting vector
        let mut v: Vec<f32> = (0..cols).map(|_| rng.random::<f32>() - 0.5).collect();
        // Normalize
        let nv = l2_norm(&v).max(1e-12);
        v.iter_mut().for_each(|x| *x /= nv);

        for _ in 0..iters {
            // u = A v
            let u = matvec(&residual, &v, rows, cols);
            // v = A^T u
            let mut v_new = vec![0.0_f32; cols];
            for r in 0..rows {
                for c in 0..cols {
                    v_new[c] += residual[r * cols + c] * u[r];
                }
            }
            // Normalize
            let nv2 = l2_norm(&v_new).max(1e-12);
            v_new.iter_mut().for_each(|x| *x /= nv2);
            v = v_new;
        }

        // Compute u = A v, sigma = ||u||
        let mut u = matvec(&residual, &v, rows, cols);
        let sigma = l2_norm(&u);
        if sigma < 1e-12 {
            break;
        }
        u.iter_mut().for_each(|x| *x /= sigma);

        // Deflate: residual -= sigma * u * v^T
        for r in 0..rows {
            for c in 0..cols {
                residual[r * cols + c] -= sigma * u[r] * v[c];
            }
        }

        s_vec.push(sigma);
        u_mat.push(u);
        vt_mat.push(v);
    }

    // Flatten U (rows × r), Vt (r × cols)
    let r = s_vec.len();
    let mut u_flat = vec![0.0_f32; rows * r];
    let mut vt_flat = vec![0.0_f32; r * cols];
    for (k, u_col) in u_mat.iter().enumerate() {
        for (i, &val) in u_col.iter().enumerate() {
            u_flat[i * r + k] = val;
        }
    }
    for (k, v_row) in vt_mat.iter().enumerate() {
        for (j, &val) in v_row.iter().enumerate() {
            vt_flat[k * cols + j] = val;
        }
    }

    (u_flat, s_vec, vt_flat)
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: LoraLayer
// ─────────────────────────────────────────────────────────────────────────────

/// Standard LoRA layer (Low-Rank Adaptation).
///
/// Adds a low-rank correction ΔW = B A (scaled by α/r) to the base weight:
///   out = W x + (B A x) * (α / r)
///
/// * A : shape (r × in_features) — down-projection
/// * B : shape (out_features × r) — up-projection (zero-initialised)
///
/// Reference: Hu et al. 2021 — "LoRA: Low-Rank Adaptation of Large Language Models"
#[derive(Debug, Clone)]
pub struct LoraLayer {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// Low-rank dimension r.
    pub rank: usize,
    /// Scaling numerator α.
    pub alpha: f32,
    /// A matrix (r × in_features), row-major.
    pub a: Vec<f32>,
    /// B matrix (out_features × r), row-major.
    pub b: Vec<f32>,
}

impl LoraLayer {
    /// Create a new LoraLayer with Gaussian A init and zero B init.
    pub fn new(in_features: usize, out_features: usize, rank: usize, alpha: f32) -> Result<Self> {
        if rank == 0 || rank > in_features || rank > out_features {
            return Err(TensorError::invalid_shape(
                "LoraLayer::new",
                &format!("rank in 1..={}", in_features.min(out_features)),
                &rank.to_string(),
            ));
        }
        let mut rng = StdRng::seed_from_u64(42);
        let std_dev = (1.0 / in_features as f32).sqrt();
        let a: Vec<f32> = (0..rank * in_features)
            .map(|_| sample_normal(&mut rng) * std_dev)
            .collect();
        let b = vec![0.0_f32; out_features * rank];
        Ok(Self {
            in_features,
            out_features,
            rank,
            alpha,
            a,
            b,
        })
    }

    /// Scaling factor α / r.
    #[inline]
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// Forward pass: compute `W x + ΔW x`.
    ///
    /// `x`: input vector (in_features,)
    /// `w`: base weight (out_features × in_features) row-major
    ///
    /// Returns output vector (out_features,).
    pub fn forward(&self, x: &[f32], w: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.in_features {
            return Err(TensorError::invalid_shape(
                "LoraLayer::forward",
                &format!("x.len()={}", self.in_features),
                &x.len().to_string(),
            ));
        }
        if w.len() != self.out_features * self.in_features {
            return Err(TensorError::invalid_shape(
                "LoraLayer::forward",
                &format!("w.len()={}", self.out_features * self.in_features),
                &w.len().to_string(),
            ));
        }
        // Base: out = W x
        let base = matvec(w, x, self.out_features, self.in_features);
        // LoRA: delta = B (A x) * scale
        let ax = matvec(&self.a, x, self.rank, self.in_features);
        let bax = matvec(&self.b, &ax, self.out_features, self.rank);
        let s = self.scale();
        let out: Vec<f32> = base
            .iter()
            .zip(bax.iter())
            .map(|(base_v, delta)| base_v + delta * s)
            .collect();
        Ok(out)
    }

    /// Merge the LoRA adapter into the base weight in-place.
    /// After merging, the adapter contribution is absorbed into `w`.
    pub fn merge_into_weight(&self, w: &mut [f32]) -> Result<()> {
        if w.len() != self.out_features * self.in_features {
            return Err(TensorError::invalid_shape(
                "LoraLayer::merge_into_weight",
                &format!("w.len()={}", self.out_features * self.in_features),
                &w.len().to_string(),
            ));
        }
        // ΔW = B A  (out × in)
        let delta = matmul_flat(
            &self.b,
            &self.a,
            self.out_features,
            self.rank,
            self.in_features,
        );
        let s = self.scale();
        for (w_val, d) in w.iter_mut().zip(delta.iter()) {
            *w_val += d * s;
        }
        Ok(())
    }

    /// Return ΔW = B A * scale as a flat (out_features × in_features) matrix.
    pub fn delta_weight(&self) -> Vec<f32> {
        let delta = matmul_flat(
            &self.b,
            &self.a,
            self.out_features,
            self.rank,
            self.in_features,
        );
        let s = self.scale();
        delta.into_iter().map(|v| v * s).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: DoraLayer
// ─────────────────────────────────────────────────────────────────────────────

/// DoRA (Weight-Decomposed Low-Rank Adaptation).
///
/// Decomposes the weight into magnitude m and unit-norm direction, then applies
/// LoRA to the directional component:
///   out\[r\] = m\[r\] * (dot(W_row + ΔW_row, x) / ||W_row + ΔW_row||)
///
/// Reference: Liu et al. 2024 — "DoRA: Weight-Decomposed Low-Rank Adaptation"
#[derive(Debug, Clone)]
pub struct DoraLayer {
    /// Underlying LoRA parameters.
    pub lora: LoraLayer,
    /// Per-row magnitude vector, length = out_features.
    pub magnitude: Vec<f32>,
}

impl DoraLayer {
    /// Create a new DoraLayer initialised from the supplied base weight.
    /// The magnitude is set to the row-wise L2 norms of `w`.
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        w: &[f32],
    ) -> Result<Self> {
        let lora = LoraLayer::new(in_features, out_features, rank, alpha)?;
        if w.len() != out_features * in_features {
            return Err(TensorError::invalid_shape(
                "DoraLayer::new",
                &format!("w.len()={}", out_features * in_features),
                &w.len().to_string(),
            ));
        }
        // Row norms: each row of W (in the out×in layout) gives one magnitude.
        let magnitude: Vec<f32> = (0..out_features)
            .map(|r| {
                let row = &w[r * in_features..(r + 1) * in_features];
                l2_norm(row).max(1e-12)
            })
            .collect();
        Ok(Self { lora, magnitude })
    }

    /// Forward pass applying weight decomposition.
    ///
    /// `x` : (in_features,)
    /// `w` : (out_features × in_features) row-major base weight
    pub fn forward(&self, x: &[f32], w: &[f32]) -> Result<Vec<f32>> {
        let out_f = self.lora.out_features;
        let in_f = self.lora.in_features;
        if x.len() != in_f {
            return Err(TensorError::invalid_shape(
                "DoraLayer::forward",
                &format!("x.len()={in_f}"),
                &x.len().to_string(),
            ));
        }
        if w.len() != out_f * in_f {
            return Err(TensorError::invalid_shape(
                "DoraLayer::forward",
                &format!("w.len()={}", out_f * in_f),
                &w.len().to_string(),
            ));
        }
        // Compute delta from LoRA
        let delta = self.lora.delta_weight(); // out × in
        let mut out = vec![0.0_f32; out_f];
        for r in 0..out_f {
            let w_row = &w[r * in_f..(r + 1) * in_f];
            // numerator: dot((W_row + delta_row), x)
            let mut num = 0.0_f32;
            // denominator: ||W_row + delta_row||
            let mut denom_sq = 0.0_f32;
            for c in 0..in_f {
                let w_plus_d = w_row[c] + delta[r * in_f + c];
                num += w_plus_d * x[c];
                denom_sq += w_plus_d * w_plus_d;
            }
            let denom = denom_sq.sqrt().max(1e-12);
            out[r] = self.magnitude[r] * num / denom;
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: AdaLoraLayer
// ─────────────────────────────────────────────────────────────────────────────

/// Importance score per singular value used by AdaLoRA.
#[derive(Debug, Clone)]
pub struct SingularValueImportance {
    /// Singular value magnitude.
    pub value: f32,
    /// Exponential moving average of gradient magnitude for this dimension.
    pub ema_grad: f32,
    /// Combined importance score.
    pub score: f32,
    /// Whether this rank dimension is currently active.
    pub active: bool,
}

/// AdaLoRA layer — adaptive rank allocation via SVD importance scoring.
///
/// Reference: Zhang et al. 2023 — "AdaLoRA: Adaptive Budget Allocation for
/// Parameter-Efficient Fine-Tuning"
#[derive(Debug, Clone)]
pub struct AdaLoraLayer {
    pub in_features: usize,
    pub out_features: usize,
    /// Maximum rank (initial capacity).
    pub max_rank: usize,
    pub alpha: f32,
    /// Left singular vectors P (out_features × max_rank), column-major wrt ranks.
    pub p: Vec<f32>,
    /// Singular values Λ (max_rank,).
    pub lambda: Vec<f32>,
    /// Right singular vectors Q (max_rank × in_features).
    pub q: Vec<f32>,
    /// Importance score per singular dimension.
    pub importance: Vec<SingularValueImportance>,
    /// EMA decay factor.
    pub ema_decay: f32,
}

impl AdaLoraLayer {
    /// Initialise AdaLoRA from a base weight (performs truncated SVD).
    pub fn new(
        in_features: usize,
        out_features: usize,
        max_rank: usize,
        alpha: f32,
        w: &[f32],
    ) -> Result<Self> {
        if max_rank == 0 {
            return Err(TensorError::invalid_shape(
                "AdaLoraLayer::new",
                "max_rank >= 1",
                "0",
            ));
        }
        if w.len() != out_features * in_features {
            return Err(TensorError::invalid_shape(
                "AdaLoraLayer::new",
                &format!("w.len()={}", out_features * in_features),
                &w.len().to_string(),
            ));
        }
        let mut rng = StdRng::seed_from_u64(123);
        let actual_rank = max_rank.min(in_features).min(out_features);
        let (u, s, vt) = truncated_svd(w, out_features, in_features, actual_rank, 30, &mut rng);
        let r = s.len();
        let importance: Vec<SingularValueImportance> = s
            .iter()
            .map(|&sv| SingularValueImportance {
                value: sv,
                ema_grad: 0.0,
                score: sv.abs(),
                active: true,
            })
            .collect();
        Ok(Self {
            in_features,
            out_features,
            max_rank: r,
            alpha,
            p: u,
            lambda: s,
            q: vt,
            importance,
            ema_decay: 0.9,
        })
    }

    /// Forward pass: compute P Λ Q x (with active ranks only) * scale + W x.
    pub fn forward(&self, x: &[f32], w: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.in_features {
            return Err(TensorError::invalid_shape(
                "AdaLoraLayer::forward",
                &format!("x.len()={}", self.in_features),
                &x.len().to_string(),
            ));
        }
        if w.len() != self.out_features * self.in_features {
            return Err(TensorError::invalid_shape(
                "AdaLoraLayer::forward",
                &format!("w.len()={}", self.out_features * self.in_features),
                &w.len().to_string(),
            ));
        }
        let scale = self.alpha / self.max_rank as f32;
        let base = matvec(w, x, self.out_features, self.in_features);
        let mut out = base;

        for k in 0..self.max_rank {
            if !self.importance[k].active {
                continue;
            }
            let sv = self.lambda[k] * scale;
            // q_row = Q[k, :], length in_features
            let q_row_start = k * self.in_features;
            let q_row = &self.q[q_row_start..q_row_start + self.in_features];
            // scalar: q_k dot x
            let qx: f32 = q_row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            // add sv * qx * P[:, k]
            for i in 0..self.out_features {
                out[i] += sv * qx * self.p[i * self.max_rank + k];
            }
        }
        Ok(out)
    }

    /// Update importance scores using the gradient of the singular value diagonal.
    /// `grad_lambda`: gradient w.r.t. each singular value (max_rank,).
    pub fn update_importance(&mut self, grad_lambda: &[f32]) {
        let n = self.max_rank.min(grad_lambda.len());
        for k in 0..n {
            let g = grad_lambda[k].abs();
            self.importance[k].ema_grad =
                self.ema_decay * self.importance[k].ema_grad + (1.0 - self.ema_decay) * g;
            self.importance[k].score = self.lambda[k].abs() * self.importance[k].ema_grad.max(1e-8);
        }
    }

    /// Prune ranks down to `target_rank` by deactivating low-importance dims.
    pub fn prune_to_budget(&mut self, target_rank: usize) {
        // Collect (score, index) for active dims
        let mut scores: Vec<(f32, usize)> = self
            .importance
            .iter()
            .enumerate()
            .filter(|(_, imp)| imp.active)
            .map(|(i, imp)| (imp.score, i))
            .collect();
        scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let active_count = scores.len();
        if active_count <= target_rank {
            return;
        }
        let to_deactivate = active_count - target_rank;
        for &(_, idx) in scores.iter().take(to_deactivate) {
            self.importance[idx].active = false;
        }
    }

    /// Return active rank count.
    pub fn active_rank(&self) -> usize {
        self.importance.iter().filter(|imp| imp.active).count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: Ia3Layer
// ─────────────────────────────────────────────────────────────────────────────

/// IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations).
///
/// Learns element-wise rescaling vectors instead of additive rank decompositions.
///
/// Reference: Liu et al. 2022 — "Few-Shot Parameter-Efficient Fine-Tuning is
/// Better and Cheaper than In-Context Learning"
#[derive(Debug, Clone)]
pub struct Ia3Layer {
    /// Rescaling for key projections (d_model,).
    pub l_k: Vec<f32>,
    /// Rescaling for value projections (d_model,).
    pub l_v: Vec<f32>,
    /// Rescaling for feed-forward intermediate activations (d_ff,).
    pub l_ff: Vec<f32>,
    pub d_model: usize,
    pub d_ff: usize,
}

impl Ia3Layer {
    /// Create a new IA³ layer initialised to ones (identity transform).
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        Self {
            l_k: vec![1.0_f32; d_model],
            l_v: vec![1.0_f32; d_model],
            l_ff: vec![1.0_f32; d_ff],
            d_model,
            d_ff,
        }
    }

    /// Apply key rescaling: `k_out[i] = l_k[i] * k[i]`.
    pub fn forward_attention_k(&self, k: &[f32]) -> Result<Vec<f32>> {
        if k.len() != self.d_model {
            return Err(TensorError::invalid_shape(
                "Ia3Layer::forward_attention_k",
                &format!("k.len()={}", self.d_model),
                &k.len().to_string(),
            ));
        }
        Ok(k.iter().zip(self.l_k.iter()).map(|(a, b)| a * b).collect())
    }

    /// Apply value rescaling: `v_out[i] = l_v[i] * v[i]`.
    pub fn forward_attention_v(&self, v: &[f32]) -> Result<Vec<f32>> {
        if v.len() != self.d_model {
            return Err(TensorError::invalid_shape(
                "Ia3Layer::forward_attention_v",
                &format!("v.len()={}", self.d_model),
                &v.len().to_string(),
            ));
        }
        Ok(v.iter().zip(self.l_v.iter()).map(|(a, b)| a * b).collect())
    }

    /// Apply feed-forward rescaling: `x_out[i] = l_ff[i] * x[i]`.
    pub fn forward_ff(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.d_ff {
            return Err(TensorError::invalid_shape(
                "Ia3Layer::forward_ff",
                &format!("x.len()={}", self.d_ff),
                &x.len().to_string(),
            ));
        }
        Ok(x.iter().zip(self.l_ff.iter()).map(|(a, b)| a * b).collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: PrefixTuning
// ─────────────────────────────────────────────────────────────────────────────

/// Prefix Tuning — prepends `prefix_len` learned tokens to key/value matrices.
///
/// Reference: Li & Liang 2021 — "Prefix-Tuning: Optimizing Continuous Prompts
/// for Generation"
#[derive(Debug, Clone)]
pub struct PrefixTuning {
    /// Number of prefix tokens to prepend.
    pub prefix_len: usize,
    /// Model hidden dimension.
    pub d_model: usize,
    /// Learned prefix key embeddings (prefix_len × d_model) row-major.
    pub prefix_keys: Vec<f32>,
    /// Learned prefix value embeddings (prefix_len × d_model) row-major.
    pub prefix_values: Vec<f32>,
}

impl PrefixTuning {
    /// Initialise with random Gaussian prefix embeddings (std=0.02).
    pub fn new(prefix_len: usize, d_model: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(17);
        let std_dev = 0.02_f32;
        let prefix_keys: Vec<f32> = (0..prefix_len * d_model)
            .map(|_| sample_normal(&mut rng) * std_dev)
            .collect();
        let prefix_values: Vec<f32> = (0..prefix_len * d_model)
            .map(|_| sample_normal(&mut rng) * std_dev)
            .collect();
        Self {
            prefix_len,
            d_model,
            prefix_keys,
            prefix_values,
        }
    }

    /// Prepend prefix tokens to existing key/value sequences.
    ///
    /// `k`: (seq_len × d_model) row-major
    /// `v`: (seq_len × d_model) row-major
    ///
    /// Returns ((seq_len + prefix_len) × d_model) key and value tensors.
    pub fn prepend_to_kv(
        &self,
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        d_model: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if d_model != self.d_model {
            return Err(TensorError::invalid_shape(
                "PrefixTuning::prepend_to_kv",
                &format!("d_model={}", self.d_model),
                &d_model.to_string(),
            ));
        }
        if k.len() != seq_len * d_model {
            return Err(TensorError::invalid_shape(
                "PrefixTuning::prepend_to_kv",
                &format!("k.len()={}", seq_len * d_model),
                &k.len().to_string(),
            ));
        }
        if v.len() != seq_len * d_model {
            return Err(TensorError::invalid_shape(
                "PrefixTuning::prepend_to_kv",
                &format!("v.len()={}", seq_len * d_model),
                &v.len().to_string(),
            ));
        }
        let total_len = self.prefix_len + seq_len;
        let mut new_k = Vec::with_capacity(total_len * d_model);
        new_k.extend_from_slice(&self.prefix_keys);
        new_k.extend_from_slice(k);
        let mut new_v = Vec::with_capacity(total_len * d_model);
        new_v.extend_from_slice(&self.prefix_values);
        new_v.extend_from_slice(v);
        Ok((new_k, new_v))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6: PromptTuning
// ─────────────────────────────────────────────────────────────────────────────

/// Prompt Tuning — prepends soft prompt embeddings to the input sequence.
///
/// Reference: Lester et al. 2021 — "The Power of Scale for Parameter-Efficient
/// Prompt Tuning"
#[derive(Debug, Clone)]
pub struct PromptTuning {
    /// Number of soft prompt tokens.
    pub num_virtual_tokens: usize,
    /// Embedding dimension.
    pub d_model: usize,
    /// Soft prompt embedding table (num_virtual_tokens × d_model) row-major.
    pub prompt_embeddings: Vec<f32>,
}

impl PromptTuning {
    /// Create a new PromptTuning module with random Gaussian initialisation (std=0.02).
    pub fn new(num_virtual_tokens: usize, d_model: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(31);
        let std_dev = 0.02_f32;
        let prompt_embeddings: Vec<f32> = (0..num_virtual_tokens * d_model)
            .map(|_| sample_normal(&mut rng) * std_dev)
            .collect();
        Self {
            num_virtual_tokens,
            d_model,
            prompt_embeddings,
        }
    }

    /// Prepend soft prompt to input embeddings.
    ///
    /// `input_embeds`: (seq_len × d_model) row-major input token embeddings.
    ///
    /// Returns (num_virtual_tokens + seq_len) × d_model embeddings.
    pub fn prepend_to_input(
        &self,
        input_embeds: &[f32],
        seq_len: usize,
        d_model: usize,
    ) -> Result<Vec<f32>> {
        if d_model != self.d_model {
            return Err(TensorError::invalid_shape(
                "PromptTuning::prepend_to_input",
                &format!("d_model={}", self.d_model),
                &d_model.to_string(),
            ));
        }
        if input_embeds.len() != seq_len * d_model {
            return Err(TensorError::invalid_shape(
                "PromptTuning::prepend_to_input",
                &format!("input_embeds.len()={}", seq_len * d_model),
                &input_embeds.len().to_string(),
            ));
        }
        let total = self.num_virtual_tokens + seq_len;
        let mut out = Vec::with_capacity(total * d_model);
        out.extend_from_slice(&self.prompt_embeddings);
        out.extend_from_slice(input_embeds);
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7: BitFit
// ─────────────────────────────────────────────────────────────────────────────

/// Trainability mask for a single parameter tensor.
#[derive(Debug, Clone)]
pub struct BitFitMask {
    /// Parameter name.
    pub name: String,
    /// Whether this parameter is trainable.
    pub trainable: bool,
    /// Number of elements in the parameter.
    pub num_elements: usize,
}

impl BitFitMask {
    /// Create a new mask for a named parameter.
    pub fn new(name: impl Into<String>, trainable: bool, num_elements: usize) -> Self {
        Self {
            name: name.into(),
            trainable,
            num_elements,
        }
    }
}

/// BitFit — bias-only fine-tuning.
///
/// Freezes all weight matrices; only bias terms remain trainable.
///
/// Reference: Ben Zaken et al. 2022 — "BitFit: Simple Parameter-efficient
/// Fine-tuning for Transformer-based Masked Language-Models"
#[derive(Debug, Clone)]
pub struct BitFit {
    /// Masks for each parameter in the model.
    pub masks: Vec<BitFitMask>,
    /// Bias values keyed by layer name.
    pub biases: HashMap<String, Vec<f32>>,
}

impl BitFit {
    /// Create an empty BitFit module.
    pub fn new() -> Self {
        Self {
            masks: Vec::new(),
            biases: HashMap::new(),
        }
    }

    /// Register a weight tensor (non-trainable) with the given name.
    pub fn register_weight(&mut self, name: impl Into<String>, num_elements: usize) {
        self.masks.push(BitFitMask::new(name, false, num_elements));
    }

    /// Register a bias tensor (trainable) with the given name and initial values.
    pub fn register_bias(&mut self, name: impl Into<String>, initial_bias: Vec<f32>) {
        let n = initial_bias.len();
        let nm: String = name.into();
        self.masks.push(BitFitMask::new(nm.clone(), true, n));
        self.biases.insert(nm, initial_bias);
    }

    /// Get trainable parameter names.
    pub fn trainable_names(&self) -> Vec<&str> {
        self.masks
            .iter()
            .filter(|m| m.trainable)
            .map(|m| m.name.as_str())
            .collect()
    }

    /// Apply the bias to an output vector in-place.
    /// `output`: mutable slice of length equal to bias length.
    pub fn apply_bias(&self, layer_name: &str, output: &mut [f32]) -> Result<()> {
        match self.biases.get(layer_name) {
            Some(bias) => {
                if bias.len() != output.len() {
                    return Err(TensorError::invalid_shape(
                        "BitFit::apply_bias",
                        &format!("output.len()={}", bias.len()),
                        &output.len().to_string(),
                    ));
                }
                for (o, b) in output.iter_mut().zip(bias.iter()) {
                    *o += b;
                }
                Ok(())
            }
            None => Err(TensorError::InvalidArgument {
                operation: "BitFit::apply_bias".to_string(),
                reason: format!("bias '{layer_name}' not found"),
                context: None,
            }),
        }
    }

    /// Update bias values for the given layer.
    pub fn update_bias(&mut self, layer_name: &str, new_bias: Vec<f32>) -> Result<()> {
        match self.biases.get_mut(layer_name) {
            Some(existing) => {
                if existing.len() != new_bias.len() {
                    return Err(TensorError::invalid_shape(
                        "BitFit::update_bias",
                        &format!("new_bias.len()={}", existing.len()),
                        &new_bias.len().to_string(),
                    ));
                }
                *existing = new_bias;
                Ok(())
            }
            None => Err(TensorError::InvalidArgument {
                operation: "BitFit::update_bias".to_string(),
                reason: format!("bias '{layer_name}' not registered"),
                context: None,
            }),
        }
    }

    /// Count trainable scalar parameters (biases only).
    pub fn num_trainable_params(&self) -> usize {
        self.biases.values().map(|b| b.len()).sum()
    }
}

impl Default for BitFit {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 8: LoraPlus
// ─────────────────────────────────────────────────────────────────────────────

/// LoRA+ — uses a larger learning rate for B and smaller for A.
///
/// Reference: Hayou et al. 2024 — "LoRA+: Efficient Low Rank Adaptation of
/// Large Models"
#[derive(Debug, Clone)]
pub struct LoraPlus {
    /// Underlying LoRA layer.
    pub lora: LoraLayer,
    /// Learning rate for matrix B (η_B = base_lr).
    pub lr_b: f32,
    /// Learning rate for matrix A (η_A = base_lr / ratio).
    pub lr_a: f32,
    /// Ratio λ = lr_b / lr_a (recommend 4 to 16).
    pub ratio: f32,
}

impl LoraPlus {
    /// Create LoRA+ wrapping a LoraLayer.
    /// `base_lr`: learning rate for B; A will use `base_lr / ratio`.
    pub fn new(lora: LoraLayer, base_lr: f32, ratio: f32) -> Result<Self> {
        if ratio <= 0.0 {
            return Err(TensorError::InvalidArgument {
                operation: "LoraPlus::new".to_string(),
                reason: format!("ratio must be > 0, got {ratio}"),
                context: None,
            });
        }
        let lr_a = base_lr / ratio;
        Ok(Self {
            lora,
            lr_b: base_lr,
            lr_a,
            ratio,
        })
    }

    /// Return `(lr_a, lr_b)` multipliers for (A, B) matrices.
    pub fn get_lr_multipliers(&self) -> (f32, f32) {
        (self.lr_a, self.lr_b)
    }

    /// Forward pass identical to LoraLayer.
    pub fn forward(&self, x: &[f32], w: &[f32]) -> Result<Vec<f32>> {
        self.lora.forward(x, w)
    }

    /// Apply one SGD step with differential learning rates.
    ///
    /// `grad_a`: gradient w.r.t. A (same shape as self.lora.a)
    /// `grad_b`: gradient w.r.t. B (same shape as self.lora.b)
    pub fn step(&mut self, grad_a: &[f32], grad_b: &[f32]) -> Result<()> {
        if grad_a.len() != self.lora.a.len() {
            return Err(TensorError::invalid_shape(
                "LoraPlus::step",
                &format!("grad_a.len()={}", self.lora.a.len()),
                &grad_a.len().to_string(),
            ));
        }
        if grad_b.len() != self.lora.b.len() {
            return Err(TensorError::invalid_shape(
                "LoraPlus::step",
                &format!("grad_b.len()={}", self.lora.b.len()),
                &grad_b.len().to_string(),
            ));
        }
        for (a, ga) in self.lora.a.iter_mut().zip(grad_a.iter()) {
            *a -= self.lr_a * ga;
        }
        for (b, gb) in self.lora.b.iter_mut().zip(grad_b.iter()) {
            *b -= self.lr_b * gb;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 9: LoftqInit
// ─────────────────────────────────────────────────────────────────────────────

/// LoftQ initialisation result.
#[derive(Debug, Clone)]
pub struct LoftqResult {
    /// Quantized base weight (dequantized back to f32).
    pub quantized_weight: Vec<f32>,
    /// Initialised A matrix for LoRA (r × cols).
    pub lora_a: Vec<f32>,
    /// Initialised B matrix for LoRA (rows × r).
    pub lora_b: Vec<f32>,
    /// Frobenius norm of the residual after rank-r approximation.
    pub residual_norm: f32,
}

/// LoftQ — Quantized LoRA Initialization.
///
/// Quantize the base weight to INT-`bits`, compute the residual (W − Q), then
/// initialise LoRA A and B from the SVD of the residual so that Q + BA ≈ W.
///
/// Reference: Liu et al. 2023 — "LoftQ: LoRA-Fine-Tuning-Aware Quantization
/// for Large Language Models"
pub struct LoftqInit;

impl LoftqInit {
    /// Quantize `weight` (rows × cols) to `bits`-bit symmetric uniform integers,
    /// compute the SVD of the residual, and return initialised LoRA matrices.
    ///
    /// `weight` : flat (rows × cols) weight
    /// `rank`   : LoRA rank
    /// `bits`   : quantization bit width (1–16)
    pub fn initialize(
        weight: &[f32],
        rows: usize,
        cols: usize,
        rank: usize,
        bits: u8,
    ) -> Result<LoftqResult> {
        if weight.len() != rows * cols {
            return Err(TensorError::invalid_shape(
                "LoftqInit::initialize",
                &format!("weight.len()={}", rows * cols),
                &weight.len().to_string(),
            ));
        }
        if bits == 0 || bits > 16 {
            return Err(TensorError::InvalidArgument {
                operation: "LoftqInit::initialize".to_string(),
                reason: format!("bits must be in 1..=16, got {bits}"),
                context: None,
            });
        }
        if rank == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "LoftqInit::initialize".to_string(),
                reason: "rank must be >= 1".to_string(),
                context: None,
            });
        }

        // Step 1: symmetric uniform quantization
        let n_levels = (1_u32 << (bits - 1)) as f32 - 1.0; // bits=8 → 127
        let abs_max = weight.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let scale = if abs_max < 1e-12 {
            1.0
        } else {
            abs_max / n_levels
        };

        let quantized_weight: Vec<f32> = weight
            .iter()
            .map(|&w| {
                let q = (w / scale).round().clamp(-n_levels, n_levels);
                q * scale
            })
            .collect();

        // Step 2: residual R = W - Q_dequant
        let residual: Vec<f32> = weight
            .iter()
            .zip(quantized_weight.iter())
            .map(|(w, q)| w - q)
            .collect();

        // Step 3: truncated SVD of residual
        let mut rng = StdRng::seed_from_u64(999);
        let actual_rank = rank.min(rows).min(cols);
        let (u, s, vt) = truncated_svd(&residual, rows, cols, actual_rank, 30, &mut rng);
        let r = s.len();

        // Step 4: B = U * sqrt(S), A = sqrt(S) * Vt
        let sqrt_s: Vec<f32> = s.iter().map(|&sv| sv.max(0.0).sqrt()).collect();
        let mut lora_b = vec![0.0_f32; rows * r];
        let mut lora_a = vec![0.0_f32; r * cols];
        for k in 0..r {
            for i in 0..rows {
                lora_b[i * r + k] = u[i * r + k] * sqrt_s[k];
            }
            for j in 0..cols {
                lora_a[k * cols + j] = sqrt_s[k] * vt[k * cols + j];
            }
        }

        // Step 5: residual norm (Frobenius)
        let residual_approx: Vec<f32> = matmul_flat(&lora_b, &lora_a, rows, r, cols);
        let residual_norm: f32 = residual
            .iter()
            .zip(residual_approx.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        Ok(LoftqResult {
            quantized_weight,
            lora_a,
            lora_b,
            residual_norm,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 10: PeftManager
// ─────────────────────────────────────────────────────────────────────────────

/// Serialized form of a LoRA adapter for save/load.
#[derive(Debug, Clone)]
struct SerializedAdapter {
    in_features: usize,
    out_features: usize,
    rank: usize,
    alpha: f32,
    a: Vec<f32>,
    b: Vec<f32>,
}

impl SerializedAdapter {
    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.in_features as u64).to_le_bytes());
        buf.extend_from_slice(&(self.out_features as u64).to_le_bytes());
        buf.extend_from_slice(&(self.rank as u64).to_le_bytes());
        buf.extend_from_slice(&self.alpha.to_le_bytes());
        buf.extend_from_slice(&(self.a.len() as u64).to_le_bytes());
        for v in &self.a {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.extend_from_slice(&(self.b.len() as u64).to_le_bytes());
        for v in &self.b {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    fn from_bytes(data: &[u8], offset: &mut usize) -> std::result::Result<Self, String> {
        fn read_u64(d: &[u8], off: &mut usize) -> std::result::Result<u64, String> {
            if *off + 8 > d.len() {
                return Err(format!("truncated: need 8 bytes at offset {off}"));
            }
            let v = u64::from_le_bytes(d[*off..*off + 8].try_into().map_err(|e| format!("{e}"))?);
            *off += 8;
            Ok(v)
        }
        fn read_f32(d: &[u8], off: &mut usize) -> std::result::Result<f32, String> {
            if *off + 4 > d.len() {
                return Err(format!("truncated: need 4 bytes at offset {off}"));
            }
            let v = f32::from_le_bytes(d[*off..*off + 4].try_into().map_err(|e| format!("{e}"))?);
            *off += 4;
            Ok(v)
        }

        let in_features = read_u64(data, offset)? as usize;
        let out_features = read_u64(data, offset)? as usize;
        let rank = read_u64(data, offset)? as usize;
        let alpha = read_f32(data, offset)?;
        let a_len = read_u64(data, offset)? as usize;
        let mut a = Vec::with_capacity(a_len);
        for _ in 0..a_len {
            a.push(read_f32(data, offset)?);
        }
        let b_len = read_u64(data, offset)? as usize;
        let mut b = Vec::with_capacity(b_len);
        for _ in 0..b_len {
            b.push(read_f32(data, offset)?);
        }
        Ok(Self {
            in_features,
            out_features,
            rank,
            alpha,
            a,
            b,
        })
    }
}

/// PeftManager — manages named LoRA adapters with apply/merge/save/load.
#[derive(Debug, Clone)]
pub struct PeftManager {
    adapters: HashMap<String, LoraLayer>,
}

impl PeftManager {
    /// Create an empty manager.
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
        }
    }

    /// Register a LoRA adapter under a layer name.
    pub fn add_adapter(&mut self, name: impl Into<String>, layer: LoraLayer) {
        self.adapters.insert(name.into(), layer);
    }

    /// Remove a named adapter (returns it if present).
    pub fn remove_adapter(&mut self, name: &str) -> Option<LoraLayer> {
        self.adapters.remove(name)
    }

    /// Check if an adapter exists for the given layer name.
    pub fn has_adapter(&self, name: &str) -> bool {
        self.adapters.contains_key(name)
    }

    /// Apply the adapter for `layer_name` to input `x` with base weight `w`.
    /// Returns `None` if no adapter is registered for that name.
    pub fn apply(&self, layer_name: &str, x: &[f32], w: &[f32]) -> Option<Result<Vec<f32>>> {
        self.adapters
            .get(layer_name)
            .map(|layer| layer.forward(x, w))
    }

    /// List all registered adapter names.
    pub fn adapter_names(&self) -> Vec<&str> {
        self.adapters.keys().map(|s| s.as_str()).collect()
    }

    /// Merge all registered adapters into corresponding weights in the map.
    /// Weight key must match the adapter name.
    pub fn merge_all(&self, weights: &mut HashMap<String, Vec<f32>>) -> Result<()> {
        for (name, adapter) in &self.adapters {
            match weights.get_mut(name) {
                Some(w) => {
                    adapter.merge_into_weight(w)?;
                }
                None => {
                    return Err(TensorError::InvalidArgument {
                        operation: "PeftManager::merge_all".to_string(),
                        reason: format!("weight key '{name}' not present in weights map"),
                        context: None,
                    });
                }
            }
        }
        Ok(())
    }

    /// Serialize all adapters to a flat byte buffer.
    ///
    /// Format: `[u64: num_adapters]` then for each:
    /// `[u64: name_len][name_bytes][adapter_bytes]`
    pub fn save_adapters(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let n = self.adapters.len() as u64;
        buf.extend_from_slice(&n.to_le_bytes());
        for (name, adapter) in &self.adapters {
            let name_bytes = name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(name_bytes);
            let sa = SerializedAdapter {
                in_features: adapter.in_features,
                out_features: adapter.out_features,
                rank: adapter.rank,
                alpha: adapter.alpha,
                a: adapter.a.clone(),
                b: adapter.b.clone(),
            };
            buf.extend_from_slice(&sa.to_bytes());
        }
        buf
    }

    /// Deserialize adapters from a byte slice produced by `save_adapters`.
    /// Replaces any existing adapters.
    pub fn load_adapters(&mut self, data: &[u8]) -> std::result::Result<(), String> {
        if data.len() < 8 {
            return Err("data too short for adapter count".to_string());
        }
        let mut offset = 0usize;

        fn read_u64(d: &[u8], off: &mut usize) -> std::result::Result<u64, String> {
            if *off + 8 > d.len() {
                return Err(format!("truncated at offset {off}"));
            }
            let v = u64::from_le_bytes(d[*off..*off + 8].try_into().map_err(|e| format!("{e}"))?);
            *off += 8;
            Ok(v)
        }

        let num_adapters = read_u64(data, &mut offset)? as usize;
        let mut loaded: HashMap<String, LoraLayer> = HashMap::with_capacity(num_adapters);

        for _ in 0..num_adapters {
            let name_len = read_u64(data, &mut offset)? as usize;
            if offset + name_len > data.len() {
                return Err("truncated name".to_string());
            }
            let name = std::str::from_utf8(&data[offset..offset + name_len])
                .map_err(|e| format!("invalid UTF-8 in name: {e}"))?
                .to_string();
            offset += name_len;

            let sa = SerializedAdapter::from_bytes(data, &mut offset)?;
            let layer = LoraLayer {
                in_features: sa.in_features,
                out_features: sa.out_features,
                rank: sa.rank,
                alpha: sa.alpha,
                a: sa.a,
                b: sa.b,
            };
            loaded.insert(name, layer);
        }

        self.adapters = loaded;
        Ok(())
    }
}

impl Default for PeftManager {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── LoraLayer ─────────────────────────────────────────────────────────────

    #[test]
    fn test_lora_new_valid() {
        let l = LoraLayer::new(8, 16, 4, 8.0).expect("should succeed");
        assert_eq!(l.in_features, 8);
        assert_eq!(l.out_features, 16);
        assert_eq!(l.rank, 4);
        assert_eq!(l.a.len(), 4 * 8);
        assert_eq!(l.b.len(), 16 * 4);
    }

    #[test]
    fn test_lora_new_rank_zero_fails() {
        assert!(LoraLayer::new(8, 16, 0, 8.0).is_err());
    }

    #[test]
    fn test_lora_b_zero_initialized() {
        let l = LoraLayer::new(8, 16, 2, 8.0).expect("ok");
        assert!(l.b.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_lora_scale() {
        let l = LoraLayer::new(8, 8, 4, 16.0).expect("ok");
        assert!((l.scale() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_lora_forward_identity_b_zero() {
        // With B=0, output should equal W x
        let in_f = 4;
        let out_f = 4;
        let w: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let x = vec![1.0_f32; in_f];
        let l = LoraLayer::new(in_f, out_f, 2, 4.0).expect("ok");
        let out = l.forward(&x, &w).expect("forward ok");
        let base: Vec<f32> = (0..out_f)
            .map(|r| (0..in_f).map(|c| w[r * in_f + c]).sum())
            .collect();
        for (a, b) in out.iter().zip(base.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn test_lora_forward_wrong_x_len_fails() {
        let w = vec![0.0_f32; 16];
        let l = LoraLayer::new(4, 4, 2, 4.0).expect("ok");
        assert!(l.forward(&[1.0; 5], &w).is_err());
    }

    #[test]
    fn test_lora_forward_wrong_w_len_fails() {
        let l = LoraLayer::new(4, 4, 2, 4.0).expect("ok");
        let x = vec![1.0_f32; 4];
        assert!(l.forward(&x, &[0.0_f32; 10]).is_err());
    }

    #[test]
    fn test_lora_merge_into_weight_identity() {
        let in_f = 4;
        let out_f = 4;
        let mut w: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let w_orig = w.clone();
        // B=0 → merge should not change w
        let l = LoraLayer::new(in_f, out_f, 2, 4.0).expect("ok");
        l.merge_into_weight(&mut w).expect("merge ok");
        for (a, b) in w.iter().zip(w_orig.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_lora_merge_adds_delta() {
        let in_f = 2;
        let out_f = 2;
        let mut l = LoraLayer::new(in_f, out_f, 1, 1.0).expect("ok");
        // A = [[1, 0]], B = [[0], [1]] → ΔW = [[0,0],[1,0]] * scale=1.0
        l.a = vec![1.0, 0.0];
        l.b = vec![0.0, 1.0];
        let mut w = vec![0.0_f32; 4];
        l.merge_into_weight(&mut w).expect("ok");
        // ΔW[1,0] = 1.0
        assert!((w[2] - 1.0).abs() < 1e-6, "w[2]={}", w[2]);
    }

    // ── DoraLayer ─────────────────────────────────────────────────────────────

    #[test]
    fn test_dora_new_valid() {
        let w: Vec<f32> = (0..16).map(|i| i as f32 * 0.1 + 0.1).collect();
        let d = DoraLayer::new(4, 4, 2, 4.0, &w).expect("ok");
        assert_eq!(d.magnitude.len(), 4);
        assert!(d.magnitude.iter().all(|&m| m > 0.0));
    }

    #[test]
    fn test_dora_forward_zero_delta() {
        let in_f = 4;
        let out_f = 4;
        let w: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
        let x = vec![1.0_f32; in_f];
        let d = DoraLayer::new(in_f, out_f, 2, 4.0, &w).expect("ok");
        let out = d.forward(&x, &w).expect("ok");
        assert_eq!(out.len(), out_f);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_dora_wrong_w_len_fails() {
        let w = vec![1.0_f32; 16];
        let d = DoraLayer::new(4, 4, 2, 4.0, &w).expect("ok");
        let x = vec![1.0_f32; 4];
        let w_bad = vec![1.0_f32; 25];
        assert!(d.forward(&x, &w_bad).is_err());
    }

    // ── AdaLoraLayer ──────────────────────────────────────────────────────────

    #[test]
    fn test_adalora_new_valid() {
        let w: Vec<f32> = (0..16).map(|i| i as f32 * 0.01).collect();
        let al = AdaLoraLayer::new(4, 4, 3, 4.0, &w).expect("ok");
        assert!(al.active_rank() <= 3);
        assert!(al.active_rank() >= 1);
    }

    #[test]
    fn test_adalora_forward_output_length() {
        let in_f = 4;
        let out_f = 6;
        let w: Vec<f32> = (0..24).map(|i| i as f32 * 0.01).collect();
        let al = AdaLoraLayer::new(in_f, out_f, 2, 4.0, &w).expect("ok");
        let x = vec![1.0_f32; in_f];
        let out = al.forward(&x, &w).expect("ok");
        assert_eq!(out.len(), out_f);
    }

    #[test]
    fn test_adalora_update_importance() {
        let w: Vec<f32> = vec![0.1, 0.5, 0.2, 0.8, 0.3, 0.1, 0.4, 0.6, 0.2];
        let mut al = AdaLoraLayer::new(3, 3, 2, 4.0, &w).expect("ok");
        let grad = vec![0.1_f32; al.max_rank];
        al.update_importance(&grad);
        assert!(al.importance.iter().all(|imp| imp.ema_grad >= 0.0));
    }

    #[test]
    fn test_adalora_prune_to_budget() {
        let w: Vec<f32> = (0..9).map(|i| (i + 1) as f32 * 0.1).collect();
        let mut al = AdaLoraLayer::new(3, 3, 3, 4.0, &w).expect("ok");
        // Inject varied importance scores
        for (k, imp) in al.importance.iter_mut().enumerate() {
            imp.score = (k + 1) as f32;
        }
        al.prune_to_budget(1);
        assert_eq!(al.active_rank(), 1);
    }

    // ── Ia3Layer ──────────────────────────────────────────────────────────────

    #[test]
    fn test_ia3_forward_k_identity() {
        let ia3 = Ia3Layer::new(4, 8);
        let k = vec![0.5_f32, 1.0, -0.5, 2.0];
        let out = ia3.forward_attention_k(&k).expect("ok");
        // l_k = ones → output should equal k
        for (a, b) in out.iter().zip(k.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ia3_forward_v_identity() {
        let ia3 = Ia3Layer::new(4, 8);
        let v = vec![0.1_f32, 0.2, 0.3, 0.4];
        let out = ia3.forward_attention_v(&v).expect("ok");
        for (a, b) in out.iter().zip(v.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ia3_forward_ff() {
        let ia3 = Ia3Layer::new(4, 8);
        let x = vec![1.0_f32; 8];
        let out = ia3.forward_ff(&x).expect("ok");
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_ia3_wrong_size_fails() {
        let ia3 = Ia3Layer::new(4, 8);
        assert!(ia3.forward_attention_k(&[1.0_f32; 5]).is_err());
        assert!(ia3.forward_ff(&[1.0_f32; 3]).is_err());
    }

    #[test]
    fn test_ia3_scaling() {
        let mut ia3 = Ia3Layer::new(2, 2);
        ia3.l_k = vec![2.0, 0.5];
        let k = vec![1.0_f32, 4.0];
        let out = ia3.forward_attention_k(&k).expect("ok");
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
    }

    // ── PrefixTuning ──────────────────────────────────────────────────────────

    #[test]
    fn test_prefix_tuning_prepend_shape() {
        let pt = PrefixTuning::new(4, 8);
        let seq_len = 5;
        let k = vec![0.1_f32; seq_len * 8];
        let v = vec![0.2_f32; seq_len * 8];
        let (new_k, new_v) = pt.prepend_to_kv(&k, &v, seq_len, 8).expect("ok");
        assert_eq!(new_k.len(), (4 + seq_len) * 8);
        assert_eq!(new_v.len(), (4 + seq_len) * 8);
    }

    #[test]
    fn test_prefix_tuning_content_correct() {
        let pt = PrefixTuning::new(2, 4);
        let seq_len = 4;
        let k = vec![9.9_f32; seq_len * 4];
        let v = vec![8.8_f32; seq_len * 4];
        let (new_k, _) = pt.prepend_to_kv(&k, &v, seq_len, 4).expect("ok");
        // The last seq_len * d_model elements should be 9.9
        for &val in &new_k[2 * 4..] {
            assert!((val - 9.9).abs() < 1e-5, "val={val}");
        }
    }

    #[test]
    fn test_prefix_tuning_wrong_d_model_fails() {
        let pt = PrefixTuning::new(2, 8);
        let k = vec![0.0_f32; 10];
        let v = vec![0.0_f32; 10];
        assert!(pt.prepend_to_kv(&k, &v, 2, 5).is_err()); // d_model != 8
    }

    // ── PromptTuning ──────────────────────────────────────────────────────────

    #[test]
    fn test_prompt_tuning_prepend_shape() {
        let ptuning = PromptTuning::new(3, 8);
        let input = vec![0.5_f32; 5 * 8]; // seq_len=5
        let out = ptuning.prepend_to_input(&input, 5, 8).expect("ok");
        assert_eq!(out.len(), (3 + 5) * 8);
    }

    #[test]
    fn test_prompt_tuning_original_tokens_preserved() {
        let ptuning = PromptTuning::new(2, 4);
        let orig = vec![7.7_f32; 3 * 4];
        let out = ptuning.prepend_to_input(&orig, 3, 4).expect("ok");
        // Last 3*4 elements should be 7.7
        for &v in &out[2 * 4..] {
            assert!((v - 7.7).abs() < 1e-5, "v={v}");
        }
    }

    #[test]
    fn test_prompt_tuning_wrong_d_model_fails() {
        let ptuning = PromptTuning::new(2, 8);
        let input = vec![1.0_f32; 5 * 4];
        assert!(ptuning.prepend_to_input(&input, 5, 4).is_err());
    }

    // ── BitFit ────────────────────────────────────────────────────────────────

    #[test]
    fn test_bitfit_register_trainable_names() {
        let mut bf = BitFit::new();
        bf.register_weight("layer0.weight", 512);
        bf.register_bias("layer0.bias", vec![0.0; 32]);
        let names = bf.trainable_names();
        assert_eq!(names.len(), 1);
        assert_eq!(names[0], "layer0.bias");
    }

    #[test]
    fn test_bitfit_apply_bias() {
        let mut bf = BitFit::new();
        bf.register_bias("fc.bias", vec![1.0_f32, 2.0, 3.0]);
        let mut output = vec![0.0_f32; 3];
        bf.apply_bias("fc.bias", &mut output).expect("ok");
        assert_eq!(output, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_bitfit_update_bias() {
        let mut bf = BitFit::new();
        bf.register_bias("fc.bias", vec![0.0_f32; 4]);
        bf.update_bias("fc.bias", vec![1.0_f32; 4]).expect("ok");
        let mut output = vec![0.0_f32; 4];
        bf.apply_bias("fc.bias", &mut output).expect("ok");
        assert!(output.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_bitfit_missing_bias_fails() {
        let bf = BitFit::new();
        let mut output = vec![0.0_f32; 4];
        assert!(bf.apply_bias("nonexistent", &mut output).is_err());
    }

    #[test]
    fn test_bitfit_num_trainable_params() {
        let mut bf = BitFit::new();
        bf.register_weight("w1", 1000);
        bf.register_bias("b1", vec![0.0_f32; 32]);
        bf.register_bias("b2", vec![0.0_f32; 64]);
        assert_eq!(bf.num_trainable_params(), 96);
    }

    // ── LoraPlus ──────────────────────────────────────────────────────────────

    #[test]
    fn test_loraplus_lr_multipliers() {
        let l = LoraLayer::new(4, 4, 2, 4.0).expect("ok");
        let lp = LoraPlus::new(l, 0.01, 4.0).expect("ok");
        let (lr_a, lr_b) = lp.get_lr_multipliers();
        assert!((lr_b - 0.01).abs() < 1e-7);
        assert!((lr_a - 0.0025).abs() < 1e-7);
    }

    #[test]
    fn test_loraplus_forward_same_as_lora() {
        let l = LoraLayer::new(4, 4, 2, 4.0).expect("ok");
        let lp = LoraPlus::new(l.clone(), 0.01, 4.0).expect("ok");
        let w: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let x = vec![1.0_f32; 4];
        let out_l = l.forward(&x, &w).expect("ok");
        let out_lp = lp.forward(&x, &w).expect("ok");
        for (a, b) in out_l.iter().zip(out_lp.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_loraplus_step_updates_params() {
        let l = LoraLayer::new(4, 4, 2, 4.0).expect("ok");
        let a_orig = l.a.clone();
        let b_orig = l.b.clone();
        let mut lp = LoraPlus::new(l, 1.0, 2.0).expect("ok");
        let grad_a = vec![1.0_f32; lp.lora.a.len()];
        let grad_b = vec![1.0_f32; lp.lora.b.len()];
        lp.step(&grad_a, &grad_b).expect("ok");
        // A updated by lr_a=0.5, B by lr_b=1.0
        for (new, old) in lp.lora.a.iter().zip(a_orig.iter()) {
            assert!((new - (old - 0.5)).abs() < 1e-6);
        }
        for (new, old) in lp.lora.b.iter().zip(b_orig.iter()) {
            assert!((new - (old - 1.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_loraplus_zero_ratio_fails() {
        let l = LoraLayer::new(4, 4, 2, 4.0).expect("ok");
        assert!(LoraPlus::new(l, 0.01, 0.0).is_err());
    }

    // ── LoftqInit ─────────────────────────────────────────────────────────────

    #[test]
    fn test_loftq_init_shapes() {
        let rows = 4;
        let cols = 4;
        let w: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let res = LoftqInit::initialize(&w, rows, cols, 2, 8).expect("ok");
        assert_eq!(res.quantized_weight.len(), 16);
        let r = res.lora_b.len() / rows;
        assert!((1..=2).contains(&r));
        assert_eq!(res.lora_a.len(), r * cols);
    }

    #[test]
    fn test_loftq_quantized_weight_bounded() {
        let w: Vec<f32> = vec![
            0.5, -0.5, 0.3, -0.3, 0.1, -0.1, 0.4, -0.4, 0.2, -0.2, 0.6, -0.6, 0.7, -0.7, 0.8, -0.8,
        ];
        let res = LoftqInit::initialize(&w, 4, 4, 2, 8).expect("ok");
        let abs_max = w.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        for &qw in &res.quantized_weight {
            assert!(qw.abs() <= abs_max + 1e-5);
        }
    }

    #[test]
    fn test_loftq_residual_norm_finite() {
        let w: Vec<f32> = (0..16).map(|i| (i as f32 - 7.5) * 0.1).collect();
        let res = LoftqInit::initialize(&w, 4, 4, 2, 4).expect("ok");
        assert!(res.residual_norm.is_finite());
    }

    #[test]
    fn test_loftq_wrong_size_fails() {
        let w = vec![0.0_f32; 10];
        assert!(LoftqInit::initialize(&w, 4, 4, 2, 8).is_err());
    }

    #[test]
    fn test_loftq_zero_bits_fails() {
        let w = vec![0.0_f32; 16];
        assert!(LoftqInit::initialize(&w, 4, 4, 2, 0).is_err());
    }

    // ── PeftManager ───────────────────────────────────────────────────────────

    #[test]
    fn test_peft_manager_add_and_has() {
        let mut mgr = PeftManager::new();
        let l = LoraLayer::new(4, 4, 2, 4.0).expect("ok");
        mgr.add_adapter("attn.q", l);
        assert!(mgr.has_adapter("attn.q"));
        assert!(!mgr.has_adapter("attn.k"));
    }

    #[test]
    fn test_peft_manager_apply_present() {
        let mut mgr = PeftManager::new();
        let l = LoraLayer::new(4, 4, 2, 4.0).expect("ok");
        mgr.add_adapter("layer0", l);
        let w = vec![0.1_f32; 16];
        let x = vec![1.0_f32; 4];
        let result = mgr.apply("layer0", &x, &w);
        assert!(result.is_some());
        assert!(result.expect("apply should return Some").is_ok());
    }

    #[test]
    fn test_peft_manager_apply_absent_returns_none() {
        let mgr = PeftManager::new();
        let w = vec![0.1_f32; 16];
        let x = vec![1.0_f32; 4];
        assert!(mgr.apply("nonexistent", &x, &w).is_none());
    }

    #[test]
    fn test_peft_manager_merge_all() {
        let mut mgr = PeftManager::new();
        let l = LoraLayer::new(2, 2, 1, 1.0).expect("ok");
        mgr.add_adapter("layer0", l);
        let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
        weights.insert("layer0".to_string(), vec![0.0_f32; 4]);
        mgr.merge_all(&mut weights).expect("merge ok");
        // B=0 → weight unchanged
        assert!(weights["layer0"].iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn test_peft_manager_merge_missing_key_fails() {
        let mut mgr = PeftManager::new();
        let l = LoraLayer::new(2, 2, 1, 1.0).expect("ok");
        mgr.add_adapter("layer0", l);
        let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
        assert!(mgr.merge_all(&mut weights).is_err());
    }

    #[test]
    fn test_peft_manager_save_load_roundtrip() {
        let mut mgr = PeftManager::new();
        let l = LoraLayer::new(4, 4, 2, 8.0).expect("ok");
        mgr.add_adapter("attn.q", l);
        let bytes = mgr.save_adapters();
        let mut mgr2 = PeftManager::new();
        mgr2.load_adapters(&bytes).expect("load ok");
        assert!(mgr2.has_adapter("attn.q"));
    }

    #[test]
    fn test_peft_manager_save_load_values_preserved() {
        let mut mgr = PeftManager::new();
        let mut l = LoraLayer::new(2, 2, 1, 2.0).expect("ok");
        l.a = vec![0.5_f32, -0.5];
        l.b = vec![1.0_f32, -1.0];
        mgr.add_adapter("fc", l);
        let bytes = mgr.save_adapters();
        let mut mgr2 = PeftManager::new();
        mgr2.load_adapters(&bytes).expect("ok");
        let a_ref = &mgr2.adapters["fc"];
        assert!((a_ref.a[0] - 0.5).abs() < 1e-6);
        assert!((a_ref.b[1] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_peft_manager_load_empty_bytes_fails() {
        let mut mgr = PeftManager::new();
        assert!(mgr.load_adapters(&[]).is_err());
    }

    #[test]
    fn test_peft_manager_adapter_names() {
        let mut mgr = PeftManager::new();
        mgr.add_adapter("a", LoraLayer::new(2, 2, 1, 1.0).expect("ok"));
        mgr.add_adapter("b", LoraLayer::new(2, 2, 1, 1.0).expect("ok"));
        let mut names = mgr.adapter_names();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn test_peft_manager_remove_adapter() {
        let mut mgr = PeftManager::new();
        mgr.add_adapter("x", LoraLayer::new(2, 2, 1, 1.0).expect("ok"));
        assert!(mgr.has_adapter("x"));
        let removed = mgr.remove_adapter("x");
        assert!(removed.is_some());
        assert!(!mgr.has_adapter("x"));
    }
}
