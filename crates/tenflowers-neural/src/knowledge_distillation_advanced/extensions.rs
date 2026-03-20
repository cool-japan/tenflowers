//! Extensions for knowledge distillation — attention transfer, scheduling,
//! and efficient transfer learning.

use super::{dot, kl_div, l2_normalize_mut, mse_slice, softmax_temp};
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// §8  Attention Transfer
// ─────────────────────────────────────────────────────────────────────────────

/// An attention map produced by a transformer layer.
///
/// `values` is a flat array of shape `[n_heads × seq_len × seq_len]`.
#[derive(Debug, Clone)]
pub struct AttentionMap {
    /// Number of attention heads.
    pub n_heads: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Flat attention weights, row-major `[n_heads × seq_len × seq_len]`.
    pub values: Vec<f32>,
}

impl AttentionMap {
    /// Access the attention weight for head `h`, query `q`, key `k`.
    pub fn get(&self, h: usize, q: usize, k: usize) -> f32 {
        let idx = h * self.seq_len * self.seq_len + q * self.seq_len + k;
        self.values.get(idx).copied().unwrap_or(0.0)
    }
}

/// Compute an attention map from query and key matrices using scaled
/// dot-product attention.
///
/// # Errors
/// Returns error on dimension mismatch.
pub fn compute_attention_map(
    q: &[f32],
    k: &[f32],
    n_heads: usize,
    seq_len: usize,
) -> Result<AttentionMap> {
    if n_heads == 0 || seq_len == 0 {
        return Err(TensorError::invalid_argument(
            "n_heads and seq_len must be positive".to_string(),
        ));
    }
    if q.len() != k.len() {
        return Err(TensorError::invalid_argument(format!(
            "q ({}) and k ({}) must have the same length",
            q.len(),
            k.len()
        )));
    }
    let total_elems = q.len();
    if total_elems % (n_heads * seq_len) != 0 {
        return Err(TensorError::invalid_argument(format!(
            "q length {total_elems} is not divisible by n_heads*seq_len = {}",
            n_heads * seq_len
        )));
    }
    let head_dim = total_elems / (n_heads * seq_len);
    let scale = (head_dim as f32).sqrt().max(1e-8);

    let mut values = vec![0.0_f32; n_heads * seq_len * seq_len];

    for h in 0..n_heads {
        for qi in 0..seq_len {
            let q_start = h * seq_len * head_dim + qi * head_dim;
            let q_vec = &q[q_start..q_start + head_dim];

            let mut row = vec![0.0_f32; seq_len];
            for ki in 0..seq_len {
                let k_start = h * seq_len * head_dim + ki * head_dim;
                let k_vec = &k[k_start..k_start + head_dim];
                row[ki] = dot(q_vec, k_vec) / scale;
            }
            let row_sm = softmax_temp(&row, 1.0);
            let out_start = h * seq_len * seq_len + qi * seq_len;
            values[out_start..out_start + seq_len].copy_from_slice(&row_sm);
        }
    }

    Ok(AttentionMap {
        n_heads,
        seq_len,
        values,
    })
}

/// Attention transfer loss and related utilities.
#[derive(Debug, Clone, Default)]
pub struct AttentionTransfer;

impl AttentionTransfer {
    /// Attention Transfer (AT) loss.
    ///
    /// # Errors
    /// Returns error if maps have different dimensions.
    pub fn at_loss(&self, student_attn: &AttentionMap, teacher_attn: &AttentionMap) -> Result<f32> {
        if student_attn.n_heads != teacher_attn.n_heads
            || student_attn.seq_len != teacher_attn.seq_len
        {
            return Err(TensorError::invalid_argument(format!(
                "attention map dimensions mismatch: student ({}, {}) vs teacher ({}, {})",
                student_attn.n_heads,
                student_attn.seq_len,
                teacher_attn.n_heads,
                teacher_attn.seq_len
            )));
        }

        let n_heads = student_attn.n_heads;
        let seq_len = student_attn.seq_len;
        let mut total_loss = 0.0_f32;

        for h in 0..n_heads {
            let mut s_map = vec![0.0_f32; seq_len];
            let mut t_map = vec![0.0_f32; seq_len];
            for qi in 0..seq_len {
                for ki in 0..seq_len {
                    s_map[ki] += student_attn.get(h, qi, ki);
                    t_map[ki] += teacher_attn.get(h, qi, ki);
                }
            }
            l2_normalize_mut(&mut s_map);
            l2_normalize_mut(&mut t_map);
            total_loss += mse_slice(&s_map, &t_map);
        }

        Ok(total_loss / n_heads as f32)
    }

    /// Gram matrix distillation loss.
    ///
    /// # Errors
    /// Returns error on length mismatch.
    pub fn gram_matrix_loss(&self, student_feat: &[f32], teacher_feat: &[f32]) -> Result<f32> {
        if student_feat.len() != teacher_feat.len() {
            return Err(TensorError::invalid_argument(format!(
                "student_feat length {} != teacher_feat length {}",
                student_feat.len(),
                teacher_feat.len()
            )));
        }
        if student_feat.is_empty() {
            return Ok(0.0);
        }
        let n = student_feat.len();
        let gs = dot(student_feat, student_feat) / n as f32;
        let gt = dot(teacher_feat, teacher_feat) / n as f32;
        Ok((gs - gt).powi(2))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  Distillation Scheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Temperature scheduling strategy for knowledge distillation.
#[derive(Debug, Clone, PartialEq)]
pub enum TempSchedule {
    /// Constant temperature throughout training.
    Constant(f32),
    /// Linearly decay from `start` to `end` over `total_steps`.
    LinearDecay { start: f32, end: f32 },
    /// Cosine annealing between `t_max` and `t_min`.
    CosineAnnealing { t_max: f32, t_min: f32 },
    /// Cyclic warm restarts: ramp up from `base` to `peak` and back.
    CyclicWarm {
        base: f32,
        peak: f32,
        cycle_len: usize,
    },
}

/// Alpha (soft/hard loss weighting) scheduling strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum AlphaSchedule {
    /// Constant alpha throughout training.
    Constant(f32),
    /// Linear rise from `start` to `end` over `total_steps`.
    LinearRise { start: f32, end: f32 },
}

/// Adaptive temperature and alpha scheduler for distillation training.
#[derive(Debug, Clone)]
pub struct KdDistilScheduler {
    /// Temperature schedule.
    pub temp_sched: TempSchedule,
    /// Alpha (soft loss weight) schedule.
    pub alpha_sched: AlphaSchedule,
    /// Current training step.
    pub current_step: usize,
    /// Total number of training steps.
    pub total_steps: usize,
}

impl KdDistilScheduler {
    /// Create a new scheduler.
    pub fn new(temp_sched: TempSchedule, alpha_sched: AlphaSchedule, total_steps: usize) -> Self {
        Self {
            temp_sched,
            alpha_sched,
            current_step: 0,
            total_steps: total_steps.max(1),
        }
    }

    /// Advance one step and return `(temperature, alpha)` for the current step.
    pub fn step(&mut self) -> (f32, f32) {
        let t = self.current_step;
        let total = self.total_steps;
        let progress = (t as f32 / total as f32).clamp(0.0, 1.0);

        let temp = match &self.temp_sched {
            TempSchedule::Constant(c) => *c,
            TempSchedule::LinearDecay { start, end } => start + (end - start) * progress,
            TempSchedule::CosineAnnealing { t_max, t_min } => {
                t_min + 0.5 * (t_max - t_min) * (1.0 + (std::f32::consts::PI * progress).cos())
            }
            TempSchedule::CyclicWarm {
                base,
                peak,
                cycle_len,
            } => {
                let cycle = if *cycle_len == 0 { 1 } else { *cycle_len };
                let pos_in_cycle = t % cycle;
                let half = cycle / 2;
                if pos_in_cycle < half {
                    let r = pos_in_cycle as f32 / half.max(1) as f32;
                    base + (peak - base) * r
                } else {
                    let r = (pos_in_cycle - half) as f32 / (cycle - half).max(1) as f32;
                    peak - (peak - base) * r
                }
            }
        };

        let alpha = match &self.alpha_sched {
            AlphaSchedule::Constant(c) => *c,
            AlphaSchedule::LinearRise { start, end } => start + (end - start) * progress,
        };

        self.current_step += 1;
        (temp.max(0.01), alpha.clamp(0.0, 1.0))
    }

    /// Reset the scheduler to step 0.
    pub fn reset(&mut self) {
        self.current_step = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  Efficient Transfer Learning
// ─────────────────────────────────────────────────────────────────────────────

/// A group of parameters that can be frozen or unfrozen together.
#[derive(Debug, Clone)]
pub struct LayerGroup {
    /// Approximate number of parameters in this group.
    pub param_count: usize,
    /// Whether the group is currently frozen (not updated during training).
    pub is_frozen: bool,
}

impl LayerGroup {
    /// Create a frozen layer group.
    pub fn new_frozen(param_count: usize) -> Self {
        Self {
            param_count,
            is_frozen: true,
        }
    }

    /// Create an unfrozen layer group.
    pub fn new_unfrozen(param_count: usize) -> Self {
        Self {
            param_count,
            is_frozen: false,
        }
    }
}

/// Schedule for gradually unfreezing layer groups.
#[derive(Debug, Clone)]
pub struct FreezeSchedule {
    /// Total number of layer groups.
    pub n_groups: usize,
    /// Unfreeze one group every this many training steps.
    pub unfreeze_every_n_steps: usize,
}

impl FreezeSchedule {
    /// Create a validated schedule.
    pub fn new(n_groups: usize, unfreeze_every_n_steps: usize) -> Result<Self> {
        if n_groups == 0 {
            return Err(TensorError::invalid_argument(
                "n_groups must be at least 1".to_string(),
            ));
        }
        if unfreeze_every_n_steps == 0 {
            return Err(TensorError::invalid_argument(
                "unfreeze_every_n_steps must be at least 1".to_string(),
            ));
        }
        Ok(Self {
            n_groups,
            unfreeze_every_n_steps,
        })
    }

    /// Return `true` if layer `group` should be unfrozen at training `step`.
    pub fn should_unfreeze_layer(&self, group: usize, step: usize) -> bool {
        let unfreeze_at_step = group * self.unfreeze_every_n_steps;
        step >= unfreeze_at_step
    }
}

/// Efficient transfer learning engine with discriminative fine-tuning and
/// gradual unfreezing.
#[derive(Debug, Clone)]
pub struct EfficientTransferLearning {
    /// Layer groups, ordered from top (index 0) to bottom (index n-1).
    pub groups: Vec<LayerGroup>,
    /// Freeze/unfreeze schedule.
    pub schedule: FreezeSchedule,
}

impl EfficientTransferLearning {
    /// Construct from a set of layer groups and a schedule.
    pub fn new(groups: Vec<LayerGroup>, schedule: FreezeSchedule) -> Self {
        Self { groups, schedule }
    }

    /// Compute the learning rate for `group` using discriminative fine-tuning.
    pub fn compute_layer_lr(&self, group: usize, base_lr: f32, decay: f32) -> f32 {
        base_lr * decay.powi(group as i32)
    }

    /// Update the frozen/unfrozen state of all layer groups based on the
    /// current training step.
    pub fn apply_unfreeze(&mut self, step: usize) -> Vec<bool> {
        let mut states = vec![false; self.groups.len()];
        for (g, group) in self.groups.iter_mut().enumerate() {
            if self.schedule.should_unfreeze_layer(g, step) {
                group.is_frozen = false;
            }
            states[g] = !group.is_frozen;
        }
        states
    }

    /// Total number of active (unfrozen) parameters.
    pub fn active_params(&self) -> usize {
        self.groups
            .iter()
            .filter(|g| !g.is_frozen)
            .map(|g| g.param_count)
            .sum()
    }

    /// Total number of frozen parameters.
    pub fn frozen_params(&self) -> usize {
        self.groups
            .iter()
            .filter(|g| g.is_frozen)
            .map(|g| g.param_count)
            .sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Suppressed re-export of pub(crate) utilities for use in the `advanced` module
// ─────────────────────────────────────────────────────────────────────────────

/// Re-export kl_div for use within the crate (tests, advanced module).
pub(crate) use super::kl_div as kl_divergence;
/// Re-export softmax_temp for use within the crate.
pub(crate) use super::softmax_temp as softmax_with_temperature;
