use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::fs;
use tenflowers_core::{Result, TensorError};

#[derive(Debug, Clone)]
pub struct SparsityReport {
    pub total_params: usize,
    pub nonzero_params: usize,
    pub sparsity: f64,
    pub layer_sparsities: Vec<f64>,
}

impl SparsityReport {
    pub fn from_weights_and_masks(weights: &[Vec<f64>], masks: &[Vec<f64>]) -> Result<Self> {
        if weights.len() != masks.len() {
            return Err(TensorError::invalid_argument(
                "weights and masks must have the same number of layers".to_string(),
            ));
        }
        let mut total = 0usize;
        let mut nonzero = 0usize;
        let mut layer_sparsities = Vec::with_capacity(weights.len());

        for (w, m) in weights.iter().zip(masks.iter()) {
            if w.len() != m.len() {
                return Err(TensorError::invalid_argument(
                    "weight and mask vectors must have the same length".to_string(),
                ));
            }
            let n = w.len();
            let nz = m.iter().filter(|&&v| v != 0.0).count();
            total += n;
            nonzero += nz;
            let sp = if n == 0 {
                0.0
            } else {
                1.0 - nz as f64 / n as f64
            };
            layer_sparsities.push(sp);
        }

        let sparsity = if total == 0 {
            0.0
        } else {
            1.0 - nonzero as f64 / total as f64
        };

        Ok(Self {
            total_params: total,
            nonzero_params: nonzero,
            sparsity,
            layer_sparsities,
        })
    }
}

/// Iterative Magnitude Pruning (IMP) as described in Frankle & Carlin (2019).
#[derive(Debug, Clone)]
pub struct LotteryTicketFinder {
    pub threshold: f64,
}

impl LotteryTicketFinder {
    pub fn new() -> Self {
        Self { threshold: 0.0 }
    }

    /// Prune a single flat weight vector to `target_sparsity`.
    pub fn prune_to_sparsity(
        &mut self,
        weights: Vec<Vec<f64>>,
        target_sparsity: f64,
    ) -> Vec<Vec<f64>> {
        let mut abs_vals: Vec<f64> = weights
            .iter()
            .flat_map(|row| row.iter().map(|&w| w.abs()))
            .collect();

        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let n = abs_vals.len();
        let prune_n = ((n as f64) * target_sparsity.clamp(0.0, 1.0)) as usize;
        self.threshold = if prune_n < n {
            abs_vals[prune_n]
        } else {
            f64::MAX
        };

        let threshold = self.threshold;
        weights
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|w| if w.abs() >= threshold { w } else { 0.0 })
                    .collect()
            })
            .collect()
    }

    /// Run the full IMP lottery-ticket search.
    /// Returns `(binary_mask, sparse_weights)` where the mask contains 1.0 for
    pub fn find_ticket(
        &mut self,
        init_weights: Vec<Vec<f64>>,
        sparsity_schedule: &[f64],
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut current = init_weights.clone();

        for &s in sparsity_schedule {
            current = self.prune_to_sparsity(current, s);
        }

        let mask: Vec<Vec<f64>> = init_weights
            .iter()
            .zip(current.iter())
            .map(|(row_init, row_pruned)| {
                row_init
                    .iter()
                    .zip(row_pruned.iter())
                    .map(|(i, p)| {
                        if p.abs() > 0.0 || i.abs() == 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();

        (mask, current)
    }
}

impl Default for LotteryTicketFinder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct GradualMagnitudePruning {
    pub s_initial: f64,
    pub s_final: f64,
    pub t_start: usize,
    pub delta_t: usize,
}

impl GradualMagnitudePruning {
    pub fn new(s_initial: f64, s_final: f64, t_start: usize, delta_t: usize) -> Self {
        Self {
            s_initial,
            s_final,
            t_start,
            delta_t,
        }
    }

    /// Compute the target sparsity at step `step` with `n_steps` total steps.
    pub fn sparsity_at(&self, step: usize, n_steps: usize, s_i: f64, s_f: f64) -> f64 {
        if step < self.t_start {
            return s_i;
        }
        let elapsed = step - self.t_start;
        let denominator = (n_steps * self.delta_t) as f64;
        if denominator <= 0.0 {
            return s_f;
        }
        let ratio = (elapsed as f64 / denominator).clamp(0.0, 1.0);
        let cubic = (1.0 - ratio).powi(3);
        s_f + (s_i - s_f) * cubic
    }
}

#[derive(Debug, Clone)]
pub struct StructuredChannelPruning;

impl StructuredChannelPruning {
    pub fn channel_importance(weights: &[Vec<f64>]) -> Vec<f64> {
        weights
            .iter()
            .map(|row| {
                let sq_sum: f64 = row.iter().map(|&w| w * w).sum();
                sq_sum.sqrt()
            })
            .collect()
    }

    /// Return a pruned weight matrix keeping only the top `keep_ratio` channels
    pub fn prune_channels(weights: Vec<Vec<f64>>, keep_ratio: f64) -> Vec<Vec<f64>> {
        let importance = Self::channel_importance(&weights);
        let n = importance.len();
        let keep_n = ((n as f64) * keep_ratio.clamp(0.0, 1.0)).floor() as usize;

        let mut indexed: Vec<(usize, f64)> = importance.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let keep_set: std::collections::HashSet<usize> =
            indexed.iter().take(keep_n).map(|&(i, _)| i).collect();

        weights
            .into_iter()
            .enumerate()
            .map(|(idx, row)| {
                if keep_set.contains(&idx) {
                    row
                } else {
                    vec![0.0; row.len()]
                }
            })
            .collect()
    }
}

/// Movement pruning (Sanh et al., 2020): score = weight × gradient.
#[derive(Debug, Clone)]
pub struct MovementPruning;

impl MovementPruning {
    pub fn movement_score(w: &[Vec<f64>], g: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if w.len() != g.len() {
            return Err(TensorError::invalid_argument(
                "weight and gradient matrices must have the same number of rows".to_string(),
            ));
        }
        let scores = w
            .iter()
            .zip(g.iter())
            .map(|(w_row, g_row)| {
                if w_row.len() != g_row.len() {
                    vec![]
                } else {
                    w_row
                        .iter()
                        .zip(g_row.iter())
                        .map(|(wi, gi)| wi * gi)
                        .collect()
                }
            })
            .collect::<Vec<_>>();

        for (i, (wr, gr)) in w.iter().zip(g.iter()).enumerate() {
            if wr.len() != gr.len() {
                return Err(TensorError::invalid_argument(format!(
                    "row {i}: weight length {} != gradient length {}",
                    wr.len(),
                    gr.len()
                )));
            }
        }

        Ok(scores)
    }
}

/// Patient Knowledge Distillation (Sun et al., 2019).
#[derive(Debug, Clone)]
pub struct PkdDistillation {
    pub layer_weight: f64,
}

impl PkdDistillation {
    pub fn new(layer_weight: f64) -> Self {
        Self { layer_weight }
    }

    pub fn pkd_loss(
        &self,
        student_layers: &[Vec<f64>],
        teacher_layers: &[Vec<f64>],
    ) -> Result<f64> {
        if student_layers.len() != teacher_layers.len() {
            return Err(TensorError::invalid_argument(
                "student and teacher must have the same number of layers".to_string(),
            ));
        }
        if student_layers.is_empty() {
            return Ok(0.0);
        }

        let mut total = 0.0f64;
        for (s, t) in student_layers.iter().zip(teacher_layers.iter()) {
            if s.len() != t.len() {
                return Err(TensorError::invalid_argument(
                    "student/teacher layer dimensions must match".to_string(),
                ));
            }
            let mse: f64 = s
                .iter()
                .zip(t.iter())
                .map(|(&sv, &tv)| (sv - tv).powi(2))
                .sum::<f64>()
                / s.len() as f64;
            total += mse;
        }

        Ok(self.layer_weight * total / student_layers.len() as f64)
    }
}

/// Relational Knowledge Distillation (Park et al., 2019).
#[derive(Debug, Clone)]
pub struct RkdDistillation {
    pub distance_weight: f64,
    pub angle_weight: f64,
}

impl RkdDistillation {
    pub fn new(distance_weight: f64, angle_weight: f64) -> Self {
        Self {
            distance_weight,
            angle_weight,
        }
    }

    fn l2_dist(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    fn distance_wise_loss(student: &[Vec<f64>], teacher: &[Vec<f64>]) -> f64 {
        let n = student.len();
        if n < 2 {
            return 0.0;
        }
        let mut loss = 0.0;
        let mut count = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let d_s = Self::l2_dist(&student[i], &student[j]);
                let d_t = Self::l2_dist(&teacher[i], &teacher[j]);
                loss += (d_s - d_t).powi(2);
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            loss / count as f64
        }
    }

    fn cosine_angle(a: &[f64], b: &[f64]) -> f64 {
        let num = Self::dot(a, b);
        let den = (Self::dot(a, a).sqrt() * Self::dot(b, b).sqrt()).max(1e-8);
        (num / den).clamp(-1.0, 1.0)
    }

    fn diff(a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
    }

    fn angle_wise_loss(student: &[Vec<f64>], teacher: &[Vec<f64>]) -> f64 {
        let n = student.len();
        if n < 3 {
            return 0.0;
        }
        let mut loss = 0.0;
        let mut count = 0usize;
        for i in 0..n {
            for j in 0..n {
                if j == i {
                    continue;
                }
                for k in (j + 1)..n {
                    if k == i {
                        continue;
                    }
                    let cos_s = Self::cosine_angle(
                        &Self::diff(&student[j], &student[i]),
                        &Self::diff(&student[k], &student[i]),
                    );
                    let cos_t = Self::cosine_angle(
                        &Self::diff(&teacher[j], &teacher[i]),
                        &Self::diff(&teacher[k], &teacher[i]),
                    );
                    loss += (cos_s - cos_t).powi(2);
                    count += 1;
                }
            }
        }
        if count == 0 {
            0.0
        } else {
            loss / count as f64
        }
    }

    pub fn rkd_loss(&self, student_embs: &[Vec<f64>], teacher_embs: &[Vec<f64>]) -> Result<f64> {
        if student_embs.len() != teacher_embs.len() {
            return Err(TensorError::invalid_argument(
                "student and teacher embeddings must have same batch size".to_string(),
            ));
        }
        let dist = Self::distance_wise_loss(student_embs, teacher_embs);
        let angle = Self::angle_wise_loss(student_embs, teacher_embs);
        Ok(self.distance_weight * dist + self.angle_weight * angle)
    }
}

/// Contrastive Representation Distillation (Tian et al., 2020).
#[derive(Debug, Clone)]
pub struct CrdDistillation {
    pub temperature: f64,
}

impl CrdDistillation {
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }

    fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let na: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt().max(1e-8);
        let nb: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt().max(1e-8);
        dot / (na * nb)
    }

    pub fn crd_loss(
        &self,
        student: &[f64],
        teacher: &[f64],
        negatives: &[Vec<f64>],
    ) -> Result<f64> {
        if student.len() != teacher.len() {
            return Err(TensorError::invalid_argument(
                "student and teacher embeddings must have the same dimension".to_string(),
            ));
        }
        let t = self.temperature;
        let pos_sim = Self::cosine_sim(student, teacher) / t;
        let neg_sims: Vec<f64> = negatives
            .iter()
            .map(|neg| Self::cosine_sim(student, neg) / t)
            .collect();

        let max_val = neg_sims.iter().cloned().fold(pos_sim, f64::max);
        let exp_pos = (pos_sim - max_val).exp();
        let exp_neg_sum: f64 = neg_sims.iter().map(|&s| (s - max_val).exp()).sum();
        let denom = (exp_pos + exp_neg_sum).max(1e-10);
        Ok(-(exp_pos / denom).ln())
    }
}

/// FitNet-style feature map alignment (Romero et al., 2015).
#[derive(Debug, Clone)]
pub struct FeatureDistillation {
    pub student_dim: usize,
    pub teacher_dim: usize,
}

impl FeatureDistillation {
    pub fn new(student_dim: usize, teacher_dim: usize) -> Self {
        Self {
            student_dim,
            teacher_dim,
        }
    }

    pub fn fit_loss(
        &self,
        student_feat: &[f64],
        teacher_feat: &[f64],
        projection: &[Vec<f64>],
    ) -> Result<f64> {
        if student_feat.len() != self.student_dim {
            return Err(TensorError::invalid_argument(format!(
                "student_feat length {} != student_dim {}",
                student_feat.len(),
                self.student_dim
            )));
        }
        if teacher_feat.len() != self.teacher_dim {
            return Err(TensorError::invalid_argument(format!(
                "teacher_feat length {} != teacher_dim {}",
                teacher_feat.len(),
                self.teacher_dim
            )));
        }
        if projection.len() != self.teacher_dim {
            return Err(TensorError::invalid_argument(
                "projection must have teacher_dim rows".to_string(),
            ));
        }

        let projected: Vec<f64> = projection
            .iter()
            .map(|row| {
                row.iter()
                    .zip(student_feat.iter())
                    .map(|(&w, &x)| w * x)
                    .sum()
            })
            .collect();

        let mse: f64 = projected
            .iter()
            .zip(teacher_feat.iter())
            .map(|(&p, &t)| (p - t).powi(2))
            .sum::<f64>()
            / self.teacher_dim as f64;

        Ok(mse)
    }
}

/// Anneal the distillation temperature T and the hard-loss weight α during
#[derive(Debug, Clone)]
pub struct DistillationScheduler {
    pub t_start: f64,
    pub t_end: f64,
    pub total_steps: usize,
}

impl DistillationScheduler {
    pub fn new(t_start: f64, t_end: f64, total_steps: usize) -> Self {
        Self {
            t_start,
            t_end,
            total_steps,
        }
    }

    pub fn temperature(&self, step: usize) -> f64 {
        if self.total_steps == 0 {
            return self.t_end;
        }
        let ratio = (step as f64 / self.total_steps as f64).clamp(0.0, 1.0);
        self.t_start + (self.t_end - self.t_start) * ratio
    }

    pub fn alpha(&self, step: usize) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        let ratio = (step as f64 / self.total_steps as f64).clamp(0.0, 1.0);
        1.0 - ratio
    }
}

/// GPTQ: Optimal Brain Quantization (Frantar et al., 2022).
#[derive(Debug, Clone)]
pub struct GptqQuantizer {
    pub bits: u32,
}

impl GptqQuantizer {
    pub fn new(bits: u32) -> Self {
        Self { bits }
    }

    /// Quantize a weight block.
    pub fn quantize_block(
        &self,
        w: &[Vec<f64>],
        h: &[f64],
        bits: u32,
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
        if w.is_empty() {
            return Ok((vec![], vec![]));
        }
        let cols = w[0].len();
        if h.len() != cols {
            return Err(TensorError::invalid_argument(format!(
                "Hessian diagonal length {} must equal weight cols {}",
                h.len(),
                cols
            )));
        }

        let levels = (1u64 << bits) as f64;
        let half_levels = levels / 2.0;

        let scales: Vec<f64> = w
            .iter()
            .map(|row| {
                let max_abs = row.iter().map(|&v| v.abs()).fold(0.0f64, f64::max);
                if max_abs < 1e-8 {
                    1.0
                } else {
                    max_abs / (half_levels - 1.0)
                }
            })
            .collect();

        let quantized: Vec<Vec<f64>> = w
            .iter()
            .zip(scales.iter())
            .map(|(row, &scale)| {
                row.iter()
                    .enumerate()
                    .map(|(col, &val)| {
                        let h_weight = h[col].max(0.0);
                        let quantized_int = (val / scale + h_weight * 1e-4)
                            .round()
                            .clamp(-half_levels, half_levels - 1.0);
                        quantized_int * scale
                    })
                    .collect()
            })
            .collect();

        Ok((quantized, scales))
    }
}

/// AWQ: Activation-Aware Weight Quantization (Lin et al., 2023).
#[derive(Debug, Clone)]
pub struct AwqQuantizer {
    pub alpha: f64,
}

impl AwqQuantizer {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    /// Compute per-channel activation-aware scales.
    pub fn compute_scale(&self, w: &[Vec<f64>], activations: &[f64]) -> Result<Vec<f64>> {
        if w.is_empty() {
            return Ok(vec![]);
        }
        let in_channels = w[0].len();
        if activations.len() != in_channels {
            return Err(TensorError::invalid_argument(format!(
                "activations length {} must equal weight in_channels {}",
                activations.len(),
                in_channels
            )));
        }

        let weight_norms: Vec<f64> = (0..in_channels)
            .map(|c| {
                let sq: f64 = w.iter().map(|row| row[c].powi(2)).sum();
                sq.sqrt()
            })
            .collect();

        let scales: Vec<f64> = activations
            .iter()
            .zip(weight_norms.iter())
            .map(|(&act, &wn)| {
                let act_part = act.abs().max(1e-8).powf(self.alpha);
                let w_part = wn.max(1e-8).powf(1.0 - self.alpha);
                act_part / w_part
            })
            .collect();

        Ok(scales)
    }
}

/// SmoothQuant (Xiao et al., 2022).
#[derive(Debug, Clone)]
pub struct SmoothQuant {
    pub alpha: f64,
}

impl SmoothQuant {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    /// Apply the smooth transformation.
    pub fn smooth(&self, w: &[Vec<f64>], act_scale: &[f64]) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
        if w.is_empty() {
            return Ok((vec![], vec![]));
        }
        let in_channels = w[0].len();
        if act_scale.len() != in_channels {
            return Err(TensorError::invalid_argument(
                "act_scale length must equal number of input channels".to_string(),
            ));
        }

        let weight_maxabs: Vec<f64> = (0..in_channels)
            .map(|j| {
                w.iter()
                    .map(|row| row[j].abs())
                    .fold(0.0f64, f64::max)
                    .max(1e-8)
            })
            .collect();

        let scales: Vec<f64> = act_scale
            .iter()
            .zip(weight_maxabs.iter())
            .map(|(&a, &wm)| a.abs().max(1e-8).powf(self.alpha) / wm.powf(1.0 - self.alpha))
            .collect();

        let smoothed: Vec<Vec<f64>> = w
            .iter()
            .map(|row| {
                row.iter()
                    .zip(scales.iter())
                    .map(|(&wij, &sj)| wij * sj)
                    .collect()
            })
            .collect();

        Ok((smoothed, scales))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Format {
    /// E4M3: 4 exponent bits, 3 mantissa bits.
    E4M3,
    /// E5M2: 5 exponent bits, 2 mantissa bits.
    E5M2,
}

#[derive(Debug, Clone)]
pub struct Fp8Quantizer;

impl Fp8Quantizer {
    fn max_val_e4m3() -> f64 {
        448.0
    }

    fn max_val_e5m2() -> f64 {
        57344.0
    }

    fn mantissa_bits(format: Fp8Format) -> u32 {
        match format {
            Fp8Format::E4M3 => 3,
            Fp8Format::E5M2 => 2,
        }
    }

    pub fn quantize_fp8(x: &[f64], format: Fp8Format) -> Vec<f64> {
        let max_val = match format {
            Fp8Format::E4M3 => Self::max_val_e4m3(),
            Fp8Format::E5M2 => Self::max_val_e5m2(),
        };
        let mantissa_bits = Self::mantissa_bits(format);
        let mantissa_levels = (1u32 << mantissa_bits) as f64;

        x.iter()
            .map(|&val| {
                if val == 0.0 {
                    return 0.0;
                }
                let sign = val.signum();
                let abs_val = val.abs().min(max_val);

                let exp = abs_val.log2().floor();
                let scale = (2.0f64).powf(exp);
                let m = abs_val / scale;
                let m_q = (m * mantissa_levels / 2.0).round() / (mantissa_levels / 2.0);
                let result = (m_q * scale).min(max_val);
                sign * result
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalibrationMethod {
    /// Min-max range calibration.
    MinMax,
    /// Percentile-based range (clips outliers).
    Percentile(f64),
    /// MSE-minimizing range search (grid search over alpha).
    MseSearch,
}

#[derive(Debug, Clone)]
pub struct CalibrationResult {
    pub scale: f64,
    pub zero_point: i32,
    pub range_min: f64,
    pub range_max: f64,
}

#[derive(Debug, Clone)]
pub struct QuantizationCalibrator {
    pub bits: u32,
    pub method: CalibrationMethod,
}

impl QuantizationCalibrator {
    pub fn new(bits: u32, method: CalibrationMethod) -> Self {
        Self { bits, method }
    }

    pub fn calibrate(&self, data: &[f64]) -> Result<CalibrationResult> {
        if data.is_empty() {
            return Err(TensorError::invalid_argument(
                "calibration data must not be empty".to_string(),
            ));
        }

        let (range_min, range_max) = match self.method {
            CalibrationMethod::MinMax => {
                let mn = data.iter().cloned().fold(f64::MAX, f64::min);
                let mx = data.iter().cloned().fold(f64::MIN, f64::max);
                (mn, mx)
            }
            CalibrationMethod::Percentile(pct) => {
                let mut sorted = data.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                let lo_idx = ((1.0 - pct / 100.0) / 2.0 * sorted.len() as f64) as usize;
                let hi_idx = sorted.len() - 1 - lo_idx;
                (sorted[lo_idx.min(sorted.len() - 1)], sorted[hi_idx])
            }
            CalibrationMethod::MseSearch => {
                let mn = data.iter().cloned().fold(f64::MAX, f64::min);
                let mx = data.iter().cloned().fold(f64::MIN, f64::max);
                let abs_max = mn.abs().max(mx.abs());
                let best_alpha = (1..=20)
                    .map(|i| 0.8 + i as f64 * 0.01)
                    .min_by(|&a1, &a2| {
                        let mse1 = Self::symmetric_mse(data, a1 * abs_max, self.bits);
                        let mse2 = Self::symmetric_mse(data, a2 * abs_max, self.bits);
                        mse1.partial_cmp(&mse2).unwrap_or(Ordering::Equal)
                    })
                    .unwrap_or(1.0);
                (-best_alpha * abs_max, best_alpha * abs_max)
            }
        };

        let levels = (1u64 << self.bits) as f64;
        let scale = (range_max - range_min).max(1e-8) / (levels - 1.0);
        let zero_point = (-(range_min / scale)).round() as i32;

        Ok(CalibrationResult {
            scale,
            zero_point,
            range_min,
            range_max,
        })
    }

    fn symmetric_mse(data: &[f64], max_val: f64, bits: u32) -> f64 {
        let levels = (1u64 << bits) as f64;
        let scale = (2.0 * max_val) / (levels - 1.0);
        data.iter()
            .map(|&x| {
                let q = (x / scale).round() * scale;
                (x - q).powi(2)
            })
            .sum::<f64>()
            / data.len() as f64
    }
}

/// Dynamic batching for inference serving.
#[derive(Debug, Clone)]
pub struct ModelBatcher {
    queue: VecDeque<Vec<f64>>,
}

impl ModelBatcher {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }

    pub fn add_request(&mut self, input: Vec<f64>) {
        self.queue.push_back(input);
    }

    /// Drain up to `max_size` requests into a batch.
    pub fn get_batch(&mut self, max_size: usize, _max_wait_ms: u64) -> Vec<Vec<f64>> {
        let take = max_size.min(self.queue.len());
        self.queue.drain(..take).collect()
    }

    pub fn pending(&self) -> usize {
        self.queue.len()
    }
}

impl Default for ModelBatcher {
    fn default() -> Self {
        Self::new()
    }
}


pub mod extensions;
pub use extensions::*;
pub mod advanced;
pub use advanced::*;

#[cfg(test)]
mod tests;
