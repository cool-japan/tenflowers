//! Calibration tools: ECE, temperature scaling, NLL, deep ensembles.

use super::helpers::{sample_normal, softmax};
use super::shared::BdlMlp;
use scirs2_core::random::{rngs::StdRng, Rng};

// ─────────────────────────────────────────────────────────────────────────────
// Calibration metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Results of a calibration evaluation.
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    pub ece: f64,
    pub mce: f64,
    pub brier_score: f64,
    /// `(bin_center_confidence, bin_accuracy, bin_count)`.
    pub reliability_diagram: Vec<(f64, f64, usize)>,
}

/// Compute calibration metrics (ECE, MCE, Brier score, reliability diagram).
pub fn compute_calibration(
    probs: &[Vec<f64>],
    labels: &[usize],
    n_bins: usize,
) -> CalibrationResult {
    assert_eq!(probs.len(), labels.len());
    let n_bins = n_bins.max(1);
    let n = probs.len();
    let bin_width = 1.0 / n_bins as f64;
    let mut bin_correct = vec![0usize; n_bins];
    let mut bin_conf_sum = vec![0.0f64; n_bins];
    let mut bin_count = vec![0usize; n_bins];

    for (p, &label) in probs.iter().zip(labels.iter()) {
        let (pred_class, max_prob) =
            p.iter()
                .enumerate()
                .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                });
        let conf = max_prob.clamp(0.0, 1.0);
        let bin_idx = ((conf / bin_width).floor() as usize).min(n_bins - 1);
        bin_count[bin_idx] += 1;
        bin_conf_sum[bin_idx] += conf;
        if pred_class == label {
            bin_correct[bin_idx] += 1;
        }
    }

    let mut ece = 0.0;
    let mut mce = 0.0f64;
    let mut reliability_diagram = Vec::with_capacity(n_bins);

    for b in 0..n_bins {
        let count = bin_count[b];
        let bin_center = (b as f64 + 0.5) * bin_width;
        if count == 0 {
            reliability_diagram.push((bin_center, 0.0, 0));
            continue;
        }
        let acc = bin_correct[b] as f64 / count as f64;
        let conf_avg = bin_conf_sum[b] / count as f64;
        let diff = (acc - conf_avg).abs();
        ece += (count as f64 / n as f64) * diff;
        if diff > mce {
            mce = diff;
        }
        reliability_diagram.push((bin_center, acc, count));
    }

    let brier_score = if n == 0 {
        0.0
    } else {
        probs
            .iter()
            .zip(labels.iter())
            .map(|(p, &label)| {
                p.iter()
                    .enumerate()
                    .map(|(c, &prob)| {
                        let t = if c == label { 1.0 } else { 0.0 };
                        (prob - t).powi(2)
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
            / n as f64
    };

    CalibrationResult {
        ece,
        mce,
        brier_score,
        reliability_diagram,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Temperature Scaling
// ─────────────────────────────────────────────────────────────────────────────

/// Post-hoc calibration via temperature scaling.
#[derive(Clone, Debug)]
pub struct TemperatureScaling {
    pub temperature: f64,
}

impl Default for TemperatureScaling {
    fn default() -> Self {
        Self::new()
    }
}

impl TemperatureScaling {
    pub fn new() -> Self {
        Self { temperature: 1.0 }
    }

    /// Optimise temperature via NLL minimisation on validation logits.
    pub fn fit(&mut self, logits: &[Vec<f64>], labels: &[usize], n_steps: usize) {
        let eps = 1e-4;
        let lr = 0.01;
        for _ in 0..n_steps {
            let nll_plus = {
                let probs = self.calibrate_with_temp(logits, self.temperature + eps);
                nll_classification(&probs, labels)
            };
            let nll_minus = {
                let probs = self.calibrate_with_temp(logits, (self.temperature - eps).max(1e-6));
                nll_classification(&probs, labels)
            };
            let grad = (nll_plus - nll_minus) / (2.0 * eps);
            self.temperature = (self.temperature - lr * grad).max(1e-3);
        }
    }

    fn calibrate_with_temp(&self, logits: &[Vec<f64>], temp: f64) -> Vec<Vec<f64>> {
        logits
            .iter()
            .map(|lgs| softmax(&lgs.iter().map(|&l| l / temp).collect::<Vec<_>>()))
            .collect()
    }

    /// Apply `softmax(logits / temperature)`.
    pub fn calibrate(&self, logits: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.calibrate_with_temp(logits, self.temperature)
    }
}

/// NLL for classification: `-mean log(probs[i][labels[i]])`.
pub fn nll_classification(probs: &[Vec<f64>], labels: &[usize]) -> f64 {
    if probs.is_empty() {
        return 0.0;
    }
    let n = probs.len() as f64;
    let total: f64 = probs
        .iter()
        .zip(labels.iter())
        .map(|(p, &label)| {
            let prob = if label < p.len() {
                p[label].max(1e-12)
            } else {
                1e-12
            };
            -prob.ln()
        })
        .sum();
    total / n
}

// ─────────────────────────────────────────────────────────────────────────────
// Deep Ensemble
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for deep ensembles.
#[derive(Clone, Debug)]
pub struct DeepEnsembleConfig {
    pub n_members: usize,
    pub lr: f64,
    pub n_epochs: usize,
    pub adversarial_training: bool,
    pub eps_adv: f64,
}

impl Default for DeepEnsembleConfig {
    fn default() -> Self {
        Self {
            n_members: 5,
            lr: 1e-3,
            n_epochs: 10,
            adversarial_training: false,
            eps_adv: 0.01,
        }
    }
}

/// Deep ensemble of independently trained BdlMlp models.
pub struct DeepEnsemble {
    pub members: Vec<BdlMlp>,
    pub config: DeepEnsembleConfig,
}

impl DeepEnsemble {
    pub fn new(layer_sizes: &[usize], config: DeepEnsembleConfig) -> Self {
        let members = (0..config.n_members)
            .map(|_| BdlMlp::new(layer_sizes))
            .collect();
        Self { members, config }
    }

    /// Train each ensemble member independently with optional FGSM augmentation.
    pub fn fit(&mut self, x_data: &[Vec<f64>], y_data: &[Vec<f64>], rng: &mut StdRng) {
        let eps_fd = 1e-5;
        let lr = self.config.lr;
        let n_epochs = self.config.n_epochs;
        let eps_adv = self.config.eps_adv;
        let use_adv = self.config.adversarial_training;

        for member in self.members.iter_mut() {
            let mut params = member.params_flat();
            for p in params.iter_mut() {
                *p += 0.01 * sample_normal(rng);
            }
            member.set_params(&params);

            for _ in 0..n_epochs {
                let mut grad = member.batch_gradient_fd(x_data, y_data, eps_fd);
                if use_adv && !x_data.is_empty() {
                    let mut adv_extra = vec![0.0f64; grad.len()];
                    for (x, y) in x_data.iter().zip(y_data.iter()) {
                        let x_grad = member.gradient_fd(x, y, eps_fd);
                        let x_adv: Vec<f64> = x
                            .iter()
                            .zip(x_grad.iter())
                            .map(|(&xi, &gi)| xi + eps_adv * gi.signum())
                            .collect();
                        let adv_grad = member.gradient_fd(&x_adv, y, eps_fd);
                        for (ae, ag) in adv_extra.iter_mut().zip(adv_grad.iter()) {
                            *ae += ag;
                        }
                    }
                    let n = x_data.len() as f64;
                    for (g, ae) in grad.iter_mut().zip(adv_extra.iter()) {
                        *g += ae / n;
                    }
                }
                let p = member.params_flat();
                let new_p: Vec<f64> = p
                    .iter()
                    .zip(grad.iter())
                    .map(|(&pi, &gi)| pi - lr * gi)
                    .collect();
                member.set_params(&new_p);
            }
        }
    }

    /// Predict by averaging ensemble members.
    pub fn predict(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        if self.members.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let preds: Vec<Vec<f64>> = self.members.iter().map(|m| m.forward(x)).collect();
        let out_dim = preds[0].len();
        let n = preds.len() as f64;
        let mean: Vec<f64> = (0..out_dim)
            .map(|d| preds.iter().map(|p| p[d]).sum::<f64>() / n)
            .collect();
        let variance: Vec<f64> = (0..out_dim)
            .map(|d| {
                let m = mean[d];
                preds.iter().map(|p| (p[d] - m).powi(2)).sum::<f64>() / n
            })
            .collect();
        (mean, variance)
    }

    /// Epistemic uncertainty: mean variance across output dimensions.
    pub fn epistemic_uncertainty(&self, x: &[f64]) -> f64 {
        let (_, variance) = self.predict(x);
        if variance.is_empty() {
            return 0.0;
        }
        variance.iter().sum::<f64>() / variance.len() as f64
    }

    /// Aleatoric uncertainty: mean within-member output variance (proxy).
    pub fn aleatoric_uncertainty(&self, x: &[f64]) -> f64 {
        if self.members.is_empty() {
            return 0.0;
        }
        let all_preds: Vec<Vec<f64>> = self.members.iter().map(|m| m.forward(x)).collect();
        let n_members = all_preds.len() as f64;
        let out_dim = all_preds[0].len();
        let total_var: f64 = (0..out_dim)
            .map(|d| {
                let mean_sq = all_preds.iter().map(|p| p[d].powi(2)).sum::<f64>() / n_members;
                let sq_mean = (all_preds.iter().map(|p| p[d]).sum::<f64>() / n_members).powi(2);
                (mean_sq - sq_mean).max(0.0)
            })
            .sum::<f64>();
        total_var / out_dim.max(1) as f64
    }
}
