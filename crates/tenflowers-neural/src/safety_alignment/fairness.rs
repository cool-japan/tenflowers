//! Section 3 — Fairness & Bias Mitigation
//!
//! - `DemographicParityChecker` + `FairnessReport`
//! - `EqualizedOdds` + `EqualizedOddsReport`
//! - `CalibrationFairness`
//! - `ReweightingDebias`
//! - `AdversarialDebias`

use super::helpers::{disparity, expected_calibration_error, sigmoid};

/// Report returned by demographic parity checks.
#[derive(Debug, Clone)]
pub struct FairnessReport {
    /// Positive prediction rate per group (group_id → rate).
    pub positive_rates: Vec<(usize, f64)>,
    /// Maximum absolute difference in positive rates across groups.
    pub max_disparity: f64,
    /// Whether the disparity is within `tolerance`.
    pub is_fair: bool,
}

/// Checks whether the positive prediction rate is equal across groups.
#[derive(Debug, Clone)]
pub struct DemographicParityChecker {
    /// Maximum allowed disparity in positive rates.
    pub tolerance: f64,
}

impl DemographicParityChecker {
    /// Create a new checker.
    pub fn new(tolerance: f64) -> Self {
        Self {
            tolerance: tolerance.max(0.0),
        }
    }

    /// Compute a [`FairnessReport`] from binary `predictions` and `groups`.
    pub fn check(&self, predictions: &[u8], groups: &[usize]) -> FairnessReport {
        let mut group_totals: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut group_positives: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();

        for (&pred, &grp) in predictions.iter().zip(groups.iter()) {
            *group_totals.entry(grp).or_insert(0) += 1;
            if pred == 1 {
                *group_positives.entry(grp).or_insert(0) += 1;
            }
        }

        let mut positive_rates: Vec<(usize, f64)> = group_totals
            .iter()
            .map(|(&grp, &total)| {
                let pos = group_positives.get(&grp).copied().unwrap_or(0);
                let rate = if total > 0 {
                    pos as f64 / total as f64
                } else {
                    0.0
                };
                (grp, rate)
            })
            .collect();
        positive_rates.sort_by_key(|(g, _)| *g);

        let rates: Vec<f64> = positive_rates.iter().map(|(_, r)| *r).collect();
        let max_rate = rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_rate = rates.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_disparity = if rates.len() >= 2 {
            (max_rate - min_rate).max(0.0)
        } else {
            0.0
        };

        FairnessReport {
            positive_rates,
            max_disparity,
            is_fair: max_disparity <= self.tolerance,
        }
    }
}

/// Report produced by the equalized-odds check.
#[derive(Debug, Clone)]
pub struct EqualizedOddsReport {
    /// True positive rate per group.
    pub tpr_per_group: Vec<(usize, f64)>,
    /// False positive rate per group.
    pub fpr_per_group: Vec<(usize, f64)>,
    /// Maximum TPR disparity.
    pub max_tpr_disparity: f64,
    /// Maximum FPR disparity.
    pub max_fpr_disparity: f64,
}

/// Checks equalized odds: TPR and FPR should be equal across groups.
#[derive(Debug, Clone)]
pub struct EqualizedOdds;

impl EqualizedOdds {
    /// Create a new checker.
    pub fn new() -> Self {
        Self
    }

    /// Compute an [`EqualizedOddsReport`].
    pub fn compute(
        &self,
        predictions: &[u8],
        labels: &[u8],
        groups: &[usize],
    ) -> EqualizedOddsReport {
        let mut tp: std::collections::HashMap<usize, usize> = Default::default();
        let mut fp: std::collections::HashMap<usize, usize> = Default::default();
        let mut fn_: std::collections::HashMap<usize, usize> = Default::default();
        let mut tn: std::collections::HashMap<usize, usize> = Default::default();

        for ((&pred, &label), &grp) in predictions.iter().zip(labels.iter()).zip(groups.iter()) {
            match (pred, label) {
                (1, 1) => *tp.entry(grp).or_insert(0) += 1,
                (1, 0) => *fp.entry(grp).or_insert(0) += 1,
                (0, 1) => *fn_.entry(grp).or_insert(0) += 1,
                _ => *tn.entry(grp).or_insert(0) += 1,
            }
        }

        let all_groups: std::collections::BTreeSet<usize> = groups.iter().cloned().collect();

        let mut tpr_per_group: Vec<(usize, f64)> = Vec::new();
        let mut fpr_per_group: Vec<(usize, f64)> = Vec::new();

        for grp in &all_groups {
            let t = tp.get(grp).copied().unwrap_or(0);
            let f = fn_.get(grp).copied().unwrap_or(0);
            let tpr = if t + f > 0 {
                t as f64 / (t + f) as f64
            } else {
                0.0
            };
            let fp_g = fp.get(grp).copied().unwrap_or(0);
            let tn_g = tn.get(grp).copied().unwrap_or(0);
            let fpr = if fp_g + tn_g > 0 {
                fp_g as f64 / (fp_g + tn_g) as f64
            } else {
                0.0
            };
            tpr_per_group.push((*grp, tpr));
            fpr_per_group.push((*grp, fpr));
        }

        let max_tpr_disparity = disparity(tpr_per_group.iter().map(|(_, r)| *r).collect());
        let max_fpr_disparity = disparity(fpr_per_group.iter().map(|(_, r)| *r).collect());

        EqualizedOddsReport {
            tpr_per_group,
            fpr_per_group,
            max_tpr_disparity,
            max_fpr_disparity,
        }
    }
}

impl Default for EqualizedOdds {
    fn default() -> Self {
        Self::new()
    }
}

/// Calibration fairness: check Expected Calibration Error per group.
#[derive(Debug, Clone)]
pub struct CalibrationFairness;

impl CalibrationFairness {
    /// Create a new calibration fairness checker.
    pub fn new() -> Self {
        Self
    }

    /// Compute ECE per group.
    pub fn ece_per_group(
        &self,
        probabilities: &[f64],
        labels: &[u8],
        groups: &[usize],
    ) -> Vec<(usize, f64)> {
        let mut group_data: std::collections::HashMap<usize, Vec<(f64, u8)>> = Default::default();
        for ((&prob, &label), &grp) in probabilities.iter().zip(labels.iter()).zip(groups.iter()) {
            group_data.entry(grp).or_default().push((prob, label));
        }

        let n_bins = 10usize;
        let mut result: Vec<(usize, f64)> = group_data
            .iter()
            .map(|(&grp, data)| {
                let ece = expected_calibration_error(data, n_bins);
                (grp, ece)
            })
            .collect();
        result.sort_by_key(|(g, _)| *g);
        result
    }
}

impl Default for CalibrationFairness {
    fn default() -> Self {
        Self::new()
    }
}

/// Reweighting debiasing: compute instance weights to enforce demographic parity.
#[derive(Debug, Clone, Default)]
pub struct ReweightingDebias;

impl ReweightingDebias {
    /// Create a new reweighter.
    pub fn new() -> Self {
        Self
    }

    /// Compute instance weights from binary `labels` and `groups`.
    pub fn compute_weights(&self, labels: &[u8], groups: &[usize]) -> Vec<f64> {
        let n = labels.len();
        if n == 0 {
            return Vec::new();
        }

        let p_y1 = labels.iter().filter(|&&l| l == 1).count() as f64 / n as f64;
        let p_y0 = 1.0 - p_y1;

        let mut group_total: std::collections::HashMap<usize, usize> = Default::default();
        let mut group_pos: std::collections::HashMap<usize, usize> = Default::default();
        for (&label, &grp) in labels.iter().zip(groups.iter()) {
            *group_total.entry(grp).or_insert(0) += 1;
            if label == 1 {
                *group_pos.entry(grp).or_insert(0) += 1;
            }
        }

        labels
            .iter()
            .zip(groups.iter())
            .map(|(&label, &grp)| {
                let total_g = *group_total.get(&grp).unwrap_or(&1) as f64;
                let pos_g = *group_pos.get(&grp).unwrap_or(&0) as f64;
                let p_y_given_g = if label == 1 {
                    (pos_g / total_g).max(1e-12)
                } else {
                    ((total_g - pos_g) / total_g).max(1e-12)
                };
                let p_y = if label == 1 { p_y1 } else { p_y0 };
                p_y / p_y_given_g
            })
            .collect()
    }
}

/// Adversarial debiasing via a linear adversary trained on representations.
#[derive(Debug, Clone)]
pub struct AdversarialDebias {
    /// Learning rate for the adversary.
    pub adversary_lr: f64,
    /// Number of adversary update steps.
    pub adversary_steps: usize,
    /// Feature dimensionality.
    pub n_features: usize,
    /// Adversary linear weights.
    pub adversary_weights: Vec<f64>,
}

impl AdversarialDebias {
    /// Create a new adversarial debiasr.
    pub fn new(n_features: usize, adversary_lr: f64, adversary_steps: usize) -> Self {
        Self {
            adversary_lr,
            adversary_steps,
            n_features,
            adversary_weights: vec![0.0; n_features],
        }
    }

    /// Train the adversary; returns accuracy on protected attribute prediction.
    pub fn train_adversary(
        &mut self,
        representations: &[Vec<f64>],
        protected_labels: &[u8],
    ) -> f64 {
        let n = representations.len().min(protected_labels.len());
        if n == 0 || self.n_features == 0 {
            return 0.0;
        }

        for _ in 0..self.adversary_steps {
            for i in 0..n {
                let rep = &representations[i];
                let label = protected_labels[i] as f64;
                let logit: f64 = rep
                    .iter()
                    .zip(self.adversary_weights.iter())
                    .map(|(r, w)| r * w)
                    .sum();
                let pred = sigmoid(logit);
                let error = pred - label;
                for (j, w) in self.adversary_weights.iter_mut().enumerate() {
                    let feat = rep.get(j).copied().unwrap_or(0.0);
                    *w -= self.adversary_lr * error * feat;
                }
            }
        }

        let correct = representations
            .iter()
            .zip(protected_labels.iter())
            .filter(|(rep, &label)| {
                let logit: f64 = rep
                    .iter()
                    .zip(self.adversary_weights.iter())
                    .map(|(r, w)| r * w)
                    .sum();
                let pred = if sigmoid(logit) >= 0.5 { 1u8 } else { 0u8 };
                pred == label
            })
            .count();
        correct as f64 / n as f64
    }
}
