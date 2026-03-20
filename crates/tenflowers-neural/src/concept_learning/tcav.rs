//! TCAV — Testing with Concept Activation Vectors (Kim et al. 2018).

use super::helpers::{dot_f64, l2_norm_f64, sigmoid_f64};

/// Configuration for TCAV analysis.
#[derive(Debug, Clone)]
pub struct TcavConfig {
    /// Dimensionality of internal activations.
    pub layer_dim: usize,
    /// Number of concepts.
    pub n_concepts: usize,
    /// Number of samples to estimate directional derivatives.
    pub n_tcav_samples: usize,
}

/// A Concept Activation Vector learned via logistic regression in activation space.
#[derive(Debug, Clone)]
pub struct ConceptActivationVector {
    pub concept_name: String,
    /// Direction in activation space `[layer_dim]`.
    pub cav: Vec<f64>,
    /// Linear probe accuracy on held-out examples.
    pub probe_accuracy: f64,
}

impl ConceptActivationVector {
    /// Train a CAV via logistic regression on positive/negative activations.
    pub fn train(
        positive_acts: &[Vec<f64>],
        negative_acts: &[Vec<f64>],
        concept_name: &str,
    ) -> Self {
        let n_pos = positive_acts.len();
        let n_neg = negative_acts.len();
        let n_total = n_pos + n_neg;
        if n_total == 0 || positive_acts.is_empty() || negative_acts.is_empty() {
            return Self {
                concept_name: concept_name.to_string(),
                cav: Vec::new(),
                probe_accuracy: 0.0,
            };
        }
        let dim = positive_acts[0].len();
        if dim == 0 {
            return Self {
                concept_name: concept_name.to_string(),
                cav: Vec::new(),
                probe_accuracy: 0.0,
            };
        }

        let mut all_acts: Vec<&Vec<f64>> = Vec::with_capacity(n_total);
        let mut labels: Vec<f64> = Vec::with_capacity(n_total);
        for a in positive_acts.iter() {
            all_acts.push(a);
            labels.push(1.0);
        }
        for a in negative_acts.iter() {
            all_acts.push(a);
            labels.push(0.0);
        }

        let mut w = vec![0.0_f64; dim];
        let mut b = 0.0_f64;
        let lr = 0.01;
        let n_epochs = 50;

        for _epoch in 0..n_epochs {
            for (act, &y) in all_acts.iter().zip(labels.iter()) {
                let logit = dot_f64(&w, act) + b;
                let p = sigmoid_f64(logit);
                let err = p - y;
                for (wj, &xj) in w.iter_mut().zip(act.iter()) {
                    *wj -= lr * err * xj;
                }
                b -= lr * err;
            }
        }

        let correct = all_acts
            .iter()
            .zip(labels.iter())
            .filter(|(act, &y)| {
                let logit = dot_f64(&w, act) + b;
                let pred = if logit >= 0.0 { 1.0 } else { 0.0 };
                (pred - y).abs() < 1e-9
            })
            .count();
        let probe_accuracy = correct as f64 / n_total as f64;

        let norm = l2_norm_f64(&w).max(1e-12);
        let cav: Vec<f64> = w.iter().map(|&wi| wi / norm).collect();

        Self {
            concept_name: concept_name.to_string(),
            cav,
            probe_accuracy,
        }
    }
}

/// TCAV analyser that stores a collection of CAVs and computes TCAV scores.
pub struct TcavAnalyzer {
    pub cavs: Vec<ConceptActivationVector>,
    pub config: TcavConfig,
}

impl TcavAnalyzer {
    pub fn new(config: TcavConfig) -> Self {
        Self {
            cavs: Vec::new(),
            config,
        }
    }

    pub fn add_cav(&mut self, cav: ConceptActivationVector) {
        self.cavs.push(cav);
    }

    /// TCAV score: fraction of examples where directional derivative > 0.
    pub fn tcav_score(
        &self,
        cav_idx: usize,
        model_fn: &dyn Fn(&[f64]) -> f64,
        activations: &[Vec<f64>],
    ) -> f64 {
        let cav = match self.cavs.get(cav_idx) {
            Some(c) => &c.cav,
            None => return 0.0,
        };
        if activations.is_empty() || cav.is_empty() {
            return 0.0;
        }
        let eps = 1e-4;
        let mut positive_count = 0_usize;
        for h in activations.iter() {
            if h.len() != cav.len() {
                continue;
            }
            let h_plus: Vec<f64> = h
                .iter()
                .zip(cav.iter())
                .map(|(hi, ci)| hi + eps * ci)
                .collect();
            let h_minus: Vec<f64> = h
                .iter()
                .zip(cav.iter())
                .map(|(hi, ci)| hi - eps * ci)
                .collect();
            let dd = (model_fn(&h_plus) - model_fn(&h_minus)) / (2.0 * eps);
            if dd > 0.0 {
                positive_count += 1;
            }
        }
        positive_count as f64 / activations.len() as f64
    }

    /// Per-example directional derivatives for a given CAV.
    pub fn concept_sensitivity(
        &self,
        cav_idx: usize,
        activations: &[Vec<f64>],
        model_fn: &dyn Fn(&[f64]) -> f64,
    ) -> Vec<f64> {
        let cav = match self.cavs.get(cav_idx) {
            Some(c) => &c.cav,
            None => return vec![0.0; activations.len()],
        };
        let eps = 1e-4;
        activations
            .iter()
            .map(|h| {
                if h.len() != cav.len() {
                    return 0.0;
                }
                let h_plus: Vec<f64> = h
                    .iter()
                    .zip(cav.iter())
                    .map(|(hi, ci)| hi + eps * ci)
                    .collect();
                let h_minus: Vec<f64> = h
                    .iter()
                    .zip(cav.iter())
                    .map(|(hi, ci)| hi - eps * ci)
                    .collect();
                (model_fn(&h_plus) - model_fn(&h_minus)) / (2.0 * eps)
            })
            .collect()
    }

    /// TCAV scores for all stored CAVs.
    pub fn all_tcav_scores(
        &self,
        activations: &[Vec<f64>],
        model_fn: &dyn Fn(&[f64]) -> f64,
    ) -> Vec<(String, f64)> {
        (0..self.cavs.len())
            .map(|i| {
                let score = self.tcav_score(i, model_fn, activations);
                (self.cavs[i].concept_name.clone(), score)
            })
            .collect()
    }
}
