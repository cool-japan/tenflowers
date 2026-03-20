//! ConceptSHAP — Monte-Carlo Shapley values over concept presence.

use super::probes::{ClProbeConfig, LinearProbe};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

/// Configuration for [`ConceptShap`].
#[derive(Debug, Clone)]
pub struct ConceptShapConfig {
    pub n_concepts: usize,
    pub n_coalition_samples: usize,
    pub n_output_classes: usize,
}

/// Monte-Carlo Shapley values over concept presence.
pub struct ConceptShap {
    pub concept_probes: Vec<LinearProbe>,
    pub config: ConceptShapConfig,
}

impl ConceptShap {
    pub fn new(feat_dim: usize, config: ConceptShapConfig) -> Self {
        let probes = (0..config.n_concepts)
            .map(|_| {
                LinearProbe::new(
                    feat_dim,
                    ClProbeConfig {
                        hidden_dim: 0,
                        n_classes: 2,
                        n_epochs: 10,
                        lr: 0.01,
                        l2_penalty: 1e-4,
                    },
                )
            })
            .collect();
        Self {
            concept_probes: probes,
            config,
        }
    }

    /// P(concept_k present | x) for each concept.
    pub fn concept_presence(&self, x: &[f64]) -> Vec<f64> {
        self.concept_probes
            .iter()
            .map(|probe| {
                let proba = probe.predict_proba(x);
                if proba.len() >= 2 {
                    proba[1]
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Monte-Carlo Shapley values for each concept given a target class.
    pub fn shapley_value(
        &self,
        x: &[f64],
        target_class: usize,
        model_fn: &dyn Fn(&[Vec<f64>]) -> Vec<f64>,
    ) -> Vec<f64> {
        let k = self.config.n_concepts;
        let m = self.config.n_coalition_samples;
        if k == 0 || m == 0 {
            return vec![0.0; k];
        }

        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        let presence = self.concept_presence(x);
        let mut shap = vec![0.0_f64; k];

        for _ in 0..m {
            let mut perm: Vec<usize> = (0..k).collect();
            for i in (1..k).rev() {
                let j = rng.random_range(0..=i);
                perm.swap(i, j);
            }
            let mut coalition: Vec<f64> = vec![0.0; k];
            let mut prev_val = {
                let inp = vec![coalition.clone()];
                let out = model_fn(&inp);
                out.get(target_class).copied().unwrap_or(0.0)
            };
            for &concept_idx in perm.iter() {
                coalition[concept_idx] = presence[concept_idx];
                let inp = vec![coalition.clone()];
                let out = model_fn(&inp);
                let new_val = out.get(target_class).copied().unwrap_or(0.0);
                shap[concept_idx] += new_val - prev_val;
                prev_val = new_val;
            }
        }

        shap.iter_mut().for_each(|v| *v /= m as f64);
        shap
    }

    /// Shapley values sorted by absolute magnitude (descending).
    pub fn concept_importance_ranking(
        &self,
        x: &[f64],
        target_class: usize,
        model_fn: &dyn Fn(&[Vec<f64>]) -> Vec<f64>,
    ) -> Vec<(usize, f64)> {
        let shap = self.shapley_value(x, target_class, model_fn);
        let mut ranked: Vec<(usize, f64)> = shap.into_iter().enumerate().collect();
        ranked.sort_by(|(_, a), (_, b)| {
            b.abs()
                .partial_cmp(&a.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked
    }
}
