//! Linear probing classifiers: LinearProbe and MultiProbe.

use super::helpers::{dot_f64, softmax_f64};

/// Configuration for a [`LinearProbe`] or [`MultiProbe`].
///
/// Named `ClProbeConfig` to avoid collision with `mechanistic_interpretability::ProbeConfig`.
#[derive(Debug, Clone)]
pub struct ClProbeConfig {
    /// If > 0, insert one hidden layer of this width (MLP probe); otherwise linear.
    pub hidden_dim: usize,
    pub n_classes: usize,
    pub n_epochs: usize,
    pub lr: f64,
    /// L2 weight-decay coefficient.
    pub l2_penalty: f64,
}

/// Linear (or MLP) probing classifier.
pub struct LinearProbe {
    /// Weight matrix `[n_classes][feat_dim]`.
    pub w: Vec<Vec<f64>>,
    /// Bias vector `[n_classes]`.
    pub b: Vec<f64>,
    pub config: ClProbeConfig,
    feat_dim: usize,
}

impl LinearProbe {
    pub fn new(feat_dim: usize, config: ClProbeConfig) -> Self {
        let n = config.n_classes;
        Self {
            w: vec![vec![0.0; feat_dim]; n],
            b: vec![0.0; n],
            config,
            feat_dim,
        }
    }

    fn logits(&self, x: &[f64]) -> Vec<f64> {
        self.w
            .iter()
            .zip(self.b.iter())
            .map(|(row, &bi)| dot_f64(row, x) + bi)
            .collect()
    }

    /// Fit using SGD with cross-entropy + L2 regularisation.
    pub fn fit(&mut self, features: &[Vec<f64>], labels: &[usize]) -> Vec<f64> {
        let n_samples = features.len().min(labels.len());
        let n_cls = self.config.n_classes;
        let d = self.feat_dim;
        let lr = self.config.lr;
        let lambda = self.config.l2_penalty;
        let mut history = Vec::with_capacity(self.config.n_epochs);

        for _ep in 0..self.config.n_epochs {
            let mut epoch_loss = 0.0_f64;
            for idx in 0..n_samples {
                let x = &features[idx];
                let y = labels[idx].min(n_cls - 1);
                let logits_v = self.logits(x);
                let probs = softmax_f64(&logits_v);
                epoch_loss -= (probs[y].max(1e-300)).ln();

                let mut grad_sm = probs.clone();
                grad_sm[y] -= 1.0;

                for c in 0..n_cls {
                    self.b[c] -= lr * grad_sm[c];
                    for j in 0..d.min(x.len()) {
                        self.w[c][j] -= lr * (grad_sm[c] * x[j] + lambda * self.w[c][j]);
                    }
                }
            }
            history.push(epoch_loss / n_samples.max(1) as f64);
        }
        history
    }

    /// Predict class (argmax of logits).
    pub fn predict(&self, x: &[f64]) -> usize {
        let logits_v = self.logits(x);
        logits_v
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Softmax class probabilities.
    pub fn predict_proba(&self, x: &[f64]) -> Vec<f64> {
        softmax_f64(&self.logits(x))
    }

    /// Classification accuracy.
    pub fn accuracy(&self, features: &[Vec<f64>], labels: &[usize]) -> f64 {
        let n = features.len().min(labels.len());
        if n == 0 {
            return 0.0;
        }
        let correct = features[..n]
            .iter()
            .zip(labels[..n].iter())
            .filter(|(x, &y)| self.predict(x) == y)
            .count();
        correct as f64 / n as f64
    }

    /// Selectivity: `max|w_j| / sum|w_j|`.
    pub fn selectivity_score(&self) -> f64 {
        let flat: Vec<f64> = self.w.iter().flat_map(|row| row.iter().cloned()).collect();
        let abs_vals: Vec<f64> = flat.iter().map(|v| v.abs()).collect();
        let total: f64 = abs_vals.iter().sum();
        if total < 1e-12 {
            return 0.0;
        }
        let max_val = abs_vals.iter().cloned().fold(0.0_f64, f64::max);
        (max_val / total).min(1.0)
    }
}

/// A collection of binary probes, one per concept.
pub struct MultiProbe {
    pub probes: Vec<LinearProbe>,
    pub concept_names: Vec<String>,
}

impl MultiProbe {
    pub fn new(feat_dim: usize, n_concepts: usize, config: ClProbeConfig) -> Self {
        let probes = (0..n_concepts)
            .map(|_| {
                LinearProbe::new(
                    feat_dim,
                    ClProbeConfig {
                        n_classes: 2,
                        ..config.clone()
                    },
                )
            })
            .collect();
        let concept_names = (0..n_concepts).map(|k| format!("concept_{k}")).collect();
        Self {
            probes,
            concept_names,
        }
    }

    /// Train each probe on its binary concept column.
    pub fn fit(&mut self, features: &[Vec<f64>], concept_labels: &[Vec<bool>]) {
        let n = features.len().min(concept_labels.len());
        for (k, probe) in self.probes.iter_mut().enumerate() {
            let labels: Vec<usize> = (0..n)
                .map(|i| {
                    if k < concept_labels[i].len() && concept_labels[i][k] {
                        1
                    } else {
                        0
                    }
                })
                .collect();
            probe.fit(features, &labels);
        }
    }

    /// Predict all concept booleans for a single example.
    pub fn predict_concepts(&self, x: &[f64]) -> Vec<bool> {
        self.probes.iter().map(|p| p.predict(x) == 1).collect()
    }

    /// Per-concept classification accuracy.
    pub fn accuracy_per_concept(
        &self,
        features: &[Vec<f64>],
        concept_labels: &[Vec<bool>],
    ) -> Vec<f64> {
        let n = features.len().min(concept_labels.len());
        self.probes
            .iter()
            .enumerate()
            .map(|(k, probe)| {
                let labels: Vec<usize> = (0..n)
                    .map(|i| {
                        if k < concept_labels[i].len() && concept_labels[i][k] {
                            1
                        } else {
                            0
                        }
                    })
                    .collect();
                probe.accuracy(features, &labels)
            })
            .collect()
    }
}
