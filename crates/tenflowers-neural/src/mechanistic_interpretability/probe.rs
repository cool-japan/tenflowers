//! Linear probe classifier trained on cached activations.

use super::helpers::{dot, softmax, zeros_vec};

/// Configuration for the linear probe.
#[derive(Debug, Clone)]
pub struct ProbeConfig {
    /// Dimensionality of the input activation vectors.
    pub d_input: usize,
    /// Number of classes.
    pub n_classes: usize,
    /// SGD learning rate.
    pub learning_rate: f64,
    /// Number of training epochs.
    pub n_epochs: usize,
}

/// Lightweight linear classifier trained on cached activations to test whether
/// a concept is linearly decodable from the representation.
pub struct ProbeClassifier {
    /// Probe configuration.
    pub config: ProbeConfig,
    /// Weight matrix `[n_classes][d_input]`.
    pub weights: Vec<Vec<f64>>,
    /// Bias vector `[n_classes]`.
    pub bias: Vec<f64>,
}

impl ProbeClassifier {
    /// Create a zero-initialised probe.
    pub fn new(config: ProbeConfig) -> Self {
        let n = config.n_classes;
        let d = config.d_input;
        Self {
            config,
            weights: vec![zeros_vec(d); n],
            bias: zeros_vec(n),
        }
    }

    fn logits(&self, x: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(self.bias.iter())
            .map(|(w, &b)| dot(w, x) + b)
            .collect()
    }

    /// Fit the probe using SGD with cross-entropy loss.  Returns per-epoch mean loss.
    pub fn fit(&mut self, activations: &[Vec<f64>], labels: &[usize]) -> Vec<f64> {
        let n_samples = activations.len().min(labels.len());
        let n_cls = self.config.n_classes;
        let d = self.config.d_input;
        let lr = self.config.learning_rate;
        let mut history = Vec::with_capacity(self.config.n_epochs);

        for _ in 0..self.config.n_epochs {
            let mut epoch_loss = 0.0_f64;
            for idx in 0..n_samples {
                let x = &activations[idx];
                let y = labels[idx].min(n_cls - 1);

                let logits_v = self.logits(x);
                let probs = softmax(&logits_v);
                epoch_loss -= (probs[y] + 1e-30).ln();

                let mut grad = probs.clone();
                grad[y] -= 1.0;

                for c in 0..n_cls {
                    self.bias[c] -= lr * grad[c];
                    for j in 0..d {
                        self.weights[c][j] -= lr * grad[c] * x[j];
                    }
                }
            }
            history.push(epoch_loss / n_samples as f64);
        }

        history
    }

    /// Predict class index (argmax of logits) for a single activation vector.
    pub fn predict(&self, x: &[f64]) -> usize {
        let logits_v = self.logits(x);
        logits_v
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Return class probabilities (softmax of logits).
    pub fn predict_proba(&self, x: &[f64]) -> Vec<f64> {
        softmax(&self.logits(x))
    }

    /// Fraction of correctly classified samples.
    pub fn accuracy(&self, activations: &[Vec<f64>], labels: &[usize]) -> f64 {
        let n = activations.len().min(labels.len());
        if n == 0 {
            return 0.0;
        }
        let correct = activations[..n]
            .iter()
            .zip(labels[..n].iter())
            .filter(|(x, &y)| self.predict(x) == y)
            .count();
        correct as f64 / n as f64
    }
}
