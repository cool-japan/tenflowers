//! Concept Bottleneck Model (Koh et al. 2020).

use super::helpers::{sigmoid_f64, softmax_f64};
use super::shared::ClMlp;

/// How concept logits are transformed before being passed to the label predictor.
#[derive(Debug, Clone)]
pub enum ConceptActivation {
    /// Binary concepts via sigmoid (BCE concept loss).
    Sigmoid,
    /// Continuous concepts, identity activation (MSE concept loss).
    Linear,
}

/// Configuration for a [`ConceptBottleneckModel`].
#[derive(Debug, Clone)]
pub struct CbmConfig {
    pub input_dim: usize,
    pub n_concepts: usize,
    pub n_classes: usize,
    pub encoder_hidden: Vec<usize>,
    pub predictor_hidden: Vec<usize>,
    pub concept_activation: ConceptActivation,
    /// Weight on the concept prediction loss in the joint objective.
    pub lambda_concept: f64,
}

/// Concept Bottleneck Model (Koh et al., 2020).
///
/// `x → concept_encoder → [K concepts] → label_predictor → y`
pub struct ConceptBottleneckModel {
    pub concept_encoder: ClMlp,
    pub label_predictor: ClMlp,
    pub config: CbmConfig,
}

impl ConceptBottleneckModel {
    pub fn new(config: CbmConfig) -> Self {
        let mut enc_sizes = vec![config.input_dim];
        enc_sizes.extend_from_slice(&config.encoder_hidden);
        enc_sizes.push(config.n_concepts);

        let mut pred_sizes = vec![config.n_concepts];
        pred_sizes.extend_from_slice(&config.predictor_hidden);
        pred_sizes.push(config.n_classes);

        Self {
            concept_encoder: ClMlp::new(&enc_sizes),
            label_predictor: ClMlp::new(&pred_sizes),
            config,
        }
    }

    /// Encode input to concept probabilities `[n_concepts]`.
    pub fn encode_concepts(&self, x: &[f64]) -> Vec<f64> {
        let logits = self.concept_encoder.forward(x);
        match self.config.concept_activation {
            ConceptActivation::Sigmoid => logits.into_iter().map(sigmoid_f64).collect(),
            ConceptActivation::Linear => logits,
        }
    }

    /// Predict label logits from concept activations `[n_classes]`.
    pub fn predict_from_concepts(&self, concepts: &[f64]) -> Vec<f64> {
        self.label_predictor.forward(concepts)
    }

    /// Full forward pass: returns `(concept_probs, label_logits)`.
    pub fn forward(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let concepts = self.encode_concepts(x);
        let logits = self.predict_from_concepts(&concepts);
        (concepts, logits)
    }

    /// Concept prediction loss (BCE for Sigmoid, MSE for Linear).
    pub fn concept_loss(&self, x: &[f64], concept_targets: &[f64]) -> f64 {
        let concepts = self.encode_concepts(x);
        let n = concepts.len().min(concept_targets.len());
        if n == 0 {
            return 0.0;
        }
        match self.config.concept_activation {
            ConceptActivation::Sigmoid => {
                let mut loss = 0.0_f64;
                for i in 0..n {
                    let p = concepts[i].clamp(1e-7, 1.0 - 1e-7);
                    let t = concept_targets[i].clamp(0.0, 1.0);
                    loss -= t * p.ln() + (1.0 - t) * (1.0 - p).ln();
                }
                loss / n as f64
            }
            ConceptActivation::Linear => {
                concepts
                    .iter()
                    .zip(concept_targets.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>()
                    / n as f64
            }
        }
    }

    /// Cross-entropy task loss.
    pub fn task_loss(&self, x: &[f64], label: usize) -> f64 {
        let (concepts, logits) = self.forward(x);
        let _ = concepts;
        let probs = softmax_f64(&logits);
        let y = label.min(probs.len().saturating_sub(1));
        -(probs[y].max(1e-300)).ln()
    }

    /// Joint loss: `lambda_concept * concept_loss + task_loss`.
    pub fn joint_loss(&self, x: &[f64], concept_targets: &[f64], label: usize) -> f64 {
        let c_loss = self.concept_loss(x, concept_targets);
        let t_loss = self.task_loss(x, label);
        self.config.lambda_concept * c_loss + t_loss
    }

    /// One training step via finite-difference gradients.
    pub fn train_step(
        &mut self,
        x: &[f64],
        concept_targets: &[f64],
        label: usize,
        lr: f64,
    ) -> (f64, f64) {
        let c_loss = self.concept_loss(x, concept_targets);
        let t_loss = self.task_loss(x, label);

        let enc_target = concept_targets.to_vec();
        let enc_in = x.to_vec();
        let eps = 1e-5;
        let concept_loss_base = c_loss;

        let enc_layers = self.concept_encoder.layers.len();
        for l in 0..enc_layers {
            let out_dim = self.concept_encoder.layers[l].w.len();
            let in_dim = if out_dim == 0 {
                0
            } else {
                self.concept_encoder.layers[l].w[0].len()
            };
            for i in 0..out_dim {
                for j in 0..in_dim {
                    let orig = self.concept_encoder.layers[l].w[i][j];
                    self.concept_encoder.layers[l].w[i][j] += eps;
                    let loss_p = self.concept_loss(&enc_in, &enc_target);
                    self.concept_encoder.layers[l].w[i][j] = orig;
                    let grad = (loss_p - concept_loss_base) / eps;
                    self.concept_encoder.layers[l].w[i][j] -=
                        lr * self.config.lambda_concept * grad;
                }
                let orig = self.concept_encoder.layers[l].b[i];
                self.concept_encoder.layers[l].b[i] += eps;
                let loss_p = self.concept_loss(&enc_in, &enc_target);
                self.concept_encoder.layers[l].b[i] = orig;
                let grad = (loss_p - concept_loss_base) / eps;
                self.concept_encoder.layers[l].b[i] -= lr * self.config.lambda_concept * grad;
            }
        }

        let task_loss_base = t_loss;
        let pred_layers = self.label_predictor.layers.len();
        for l in 0..pred_layers {
            let out_dim = self.label_predictor.layers[l].w.len();
            let in_dim = if out_dim == 0 {
                0
            } else {
                self.label_predictor.layers[l].w[0].len()
            };
            for i in 0..out_dim {
                for j in 0..in_dim {
                    let orig = self.label_predictor.layers[l].w[i][j];
                    self.label_predictor.layers[l].w[i][j] += eps;
                    let loss_p = self.task_loss(x, label);
                    self.label_predictor.layers[l].w[i][j] = orig;
                    let grad = (loss_p - task_loss_base) / eps;
                    self.label_predictor.layers[l].w[i][j] -= lr * grad;
                }
                let orig = self.label_predictor.layers[l].b[i];
                self.label_predictor.layers[l].b[i] += eps;
                let loss_p = self.task_loss(x, label);
                self.label_predictor.layers[l].b[i] = orig;
                let grad = (loss_p - task_loss_base) / eps;
                self.label_predictor.layers[l].b[i] -= lr * grad;
            }
        }

        (c_loss, t_loss)
    }

    /// Test-time intervention: override concept `concept_idx` with `concept_value`.
    pub fn intervene(&self, x: &[f64], concept_idx: usize, concept_value: f64) -> Vec<f64> {
        let mut concepts = self.encode_concepts(x);
        if concept_idx < concepts.len() {
            concepts[concept_idx] = concept_value;
        }
        self.predict_from_concepts(&concepts)
    }

    /// Per-concept accuracy using 0.5 threshold.
    pub fn concept_accuracy(&self, x_batch: &[Vec<f64>], concept_batch: &[Vec<f64>]) -> Vec<f64> {
        let n_concepts = self.config.n_concepts;
        let n = x_batch.len().min(concept_batch.len());
        if n == 0 {
            return vec![0.0; n_concepts];
        }
        let mut correct = vec![0_usize; n_concepts];
        for (x, ctarget) in x_batch.iter().zip(concept_batch.iter()) {
            let preds = self.encode_concepts(x);
            for k in 0..n_concepts.min(ctarget.len()).min(preds.len()) {
                let pred_bin = if preds[k] >= 0.5 { 1.0_f64 } else { 0.0_f64 };
                let tgt_bin = if ctarget[k] >= 0.5 { 1.0_f64 } else { 0.0_f64 };
                if (pred_bin - tgt_bin).abs() < 1e-9 {
                    correct[k] += 1;
                }
            }
        }
        correct.into_iter().map(|c| c as f64 / n as f64).collect()
    }
}
