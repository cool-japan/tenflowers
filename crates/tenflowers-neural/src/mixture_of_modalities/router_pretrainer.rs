//! Modality-based MoE router and multi-modal pretraining objectives.

use scirs2_core::random::{rngs::StdRng, SeedableRng};

use super::embedding_transformer::{ModalityEmbedding, UnifiedTransformer};
use super::types::{
    cosine_similarity, gelu, mat_vec, pool_mean, softmax, xavier_init, ModalSequence,
    MomModalityType,
};

// ─────────────────────────────────────────────────────────────────────────────
// §8 Modality Router
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the modality-based MoE router.
#[derive(Debug, Clone)]
pub struct ModalityRouterConfig {
    pub d_model: usize,
    pub n_modalities: usize,
    pub expert_dim: usize,
    pub top_k: usize,
}

/// A single modality-specific feed-forward expert.
pub struct ModalityExpert {
    /// Up-projection: `[expert_dim][d_model]`.
    pub w1: Vec<Vec<f64>>,
    /// Down-projection: `[d_model][expert_dim]`.
    pub w2: Vec<Vec<f64>>,
}

impl ModalityExpert {
    pub(super) fn new(d_model: usize, expert_dim: usize, rng: &mut StdRng) -> Self {
        Self {
            w1: xavier_init(rng, expert_dim, d_model),
            w2: xavier_init(rng, d_model, expert_dim),
        }
    }

    pub(super) fn forward(&self, x: &[f64]) -> Vec<f64> {
        let h: Vec<f64> = mat_vec(&self.w1, x).iter().map(|&v| gelu(v)).collect();
        mat_vec(&self.w2, &h)
    }
}

/// Routes tokens to modality-specific experts.
pub struct ModalityRouter {
    pub config: ModalityRouterConfig,
    /// Router weight matrix: `[n_modalities][d_model]`.
    pub router_weights: Vec<Vec<f64>>,
    pub experts: Vec<ModalityExpert>,
}

impl ModalityRouter {
    /// Create a new router with one expert per modality.
    pub fn new(config: ModalityRouterConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(2024);
        let n = config.n_modalities;
        let d = config.d_model;
        let ed = config.expert_dim;
        let router_weights = xavier_init(&mut rng, n, d);
        let experts: Vec<ModalityExpert> = (0..n)
            .map(|_| ModalityExpert::new(d, ed, &mut rng))
            .collect();
        Self {
            config,
            router_weights,
            experts,
        }
    }

    /// Route a single token vector using hard routing by modality type.
    pub fn route(&self, x: &[f64], modality: &MomModalityType) -> Vec<f64> {
        let expert_idx =
            ModalityEmbedding::modality_type_index(modality) % self.config.n_modalities;
        self.experts[expert_idx].forward(x)
    }

    /// Route a sequence of token vectors.
    pub fn route_sequence(&self, xs: &[Vec<f64>], modalities: &[MomModalityType]) -> Vec<Vec<f64>> {
        xs.iter()
            .zip(modalities.iter())
            .map(|(x, m)| self.route(x, m))
            .collect()
    }

    /// Compute load balance loss as variance of expert utilization.
    pub fn load_balance_loss(&self, xs: &[Vec<f64>], modalities: &[MomModalityType]) -> f64 {
        if xs.is_empty() {
            return 0.0;
        }
        let n = self.config.n_modalities;
        let mut counts = vec![0.0f64; n];
        for m in modalities {
            let idx = ModalityEmbedding::modality_type_index(m) % n;
            counts[idx] += 1.0;
        }
        let total = xs.len() as f64;
        let fracs: Vec<f64> = counts.iter().map(|&c| c / total).collect();
        let mean = fracs.iter().sum::<f64>() / n as f64;
        fracs.iter().map(|&f| (f - mean).powi(2)).sum::<f64>() / n as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §9 Multi-Modal Pretraining Objectives
// ─────────────────────────────────────────────────────────────────────────────

/// Weights and hyper-parameters for pretraining.
#[derive(Debug, Clone)]
pub struct PretrainingConfig {
    pub contrastive_weight: f64,
    pub generative_weight: f64,
    pub masked_weight: f64,
    pub mask_ratio: f64,
    pub temperature: f64,
}

impl Default for PretrainingConfig {
    fn default() -> Self {
        Self {
            contrastive_weight: 1.0,
            generative_weight: 1.0,
            masked_weight: 1.0,
            mask_ratio: 0.15,
            temperature: 0.07,
        }
    }
}

/// Combines contrastive, generative, and masked-token pretraining.
pub struct MultiModalPretrainer {
    pub config: PretrainingConfig,
    pub model: UnifiedTransformer,
}

impl MultiModalPretrainer {
    /// Create a new pretrainer.
    pub fn new(config: PretrainingConfig, model: UnifiedTransformer) -> Self {
        Self { config, model }
    }

    /// Symmetric InfoNCE contrastive loss between two sets of embeddings.
    pub fn contrastive_loss(&self, embeds_a: &[Vec<f64>], embeds_b: &[Vec<f64>]) -> f64 {
        if embeds_a.is_empty() || embeds_a.len() != embeds_b.len() {
            return 0.0;
        }
        let n = embeds_a.len();
        let t = self.config.temperature.max(1e-6);

        let mut loss = 0.0;
        for i in 0..n {
            let ai = &embeds_a[i];
            let pos_sim = cosine_similarity(ai, &embeds_b[i]) / t;
            let denom_log: f64 = {
                let log_sum = embeds_b
                    .iter()
                    .map(|bj| cosine_similarity(ai, bj) / t)
                    .fold(f64::NEG_INFINITY, |acc, s| {
                        let max_s = acc.max(s);
                        max_s + ((acc - max_s).exp() + (s - max_s).exp()).ln()
                    });
                log_sum
            };
            loss += denom_log - pos_sim;
        }
        (loss / n as f64).max(0.0)
    }

    /// Cross-entropy loss for next-token prediction.
    pub fn generative_loss(&self, seq: &ModalSequence, labels: &[usize]) -> f64 {
        if seq.total_tokens() == 0 || labels.is_empty() {
            return 0.0;
        }
        let logits = self.model.forward(seq);
        let n = logits.len().min(labels.len());
        let vocab = self.model.config.total_vocab;
        let mut loss = 0.0;
        for i in 0..n {
            let probs = softmax(&logits[i]);
            let label = labels[i] % vocab;
            let p = probs[label].max(1e-10);
            loss -= p.ln();
        }
        loss / n as f64
    }

    /// Masked token modeling loss.
    pub fn masked_loss(&self, seq: &ModalSequence) -> f64 {
        if seq.total_tokens() == 0 {
            return 0.0;
        }
        let n = seq.total_tokens();
        let n_masked = ((n as f64 * self.config.mask_ratio).ceil() as usize).max(1);

        let mut masked_seq = seq.clone();
        let step = n / n_masked;
        let mut masked_positions = Vec::with_capacity(n_masked);
        for i in 0..n_masked {
            let pos = i * step;
            if pos < masked_seq.tokens.len() {
                masked_positions.push((pos, masked_seq.tokens[pos].token_id));
                masked_seq.tokens[pos].token_id = 0;
            }
        }

        let logits = self.model.forward(&masked_seq);
        let vocab = self.model.config.total_vocab;
        let mut loss = 0.0;
        for (pos, label) in &masked_positions {
            if *pos < logits.len() {
                let probs = softmax(&logits[*pos]);
                let true_label = label % vocab;
                let p = probs[true_label].max(1e-10);
                loss -= p.ln();
            }
        }
        let count = masked_positions.len().max(1);
        (loss / count as f64).max(0.0)
    }

    /// Combined weighted loss.
    pub fn total_loss(
        &self,
        seq: &ModalSequence,
        paired_seqs: Option<(&ModalSequence, &ModalSequence)>,
    ) -> f64 {
        let labels: Vec<usize> = seq.tokens.iter().skip(1).map(|t| t.token_id).collect();
        let gen_loss = self.generative_loss(seq, &labels);
        let mask_loss = self.masked_loss(seq);

        let cont_loss = if let Some((sa, sb)) = paired_seqs {
            let ea = self.model.forward(sa);
            let eb = self.model.forward(sb);
            if ea.is_empty() || eb.is_empty() {
                0.0
            } else {
                self.contrastive_loss(&ea, &eb)
            }
        } else {
            0.0
        };

        self.config.contrastive_weight * cont_loss
            + self.config.generative_weight * gen_loss
            + self.config.masked_weight * mask_loss
    }
}
