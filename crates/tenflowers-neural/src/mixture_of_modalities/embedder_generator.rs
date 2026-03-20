//! AnyModal embedder (cross-modal projection) and cross-modal generator.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

use super::embedding_transformer::UnifiedTransformer;
use super::types::{
    gelu, mat_vec, softmax, vec_add, xavier_init, ModalSequence, ModalToken, MomModalityType,
};

// ─────────────────────────────────────────────────────────────────────────────
// §6 AnyModal Embedder
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the any-modal embedder.
#[derive(Debug, Clone)]
pub struct AnyModalEmbedderConfig {
    /// Input feature dimensions per modality.
    pub source_dims: Vec<(MomModalityType, usize)>,
    /// Shared embedding dimension.
    pub d_model: usize,
}

/// Linear projector from modality feature space to shared embedding space.
pub struct ModalProjector {
    /// Weight matrix: `[d_model][src_dim]`.
    pub w: Vec<Vec<f64>>,
    /// Bias: `[d_model]`.
    pub b: Vec<f64>,
}

impl ModalProjector {
    pub(super) fn new(d_model: usize, src_dim: usize, rng: &mut StdRng) -> Self {
        Self {
            w: xavier_init(rng, d_model, src_dim),
            b: (0..d_model)
                .map(|_| rng.random::<f64>() * 0.02 - 0.01)
                .collect(),
        }
    }

    pub(super) fn project(&self, x: &[f64]) -> Vec<f64> {
        let out = vec_add(&mat_vec(&self.w, x), &self.b);
        out.iter().map(|&v| gelu(v)).collect()
    }
}

/// Projects any modality's raw features into a shared d_model-dimensional space.
pub struct AnyModalEmbedder {
    pub config: AnyModalEmbedderConfig,
    pub projectors: Vec<ModalProjector>,
}

impl AnyModalEmbedder {
    /// Create a new embedder with one projector per source modality.
    pub fn new(config: AnyModalEmbedderConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(5678);
        let projectors: Vec<ModalProjector> = config
            .source_dims
            .iter()
            .map(|(_, src_dim)| ModalProjector::new(config.d_model, *src_dim, &mut rng))
            .collect();
        Self { config, projectors }
    }

    fn projector_index(&self, modality: &MomModalityType) -> Option<usize> {
        self.config
            .source_dims
            .iter()
            .position(|(m, _)| m == modality)
    }

    /// Project a single feature vector for the given modality.
    pub fn project(&self, data: &[f64], modality: &MomModalityType) -> Vec<f64> {
        match self.projector_index(modality) {
            Some(idx) => self.projectors[idx].project(data),
            None => vec![0.0; self.config.d_model],
        }
    }

    /// Project a sequence of feature vectors.
    pub fn project_sequence(&self, seq: &[Vec<f64>], modality: &MomModalityType) -> Vec<Vec<f64>> {
        seq.iter().map(|x| self.project(x, modality)).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7 Cross-Modal Generation
// ─────────────────────────────────────────────────────────────────────────────

/// Which cross-modal generation direction to use.
#[derive(Debug, Clone)]
pub enum GenerationMode {
    TextFromImage,
    TextFromAudio,
    ImageFromText,
    AudioFromText,
    AnyToAny(MomModalityType, MomModalityType),
}

/// Configuration for the cross-modal generator.
#[derive(Debug, Clone)]
pub struct CrossModalGeneratorConfig {
    pub mode: GenerationMode,
    pub max_gen_tokens: usize,
    pub temperature: f64,
    pub top_k: usize,
}

/// Generates tokens in a target modality conditioned on a source modality sequence.
pub struct CrossModalGenerator {
    pub config: CrossModalGeneratorConfig,
    pub model: UnifiedTransformer,
}

impl CrossModalGenerator {
    /// Create a new generator.
    pub fn new(config: CrossModalGeneratorConfig, model: UnifiedTransformer) -> Self {
        Self { config, model }
    }

    /// Sample the next token from logits using temperature-scaled top-k sampling.
    pub fn sample_next(&self, logits: &[f64]) -> usize {
        if logits.is_empty() {
            return 0;
        }
        let temp = self.config.temperature.max(1e-6);
        let scaled: Vec<f64> = logits.iter().map(|&l| l / temp).collect();

        let k = self.config.top_k.min(logits.len()).max(1);
        let mut indexed: Vec<(usize, f64)> = scaled.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);

        let probs = softmax(&indexed.iter().map(|(_, v)| *v).collect::<Vec<_>>());

        let mut rng = StdRng::seed_from_u64(42);
        let r: f64 = rng.random();
        let mut cumsum = 0.0;
        for (i, p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return indexed[i].0;
            }
        }
        indexed[0].0
    }

    /// Auto-regressive generation: returns generated token IDs.
    pub fn generate(&self, source: &ModalSequence) -> Vec<usize> {
        if source.total_tokens() == 0 {
            return Vec::new();
        }
        let mut generated = Vec::with_capacity(self.config.max_gen_tokens);
        let mut current_seq = source.clone();

        for _ in 0..self.config.max_gen_tokens {
            let logits_all = self.model.forward(&current_seq);
            let last_logits = match logits_all.last() {
                Some(l) => l,
                None => break,
            };
            let next_id = self.sample_next(last_logits);
            generated.push(next_id);

            let target_modality = match &self.config.mode {
                GenerationMode::TextFromImage => MomModalityType::Text,
                GenerationMode::TextFromAudio => MomModalityType::Text,
                GenerationMode::ImageFromText => MomModalityType::Image,
                GenerationMode::AudioFromText => MomModalityType::Audio,
                GenerationMode::AnyToAny(_, target) => target.clone(),
            };

            let pos = current_seq.total_tokens();
            let new_tok = ModalToken {
                modality: target_modality,
                token_id: next_id,
                position: pos,
            };
            current_seq.append(vec![new_tok]);
        }
        generated
    }
}
