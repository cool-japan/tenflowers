//! Generation extensions — speculative decoding, constrained generation, and configuration.

use super::{softmax, log_softmax};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::sync::Mutex;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// 4. Speculative Decoding
// ─────────────────────────────────────────────────────────────────────────────

/// Two-layer MLP draft model for speculative decoding (greedy token generation).
pub struct DraftModel {
    pub(crate) vocab_size: usize,
    w1: Vec<f64>, // [vocab_size × hidden_dim]
    w2: Vec<f64>, // [hidden_dim × vocab_size]
    hidden_dim: usize,
}

impl DraftModel {
    /// Create a draft model with random weights.
    pub fn new(vocab_size: usize, hidden_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (2.0 / vocab_size as f64).sqrt();
        let w1: Vec<f64> = (0..vocab_size * hidden_dim)
            .map(|_| {
                let u: f64 = rng.random::<f64>();
                (u * 2.0 - 1.0) * scale
            })
            .collect();
        let w2: Vec<f64> = (0..hidden_dim * vocab_size)
            .map(|_| {
                let u: f64 = rng.random::<f64>();
                (u * 2.0 - 1.0) * scale
            })
            .collect();
        Self {
            vocab_size,
            w1,
            w2,
            hidden_dim,
        }
    }

    /// Compute logits for a given input token id.
    pub(crate) fn logits_for_token(&self, token_id: usize) -> Vec<f64> {
        // One-hot encode the token.
        let mut one_hot = vec![0.0_f64; self.vocab_size];
        if token_id < self.vocab_size {
            one_hot[token_id] = 1.0;
        }

        // Hidden layer: h = ReLU(W1 @ x).
        let mut hidden = vec![0.0_f64; self.hidden_dim];
        for h in 0..self.hidden_dim {
            let mut acc = 0.0;
            for v in 0..self.vocab_size {
                acc += self.w1[v * self.hidden_dim + h] * one_hot[v];
            }
            hidden[h] = acc.max(0.0); // ReLU
        }

        // Output layer: logits = W2 @ h.
        let mut logits = vec![0.0_f64; self.vocab_size];
        for v in 0..self.vocab_size {
            let mut acc = 0.0;
            for h in 0..self.hidden_dim {
                acc += self.w2[h * self.vocab_size + v] * hidden[h];
            }
            logits[v] = acc;
        }
        logits
    }

    /// Greedily generate `n_draft` tokens from the last token of `prompt`.
    pub fn draft_tokens(&self, prompt: &[usize], n_draft: usize) -> Vec<usize> {
        if prompt.is_empty() || n_draft == 0 {
            return vec![];
        }
        let mut tokens = Vec::with_capacity(n_draft);
        let mut current = prompt.last().copied().unwrap_or(0);
        for _ in 0..n_draft {
            let logits = self.logits_for_token(current);
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            tokens.push(next);
            current = next;
        }
        tokens
    }

    /// Softmax probability of `target_token` given `context_token`.
    pub fn token_prob(&self, context_token: usize, target_token: usize) -> f64 {
        let logits = self.logits_for_token(context_token);
        let probs = softmax(&logits);
        probs.get(target_token).copied().unwrap_or(0.0)
    }
}

/// Speculative decoding verification: accepts draft tokens with prob min(1, p_t/p_d).
pub struct VerificationStep;

impl VerificationStep {
    /// Acceptance probability: min(1, p_target / p_draft).
    pub fn accept_prob(p_target: f64, p_draft: f64) -> f64 {
        if p_draft <= 0.0 {
            return 1.0;
        }
        (p_target / p_draft).min(1.0)
    }

    /// Accept a prefix of `draft_tokens`; return empty slice on first rejection.
    pub fn verify(
        &self,
        draft_tokens: &[usize],
        target_probs: &[f64],
        draft_probs: &[f64],
        rng: &mut StdRng,
    ) -> Vec<usize> {
        let n = draft_tokens
            .len()
            .min(target_probs.len())
            .min(draft_probs.len());
        let mut accepted = Vec::new();
        for i in 0..n {
            let ap = Self::accept_prob(target_probs[i], draft_probs[i]);
            let u: f64 = rng.random::<f64>();
            if u < ap {
                accepted.push(draft_tokens[i]);
            } else {
                break;
            }
        }
        accepted
    }
}

/// Speculative decoder: draft model proposes `n_draft` tokens; target verifies them.
pub struct SpeculativeDecoder {
    draft: DraftModel,
    target: DraftModel,
    n_draft: usize,
    verifier: VerificationStep,
}

impl SpeculativeDecoder {
    /// Compose draft model, target model, and draft budget.
    pub fn new(draft: DraftModel, target: DraftModel, n_draft: usize) -> Self {
        Self {
            draft,
            target,
            n_draft,
            verifier: VerificationStep,
        }
    }

    /// Decode up to `max_len` tokens from `prompt`.
    pub fn decode(&self, prompt: &[usize], max_len: usize, seed: u64) -> Vec<usize> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut output: Vec<usize> = prompt.to_vec();

        while output.len() < prompt.len() + max_len {
            let remaining = prompt.len() + max_len - output.len();
            let n = self.n_draft.min(remaining);
            let draft_tokens = self.draft.draft_tokens(&output, n);
            if draft_tokens.is_empty() {
                break;
            }
            let context = output.last().copied().unwrap_or(0);
            let target_probs: Vec<f64> = draft_tokens
                .iter()
                .map(|&t| self.target.token_prob(context, t))
                .collect();
            let draft_probs: Vec<f64> = draft_tokens
                .iter()
                .map(|&t| self.draft.token_prob(context, t))
                .collect();
            let accepted =
                self.verifier
                    .verify(&draft_tokens, &target_probs, &draft_probs, &mut rng);
            if accepted.is_empty() {
                // Rejection: fall back to a single target greedy step.
                let target_logits = self.target.logits_for_token(context);
                let fallback = target_logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                output.push(fallback);
            } else {
                output.extend_from_slice(&accepted);
            }
        }
        output
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Constrained Generation
// ─────────────────────────────────────────────────────────────────────────────

/// In-place logit processor applied before sampling.
pub trait LogitProcessor: Send + Sync {
    /// Modify `logits` given previously `generated` token ids.
    fn process(&self, logits: &mut Vec<f64>, generated: &[usize]);
}

/// Forces output to start with a given prefix sequence.
pub struct PrefixConstraint {
    prefix: Vec<usize>,
}

impl PrefixConstraint {
    /// Create from token id sequence.
    pub fn new(prefix: Vec<usize>) -> Self {
        Self { prefix }
    }
    /// Required prefix length.
    pub fn prefix_len(&self) -> usize {
        self.prefix.len()
    }
}

impl LogitProcessor for PrefixConstraint {
    fn process(&self, logits: &mut Vec<f64>, generated: &[usize]) {
        let step = generated.len();
        if step < self.prefix.len() {
            let required = self.prefix[step];
            for (i, l) in logits.iter_mut().enumerate() {
                if i != required {
                    *l = f64::NEG_INFINITY;
                }
            }
        }
    }
}

/// JSON structure tracking state for constrained generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JsonState {
    Start,
    Object,
    Array,
    StringValue,
    Done,
}

struct JsonSchemaState {
    stack: Vec<JsonState>,
    current: JsonState,
}

/// Lightweight JSON-structure state machine constraint (thread-safe via `Mutex`).
pub struct JsonSchemaConstraint {
    state: Mutex<JsonSchemaState>,
    open_brace_token: usize,
    close_brace_token: usize,
    open_bracket_token: usize,
    close_bracket_token: usize,
    quote_token: usize,
}

impl JsonSchemaConstraint {
    /// Create with vocabulary-specific token ids for `{`, `}`, `[`, `]`, `"`.
    pub fn new(
        open_brace_token: usize,
        close_brace_token: usize,
        open_bracket_token: usize,
        close_bracket_token: usize,
        quote_token: usize,
    ) -> Self {
        Self {
            state: Mutex::new(JsonSchemaState {
                stack: Vec::new(),
                current: JsonState::Start,
            }),
            open_brace_token,
            close_brace_token,
            open_bracket_token,
            close_bracket_token,
            quote_token,
        }
    }

    /// Current JSON state.
    pub fn state(&self) -> JsonState {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .current
            .clone()
    }

    /// Advance the state machine for `token_id`.
    pub fn advance(&self, token_id: usize) {
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let s = &mut *guard;
        match s.current {
            JsonState::Start | JsonState::Object | JsonState::Array => {
                if token_id == self.open_brace_token {
                    s.stack.push(s.current.clone());
                    s.current = JsonState::Object;
                } else if token_id == self.open_bracket_token {
                    s.stack.push(s.current.clone());
                    s.current = JsonState::Array;
                } else if token_id == self.close_brace_token && s.current == JsonState::Object {
                    s.current = s.stack.pop().unwrap_or(JsonState::Done);
                } else if token_id == self.close_bracket_token && s.current == JsonState::Array {
                    s.current = s.stack.pop().unwrap_or(JsonState::Done);
                } else if token_id == self.quote_token {
                    s.stack.push(s.current.clone());
                    s.current = JsonState::StringValue;
                } else if token_id == self.close_brace_token || token_id == self.close_bracket_token
                {
                    // closing bracket with nothing on stack → Done
                    s.current = JsonState::Done;
                }
            }
            JsonState::StringValue => {
                if token_id == self.quote_token {
                    s.current = s.stack.pop().unwrap_or(JsonState::Done);
                }
            }
            JsonState::Done => {}
        }
    }
}

impl LogitProcessor for JsonSchemaConstraint {
    fn process(&self, logits: &mut Vec<f64>, generated: &[usize]) {
        // Advance state for each generated token.
        for &tok in generated {
            self.advance(tok);
        }
        let current = self.state();
        // In Done state: discourage further token generation.
        if current == JsonState::Done {
            for l in logits.iter_mut() {
                *l = f64::NEG_INFINITY;
            }
        }
    }
}

/// Blocks specific token n-gram sequences from being generated.
pub struct BanWordConstraint {
    banned: Vec<Vec<usize>>,
}

impl BanWordConstraint {
    /// Create from a list of banned token id sequences.
    pub fn new(banned: Vec<Vec<usize>>) -> Self {
        Self { banned }
    }
    /// Add a banned sequence.
    pub fn add_sequence(&mut self, seq: Vec<usize>) {
        self.banned.push(seq);
    }
}

impl LogitProcessor for BanWordConstraint {
    fn process(&self, logits: &mut Vec<f64>, generated: &[usize]) {
        for banned_seq in &self.banned {
            if banned_seq.is_empty() {
                continue;
            }
            let n = banned_seq.len();
            // The last (n-1) tokens of generated must match the first (n-1) of banned.
            let prefix_len = n - 1;
            if generated.len() >= prefix_len {
                let tail = &generated[generated.len() - prefix_len..];
                if tail == &banned_seq[..prefix_len] {
                    // Block the final token of the banned sequence.
                    let block_tok = banned_seq[n - 1];
                    if block_tok < logits.len() {
                        logits[block_tok] = f64::NEG_INFINITY;
                    }
                }
            }
        }
    }
}

/// Configuration for text generation.
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub max_length: usize,
    pub repetition_penalty: f64,
    pub eos_token_id: Option<usize>,
    pub pad_token_id: Option<usize>,
}

impl GenerationConfig {
    /// Greedy decoding config.
    pub fn greedy(max_length: usize) -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            max_length,
            repetition_penalty: 1.0,
            eos_token_id: None,
            pad_token_id: None,
        }
    }

    /// Nucleus sampling config.
    pub fn nucleus(temperature: f64, top_p: f64, max_length: usize) -> Self {
        Self {
            temperature,
            top_k: 0,
            top_p,
            max_length,
            repetition_penalty: 1.0,
            eos_token_id: None,
            pad_token_id: None,
        }
    }

    /// Validate temperature, top_p, and repetition_penalty are in valid ranges.
    pub fn validate(&self) -> Result<()> {
        if self.temperature <= 0.0 {
            return Err(TensorError::invalid_argument(
                "temperature must be positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.top_p) {
            return Err(TensorError::invalid_argument(
                "top_p must be in [0.0, 1.0]".to_string(),
            ));
        }
        if self.repetition_penalty <= 0.0 {
            return Err(TensorError::invalid_argument(
                "repetition_penalty must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self::greedy(128)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (ported from original generation.rs)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::{
        BeamSearchDecoder, BeamHypothesis, BM25Retriever, ContrastiveDecoding, DocumentChunk,
        FewShotPrompter, HybridRetriever, InstructionTuningFormatter, KvCache, NucleusTopKSampler,
        PrefixLmDecoder, PromptTemplate, RagPipeline, RepetitionPenalty, TypicalSampler,
        VectorStore,
    };
    use scirs2_core::random::{rngs::StdRng, SeedableRng};
    use std::collections::HashMap;

    // ── Decoding ──────────────────────────────────────────────────────────────

    #[test]
    fn test_beam_search_construction() {
        let decoder = BeamSearchDecoder::new(4, 50, 0.6);
        assert_eq!(decoder.beam_width, 4);
        assert_eq!(decoder.max_length, 50);
    }

    #[test]
    fn test_beam_search_produces_tokens() {
        let vocab_size = 10_usize;
        let decoder = BeamSearchDecoder::new(2, 20, 0.6);
        let eos = 1_usize;
        let logits_fn = |token_id: usize| -> Vec<f64> {
            let mut l = vec![0.0_f64; vocab_size];
            l[(token_id + 1) % vocab_size] = 5.0;
            l[eos] = -3.0;
            l
        };
        let result = decoder.decode(logits_fn, 0, eos, 5);
        assert!(!result.is_empty());
        assert!(result.len() <= 6); // start token + up to 5 steps
    }

    #[test]
    fn test_beam_search_length_normalization() {
        let h1 = BeamHypothesis {
            tokens: vec![0, 1, 2, 3, 4],
            log_prob: -2.0,
        };
        let h2 = BeamHypothesis {
            tokens: vec![0, 1],
            log_prob: -1.0,
        };
        let s1 = h1.score(0.6);
        let s2 = h2.score(0.6);
        assert!(s1.is_finite());
        assert!(s2.is_finite());
    }

    #[test]
    fn test_nucleus_sampling_valid_token() {
        let sampler = NucleusTopKSampler::new();
        let logits = vec![1.0, 2.0, 0.5, -1.0, 3.0];
        let mut rng = StdRng::seed_from_u64(42);
        let token = sampler.sample(&logits, 0.9, 3, 1.0, &mut rng);
        assert!(token < logits.len());
    }

    #[test]
    fn test_top_k_filtering() {
        let sampler = NucleusTopKSampler::new();
        // With top_k=1 and very low temperature it should always pick the argmax.
        let logits = vec![0.0, 0.0, 10.0, 0.0, 0.0];
        let mut rng = StdRng::seed_from_u64(7);
        let token = sampler.sample(&logits, 1.0, 1, 0.01, &mut rng);
        assert_eq!(token, 2);
    }

    #[test]
    fn test_repetition_penalty_applied() {
        let logits = vec![1.0, 2.0, 3.0];
        let generated = vec![2_usize]; // penalise token 2
        let result = RepetitionPenalty::apply(&logits, &generated, 2.0);
        assert!((result[2] - 1.5).abs() < 1e-10);
        assert!((result[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_repetition_penalty_negative_logit() {
        let logits = vec![1.0, -2.0, 3.0];
        let generated = vec![1_usize]; // penalise token 1 (negative)
        let result = RepetitionPenalty::apply(&logits, &generated, 2.0);
        // Negative logit should be multiplied by penalty → more negative.
        assert!(result[1] < logits[1]);
    }

    #[test]
    fn test_contrastive_decoding() {
        let cd = ContrastiveDecoding::new(0.5);
        let expert = vec![1.0, 3.0, 2.0];
        let amateur = vec![2.0, 1.0, 2.0];
        let tok = cd.decode_contrastive(&expert, &amateur).expect("decode_contrastive failed");
        assert!(tok < expert.len());
    }

    #[test]
    fn test_contrastive_decoding_length_mismatch() {
        let cd = ContrastiveDecoding::new(0.5);
        let err = cd.decode_contrastive(&[1.0, 2.0], &[1.0]);
        assert!(err.is_err());
    }

    #[test]
    fn test_typical_sampler() {
        let sampler = TypicalSampler::new(0.95, 1.0);
        let logits = vec![0.5, 1.0, 2.0, 1.5, 0.1];
        let mut rng = StdRng::seed_from_u64(123);
        let tok = sampler.sample(&logits, &mut rng);
        assert!(tok < logits.len());
    }

    // ── Prefix LM ─────────────────────────────────────────────────────────────

    #[test]
    fn test_kv_cache_update() {
        let mut cache = KvCache::new(3, 8);
        cache.update(0, vec![0.1; 8], vec![0.2; 8]).expect("cache update failed");
        assert_eq!(cache.sequence_length(), 1);
        assert_eq!(cache.num_layers(), 3);
    }

    #[test]
    fn test_kv_cache_out_of_bounds() {
        let mut cache = KvCache::new(2, 4);
        let err = cache.update(5, vec![0.0; 4], vec![0.0; 4]);
        assert!(err.is_err());
    }

    #[test]
    fn test_kv_cache_clear() {
        let mut cache = KvCache::new(1, 4);
        cache.update(0, vec![1.0; 4], vec![1.0; 4]).expect("cache update failed");
        cache.clear();
        assert_eq!(cache.sequence_length(), 0);
    }

    #[test]
    fn test_prefix_lm_decoder_forward() {
        let decoder = PrefixLmDecoder::new(16, 8, 2, 0);
        let mut cache = KvCache::new(1, 4);
        let logits = decoder.forward_cached(3, &mut cache).expect("forward_cached failed");
        assert_eq!(logits.len(), 16);
        assert_eq!(cache.sequence_length(), 1);
    }

    #[test]
    fn test_prefix_lm_decoder_invalid_token() {
        let decoder = PrefixLmDecoder::new(8, 4, 1, 0);
        let mut cache = KvCache::new(1, 4);
        let err = decoder.forward_cached(100, &mut cache);
        assert!(err.is_err());
    }

    #[test]
    fn test_prompt_template_format() {
        let tmpl = PromptTemplate::new("Hello, {name}! You asked: {question}");
        let mut fields = HashMap::new();
        fields.insert("name", "Alice");
        fields.insert("question", "how are you?");
        let out = tmpl.format(&fields);
        assert!(out.contains("Alice"));
        assert!(out.contains("how are you?"));
        assert!(!out.contains("{name}"));
    }

    #[test]
    fn test_prompt_template_missing_key() {
        let tmpl = PromptTemplate::new("Hi {x} and {y}");
        let mut fields = HashMap::new();
        fields.insert("x", "foo");
        let out = tmpl.format(&fields);
        assert!(out.contains("foo"));
        assert!(out.contains("{y}"));
    }

    #[test]
    fn test_few_shot_prompter() {
        let mut prompter = FewShotPrompter::new("\n");
        prompter.add_example("cat", "animal");
        let prompt = prompter.build_prompt(&[("dog", "animal")], "fish?");
        assert!(prompt.contains("cat"));
        assert!(prompt.contains("animal"));
        assert!(prompt.contains("fish?"));
        assert_eq!(prompter.num_examples(), 1);
    }

    #[test]
    fn test_few_shot_prompter_empty() {
        let prompter = FewShotPrompter::new("\n---\n");
        let prompt = prompter.build_prompt(&[], "what is 2+2?");
        assert!(prompt.contains("2+2"));
    }

    #[test]
    fn test_instruction_tuning_format() {
        let fmt = InstructionTuningFormatter::alpaca();
        let out = fmt.format("Be helpful.", "What is Rust?", "");
        assert!(out.contains("Be helpful."));
        assert!(out.contains("What is Rust?"));
        assert!(out.contains("### System:"));
    }

    #[test]
    fn test_instruction_tuning_llama2() {
        let fmt = InstructionTuningFormatter::llama2();
        let out = fmt.format("System msg", "User msg", "Assistant start");
        assert!(out.contains("<<SYS>>"));
        assert!(out.contains("[INST]"));
        assert!(out.contains("Assistant start"));
    }

    // ── RAG ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_document_chunk_construction() {
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), "wiki".to_string());
        let chunk = DocumentChunk::new("Some text.", vec![1.0, 0.0], meta);
        assert_eq!(chunk.text, "Some text.");
        assert_eq!(chunk.embedding.len(), 2);
        assert_eq!(chunk.metadata["source"], "wiki");
    }

    #[test]
    fn test_vector_store_add_search() {
        let mut store = VectorStore::new(3);
        let c1 = DocumentChunk::new("doc1", vec![1.0, 0.0, 0.0], HashMap::new());
        let c2 = DocumentChunk::new("doc2", vec![0.0, 1.0, 0.0], HashMap::new());
        let c3 = DocumentChunk::new("doc3", vec![0.0, 0.0, 1.0], HashMap::new());
        store.add(c1);
        store.add(c2);
        store.add(c3);
        assert_eq!(store.len(), 3);
        let results = store.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "doc1");
    }

    #[test]
    fn test_vector_store_empty_search() {
        let store = VectorStore::new(4);
        let results = store.search(&[1.0, 0.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_retriever_build() {
        let docs = vec![
            vec!["hello".to_string(), "world".to_string()],
            vec!["foo".to_string(), "bar".to_string()],
        ];
        let retriever = BM25Retriever::build(docs);
        assert_eq!(retriever.num_docs(), 2);
    }

    #[test]
    fn test_bm25_scoring() {
        let docs = vec![
            vec![
                "rust".to_string(),
                "programming".to_string(),
                "rust".to_string(),
            ],
            vec!["python".to_string(), "programming".to_string()],
        ];
        let retriever = BM25Retriever::build(docs);
        let results = retriever.search(&["rust"], 2);
        assert!(!results.is_empty());
        // Document 0 mentions "rust" twice → should rank first.
        assert_eq!(results[0], 0);
    }

    #[test]
    fn test_bm25_retriever_no_match() {
        let docs = vec![vec!["alpha".to_string()]];
        let retriever = BM25Retriever::build(docs);
        let results = retriever.search(&["zzz"], 1);
        // Still returns indices (score = 0 for all).
        assert!(results.len() <= 1);
    }

    #[test]
    fn test_hybrid_retriever() {
        let bm25_docs = vec![
            vec!["machine".to_string(), "learning".to_string()],
            vec!["deep".to_string(), "learning".to_string()],
        ];
        let bm25 = BM25Retriever::build(bm25_docs);
        let mut vstore = VectorStore::new(2);
        vstore.add(DocumentChunk::new("doc A", vec![1.0, 0.0], HashMap::new()));
        vstore.add(DocumentChunk::new("doc B", vec![0.5, 0.5], HashMap::new()));

        let hybrid = HybridRetriever::new(bm25, vstore, 0.5);
        let results = hybrid.retrieve(&["machine"], &[1.0, 0.0], 2);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_rag_pipeline() {
        let bm25_docs = vec![
            vec!["neural".to_string(), "networks".to_string()],
            vec!["gradient".to_string(), "descent".to_string()],
        ];
        let bm25 = BM25Retriever::build(bm25_docs);
        let mut vstore = VectorStore::new(2);
        vstore.add(DocumentChunk::new(
            "Neural networks are universal function approximators.",
            vec![1.0, 0.0],
            HashMap::new(),
        ));
        vstore.add(DocumentChunk::new(
            "Gradient descent minimises the loss.",
            vec![0.0, 1.0],
            HashMap::new(),
        ));
        let hybrid = HybridRetriever::new(bm25, vstore, 0.5);
        let rag = RagPipeline::new(hybrid, 2);
        let augmented = rag.augment("What is a neural network?", &["neural"], &[1.0, 0.0]);
        // Should contain either the context header or the query text.
        assert!(
            augmented.contains("Context:")
                || augmented.contains("neural")
                || augmented.contains("What")
        );
    }

    // ── Speculative ───────────────────────────────────────────────────────────

    #[test]
    fn test_draft_model_tokens() {
        let draft = DraftModel::new(10, 8, 42);
        let tokens = draft.draft_tokens(&[3], 4);
        assert_eq!(tokens.len(), 4);
        for &t in &tokens {
            assert!(t < 10);
        }
    }

    #[test]
    fn test_draft_model_empty_prompt() {
        let draft = DraftModel::new(10, 8, 0);
        let tokens = draft.draft_tokens(&[], 3);
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_draft_model_token_prob_sums_to_one() {
        let draft = DraftModel::new(8, 4, 1);
        let total: f64 = (0..8).map(|t| draft.token_prob(0, t)).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_verification_acceptance() {
        let vs = VerificationStep;
        let mut rng = StdRng::seed_from_u64(0);
        // p_target >= p_draft → accept probability = 1 → always accepted.
        let accepted = vs.verify(&[1, 2], &[0.5, 0.3], &[0.2, 0.1], &mut rng);
        assert_eq!(accepted, vec![1, 2]);
    }

    #[test]
    fn test_verification_rejection() {
        // Draft prob much higher than target → high rejection probability.
        let vs = VerificationStep;
        let mut rng = StdRng::seed_from_u64(99);
        let mut rejected = false;
        for _ in 0..100 {
            let accepted = vs.verify(&[5], &[0.001], &[0.999], &mut rng);
            if accepted.is_empty() {
                rejected = true;
                break;
            }
        }
        assert!(rejected);
    }

    #[test]
    fn test_speculative_decoder_output() {
        let draft = DraftModel::new(12, 6, 10);
        let target = DraftModel::new(12, 6, 20);
        let decoder = SpeculativeDecoder::new(draft, target, 3);
        let prompt = vec![0_usize, 1];
        let output = decoder.decode(&prompt, 5, 42);
        assert!(output.len() >= prompt.len());
        for &t in &output {
            assert!(t < 12);
        }
    }

    // ── Constrained ───────────────────────────────────────────────────────────

    #[test]
    fn test_prefix_constraint() {
        let constraint = PrefixConstraint::new(vec![3, 7]);
        let mut logits = vec![1.0_f64; 10];
        constraint.process(&mut logits, &[]);
        // Only token 3 should be non-NEG_INFINITY.
        assert!(logits[3].is_finite());
        for (i, &l) in logits.iter().enumerate() {
            if i != 3 {
                assert_eq!(l, f64::NEG_INFINITY);
            }
        }
    }

    #[test]
    fn test_prefix_constraint_second_step() {
        let constraint = PrefixConstraint::new(vec![3, 7]);
        let mut logits = vec![1.0_f64; 10];
        constraint.process(&mut logits, &[3]);
        assert!(logits[7].is_finite());
        for (i, &l) in logits.iter().enumerate() {
            if i != 7 {
                assert_eq!(l, f64::NEG_INFINITY);
            }
        }
    }

    #[test]
    fn test_prefix_constraint_after_prefix_completed() {
        let constraint = PrefixConstraint::new(vec![3]);
        let mut logits = vec![1.0_f64; 5];
        constraint.process(&mut logits, &[3]);
        for &l in &logits {
            assert_eq!(l, 1.0);
        }
    }

    #[test]
    fn test_ban_word_constraint() {
        let constraint = BanWordConstraint::new(vec![vec![2_usize, 5]]);
        let mut logits = vec![1.0_f64; 10];
        constraint.process(&mut logits, &[2]);
        assert_eq!(logits[5], f64::NEG_INFINITY);
        assert_eq!(logits[0], 1.0);
    }

    #[test]
    fn test_ban_word_constraint_no_match() {
        let constraint = BanWordConstraint::new(vec![vec![2_usize, 5]]);
        let mut logits = vec![1.0_f64; 10];
        constraint.process(&mut logits, &[9]);
        for &l in &logits {
            assert_eq!(l, 1.0);
        }
    }

    #[test]
    fn test_generation_config() {
        let config = GenerationConfig::nucleus(0.8, 0.95, 256);
        assert!(config.validate().is_ok());
        assert_eq!(config.temperature, 0.8);
        assert_eq!(config.top_p, 0.95);
        assert_eq!(config.max_length, 256);
    }

    #[test]
    fn test_generation_config_invalid_temperature() {
        let config = GenerationConfig {
            temperature: -1.0,
            ..GenerationConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_generation_config_invalid_top_p() {
        let config = GenerationConfig {
            top_p: 1.5,
            ..GenerationConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_logit_processor_trait() {
        struct ZeroProcessor;
        impl LogitProcessor for ZeroProcessor {
            fn process(&self, logits: &mut Vec<f64>, _generated: &[usize]) {
                for l in logits.iter_mut() {
                    *l = 0.0;
                }
            }
        }
        let processor = ZeroProcessor;
        let mut logits = vec![1.0, 2.0, 3.0];
        processor.process(&mut logits, &[]);
        assert!(logits.iter().all(|&l| l == 0.0));
    }

    #[test]
    fn test_json_schema_constraint() {
        // Token ids: { = 1, } = 2, [ = 3, ] = 4, " = 5
        let constraint = JsonSchemaConstraint::new(1, 2, 3, 4, 5);
        assert_eq!(constraint.state(), JsonState::Start);

        constraint.advance(1); // open brace → Object
        assert_eq!(constraint.state(), JsonState::Object);

        constraint.advance(5); // quote → StringValue
        assert_eq!(constraint.state(), JsonState::StringValue);

        constraint.advance(5); // close quote → back to Object
        assert_eq!(constraint.state(), JsonState::Object);

        constraint.advance(2); // close brace → back to Start (stack pop)
        assert_eq!(constraint.state(), JsonState::Start);
    }

    #[test]
    fn test_json_schema_done_state_blocks_logits() {
        let constraint = JsonSchemaConstraint::new(1, 2, 3, 4, 5);
        // Advance to Done by closing brace with empty stack.
        constraint.advance(2);
        assert_eq!(constraint.state(), JsonState::Done);
        let mut logits = vec![1.0_f64; 5];
        constraint.process(&mut logits, &[]);
        for &l in &logits {
            assert_eq!(l, f64::NEG_INFINITY);
        }
    }

    #[test]
    fn test_kv_cache_head_dim() {
        let cache = KvCache::new(4, 16);
        assert_eq!(cache.head_dim(), 16);
        assert_eq!(cache.num_layers(), 4);
    }

    #[test]
    fn test_vector_store_top_k() {
        let mut store = VectorStore::new(3);
        for i in 0..5 {
            let emb = vec![(i as f64) * 0.1, 0.0, 0.0];
            store.add(DocumentChunk::new(format!("doc{}", i), emb, HashMap::new()));
        }
        let results = store.search(&[0.4, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_instruction_tuning_custom() {
        let fmt = InstructionTuningFormatter::custom("SYS:", "\nUSER:", "\nASS:");
        let out = fmt.format("be nice", "hello", "hi");
        assert!(out.starts_with("SYS:be nice"));
        assert!(out.contains("hi"));
    }

    #[test]
    fn test_softmax_sums_to_one() {
        use crate::generation::softmax as gen_softmax;
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = gen_softmax(&logits);
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        use crate::generation::cosine_similarity as gen_cos;
        let v = vec![1.0, 2.0, 3.0];
        assert!((gen_cos(&v, &v) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        use crate::generation::cosine_similarity as gen_cos;
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(gen_cos(&a, &b).abs() < 1e-10);
    }
}
