//! Code/Text Generation & Sequence Model Utilities — core module.
//!
//! Provides beam search, nucleus sampling, contrastive decoding, typical sampling,
//! KV-cache prefix LM, RAG pipeline, speculative decoding, and constrained generation.

pub mod extensions;
pub use extensions::*;
pub mod advanced;
pub use advanced::*;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;
use std::sync::Mutex;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically-stable softmax over a slice.
pub(crate) fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return vec![];
    }
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / logits.len() as f64; logits.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

/// Log-softmax over a slice.
pub(crate) fn log_softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return vec![];
    }
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let log_sum = sum.ln();
    logits.iter().map(|&x| x - max_val - log_sum).collect()
}

/// Cosine similarity between two equal-length slices.
pub(crate) fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
    let norm_a: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Advanced Decoding Strategies
// ─────────────────────────────────────────────────────────────────────────────

/// A single hypothesis in beam search.
#[derive(Clone, Debug)]
pub(crate) struct BeamHypothesis {
    /// Token ids generated so far.
    pub tokens: Vec<usize>,
    /// Accumulated log-probability.
    pub log_prob: f64,
}

impl BeamHypothesis {
    pub fn new(start_token: usize) -> Self {
        Self {
            tokens: vec![start_token],
            log_prob: 0.0,
        }
    }

    /// Length-normalised score. α is the length-normalization exponent (Wu et al. 2016).
    pub fn score(&self, alpha: f64) -> f64 {
        let len = self.tokens.len() as f64;
        let lp = ((5.0 + len) / 6.0).powf(alpha);
        self.log_prob / lp
    }
}

/// Beam search decoder with Wu-et-al. (2016) length normalization.
pub struct BeamSearchDecoder {
    pub beam_width: usize,
    pub max_length: usize,
    /// Length normalisation exponent α (0.6 is standard).
    pub alpha: f64,
}

impl BeamSearchDecoder {
    /// Create a new beam search decoder (`beam_width`, `max_length`, `alpha`).
    pub fn new(beam_width: usize, max_length: usize, alpha: f64) -> Self {
        Self {
            beam_width: beam_width.max(1),
            max_length,
            alpha,
        }
    }

    /// Decode using `logits_fn(token_id) -> Vec<f64>`. Returns the best token sequence.
    pub fn decode<F>(
        &self,
        mut logits_fn: F,
        start_token: usize,
        eos_token: usize,
        max_len: usize,
    ) -> Vec<usize>
    where
        F: FnMut(usize) -> Vec<f64>,
    {
        let cap = max_len.min(self.max_length);
        let mut beams = vec![BeamHypothesis::new(start_token)];
        let mut finished: Vec<BeamHypothesis> = Vec::new();

        for _step in 0..cap {
            if beams.is_empty() {
                break;
            }
            let mut candidates: Vec<BeamHypothesis> = Vec::new();

            for beam in &beams {
                let last_token = beam.tokens.last().copied().unwrap_or(start_token);
                let logits = logits_fn(last_token);
                if logits.is_empty() {
                    continue;
                }
                let log_probs = log_softmax(&logits);
                let mut indexed: Vec<(usize, f64)> =
                    log_probs.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for &(tok_id, lp) in indexed.iter().take(self.beam_width) {
                    let mut new_beam = beam.clone();
                    new_beam.tokens.push(tok_id);
                    new_beam.log_prob += lp;
                    if tok_id == eos_token {
                        finished.push(new_beam);
                    } else {
                        candidates.push(new_beam);
                    }
                }
            }

            // Keep only top-K by normalised score.
            candidates.sort_by(|a, b| {
                b.score(self.alpha)
                    .partial_cmp(&a.score(self.alpha))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(self.beam_width);
            beams = candidates;
        }
        finished.extend(beams);
        finished.sort_by(|a, b| {
            b.score(self.alpha)
                .partial_cmp(&a.score(self.alpha))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        finished
            .into_iter()
            .next()
            .map(|h| h.tokens)
            .unwrap_or_else(|| vec![start_token])
    }
}

/// Combined top-p (nucleus) + top-k sampler with temperature scaling.
pub struct NucleusTopKSampler;

impl NucleusTopKSampler {
    /// Create a new sampler.
    pub fn new() -> Self {
        Self
    }

    /// Sample a token using top-p + top-k filtering and temperature scaling.
    pub fn sample(
        &self,
        logits: &[f64],
        top_p: f64,
        top_k: usize,
        temperature: f64,
        rng: &mut StdRng,
    ) -> usize {
        if logits.is_empty() {
            return 0;
        }
        let temp = temperature.max(1e-8);

        let scaled: Vec<f64> = logits.iter().map(|&x| x / temp).collect();
        let mut indexed: Vec<(usize, f64)> = scaled.iter().copied().enumerate().collect();
        if top_k > 0 && top_k < indexed.len() {
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(top_k);
        }
        let logit_vals: Vec<f64> = indexed.iter().map(|&(_, l)| l).collect();
        let probs = softmax(&logit_vals);
        let mut prob_indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
        prob_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cum_prob = 0.0_f64;
        let mut keep_len = prob_indexed.len();
        for (rank, &(_, p)) in prob_indexed.iter().enumerate() {
            cum_prob += p;
            if cum_prob >= top_p {
                keep_len = rank + 1;
                break;
            }
        }
        let nucleus_ids: std::collections::HashSet<usize> = prob_indexed
            .iter()
            .take(keep_len)
            .map(|&(i, _)| i)
            .collect();
        let final_candidates: Vec<(usize, f64)> = indexed
            .iter()
            .enumerate()
            .filter(|(i, _)| nucleus_ids.contains(i))
            .map(|(_, &(orig_id, logit))| (orig_id, logit))
            .collect();
        if final_candidates.is_empty() {
            return indexed.first().map(|&(id, _)| id).unwrap_or(0);
        }
        let final_probs = softmax(&final_candidates.iter().map(|&(_, l)| l).collect::<Vec<_>>());
        let u: f64 = rng.random::<f64>();
        let mut cum = 0.0;
        for (i, &p) in final_probs.iter().enumerate() {
            cum += p;
            if u < cum {
                return final_candidates[i].0;
            }
        }
        final_candidates.last().map(|&(id, _)| id).unwrap_or(0)
    }
}

impl Default for NucleusTopKSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Contrastive Decoding (CD-α): expert - α·amateur logit subtraction (Li et al. 2022).
pub struct ContrastiveDecoding {
    alpha: f64,
}

impl ContrastiveDecoding {
    /// Create with mixing coefficient `alpha`.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    /// Subtract `alpha * log_softmax(amateur)` from `log_softmax(expert)` and return argmax.
    pub fn decode_contrastive(
        &self,
        expert_logits: &[f64],
        amateur_logits: &[f64],
    ) -> Result<usize> {
        if expert_logits.len() != amateur_logits.len() {
            return Err(TensorError::invalid_argument(
                "expert and amateur logit vectors must have equal length".to_string(),
            ));
        }
        if expert_logits.is_empty() {
            return Err(TensorError::invalid_argument(
                "logit vectors must not be empty".to_string(),
            ));
        }
        let expert_lp = log_softmax(expert_logits);
        let amateur_lp = log_softmax(amateur_logits);

        let adjusted: Vec<f64> = expert_lp
            .iter()
            .zip(amateur_lp.iter())
            .map(|(&e, &a)| e - self.alpha * a)
            .collect();
        let best = adjusted
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(best)
    }
}

/// Typical sampling (Meister et al. 2022): filters tokens by |I(x) - H| criterion.
pub struct TypicalSampler {
    typical_p: f64,
    temperature: f64,
}

impl TypicalSampler {
    /// Create with `typical_p` mass threshold and `temperature`.
    pub fn new(typical_p: f64, temperature: f64) -> Self {
        Self {
            typical_p: typical_p.clamp(0.0, 1.0),
            temperature: temperature.max(1e-8),
        }
    }

    /// Sample a token using typical sampling.
    pub fn sample(&self, logits: &[f64], rng: &mut StdRng) -> usize {
        if logits.is_empty() {
            return 0;
        }
        let probs = softmax(
            &logits
                .iter()
                .map(|&x| x / self.temperature)
                .collect::<Vec<_>>(),
        );
        // H = Σ p * -ln(p)
        let entropy: f64 = probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        let mut scored: Vec<(usize, f64, f64)> = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(i, &p)| (i, p, (-p.ln() - entropy).abs()))
            .collect();
        scored.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        let mut cum = 0.0_f64;
        let mut candidates: Vec<(usize, f64)> = Vec::new();
        for &(i, p, _) in &scored {
            candidates.push((i, p));
            cum += p;
            if cum >= self.typical_p {
                break;
            }
        }
        if candidates.is_empty() {
            return 0;
        }
        let total: f64 = candidates.iter().map(|&(_, p)| p).sum();
        let u: f64 = rng.random::<f64>();
        let mut acc = 0.0;
        for &(tok, p) in &candidates {
            acc += p / total;
            if u < acc {
                return tok;
            }
        }
        candidates.last().map(|&(i, _)| i).unwrap_or(0)
    }
}

/// Repetition penalty: divides positive logits / multiplies negative logits of seen tokens.
pub struct RepetitionPenalty;

impl RepetitionPenalty {
    /// Apply `penalty` to logits of tokens in `generated_ids`. Returns modified logits.
    pub fn apply(logits: &[f64], generated_ids: &[usize], penalty: f64) -> Vec<f64> {
        let mut out = logits.to_vec();
        let p = penalty.max(1e-8);
        for &id in generated_ids {
            if id < out.len() {
                if out[id] > 0.0 {
                    out[id] /= p;
                } else {
                    out[id] *= p;
                }
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Prefix Language Model / Prompting
// ─────────────────────────────────────────────────────────────────────────────

/// Key-value cache for autoregressive transformer decoding.
pub struct KvCache {
    /// Per-layer list of (key, value) pairs, one per decoded position.
    pub layers: Vec<Vec<(Vec<f64>, Vec<f64>)>>,
    head_dim: usize,
}

impl KvCache {
    /// Create an empty cache with `num_layers` layers and `head_dim`-dimensional K/V.
    pub fn new(num_layers: usize, head_dim: usize) -> Self {
        Self {
            layers: vec![Vec::new(); num_layers],
            head_dim,
        }
    }

    /// Append `(key, value)` to the given layer.
    pub fn update(&mut self, layer: usize, key: Vec<f64>, value: Vec<f64>) -> Result<()> {
        if layer >= self.layers.len() {
            return Err(TensorError::invalid_argument(
                "layer index out of bounds for KvCache".to_string(),
            ));
        }
        self.layers[layer].push((key, value));
        Ok(())
    }

    /// Retrieve all (key, value) pairs for a layer.
    pub fn get_layer(&self, layer: usize) -> Option<&Vec<(Vec<f64>, Vec<f64>)>> {
        self.layers.get(layer)
    }
    /// Total cached positions (first layer length).
    pub fn sequence_length(&self) -> usize {
        self.layers.first().map(|l| l.len()).unwrap_or(0)
    }
    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
    /// K/V head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
    /// Clear all cached entries.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }
}

/// Causal transformer decoder stub backed by a KV cache.
pub struct PrefixLmDecoder {
    vocab_size: usize,
    model_dim: usize,
    num_heads: usize,
    embeddings: Vec<f64>,
    out_proj: Vec<f64>,
}

impl PrefixLmDecoder {
    /// Create a decoder with random weights (seed for reproducibility).
    pub fn new(vocab_size: usize, model_dim: usize, num_heads: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (1.0 / model_dim as f64).sqrt();
        let embed_size = vocab_size * model_dim;
        let embeddings: Vec<f64> = (0..embed_size)
            .map(|_| {
                let u: f64 = rng.random::<f64>();
                (u * 2.0 - 1.0) * scale
            })
            .collect();
        let out_size = model_dim * vocab_size;
        let out_proj: Vec<f64> = (0..out_size)
            .map(|_| {
                let u: f64 = rng.random::<f64>();
                (u * 2.0 - 1.0) * scale
            })
            .collect();
        Self {
            vocab_size,
            model_dim,
            num_heads,
            embeddings,
            out_proj,
        }
    }

    /// Single decoding step with KV cache update. Returns vocabulary logits.
    pub fn forward_cached(&self, token_id: usize, kv_cache: &mut KvCache) -> Result<Vec<f64>> {
        if token_id >= self.vocab_size {
            return Err(TensorError::invalid_argument(
                "token_id out of vocabulary".to_string(),
            ));
        }
        // Look up embedding.
        let emb_start = token_id * self.model_dim;
        let hidden: Vec<f64> = self.embeddings[emb_start..emb_start + self.model_dim].to_vec();

        // Simulate a KV update (layer 0).
        if kv_cache.num_layers() > 0 {
            let head_dim = self.model_dim / self.num_heads.max(1);
            let key = hidden[..head_dim.min(hidden.len())].to_vec();
            let val = hidden[..head_dim.min(hidden.len())].to_vec();
            kv_cache.update(0, key, val)?;
        }

        // Project to vocabulary via out_proj.
        let mut logits = vec![0.0_f64; self.vocab_size];
        for v in 0..self.vocab_size {
            let mut acc = 0.0_f64;
            for d in 0..self.model_dim {
                let proj_idx = d * self.vocab_size + v;
                if proj_idx < self.out_proj.len() {
                    acc += hidden[d] * self.out_proj[proj_idx];
                }
            }
            logits[v] = acc;
        }
        Ok(logits)
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// Prompt template with `{key}` placeholder substitution.
pub struct PromptTemplate {
    template: String,
}

impl PromptTemplate {
    /// Create from a string with `{key}` placeholders.
    pub fn new(template: impl Into<String>) -> Self {
        Self {
            template: template.into(),
        }
    }

    /// Substitute all `{key}` placeholders from `fields`.
    pub fn format(&self, fields: &HashMap<&str, &str>) -> String {
        let mut result = self.template.clone();
        for (&key, &value) in fields {
            let placeholder = format!("{{{}}}", key);
            result = result.replace(&placeholder, value);
        }
        result
    }

    /// Raw template string.
    pub fn template(&self) -> &str {
        &self.template
    }
}

/// Few-shot prompter: prepends N stored examples before a query.
pub struct FewShotPrompter {
    examples: Vec<(String, String)>,
    separator: String,
    input_prefix: String,
    output_prefix: String,
}

impl FewShotPrompter {
    /// Create with the given separator string.
    pub fn new(separator: impl Into<String>) -> Self {
        Self {
            examples: Vec::new(),
            separator: separator.into(),
            input_prefix: "Input: ".to_string(),
            output_prefix: "Output: ".to_string(),
        }
    }

    /// Set custom input / output prefixes.
    pub fn with_prefixes(
        mut self,
        input_prefix: impl Into<String>,
        output_prefix: impl Into<String>,
    ) -> Self {
        self.input_prefix = input_prefix.into();
        self.output_prefix = output_prefix.into();
        self
    }

    /// Add an (input, output) example.
    pub fn add_example(&mut self, input: impl Into<String>, output: impl Into<String>) {
        self.examples.push((input.into(), output.into()));
    }

    /// Build the prompt: stored examples + runtime examples + query.
    pub fn build_prompt(&self, examples: &[(&str, &str)], query: &str) -> String {
        let mut parts: Vec<String> = Vec::new();
        // Stored examples first.
        for (inp, out) in &self.examples {
            parts.push(format!(
                "{}{}{}{}{}",
                self.input_prefix, inp, self.separator, self.output_prefix, out
            ));
        }
        // Runtime examples.
        for &(inp, out) in examples {
            parts.push(format!(
                "{}{}{}{}{}",
                self.input_prefix, inp, self.separator, self.output_prefix, out
            ));
        }
        // Final query.
        parts.push(format!("{}{}", self.input_prefix, query));
        parts.join(&self.separator)
    }

    /// Number of stored examples.
    pub fn num_examples(&self) -> usize {
        self.examples.len()
    }
}

/// Instruction tuning formatter (Alpaca / LLaMA-style) for system + user + assistant turns.
pub struct InstructionTuningFormatter {
    system_prefix: String,
    user_prefix: String,
    assistant_prefix: String,
}

impl InstructionTuningFormatter {
    /// Alpaca-style formatter.
    pub fn alpaca() -> Self {
        Self {
            system_prefix: "### System:\n".to_string(),
            user_prefix: "\n### Instruction:\n".to_string(),
            assistant_prefix: "\n### Response:\n".to_string(),
        }
    }

    /// LLaMA-2 chat style formatter.
    pub fn llama2() -> Self {
        Self {
            system_prefix: "<<SYS>>\n".to_string(),
            user_prefix: "<</SYS>>\n[INST] ".to_string(),
            assistant_prefix: " [/INST]\n".to_string(),
        }
    }

    /// Custom formatter with user-defined prefixes.
    pub fn custom(
        system_prefix: impl Into<String>,
        user_prefix: impl Into<String>,
        assistant_prefix: impl Into<String>,
    ) -> Self {
        Self {
            system_prefix: system_prefix.into(),
            user_prefix: user_prefix.into(),
            assistant_prefix: assistant_prefix.into(),
        }
    }

    /// Format `system_message`, `user_turn`, and optional `assistant_turn_prefix`.
    pub fn format(
        &self,
        system_message: &str,
        user_turn: &str,
        assistant_turn_prefix: &str,
    ) -> String {
        format!(
            "{}{}{}{}{}{}",
            self.system_prefix,
            system_message,
            self.user_prefix,
            user_turn,
            self.assistant_prefix,
            assistant_turn_prefix,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Retrieval-Augmented Generation (RAG)
// ─────────────────────────────────────────────────────────────────────────────

/// Text chunk with dense embedding and metadata for RAG.
#[derive(Clone, Debug)]
pub struct DocumentChunk {
    pub text: String,
    pub embedding: Vec<f64>,
    pub metadata: HashMap<String, String>,
}

impl DocumentChunk {
    /// Create a chunk.
    pub fn new(
        text: impl Into<String>,
        embedding: Vec<f64>,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            text: text.into(),
            embedding,
            metadata,
        }
    }
}

/// Brute-force cosine-similarity vector store.
pub struct VectorStore {
    pub(crate) chunks: Vec<DocumentChunk>,
    dim: usize,
}

impl VectorStore {
    /// Create empty store for embeddings of dimension `dim`.
    pub fn new(dim: usize) -> Self {
        Self {
            chunks: Vec::new(),
            dim,
        }
    }

    /// Add a chunk.
    pub fn add(&mut self, chunk: DocumentChunk) {
        self.chunks.push(chunk);
    }

    /// Return the `k` most cosine-similar chunks to `query_emb`.
    pub fn search(&self, query_emb: &[f64], k: usize) -> Vec<DocumentChunk> {
        if self.chunks.is_empty() || k == 0 {
            return vec![];
        }
        let mut scored: Vec<(usize, f64)> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (i, cosine_similarity(query_emb, &c.embedding)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .iter()
            .take(k)
            .map(|&(i, _)| self.chunks[i].clone())
            .collect()
    }

    /// Number of chunks.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }
    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
    /// Expected embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// BM25 retriever (Robertson et al.; k1=1.5, b=0.75).
pub struct BM25Retriever {
    pub(crate) docs: Vec<Vec<String>>,
    df: HashMap<String, usize>,
    avg_dl: f64,
    k1: f64,
    b: f64,
}

impl BM25Retriever {
    const DEFAULT_K1: f64 = 1.5;
    const DEFAULT_B: f64 = 0.75;

    /// Build a BM25 index from tokenized documents.
    pub fn build(docs: Vec<Vec<String>>) -> Self {
        let mut df: HashMap<String, usize> = HashMap::new();
        let total_len: usize = docs.iter().map(|d| d.len()).sum();
        let avg_dl = if docs.is_empty() {
            0.0
        } else {
            total_len as f64 / docs.len() as f64
        };
        for doc in &docs {
            let unique_terms: std::collections::HashSet<&str> =
                doc.iter().map(|t| t.as_str()).collect();
            for term in unique_terms {
                *df.entry(term.to_string()).or_insert(0) += 1;
            }
        }
        Self {
            docs,
            df,
            avg_dl,
            k1: Self::DEFAULT_K1,
            b: Self::DEFAULT_B,
        }
    }

    /// BM25 score for `doc` against `query_terms`.
    pub fn score(&self, doc: &[String], query_terms: &[&str]) -> f64 {
        let n = self.docs.len() as f64;
        let dl = doc.len() as f64;
        let mut total = 0.0_f64;
        let avg = if self.avg_dl == 0.0 { 1.0 } else { self.avg_dl };
        for &term in query_terms {
            let df_t = self.df.get(term).copied().unwrap_or(0) as f64;
            if df_t == 0.0 {
                continue;
            }
            // IDF with Robertson smoothing.
            let idf = ((n - df_t + 0.5) / (df_t + 0.5) + 1.0).ln();
            // Term frequency in this document.
            let tf = doc.iter().filter(|t| t.as_str() == term).count() as f64;
            let tf_norm =
                tf * (self.k1 + 1.0) / (tf + self.k1 * (1.0 - self.b + self.b * dl / avg));
            total += idf * tf_norm;
        }
        total
    }

    /// Top-`k` document indices for `query_terms`.
    pub fn search(&self, query_terms: &[&str], k: usize) -> Vec<usize> {
        if self.docs.is_empty() || k == 0 {
            return vec![];
        }
        let mut scored: Vec<(usize, f64)> = self
            .docs
            .iter()
            .enumerate()
            .map(|(i, doc)| (i, self.score(doc, query_terms)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.iter().take(k).map(|&(i, _)| i).collect()
    }

    /// Number of indexed documents.
    pub fn num_docs(&self) -> usize {
        self.docs.len()
    }
}

/// Hybrid BM25 + dense vector retriever with α-weighted score fusion.
pub struct HybridRetriever {
    bm25: BM25Retriever,
    vector_store: VectorStore,
    alpha: f64,
}

impl HybridRetriever {
    /// Create (`alpha` = BM25 weight; 1-alpha = vector weight).
    pub fn new(bm25: BM25Retriever, vector_store: VectorStore, alpha: f64) -> Self {
        Self {
            bm25,
            vector_store,
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Top-`k` chunks using min-max normalised BM25 + cosine score fusion.
    pub fn retrieve(
        &self,
        query_terms: &[&str],
        query_emb: &[f64],
        k: usize,
    ) -> Vec<DocumentChunk> {
        if k == 0 {
            return vec![];
        }
        let n = self.vector_store.len();
        if n == 0 {
            return vec![];
        }

        // BM25 scores (one per BM25 doc, or 0.0 if out of bounds).
        let bm25_scores: Vec<f64> = (0..n)
            .map(|i| {
                if i < self.bm25.docs.len() {
                    self.bm25.score(&self.bm25.docs[i], query_terms)
                } else {
                    0.0
                }
            })
            .collect();

        // Vector scores.
        let vec_scores: Vec<f64> = self
            .vector_store
            .chunks
            .iter()
            .map(|c| cosine_similarity(query_emb, &c.embedding))
            .collect();

        // Normalise each score array to [0, 1].
        let normalize = |scores: &[f64]| -> Vec<f64> {
            let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = max - min;
            if range < 1e-12 {
                return vec![0.5; scores.len()];
            }
            scores.iter().map(|&s| (s - min) / range).collect()
        };

        let bm25_norm = normalize(&bm25_scores);
        let vec_norm = normalize(&vec_scores);

        let mut combined: Vec<(usize, f64)> = (0..n.min(bm25_norm.len()).min(vec_norm.len()))
            .map(|i| {
                let score = self.alpha * bm25_norm[i] + (1.0 - self.alpha) * vec_norm[i];
                (i, score)
            })
            .collect();

        combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        combined
            .iter()
            .take(k)
            .filter_map(|&(i, _)| self.vector_store.chunks.get(i).cloned())
            .collect()
    }
}

/// RAG pipeline: retrieve relevant chunks and prepend them as context.
pub struct RagPipeline {
    retriever: HybridRetriever,
    context_prefix: String,
    context_separator: String,
    max_context_chunks: usize,
}

impl RagPipeline {
    /// Create a RAG pipeline.
    pub fn new(retriever: HybridRetriever, max_context_chunks: usize) -> Self {
        Self {
            retriever,
            context_prefix: "Context:\n".to_string(),
            context_separator: "\n---\n".to_string(),
            max_context_chunks,
        }
    }

    /// Set context prefix and separator strings.
    pub fn with_format(
        mut self,
        context_prefix: impl Into<String>,
        context_separator: impl Into<String>,
    ) -> Self {
        self.context_prefix = context_prefix.into();
        self.context_separator = context_separator.into();
        self
    }

    /// Retrieve context and prepend to `query`. Returns augmented prompt.
    pub fn augment(&self, query: &str, query_terms: &[&str], query_emb: &[f64]) -> String {
        let chunks = self
            .retriever
            .retrieve(query_terms, query_emb, self.max_context_chunks);

        if chunks.is_empty() {
            return query.to_string();
        }

        let context_body: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let context = context_body.join(&self.context_separator);
        format!("{}{}\n\nQuery: {}", self.context_prefix, context, query)
    }
}
