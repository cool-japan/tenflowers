//! NLP & Text Understanding Components — Track NLP.
//!
//! Provides a comprehensive suite of natural language processing primitives
//! including text preprocessing, sequence labeling, semantic understanding,
//! question answering, and text generation/summarization.
//!
//! All components operate on plain Rust `Vec<f64>` / `Vec<usize>` buffers so
//! there is no dependency on the `Tensor` type, making the module easy to
//! compose and test in isolation.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn l2_norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Numerically-stable softmax.
fn softmax_f64(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum.max(1e-300)).collect()
}

/// Row-major matrix–vector product with bias.
fn linear(x: &[f64], weights: &[Vec<f64>], bias: &[f64]) -> Vec<f64> {
    weights
        .iter()
        .zip(bias.iter())
        .map(|(row, b)| dot(row, x) + b)
        .collect()
}

/// ReLU: `max(0, x)`.
#[inline]
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Element-wise ReLU.
fn relu_vec(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| relu(x)).collect()
}

/// Glorot-scaled random weight matrix (`rows × cols`).
fn init_weights(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let scale = (2.0_f64 / (rows + cols) as f64).sqrt();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    let u1: f64 = rng.random::<f64>().max(1e-15);
                    let u2: f64 = rng.random::<f64>();
                    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos() * scale
                })
                .collect()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Text Preprocessing
// ─────────────────────────────────────────────────────────────────────────────

/// BPE + WordPiece hybrid subword tokenizer.
pub struct SubwordTokenizer;

impl SubwordTokenizer {
    /// Encode `text` into vocabulary IDs using WordPiece-style longest-match.
    pub fn encode(text: &str, vocab: &HashMap<String, usize>) -> Vec<usize> {
        let unk_id = *vocab.get("[UNK]").unwrap_or(&0);
        let mut ids = Vec::new();

        for word in text.split_whitespace() {
            let chars: Vec<char> = word.chars().collect();
            let n = chars.len();
            let mut start = 0;
            let mut is_first = true;

            while start < n {
                // Try longest match from `start`.
                let mut matched_end = None;
                for end in (start + 1..=n).rev() {
                    let sub: String = chars[start..end].iter().collect();
                    let key = if is_first {
                        sub.clone()
                    } else {
                        format!("##{}", sub)
                    };
                    if vocab.contains_key(&key) {
                        matched_end = Some((end, key));
                        break;
                    }
                }
                match matched_end {
                    Some((end, key)) => {
                        ids.push(*vocab.get(&key).unwrap_or(&unk_id));
                        start = end;
                        is_first = false;
                    }
                    None => {
                        // Single character fallback.
                        let ch: String = chars[start..start + 1].iter().collect();
                        let key = if is_first {
                            ch
                        } else {
                            format!("##{}", chars[start..start + 1].iter().collect::<String>())
                        };
                        ids.push(*vocab.get(&key).unwrap_or(&unk_id));
                        start += 1;
                        is_first = false;
                    }
                }
            }
        }
        ids
    }

    /// Decode a sequence of IDs back to a string, merging `##` continuation tokens.
    pub fn decode(ids: &[usize], vocab: &HashMap<usize, String>) -> String {
        let mut parts: Vec<String> = Vec::new();
        for &id in ids {
            let token = vocab
                .get(&id)
                .cloned()
                .unwrap_or_else(|| "[UNK]".to_string());
            if let Some(rest) = token.strip_prefix("##") {
                // Merge with previous token without space.
                if let Some(last) = parts.last_mut() {
                    last.push_str(rest);
                } else {
                    parts.push(rest.to_string());
                }
            } else {
                parts.push(token);
            }
        }
        parts.join(" ")
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Unigram language model (SentencePiece-style) tokenizer.
pub struct SentencePieceModel;

impl SentencePieceModel {
    /// Build a unigram vocabulary (token → log-prob) from `corpus`.
    pub fn build_vocab(corpus: &[&str], vocab_size: usize) -> HashMap<String, f64> {
        let mut freq: HashMap<String, usize> = HashMap::new();
        for sentence in corpus {
            for word in sentence.split_whitespace() {
                let marked = format!("▁{}", word);
                let chars: Vec<char> = marked.chars().collect();
                for &ch in &chars {
                    *freq.entry(ch.to_string()).or_insert(0) += 1;
                }
                for pair in chars.windows(2) {
                    *freq.entry(pair.iter().collect::<String>()).or_insert(0) += 1;
                }
            }
        }
        let mut entries: Vec<(String, usize)> = freq.into_iter().collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        entries.truncate(vocab_size.max(1));

        let total: usize = entries.iter().map(|(_, c)| c).sum();
        let total_f = total.max(1) as f64;

        entries
            .into_iter()
            .map(|(tok, cnt)| (tok, (cnt as f64 / total_f).ln()))
            .collect()
    }

    /// Encode `text` using greedy Viterbi segmentation; returns token strings.
    pub fn encode(text: &str, vocab: &HashMap<String, f64>) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        if n == 0 {
            return Vec::new();
        }

        // Forward Viterbi: best[i] = (score, back_len) ending at position i.
        let neg_inf = f64::NEG_INFINITY;
        let mut best_score = vec![neg_inf; n + 1];
        let mut back_len = vec![0usize; n + 1];
        best_score[0] = 0.0;

        for i in 0..n {
            if best_score[i] == neg_inf {
                continue;
            }
            for j in (i + 1)..=n {
                let token: String = chars[i..j].iter().collect();
                if let Some(&log_p) = vocab.get(&token) {
                    let new_score = best_score[i] + log_p;
                    if new_score > best_score[j] {
                        best_score[j] = new_score;
                        back_len[j] = j - i;
                    }
                }
            }
        }

        let mut tokens = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let len = back_len[pos];
            if len == 0 {
                // Fallback: single character.
                tokens.push(chars[pos - 1].to_string());
                pos -= 1;
            } else {
                let token: String = chars[pos - len..pos].iter().collect();
                tokens.push(token);
                pos -= len;
            }
        }
        tokens.reverse();
        tokens
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// FastText-style character n-gram feature extractor.
pub struct CharacterNgram;

impl CharacterNgram {
    /// Extract character n-grams (lengths `n_min..=n_max`) from `word` (wrapped in `<>`).
    pub fn extract_ngrams(word: &str, n_min: usize, n_max: usize) -> Vec<String> {
        let padded = format!("<{}>", word);
        let chars: Vec<char> = padded.chars().collect();
        let len = chars.len();
        let mut ngrams = Vec::new();

        for n in n_min..=n_max {
            if n > len {
                break;
            }
            for start in 0..=(len - n) {
                let ng: String = chars[start..start + n].iter().collect();
                ngrams.push(ng);
            }
        }
        ngrams
    }

    /// Hash a character n-gram to a bucket index using FNV-1a hashing.
    pub fn hash_ngram(ngram: &str, bucket_size: usize) -> usize {
        let mut h: u64 = 2_166_136_261_u64;
        for byte in ngram.bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(16_777_619);
        }
        (h as usize) % bucket_size.max(1)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Unicode-aware text normalizer (lowercase, remove ASCII punctuation, collapse whitespace).
pub struct TextNormalizer;

impl TextNormalizer {
    /// Normalize: lowercase → strip ASCII punctuation → collapse whitespace.
    pub fn normalize(text: &str) -> String {
        let lower = text.to_lowercase();
        let cleaned: String = lower
            .chars()
            .map(|c| if c.is_ascii_punctuation() { ' ' } else { c })
            .collect();
        // Collapse whitespace.
        let mut result = String::with_capacity(cleaned.len());
        let mut prev_space = false;
        for ch in cleaned.chars() {
            if ch.is_whitespace() {
                if !prev_space {
                    result.push(' ');
                }
                prev_space = true;
            } else {
                result.push(ch);
                prev_space = false;
            }
        }
        result.trim().to_string()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Batch collator: pad/truncate token-ID sequences to a uniform length.
pub struct DataCollator;

impl DataCollator {
    /// Pad/truncate `sequences` to `max_len`, using `pad_id` for padding.
    pub fn collate(sequences: &[Vec<usize>], max_len: usize, pad_id: usize) -> Vec<Vec<usize>> {
        sequences
            .iter()
            .map(|seq| {
                let mut row = seq.clone();
                row.truncate(max_len);
                while row.len() < max_len {
                    row.push(pad_id);
                }
                row
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Sequence Labeling
// ─────────────────────────────────────────────────────────────────────────────

/// Linear-chain Conditional Random Field: Viterbi decoding + log-partition.
pub struct CrfLayer;

impl CrfLayer {
    /// Viterbi decode emissions `[T, num_tags]` with transition log-potentials.
    pub fn viterbi_decode(emissions: &[Vec<f64>], transitions: &[Vec<f64>]) -> Vec<usize> {
        let t = emissions.len();
        if t == 0 {
            return Vec::new();
        }
        let n = emissions[0].len();
        if n == 0 {
            return Vec::new();
        }

        let mut dp: Vec<f64> = emissions[0].clone();
        let mut back: Vec<Vec<usize>> = vec![vec![0usize; n]; t];

        for step in 1..t {
            let mut new_dp = vec![f64::NEG_INFINITY; n];
            for j in 0..n {
                for i in 0..n {
                    let score = dp[i]
                        + transitions
                            .get(i)
                            .and_then(|r| r.get(j))
                            .copied()
                            .unwrap_or(0.0)
                        + emissions[step][j];
                    if score > new_dp[j] {
                        new_dp[j] = score;
                        back[step][j] = i;
                    }
                }
            }
            dp = new_dp;
        }

        let mut best_last = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (j, &s) in dp.iter().enumerate() {
            if s > best_score {
                best_score = s;
                best_last = j;
            }
        }

        // backtrack
        let mut path = vec![0usize; t];
        path[t - 1] = best_last;
        for step in (1..t).rev() {
            path[step - 1] = back[step][path[step]];
        }
        path
    }

    /// Compute `log Z` (log-partition) via the forward algorithm.
    pub fn forward_backward(emissions: &[Vec<f64>], transitions: &[Vec<f64>]) -> f64 {
        let t = emissions.len();
        if t == 0 {
            return 0.0;
        }
        let n = emissions[0].len();
        if n == 0 {
            return 0.0;
        }

        let mut alpha: Vec<f64> = emissions[0].clone();

        for step in 1..t {
            let mut new_alpha = vec![f64::NEG_INFINITY; n];
            for j in 0..n {
                let scores: Vec<f64> = (0..n)
                    .map(|i| {
                        alpha[i]
                            + transitions
                                .get(i)
                                .and_then(|r| r.get(j))
                                .copied()
                                .unwrap_or(0.0)
                    })
                    .collect();
                let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                new_alpha[j] = max_s
                    + scores.iter().map(|&s| (s - max_s).exp()).sum::<f64>().ln()
                    + emissions[step][j];
            }
            alpha = new_alpha;
        }

        let max_a = alpha.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        max_a + alpha.iter().map(|&a| (a - max_a).exp()).sum::<f64>().ln()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Named Entity Recognition (MLP emission scorer + CRF Viterbi decoding).
pub struct NerModel {
    embed_dim: usize,
    hidden_dim: usize,
    num_tags: usize,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
    transitions: Vec<Vec<f64>>,
}

impl NerModel {
    /// Create a NER model (`vocab_size` one-hot input, `hidden_dim`, `num_tags` BIO outputs).
    pub fn new(vocab_size: usize, hidden_dim: usize, num_tags: usize) -> Result<Self> {
        if vocab_size == 0 || hidden_dim == 0 || num_tags == 0 {
            return Err(TensorError::invalid_argument(
                "NerModel: dimensions must be > 0".to_string(),
            ));
        }
        Ok(Self {
            embed_dim: vocab_size,
            hidden_dim,
            num_tags,
            w1: init_weights(hidden_dim, vocab_size, 1),
            b1: vec![0.0; hidden_dim],
            w2: init_weights(num_tags, hidden_dim, 2),
            b2: vec![0.0; num_tags],
            transitions: init_weights(num_tags, num_tags, 3),
        })
    }

    /// Run a forward pass and return a BIO tag index for each token.
    pub fn forward(&self, token_ids: &[usize]) -> Vec<usize> {
        // Compute emission scores for each token.
        let emissions: Vec<Vec<f64>> = token_ids
            .iter()
            .map(|&id| {
                let mut emb = vec![0.0_f64; self.embed_dim];
                if id < self.embed_dim {
                    emb[id] = 1.0;
                }
                let h = relu_vec(&linear(&emb, &self.w1, &self.b1));
                linear(&h, &self.w2, &self.b2)
            })
            .collect();

        CrfLayer::viterbi_decode(&emissions, &self.transitions)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Part-of-speech tagger (single-layer MLP over one-hot token embeddings).
pub struct PosTagging {
    vocab_size: usize,
    num_pos: usize,
    w: Vec<Vec<f64>>,
    b: Vec<f64>,
}

impl PosTagging {
    /// Create a new POS tagger.
    pub fn new(vocab_size: usize, num_pos: usize) -> Result<Self> {
        if vocab_size == 0 || num_pos == 0 {
            return Err(TensorError::invalid_argument(
                "PosTagging: dimensions must be > 0".to_string(),
            ));
        }
        Ok(Self {
            vocab_size,
            num_pos,
            w: init_weights(num_pos, vocab_size, 42),
            b: vec![0.0; num_pos],
        })
    }

    /// Assign a POS tag index to each token ID.
    pub fn tag(&self, token_ids: &[usize]) -> Vec<usize> {
        token_ids
            .iter()
            .map(|&id| {
                let mut emb = vec![0.0_f64; self.vocab_size];
                if id < self.vocab_size {
                    emb[id] = 1.0;
                }
                let logits = linear(&emb, &self.w, &self.b);
                logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// IOB2 span decoder: 0=O, odd=B-type, even=I-type.
pub struct ChunkingDecoder;

impl ChunkingDecoder {
    /// Decode BIO tags into `(start, end_exclusive, entity_type)` spans.
    pub fn decode(bio_tags: &[usize]) -> Vec<(usize, usize, usize)> {
        let mut spans = Vec::new();
        let mut in_entity: Option<(usize, usize, usize)> = None; // (start, entity_type, current_pos)

        for (i, &tag) in bio_tags.iter().enumerate() {
            if tag == 0 {
                // O tag — close any open entity.
                if let Some((start, etype, _)) = in_entity.take() {
                    spans.push((start, i, etype));
                }
            } else if tag % 2 == 1 {
                // B-tag — entity type = (tag + 1) / 2.
                if let Some((start, etype, _)) = in_entity.take() {
                    spans.push((start, i, etype));
                }
                let etype = (tag + 1) / 2;
                in_entity = Some((i, etype, i));
            } else {
                // I-tag.
                let etype = tag / 2;
                match &mut in_entity {
                    Some((_, cur_type, _)) if *cur_type == etype => {
                        // Continue current entity.
                    }
                    _ => {
                        // Mismatched or no open entity — treat as B.
                        if let Some((start, et, _)) = in_entity.take() {
                            spans.push((start, i, et));
                        }
                        in_entity = Some((i, etype, i));
                    }
                }
            }
        }
        if let Some((start, etype, _)) = in_entity {
            spans.push((start, bio_tags.len(), etype));
        }
        spans
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// CYK constituency parser: greedy non-overlapping span selection by score.
pub struct ConstituencyParser;

impl ConstituencyParser {
    /// Parse `n_tokens` tokens using `span_scores[i][j]`; returns non-overlapping `(start, end)` spans.
    pub fn parse(n_tokens: usize, span_scores: &[Vec<f64>]) -> Vec<(usize, usize)> {
        if n_tokens == 0 {
            return Vec::new();
        }

        // Collect all scored spans.
        let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
        for i in 0..n_tokens {
            if i >= span_scores.len() {
                break;
            }
            for j in i..n_tokens {
                if j >= span_scores[i].len() {
                    break;
                }
                candidates.push((span_scores[i][j], i, j));
            }
        }

        // Sort descending by score.
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Greedy non-overlapping selection.
        let mut selected: Vec<(usize, usize)> = Vec::new();
        let mut covered = vec![false; n_tokens];

        for (_, start, end) in candidates {
            // Check if any position in [start, end] is already covered.
            let conflict = (start..=end).any(|pos| covered[pos]);
            if !conflict {
                for pos in start..=end {
                    covered[pos] = true;
                }
                selected.push((start, end));
            }
        }
        selected
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Semantic Understanding
// ─────────────────────────────────────────────────────────────────────────────

/// Sentence encoder: mean-pooling over token embeddings.
pub struct SentenceEncoder;

impl SentenceEncoder {
    /// Produce a sentence embedding by mean-pooling all token embeddings.
    pub fn encode(tokens: &[Vec<f64>]) -> Vec<f64> {
        if tokens.is_empty() {
            return Vec::new();
        }
        let dim = tokens[0].len();
        let n = tokens.len() as f64;
        let mut mean = vec![0.0_f64; dim];
        for tok in tokens {
            for (i, &v) in tok.iter().enumerate() {
                if i < dim {
                    mean[i] += v / n;
                }
            }
        }
        mean
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Cosine similarity and cross-encoder scoring utilities.
pub struct SemanticSimilarity;

impl SemanticSimilarity {
    /// Cosine similarity in `[-1, 1]`; returns `0.0` for zero vectors.
    pub fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
        let norm_a = l2_norm(a);
        let norm_b = l2_norm(b);
        if norm_a < 1e-15 || norm_b < 1e-15 {
            return 0.0;
        }
        (dot(a, b) / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    /// Cross-encoder score in `[0,1]`: pool concatenated sequences then project.
    pub fn cross_encode(a_tokens: &[Vec<f64>], b_tokens: &[Vec<f64>]) -> f64 {
        let mut combined = a_tokens.to_vec();
        combined.extend_from_slice(b_tokens);
        let pooled = SentenceEncoder::encode(&combined);
        if pooled.is_empty() {
            return 0.5;
        }
        // Use mean of pooled vector (tanh-squashed to [0, 1]).
        let mean_val = pooled.iter().sum::<f64>() / pooled.len() as f64;
        (mean_val.tanh() + 1.0) / 2.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// NLI classifier (entailment / contradiction / neutral) via two-layer MLP.
pub struct TextualEntailment {
    in_dim: usize,
    hidden: usize,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
}

impl TextualEntailment {
    /// Create a TextualEntailment classifier for sentence embeddings of dimension `emb_dim`.
    pub fn new(emb_dim: usize) -> Result<Self> {
        if emb_dim == 0 {
            return Err(TensorError::invalid_argument(
                "TextualEntailment: emb_dim must be > 0".to_string(),
            ));
        }
        let in_dim = emb_dim * 2;
        let hidden = (in_dim / 2).max(4);
        Ok(Self {
            in_dim,
            hidden,
            w1: init_weights(hidden, in_dim, 10),
            b1: vec![0.0; hidden],
            w2: init_weights(3, hidden, 11),
            b2: vec![0.0; 3],
        })
    }

    /// Returns softmax probabilities `[entailment, contradiction, neutral]`.
    pub fn predict(&self, premise: &[f64], hypothesis: &[f64]) -> [f64; 3] {
        let mut input = Vec::with_capacity(self.in_dim);
        input.extend_from_slice(premise);
        input.extend_from_slice(hypothesis);
        input.resize(self.in_dim, 0.0);

        let h = relu_vec(&linear(&input, &self.w1, &self.b1));
        let logits = linear(&h, &self.w2, &self.b2);
        let probs = softmax_f64(&logits);
        [probs[0], probs[1], probs[2]]
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Co-reference resolver: bilinear mention-pair scoring `m1ᵀ W m2 + bias·‖doc‖`.
pub struct CoReferenceResolver {
    dim: usize,
    w_bilinear: Vec<Vec<f64>>,
    bias: f64,
}

impl CoReferenceResolver {
    /// Create a co-reference resolver for embeddings of dimension `dim`.
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(TensorError::invalid_argument(
                "CoReferenceResolver: dim must be > 0".to_string(),
            ));
        }
        Ok(Self {
            dim,
            w_bilinear: init_weights(dim, dim, 20),
            bias: 0.1,
        })
    }

    /// Antecedent compatibility score in `[0, 1]`.
    pub fn score_pair(&self, mention1: &[f64], mention2: &[f64], doc_emb: &[f64]) -> f64 {
        let wm2 = linear(mention2, &self.w_bilinear, &vec![0.0; self.dim]);
        let raw = dot(mention1, &wm2) + l2_norm(doc_emb) * self.bias;
        1.0 / (1.0 + (-raw).exp())
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Semantic Role Labeler: assigns predicate-argument labels per token.
pub struct SemanticRoleLabeler {
    dim: usize,
    num_roles: usize,
    w: Vec<Vec<f64>>,
    b: Vec<f64>,
}

impl SemanticRoleLabeler {
    /// Role labels used by this implementation.
    pub const ROLES: &'static [&'static str] = &["O", "ARG0", "ARG1", "ARG2", "ARGM-TMP", "V"];

    /// Create a SemanticRoleLabeler.
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(TensorError::invalid_argument(
                "SemanticRoleLabeler: dim must be > 0".to_string(),
            ));
        }
        let num_roles = Self::ROLES.len();
        Ok(Self {
            dim,
            num_roles,
            w: init_weights(num_roles, dim * 2, 30),
            b: vec![0.0; num_roles],
        })
    }

    /// Returns `(token_idx, role_label)` for each token given `predicate_idx`.
    pub fn label(
        &self,
        predicate_idx: usize,
        token_embs: &[Vec<f64>],
    ) -> Vec<(usize, &'static str)> {
        let n = token_embs.len();
        if n == 0 {
            return Vec::new();
        }
        let pred_emb = if predicate_idx < n {
            token_embs[predicate_idx].clone()
        } else {
            vec![0.0; self.dim]
        };

        token_embs
            .iter()
            .enumerate()
            .map(|(i, tok_emb)| {
                let mut input = tok_emb.clone();
                input.extend_from_slice(&pred_emb);
                input.resize(self.dim * 2, 0.0);

                let logits = linear(&input, &self.w, &self.b);
                let best = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                let role_label = Self::ROLES
                    .get(best % Self::ROLES.len())
                    .copied()
                    .unwrap_or("O");
                (i, role_label)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Question Answering
// ─────────────────────────────────────────────────────────────────────────────

/// Span extraction QA: argmax over start/end logits via learned projections.
pub struct SpanExtractionQa {
    dim: usize,
    w_start: Vec<f64>,
    w_end: Vec<f64>,
}

impl SpanExtractionQa {
    /// Create a SpanExtractionQa model.
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(TensorError::invalid_argument(
                "SpanExtractionQa: dim must be > 0".to_string(),
            ));
        }
        let w_start = init_weights(1, dim, 40)[0].clone();
        let w_end = init_weights(1, dim, 41)[0].clone();
        Ok(Self {
            dim,
            w_start,
            w_end,
        })
    }

    /// Returns `(start_idx, end_idx)` over `N` context tokens (flat `N × dim`).
    pub fn forward(&self, question: &[f64], context: &[f64]) -> (usize, usize) {
        let d = self.dim;
        if d == 0 || context.len() < d {
            return (0, 0);
        }

        let n_ctx = context.len() / d;
        let mut start_logits = Vec::with_capacity(n_ctx);
        let mut end_logits = Vec::with_capacity(n_ctx);

        for i in 0..n_ctx {
            let tok = &context[i * d..(i + 1) * d];
            let fused: Vec<f64> = tok
                .iter()
                .zip(question.iter().chain(std::iter::repeat(&0.0)).take(d))
                .map(|(&t, &q)| t + q * 0.5)
                .collect();
            start_logits.push(dot(&fused, &self.w_start));
            end_logits.push(dot(&fused, &self.w_end));
        }

        let start = start_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let end = end_logits
            .iter()
            .enumerate()
            .skip(start)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(start);

        (start, end)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// TF-IDF retriever + span-extraction reader.
pub struct RetrieverReader {
    dim: usize,
    reader: SpanExtractionQa,
}

impl RetrieverReader {
    /// Create a RetrieverReader.
    pub fn new(dim: usize) -> Result<Self> {
        Ok(Self {
            dim,
            reader: SpanExtractionQa::new(dim)?,
        })
    }

    /// Returns `(doc_idx, start, end)` by TF-IDF retrieval + span extraction.
    pub fn answer(&self, question: &str, corpus: &[&str]) -> (usize, usize, usize) {
        if corpus.is_empty() {
            return (0, 0, 0);
        }
        let q_terms: std::collections::HashSet<String> = question
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();

        let best_doc = corpus
            .iter()
            .enumerate()
            .map(|(idx, doc)| {
                let score = doc
                    .split_whitespace()
                    .map(|w| w.to_lowercase())
                    .filter(|w| q_terms.contains(w))
                    .count();
                (score, idx)
            })
            .max_by_key(|(score, _)| *score)
            .map(|(_, idx)| idx)
            .unwrap_or(0);

        let doc_text = corpus[best_doc];
        let n_tokens = doc_text.split_whitespace().count().max(1);
        let d = self.dim;

        // Build simple token embeddings (one-hot hashing into dim-sized vector).
        let context_flat: Vec<f64> = doc_text
            .split_whitespace()
            .flat_map(|tok| {
                let h = CharacterNgram::hash_ngram(tok, d);
                let mut emb = vec![0.0_f64; d];
                emb[h % d] = 1.0;
                emb
            })
            .collect();

        // Build question embedding similarly.
        let q_emb: Vec<f64> = {
            let n_q = q_terms.len().max(1);
            let mut v = vec![0.0_f64; d];
            for term in &q_terms {
                let h = CharacterNgram::hash_ngram(term, d) % d;
                v[h] += 1.0 / n_q as f64;
            }
            v
        };

        let expected_len = n_tokens * d;
        let context = if context_flat.len() >= expected_len {
            context_flat[..expected_len].to_vec()
        } else {
            let mut padded = context_flat;
            padded.resize(expected_len, 0.0);
            padded
        };

        let (start, end) = self.reader.forward(&q_emb, &context);
        (best_doc, start, end)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Multi-hop QA: decomposes a question into sub-question embeddings.
pub struct MultiHopQa {
    dim: usize,
    w_decompose: Vec<Vec<f64>>,
    b_decompose: Vec<f64>,
}

impl MultiHopQa {
    /// Create a MultiHopQa module.
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(TensorError::invalid_argument(
                "MultiHopQa: dim must be > 0".to_string(),
            ));
        }
        Ok(Self {
            dim,
            w_decompose: init_weights(dim * 2, dim, 50),
            b_decompose: vec![0.0; dim * 2],
        })
    }

    /// Decompose question embedding into two sub-question embeddings.
    pub fn decompose(&self, question_emb: &[f64]) -> Vec<Vec<f64>> {
        let mut input = question_emb.to_vec();
        input.resize(self.dim, 0.0);
        let projected = relu_vec(&linear(&input, &self.w_decompose, &self.b_decompose));
        // Split into two halves.
        let half = projected.len() / 2;
        vec![projected[..half].to_vec(), projected[half..].to_vec()]
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Open-domain QA: cosine-sim retriever + span-extraction reader.
pub struct OpenDomainQa {
    dim: usize,
    reader: SpanExtractionQa,
}

impl OpenDomainQa {
    /// Create an OpenDomainQa module.
    pub fn new(dim: usize) -> Result<Self> {
        Ok(Self {
            dim,
            reader: SpanExtractionQa::new(dim)?,
        })
    }

    /// Returns `(best_doc_idx, span_logits)` for the retrieved document.
    pub fn predict(&self, question_emb: &[f64], doc_embs: &[Vec<f64>]) -> (usize, Vec<f64>) {
        if doc_embs.is_empty() {
            return (0, Vec::new());
        }
        let best_doc = doc_embs
            .iter()
            .enumerate()
            .map(|(i, doc)| (i, SemanticSimilarity::cosine_sim(question_emb, doc)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let d = self.dim;
        let mut context = doc_embs[best_doc].clone();
        context.resize(d, 0.0);
        let mut q = question_emb.to_vec();
        q.resize(d, 0.0);

        let span_logits: Vec<f64> = context
            .iter()
            .zip(q.iter())
            .map(|(&c, &qv)| c * qv)
            .collect();

        (best_doc, span_logits)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// TriviaQA-style exact match + token F1 evaluator.
pub struct TriviaQaEvaluator;

impl TriviaQaEvaluator {
    /// Returns `(exact_match, f1)` after normalizing both strings.
    pub fn eval(pred: &str, gold: &str) -> (bool, f64) {
        let norm_pred = TextNormalizer::normalize(pred);
        let norm_gold = TextNormalizer::normalize(gold);

        let exact = norm_pred == norm_gold;

        let pred_tokens: std::collections::HashSet<&str> = norm_pred.split_whitespace().collect();
        let gold_tokens: std::collections::HashSet<&str> = norm_gold.split_whitespace().collect();

        if gold_tokens.is_empty() {
            return (exact, if pred_tokens.is_empty() { 1.0 } else { 0.0 });
        }
        if pred_tokens.is_empty() {
            return (exact, 0.0);
        }

        let common = pred_tokens.intersection(&gold_tokens).count() as f64;
        let precision = common / pred_tokens.len() as f64;
        let recall = common / gold_tokens.len() as f64;
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        (exact, f1)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Text Generation & Summarization
// ─────────────────────────────────────────────────────────────────────────────

/// Extractive summarizer: top-K sentence selection by cosine sim to centroid.
pub struct ExtractiveSummarizer;

impl ExtractiveSummarizer {
    /// Returns sorted indices of the top-`k` sentences (ascending).
    pub fn summarize(sentences: &[Vec<f64>], k: usize) -> Vec<usize> {
        if sentences.is_empty() || k == 0 {
            return Vec::new();
        }
        let k_actual = k.min(sentences.len());
        let centroid = SentenceEncoder::encode(sentences);

        // Score each sentence by cosine similarity with centroid.
        let mut scored: Vec<(usize, f64)> = sentences
            .iter()
            .enumerate()
            .map(|(i, emb)| (i, SemanticSimilarity::cosine_sim(emb, &centroid)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut selected: Vec<usize> = scored[..k_actual].iter().map(|(i, _)| *i).collect();
        selected.sort_unstable();
        selected
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Abstractive summarizer: simplified encoder-decoder that produces output
/// embeddings by attending over source embeddings.
pub struct AbstractiveSummarizer {
    dim: usize,
    w_enc: Vec<Vec<f64>>,
    b_enc: Vec<f64>,
    w_dec: Vec<Vec<f64>>,
    b_dec: Vec<f64>,
}

impl AbstractiveSummarizer {
    /// Create an AbstractiveSummarizer.
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(TensorError::invalid_argument(
                "AbstractiveSummarizer: dim must be > 0".to_string(),
            ));
        }
        Ok(Self {
            dim,
            w_enc: init_weights(dim, dim, 60),
            b_enc: vec![0.0; dim],
            w_dec: init_weights(dim, dim, 61),
            b_dec: vec![0.0; dim],
        })
    }

    /// Produce output embeddings: attention-weighted source → ReLU decode per target position.
    pub fn forward(&self, source_embs: &[Vec<f64>], target_embs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if source_embs.is_empty() || target_embs.is_empty() {
            return Vec::new();
        }
        let d = self.dim;

        // Encode source tokens.
        let encoded: Vec<Vec<f64>> = source_embs
            .iter()
            .map(|emb| {
                let mut e = emb.clone();
                e.resize(d, 0.0);
                relu_vec(&linear(&e, &self.w_enc, &self.b_enc))
            })
            .collect();

        target_embs
            .iter()
            .map(|tgt| {
                let mut t = tgt.clone();
                t.resize(d, 0.0);
                let weights =
                    softmax_f64(&encoded.iter().map(|enc| dot(&t, enc)).collect::<Vec<_>>());
                let mut context = vec![0.0_f64; d];
                for (enc, &w) in encoded.iter().zip(weights.iter()) {
                    for (i, &e) in enc.iter().enumerate() {
                        if i < d {
                            context[i] += w * e;
                        }
                    }
                }
                relu_vec(&linear(&context, &self.w_dec, &self.b_dec))
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// ROUGE metric computation (ROUGE-1, ROUGE-2, ROUGE-L).
pub struct RougeMetric;

impl RougeMetric {
    /// ROUGE-1 F1: unigram overlap.
    pub fn compute_rouge1(pred_tokens: &[usize], gold_tokens: &[usize]) -> f64 {
        if gold_tokens.is_empty() {
            return if pred_tokens.is_empty() { 1.0 } else { 0.0 };
        }
        if pred_tokens.is_empty() {
            return 0.0;
        }
        let gold_set: std::collections::HashSet<usize> = gold_tokens.iter().copied().collect();
        let pred_set: std::collections::HashSet<usize> = pred_tokens.iter().copied().collect();
        let common = gold_set.intersection(&pred_set).count() as f64;
        let precision = common / pred_set.len() as f64;
        let recall = common / gold_set.len() as f64;
        if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        }
    }

    /// ROUGE-2 F1: bigram overlap.
    pub fn compute_rouge2(pred_tokens: &[usize], gold_tokens: &[usize]) -> f64 {
        fn bigrams(toks: &[usize]) -> std::collections::HashSet<(usize, usize)> {
            toks.windows(2).map(|w| (w[0], w[1])).collect()
        }
        if gold_tokens.len() < 2 {
            return if pred_tokens.len() < 2 { 1.0 } else { 0.0 };
        }
        if pred_tokens.len() < 2 {
            return 0.0;
        }
        let gold_bg = bigrams(gold_tokens);
        let pred_bg = bigrams(pred_tokens);
        let common = gold_bg.intersection(&pred_bg).count() as f64;
        let precision = common / pred_bg.len() as f64;
        let recall = common / gold_bg.len() as f64;
        if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        }
    }

    /// ROUGE-L: longest common subsequence F1.
    pub fn rouge_l(pred: &[usize], gold: &[usize]) -> f64 {
        let lcs_len = Self::lcs_length(pred, gold);
        if gold.is_empty() || pred.is_empty() {
            return if gold.is_empty() && pred.is_empty() {
                1.0
            } else {
                0.0
            };
        }
        let precision = lcs_len as f64 / pred.len() as f64;
        let recall = lcs_len as f64 / gold.len() as f64;
        if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        }
    }

    /// Standard LCS length via DP.
    fn lcs_length(a: &[usize], b: &[usize]) -> usize {
        let m = a.len();
        let n = b.len();
        if m == 0 || n == 0 {
            return 0;
        }
        // Use two-row rolling DP to keep memory O(min(m,n)).
        let (s, t, sv, tv) = if m <= n { (m, n, a, b) } else { (n, m, b, a) };
        let mut prev = vec![0usize; s + 1];
        let mut curr = vec![0usize; s + 1];
        for j in 1..=t {
            for i in 1..=s {
                curr[i] = if sv[i - 1] == tv[j - 1] {
                    prev[i - 1] + 1
                } else {
                    curr[i - 1].max(prev[i])
                };
            }
            std::mem::swap(&mut prev, &mut curr);
            for x in curr.iter_mut() {
                *x = 0;
            }
        }
        prev[s]
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// BERTScore-proxy: greedy max-cosine-sim token F1 in `[0, 1]`.
pub struct BleurtProxy;

impl BleurtProxy {
    /// BERTScore-style F1 between `pred_embs` and `gold_embs`.
    pub fn compute(pred_embs: &[Vec<f64>], gold_embs: &[Vec<f64>]) -> f64 {
        if pred_embs.is_empty() || gold_embs.is_empty() {
            return 0.0;
        }

        // Precision: for each pred token, find max similarity to any gold token.
        let precision: f64 = pred_embs
            .iter()
            .map(|p| {
                gold_embs
                    .iter()
                    .map(|g| SemanticSimilarity::cosine_sim(p, g))
                    .fold(f64::NEG_INFINITY, f64::max)
                    .max(0.0)
            })
            .sum::<f64>()
            / pred_embs.len() as f64;

        // Recall: for each gold token, find max similarity to any pred token.
        let recall: f64 = gold_embs
            .iter()
            .map(|g| {
                pred_embs
                    .iter()
                    .map(|p| SemanticSimilarity::cosine_sim(p, g))
                    .fold(f64::NEG_INFINITY, f64::max)
                    .max(0.0)
            })
            .sum::<f64>()
            / gold_embs.len() as f64;

        if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Paraphraser: Gaussian perturbation + L2-renorm to simulate back-translation.
pub struct ParaphraserModel;

impl ParaphraserModel {
    /// Add `strength`-scaled Gaussian noise to `emb` and renormalize to the original norm.
    pub fn paraphrase(emb: &[f64], strength: f64, rng: &mut StdRng) -> Vec<f64> {
        let n = emb.len();
        let mut perturbed: Vec<f64> = emb
            .iter()
            .map(|&v| {
                // Box-Muller using two uniform samples from rng.
                let u1: f64 = rng.random::<f64>().max(1e-15);
                let u2: f64 = rng.random::<f64>();
                let noise = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                v + strength * noise
            })
            .collect();

        // L2-renormalise to preserve the original norm.
        let orig_norm = l2_norm(emb);
        let new_norm = l2_norm(&perturbed);
        if new_norm > 1e-15 && orig_norm > 1e-15 {
            let scale = orig_norm / new_norm;
            for x in perturbed.iter_mut() {
                *x *= scale;
            }
        }
        perturbed
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Text Preprocessing ──────────────────────────────────────────────────

    #[test]
    fn test_subword_encode_decode() {
        let mut vocab_fwd: HashMap<String, usize> = HashMap::new();
        vocab_fwd.insert("hello".to_string(), 1);
        vocab_fwd.insert("world".to_string(), 2);
        vocab_fwd.insert("[UNK]".to_string(), 0);
        vocab_fwd.insert("##o".to_string(), 3);

        let ids = SubwordTokenizer::encode("hello world", &vocab_fwd);
        assert_eq!(ids, vec![1, 2]);

        let vocab_rev: HashMap<usize, String> =
            vocab_fwd.iter().map(|(k, &v)| (v, k.clone())).collect();
        let decoded = SubwordTokenizer::decode(&ids, &vocab_rev);
        assert!(decoded.contains("hello") && decoded.contains("world"));
    }

    #[test]
    fn test_subword_continuation_token() {
        // Test that ## prefix tokens are merged correctly during decode.
        let vocab_rev: HashMap<usize, String> = [(1, "play".to_string()), (2, "##ing".to_string())]
            .into_iter()
            .collect();
        let decoded = SubwordTokenizer::decode(&[1, 2], &vocab_rev);
        assert_eq!(decoded, "playing");
    }

    #[test]
    fn test_char_ngram_extract() {
        let ngrams = CharacterNgram::extract_ngrams("cat", 2, 3);
        // Expected: bigrams + trigrams of "<cat>"
        // <cat> has chars ['<','c','a','t','>']
        // bigrams: <c, ca, at, t>   (5-1=4)
        // trigrams: <ca, cat, at>   (5-2=3)
        assert!(ngrams.len() >= 4);
        assert!(ngrams.contains(&"<c".to_string()));
        assert!(ngrams.contains(&"<ca".to_string()));
    }

    #[test]
    fn test_char_ngram_hash_range() {
        let bucket_size = 100_000;
        let h = CharacterNgram::hash_ngram("hello", bucket_size);
        assert!(h < bucket_size);
    }

    #[test]
    fn test_text_normalizer() {
        let result = TextNormalizer::normalize("Hello, World! How are You?");
        // Lowercase, punctuation removed, whitespace collapsed.
        assert!(!result.chars().any(|c| c.is_uppercase()));
        assert!(!result.contains(','));
        assert!(!result.contains('!'));
        assert!(!result.contains('?'));
    }

    #[test]
    fn test_text_normalizer_unicode() {
        let result = TextNormalizer::normalize("Café Résumé");
        // Should not crash and should be lowercase.
        assert!(!result.is_empty());
        assert!(!result.chars().any(|c| c.is_ascii_uppercase()));
    }

    #[test]
    fn test_data_collator_padding() {
        let seqs = vec![vec![1, 2, 3], vec![4, 5], vec![6]];
        let batch = DataCollator::collate(&seqs, 4, 0);
        assert_eq!(batch.len(), 3);
        for row in &batch {
            assert_eq!(row.len(), 4);
        }
        assert_eq!(batch[1], vec![4, 5, 0, 0]);
        assert_eq!(batch[2], vec![6, 0, 0, 0]);
    }

    #[test]
    fn test_data_collator_truncation() {
        let seqs = vec![vec![1, 2, 3, 4, 5]];
        let batch = DataCollator::collate(&seqs, 3, 0);
        assert_eq!(batch[0], vec![1, 2, 3]);
    }

    #[test]
    fn test_sentence_piece_vocab() {
        let corpus = vec!["hello world", "hello rust", "world peace"];
        let vocab = SentencePieceModel::build_vocab(&corpus, 20);
        // Vocabulary must be non-empty.
        assert!(!vocab.is_empty());
        // All log-probs must be ≤ 0.
        for &lp in vocab.values() {
            assert!(lp <= 0.0, "log-prob must be non-positive: {}", lp);
        }
    }

    #[test]
    fn test_sentence_piece_encode() {
        let corpus = vec!["hello world"];
        let vocab = SentencePieceModel::build_vocab(&corpus, 50);
        let tokens = SentencePieceModel::encode("hello", &vocab);
        // Must produce at least one token.
        assert!(!tokens.is_empty());
        // Concatenation of tokens must equal input.
        let reconstructed: String = tokens.concat();
        assert_eq!(reconstructed, "hello");
    }

    // ── Sequence Labeling ───────────────────────────────────────────────────

    #[test]
    fn test_crf_viterbi_length() {
        let emissions = vec![
            vec![0.1, 0.9, 0.2],
            vec![0.8, 0.1, 0.3],
            vec![0.2, 0.3, 0.7],
        ];
        let transitions = vec![
            vec![0.0, 0.1, -0.1],
            vec![0.1, 0.0, 0.2],
            vec![-0.1, 0.2, 0.0],
        ];
        let tags = CrfLayer::viterbi_decode(&emissions, &transitions);
        assert_eq!(tags.len(), 3);
        for &t in &tags {
            assert!(t < 3);
        }
    }

    #[test]
    fn test_crf_log_partition() {
        let emissions = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let transitions = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let log_z = CrfLayer::forward_backward(&emissions, &transitions);
        // Should be a finite number.
        assert!(log_z.is_finite());
    }

    #[test]
    fn test_crf_empty() {
        let tags = CrfLayer::viterbi_decode(&[], &[]);
        assert!(tags.is_empty());
        let lz = CrfLayer::forward_backward(&[], &[]);
        assert_eq!(lz, 0.0);
    }

    #[test]
    fn test_ner_model_bio_tags() {
        let model = NerModel::new(10, 8, 5).expect("NerModel creation failed");
        let tags = model.forward(&[0, 1, 2, 3]);
        assert_eq!(tags.len(), 4);
        for &t in &tags {
            assert!(t < 5, "tag {} out of range", t);
        }
    }

    #[test]
    fn test_pos_tagging() {
        let model = PosTagging::new(8, 4).expect("PosTagging creation failed");
        let tags = model.tag(&[0, 1, 2]);
        assert_eq!(tags.len(), 3);
        for &t in &tags {
            assert!(t < 4);
        }
    }

    #[test]
    fn test_chunking_decoder() {
        // B-1 I-1 O B-2 I-2 I-2
        // B-type1 = tag 1, I-type1 = tag 2, O = 0, B-type2 = tag 3, I-type2 = tag 4
        let tags = vec![1, 2, 0, 3, 4, 4];
        let spans = ChunkingDecoder::decode(&tags);
        assert_eq!(spans.len(), 2);
        // First span: positions 0..2, type 1.
        assert_eq!(spans[0], (0, 2, 1));
        // Second span: positions 3..6, type 2.
        assert_eq!(spans[1], (3, 6, 2));
    }

    #[test]
    fn test_chunking_decoder_empty() {
        let spans = ChunkingDecoder::decode(&[]);
        assert!(spans.is_empty());
    }

    #[test]
    fn test_constituency_parser() {
        let scores = vec![
            vec![5.0, 3.0, 1.0], // spans starting at 0: (0,0)=5, (0,1)=3, (0,2)=1
            vec![f64::NEG_INFINITY, 4.0, 2.0], // starts at 1
            vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 3.0], // starts at 2
        ];
        let spans = ConstituencyParser::parse(3, &scores);
        // Should have selected some spans.
        assert!(!spans.is_empty());
        // Spans should be non-overlapping.
        let mut covered = [false; 3];
        for (s, e) in &spans {
            for pos in *s..=*e {
                assert!(!covered[pos], "overlapping spans detected");
                covered[pos] = true;
            }
        }
    }

    // ── Semantic Understanding ───────────────────────────────────────────────

    #[test]
    fn test_sentence_encoder_shape() {
        let tokens = vec![vec![1.0, 0.0, 0.5], vec![0.0, 1.0, 0.5]];
        let emb = SentenceEncoder::encode(&tokens);
        assert_eq!(emb.len(), 3);
        // Mean of the two tokens.
        assert!((emb[0] - 0.5).abs() < 1e-10);
        assert!((emb[1] - 0.5).abs() < 1e-10);
        assert!((emb[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sentence_encoder_empty() {
        let emb = SentenceEncoder::encode(&[]);
        assert!(emb.is_empty());
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = SemanticSimilarity::cosine_sim(&v, &v);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = SemanticSimilarity::cosine_sim(&a, &b);
        assert!((sim).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        let sim = SemanticSimilarity::cosine_sim(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_textual_entailment_shape() {
        let model = TextualEntailment::new(4).expect("TextualEntailment creation failed");
        let p = vec![1.0, 0.0, 0.5, -0.5];
        let h = vec![0.5, 0.5, 0.0, 1.0];
        let probs = model.predict(&p, &h);
        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "probs sum={}", sum);
        for &p in &probs {
            assert!((0.0..=1.0).contains(&p));
        }
    }

    #[test]
    fn test_coref_score_range() {
        let resolver = CoReferenceResolver::new(4).expect("CoReferenceResolver creation failed");
        let m1 = vec![1.0, 0.0, 0.5, -0.5];
        let m2 = vec![-0.5, 1.0, 0.0, 0.5];
        let doc = vec![0.1, 0.2, 0.3, 0.4];
        let score = resolver.score_pair(&m1, &m2, &doc);
        assert!((0.0..=1.0).contains(&score), "score={}", score);
    }

    #[test]
    fn test_srl_output() {
        let srl = SemanticRoleLabeler::new(4).expect("SemanticRoleLabeler creation failed");
        let tokens = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let labels = srl.label(1, &tokens);
        assert_eq!(labels.len(), 3);
        for (_, role) in &labels {
            assert!(SemanticRoleLabeler::ROLES.contains(role));
        }
    }

    // ── Question Answering ───────────────────────────────────────────────────

    #[test]
    fn test_span_extraction_bounds() {
        let dim = 4;
        let qa = SpanExtractionQa::new(dim).expect("SpanExtractionQa creation failed");
        let question = vec![1.0, 0.0, 0.5, -0.5];
        // 5 context tokens, each of dimension 4.
        let context: Vec<f64> = (0..20).map(|i| (i as f64) * 0.1).collect();
        let (start, end) = qa.forward(&question, &context);
        assert!(start < 5, "start={}", start);
        assert!(end < 5, "end={}", end);
        assert!(end >= start, "end < start: {} < {}", end, start);
    }

    #[test]
    fn test_span_extraction_empty_context() {
        let qa = SpanExtractionQa::new(4).expect("SpanExtractionQa creation failed");
        let question = vec![1.0, 0.0, 0.5, -0.5];
        let (start, end) = qa.forward(&question, &[]);
        assert_eq!((start, end), (0, 0));
    }

    #[test]
    fn test_retriever_reader() {
        let rr = RetrieverReader::new(8).expect("RetrieverReader creation failed");
        let corpus = vec!["the cat sat on the mat", "dogs are great pets"];
        let (doc, start, end) = rr.answer("cat sat", &corpus);
        assert!(doc < 2, "doc={}", doc);
        assert!(end >= start, "end < start");
        // The retriever should prefer the first document (contains "cat" and "sat").
        assert_eq!(doc, 0);
    }

    #[test]
    fn test_multihop_qa_decompose() {
        let mhqa = MultiHopQa::new(4).expect("MultiHopQa creation failed");
        let q_emb = vec![1.0, -1.0, 0.5, 0.0];
        let sub_qs = mhqa.decompose(&q_emb);
        assert_eq!(sub_qs.len(), 2);
        // Each sub-question embedding should be non-empty.
        for sq in &sub_qs {
            assert!(!sq.is_empty());
        }
    }

    #[test]
    fn test_open_domain_qa() {
        let qdqa = OpenDomainQa::new(4).expect("OpenDomainQa creation failed");
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let docs = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let (best_doc, span_logits) = qdqa.predict(&q, &docs);
        assert!(best_doc < 2);
        assert!(!span_logits.is_empty());
    }

    #[test]
    fn test_triviaqa_exact_match() {
        let (em, f1) = TriviaQaEvaluator::eval("Albert Einstein", "Albert Einstein");
        assert!(em);
        assert!((f1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_triviaqa_f1() {
        let (em, f1) = TriviaQaEvaluator::eval("Albert Einstein physicist", "Albert Einstein");
        assert!(!em);
        assert!(f1 > 0.0 && f1 <= 1.0);
    }

    #[test]
    fn test_triviaqa_no_overlap() {
        let (em, f1) = TriviaQaEvaluator::eval("foo bar", "baz qux");
        assert!(!em);
        assert_eq!(f1, 0.0);
    }

    // ── Text Generation & Summarization ─────────────────────────────────────

    #[test]
    fn test_extractive_summarizer_k() {
        let sentences = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.5, 0.5, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let selected = ExtractiveSummarizer::summarize(&sentences, 2);
        assert_eq!(selected.len(), 2);
        // Indices should be valid and sorted.
        for &idx in &selected {
            assert!(idx < 4);
        }
        assert!(selected[0] <= selected[1]);
    }

    #[test]
    fn test_extractive_summarizer_k_larger_than_n() {
        let sentences = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let selected = ExtractiveSummarizer::summarize(&sentences, 10);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_abstractive_summarizer_shape() {
        let summ = AbstractiveSummarizer::new(4).expect("AbstractiveSummarizer creation failed");
        let src = vec![vec![1.0, 0.0, 0.5, -0.5], vec![-0.5, 1.0, 0.0, 0.5]];
        let tgt = vec![vec![0.1, 0.2, 0.3, 0.4]];
        let out = summ.forward(&src, &tgt);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 4);
    }

    #[test]
    fn test_rouge1_identical() {
        let tokens = vec![1usize, 2, 3, 4];
        let score = RougeMetric::compute_rouge1(&tokens, &tokens);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rouge1_no_overlap() {
        let pred = vec![1usize, 2];
        let gold = vec![3usize, 4];
        let score = RougeMetric::compute_rouge1(&pred, &gold);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_rouge2() {
        let pred = vec![1usize, 2, 3];
        let gold = vec![1usize, 2, 4];
        let score = RougeMetric::compute_rouge2(&pred, &gold);
        // Shared bigrams: (1,2) → 1 out of pred_bg={1,2},{2,3} and gold_bg={1,2},{2,4}.
        assert!(score > 0.0 && score < 1.0);
    }

    #[test]
    fn test_rouge_l_subsequence() {
        let pred = vec![1usize, 2, 3, 4, 5];
        let gold = vec![1usize, 3, 5];
        let score = RougeMetric::rouge_l(&pred, &gold);
        // LCS length is 3 ([1,3,5]).
        assert!(score > 0.0 && score <= 1.0);
        // Precision: 3/5, Recall: 3/3=1.0 → F1 = 2*(0.6*1.0)/(0.6+1.0) = 0.75.
        assert!((score - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_rouge_l_identical() {
        let tokens = vec![1usize, 2, 3];
        assert!((RougeMetric::rouge_l(&tokens, &tokens) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bleurt_proxy_range() {
        let pred = vec![vec![1.0, 0.0], vec![0.5, 0.5]];
        let gold = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let score = BleurtProxy::compute(&pred, &gold);
        assert!((0.0..=1.0).contains(&score), "score={}", score);
    }

    #[test]
    fn test_bleurt_proxy_identical() {
        let embs = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let score = BleurtProxy::compute(&embs, &embs);
        assert!((score - 1.0).abs() < 1e-9, "score={}", score);
    }

    #[test]
    fn test_paraphraser_shape() {
        let emb = vec![1.0, 0.5, -0.3, 0.8];
        let mut rng = StdRng::seed_from_u64(123);
        let para = ParaphraserModel::paraphrase(&emb, 0.1, &mut rng);
        assert_eq!(para.len(), emb.len());
    }

    #[test]
    fn test_paraphraser_norm_preserved() {
        let emb = vec![3.0, 4.0]; // norm = 5.0
        let mut rng = StdRng::seed_from_u64(42);
        let para = ParaphraserModel::paraphrase(&emb, 0.1, &mut rng);
        let orig_norm = l2_norm(&emb);
        let para_norm = l2_norm(&para);
        assert!(
            (orig_norm - para_norm).abs() < 1e-9,
            "norms differ: {} vs {}",
            orig_norm,
            para_norm
        );
    }

    #[test]
    fn test_paraphraser_not_identical() {
        // With strength > 0 the paraphrase should differ from the original.
        let emb = vec![1.0, 0.0, 0.0, 0.0];
        let mut rng = StdRng::seed_from_u64(7);
        let para = ParaphraserModel::paraphrase(&emb, 1.0, &mut rng);
        let diff: f64 = emb
            .iter()
            .zip(para.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-10, "paraphrase should differ from original");
    }
}
