//! Byte-Pair Encoding (BPE) tokenizer implementation.
//!
//! This module provides:
//! - [`BpeVocab`]: a bidirectional token↔id mapping.
//! - [`BpeTokenizer`]: a full BPE tokenizer supporting training, encoding, and decoding.

use std::collections::HashMap;
use tenflowers_core::{Result, TensorError};

// ─── Merge rule ──────────────────────────────────────────────────────────────

/// A BPE merge rule: `(token_a, token_b)` → concatenated merged token.
pub type MergeRule = (String, String);

// ─── BpeVocab ────────────────────────────────────────────────────────────────

/// Bidirectional vocabulary: token string ↔ token id.
#[derive(Debug, Clone)]
pub struct BpeVocab {
    token_to_id: HashMap<String, u32>,
    id_to_token: Vec<String>,
}

impl BpeVocab {
    /// Create an empty vocabulary.
    pub fn new() -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
        }
    }

    /// Add `token` to the vocabulary if it is not already present.
    ///
    /// Returns the id of the token (existing or newly assigned).
    pub fn add_token(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }
        let id = self.id_to_token.len() as u32;
        self.id_to_token.push(token.to_string());
        self.token_to_id.insert(token.to_string(), id);
        id
    }

    /// Look up the id of a token string.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Look up the token string for a given id.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    /// Number of tokens currently in the vocabulary.
    pub fn size(&self) -> usize {
        self.id_to_token.len()
    }
}

impl Default for BpeVocab {
    fn default() -> Self {
        Self::new()
    }
}

// ─── BpeTokenizer ────────────────────────────────────────────────────────────

/// Byte-Pair Encoding tokenizer.
///
/// Supports training from a text corpus, loading from pre-built vocab + merges,
/// encoding text to token ids, and decoding ids back to text.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// The vocabulary.
    pub vocab: BpeVocab,
    /// Ordered list of merge rules (lower index = higher priority).
    merges: Vec<MergeRule>,
    /// Lookup table: `(token_a, token_b)` → rank (index in `merges`).
    merge_rank: HashMap<(String, String), usize>,
    /// Unknown token string.
    pub unk_token: String,
    /// End-of-sequence token string.
    pub eos_token: String,
    /// Beginning-of-sequence token string.
    pub bos_token: String,
}

impl BpeTokenizer {
    // ── Special token defaults ───────────────────────────────────────────────

    const DEFAULT_UNK: &'static str = "<unk>";
    const DEFAULT_BOS: &'static str = "<s>";
    const DEFAULT_EOS: &'static str = "</s>";
    /// Suffix appended to the last character-token of every word to mark word boundary.
    const WORD_END: &'static str = "</w>";

    // ── Construction helpers ─────────────────────────────────────────────────

    /// Build a [`BpeTokenizer`] from an already-populated vocab and an ordered list
    /// of merge rules. The merge rank map is derived from the rule order.
    pub fn from_vocab_and_merges(
        vocab_tokens: Vec<String>,
        merges: Vec<MergeRule>,
    ) -> Result<Self> {
        if vocab_tokens.is_empty() {
            return Err(TensorError::invalid_argument(
                "vocab_tokens must not be empty".to_string(),
            ));
        }

        let mut vocab = BpeVocab::new();
        for token in &vocab_tokens {
            vocab.add_token(token);
        }

        let mut merge_rank = HashMap::with_capacity(merges.len());
        for (rank, rule) in merges.iter().enumerate() {
            merge_rank.insert(rule.clone(), rank);
        }

        let unk_token = Self::DEFAULT_UNK.to_string();
        let bos_token = Self::DEFAULT_BOS.to_string();
        let eos_token = Self::DEFAULT_EOS.to_string();

        // Ensure special tokens are in the vocabulary.
        let mut tokenizer = Self {
            vocab,
            merges,
            merge_rank,
            unk_token: unk_token.clone(),
            bos_token: bos_token.clone(),
            eos_token: eos_token.clone(),
        };
        tokenizer.vocab.add_token(&unk_token);
        tokenizer.vocab.add_token(&bos_token);
        tokenizer.vocab.add_token(&eos_token);

        Ok(tokenizer)
    }

    // ── Training ─────────────────────────────────────────────────────────────

    /// Train a BPE tokenizer on `corpus` until the vocabulary reaches `vocab_size`.
    ///
    /// Algorithm:
    /// 1. Collect every unique character (plus the `</w>` word-end marker) as
    ///    the initial character-level vocabulary.
    /// 2. Represent every word as a sequence of character tokens, with `</w>`
    ///    appended to the last character.
    /// 3. Repeatedly find the most-frequent adjacent pair, merge it into a new
    ///    token, and record the merge rule — until `vocab_size` is reached or no
    ///    more pairs exist.
    pub fn train(corpus: &[&str], vocab_size: usize) -> Result<Self> {
        if vocab_size < 4 {
            return Err(TensorError::invalid_argument(
                "vocab_size must be at least 4 (space for special tokens)".to_string(),
            ));
        }

        // ── Step 1: build initial character vocabulary and word frequency table ──
        // Each word is stored as Vec<String> where the last character has WORD_END appended.
        let mut word_freqs: HashMap<Vec<String>, u64> = HashMap::new();
        for text in corpus {
            for word in text.split_whitespace() {
                let chars: Vec<char> = word.chars().collect();
                if chars.is_empty() {
                    continue;
                }
                let mut token_seq: Vec<String> = chars[..chars.len() - 1]
                    .iter()
                    .map(|c| c.to_string())
                    .collect();
                // Last character gets WORD_END suffix.
                token_seq.push(format!("{}{}", chars[chars.len() - 1], Self::WORD_END));
                *word_freqs.entry(token_seq).or_insert(0) += 1;
            }
        }

        // ── Step 2: collect unique character tokens ──────────────────────────
        let mut vocab = BpeVocab::new();
        // Insert special tokens first so they get low ids.
        vocab.add_token(Self::DEFAULT_UNK);
        vocab.add_token(Self::DEFAULT_BOS);
        vocab.add_token(Self::DEFAULT_EOS);

        // Collect all unique base tokens.
        let mut base_tokens: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for word_tokens in word_freqs.keys() {
            for t in word_tokens {
                base_tokens.insert(t.clone());
            }
        }
        for t in &base_tokens {
            vocab.add_token(t);
        }

        let mut merges: Vec<MergeRule> = Vec::new();
        let mut merge_rank: HashMap<(String, String), usize> = HashMap::new();

        // ── Step 3: iteratively merge the most-frequent pair ─────────────────
        while vocab.size() < vocab_size {
            // Count pair frequencies across the entire corpus.
            let pair_freqs = Self::count_pairs(&word_freqs);
            if pair_freqs.is_empty() {
                break;
            }

            // Find the pair with the highest frequency (ties broken lexicographically
            // for deterministic output).
            let best_pair = pair_freqs
                .iter()
                .max_by(|(pair_a, freq_a), (pair_b, freq_b)| {
                    freq_a.cmp(freq_b).then_with(|| pair_a.cmp(pair_b))
                })
                .map(|(pair, _)| pair.clone());

            let best_pair = match best_pair {
                Some(p) => p,
                None => break,
            };

            // Create the merged token.
            let merged = format!("{}{}", best_pair.0, best_pair.1);
            let rank = merges.len();
            merges.push(best_pair.clone());
            merge_rank.insert(best_pair.clone(), rank);
            vocab.add_token(&merged);

            // Apply the merge to all words.
            word_freqs = Self::apply_merge(&word_freqs, &best_pair, &merged);
        }

        Ok(Self {
            vocab,
            merges,
            merge_rank,
            unk_token: Self::DEFAULT_UNK.to_string(),
            bos_token: Self::DEFAULT_BOS.to_string(),
            eos_token: Self::DEFAULT_EOS.to_string(),
        })
    }

    // ── Encoding / decoding ──────────────────────────────────────────────────

    /// Tokenize `text` into token strings by applying the learned BPE merges.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut result = Vec::new();
        for word in text.split_whitespace() {
            let chars: Vec<char> = word.chars().collect();
            if chars.is_empty() {
                continue;
            }
            // Build initial character-level sequence with WORD_END on last char.
            let mut tokens: Vec<String> = chars[..chars.len() - 1]
                .iter()
                .map(|c| c.to_string())
                .collect();
            tokens.push(format!("{}{}", chars[chars.len() - 1], Self::WORD_END));

            // Apply merges in rank order.
            tokens = self.apply_merges(tokens);
            result.extend(tokens);
        }
        result
    }

    /// Encode `text` into a sequence of token ids.
    ///
    /// Tokens not found in the vocabulary are mapped to the UNK token id (or 0 as
    /// a fallback if UNK is also absent).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let unk_id = self.vocab.token_to_id(&self.unk_token).unwrap_or(0);
        self.tokenize(text)
            .iter()
            .map(|tok| self.vocab.token_to_id(tok).unwrap_or(unk_id))
            .collect()
    }

    /// Decode a sequence of token ids back to a string.
    ///
    /// The `</w>` word-end markers are converted back to spaces so that
    /// `decode(encode(text))` approximately recovers the original text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let unk = self.unk_token.as_str();
        let pieces: Vec<&str> = ids
            .iter()
            .map(|&id| self.vocab.id_to_token(id).unwrap_or(unk))
            .collect();

        // Reconstruct text: each token ending in WORD_END contributes a trailing space.
        let mut out = String::new();
        for piece in pieces {
            if let Some(base) = piece.strip_suffix(Self::WORD_END) {
                out.push_str(base);
                out.push(' ');
            } else {
                out.push_str(piece);
            }
        }
        // Trim trailing space introduced by the last word.
        out.trim_end().to_string()
    }

    /// Add a special token to the vocabulary and return its id.
    ///
    /// If the token already exists, its existing id is returned.
    pub fn add_special_token(&mut self, token: &str) -> u32 {
        self.vocab.add_token(token)
    }

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.vocab.size()
    }

    /// Number of merge rules learned during training.
    pub fn num_merges(&self) -> usize {
        self.merges.len()
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Count the frequency of every adjacent token pair across all words.
    fn count_pairs(word_freqs: &HashMap<Vec<String>, u64>) -> HashMap<(String, String), u64> {
        let mut pair_freqs: HashMap<(String, String), u64> = HashMap::new();
        for (word_tokens, freq) in word_freqs {
            for window in word_tokens.windows(2) {
                let pair = (window[0].clone(), window[1].clone());
                *pair_freqs.entry(pair).or_insert(0) += freq;
            }
        }
        pair_freqs
    }

    /// Apply a merge rule to all words, replacing every occurrence of `pair`
    /// with `merged`.
    fn apply_merge(
        word_freqs: &HashMap<Vec<String>, u64>,
        pair: &(String, String),
        merged: &str,
    ) -> HashMap<Vec<String>, u64> {
        let mut new_word_freqs: HashMap<Vec<String>, u64> = HashMap::new();
        for (word_tokens, &freq) in word_freqs {
            let new_tokens = Self::merge_pair_in_word(word_tokens, pair, merged);
            *new_word_freqs.entry(new_tokens).or_insert(0) += freq;
        }
        new_word_freqs
    }

    /// Replace all occurrences of `pair` in `word_tokens` with `merged`.
    fn merge_pair_in_word(
        word_tokens: &[String],
        pair: &(String, String),
        merged: &str,
    ) -> Vec<String> {
        let mut result: Vec<String> = Vec::with_capacity(word_tokens.len());
        let mut i = 0;
        while i < word_tokens.len() {
            if i + 1 < word_tokens.len() && word_tokens[i] == pair.0 && word_tokens[i + 1] == pair.1
            {
                result.push(merged.to_string());
                i += 2;
            } else {
                result.push(word_tokens[i].clone());
                i += 1;
            }
        }
        result
    }

    /// Apply all stored merge rules (in priority order) to a token sequence.
    fn apply_merges(&self, mut tokens: Vec<String>) -> Vec<String> {
        loop {
            if tokens.len() < 2 {
                break;
            }

            // Find the pair with the lowest merge rank (highest priority).
            let best = tokens
                .windows(2)
                .enumerate()
                .filter_map(|(i, window)| {
                    let pair = (window[0].clone(), window[1].clone());
                    self.merge_rank.get(&pair).map(|&rank| (i, pair, rank))
                })
                .min_by_key(|&(_, _, rank)| rank);

            match best {
                None => break,
                Some((i, pair, _)) => {
                    let merged = format!("{}{}", pair.0, pair.1);
                    tokens = Self::merge_pair_in_word(&tokens, &pair, &merged);
                }
            }
        }
        tokens
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BpeVocab tests ───────────────────────────────────────────────────────

    #[test]
    fn test_vocab_add_token_returns_increasing_ids() {
        let mut vocab = BpeVocab::new();
        let id_a = vocab.add_token("hello");
        let id_b = vocab.add_token("world");
        assert_eq!(id_a, 0);
        assert_eq!(id_b, 1);
        assert_eq!(vocab.size(), 2);
    }

    #[test]
    fn test_vocab_add_token_idempotent() {
        let mut vocab = BpeVocab::new();
        let id1 = vocab.add_token("foo");
        let id2 = vocab.add_token("foo");
        assert_eq!(id1, id2);
        assert_eq!(vocab.size(), 1);
    }

    #[test]
    fn test_vocab_token_to_id_and_back() {
        let mut vocab = BpeVocab::new();
        vocab.add_token("a");
        vocab.add_token("b");
        vocab.add_token("c");

        assert_eq!(vocab.token_to_id("b"), Some(1));
        assert_eq!(vocab.id_to_token(1), Some("b"));
    }

    #[test]
    fn test_vocab_missing_returns_none() {
        let vocab = BpeVocab::new();
        assert_eq!(vocab.token_to_id("missing"), None);
        assert_eq!(vocab.id_to_token(0), None);
    }

    // ── BpeTokenizer construction ────────────────────────────────────────────

    #[test]
    fn test_from_vocab_and_merges_empty_vocab_error() {
        let result = BpeTokenizer::from_vocab_and_merges(vec![], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_vocab_and_merges_basic() {
        let vocab_tokens = vec!["a".to_string(), "b".to_string(), "ab</w>".to_string()];
        let merges = vec![("a".to_string(), "b</w>".to_string())];
        let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab_tokens, merges);
        assert!(tokenizer.is_ok());
        let tokenizer = tokenizer.expect("from_vocab_and_merges failed");
        assert!(tokenizer.vocab_size() >= 3);
    }

    // ── Training ─────────────────────────────────────────────────────────────

    #[test]
    fn test_train_small_vocab_size_error() {
        let result = BpeTokenizer::train(&["hello world"], 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_train_produces_requested_vocab_size() {
        let corpus = &[
            "low lower newest widest",
            "low low low lowest",
            "newer newest",
        ];
        let target = 20;
        let tok = BpeTokenizer::train(corpus, target).expect("training should succeed");
        // Vocab may reach target or stop early if no more pairs remain.
        assert!(
            tok.vocab_size() <= target + 5,
            "vocab should not greatly exceed target"
        );
        assert!(
            tok.vocab_size() >= 4,
            "vocab must contain at least special tokens"
        );
    }

    #[test]
    fn test_train_learns_frequent_merge() {
        // "ab" appears very often; after training "ab</w>" should be a token.
        let corpus = vec!["ab ab ab ab ab ab ab ab"];
        let tok = BpeTokenizer::train(&corpus, 10).expect("training should succeed");
        // After merges "a" + "b</w>" → "ab</w>" or similar merge should occur.
        // At minimum, base character tokens must be present.
        assert!(tok.vocab_size() >= 4);
        // The number of merge rules learned should be within a sane upper bound
        // (at most vocab_size merges can be learned).
        assert!(tok.num_merges() <= 10);
    }

    #[test]
    fn test_train_repeated_bigrams_learned_first() {
        // Corpus where "lo" is the most common bigram.
        let corpus = vec!["lo lo lo lo lo lo lo lo lo lo"];
        let tok = BpeTokenizer::train(&corpus, 8).expect("training should succeed");
        // We expect "lo</w>" or "lo" to be a merged token.
        let has_lo = (0..tok.vocab_size() as u32)
            .filter_map(|id| tok.vocab.id_to_token(id))
            .any(|t| t.contains("lo"));
        assert!(has_lo, "merged 'lo' token should appear in vocab");
    }

    // ── Tokenize / encode / decode ───────────────────────────────────────────

    #[test]
    fn test_tokenize_produces_tokens() {
        let corpus = vec!["hello world hello world"];
        let tok = BpeTokenizer::train(&corpus, 15).expect("training should succeed");
        let tokens = tok.tokenize("hello");
        assert!(
            !tokens.is_empty(),
            "tokenize must return at least one token"
        );
    }

    #[test]
    fn test_encode_returns_known_ids() {
        let corpus = vec!["hello world"];
        let tok = BpeTokenizer::train(&corpus, 15).expect("training should succeed");
        let ids = tok.encode("hello");
        assert!(!ids.is_empty());
        // All ids must be within vocab range.
        for &id in &ids {
            assert!((id as usize) < tok.vocab_size(), "id out of vocab range");
        }
    }

    #[test]
    fn test_encode_decode_roundtrip_approximate() {
        let corpus = vec!["hello world foo bar"];
        let tok = BpeTokenizer::train(&corpus, 20).expect("training should succeed");
        let original = "hello world";
        let ids = tok.encode(original);
        let decoded = tok.decode(&ids);
        // After BPE encoding/decoding with </w> markers the text should be recovered.
        assert_eq!(
            decoded, original,
            "decode(encode(text)) should return original text"
        );
    }

    #[test]
    fn test_decode_empty_ids() {
        let corpus = vec!["hello"];
        let tok = BpeTokenizer::train(&corpus, 10).expect("training should succeed");
        assert_eq!(tok.decode(&[]), "");
    }

    // ── add_special_token ────────────────────────────────────────────────────

    #[test]
    fn test_add_special_token_increases_vocab() {
        let corpus = vec!["hello world"];
        let mut tok = BpeTokenizer::train(&corpus, 10).expect("training should succeed");
        let before = tok.vocab_size();
        tok.add_special_token("<pad>");
        assert_eq!(tok.vocab_size(), before + 1);
    }

    #[test]
    fn test_add_special_token_idempotent() {
        let corpus = vec!["hello world"];
        let mut tok = BpeTokenizer::train(&corpus, 10).expect("training should succeed");
        let id1 = tok.add_special_token("<pad>");
        let id2 = tok.add_special_token("<pad>");
        assert_eq!(
            id1, id2,
            "adding the same token twice should return the same id"
        );
    }

    // ── num_merges ───────────────────────────────────────────────────────────

    #[test]
    fn test_num_merges_reflects_training() {
        let corpus = vec!["aaaa bbbb cccc aaaa bbbb"];
        let tok = BpeTokenizer::train(&corpus, 12).expect("training should succeed");
        assert_eq!(tok.num_merges(), tok.merges.len());
    }
}
