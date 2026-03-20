//! Vocabulary utilities, special tokens, padding helpers, and tokenizer configuration.

/// Configuration for a tokenizer, describing special tokens and vocabulary limits.
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Target vocabulary size.
    pub vocab_size: usize,
    /// Unknown token string.
    pub unk_token: String,
    /// Beginning-of-sequence token string.
    pub bos_token: String,
    /// End-of-sequence token string.
    pub eos_token: String,
    /// Padding token string.
    pub pad_token: String,
    /// Optional maximum sequence length.
    pub max_length: Option<usize>,
}

impl TokenizerConfig {
    /// Create a GPT-style configuration.
    ///
    /// GPT models typically use a single `<|endoftext|>` token for both BOS and EOS,
    /// have no explicit padding token, and use `<|unk|>` for unknown sequences.
    pub fn default_gpt_style() -> Self {
        Self {
            vocab_size: 50257,
            unk_token: "<|unk|>".to_string(),
            bos_token: "<|endoftext|>".to_string(),
            eos_token: "<|endoftext|>".to_string(),
            pad_token: "<|endoftext|>".to_string(),
            max_length: Some(1024),
        }
    }

    /// Create a BERT-style configuration.
    ///
    /// BERT models use `[UNK]`, `[CLS]` (BOS), `[SEP]` (EOS), `[PAD]`, and `[MASK]`
    /// as special tokens.
    pub fn default_bert_style() -> Self {
        Self {
            vocab_size: 30522,
            unk_token: "[UNK]".to_string(),
            bos_token: "[CLS]".to_string(),
            eos_token: "[SEP]".to_string(),
            pad_token: "[PAD]".to_string(),
            max_length: Some(512),
        }
    }
}

/// Pad or truncate a sequence of token IDs to a fixed `length`.
///
/// - If `ids.len() >= length`, the sequence is truncated to `length`.
/// - If `ids.len() < length`, the sequence is right-padded with `pad_id`.
pub fn pad_sequence(ids: &[u32], length: usize, pad_id: u32) -> Vec<u32> {
    if ids.len() >= length {
        ids[..length].to_vec()
    } else {
        let mut result = ids.to_vec();
        result.resize(length, pad_id);
        result
    }
}

/// Create an attention mask for a padded sequence.
///
/// Returns a `Vec<u8>` of the same length as `ids`:
/// - `1` for real tokens (ids that differ from `pad_id`).
/// - `0` for padding tokens.
pub fn attention_mask(ids: &[u32], pad_id: u32) -> Vec<u8> {
    ids.iter()
        .map(|&id| if id != pad_id { 1u8 } else { 0u8 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_sequence_extends() {
        let ids = vec![1u32, 2, 3];
        let padded = pad_sequence(&ids, 5, 0);
        assert_eq!(padded, vec![1, 2, 3, 0, 0]);
    }

    #[test]
    fn test_pad_sequence_truncates() {
        let ids = vec![1u32, 2, 3, 4, 5];
        let padded = pad_sequence(&ids, 3, 0);
        assert_eq!(padded, vec![1, 2, 3]);
    }

    #[test]
    fn test_pad_sequence_exact_length() {
        let ids = vec![1u32, 2, 3];
        let padded = pad_sequence(&ids, 3, 0);
        assert_eq!(padded, vec![1, 2, 3]);
    }

    #[test]
    fn test_pad_sequence_empty() {
        let ids: Vec<u32> = vec![];
        let padded = pad_sequence(&ids, 4, 99);
        assert_eq!(padded, vec![99, 99, 99, 99]);
    }

    #[test]
    fn test_attention_mask_no_padding() {
        let ids = vec![1u32, 2, 3];
        let mask = attention_mask(&ids, 0);
        assert_eq!(mask, vec![1, 1, 1]);
    }

    #[test]
    fn test_attention_mask_with_padding() {
        let ids = vec![1u32, 2, 0, 0];
        let mask = attention_mask(&ids, 0);
        assert_eq!(mask, vec![1, 1, 0, 0]);
    }

    #[test]
    fn test_attention_mask_all_padding() {
        let ids = vec![0u32, 0, 0];
        let mask = attention_mask(&ids, 0);
        assert_eq!(mask, vec![0, 0, 0]);
    }

    #[test]
    fn test_tokenizer_config_gpt_style() {
        let cfg = TokenizerConfig::default_gpt_style();
        assert_eq!(cfg.eos_token, "<|endoftext|>");
        assert_eq!(cfg.bos_token, "<|endoftext|>");
        assert_eq!(cfg.vocab_size, 50257);
        assert!(cfg.max_length.is_some());
    }

    #[test]
    fn test_tokenizer_config_bert_style() {
        let cfg = TokenizerConfig::default_bert_style();
        assert_eq!(cfg.unk_token, "[UNK]");
        assert_eq!(cfg.bos_token, "[CLS]");
        assert_eq!(cfg.eos_token, "[SEP]");
        assert_eq!(cfg.pad_token, "[PAD]");
        assert_eq!(cfg.vocab_size, 30522);
    }
}
