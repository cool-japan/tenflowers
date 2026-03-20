//! Information extraction, dense retrieval, and question answering for documents.

// ─── InformationExtractor ────────────────────────────────────────────────────

/// Entity type for document NER.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityType {
    Date,
    Amount,
    Name,
    Address,
    Phone,
    Email,
    TaxId,
    InvoiceNumber,
    Other,
}

impl EntityType {
    /// Return string label.
    pub fn label(&self) -> &'static str {
        match self {
            EntityType::Date => "DATE",
            EntityType::Amount => "AMOUNT",
            EntityType::Name => "NAME",
            EntityType::Address => "ADDRESS",
            EntityType::Phone => "PHONE",
            EntityType::Email => "EMAIL",
            EntityType::TaxId => "TAX_ID",
            EntityType::InvoiceNumber => "INVOICE_NUMBER",
            EntityType::Other => "OTHER",
        }
    }
}

/// A recognized named entity in a document.
#[derive(Debug, Clone)]
pub struct DocEntity {
    /// Type of the entity
    pub entity_type: EntityType,
    /// Surface text
    pub text: String,
    /// Character-level start offset in source text
    pub start: usize,
    /// Character-level end offset (exclusive) in source text
    pub end: usize,
    /// Confidence score
    pub confidence: f32,
}

impl DocEntity {
    /// Create a new DocEntity.
    pub fn new(
        entity_type: EntityType,
        text: impl Into<String>,
        start: usize,
        end: usize,
        confidence: f32,
    ) -> Self {
        Self {
            entity_type,
            text: text.into(),
            start,
            end,
            confidence,
        }
    }
}

/// Rule-based information extractor for document entities.
pub struct InformationExtractor;

impl InformationExtractor {
    /// Create a new extractor.
    pub fn new() -> Self {
        Self
    }

    /// Extract date entities using a state machine for common formats:
    /// YYYY-MM-DD, DD/MM/YYYY, DD.MM.YYYY, MM/DD/YYYY.
    pub fn extract_dates(text: &str) -> Vec<DocEntity> {
        let chars: Vec<char> = text.chars().collect();
        let bytes = text.as_bytes();
        let mut results = Vec::new();
        let n = bytes.len();
        let mut i = 0;

        while i < n {
            // Try YYYY-MM-DD or YYYY/MM/DD
            if i + 9 < n && Self::is_digit4(&chars, i) {
                let sep = chars.get(i + 4).copied().unwrap_or(' ');
                if (sep == '-' || sep == '/') && Self::is_digit2(&chars, i + 5) {
                    let sep2 = chars.get(i + 7).copied().unwrap_or(' ');
                    if sep2 == sep && Self::is_digit2(&chars, i + 8) {
                        let end = i + 10;
                        let span = Self::char_range_to_byte_range(text, i, end);
                        results.push(DocEntity::new(
                            EntityType::Date,
                            &text[span.0..span.1],
                            span.0,
                            span.1,
                            0.95,
                        ));
                        i = end;
                        continue;
                    }
                }
            }
            // Try DD/MM/YYYY or MM/DD/YYYY or DD.MM.YYYY
            if i + 9 < n && Self::is_digit2(&chars, i) {
                let sep = chars.get(i + 2).copied().unwrap_or(' ');
                if (sep == '/' || sep == '.') && Self::is_digit2(&chars, i + 3) {
                    let sep2 = chars.get(i + 5).copied().unwrap_or(' ');
                    if sep2 == sep && Self::is_digit4(&chars, i + 6) {
                        let end = i + 10;
                        let span = Self::char_range_to_byte_range(text, i, end);
                        results.push(DocEntity::new(
                            EntityType::Date,
                            &text[span.0..span.1],
                            span.0,
                            span.1,
                            0.90,
                        ));
                        i = end;
                        continue;
                    }
                }
            }
            i += 1;
        }
        results
    }

    /// Extract currency amount entities.
    pub fn extract_amounts(text: &str) -> Vec<DocEntity> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mut results = Vec::new();
        let mut i = 0;

        while i < n {
            let ch = chars[i];
            let has_symbol = matches!(ch, '$' | '€' | '£' | '¥');
            let start_idx = i;
            if has_symbol {
                i += 1;
            }

            let num_start = i;
            if has_symbol && i < n && chars[i] == ' ' {
                i += 1;
            }

            if i < n && chars[i].is_ascii_digit() {
                let num_begin = i;
                while i < n && (chars[i].is_ascii_digit() || chars[i] == ',' || chars[i] == '_') {
                    i += 1;
                }
                if i + 1 < n && chars[i] == '.' && chars[i + 1].is_ascii_digit() {
                    i += 1;
                    while i < n && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                let _has_suffix = if i < n && matches!(chars[i], '$' | '€' | '£' | '¥') {
                    i += 1;
                    true
                } else {
                    false
                };

                if i > num_begin {
                    let span = Self::char_range_to_byte_range(text, start_idx, i);
                    results.push(DocEntity::new(
                        EntityType::Amount,
                        &text[span.0..span.1],
                        span.0,
                        span.1,
                        0.88,
                    ));
                }
                let _ = num_start;
            } else if has_symbol {
                i = start_idx + 1;
            } else {
                i = start_idx + 1;
            }
        }
        results
    }

    /// Extract email addresses using a state machine.
    pub fn extract_emails(text: &str) -> Vec<DocEntity> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mut results = Vec::new();
        let mut i = 0;

        while i < n {
            if chars[i] != '@' {
                i += 1;
                continue;
            }
            let at = i;
            let mut local_start = at;
            let mut j = at as isize - 1;
            while j >= 0 {
                let c = chars[j as usize];
                if c.is_alphanumeric() || matches!(c, '.' | '_' | '-' | '+') {
                    local_start = j as usize;
                    j -= 1;
                } else {
                    break;
                }
            }
            if local_start == at {
                i += 1;
                continue;
            }
            let mut domain_end = at + 1;
            while domain_end < n {
                let c = chars[domain_end];
                if c.is_alphanumeric() || matches!(c, '.' | '-') {
                    domain_end += 1;
                } else {
                    break;
                }
            }
            let domain: String = chars[at + 1..domain_end].iter().collect();
            if domain.contains('.') && domain_end > at + 3 {
                let span = Self::char_range_to_byte_range(text, local_start, domain_end);
                results.push(DocEntity::new(
                    EntityType::Email,
                    &text[span.0..span.1],
                    span.0,
                    span.1,
                    0.97,
                ));
            }
            i = domain_end;
        }
        results
    }

    fn is_digit4(chars: &[char], pos: usize) -> bool {
        (pos..pos + 4).all(|k| chars.get(k).map(|c| c.is_ascii_digit()).unwrap_or(false))
    }

    fn is_digit2(chars: &[char], pos: usize) -> bool {
        (pos..pos + 2).all(|k| chars.get(k).map(|c| c.is_ascii_digit()).unwrap_or(false))
    }

    fn char_range_to_byte_range(text: &str, char_start: usize, char_end: usize) -> (usize, usize) {
        let mut byte_start = text.len();
        let mut byte_end = text.len();
        for (ci, (bi, _)) in text.char_indices().enumerate() {
            if ci == char_start {
                byte_start = bi;
            }
            if ci == char_end {
                byte_end = bi;
                break;
            }
        }
        (byte_start, byte_end)
    }
}

impl Default for InformationExtractor {
    fn default() -> Self {
        Self::new()
    }
}

// ─── DocumentEmbedder ────────────────────────────────────────────────────────

/// Pooling strategy for document embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Use the \[CLS\] token (first token) embedding.
    CLS,
    /// Mean-pool all token embeddings.
    MeanPool,
    /// Max-pool across token embeddings (element-wise max).
    MaxPool,
    /// Attention-weighted pool.
    AttentionPool,
}

/// Configuration for document embedder.
#[derive(Debug, Clone)]
pub struct DocEmbedderConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Output embedding dimension
    pub embed_dim: usize,
    /// Pooling strategy
    pub pool: PoolingStrategy,
}

impl DocEmbedderConfig {
    /// Create a new configuration.
    pub fn new(vocab_size: usize, embed_dim: usize, pool: PoolingStrategy) -> Self {
        Self {
            vocab_size,
            embed_dim,
            pool,
        }
    }
}

/// Dense retrieval document embedder.
pub struct DocEmbedder {
    /// Token embedding table [vocab_size x embed_dim]
    pub token_embed: Vec<f32>,
    /// Attention pool query vector \[embed_dim\]
    pub attn_query: Vec<f32>,
    /// Configuration
    pub config: DocEmbedderConfig,
}

impl DocEmbedder {
    /// Create a new DocEmbedder.
    pub fn new(config: DocEmbedderConfig) -> Self {
        let d = config.embed_dim;
        let scale = 0.02_f32;
        let sinit =
            |n: usize| -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.09).sin()) * scale).collect() };
        Self {
            token_embed: sinit(config.vocab_size * d),
            attn_query: sinit(d),
            config,
        }
    }

    /// Embed a document: lookup token embeddings, apply pooling strategy.
    pub fn embed_document(&self, tokens: &[usize]) -> Vec<f32> {
        if tokens.is_empty() {
            return vec![0.0; self.config.embed_dim];
        }
        let d = self.config.embed_dim;
        let vocab = self.config.vocab_size;
        let embeds: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&tok| {
                let idx = tok.min(vocab.saturating_sub(1));
                (0..d)
                    .map(|k| self.token_embed.get(idx * d + k).copied().unwrap_or(0.0))
                    .collect()
            })
            .collect();

        match self.config.pool {
            PoolingStrategy::CLS => embeds.into_iter().next().unwrap_or_else(|| vec![0.0; d]),
            PoolingStrategy::MeanPool => {
                let n = embeds.len() as f32;
                let mut out = vec![0.0_f32; d];
                for e in &embeds {
                    for (i, v) in e.iter().enumerate() {
                        if let Some(cell) = out.get_mut(i) {
                            *cell += v;
                        }
                    }
                }
                out.iter_mut().for_each(|v| *v /= n);
                out
            }
            PoolingStrategy::MaxPool => {
                let mut out = vec![f32::NEG_INFINITY; d];
                for e in &embeds {
                    for (i, &v) in e.iter().enumerate() {
                        if let Some(cell) = out.get_mut(i) {
                            if v > *cell {
                                *cell = v;
                            }
                        }
                    }
                }
                out
            }
            PoolingStrategy::AttentionPool => {
                let scores: Vec<f32> = embeds
                    .iter()
                    .map(|e| {
                        e.iter()
                            .zip(self.attn_query.iter())
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                    })
                    .collect();
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores.iter().map(|s| (s - max_s).exp()).sum();
                let weights: Vec<f32> = scores
                    .iter()
                    .map(|s| (s - max_s).exp() / exp_sum.max(1e-9))
                    .collect();
                let mut out = vec![0.0_f32; d];
                for (w, e) in weights.iter().zip(embeds.iter()) {
                    for (i, v) in e.iter().enumerate() {
                        if let Some(cell) = out.get_mut(i) {
                            *cell += w * v;
                        }
                    }
                }
                out
            }
        }
    }

    /// Cosine similarity between two embedding vectors.
    pub fn similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let denom = na * nb;
        if denom < 1e-9 {
            0.0
        } else {
            (dot / denom).clamp(-1.0, 1.0)
        }
    }

    /// Retrieve top-k most similar documents to a query embedding.
    pub fn retrieve(
        &self,
        query_embed: &[f32],
        doc_embeds: &[Vec<f32>],
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = doc_embeds
            .iter()
            .enumerate()
            .map(|(i, doc)| (i, Self::similarity(query_embed, doc)))
            .collect();
        scored.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }
}

// ─── QuestionAnsweringDoc ─────────────────────────────────────────────────────

/// Configuration for document question answering.
#[derive(Debug, Clone)]
pub struct DocQaConfig {
    /// Maximum total sequence length (question + context)
    pub max_seq_len: usize,
    /// Stride when sliding window over long documents
    pub doc_stride: usize,
    /// Maximum number of answer tokens
    pub max_answer_len: usize,
}

impl DocQaConfig {
    /// Create a new DocQaConfig.
    pub fn new(max_seq_len: usize, doc_stride: usize, max_answer_len: usize) -> Self {
        Self {
            max_seq_len,
            doc_stride,
            max_answer_len,
        }
    }
}

/// A single sliding-window span over a long document.
#[derive(Debug, Clone)]
pub struct InputSpan {
    /// Token ids for \[CLS\] + question + \[SEP\] + context_chunk + \[SEP\]
    pub tokens: Vec<usize>,
    /// Character-level offset for each token: (char_start, char_end)
    pub offset_mapping: Vec<(usize, usize)>,
    /// Number of question tokens (excluding special tokens)
    pub question_len: usize,
}

/// Create overlapping input spans from question and document tokens.
pub fn create_spans(
    question_tokens: &[usize],
    doc_tokens: &[usize],
    config: &DocQaConfig,
) -> Vec<InputSpan> {
    let cls_id = 0usize;
    let sep_id = 1usize;
    let q_len = question_tokens.len();
    let overhead = q_len + 3;
    let max_ctx = config.max_seq_len.saturating_sub(overhead);

    if max_ctx == 0 || doc_tokens.is_empty() {
        let mut tokens = vec![cls_id];
        tokens.extend_from_slice(question_tokens);
        tokens.push(sep_id);
        let take = doc_tokens
            .len()
            .min(config.max_seq_len.saturating_sub(q_len + 3));
        tokens.extend_from_slice(&doc_tokens[..take]);
        tokens.push(sep_id);

        let q_offset = q_len + 2;
        let offset_mapping: Vec<(usize, usize)> = tokens
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if i >= q_offset && i < q_offset + take {
                    let ci = i - q_offset;
                    (ci, ci + 1)
                } else {
                    (0, 0)
                }
            })
            .collect();

        return vec![InputSpan {
            tokens,
            offset_mapping,
            question_len: q_len,
        }];
    }

    let stride = config.doc_stride.max(1);
    let mut spans = Vec::new();
    let mut start = 0;

    while start < doc_tokens.len() {
        let end = (start + max_ctx).min(doc_tokens.len());
        let chunk = &doc_tokens[start..end];

        let mut tokens = vec![cls_id];
        tokens.extend_from_slice(question_tokens);
        tokens.push(sep_id);
        tokens.extend_from_slice(chunk);
        tokens.push(sep_id);

        let q_offset = q_len + 2;
        let offset_mapping: Vec<(usize, usize)> = tokens
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if i >= q_offset && i < q_offset + chunk.len() {
                    let ci = start + (i - q_offset);
                    (ci, ci + 1)
                } else {
                    (0, 0)
                }
            })
            .collect();

        spans.push(InputSpan {
            tokens,
            offset_mapping,
            question_len: q_len,
        });

        if end == doc_tokens.len() {
            break;
        }
        start += stride;
    }

    spans
}

/// Extract a text answer from start/end logits over a span.
pub fn extract_answer(
    start_logits: &[f32],
    end_logits: &[f32],
    span: &InputSpan,
    text: &str,
) -> Option<String> {
    let n = start_logits.len().min(end_logits.len());
    if n == 0 {
        return None;
    }

    let ctx_start = span.question_len + 2;

    let mut best_score = f32::NEG_INFINITY;
    let mut best_start = 0usize;
    let mut best_end = 0usize;

    for s in ctx_start..n {
        let max_e = (s + span.tokens.len().saturating_sub(1)).min(n - 1);
        let max_e_constrained = (s + 1).min(max_e + 1);
        let end_limit = (s + 30).min(n);
        for e in max_e_constrained..=end_limit.min(n) {
            let score = start_logits.get(s).copied().unwrap_or(f32::NEG_INFINITY)
                + end_logits.get(e - 1).copied().unwrap_or(f32::NEG_INFINITY);
            if score > best_score {
                best_score = score;
                best_start = s;
                best_end = e;
            }
        }
    }

    if best_start >= best_end {
        return None;
    }

    let char_start = span.offset_mapping.get(best_start).map(|(s, _)| *s)?;
    let char_end = span.offset_mapping.get(best_end - 1).map(|(_, e)| *e)?;

    if char_start >= char_end {
        return None;
    }

    let mut byte_start = text.len();
    let mut byte_end = text.len();
    for (ci, (bi, _)) in text.char_indices().enumerate() {
        if ci == char_start {
            byte_start = bi;
        }
        if ci == char_end {
            byte_end = bi;
        }
    }
    if char_end >= text.chars().count() {
        byte_end = text.len();
    }

    if byte_start >= byte_end || byte_end > text.len() {
        return None;
    }

    Some(text[byte_start..byte_end].trim().to_string())
}
