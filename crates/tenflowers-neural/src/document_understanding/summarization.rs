//! Document summarization, form parsing, and evaluation metrics.

use super::OcrBox;
use std::collections::HashMap;

// ─── DocumentSummarizer ──────────────────────────────────────────────────────

/// TF-IDF-based sentence scorer for extractive summarization.
pub struct SentenceScorer {
    /// IDF weights per term
    pub tfidf_weights: HashMap<String, f32>,
}

impl SentenceScorer {
    /// Create a SentenceScorer by computing TF-IDF from a corpus of sentences.
    pub fn from_sentences(sentences: &[&str]) -> Self {
        let n_docs = sentences.len() as f32;
        let mut df: HashMap<String, usize> = HashMap::new();
        for sent in sentences {
            let words: std::collections::HashSet<String> =
                Self::tokenize(sent).into_iter().collect();
            for w in words {
                *df.entry(w).or_insert(0) += 1;
            }
        }
        let tfidf_weights = df
            .into_iter()
            .map(|(w, df_count)| {
                let idf = ((n_docs + 1.0) / (df_count as f32 + 1.0)).ln() + 1.0;
                (w, idf)
            })
            .collect();
        Self { tfidf_weights }
    }

    /// Score a sentence based on the sum of TF-IDF weights of its words.
    pub fn score_sentence(&self, sentence: &str, doc_tfidf: &HashMap<String, f32>) -> f32 {
        Self::tokenize(sentence)
            .iter()
            .map(|w| doc_tfidf.get(w).copied().unwrap_or(0.0))
            .sum()
    }

    /// Simple whitespace + punctuation tokenizer, lowercased.
    pub(crate) fn tokenize(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphabetic())
            .filter(|s| !s.is_empty() && s.len() > 1)
            .map(|s| s.to_lowercase())
            .collect()
    }

    /// Compute TF-IDF vector for a document (used for MMR).
    pub fn doc_tfidf(text: &str, idf: &HashMap<String, f32>) -> HashMap<String, f32> {
        let words = Self::tokenize(text);
        let n = words.len() as f32;
        if n == 0.0 {
            return HashMap::new();
        }
        let mut tf: HashMap<String, f32> = HashMap::new();
        for w in &words {
            *tf.entry(w.clone()).or_insert(0.0) += 1.0 / n;
        }
        tf.into_iter()
            .map(|(w, tf_val)| {
                let idf_val = idf.get(&w).copied().unwrap_or(1.0);
                (w, tf_val * idf_val)
            })
            .collect()
    }

    /// Cosine similarity between two TF-IDF maps.
    pub(crate) fn cosine_tfidf(a: &HashMap<String, f32>, b: &HashMap<String, f32>) -> f32 {
        let dot: f32 = a
            .iter()
            .map(|(k, v)| b.get(k).copied().unwrap_or(0.0) * v)
            .sum();
        let na: f32 = a.values().map(|v| v * v).sum::<f32>().sqrt();
        let nb: f32 = b.values().map(|v| v * v).sum::<f32>().sqrt();
        let denom = na * nb;
        if denom < 1e-9 {
            0.0
        } else {
            dot / denom
        }
    }
}

/// Split text into sentences by '.', '!', '?'.
pub(crate) fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }
    sentences
}

/// Extractive summarization: select top-n highest-scored sentences in original order.
pub fn extractive_summarize(text: &str, n_sentences: usize) -> String {
    let sentences = split_sentences(text);
    if sentences.is_empty() || n_sentences == 0 {
        return String::new();
    }
    let n = n_sentences.min(sentences.len());

    let scorer =
        SentenceScorer::from_sentences(&sentences.iter().map(|s| s.as_str()).collect::<Vec<_>>());
    let doc_tfidf = SentenceScorer::doc_tfidf(text, &scorer.tfidf_weights);

    let mut scored: Vec<(usize, f32)> = sentences
        .iter()
        .enumerate()
        .map(|(i, s)| (i, scorer.score_sentence(s, &doc_tfidf)))
        .collect();

    scored.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut selected_indices: Vec<usize> = scored.iter().take(n).map(|(i, _)| *i).collect();
    selected_indices.sort_unstable();

    selected_indices
        .iter()
        .map(|&i| sentences[i].as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// MMR (Maximal Marginal Relevance) summarization.
///
/// `lambda` controls relevance-diversity tradeoff (0=max diversity, 1=max relevance).
pub fn mmr_summarize(text: &str, n_sentences: usize, lambda: f32) -> String {
    let sentences = split_sentences(text);
    if sentences.is_empty() || n_sentences == 0 {
        return String::new();
    }
    let n = n_sentences.min(sentences.len());

    let scorer =
        SentenceScorer::from_sentences(&sentences.iter().map(|s| s.as_str()).collect::<Vec<_>>());
    let doc_tfidf = SentenceScorer::doc_tfidf(text, &scorer.tfidf_weights);

    let sent_tfidf: Vec<HashMap<String, f32>> = sentences
        .iter()
        .map(|s| SentenceScorer::doc_tfidf(s, &scorer.tfidf_weights))
        .collect();

    let relevance: Vec<f32> = sent_tfidf
        .iter()
        .map(|sv| SentenceScorer::cosine_tfidf(sv, &doc_tfidf))
        .collect();

    let mut selected: Vec<usize> = Vec::new();
    let mut remaining: Vec<usize> = (0..sentences.len()).collect();

    for _ in 0..n {
        if remaining.is_empty() {
            break;
        }
        let best = remaining
            .iter()
            .max_by(|&&i, &&j| {
                let sim_selected_i = if selected.is_empty() {
                    0.0_f32
                } else {
                    selected
                        .iter()
                        .map(|&s| SentenceScorer::cosine_tfidf(&sent_tfidf[i], &sent_tfidf[s]))
                        .fold(f32::NEG_INFINITY, f32::max)
                };
                let sim_selected_j = if selected.is_empty() {
                    0.0_f32
                } else {
                    selected
                        .iter()
                        .map(|&s| SentenceScorer::cosine_tfidf(&sent_tfidf[j], &sent_tfidf[s]))
                        .fold(f32::NEG_INFINITY, f32::max)
                };
                let score_i = lambda * relevance[i] - (1.0 - lambda) * sim_selected_i;
                let score_j = lambda * relevance[j] - (1.0 - lambda) * sim_selected_j;
                score_i
                    .partial_cmp(&score_j)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied();

        if let Some(best_idx) = best {
            selected.push(best_idx);
            remaining.retain(|&i| i != best_idx);
        }
    }

    selected.sort_unstable();
    selected
        .iter()
        .map(|&i| sentences[i].as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

// ─── FormParser ──────────────────────────────────────────────────────────────

/// Type of a form field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    Text,
    Checkbox,
    Radio,
    Dropdown,
    Date,
    Number,
}

/// A parsed form field with key-value and spatial information.
#[derive(Debug, Clone)]
pub struct FormField {
    /// Field label / key
    pub key: String,
    /// Field value
    pub value: String,
    /// Type of the field
    pub field_type: FieldType,
    /// Bounding box of the value region
    pub bbox: (f32, f32, f32, f32),
}

impl FormField {
    /// Create a new FormField.
    pub fn new(
        key: impl Into<String>,
        value: impl Into<String>,
        field_type: FieldType,
        bbox: (f32, f32, f32, f32),
    ) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
            field_type,
            bbox,
        }
    }
}

/// Parse form from OCR boxes by detecting label-value pairs by proximity.
pub fn parse_form(boxes: &[OcrBox]) -> Vec<FormField> {
    let mut fields = Vec::new();
    let proximity_x = 200.0_f32;
    let proximity_y = 30.0_f32;

    for (i, label_box) in boxes.iter().enumerate() {
        let text = label_box.text.trim();
        if !text.ends_with(':') && !text.ends_with('?') {
            continue;
        }
        let lx = label_box.bbox.0 + label_box.bbox.2;
        let ly = label_box.y_center();
        let key = text.trim_end_matches(':').trim_end_matches('?').trim();

        let value_box = boxes
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .filter_map(|(_, vb)| {
                let vx = vb.bbox.0;
                let vy = vb.y_center();
                let dx = vx - lx;
                let dy = (vy - ly).abs();
                if (dx >= 0.0 && dx < proximity_x && dy < proximity_y)
                    || (dy > 0.0
                        && dy < proximity_y * 2.0
                        && (vb.bbox.0 - label_box.bbox.0).abs() < proximity_x)
                {
                    Some((dx * dx + dy * dy, vb))
                } else {
                    None
                }
            })
            .min_by(|(da, _), (db, _)| da.partial_cmp(db).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, vb)| vb);

        if let Some(vb) = value_box {
            let field_type = infer_field_type(&vb.text);
            fields.push(FormField::new(key, &vb.text, field_type, vb.bbox));
        }
    }

    fields
}

/// Infer field type from value text.
pub(crate) fn infer_field_type(value: &str) -> FieldType {
    let v = value.trim();
    if matches!(
        v,
        "☑" | "☐" | "✓" | "✗" | "yes" | "no" | "Yes" | "No" | "true" | "false"
    ) {
        return FieldType::Checkbox;
    }
    if v.len() == 1 && v.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false) {
        return FieldType::Radio;
    }
    if v.len() == 10
        && (v.contains('-') || v.contains('/'))
        && v.chars().filter(|c| c.is_ascii_digit()).count() == 8
    {
        return FieldType::Date;
    }
    if v.chars()
        .all(|c| c.is_ascii_digit() || matches!(c, '.' | ',' | '-' | '+'))
        && !v.is_empty()
    {
        return FieldType::Number;
    }
    FieldType::Text
}

/// Validate a form field based on its type.
pub fn validate_field(field: &FormField) -> bool {
    let v = field.value.trim();
    match field.field_type {
        FieldType::Text => !v.is_empty(),
        FieldType::Checkbox => matches!(
            v,
            "☑" | "☐" | "✓" | "✗" | "yes" | "no" | "Yes" | "No" | "true" | "false" | "1" | "0"
        ),
        FieldType::Radio => v.len() <= 3 && !v.is_empty(),
        FieldType::Dropdown => !v.is_empty(),
        FieldType::Date => {
            v.len() == 10
                && (v.contains('-') || v.contains('/'))
                && v.chars().filter(|c| c.is_ascii_digit()).count() == 8
        }
        FieldType::Number => {
            v.parse::<f64>().is_ok() || v.replace([',', '_'], "").parse::<f64>().is_ok()
        }
    }
}

// ─── DocumentMetrics ─────────────────────────────────────────────────────────

use super::extraction::DocEntity;

/// Compute token-level F1 score for NER spans (exact match on span boundaries).
pub fn f1_score_ner(pred: &[DocEntity], gold: &[DocEntity]) -> f32 {
    if pred.is_empty() && gold.is_empty() {
        return 1.0;
    }
    if pred.is_empty() || gold.is_empty() {
        return 0.0;
    }
    let mut tp = 0usize;
    for p in pred {
        let matched = gold
            .iter()
            .any(|g| g.start == p.start && g.end == p.end && g.entity_type == p.entity_type);
        if matched {
            tp += 1;
        }
    }
    let precision = tp as f32 / pred.len() as f32;
    let recall = tp as f32 / gold.len() as f32;
    if precision + recall < 1e-9 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

/// Check exact string match (case-insensitive, whitespace-normalized).
pub fn exact_match(pred: &str, gold: &str) -> bool {
    normalize_answer(pred) == normalize_answer(gold)
}

fn normalize_answer(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Token-level F1 score for QA (as used in SQuAD).
pub fn f1_answer(pred: &str, gold: &str) -> f32 {
    let pred_tokens: Vec<&str> = pred.split_whitespace().collect();
    let gold_tokens: Vec<&str> = gold.split_whitespace().collect();
    if pred_tokens.is_empty() && gold_tokens.is_empty() {
        return 1.0;
    }
    if pred_tokens.is_empty() || gold_tokens.is_empty() {
        return 0.0;
    }
    let mut gold_counts: HashMap<&str, usize> = HashMap::new();
    for t in &gold_tokens {
        *gold_counts.entry(t).or_insert(0) += 1;
    }
    let mut common = 0usize;
    let mut pred_counts: HashMap<&str, usize> = HashMap::new();
    for t in &pred_tokens {
        *pred_counts.entry(t).or_insert(0) += 1;
    }
    for (t, &pc) in &pred_counts {
        let gc = gold_counts.get(t).copied().unwrap_or(0);
        common += pc.min(gc);
    }
    let precision = common as f32 / pred_tokens.len() as f32;
    let recall = common as f32 / gold_tokens.len() as f32;
    if precision + recall < 1e-9 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

/// ROUGE-N score: n-gram overlap between hypothesis and reference.
pub fn rouge_n(hypothesis: &str, reference: &str, n: usize) -> f32 {
    if n == 0 {
        return 0.0;
    }
    let hyp_ngrams = build_ngrams(hypothesis, n);
    let ref_ngrams = build_ngrams(reference, n);

    if hyp_ngrams.is_empty() || ref_ngrams.is_empty() {
        return 0.0;
    }

    let mut ref_counts: HashMap<Vec<String>, usize> = HashMap::new();
    for ng in &ref_ngrams {
        *ref_counts.entry(ng.clone()).or_insert(0) += 1;
    }

    let mut hyp_counts: HashMap<Vec<String>, usize> = HashMap::new();
    for ng in &hyp_ngrams {
        *hyp_counts.entry(ng.clone()).or_insert(0) += 1;
    }

    let mut overlap = 0usize;
    for (ng, &hc) in &hyp_counts {
        let rc = ref_counts.get(ng).copied().unwrap_or(0);
        overlap += hc.min(rc);
    }

    overlap as f32 / ref_ngrams.len() as f32
}

fn build_ngrams(text: &str, n: usize) -> Vec<Vec<String>> {
    let tokens: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect();
    if tokens.len() < n {
        return Vec::new();
    }
    tokens.windows(n).map(|w| w.to_vec()).collect()
}
