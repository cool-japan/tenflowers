use super::extraction::{
    create_spans, extract_answer, DocEmbedder, DocEmbedderConfig, DocEntity, DocQaConfig,
    EntityType, InformationExtractor, PoolingStrategy,
};
use super::summarization::{
    exact_match, extractive_summarize, f1_answer, f1_score_ner, infer_field_type, mmr_summarize,
    parse_form, rouge_n, validate_field, FieldType, FormField, SentenceScorer,
};
use super::*;

// ── LayoutLM ──────────────────────────────────────────────────────────────

#[test]
fn test_layout_lm_config() {
    let cfg = LayoutLmConfig::new(1000, 128, 64, 8, 2);
    assert_eq!(cfg.vocab_size, 1000);
    assert_eq!(cfg.embed_dim, 64);
    assert_eq!(cfg.n_heads, 8);
}

#[test]
fn test_bbox_embedding_lookup() {
    let be = BboxEmbedding::new(1001, 16);
    let v = be.lookup(100, 200, 50, 30);
    assert_eq!(v.len(), 16);
}

#[test]
fn test_bbox_embedding_clamping() {
    let be = BboxEmbedding::new(1001, 8);
    let v1 = be.lookup(2000, 2000, 0, 0);
    let v2 = be.lookup(1000, 1000, 0, 0);
    assert_eq!(v1, v2, "clamped values should match");
}

#[test]
fn test_layout_lm_forward_shape() {
    let cfg = LayoutLmConfig::new(100, 32, 8, 2, 1);
    let model = LayoutLm::new(cfg);
    let tokens = vec![0usize, 5, 10, 15];
    let bboxes = vec![(10u32, 20u32, 100u32, 20u32); 4];
    let out = model.forward(&tokens, &bboxes);
    assert_eq!(out.len(), 4);
    assert_eq!(out[0].len(), 8);
}

#[test]
fn test_layout_lm_empty_sequence() {
    let cfg = LayoutLmConfig::new(100, 32, 8, 2, 1);
    let model = LayoutLm::new(cfg);
    let out = model.forward(&[], &[]);
    assert_eq!(out.len(), 0);
}

#[test]
fn test_layout_lm_layer_forward() {
    let layer = LayoutLmLayer::new(8, 2);
    let hidden = vec![vec![0.1_f32; 8], vec![0.2_f32; 8]];
    let out = layer.forward(&hidden, None);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].len(), 8);
}

// ── DocumentOCR ─────────────────────────────────────────────────────────────

#[test]
fn test_sort_reading_order() {
    let mut boxes = vec![
        OcrBox::new("C", (50.0, 50.0, 20.0, 10.0), 0.9),
        OcrBox::new("A", (10.0, 10.0, 20.0, 10.0), 0.9),
        OcrBox::new("B", (10.0, 50.0, 20.0, 10.0), 0.9),
    ];
    sort_reading_order(&mut boxes);
    assert_eq!(boxes[0].text, "A");
}

#[test]
fn test_sort_reading_order_same_line() {
    let mut boxes = vec![
        OcrBox::new("World", (100.0, 10.0, 40.0, 12.0), 0.9),
        OcrBox::new("Hello", (10.0, 10.0, 40.0, 12.0), 0.9),
    ];
    sort_reading_order(&mut boxes);
    assert_eq!(boxes[0].text, "Hello");
    assert_eq!(boxes[1].text, "World");
}

#[test]
fn test_merge_lines() {
    let boxes = vec![
        OcrBox::new("Hello", (0.0, 0.0, 50.0, 10.0), 0.9),
        OcrBox::new("World", (60.0, 2.0, 50.0, 10.0), 0.9),
        OcrBox::new("Next", (0.0, 20.0, 50.0, 10.0), 0.9),
    ];
    let lines = merge_lines(&boxes, 8.0);
    assert_eq!(lines.len(), 2);
    assert!(lines[0].contains("Hello") && lines[0].contains("World"));
}

#[test]
fn test_normalize_bbox() {
    let nb = normalize_bbox((100.0, 200.0, 50.0, 30.0), 1000.0, 1000.0);
    assert_eq!(nb.0, 100);
    assert_eq!(nb.1, 200);
    assert_eq!(nb.2, 50);
    assert_eq!(nb.3, 30);
}

#[test]
fn test_normalize_bbox_clamping() {
    let nb = normalize_bbox((1200.0, 0.0, 0.0, 0.0), 1000.0, 1000.0);
    assert_eq!(nb.0, 1000);
}

#[test]
fn test_ocr_box_centers() {
    let b = OcrBox::new("hi", (10.0, 20.0, 40.0, 30.0), 0.8);
    assert!((b.x_center() - 30.0).abs() < 1e-3);
    assert!((b.y_center() - 35.0).abs() < 1e-3);
}

// ── TableExtractor ────────────────────────────────────────────────────────

#[test]
fn test_detect_table_structure() {
    let boxes = vec![
        OcrBox::new("Name", (0.0, 0.0, 60.0, 12.0), 0.9),
        OcrBox::new("Age", (80.0, 0.0, 30.0, 12.0), 0.9),
        OcrBox::new("Alice", (0.0, 20.0, 60.0, 12.0), 0.9),
        OcrBox::new("30", (80.0, 20.0, 30.0, 12.0), 0.9),
    ];
    let tables = detect_table_structure(&boxes);
    assert_eq!(tables.len(), 1);
    assert_eq!(tables[0].n_rows, 2);
}

#[test]
fn test_table_to_csv() {
    let mut t = Table::new(2, 2);
    t.cells.push(TableCell::new("Name", 0, 0));
    t.cells.push(TableCell::new("Age", 0, 1));
    t.cells.push(TableCell::new("Alice", 1, 0));
    t.cells.push(TableCell::new("30", 1, 1));
    let csv = table_to_csv(&t);
    assert!(csv.contains("Name,Age"));
    assert!(csv.contains("Alice,30"));
}

#[test]
fn test_table_to_csv_quoting() {
    let mut t = Table::new(1, 1);
    t.cells.push(TableCell::new("hello, world", 0, 0));
    let csv = table_to_csv(&t);
    assert!(csv.contains('"'));
}

#[test]
fn test_table_to_json() {
    let mut t = Table::new(3, 2);
    t.cells.push(TableCell::new("Name", 0, 0));
    t.cells.push(TableCell::new("Score", 0, 1));
    t.cells.push(TableCell::new("Bob", 1, 0));
    t.cells.push(TableCell::new("95", 1, 1));
    t.cells.push(TableCell::new("Carol", 2, 0));
    t.cells.push(TableCell::new("87", 2, 1));
    let json = table_to_json(&t);
    assert!(json.contains("Bob"));
    assert!(json.contains("95"));
    assert!(json.starts_with('['));
    assert!(json.ends_with(']'));
}

// ── DocumentClassifier ────────────────────────────────────────────────────

#[test]
fn test_doc_classifier_returns_class() {
    let clf = DocClassifier::new(16);
    let embeddings = vec![vec![0.1_f32; 16]; 10];
    let cls = clf.classify(&embeddings);
    let _ = cls.label();
}

#[test]
fn test_doc_classifier_with_confidence() {
    let clf = DocClassifier::new(16);
    let embeddings = vec![vec![0.5_f32; 16]; 5];
    let probs = clf.classify_with_confidence(&embeddings);
    assert_eq!(probs.len(), 7);
    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    assert!((sum - 1.0).abs() < 1e-4, "probabilities must sum to 1");
}

#[test]
fn test_doc_classifier_empty_embeddings() {
    let clf = DocClassifier::new(16);
    let cls = clf.classify(&[]);
    let _ = cls.label();
}

#[test]
fn test_doc_class_labels() {
    assert_eq!(DocClass::Invoice.label(), "Invoice");
    assert_eq!(DocClass::Resume.label(), "Resume");
    assert_eq!(DocClass::Other.label(), "Other");
}

// ── InformationExtractor ──────────────────────────────────────────────────

#[test]
fn test_extract_dates_iso() {
    let text = "Contract date: 2024-03-15 and end date: 2025-12-31.";
    let entities = InformationExtractor::extract_dates(text);
    assert!(entities.len() >= 2);
    assert!(entities.iter().any(|e| e.text == "2024-03-15"));
    assert!(entities.iter().any(|e| e.text == "2025-12-31"));
}

#[test]
fn test_extract_dates_european() {
    let text = "Invoice date: 15/03/2024";
    let entities = InformationExtractor::extract_dates(text);
    assert!(entities.len() >= 1);
    assert!(entities.iter().any(|e| e.text.contains("15")));
}

#[test]
fn test_extract_amounts() {
    let text = "Total: $1,234.56 and tax: $99.00";
    let entities = InformationExtractor::extract_amounts(text);
    assert!(!entities.is_empty());
    assert!(entities.iter().all(|e| e.entity_type == EntityType::Amount));
}

#[test]
fn test_extract_emails() {
    let text = "Contact us at info@example.com or support@company.org for help.";
    let entities = InformationExtractor::extract_emails(text);
    assert!(entities.len() >= 2);
    assert!(entities.iter().any(|e| e.text.contains("info@example.com")));
}

#[test]
fn test_extract_emails_none() {
    let text = "No emails here, just text.";
    let entities = InformationExtractor::extract_emails(text);
    assert!(entities.is_empty());
}

#[test]
fn test_entity_type_labels() {
    assert_eq!(EntityType::Date.label(), "DATE");
    assert_eq!(EntityType::Email.label(), "EMAIL");
    assert_eq!(EntityType::InvoiceNumber.label(), "INVOICE_NUMBER");
}

// ── DocEmbedder ───────────────────────────────────────────────────────────

#[test]
fn test_doc_embedder_mean_pool() {
    let cfg = DocEmbedderConfig::new(100, 16, PoolingStrategy::MeanPool);
    let emb = DocEmbedder::new(cfg);
    let tokens = vec![0usize, 5, 10, 20];
    let out = emb.embed_document(&tokens);
    assert_eq!(out.len(), 16);
}

#[test]
fn test_doc_embedder_cls() {
    let cfg = DocEmbedderConfig::new(100, 16, PoolingStrategy::CLS);
    let emb = DocEmbedder::new(cfg);
    let out = emb.embed_document(&[0, 1, 2]);
    assert_eq!(out.len(), 16);
}

#[test]
fn test_doc_embedder_max_pool() {
    let cfg = DocEmbedderConfig::new(100, 16, PoolingStrategy::MaxPool);
    let emb = DocEmbedder::new(cfg);
    let out = emb.embed_document(&[3, 7, 11]);
    assert_eq!(out.len(), 16);
}

#[test]
fn test_doc_embedder_attention_pool() {
    let cfg = DocEmbedderConfig::new(100, 16, PoolingStrategy::AttentionPool);
    let emb = DocEmbedder::new(cfg);
    let out = emb.embed_document(&[1, 2, 3]);
    assert_eq!(out.len(), 16);
}

#[test]
fn test_cosine_similarity() {
    let a = vec![1.0_f32, 0.0, 0.0];
    let b = vec![1.0_f32, 0.0, 0.0];
    assert!((DocEmbedder::similarity(&a, &b) - 1.0).abs() < 1e-5);

    let c = vec![-1.0_f32, 0.0, 0.0];
    assert!((DocEmbedder::similarity(&a, &c) + 1.0).abs() < 1e-5);
}

#[test]
fn test_retrieve_top_k() {
    let cfg = DocEmbedderConfig::new(50, 8, PoolingStrategy::MeanPool);
    let emb = DocEmbedder::new(cfg);
    let query = vec![1.0_f32; 8];
    let docs = vec![vec![1.0_f32; 8], vec![-1.0_f32; 8], vec![0.5_f32; 8]];
    let results = emb.retrieve(&query, &docs, 2);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 0);
}

// ── QuestionAnsweringDoc ──────────────────────────────────────────────────

#[test]
fn test_create_spans_single() {
    let cfg = DocQaConfig::new(20, 5, 10);
    let q_tokens = vec![2usize, 3, 4];
    let d_tokens = vec![5usize, 6, 7, 8];
    let spans = create_spans(&q_tokens, &d_tokens, &cfg);
    assert!(!spans.is_empty());
    assert_eq!(spans[0].question_len, 3);
}

#[test]
fn test_create_spans_multiple() {
    let cfg = DocQaConfig::new(10, 3, 5);
    let q_tokens = vec![1usize, 2];
    let d_tokens: Vec<usize> = (10..30).collect();
    let spans = create_spans(&q_tokens, &d_tokens, &cfg);
    assert!(
        spans.len() > 1,
        "should create multiple spans for long document"
    );
}

#[test]
fn test_extract_answer_basic() {
    let cfg = DocQaConfig::new(20, 5, 10);
    let q_tokens = vec![1usize, 2];
    let d_tokens = vec![10usize, 11, 12, 13];
    let spans = create_spans(&q_tokens, &d_tokens, &cfg);
    let text = "Hello World test doc";
    let n = spans[0].tokens.len();
    let mut start_logits = vec![0.0_f32; n];
    let mut end_logits = vec![0.0_f32; n];
    let ctx_start = q_tokens.len() + 2;
    if let Some(s) = start_logits.get_mut(ctx_start) {
        *s = 10.0;
    }
    if let Some(e) = end_logits.get_mut(ctx_start) {
        *e = 10.0;
    }
    let ans = extract_answer(&start_logits, &end_logits, &spans[0], text);
    let _ = ans;
}

// ── DocumentSummarizer ────────────────────────────────────────────────────

#[test]
fn test_extractive_summarize() {
    let text = "The quick brown fox jumps over the lazy dog. \
                Machine learning is a subfield of artificial intelligence. \
                Natural language processing enables computers to understand text. \
                Deep learning uses neural networks with many layers.";
    let summary = extractive_summarize(text, 2);
    assert!(!summary.is_empty());
    let sent_count = summary.split(". ").count();
    assert!(sent_count >= 1);
}

#[test]
fn test_extractive_summarize_single() {
    let text = "One sentence only.";
    let summary = extractive_summarize(text, 3);
    assert!(!summary.is_empty());
}

#[test]
fn test_mmr_summarize() {
    let text = "The quick brown fox jumps. \
                Deep learning is powerful. \
                Neural networks learn features. \
                Transformers use attention mechanisms.";
    let summary = mmr_summarize(text, 2, 0.7);
    assert!(!summary.is_empty());
}

#[test]
fn test_sentence_scorer() {
    let sents = ["machine learning is great", "learning is fun"];
    let scorer = SentenceScorer::from_sentences(&sents);
    let doc_tfidf = SentenceScorer::doc_tfidf("machine learning is great", &scorer.tfidf_weights);
    let score = scorer.score_sentence("machine learning", &doc_tfidf);
    assert!(score >= 0.0);
}

// ── FormParser ────────────────────────────────────────────────────────────

#[test]
fn test_parse_form_basic() {
    let boxes = vec![
        OcrBox::new("Name:", (0.0, 10.0, 50.0, 12.0), 0.9),
        OcrBox::new("John Doe", (60.0, 10.0, 80.0, 12.0), 0.9),
        OcrBox::new("Date:", (0.0, 30.0, 50.0, 12.0), 0.9),
        OcrBox::new("2024-01-15", (60.0, 30.0, 80.0, 12.0), 0.9),
    ];
    let fields = parse_form(&boxes);
    assert!(!fields.is_empty());
    let names: Vec<_> = fields.iter().map(|f| f.key.as_str()).collect();
    assert!(names.iter().any(|&n| n.contains("Name")));
}

#[test]
fn test_validate_field_text() {
    let f = FormField::new("Name", "John", FieldType::Text, (0.0, 0.0, 50.0, 10.0));
    assert!(validate_field(&f));
}

#[test]
fn test_validate_field_empty_text() {
    let f = FormField::new("Name", "", FieldType::Text, (0.0, 0.0, 50.0, 10.0));
    assert!(!validate_field(&f));
}

#[test]
fn test_validate_field_date() {
    let f = FormField::new(
        "Date",
        "2024-03-15",
        FieldType::Date,
        (0.0, 0.0, 50.0, 10.0),
    );
    assert!(validate_field(&f));
    let f2 = FormField::new(
        "Date",
        "not-a-date",
        FieldType::Date,
        (0.0, 0.0, 50.0, 10.0),
    );
    assert!(!validate_field(&f2));
}

#[test]
fn test_validate_field_number() {
    let f = FormField::new(
        "Amount",
        "1234.56",
        FieldType::Number,
        (0.0, 0.0, 50.0, 10.0),
    );
    assert!(validate_field(&f));
    let f2 = FormField::new("Amount", "abc", FieldType::Number, (0.0, 0.0, 50.0, 10.0));
    assert!(!validate_field(&f2));
}

#[test]
fn test_infer_field_type() {
    assert_eq!(infer_field_type("yes"), FieldType::Checkbox);
    assert_eq!(infer_field_type("2024-03-15"), FieldType::Date);
    assert_eq!(infer_field_type("1234.56"), FieldType::Number);
    assert_eq!(infer_field_type("Some long text"), FieldType::Text);
}

// ── DocumentMetrics ───────────────────────────────────────────────────────

#[test]
fn test_f1_score_ner_perfect() {
    let ents = vec![
        DocEntity::new(EntityType::Date, "2024-01-01", 0, 10, 0.9),
        DocEntity::new(EntityType::Amount, "$100", 20, 24, 0.9),
    ];
    let score = f1_score_ner(&ents, &ents);
    assert!((score - 1.0).abs() < 1e-5);
}

#[test]
fn test_f1_score_ner_zero() {
    let pred = vec![DocEntity::new(EntityType::Date, "2024-01-01", 0, 10, 0.9)];
    let gold = vec![DocEntity::new(EntityType::Amount, "$100", 20, 24, 0.9)];
    let score = f1_score_ner(&pred, &gold);
    assert!(score.abs() < 1e-5);
}

#[test]
fn test_exact_match() {
    assert!(exact_match("hello world", "Hello World"));
    assert!(!exact_match("hello", "world"));
    assert!(exact_match("  foo  bar  ", "foo bar"));
}

#[test]
fn test_f1_answer_perfect() {
    assert!((f1_answer("hello world", "hello world") - 1.0).abs() < 1e-5);
}

#[test]
fn test_f1_answer_partial() {
    let score = f1_answer("hello world", "hello there");
    assert!(score > 0.0 && score < 1.0);
}

#[test]
fn test_f1_answer_empty() {
    assert!((f1_answer("", "")).abs() - 1.0 < 1e-5);
    assert!(f1_answer("hello", "").abs() < 1e-5);
}

#[test]
fn test_rouge_1() {
    let hyp = "the cat sat on the mat";
    let ref_ = "the cat is on the mat";
    let r1 = rouge_n(hyp, ref_, 1);
    assert!(r1 > 0.5, "rouge-1 should be high for similar sentences");
}

#[test]
fn test_rouge_2() {
    let hyp = "the cat sat on the mat";
    let ref_ = "the cat sat on the mat";
    let r2 = rouge_n(hyp, ref_, 2);
    assert!(
        (r2 - 1.0).abs() < 1e-5,
        "rouge-2 should be 1.0 for identical"
    );
}

#[test]
fn test_rouge_n_empty() {
    assert_eq!(rouge_n("", "reference text", 1), 0.0);
    assert_eq!(rouge_n("hypothesis", "", 1), 0.0);
}
