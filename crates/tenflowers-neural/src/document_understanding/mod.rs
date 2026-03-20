//! Document Understanding Module
//!
//! Provides comprehensive document AI capabilities including:
//! - Layout-aware language models (LayoutLM)
//! - OCR post-processing utilities
//! - Table extraction and structured parsing
//! - Document classification
//! - Information extraction (NER for documents)
//! - Dense retrieval embeddings
//! - Span-extraction Question Answering
//! - Extractive and MMR summarization
//! - Form parsing and validation
//! - Document evaluation metrics

pub mod extraction;
pub mod summarization;

pub use extraction::{
    create_spans, extract_answer, DocEmbedder, DocEmbedderConfig, DocEntity, DocQaConfig,
    EntityType, InformationExtractor, InputSpan, PoolingStrategy,
};
pub use summarization::{
    exact_match, extractive_summarize, f1_answer, f1_score_ner, mmr_summarize, parse_form, rouge_n,
    validate_field, FieldType, FormField, SentenceScorer,
};

// ─── LayoutLM ────────────────────────────────────────────────────────────────

/// Configuration for a layout-aware language model.
#[derive(Debug, Clone)]
pub struct LayoutLmConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of transformer layers
    pub n_layers: usize,
}

impl LayoutLmConfig {
    /// Create a new LayoutLmConfig.
    pub fn new(
        vocab_size: usize,
        max_seq_len: usize,
        embed_dim: usize,
        n_heads: usize,
        n_layers: usize,
    ) -> Self {
        Self {
            vocab_size,
            max_seq_len,
            embed_dim,
            n_heads,
            n_layers,
        }
    }
}

/// Separate embeddings for each bounding-box coordinate dimension.
#[derive(Debug, Clone)]
pub struct BboxEmbedding {
    /// Embeddings for x-coordinate (left)
    pub x_embed: Vec<f32>,
    /// Embeddings for y-coordinate (top)
    pub y_embed: Vec<f32>,
    /// Embeddings for width
    pub w_embed: Vec<f32>,
    /// Embeddings for height
    pub h_embed: Vec<f32>,
    /// Embedding dimension
    embed_dim: usize,
    /// Coordinate range (max value, e.g. 1000)
    coord_range: usize,
}

impl BboxEmbedding {
    /// Create a new BboxEmbedding with given coordinate range and embedding dim.
    pub fn new(coord_range: usize, embed_dim: usize) -> Self {
        let init = |size: usize| -> Vec<f32> {
            (0..size)
                .map(|i| ((i as f32) * 0.01_f32).sin() * 0.02_f32)
                .collect()
        };
        let table_size = coord_range * embed_dim;
        Self {
            x_embed: init(table_size),
            y_embed: init(table_size),
            w_embed: init(table_size),
            h_embed: init(table_size),
            embed_dim,
            coord_range,
        }
    }

    /// Lookup a bbox embedding vector for given coordinates (x, y, w, h).
    /// Coordinates are clamped to [0, coord_range).
    pub fn lookup(&self, x: u32, y: u32, w: u32, h: u32) -> Vec<f32> {
        let clamp = |v: u32| (v as usize).min(self.coord_range.saturating_sub(1));
        let xi = clamp(x);
        let yi = clamp(y);
        let wi = clamp(w);
        let hi = clamp(h);

        let mut out = vec![0.0_f32; self.embed_dim];
        let d = self.embed_dim;
        for k in 0..d {
            let xe = self.x_embed.get(xi * d + k).copied().unwrap_or(0.0);
            let ye = self.y_embed.get(yi * d + k).copied().unwrap_or(0.0);
            let we = self.w_embed.get(wi * d + k).copied().unwrap_or(0.0);
            let he = self.h_embed.get(hi * d + k).copied().unwrap_or(0.0);
            out[k] = xe + ye + we + he;
        }
        out
    }
}

/// A single transformer layer conditioned on bounding-box information.
#[derive(Debug, Clone)]
pub struct LayoutLmLayer {
    /// Weight matrices for Q, K, V projections (flattened [embed_dim x embed_dim])
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    /// Output projection
    pub wo: Vec<f32>,
    /// Feed-forward weights (embed_dim → 4*embed_dim → embed_dim)
    pub ff1: Vec<f32>,
    pub ff2: Vec<f32>,
    /// Layer norm parameters
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub embed_dim: usize,
    pub n_heads: usize,
}

impl LayoutLmLayer {
    /// Create a new layer initialized with scaled sinusoidal values.
    pub fn new(embed_dim: usize, n_heads: usize) -> Self {
        let ff_dim = 4 * embed_dim;
        let scale = 0.02_f32;
        let sinit =
            |n: usize| -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.1).sin()) * scale).collect() };

        Self {
            wq: sinit(embed_dim * embed_dim),
            wk: sinit(embed_dim * embed_dim),
            wv: sinit(embed_dim * embed_dim),
            wo: sinit(embed_dim * embed_dim),
            ff1: sinit(embed_dim * ff_dim),
            ff2: sinit(ff_dim * embed_dim),
            ln1_w: vec![1.0_f32; embed_dim],
            ln1_b: vec![0.0_f32; embed_dim],
            ln2_w: vec![1.0_f32; embed_dim],
            ln2_b: vec![0.0_f32; embed_dim],
            embed_dim,
            n_heads,
        }
    }

    /// Apply layer norm to a vector.
    fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let std_inv = 1.0 / (var + eps).sqrt();
        x.iter()
            .enumerate()
            .map(|(i, v)| {
                let normed = (v - mean) * std_inv;
                normed * w.get(i).copied().unwrap_or(1.0) + b.get(i).copied().unwrap_or(0.0)
            })
            .collect()
    }

    /// Matrix-vector multiply: out = W * x, W is [out_dim x in_dim] row-major.
    fn matmul_vec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; out_dim];
        for i in 0..out_dim {
            let mut acc = 0.0_f32;
            for j in 0..in_dim {
                acc += w.get(i * in_dim + j).copied().unwrap_or(0.0)
                    * x.get(j).copied().unwrap_or(0.0);
            }
            out[i] = acc;
        }
        out
    }

    /// GELU activation approximation.
    fn gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x.powi(3))).tanh())
    }

    /// Forward pass with optional bbox bias (additive to attention logits).
    /// Input: sequence of token hidden states [seq_len x embed_dim].
    /// bbox_biases: optional [seq_len x seq_len] additive attention biases.
    pub fn forward(
        &self,
        hidden: &[Vec<f32>],
        bbox_biases: Option<&Vec<Vec<f32>>>,
    ) -> Vec<Vec<f32>> {
        let seq_len = hidden.len();
        let d = self.embed_dim;
        let h = self.n_heads;
        let head_dim = d / h.max(1);
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Compute Q, K, V projections for all tokens
        let queries: Vec<Vec<f32>> = hidden
            .iter()
            .map(|x| Self::matmul_vec(&self.wq, x, d, d))
            .collect();
        let keys: Vec<Vec<f32>> = hidden
            .iter()
            .map(|x| Self::matmul_vec(&self.wk, x, d, d))
            .collect();
        let values: Vec<Vec<f32>> = hidden
            .iter()
            .map(|x| Self::matmul_vec(&self.wv, x, d, d))
            .collect();

        // Multi-head attention
        let mut attn_out = vec![vec![0.0_f32; d]; seq_len];
        for hi in 0..h {
            let start = hi * head_dim;
            let end = start + head_dim;
            for i in 0..seq_len {
                let mut scores: Vec<f32> = (0..seq_len)
                    .map(|j| {
                        let q_slice = &queries[i][start..end];
                        let k_slice = &keys[j][start..end];
                        let dot: f32 = q_slice.iter().zip(k_slice.iter()).map(|(a, b)| a * b).sum();
                        let bias = bbox_biases
                            .and_then(|bb| bb.get(i))
                            .and_then(|row| row.get(j))
                            .copied()
                            .unwrap_or(0.0);
                        dot * scale + bias
                    })
                    .collect();

                // Softmax
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores.iter().map(|s| (s - max_s).exp()).sum();
                for s in &mut scores {
                    *s = (*s - max_s).exp() / exp_sum.max(1e-9);
                }

                // Weighted sum of values
                for j in 0..seq_len {
                    let v_slice = &values[j][start..end];
                    for (k, &v) in v_slice.iter().enumerate() {
                        if let Some(cell) = attn_out[i].get_mut(start + k) {
                            *cell += scores[j] * v;
                        }
                    }
                }
            }
        }

        // Apply output projection + residual + LN1
        let mut after_attn: Vec<Vec<f32>> = attn_out
            .iter()
            .enumerate()
            .map(|(i, ao)| {
                let proj = Self::matmul_vec(&self.wo, ao, d, d);
                let residual: Vec<f32> = proj
                    .iter()
                    .zip(hidden[i].iter())
                    .map(|(a, b)| a + b)
                    .collect();
                Self::layer_norm(&residual, &self.ln1_w, &self.ln1_b, 1e-5)
            })
            .collect();

        // Feed-forward
        let ff_dim = 4 * d;
        for i in 0..seq_len {
            let ff_in = after_attn[i].clone();
            let ff_mid: Vec<f32> = Self::matmul_vec(&self.ff1, &ff_in, ff_dim, d)
                .into_iter()
                .map(Self::gelu)
                .collect();
            let ff_out_proj = Self::matmul_vec(&self.ff2, &ff_mid, d, ff_dim);
            let residual: Vec<f32> = ff_out_proj
                .iter()
                .zip(after_attn[i].iter())
                .map(|(a, b)| a + b)
                .collect();
            after_attn[i] = Self::layer_norm(&residual, &self.ln2_w, &self.ln2_b, 1e-5);
        }

        after_attn
    }
}

/// Layout-aware language model combining token embeddings with spatial bbox embeddings.
pub struct LayoutLm {
    /// Token embedding table [vocab_size x embed_dim]
    pub token_embed: Vec<f32>,
    /// Position embedding table [max_seq_len x embed_dim]
    pub pos_embed: Vec<f32>,
    /// Bounding-box coordinate embeddings
    pub bbox_embed: BboxEmbedding,
    /// Transformer layers
    pub layers: Vec<LayoutLmLayer>,
    /// Configuration
    pub config: LayoutLmConfig,
}

impl LayoutLm {
    /// Create a new LayoutLm from configuration.
    pub fn new(config: LayoutLmConfig) -> Self {
        let d = config.embed_dim;
        let scale = 0.02_f32;
        let sinit =
            |n: usize| -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.07).sin()) * scale).collect() };
        let token_embed = sinit(config.vocab_size * d);
        let pos_embed = sinit(config.max_seq_len * d);
        let bbox_embed = BboxEmbedding::new(1001, d);
        let layers = (0..config.n_layers)
            .map(|_| LayoutLmLayer::new(d, config.n_heads))
            .collect();

        Self {
            token_embed,
            pos_embed,
            bbox_embed,
            layers,
            config,
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// - `tokens`: token ids
    /// - `bboxes`: per-token bounding boxes `(x, y, w, h)` in 0..1000 range
    ///
    /// Returns per-token hidden states of shape `[seq_len x embed_dim]`.
    pub fn forward(&self, tokens: &[usize], bboxes: &[(u32, u32, u32, u32)]) -> Vec<Vec<f32>> {
        let d = self.config.embed_dim;
        let seq_len = tokens.len();

        // Build initial hidden states: token_embed + pos_embed + bbox_embed
        let mut hidden: Vec<Vec<f32>> = tokens
            .iter()
            .enumerate()
            .map(|(pos, &tok_id)| {
                let tok_id = tok_id.min(self.config.vocab_size.saturating_sub(1));
                let pos_clamped = pos.min(self.config.max_seq_len.saturating_sub(1));
                let bbox = bboxes.get(pos).copied().unwrap_or((0, 0, 0, 0));

                let tok_vec: Vec<f32> = (0..d)
                    .map(|k| self.token_embed.get(tok_id * d + k).copied().unwrap_or(0.0))
                    .collect();
                let pos_vec: Vec<f32> = (0..d)
                    .map(|k| {
                        self.pos_embed
                            .get(pos_clamped * d + k)
                            .copied()
                            .unwrap_or(0.0)
                    })
                    .collect();
                let bbox_vec = self.bbox_embed.lookup(bbox.0, bbox.1, bbox.2, bbox.3);

                tok_vec
                    .iter()
                    .zip(pos_vec.iter())
                    .zip(bbox_vec.iter())
                    .map(|((t, p), b)| t + p + b)
                    .collect()
            })
            .collect();

        // Compute bbox-conditioned attention biases [seq_len x seq_len]
        let bbox_biases: Vec<Vec<f32>> = (0..seq_len)
            .map(|i| {
                let (xi, yi, _, _) = bboxes.get(i).copied().unwrap_or((0, 0, 0, 0));
                (0..seq_len)
                    .map(|j| {
                        let (xj, yj, _, _) = bboxes.get(j).copied().unwrap_or((0, 0, 0, 0));
                        let dx = (xi as f32 - xj as f32).powi(2);
                        let dy = (yi as f32 - yj as f32).powi(2);
                        let dist = (dx + dy).sqrt();
                        -dist * 0.001
                    })
                    .collect()
            })
            .collect();

        // Apply transformer layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, Some(&bbox_biases));
        }

        hidden
    }
}

// ─── DocumentOCR ─────────────────────────────────────────────────────────────

/// A single OCR detection result.
#[derive(Debug, Clone)]
pub struct OcrBox {
    /// Recognized text
    pub text: String,
    /// Bounding box (x, y, width, height) in pixel coordinates
    pub bbox: (f32, f32, f32, f32),
    /// Recognition confidence in [0, 1]
    pub confidence: f32,
}

impl OcrBox {
    /// Create a new OcrBox.
    pub fn new(text: impl Into<String>, bbox: (f32, f32, f32, f32), confidence: f32) -> Self {
        Self {
            text: text.into(),
            bbox,
            confidence,
        }
    }

    /// Return the y-center of the box.
    #[inline]
    pub fn y_center(&self) -> f32 {
        self.bbox.1 + self.bbox.3 * 0.5
    }

    /// Return the x-center of the box.
    #[inline]
    pub fn x_center(&self) -> f32 {
        self.bbox.0 + self.bbox.2 * 0.5
    }
}

/// Sort OCR boxes in natural reading order: top-to-bottom, with left-to-right
/// column detection using a horizontal gap heuristic.
pub fn sort_reading_order(boxes: &mut [OcrBox]) {
    if boxes.is_empty() {
        return;
    }

    let mut heights: Vec<f32> = boxes.iter().map(|b| b.bbox.3).collect();
    heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_h = heights[heights.len() / 2];
    let line_thresh = median_h * 0.5;

    boxes.sort_by(|a, b| {
        let ay = a.bbox.1;
        let by_ = b.bbox.1;
        let diff = ay - by_;
        if diff.abs() < line_thresh {
            a.bbox
                .0
                .partial_cmp(&b.bbox.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        } else {
            ay.partial_cmp(&by_).unwrap_or(std::cmp::Ordering::Equal)
        }
    });
}

/// Merge OCR boxes whose y-centers are within `y_threshold` into text lines.
pub fn merge_lines(boxes: &[OcrBox], y_threshold: f32) -> Vec<String> {
    if boxes.is_empty() {
        return Vec::new();
    }

    let mut lines: Vec<(f32, Vec<&OcrBox>)> = Vec::new();

    for b in boxes {
        let y = b.y_center();
        let matched = lines
            .iter_mut()
            .find(|(ly, _)| (y - ly).abs() < y_threshold);
        match matched {
            Some((ref mut ly, ref mut members)) => {
                members.push(b);
                *ly = (*ly + y) / 2.0;
            }
            None => {
                lines.push((y, vec![b]));
            }
        }
    }

    lines.sort_by(|(ya, _), (yb, _)| ya.partial_cmp(yb).unwrap_or(std::cmp::Ordering::Equal));

    lines
        .into_iter()
        .map(|(_, mut members)| {
            members.sort_by(|a, b| {
                a.bbox
                    .0
                    .partial_cmp(&b.bbox.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            members
                .iter()
                .map(|b| b.text.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

/// Normalize a pixel bounding box to the 0-1000 range used by LayoutLM.
pub fn normalize_bbox(
    bbox: (f32, f32, f32, f32),
    page_w: f32,
    page_h: f32,
) -> (u32, u32, u32, u32) {
    let pw = page_w.max(1.0);
    let ph = page_h.max(1.0);
    let clamp = |v: f32| (v.clamp(0.0, 1000.0)) as u32;
    (
        clamp(bbox.0 / pw * 1000.0),
        clamp(bbox.1 / ph * 1000.0),
        clamp(bbox.2 / pw * 1000.0),
        clamp(bbox.3 / ph * 1000.0),
    )
}

// ─── TableExtractor ──────────────────────────────────────────────────────────

/// A single cell in an extracted table.
#[derive(Debug, Clone)]
pub struct TableCell {
    /// Text content of the cell
    pub text: String,
    /// Row index (0-based)
    pub row: usize,
    /// Column index (0-based)
    pub col: usize,
    /// Number of rows spanned
    pub row_span: usize,
    /// Number of columns spanned
    pub col_span: usize,
}

impl TableCell {
    /// Create a simple single-cell entry.
    pub fn new(text: impl Into<String>, row: usize, col: usize) -> Self {
        Self {
            text: text.into(),
            row,
            col,
            row_span: 1,
            col_span: 1,
        }
    }
}

/// A structured table extracted from a document.
#[derive(Debug, Clone)]
pub struct Table {
    /// All cells in the table
    pub cells: Vec<TableCell>,
    /// Number of rows
    pub n_rows: usize,
    /// Number of columns
    pub n_cols: usize,
}

impl Table {
    /// Create a new empty table.
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        Self {
            cells: Vec::new(),
            n_rows,
            n_cols,
        }
    }

    /// Get text at (row, col), returns empty string if not found.
    pub fn get(&self, row: usize, col: usize) -> &str {
        self.cells
            .iter()
            .find(|c| c.row == row && c.col == col)
            .map(|c| c.text.as_str())
            .unwrap_or("")
    }
}

/// Detect table structure from OCR boxes by clustering into rows and columns.
pub fn detect_table_structure(boxes: &[OcrBox]) -> Vec<Table> {
    if boxes.is_empty() {
        return Vec::new();
    }

    let mut heights: Vec<f32> = boxes.iter().map(|b| b.bbox.3).collect();
    heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_h = heights[heights.len() / 2];
    let row_thresh = median_h * 0.6;

    let mut row_bands: Vec<(f32, Vec<&OcrBox>)> = Vec::new();
    for b in boxes {
        let y = b.y_center();
        let found = row_bands
            .iter_mut()
            .find(|(ry, _)| (y - ry).abs() < row_thresh);
        match found {
            Some((ref mut ry, ref mut members)) => {
                members.push(b);
                *ry = (*ry + y) / 2.0;
            }
            None => row_bands.push((y, vec![b])),
        }
    }
    row_bands.sort_by(|(ya, _), (yb, _)| ya.partial_cmp(yb).unwrap_or(std::cmp::Ordering::Equal));

    let n_rows = row_bands.len();

    let mut all_x: Vec<f32> = boxes.iter().map(|b| b.bbox.0).collect();
    all_x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_x.dedup_by(|a, b| (*a - *b).abs() < 10.0);
    let n_cols = all_x.len().max(1);

    let col_for_x = |x: f32| -> usize {
        all_x
            .iter()
            .enumerate()
            .min_by(|(_, &ax), (_, &bx)| {
                (x - ax)
                    .abs()
                    .partial_cmp(&(x - bx).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    };

    let mut table = Table::new(n_rows, n_cols);
    for (row_idx, (_, members)) in row_bands.iter().enumerate() {
        let mut sorted_members: Vec<&&OcrBox> = members.iter().collect();
        sorted_members.sort_by(|a, b| {
            a.bbox
                .0
                .partial_cmp(&b.bbox.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for m in sorted_members {
            let col_idx = col_for_x(m.bbox.0);
            table.cells.push(TableCell::new(&m.text, row_idx, col_idx));
        }
    }

    vec![table]
}

/// Convert a table to CSV string.
pub fn table_to_csv(table: &Table) -> String {
    let mut rows: Vec<String> = Vec::with_capacity(table.n_rows);
    for r in 0..table.n_rows {
        let cols: Vec<String> = (0..table.n_cols)
            .map(|c| {
                let text = table.get(r, c);
                if text.contains(',') || text.contains('"') {
                    format!("\"{}\"", text.replace('"', "\"\""))
                } else {
                    text.to_string()
                }
            })
            .collect();
        rows.push(cols.join(","));
    }
    rows.join("\n")
}

/// Convert a table to a JSON string (array of objects).
pub fn table_to_json(table: &Table) -> String {
    let headers: Vec<String> = if table.n_rows > 0 {
        (0..table.n_cols)
            .map(|c| table.get(0, c).to_string())
            .collect()
    } else {
        return "[]".to_string();
    };

    let mut objects: Vec<String> = Vec::new();
    for r in 1..table.n_rows {
        let fields: Vec<String> = (0..table.n_cols)
            .map(|c| {
                let key = escape_json_str(headers.get(c).map(|s| s.as_str()).unwrap_or(""));
                let val = escape_json_str(table.get(r, c));
                format!("\"{}\":\"{}\"", key, val)
            })
            .collect();
        objects.push(format!("{{{}}}", fields.join(",")));
    }
    format!("[{}]", objects.join(","))
}

pub(crate) fn escape_json_str(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// ─── DocumentClassifier ──────────────────────────────────────────────────────

/// Document type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DocClass {
    Invoice,
    Receipt,
    Contract,
    Resume,
    Report,
    Letter,
    Other,
}

impl DocClass {
    /// Return a human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            DocClass::Invoice => "Invoice",
            DocClass::Receipt => "Receipt",
            DocClass::Contract => "Contract",
            DocClass::Resume => "Resume",
            DocClass::Report => "Report",
            DocClass::Letter => "Letter",
            DocClass::Other => "Other",
        }
    }

    pub(crate) fn all() -> [DocClass; 7] {
        [
            DocClass::Invoice,
            DocClass::Receipt,
            DocClass::Contract,
            DocClass::Resume,
            DocClass::Report,
            DocClass::Letter,
            DocClass::Other,
        ]
    }
}

/// Document classifier: mean-pool token embeddings + linear head + argmax.
pub struct DocClassifier {
    /// Linear layer weights [n_classes x embed_dim]
    pub class_weights: Vec<f32>,
    /// Linear layer biases \[n_classes\]
    pub class_biases: Vec<f32>,
    /// Embedding dimension
    pub embed_dim: usize,
}

impl DocClassifier {
    const N_CLASSES: usize = 7;

    /// Create with given embedding dimension.
    pub fn new(embed_dim: usize) -> Self {
        let scale = 0.02_f32;
        let weights = (0..Self::N_CLASSES * embed_dim)
            .map(|i| ((i as f32 * 0.13).sin()) * scale)
            .collect();
        Self {
            class_weights: weights,
            class_biases: vec![0.0; Self::N_CLASSES],
            embed_dim,
        }
    }

    fn mean_pool(token_embeddings: &[Vec<f32>]) -> Vec<f32> {
        if token_embeddings.is_empty() {
            return Vec::new();
        }
        let d = token_embeddings[0].len();
        let n = token_embeddings.len() as f32;
        let mut out = vec![0.0_f32; d];
        for emb in token_embeddings {
            for (i, v) in emb.iter().enumerate() {
                if let Some(cell) = out.get_mut(i) {
                    *cell += v;
                }
            }
        }
        out.iter_mut().for_each(|v| *v /= n);
        out
    }

    fn logits(&self, pooled: &[f32]) -> Vec<f32> {
        let d = self.embed_dim;
        (0..Self::N_CLASSES)
            .map(|c| {
                let bias = self.class_biases.get(c).copied().unwrap_or(0.0);
                let dot: f32 = (0..d)
                    .map(|k| {
                        self.class_weights.get(c * d + k).copied().unwrap_or(0.0)
                            * pooled.get(k).copied().unwrap_or(0.0)
                    })
                    .sum();
                dot + bias
            })
            .collect()
    }

    /// Classify document; returns the predicted DocClass.
    pub fn classify(&self, token_embeddings: &[Vec<f32>]) -> DocClass {
        let pooled = Self::mean_pool(token_embeddings);
        let logits = self.logits(&pooled);
        let best = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(Self::N_CLASSES - 1);
        DocClass::all()[best.min(Self::N_CLASSES - 1)]
    }

    /// Classify with softmax probabilities for all classes.
    pub fn classify_with_confidence(&self, token_embeddings: &[Vec<f32>]) -> Vec<(DocClass, f32)> {
        let pooled = Self::mean_pool(token_embeddings);
        let logits = self.logits(&pooled);
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|l| (l - max_l).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|l| (l - max_l).exp() / exp_sum.max(1e-9))
            .collect();
        DocClass::all()
            .iter()
            .enumerate()
            .map(|(i, &cls)| (cls, probs.get(i).copied().unwrap_or(0.0)))
            .collect()
    }
}

#[cfg(test)]
mod tests;
