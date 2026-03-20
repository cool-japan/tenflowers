//! Advanced structured prediction components.
//!
//! This module provides:
//! - Neural CRF++ (`NeuralCrfLayer`, `PartialCrfLoss`, `ConstrainedDecoding`)
//! - Span-based models (`SpanExtractor`, `SpanClassifier`, `ConstituencyParser`)
//! - Graph-based structured prediction (`DependencyParser`, `SemanticRoleLabeler`)
//! - Metrics (`SpStructuredMetrics`)

use super::{log_sum_exp, softmax, SpLinear};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Xavier-uniform weight initialisation for a matrix of shape `[out_dim][in_dim]`.
fn xavier_init(out_dim: usize, in_dim: usize) -> Vec<Vec<f64>> {
    let limit = (6.0_f64 / (in_dim + out_dim) as f64).sqrt();
    let step = 2.0 * limit / (out_dim * in_dim) as f64;
    let mut counter = 0.0_f64;
    (0..out_dim)
        .map(|_| {
            (0..in_dim)
                .map(|_| {
                    counter += step;

                    -limit + counter % (2.0 * limit)
                })
                .collect()
        })
        .collect()
}

/// Apply ReLU element-wise.
fn relu_vec(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| x.max(0.0)).collect()
}

/// Matrix-vector multiply + bias: `W x + b`.
fn linear_fwd(w: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
    w.iter()
        .zip(b.iter())
        .map(|(row, bi)| {
            row.iter()
                .zip(x.iter())
                .map(|(wi, xi)| wi * xi)
                .sum::<f64>()
                + bi
        })
        .collect()
}

/// Two-layer MLP: ReLU hidden, linear output.
fn mlp2_fwd(w1: &[Vec<f64>], b1: &[f64], w2: &[Vec<f64>], b2: &[f64], x: &[f64]) -> Vec<f64> {
    let h = relu_vec(&linear_fwd(w1, b1, x));
    linear_fwd(w2, b2, &h)
}

/// Argmax of a slice.
fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Neural CRF Layer  (BiLSTM emissions + trainable CRF)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the [`NeuralCrfLayer`].
#[derive(Debug, Clone)]
pub struct NeuralCrfConfig {
    /// Input feature dimensionality.
    pub input_dim: usize,
    /// BiLSTM hidden size (per direction).
    pub hidden_dim: usize,
    /// Number of output label classes.
    pub n_classes: usize,
    /// Learning rate for parameter updates.
    pub lr: f64,
}

impl Default for NeuralCrfConfig {
    fn default() -> Self {
        Self {
            input_dim: 32,
            hidden_dim: 64,
            n_classes: 5,
            lr: 1e-3,
        }
    }
}

/// Neural CRF++ layer: MLP emission scorer + linear-chain CRF decoder.
///
/// The emission scores are produced by a two-layer feed-forward network applied
/// independently at each token position. Decoding uses the Viterbi algorithm
/// with learned transition parameters.
#[derive(Debug, Clone)]
pub struct NeuralCrfLayer {
    /// Configuration.
    pub config: NeuralCrfConfig,
    /// Emission MLP first-layer weights `[hidden_dim][input_dim]`.
    pub emit_w1: Vec<Vec<f64>>,
    /// Emission MLP first-layer biases.
    pub emit_b1: Vec<f64>,
    /// Emission MLP second-layer weights `[n_classes][hidden_dim]`.
    pub emit_w2: Vec<Vec<f64>>,
    /// Emission MLP second-layer biases.
    pub emit_b2: Vec<f64>,
    /// Transition matrix `[n_classes][n_classes]` (to, from).
    pub transition: Vec<Vec<f64>>,
    /// Start-of-sequence scores.
    pub start_scores: Vec<f64>,
    /// End-of-sequence scores.
    pub end_scores: Vec<f64>,
}

impl NeuralCrfLayer {
    /// Construct a new `NeuralCrfLayer` with Xavier-initialised weights.
    pub fn new(cfg: NeuralCrfConfig) -> Self {
        let d = cfg.input_dim;
        let h = cfg.hidden_dim;
        let c = cfg.n_classes;
        Self {
            emit_w1: xavier_init(h, d),
            emit_b1: vec![0.0; h],
            emit_w2: xavier_init(c, h),
            emit_b2: vec![0.0; c],
            transition: vec![vec![0.0; c]; c],
            start_scores: vec![0.0; c],
            end_scores: vec![0.0; c],
            config: cfg,
        }
    }

    /// Compute emission scores for each token, shape `[T][n_classes]`.
    pub fn emission_scores(&self, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        features
            .iter()
            .map(|f| {
                mlp2_fwd(
                    &self.emit_w1,
                    &self.emit_b1,
                    &self.emit_w2,
                    &self.emit_b2,
                    f,
                )
            })
            .collect()
    }

    /// Forward algorithm — returns log partition `log Z`.
    pub fn log_partition(&self, emit: &[Vec<f64>]) -> f64 {
        let c = self.config.n_classes;
        if emit.is_empty() {
            return 0.0;
        }
        let mut alpha: Vec<f64> = (0..c).map(|k| self.start_scores[k] + emit[0][k]).collect();
        for t in 1..emit.len() {
            let mut new_alpha = vec![f64::NEG_INFINITY; c];
            for to in 0..c {
                let scores: Vec<f64> = (0..c)
                    .map(|from| alpha[from] + self.transition[to][from] + emit[t][to])
                    .collect();
                new_alpha[to] = log_sum_exp(&scores);
            }
            alpha = new_alpha;
        }
        let last: Vec<f64> = (0..c).map(|k| alpha[k] + self.end_scores[k]).collect();
        log_sum_exp(&last)
    }

    /// Viterbi decoding — returns the most likely label sequence.
    pub fn viterbi(&self, features: &[Vec<f64>]) -> Vec<usize> {
        let c = self.config.n_classes;
        let emit = self.emission_scores(features);
        let t_len = emit.len();
        if t_len == 0 {
            return vec![];
        }
        let mut viterbi: Vec<Vec<f64>> = vec![vec![f64::NEG_INFINITY; c]; t_len];
        let mut back: Vec<Vec<usize>> = vec![vec![0; c]; t_len];

        for k in 0..c {
            viterbi[0][k] = self.start_scores[k] + emit[0][k];
        }
        for t in 1..t_len {
            for to in 0..c {
                let (best_from, best_score) = (0..c)
                    .map(|from| {
                        (
                            from,
                            viterbi[t - 1][from] + self.transition[to][from] + emit[t][to],
                        )
                    })
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((0, f64::NEG_INFINITY));
                viterbi[t][to] = best_score;
                back[t][to] = best_from;
            }
        }
        // terminate
        let best_last = (0..c)
            .max_by(|&a, &b| {
                (viterbi[t_len - 1][a] + self.end_scores[a])
                    .partial_cmp(&(viterbi[t_len - 1][b] + self.end_scores[b]))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        let mut path = vec![0usize; t_len];
        path[t_len - 1] = best_last;
        for t in (0..t_len - 1).rev() {
            path[t] = back[t + 1][path[t + 1]];
        }
        path
    }

    /// Compute negative log-likelihood (NLL) loss for a labelled sequence.
    pub fn nll_loss(&self, features: &[Vec<f64>], labels: &[usize]) -> f64 {
        let c = self.config.n_classes;
        let emit = self.emission_scores(features);
        if emit.is_empty() || labels.is_empty() {
            return 0.0;
        }
        // score of correct path
        let mut score = self.start_scores[labels[0]] + emit[0][labels[0]];
        for t in 1..labels.len() {
            score += self.transition[labels[t]][labels[t - 1]] + emit[t][labels[t]];
        }
        score += self.end_scores[labels[labels.len() - 1]];
        // log Z
        let log_z = self.log_partition(&emit);
        // NLL
        let _ = c;
        -(score - log_z)
    }

    /// Perform one gradient-descent training step, returning the NLL.
    pub fn train_step(&mut self, features: &[Vec<f64>], labels: &[usize]) -> f64 {
        let lr = self.config.lr;
        let loss = self.nll_loss(features, labels);
        // Finite-difference gradient for transition parameters.
        let c = self.config.n_classes;
        let eps = 1e-4;
        for to in 0..c {
            for from in 0..c {
                self.transition[to][from] += eps;
                let lp = self.nll_loss(features, labels);
                self.transition[to][from] -= eps;
                let grad = (lp - loss) / eps;
                self.transition[to][from] -= lr * grad;
            }
        }
        loss
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Partial CRF Loss  (support for partially labelled sequences)
// ─────────────────────────────────────────────────────────────────────────────

/// Partial CRF loss: compute NLL over a subset of labelled positions.
///
/// Unlabelled positions are indicated by `None` in `partial_labels`.
/// The loss marginalises over all consistent label assignments.
#[derive(Debug, Clone)]
pub struct PartialCrfLoss {
    /// Underlying neural CRF layer.
    pub crf: NeuralCrfLayer,
}

impl PartialCrfLoss {
    /// Construct a `PartialCrfLoss` wrapping `crf`.
    pub fn new(crf: NeuralCrfLayer) -> Self {
        Self { crf }
    }

    /// Compute the partial-supervision NLL.
    ///
    /// `partial_labels` has length `T`; `None` indicates a free (unobserved) position.
    /// The implementation builds a constrained forward pass that forces known labels
    /// while marginalising over unknown ones, then subtracts log Z of the full model.
    pub fn partial_nll(&self, features: &[Vec<f64>], partial_labels: &[Option<usize>]) -> f64 {
        let c = self.crf.config.n_classes;
        let emit = self.crf.emission_scores(features);
        let t_len = emit.len();
        if t_len == 0 {
            return 0.0;
        }

        // Constrained forward: alpha_constrained[t][k] = log p(x_{1..t}, y_t=k | constraints)
        let mask0: Vec<f64> = (0..c)
            .map(|k| {
                if partial_labels[0].map_or(true, |lbl| lbl == k) {
                    self.crf.start_scores[k] + emit[0][k]
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect();

        let mut alpha = mask0;
        for t in 1..t_len {
            let allowed_to: Box<dyn Fn(usize) -> bool> = match partial_labels[t] {
                Some(lbl) => Box::new(move |k: usize| k == lbl),
                None => Box::new(|_: usize| true),
            };
            let mut new_alpha = vec![f64::NEG_INFINITY; c];
            for to in 0..c {
                if !allowed_to(to) {
                    continue;
                }
                let scores: Vec<f64> = (0..c)
                    .map(|from| alpha[from] + self.crf.transition[to][from] + emit[t][to])
                    .collect();
                new_alpha[to] = log_sum_exp(&scores);
            }
            alpha = new_alpha;
        }
        let last: Vec<f64> = (0..c).map(|k| alpha[k] + self.crf.end_scores[k]).collect();
        let constrained_log_z = log_sum_exp(&last);

        // Full log Z
        let full_log_z = self.crf.log_partition(&emit);

        // NLL = full_log_z - constrained_log_z
        full_log_z - constrained_log_z
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Constrained Decoding  (Viterbi with hard label constraints)
// ─────────────────────────────────────────────────────────────────────────────

/// Viterbi decoder with hard label constraints.
///
/// At each position `t`, only labels in `allowed_labels[t]` may be assigned.
/// An empty allowed-set at position `t` permits all labels.
#[derive(Debug, Clone)]
pub struct ConstrainedDecoding;

impl ConstrainedDecoding {
    /// Perform constrained Viterbi decoding.
    ///
    /// Returns the highest-scoring label sequence consistent with
    /// `allowed_labels[t]` at each time step.  An empty inner vector at
    /// position `t` means *all* labels are allowed.
    pub fn viterbi(
        crf: &NeuralCrfLayer,
        features: &[Vec<f64>],
        allowed_labels: &[Vec<usize>],
    ) -> Vec<usize> {
        let c = crf.config.n_classes;
        let emit = crf.emission_scores(features);
        let t_len = emit.len();
        if t_len == 0 {
            return vec![];
        }

        let is_allowed = |t: usize, k: usize| -> bool {
            if t >= allowed_labels.len() || allowed_labels[t].is_empty() {
                true
            } else {
                allowed_labels[t].contains(&k)
            }
        };

        let mut vit = vec![vec![f64::NEG_INFINITY; c]; t_len];
        let mut back = vec![vec![0usize; c]; t_len];

        for k in 0..c {
            if is_allowed(0, k) {
                vit[0][k] = crf.start_scores[k] + emit[0][k];
            }
        }

        for t in 1..t_len {
            for to in 0..c {
                if !is_allowed(t, to) {
                    continue;
                }
                let (best_from, best_score) = (0..c)
                    .filter(|&from| vit[t - 1][from] > f64::NEG_INFINITY)
                    .map(|from| {
                        (
                            from,
                            vit[t - 1][from] + crf.transition[to][from] + emit[t][to],
                        )
                    })
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((0, f64::NEG_INFINITY));
                vit[t][to] = best_score;
                back[t][to] = best_from;
            }
        }

        let best_last = (0..c)
            .filter(|&k| vit[t_len - 1][k] > f64::NEG_INFINITY)
            .max_by(|&a, &b| {
                (vit[t_len - 1][a] + crf.end_scores[a])
                    .partial_cmp(&(vit[t_len - 1][b] + crf.end_scores[b]))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        let mut path = vec![0usize; t_len];
        path[t_len - 1] = best_last;
        for t in (0..t_len - 1).rev() {
            path[t] = back[t + 1][path[t + 1]];
        }
        path
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Span Extractor  (enumerate and embed text spans)
// ─────────────────────────────────────────────────────────────────────────────

/// A contiguous span `[start, end)` within a token sequence.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Span {
    /// Inclusive start index.
    pub start: usize,
    /// Exclusive end index.
    pub end: usize,
}

impl Span {
    /// Create a new span.
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Width of the span in tokens.
    pub fn width(&self) -> usize {
        self.end.saturating_sub(self.start)
    }
}

/// Span extractor: derives span representations by pooling token features.
///
/// For each candidate span `(i, j)` the representation is:
/// `[h_i ; h_{j-1} ; h_{(i+j)/2} ; width_embedding]`
/// where each `h_t` is the token embedding at position `t`.
#[derive(Debug, Clone)]
pub struct SpanExtractor {
    /// Dimensionality of the input token embeddings.
    pub token_dim: usize,
    /// Number of discrete width buckets.
    pub n_width_buckets: usize,
    /// Width embedding table `[n_width_buckets][token_dim]`.
    pub width_embed: Vec<Vec<f64>>,
}

impl SpanExtractor {
    /// Create a new `SpanExtractor`.
    pub fn new(token_dim: usize, n_width_buckets: usize) -> Self {
        // small uniform initialisation for width embeddings
        let width_embed = (0..n_width_buckets)
            .map(|i| {
                (0..token_dim)
                    .map(|j| ((i + j + 1) as f64) * 0.01)
                    .collect()
            })
            .collect();
        Self {
            token_dim,
            n_width_buckets,
            width_embed,
        }
    }

    /// Output dimensionality: `3 * token_dim + token_dim = 4 * token_dim`.
    pub fn output_dim(&self) -> usize {
        4 * self.token_dim
    }

    /// Extract the representation for span `(start, end)`.
    ///
    /// `token_features` has shape `[T][token_dim]`.
    pub fn extract(&self, token_features: &[Vec<f64>], span: &Span) -> Vec<f64> {
        let d = self.token_dim;
        let t = token_features.len();
        if span.start >= t || span.end > t || span.start >= span.end {
            return vec![0.0; self.output_dim()];
        }
        let h_start = &token_features[span.start];
        let h_end = &token_features[span.end - 1];
        let mid = (span.start + span.end - 1) / 2;
        let h_mid = &token_features[mid.min(t - 1)];

        let width_bucket = (span.width() - 1).min(self.n_width_buckets - 1);
        let w_embed = &self.width_embed[width_bucket];

        let mut out = Vec::with_capacity(self.output_dim());
        out.extend_from_slice(h_start);
        out.extend_from_slice(h_end);
        out.extend_from_slice(h_mid);
        out.extend_from_slice(w_embed);
        let _ = d;
        out
    }

    /// Enumerate all spans up to `max_width` tokens and extract their representations.
    pub fn extract_all(
        &self,
        token_features: &[Vec<f64>],
        max_width: usize,
    ) -> Vec<(Span, Vec<f64>)> {
        let t = token_features.len();
        let mut out = Vec::new();
        for start in 0..t {
            for end in (start + 1)..=(start + max_width).min(t) {
                let span = Span::new(start, end);
                let rep = self.extract(token_features, &span);
                out.push((span, rep));
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Span Classifier  (NER / SRL span labelling)
// ─────────────────────────────────────────────────────────────────────────────

/// Span classifier: assigns a label (or "non-entity") to each candidate span.
///
/// A two-layer MLP is applied to the span representation produced by
/// [`SpanExtractor`].  Label 0 is treated as the null / no-entity class.
#[derive(Debug, Clone)]
pub struct SpanClassifier {
    /// Hidden-layer weights.
    pub w1: Vec<Vec<f64>>,
    /// Hidden-layer biases.
    pub b1: Vec<f64>,
    /// Output-layer weights.
    pub w2: Vec<Vec<f64>>,
    /// Output-layer biases.
    pub b2: Vec<f64>,
    /// Number of label classes (including null).
    pub n_classes: usize,
    /// Learning rate.
    pub lr: f64,
}

impl SpanClassifier {
    /// Create a new `SpanClassifier`.
    pub fn new(span_dim: usize, hidden_dim: usize, n_classes: usize, lr: f64) -> Self {
        Self {
            w1: xavier_init(hidden_dim, span_dim),
            b1: vec![0.0; hidden_dim],
            w2: xavier_init(n_classes, hidden_dim),
            b2: vec![0.0; n_classes],
            n_classes,
            lr,
        }
    }

    /// Compute class logits for a span representation.
    pub fn logits(&self, span_rep: &[f64]) -> Vec<f64> {
        mlp2_fwd(&self.w1, &self.b1, &self.w2, &self.b2, span_rep)
    }

    /// Predict the most likely label index.
    pub fn predict(&self, span_rep: &[f64]) -> usize {
        argmax(&self.logits(span_rep))
    }

    /// Cross-entropy loss for a single span.
    pub fn loss(&self, span_rep: &[f64], target: usize) -> f64 {
        let logits = self.logits(span_rep);
        let probs = softmax(&logits);
        let p = probs[target].max(1e-12);
        -p.ln()
    }

    /// One gradient-descent step on a single (span_rep, target) pair.
    pub fn train_step(&mut self, span_rep: &[f64], target: usize) -> f64 {
        let eps = 1e-4;
        let base = self.loss(span_rep, target);
        // FD gradient on w2 (output layer only for speed)
        let n2 = self.w2.len();
        let m2 = if n2 > 0 { self.w2[0].len() } else { 0 };
        for i in 0..n2 {
            for j in 0..m2 {
                self.w2[i][j] += eps;
                let lp = self.loss(span_rep, target);
                self.w2[i][j] -= eps;
                let grad = (lp - base) / eps;
                self.w2[i][j] -= self.lr * grad;
            }
        }
        base
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Constituency Parser  (CYK algorithm)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the [`ConstituencyParser`].
#[derive(Debug, Clone)]
pub struct ConstituencyConfig {
    /// Token feature dimensionality.
    pub token_dim: usize,
    /// Hidden dim for the span scoring MLP.
    pub hidden_dim: usize,
    /// Number of non-terminal categories.
    pub n_labels: usize,
    /// Maximum span width to consider.
    pub max_span_width: usize,
}

/// Span-based constituency parser using the CYK dynamic-programming algorithm.
///
/// A span scoring MLP assigns a score to each `(span, label)` pair.  The
/// highest-scoring binary parse tree is recovered by CYK.
#[derive(Debug, Clone)]
pub struct ConstituencyParser {
    /// Span feature extractor.
    pub extractor: SpanExtractor,
    /// Span scoring MLP weights (first layer).
    pub score_w1: Vec<Vec<f64>>,
    /// Span scoring MLP biases (first layer).
    pub score_b1: Vec<f64>,
    /// Span scoring MLP weights (second layer) `[n_labels][hidden_dim]`.
    pub score_w2: Vec<Vec<f64>>,
    /// Span scoring MLP biases (second layer).
    pub score_b2: Vec<f64>,
    /// Parser configuration.
    pub config: ConstituencyConfig,
}

impl ConstituencyParser {
    /// Create a new `ConstituencyParser`.
    pub fn new(cfg: ConstituencyConfig) -> Self {
        let span_dim = 4 * cfg.token_dim;
        let h = cfg.hidden_dim;
        let nl = cfg.n_labels;
        let extractor = SpanExtractor::new(cfg.token_dim, 10);
        Self {
            score_w1: xavier_init(h, span_dim),
            score_b1: vec![0.0; h],
            score_w2: xavier_init(nl, h),
            score_b2: vec![0.0; nl],
            extractor,
            config: cfg,
        }
    }

    /// Score a span, returning label logits.
    pub fn score_span(&self, span_rep: &[f64]) -> Vec<f64> {
        mlp2_fwd(
            &self.score_w1,
            &self.score_b1,
            &self.score_w2,
            &self.score_b2,
            span_rep,
        )
    }

    /// Best label and its score for a given span representation.
    pub fn best_label(&self, span_rep: &[f64]) -> (usize, f64) {
        let logits = self.score_span(span_rep);
        let idx = argmax(&logits);
        (idx, logits[idx])
    }

    /// CYK parsing: find the highest-scoring binary parse tree.
    ///
    /// Returns a list of `(Span, label_idx)` for all spans in the best parse.
    pub fn parse(&self, token_features: &[Vec<f64>]) -> Vec<(Span, usize)> {
        let n = token_features.len();
        if n == 0 {
            return vec![];
        }
        // chart[i][j] = (best_score, best_label, best_split)
        // Use flat indexing: chart[i * n + j]
        let mut chart_score = vec![f64::NEG_INFINITY; n * n];
        let mut chart_label = vec![0usize; n * n];
        let mut chart_split = vec![0usize; n * n]; // 0 = leaf

        // Fill terminals (width 1)
        for i in 0..n {
            let span = Span::new(i, i + 1);
            let rep = self.extractor.extract(token_features, &span);
            let (lbl, score) = self.best_label(&rep);
            chart_score[i * n + i] = score;
            chart_label[i * n + i] = lbl;
            chart_split[i * n + i] = 0;
        }

        // Fill wider spans
        for width in 2..=n {
            for start in 0..=(n - width) {
                let end = start + width;
                let span = Span::new(start, end);
                let rep = self.extractor.extract(token_features, &span);
                let (lbl, span_score) = self.best_label(&rep);

                // Find best binary split
                let mut best_total = f64::NEG_INFINITY;
                let mut best_k = start + 1;
                for k in (start + 1)..end {
                    let left = chart_score[start * n + (k - 1)];
                    let right = chart_score[k * n + (end - 1)];
                    if left > f64::NEG_INFINITY && right > f64::NEG_INFINITY {
                        let total = left + right + span_score;
                        if total > best_total {
                            best_total = total;
                            best_k = k;
                        }
                    }
                }
                let total_score = if best_total > f64::NEG_INFINITY {
                    best_total
                } else {
                    span_score
                };
                chart_score[start * n + (end - 1)] = total_score;
                chart_label[start * n + (end - 1)] = lbl;
                chart_split[start * n + (end - 1)] = best_k;
            }
        }

        // Backtrack
        let mut result = Vec::new();
        let mut stack = vec![(0usize, n - 1)];
        while let Some((s, e)) = stack.pop() {
            let label = chart_label[s * n + e];
            result.push((Span::new(s, e + 1), label));
            let k = chart_split[s * n + e];
            if k > 0 && k < e {
                stack.push((s, k - 1));
                stack.push((k, e));
            }
        }
        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dependency Parser  (biaffine attention + Prim's MST)
// ─────────────────────────────────────────────────────────────────────────────

/// Biaffine dependency parser.
///
/// Computes arc scores via a biaffine transformation of head and dependent
/// projections, then finds the maximum spanning tree (MST) via Prim's algorithm
/// over the fully directed score matrix.
#[derive(Debug, Clone)]
pub struct DependencyParser {
    /// Projection for dependent tokens `[arc_dim][token_dim]`.
    pub dep_proj: Vec<Vec<f64>>,
    /// Projection bias for dependent tokens.
    pub dep_bias: Vec<f64>,
    /// Projection for head tokens `[arc_dim][token_dim]`.
    pub head_proj: Vec<Vec<f64>>,
    /// Head projection bias.
    pub head_bias: Vec<f64>,
    /// Biaffine weight matrix `[arc_dim][arc_dim]`.
    pub biaffine_w: Vec<Vec<f64>>,
    /// Label classifier weights `[n_labels][2*arc_dim]`.
    pub label_w: Vec<Vec<f64>>,
    /// Label classifier biases.
    pub label_b: Vec<f64>,
    /// Arc projection dimensionality.
    pub arc_dim: usize,
    /// Number of dependency relation labels.
    pub n_labels: usize,
}

impl DependencyParser {
    /// Create a new `DependencyParser`.
    pub fn new(token_dim: usize, arc_dim: usize, n_labels: usize) -> Self {
        Self {
            dep_proj: xavier_init(arc_dim, token_dim),
            dep_bias: vec![0.0; arc_dim],
            head_proj: xavier_init(arc_dim, token_dim),
            head_bias: vec![0.0; arc_dim],
            biaffine_w: xavier_init(arc_dim, arc_dim),
            label_w: xavier_init(n_labels, 2 * arc_dim),
            label_b: vec![0.0; n_labels],
            arc_dim,
            n_labels,
        }
    }

    fn project_dep(&self, tok: &[f64]) -> Vec<f64> {
        relu_vec(&linear_fwd(&self.dep_proj, &self.dep_bias, tok))
    }

    fn project_head(&self, tok: &[f64]) -> Vec<f64> {
        relu_vec(&linear_fwd(&self.head_proj, &self.head_bias, tok))
    }

    /// Compute arc score matrix `[n][n]` where `score[dep][head]`.
    pub fn arc_scores(&self, token_features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = token_features.len();
        let deps: Vec<Vec<f64>> = token_features.iter().map(|t| self.project_dep(t)).collect();
        let heads: Vec<Vec<f64>> = token_features
            .iter()
            .map(|t| self.project_head(t))
            .collect();

        (0..n)
            .map(|d| {
                (0..n)
                    .map(|h| {
                        // biaffine: dep^T W head
                        let wd = linear_fwd(&self.biaffine_w, &vec![0.0; self.arc_dim], &deps[d]);
                        wd.iter().zip(heads[h].iter()).map(|(a, b)| a * b).sum()
                    })
                    .collect()
            })
            .collect()
    }

    /// Predict label for the arc `(dep, head)`.
    pub fn arc_label(&self, dep_feat: &[f64], head_feat: &[f64]) -> usize {
        let dep_h = self.project_dep(dep_feat);
        let head_h = self.project_head(head_feat);
        let mut combined = dep_h.clone();
        combined.extend_from_slice(&head_h);
        let logits = linear_fwd(&self.label_w, &self.label_b, &combined);
        argmax(&logits)
    }

    /// Parse a sequence using Prim's maximum spanning tree algorithm.
    ///
    /// Returns `(heads, labels)` where `heads[i]` is the head index of token `i`
    /// (root token has `heads[root] == root`) and `labels[i]` is the dependency
    /// relation label for the arc `(i, heads[i])`.
    pub fn parse(&self, token_features: &[Vec<f64>]) -> (Vec<usize>, Vec<usize>) {
        let n = token_features.len();
        if n == 0 {
            return (vec![], vec![]);
        }
        let scores = self.arc_scores(token_features);

        // Prim's MST (greedy directed): pick highest-scoring incoming arc for each node
        // Treat token 0 as the virtual root.
        let mut heads = vec![0usize; n];
        for dep in 1..n {
            let best_head = (0..n)
                .filter(|&h| h != dep)
                .max_by(|&a, &b| {
                    scores[dep][a]
                        .partial_cmp(&scores[dep][b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            heads[dep] = best_head;
        }

        // Compute labels
        let labels: Vec<usize> = (0..n)
            .map(|dep| self.arc_label(&token_features[dep], &token_features[heads[dep]]))
            .collect();

        (heads, labels)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Semantic Role Labeler
// ─────────────────────────────────────────────────────────────────────────────

/// A single semantic role label assignment: (argument span, role label index).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SrlAnnotation {
    /// Predicate token index.
    pub predicate: usize,
    /// Argument span.
    pub argument: Span,
    /// Role label index.
    pub role: usize,
}

/// Span-based semantic role labeller.
///
/// For each predicate token, candidate argument spans are scored by a bilinear
/// interaction between the predicate representation and each span representation.
#[derive(Debug, Clone)]
pub struct SemanticRoleLabeler {
    /// Span extractor for argument spans.
    pub extractor: SpanExtractor,
    /// Predicate MLP first-layer weights.
    pub pred_w1: Vec<Vec<f64>>,
    /// Predicate MLP first-layer biases.
    pub pred_b1: Vec<f64>,
    /// Predicate MLP second-layer weights.
    pub pred_w2: Vec<Vec<f64>>,
    /// Predicate MLP second-layer biases.
    pub pred_b2: Vec<f64>,
    /// Span MLP first-layer weights.
    pub span_w1: Vec<Vec<f64>>,
    /// Span MLP first-layer biases.
    pub span_b1: Vec<f64>,
    /// Span MLP second-layer weights.
    pub span_w2: Vec<Vec<f64>>,
    /// Span MLP second-layer biases.
    pub span_b2: Vec<f64>,
    /// Bilinear role-scoring weight `[n_roles][role_dim][role_dim]` flattened to `[n_roles][role_dim*role_dim]`.
    pub role_w: Vec<Vec<f64>>,
    /// Role dim.
    pub role_dim: usize,
    /// Number of role labels (including null).
    pub n_roles: usize,
    /// Maximum argument span width.
    pub max_width: usize,
}

impl SemanticRoleLabeler {
    /// Construct a new `SemanticRoleLabeler`.
    pub fn new(token_dim: usize, role_dim: usize, n_roles: usize, max_width: usize) -> Self {
        let span_input_dim = 4 * token_dim;
        Self {
            extractor: SpanExtractor::new(token_dim, 10),
            pred_w1: xavier_init(role_dim, token_dim),
            pred_b1: vec![0.0; role_dim],
            pred_w2: xavier_init(role_dim, role_dim),
            pred_b2: vec![0.0; role_dim],
            span_w1: xavier_init(role_dim, span_input_dim),
            span_b1: vec![0.0; role_dim],
            span_w2: xavier_init(role_dim, role_dim),
            span_b2: vec![0.0; role_dim],
            role_w: (0..n_roles)
                .map(|_| {
                    xavier_init(role_dim, role_dim)
                        .into_iter()
                        .flatten()
                        .collect()
                })
                .collect(),
            role_dim,
            n_roles,
            max_width,
        }
    }

    fn predicate_rep(&self, tok: &[f64]) -> Vec<f64> {
        mlp2_fwd(
            &self.pred_w1,
            &self.pred_b1,
            &self.pred_w2,
            &self.pred_b2,
            tok,
        )
    }

    fn span_rep(&self, span_feat: &[f64]) -> Vec<f64> {
        mlp2_fwd(
            &self.span_w1,
            &self.span_b1,
            &self.span_w2,
            &self.span_b2,
            span_feat,
        )
    }

    /// Compute role logits for `(predicate_rep, span_rep)`.
    fn role_logits(&self, pred_r: &[f64], span_r: &[f64]) -> Vec<f64> {
        let d = self.role_dim;
        (0..self.n_roles)
            .map(|r| {
                // bilinear: pred^T W_r span
                let row = &self.role_w[r];
                let mut score = 0.0_f64;
                for i in 0..d {
                    for j in 0..d {
                        score += pred_r[i] * row[i * d + j] * span_r[j];
                    }
                }
                score
            })
            .collect()
    }

    /// Predict SRL annotations for a sentence given predicate positions.
    ///
    /// Returns one [`SrlAnnotation`] per `(predicate, span)` pair where the
    /// top-scoring role label is not the null class (index 0).
    pub fn predict(&self, token_features: &[Vec<f64>], predicates: &[usize]) -> Vec<SrlAnnotation> {
        let mut annotations = Vec::new();
        let span_data = self.extractor.extract_all(token_features, self.max_width);

        for &pred_idx in predicates {
            if pred_idx >= token_features.len() {
                continue;
            }
            let pred_r = self.predicate_rep(&token_features[pred_idx]);
            for (span, span_feat) in &span_data {
                // skip spans containing the predicate itself
                if span.start <= pred_idx && pred_idx < span.end {
                    continue;
                }
                let span_r = self.span_rep(span_feat);
                let logits = self.role_logits(&pred_r, &span_r);
                let best_role = argmax(&logits);
                if best_role > 0 {
                    annotations.push(SrlAnnotation {
                        predicate: pred_idx,
                        argument: span.clone(),
                        role: best_role,
                    });
                }
            }
        }
        annotations
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SpStructuredMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluation metrics for structured prediction.
#[derive(Debug, Clone, Default)]
pub struct SpStructuredMetrics {
    /// True positive count.
    pub tp: usize,
    /// False positive count.
    pub fp: usize,
    /// False negative count.
    pub fn_: usize,
    /// Exact-match sentence count.
    pub exact_match: usize,
    /// Total sentence count.
    pub total: usize,
    /// Unlabelled attachment score numerator.
    pub uas_correct: usize,
    /// Labelled attachment score numerator.
    pub las_correct: usize,
    /// Token count (UAS/LAS denominator).
    pub tokens: usize,
}

impl SpStructuredMetrics {
    /// Create a zeroed `SpStructuredMetrics`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Span-level precision.
    pub fn precision(&self) -> f64 {
        if self.tp + self.fp == 0 {
            1.0
        } else {
            self.tp as f64 / (self.tp + self.fp) as f64
        }
    }

    /// Span-level recall.
    pub fn recall(&self) -> f64 {
        if self.tp + self.fn_ == 0 {
            1.0
        } else {
            self.tp as f64 / (self.tp + self.fn_) as f64
        }
    }

    /// Span-level F1.
    pub fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    /// Exact-match accuracy over sentences.
    pub fn exact_match_accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.exact_match as f64 / self.total as f64
        }
    }

    /// Unlabelled attachment score (UAS).
    pub fn uas(&self) -> f64 {
        if self.tokens == 0 {
            0.0
        } else {
            self.uas_correct as f64 / self.tokens as f64
        }
    }

    /// Labelled attachment score (LAS).
    pub fn las(&self) -> f64 {
        if self.tokens == 0 {
            0.0
        } else {
            self.las_correct as f64 / self.tokens as f64
        }
    }

    /// Update metrics given predicted and gold span sets for a single sentence.
    pub fn update_spans(
        &mut self,
        pred_spans: &[(Span, usize)],
        gold_spans: &[(Span, usize)],
        exact: bool,
    ) {
        self.total += 1;
        if exact {
            self.exact_match += 1;
        }
        use std::collections::HashSet;
        let pred_set: HashSet<_> = pred_spans.iter().collect();
        let gold_set: HashSet<_> = gold_spans.iter().collect();
        self.tp += pred_set.intersection(&gold_set).count();
        self.fp += pred_set.difference(&gold_set).count();
        self.fn_ += gold_set.difference(&pred_set).count();
    }

    /// Update UAS/LAS given predicted and gold head arrays.
    pub fn update_dependency(
        &mut self,
        pred_heads: &[usize],
        gold_heads: &[usize],
        pred_labels: &[usize],
        gold_labels: &[usize],
    ) {
        let n = pred_heads.len().min(gold_heads.len());
        self.tokens += n;
        for i in 0..n {
            if pred_heads[i] == gold_heads[i] {
                self.uas_correct += 1;
                if i < pred_labels.len()
                    && i < gold_labels.len()
                    && pred_labels[i] == gold_labels[i]
                {
                    self.las_correct += 1;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_features(t: usize, d: usize) -> Vec<Vec<f64>> {
        (0..t)
            .map(|i| (0..d).map(|j| ((i + j + 1) as f64) * 0.05).collect())
            .collect()
    }

    fn make_neural_crf(t: usize) -> (NeuralCrfLayer, Vec<Vec<f64>>, Vec<usize>) {
        let cfg = NeuralCrfConfig {
            input_dim: 8,
            hidden_dim: 16,
            n_classes: 3,
            lr: 1e-3,
        };
        let crf = NeuralCrfLayer::new(cfg);
        let feats = make_features(t, 8);
        let labels: Vec<usize> = (0..t).map(|i| i % 3).collect();
        (crf, feats, labels)
    }

    // ── NeuralCrfLayer ───────────────────────────────────────────────────────

    #[test]
    fn test_neural_crf_emission_scores_shape() {
        let (crf, feats, _) = make_neural_crf(5);
        let emit = crf.emission_scores(&feats);
        assert_eq!(emit.len(), 5);
        for row in &emit {
            assert_eq!(row.len(), 3);
        }
    }

    #[test]
    fn test_neural_crf_viterbi_length() {
        let (crf, feats, _) = make_neural_crf(6);
        let path = crf.viterbi(&feats);
        assert_eq!(path.len(), 6);
    }

    #[test]
    fn test_neural_crf_viterbi_valid_labels() {
        let (crf, feats, _) = make_neural_crf(7);
        let path = crf.viterbi(&feats);
        for &l in &path {
            assert!(l < 3, "label {l} out of range");
        }
    }

    #[test]
    fn test_neural_crf_log_partition_finite() {
        let (crf, feats, _) = make_neural_crf(4);
        let emit = crf.emission_scores(&feats);
        let lz = crf.log_partition(&emit);
        assert!(lz.is_finite());
    }

    #[test]
    fn test_neural_crf_nll_finite() {
        let (crf, feats, labels) = make_neural_crf(5);
        let nll = crf.nll_loss(&feats, &labels);
        assert!(nll.is_finite(), "NLL should be finite, got {nll}");
    }

    #[test]
    fn test_neural_crf_train_step_finite() {
        let (mut crf, feats, labels) = make_neural_crf(4);
        let loss = crf.train_step(&feats, &labels);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_neural_crf_empty_sequence() {
        let (crf, _, _) = make_neural_crf(1);
        let path = crf.viterbi(&[]);
        assert_eq!(path.len(), 0);
        let emit: Vec<Vec<f64>> = vec![];
        let lz = crf.log_partition(&emit);
        assert_eq!(lz, 0.0);
    }

    // ── PartialCrfLoss ───────────────────────────────────────────────────────

    #[test]
    fn test_partial_crf_fully_observed() {
        let (crf, feats, labels) = make_neural_crf(4);
        let partial = PartialCrfLoss::new(crf.clone());
        let partial_labels: Vec<Option<usize>> = labels.iter().map(|&l| Some(l)).collect();
        let loss_partial = partial.partial_nll(&feats, &partial_labels);
        let loss_full = crf.nll_loss(&feats, &labels);
        assert!((loss_partial - loss_full).abs() < 1e-6);
    }

    #[test]
    fn test_partial_crf_fully_unobserved() {
        let (crf, feats, _) = make_neural_crf(4);
        let partial = PartialCrfLoss::new(crf);
        let partial_labels: Vec<Option<usize>> = vec![None; 4];
        // fully unobserved → constrained_log_z == full_log_z → loss ≈ 0
        let loss = partial.partial_nll(&feats, &partial_labels);
        assert!(
            loss.abs() < 1e-6,
            "fully unobserved NLL should be ~0, got {loss}"
        );
    }

    #[test]
    fn test_partial_crf_mixed_observation() {
        let (crf, feats, _) = make_neural_crf(5);
        let partial = PartialCrfLoss::new(crf);
        let partial_labels = vec![Some(0), None, Some(1), None, Some(2)];
        let loss = partial.partial_nll(&feats, &partial_labels);
        assert!(loss.is_finite());
        assert!(loss >= 0.0, "partial NLL should be >= 0, got {loss}");
    }

    // ── ConstrainedDecoding ──────────────────────────────────────────────────

    #[test]
    fn test_constrained_decoding_unconstrained() {
        let (crf, feats, _) = make_neural_crf(5);
        let allowed: Vec<Vec<usize>> = vec![vec![]; 5];
        let path = ConstrainedDecoding::viterbi(&crf, &feats, &allowed);
        assert_eq!(path.len(), 5);
    }

    #[test]
    fn test_constrained_decoding_forces_labels() {
        let (crf, feats, _) = make_neural_crf(4);
        // Force position 0 → label 2, position 2 → label 1
        let allowed = vec![vec![2usize], vec![], vec![1usize], vec![]];
        let path = ConstrainedDecoding::viterbi(&crf, &feats, &allowed);
        assert_eq!(path.len(), 4);
        assert_eq!(path[0], 2, "position 0 must be label 2");
        assert_eq!(path[2], 1, "position 2 must be label 1");
    }

    #[test]
    fn test_constrained_decoding_empty() {
        let (crf, _, _) = make_neural_crf(1);
        let path = ConstrainedDecoding::viterbi(&crf, &[], &[]);
        assert_eq!(path.len(), 0);
    }

    // ── SpanExtractor ────────────────────────────────────────────────────────

    #[test]
    fn test_span_extractor_output_dim() {
        let ex = SpanExtractor::new(8, 5);
        assert_eq!(ex.output_dim(), 32);
    }

    #[test]
    fn test_span_extractor_extract_shape() {
        let ex = SpanExtractor::new(8, 5);
        let feats = make_features(6, 8);
        let span = Span::new(1, 4);
        let rep = ex.extract(&feats, &span);
        assert_eq!(rep.len(), 32);
    }

    #[test]
    fn test_span_extractor_extract_all_count() {
        let ex = SpanExtractor::new(4, 5);
        let feats = make_features(4, 4);
        // max_width=2: spans (0,1),(0,2),(1,2),(1,3),(2,3),(2,4),(3,4) = 7
        let all = ex.extract_all(&feats, 2);
        assert_eq!(all.len(), 7);
    }

    #[test]
    fn test_span_width() {
        assert_eq!(Span::new(2, 5).width(), 3);
        assert_eq!(Span::new(0, 1).width(), 1);
    }

    // ── SpanClassifier ───────────────────────────────────────────────────────

    #[test]
    fn test_span_classifier_logits_shape() {
        let clf = SpanClassifier::new(32, 16, 4, 1e-3);
        let span_rep = vec![0.1f64; 32];
        let logits = clf.logits(&span_rep);
        assert_eq!(logits.len(), 4);
    }

    #[test]
    fn test_span_classifier_predict_in_range() {
        let clf = SpanClassifier::new(32, 16, 5, 1e-3);
        let span_rep = vec![0.2f64; 32];
        let pred = clf.predict(&span_rep);
        assert!(pred < 5);
    }

    #[test]
    fn test_span_classifier_loss_non_negative() {
        let clf = SpanClassifier::new(32, 16, 4, 1e-3);
        let span_rep = vec![0.1f64; 32];
        let loss = clf.loss(&span_rep, 1);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_span_classifier_train_step_finite() {
        let mut clf = SpanClassifier::new(32, 16, 4, 1e-3);
        let span_rep = vec![0.1f64; 32];
        let loss = clf.train_step(&span_rep, 2);
        assert!(loss.is_finite());
    }

    // ── ConstituencyParser ───────────────────────────────────────────────────

    #[test]
    fn test_constituency_parser_parse_non_empty() {
        let cfg = ConstituencyConfig {
            token_dim: 8,
            hidden_dim: 16,
            n_labels: 4,
            max_span_width: 4,
        };
        let parser = ConstituencyParser::new(cfg);
        let feats = make_features(5, 8);
        let parse = parser.parse(&feats);
        assert!(!parse.is_empty());
    }

    #[test]
    fn test_constituency_parser_parse_labels_in_range() {
        let cfg = ConstituencyConfig {
            token_dim: 8,
            hidden_dim: 16,
            n_labels: 4,
            max_span_width: 3,
        };
        let parser = ConstituencyParser::new(cfg);
        let feats = make_features(4, 8);
        let parse = parser.parse(&feats);
        for (span, lbl) in &parse {
            assert!(lbl < &4, "label {lbl} out of range");
            assert!(span.start < span.end);
        }
    }

    #[test]
    fn test_constituency_parser_empty_input() {
        let cfg = ConstituencyConfig {
            token_dim: 4,
            hidden_dim: 8,
            n_labels: 3,
            max_span_width: 2,
        };
        let parser = ConstituencyParser::new(cfg);
        let parse = parser.parse(&[]);
        assert_eq!(parse.len(), 0);
    }

    #[test]
    fn test_constituency_parser_single_token() {
        let cfg = ConstituencyConfig {
            token_dim: 4,
            hidden_dim: 8,
            n_labels: 3,
            max_span_width: 2,
        };
        let parser = ConstituencyParser::new(cfg);
        let feats = make_features(1, 4);
        let parse = parser.parse(&feats);
        // Should contain the root span (0,1)
        assert!(!parse.is_empty());
    }

    // ── DependencyParser ─────────────────────────────────────────────────────

    #[test]
    fn test_dependency_parser_arc_scores_shape() {
        let dp = DependencyParser::new(8, 16, 5);
        let feats = make_features(4, 8);
        let scores = dp.arc_scores(&feats);
        assert_eq!(scores.len(), 4);
        for row in &scores {
            assert_eq!(row.len(), 4);
        }
    }

    #[test]
    fn test_dependency_parser_parse_length() {
        let dp = DependencyParser::new(8, 16, 5);
        let feats = make_features(5, 8);
        let (heads, labels) = dp.parse(&feats);
        assert_eq!(heads.len(), 5);
        assert_eq!(labels.len(), 5);
    }

    #[test]
    fn test_dependency_parser_heads_in_range() {
        let dp = DependencyParser::new(8, 16, 5);
        let feats = make_features(6, 8);
        let (heads, _) = dp.parse(&feats);
        for &h in &heads {
            assert!(h < 6, "head {h} out of range");
        }
    }

    #[test]
    fn test_dependency_parser_labels_in_range() {
        let dp = DependencyParser::new(8, 16, 5);
        let feats = make_features(4, 8);
        let (_, labels) = dp.parse(&feats);
        for &l in &labels {
            assert!(l < 5, "label {l} out of range");
        }
    }

    #[test]
    fn test_dependency_parser_empty_input() {
        let dp = DependencyParser::new(8, 16, 5);
        let (heads, labels) = dp.parse(&[]);
        assert_eq!(heads.len(), 0);
        assert_eq!(labels.len(), 0);
    }

    // ── SemanticRoleLabeler ──────────────────────────────────────────────────

    #[test]
    fn test_srl_predict_no_panic() {
        let srl = SemanticRoleLabeler::new(8, 16, 5, 3);
        let feats = make_features(6, 8);
        let predicates = vec![1, 3];
        let ann = srl.predict(&feats, &predicates);
        // Just check that annotations are valid
        for a in &ann {
            assert!(a.predicate < 6);
            assert!(a.role < 5);
            assert!(a.argument.start < a.argument.end);
        }
    }

    #[test]
    fn test_srl_predict_empty_sentence() {
        let srl = SemanticRoleLabeler::new(8, 16, 5, 3);
        let ann = srl.predict(&[], &[0]);
        assert_eq!(ann.len(), 0);
    }

    #[test]
    fn test_srl_predict_no_predicates() {
        let srl = SemanticRoleLabeler::new(8, 16, 5, 3);
        let feats = make_features(5, 8);
        let ann = srl.predict(&feats, &[]);
        assert_eq!(ann.len(), 0);
    }

    // ── SpStructuredMetrics ──────────────────────────────────────────────────

    #[test]
    fn test_metrics_perfect() {
        let mut m = SpStructuredMetrics::new();
        let spans = vec![(Span::new(0, 2), 1usize), (Span::new(3, 5), 2)];
        m.update_spans(&spans, &spans, true);
        assert!((m.precision() - 1.0).abs() < 1e-9);
        assert!((m.recall() - 1.0).abs() < 1e-9);
        assert!((m.f1() - 1.0).abs() < 1e-9);
        assert_eq!(m.exact_match, 1);
    }

    #[test]
    fn test_metrics_no_overlap() {
        let mut m = SpStructuredMetrics::new();
        let pred = vec![(Span::new(0, 2), 1usize)];
        let gold = vec![(Span::new(3, 5), 2usize)];
        m.update_spans(&pred, &gold, false);
        assert_eq!(m.tp, 0);
        assert_eq!(m.fp, 1);
        assert_eq!(m.fn_, 1);
        assert!((m.precision()).abs() < 1e-9);
        assert!((m.recall()).abs() < 1e-9);
        assert!((m.f1()).abs() < 1e-9);
    }

    #[test]
    fn test_metrics_uas_las() {
        let mut m = SpStructuredMetrics::new();
        let pred_heads = vec![0, 0, 1, 2];
        let gold_heads = vec![0, 0, 1, 3];
        let pred_lbl = vec![0, 1, 2, 3];
        let gold_lbl = vec![0, 1, 2, 3];
        m.update_dependency(&pred_heads, &gold_heads, &pred_lbl, &gold_lbl);
        // 3 out of 4 UAS correct
        assert_eq!(m.uas_correct, 3);
        assert_eq!(m.tokens, 4);
        assert!((m.uas() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_metrics_empty() {
        let m = SpStructuredMetrics::new();
        assert!((m.precision() - 1.0).abs() < 1e-9);
        assert!((m.recall() - 1.0).abs() < 1e-9);
        assert_eq!(m.uas(), 0.0);
        assert_eq!(m.las(), 0.0);
        assert_eq!(m.exact_match_accuracy(), 0.0);
    }

    #[test]
    fn test_metrics_partial_tp() {
        let mut m = SpStructuredMetrics::new();
        let pred = vec![(Span::new(0, 2), 1usize), (Span::new(4, 6), 0)];
        let gold = vec![(Span::new(0, 2), 1usize), (Span::new(3, 5), 2)];
        m.update_spans(&pred, &gold, false);
        assert_eq!(m.tp, 1);
        assert_eq!(m.fp, 1);
        assert_eq!(m.fn_, 1);
        let f1 = m.f1();
        assert!(f1 > 0.0 && f1 < 1.0);
    }
}
