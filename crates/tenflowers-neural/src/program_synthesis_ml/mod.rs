//! # Program Synthesis ML
//!
//! ML methods for code understanding, analysis and program synthesis.
//!
//! This module provides:
//!
//! * **[`CodeTokenizer`]** — language-agnostic lexer for source code
//! * **[`ASTEncoder`]** — tree-positional encoding for AST nodes
//! * **[`CodeBert`]** — masked language model for code (CodeBERT-style)
//! * **[`CodeContrastive`]** — contrastive learning between code and docstrings
//! * **[`FlashFillSolver`]** — example-based string transformation via DSL
//! * **\[`NeuralProgramInducer`\]** — differentiable interpreter for program induction
//! * **[`CodeSummarizer`]** — pointer-generator network for code summarization
//! * **[`BugLocalizerGnn`]** — graph neural network to locate buggy lines
//! * **[`TestCaseGenerator`]** — mutation-based test case generation
//! * **[`CodeMetrics`]** — static analysis metrics (cyclomatic, Halstead, MI)
//!
//! ## Randomness Policy
//!
//! All randomness is sourced from `scirs2_core::random` — never from `rand`.


pub mod extensions;
pub use extensions::*;

pub mod neural_exec;
pub use neural_exec::*;

#[cfg(test)]
mod tests;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::{HashMap, VecDeque};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// 1. CodeTokenizer
// ---------------------------------------------------------------------------

/// Programming language for tokenization hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Python,
    Rust,
    C,
    Generic,
}

/// Coarse token category.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    Keyword,
    Identifier,
    Literal,
    Operator,
    Punct,
    Comment,
    Whitespace,
}

/// A single lexical token produced by [`CodeTokenizer`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub text: String,
}

/// Language-agnostic lexer for source code.
pub struct CodeTokenizer {
    python_keywords: Vec<&'static str>,
    rust_keywords: Vec<&'static str>,
    c_keywords: Vec<&'static str>,
}

impl CodeTokenizer {
    pub fn new() -> Self {
        Self {
            python_keywords: vec![
                "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
                "continue", "def", "del", "elif", "else", "except", "finally", "for", "from",
                "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass",
                "raise", "return", "try", "while", "with", "yield",
            ],
            rust_keywords: vec![
                "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else",
                "enum", "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop", "match",
                "mod", "move", "mut", "pub", "ref", "return", "self", "Self", "static", "struct",
                "super", "trait", "true", "type", "unsafe", "use", "where", "while",
            ],
            c_keywords: vec![
                "auto", "break", "case", "char", "const", "continue", "default", "do", "double",
                "else", "enum", "extern", "float", "for", "goto", "if", "int", "long", "register",
                "return", "short", "signed", "sizeof", "static", "struct", "switch", "typedef",
                "union", "unsigned", "void", "volatile", "while",
            ],
        }
    }

    /// Tokenize `source` according to the given `lang` hint.
    pub fn tokenize(&self, source: &str, lang: Language) -> Vec<Token> {
        let keywords: &[&str] = match lang {
            Language::Python => &self.python_keywords,
            Language::Rust => &self.rust_keywords,
            Language::C => &self.c_keywords,
            Language::Generic => &[],
        };
        let mut tokens = Vec::new();
        let chars: Vec<char> = source.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let ch = chars[i];

            // Whitespace
            if ch.is_whitespace() {
                let start = i;
                while i < chars.len() && chars[i].is_whitespace() {
                    i += 1;
                }
                tokens.push(Token {
                    kind: TokenKind::Whitespace,
                    text: chars[start..i].iter().collect(),
                });
                continue;
            }

            // Line comment (// or #)
            if ch == '/' && i + 1 < chars.len() && chars[i + 1] == '/' {
                let start = i;
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
                tokens.push(Token {
                    kind: TokenKind::Comment,
                    text: chars[start..i].iter().collect(),
                });
                continue;
            }
            if ch == '#' {
                let start = i;
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
                tokens.push(Token {
                    kind: TokenKind::Comment,
                    text: chars[start..i].iter().collect(),
                });
                continue;
            }
            // Block comment /* ... */
            if ch == '/' && i + 1 < chars.len() && chars[i + 1] == '*' {
                let start = i;
                i += 2;
                while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                    i += 1;
                }
                i += 2; // consume */
                tokens.push(Token {
                    kind: TokenKind::Comment,
                    text: chars[start..i.min(chars.len())].iter().collect(),
                });
                continue;
            }

            // String literal (double-quote)
            if ch == '"' {
                let start = i;
                i += 1;
                while i < chars.len() {
                    if chars[i] == '\\' {
                        i += 2;
                    } else if chars[i] == '"' {
                        i += 1;
                        break;
                    } else {
                        i += 1;
                    }
                }
                tokens.push(Token {
                    kind: TokenKind::Literal,
                    text: chars[start..i].iter().collect(),
                });
                continue;
            }
            // String literal (single-quote)
            if ch == '\'' {
                let start = i;
                i += 1;
                while i < chars.len() {
                    if chars[i] == '\\' {
                        i += 2;
                    } else if chars[i] == '\'' {
                        i += 1;
                        break;
                    } else {
                        i += 1;
                    }
                }
                tokens.push(Token {
                    kind: TokenKind::Literal,
                    text: chars[start..i].iter().collect(),
                });
                continue;
            }

            // Numeric literal
            if ch.is_ascii_digit()
                || (ch == '.' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit())
            {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_alphanumeric() || chars[i] == '.' || chars[i] == '_')
                {
                    i += 1;
                }
                tokens.push(Token {
                    kind: TokenKind::Literal,
                    text: chars[start..i].iter().collect(),
                });
                continue;
            }

            // Identifier or keyword
            if ch.is_alphabetic() || ch == '_' {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                let kind = if keywords.contains(&word.as_str()) {
                    TokenKind::Keyword
                } else {
                    TokenKind::Identifier
                };
                tokens.push(Token { kind, text: word });
                continue;
            }

            // Operators (multi-char first)
            let op2: Option<String> = if i + 1 < chars.len() {
                let s: String = chars[i..i + 2].iter().collect();
                match s.as_str() {
                    "==" | "!=" | "<=" | ">=" | "->" | "=>" | "::" | "&&" | "||" | "+=" | "-="
                    | "*=" | "/=" | "&=" | "|=" | "^=" | "<<" | ">>" => Some(s),
                    _ => None,
                }
            } else {
                None
            };
            if let Some(op) = op2 {
                i += 2;
                tokens.push(Token {
                    kind: TokenKind::Operator,
                    text: op,
                });
                continue;
            }

            // Single-char operators / punctuation
            let kind = match ch {
                '+' | '-' | '*' | '/' | '%' | '=' | '<' | '>' | '!' | '&' | '|' | '^' | '~' => {
                    TokenKind::Operator
                }
                '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | ':' | '.' | '@' => TokenKind::Punct,
                _ => TokenKind::Punct,
            };
            tokens.push(Token {
                kind,
                text: ch.to_string(),
            });
            i += 1;
        }
        tokens
    }
}

impl Default for CodeTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// 2. ASTEncoder
// ---------------------------------------------------------------------------

/// A node in an Abstract Syntax Tree.
#[derive(Debug, Clone)]
pub struct AstNode {
    pub node_type: String,
    pub children: Vec<AstNode>,
    pub depth: usize,
    pub sibling_idx: usize,
}

impl AstNode {
    pub fn new(node_type: impl Into<String>, depth: usize, sibling_idx: usize) -> Self {
        Self {
            node_type: node_type.into(),
            children: Vec::new(),
            depth,
            sibling_idx,
        }
    }

    pub fn add_child(&mut self, child: AstNode) {
        self.children.push(child);
    }
}

/// Tree-positional encoder for AST nodes using sinusoidal encoding.
pub struct ASTEncoder {
    pub embed_dim: usize,
}

impl ASTEncoder {
    pub fn new(embed_dim: usize) -> Self {
        Self { embed_dim }
    }

    /// Encode a single node using depth + sibling-index sinusoidal encoding.
    pub fn encode_node(&self, node: &AstNode, embed_dim: usize) -> Vec<f32> {
        let mut enc = vec![0.0f32; embed_dim];
        let d = embed_dim / 2;
        for k in 0..d {
            let denom = 10_000_f32.powf(2.0 * k as f32 / embed_dim as f32);
            // First half: depth encoding
            enc[2 * k] = (node.depth as f32 / denom).sin();
            if 2 * k + 1 < embed_dim {
                enc[2 * k + 1] = (node.depth as f32 / denom).cos();
            }
        }
        // Second half: sibling index encoding (interleaved with depth if dim is small)
        let half = embed_dim / 2;
        for k in 0..half {
            let denom = 10_000_f32.powf(2.0 * k as f32 / embed_dim as f32);
            let idx = half + 2 * k;
            if idx < embed_dim {
                enc[idx] = (node.sibling_idx as f32 / denom).sin();
            }
            if idx + 1 < embed_dim {
                enc[idx + 1] = (node.sibling_idx as f32 / denom).cos();
            }
        }
        // Add a node-type hash component into the first position
        let type_hash = node
            .node_type
            .bytes()
            .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
        let type_signal = ((type_hash as f32 / u32::MAX as f32) * 2.0 - 1.0) * 0.1;
        enc[0] += type_signal;
        enc
    }

    /// BFS-order encoding of an entire tree.
    pub fn encode_tree(&self, root: &AstNode) -> Vec<Vec<f32>> {
        let mut result = Vec::new();
        let mut queue: VecDeque<&AstNode> = VecDeque::new();
        queue.push_back(root);
        while let Some(node) = queue.pop_front() {
            result.push(self.encode_node(node, self.embed_dim));
            for child in &node.children {
                queue.push_back(child);
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// 3. CodeBert
// ---------------------------------------------------------------------------

/// Configuration for the CodeBERT masked language model.
#[derive(Debug, Clone)]
pub struct CodeBertConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub max_seq_len: usize,
}

impl CodeBertConfig {
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        n_heads: usize,
        n_layers: usize,
        max_seq_len: usize,
    ) -> Self {
        Self {
            vocab_size,
            embed_dim,
            n_heads,
            n_layers,
            max_seq_len,
        }
    }
}

/// CodeBERT: masked language model for code.
pub struct CodeBert {
    config: CodeBertConfig,
    /// Token embedding table: [vocab_size x embed_dim]
    token_embed: Vec<Vec<f32>>,
    /// Positional embedding table: [max_seq_len x embed_dim]
    pos_embed: Vec<Vec<f32>>,
    /// Per-layer attention weight matrices (Q, K, V, O) each [embed_dim x embed_dim]
    layers: Vec<CodeBertLayer>,
    /// Output projection [embed_dim x vocab_size]
    output_proj: Vec<Vec<f32>>,
}

struct CodeBertLayer {
    wq: Vec<Vec<f32>>,
    wk: Vec<Vec<f32>>,
    wv: Vec<Vec<f32>>,
    wo: Vec<Vec<f32>>,
    ff1: Vec<Vec<f32>>,
    ff2: Vec<Vec<f32>>,
}

impl CodeBertLayer {
    fn new(embed_dim: usize, ff_dim: usize, rng: &mut StdRng) -> Self {
        let init = |rows: usize, cols: usize, rng: &mut StdRng| -> Vec<Vec<f32>> {
            let scale = (2.0 / (rows + cols) as f32).sqrt();
            (0..rows)
                .map(|_| {
                    (0..cols)
                        .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
                        .collect()
                })
                .collect()
        };
        Self {
            wq: init(embed_dim, embed_dim, rng),
            wk: init(embed_dim, embed_dim, rng),
            wv: init(embed_dim, embed_dim, rng),
            wo: init(embed_dim, embed_dim, rng),
            // ff1: maps embed_dim -> ff_dim  (ff_dim rows, each of embed_dim cols)
            ff1: init(ff_dim, embed_dim, rng),
            // ff2: maps ff_dim -> embed_dim  (embed_dim rows, each of ff_dim cols)
            ff2: init(embed_dim, ff_dim, rng),
        }
    }

    fn forward(&self, x: &[Vec<f32>], n_heads: usize) -> Vec<Vec<f32>> {
        let seq_len = x.len();
        let embed_dim = x[0].len();
        let head_dim = embed_dim / n_heads;
        let scale = (head_dim as f32).sqrt();

        // Project Q, K, V
        let proj = |input: &[Vec<f32>], w: &[Vec<f32>]| -> Vec<Vec<f32>> {
            input
                .iter()
                .map(|xi| {
                    w.iter()
                        .map(|row| row.iter().zip(xi.iter()).map(|(a, b)| a * b).sum::<f32>())
                        .collect()
                })
                .collect()
        };
        let q = proj(x, &self.wq);
        let k = proj(x, &self.wk);
        let v = proj(x, &self.wv);

        // Multi-head attention
        let mut attn_out = vec![vec![0.0f32; embed_dim]; seq_len];
        for h in 0..n_heads {
            let s = h * head_dim;
            let e = s + head_dim;
            // Compute scores
            let mut scores = vec![vec![0.0f32; seq_len]; seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    scores[i][j] = q[i][s..e]
                        .iter()
                        .zip(k[j][s..e].iter())
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        / scale;
                }
            }
            // Softmax per row
            for i in 0..seq_len {
                let max_s = scores[i].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp: Vec<f32> = scores[i].iter().map(|&x| (x - max_s).exp()).collect();
                let sum: f32 = exp.iter().sum();
                for j in 0..seq_len {
                    scores[i][j] = exp[j] / sum.max(1e-8);
                }
            }
            // Weighted sum of V
            for i in 0..seq_len {
                for j in 0..seq_len {
                    for k_d in s..e {
                        attn_out[i][k_d] += scores[i][j] * v[j][k_d];
                    }
                }
            }
        }
        // Output projection
        let out = proj(&attn_out, &self.wo);
        // Residual + layer norm
        let mut res: Vec<Vec<f32>> = out
            .iter()
            .zip(x.iter())
            .map(|(o, xi)| o.iter().zip(xi.iter()).map(|(a, b)| a + b).collect())
            .collect();
        layer_norm_2d(&mut res);

        // Feed-forward
        let mut ff_out = vec![vec![0.0f32; embed_dim]; seq_len];
        for (i, xi) in res.iter().enumerate() {
            let h1: Vec<f32> = self
                .ff1
                .iter()
                .map(|row| {
                    let sum: f32 = row.iter().zip(xi.iter()).map(|(a, b)| a * b).sum();
                    gelu(sum)
                })
                .collect();
            for (j, row) in self.ff2.iter().enumerate() {
                ff_out[i][j] = row.iter().zip(h1.iter()).map(|(a, b)| a * b).sum::<f32>();
            }
        }
        // Residual + layer norm
        let mut final_out: Vec<Vec<f32>> = ff_out
            .iter()
            .zip(res.iter())
            .map(|(f, r)| f.iter().zip(r.iter()).map(|(a, b)| a + b).collect())
            .collect();
        layer_norm_2d(&mut final_out);
        final_out
    }
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

fn layer_norm_2d(x: &mut [Vec<f32>]) {
    for xi in x.iter_mut() {
        let mean = xi.iter().sum::<f32>() / xi.len() as f32;
        let var = xi.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / xi.len() as f32;
        let std = (var + 1e-5).sqrt();
        for v in xi.iter_mut() {
            *v = (*v - mean) / std;
        }
    }
}

impl CodeBert {
    /// Create a new CodeBERT model with Xavier-initialised weights.
    pub fn new(config: CodeBertConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let scale_emb = (1.0 / config.embed_dim as f32).sqrt();
        let token_embed: Vec<Vec<f32>> = (0..config.vocab_size)
            .map(|_| {
                (0..config.embed_dim)
                    .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale_emb)
                    .collect()
            })
            .collect();
        let pos_embed: Vec<Vec<f32>> = (0..config.max_seq_len)
            .map(|pos| {
                (0..config.embed_dim)
                    .enumerate()
                    .map(|(i, _)| {
                        let denom = 10_000_f32.powf(2.0 * (i / 2) as f32 / config.embed_dim as f32);
                        if i % 2 == 0 {
                            (pos as f32 / denom).sin()
                        } else {
                            (pos as f32 / denom).cos()
                        }
                    })
                    .collect()
            })
            .collect();
        let ff_dim = config.embed_dim * 4;
        let layers = (0..config.n_layers)
            .map(|_| CodeBertLayer::new(config.embed_dim, ff_dim, &mut rng))
            .collect();
        let scale_out = (2.0 / (config.embed_dim + config.vocab_size) as f32).sqrt();
        // output_proj: [vocab_size x embed_dim] — each row is a vocab entry's projection vector
        let output_proj: Vec<Vec<f32>> = (0..config.vocab_size)
            .map(|_| {
                (0..config.embed_dim)
                    .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale_out)
                    .collect()
            })
            .collect();
        Self {
            config,
            token_embed,
            pos_embed,
            layers,
            output_proj,
        }
    }

    /// Forward pass: returns contextual embeddings [seq_len x embed_dim].
    pub fn forward(&self, token_ids: &[usize]) -> Vec<Vec<f32>> {
        let seq_len = token_ids.len().min(self.config.max_seq_len);
        // Embedding lookup + positional
        let mut x: Vec<Vec<f32>> = token_ids[..seq_len]
            .iter()
            .enumerate()
            .map(|(pos, &tid)| {
                let t_idx = tid % self.config.vocab_size;
                self.token_embed[t_idx]
                    .iter()
                    .zip(self.pos_embed[pos].iter())
                    .map(|(t, p)| t + p)
                    .collect()
            })
            .collect();
        // Transformer layers
        for layer in &self.layers {
            x = layer.forward(&x, self.config.n_heads);
        }
        x
    }

    /// Compute per-position vocab logits for masked language modelling.
    pub fn mlm_logits(&self, hidden: &[Vec<f32>]) -> Vec<Vec<f32>> {
        hidden
            .iter()
            .map(|h| {
                self.output_proj
                    .iter()
                    .map(|row| row.iter().zip(h.iter()).map(|(a, b)| a * b).sum::<f32>())
                    .collect()
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// 4. CodeContrastive
// ---------------------------------------------------------------------------

/// Contrastive learning between code and natural-language docstrings.
pub struct CodeContrastive {
    pub embed_dim: usize,
    code_proj: Vec<Vec<f32>>,
    doc_proj: Vec<Vec<f32>>,
    token_embed: Vec<Vec<f32>>,
    vocab_size: usize,
}

impl CodeContrastive {
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(123);
        let scale = (1.0 / embed_dim as f32).sqrt();
        let random_matrix = |rng: &mut StdRng| -> Vec<Vec<f32>> {
            (0..embed_dim)
                .map(|_| {
                    (0..embed_dim)
                        .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
                        .collect()
                })
                .collect()
        };
        let token_embed: Vec<Vec<f32>> = (0..vocab_size)
            .map(|_| {
                (0..embed_dim)
                    .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
                    .collect()
            })
            .collect();
        Self {
            embed_dim,
            code_proj: random_matrix(&mut rng),
            doc_proj: random_matrix(&mut rng),
            token_embed,
            vocab_size,
        }
    }

    fn mean_pool(&self, tokens: &[usize]) -> Vec<f32> {
        if tokens.is_empty() {
            return vec![0.0; self.embed_dim];
        }
        let mut sum = vec![0.0f32; self.embed_dim];
        for &tid in tokens {
            let idx = tid % self.vocab_size;
            for (s, e) in sum.iter_mut().zip(self.token_embed[idx].iter()) {
                *s += e;
            }
        }
        let n = tokens.len() as f32;
        sum.iter_mut().for_each(|v| *v /= n);
        sum
    }

    fn project(vec: &[f32], mat: &[Vec<f32>]) -> Vec<f32> {
        mat.iter()
            .map(|row| row.iter().zip(vec.iter()).map(|(a, b)| a * b).sum::<f32>())
            .collect()
    }

    fn l2_norm(v: &mut [f32]) {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        v.iter_mut().for_each(|x| *x /= norm);
    }

    /// Encode a token sequence as a mean-pooled code embedding.
    pub fn encode_code(&self, tokens: &[usize]) -> Vec<f32> {
        let pooled = self.mean_pool(tokens);
        let mut proj = Self::project(&pooled, &self.code_proj);
        Self::l2_norm(&mut proj);
        proj
    }

    /// Encode a token sequence as a mean-pooled doc embedding.
    pub fn encode_doc(&self, tokens: &[usize]) -> Vec<f32> {
        let pooled = self.mean_pool(tokens);
        let mut proj = Self::project(&pooled, &self.doc_proj);
        Self::l2_norm(&mut proj);
        proj
    }

    /// InfoNCE contrastive loss with in-batch negatives.
    ///
    /// `code_embeds` and `doc_embeds` must have the same length N;
    /// diagonal entries are positive pairs and off-diagonal are negatives.
    pub fn contrastive_loss(
        &self,
        code_embeds: &[Vec<f32>],
        doc_embeds: &[Vec<f32>],
        temperature: f32,
    ) -> f32 {
        let n = code_embeds.len();
        if n == 0 {
            return 0.0;
        }
        let dot =
            |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };
        let mut total_loss = 0.0f32;
        for i in 0..n {
            // Code → Doc direction
            let logits: Vec<f32> = (0..n)
                .map(|j| dot(&code_embeds[i], &doc_embeds[j]) / temperature)
                .collect();
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
            let sum: f32 = exp.iter().sum();
            total_loss -= (exp[i] / sum.max(1e-8)).ln();
        }
        for j in 0..n {
            // Doc → Code direction
            let logits: Vec<f32> = (0..n)
                .map(|i| dot(&doc_embeds[j], &code_embeds[i]) / temperature)
                .collect();
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
            let sum: f32 = exp.iter().sum();
            total_loss -= (exp[j] / sum.max(1e-8)).ln();
        }
        total_loss / (2.0 * n as f32)
    }
}

// ---------------------------------------------------------------------------
// 5. FlashFillSolver
// ---------------------------------------------------------------------------

/// DSL operations for string transformations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StringDsl {
    /// Concatenate sub-results of two sub-programs.
    Concat(Box<StringDsl>, Box<StringDsl>),
    /// Extract substring: Substr(start, end) — both inclusive, 0-indexed.
    Substr(usize, usize),
    /// Replace all occurrences of first with second.
    Replace(String, String),
    /// Convert to uppercase.
    Upper,
    /// Convert to lowercase.
    Lower,
    /// Strip leading/trailing whitespace.
    Strip,
    /// Split on delimiter and return the n-th piece.
    Split(String, usize),
    /// Join a split result with a new delimiter (operates on whitespace split).
    Join(String),
    /// Apply a simple regex-derived constant extraction by index.
    Regex(String, usize),
}

/// A program in the string-transformation DSL.
#[derive(Debug, Clone)]
pub struct FlashFillProgram {
    pub ops: Vec<StringDsl>,
}

/// FlashFill-style example-based string synthesis.
pub struct FlashFillSolver;

impl FlashFillSolver {
    pub fn new() -> Self {
        Self
    }

    /// Execute a single DSL op on `input`.
    fn execute_op(op: &StringDsl, input: &str) -> String {
        match op {
            StringDsl::Concat(a, b) => {
                format!(
                    "{}{}",
                    Self::execute_op(a, input),
                    Self::execute_op(b, input)
                )
            }
            StringDsl::Substr(start, end) => {
                let chars: Vec<char> = input.chars().collect();
                let s = (*start).min(chars.len());
                let e = (*end + 1).min(chars.len());
                if s >= e {
                    String::new()
                } else {
                    chars[s..e].iter().collect()
                }
            }
            StringDsl::Replace(from, to) => input.replace(from.as_str(), to.as_str()),
            StringDsl::Upper => input.to_uppercase(),
            StringDsl::Lower => input.to_lowercase(),
            StringDsl::Strip => input.trim().to_string(),
            StringDsl::Split(delim, idx) => {
                let parts: Vec<&str> = input.split(delim.as_str()).collect();
                (*parts.get(*idx).unwrap_or(&"")).to_string()
            }
            StringDsl::Join(delim) => input
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(delim.as_str()),
            StringDsl::Regex(pattern, idx) => {
                // Simple pattern: treat pattern as a literal delimiter and extract idx-th piece
                let parts: Vec<&str> = input.split(pattern.as_str()).collect();
                parts.get(*idx).unwrap_or(&"").trim().to_string()
            }
        }
    }

    /// Execute a full program (sequential ops compose: each op applied to input).
    pub fn execute(program: &FlashFillProgram, input: &str) -> String {
        let mut current = input.to_string();
        for op in &program.ops {
            current = Self::execute_op(op, &current);
        }
        current
    }

    /// Verify a program against all examples.
    fn verify(program: &FlashFillProgram, examples: &[(&str, &str)]) -> bool {
        examples
            .iter()
            .all(|(inp, out)| Self::execute(program, inp) == *out)
    }

    /// Enumerate short programs and verify against examples.
    /// Returns the first program that satisfies all examples, if any.
    pub fn synthesize(&self, examples: &[(&str, &str)]) -> Option<FlashFillProgram> {
        if examples.is_empty() {
            return None;
        }

        // Build candidate atomic ops from example data
        let mut candidates: Vec<Vec<StringDsl>> = Vec::new();

        // Identity-like ops
        candidates.push(vec![StringDsl::Strip]);
        candidates.push(vec![StringDsl::Upper]);
        candidates.push(vec![StringDsl::Lower]);

        // Collect delimiters/replacements from examples
        let common_delimiters = [" ", "-", "_", ",", ".", "/", ":"];
        for &delim in &common_delimiters {
            for idx in 0..4 {
                candidates.push(vec![StringDsl::Split(delim.to_string(), idx)]);
            }
            candidates.push(vec![StringDsl::Join(delim.to_string())]);
        }

        // Substring ranges based on example output lengths
        let max_len = examples
            .iter()
            .map(|(i, _)| i.chars().count())
            .max()
            .unwrap_or(0);
        for start in 0..max_len.min(8) {
            for end in start..max_len.min(16) {
                candidates.push(vec![StringDsl::Substr(start, end)]);
            }
        }

        // Replace ops from example pairs
        for &(inp, out) in examples {
            // Simple prefix strip
            if out.len() < inp.len() && inp.starts_with(out) {
                // output is a prefix
            }
            // Replacement heuristics
            for &delim in &common_delimiters {
                if inp.contains(delim) {
                    for &new_delim in &common_delimiters {
                        if new_delim != delim {
                            candidates.push(vec![StringDsl::Replace(
                                delim.to_string(),
                                new_delim.to_string(),
                            )]);
                        }
                    }
                    candidates.push(vec![StringDsl::Replace(delim.to_string(), "".to_string())]);
                }
            }
        }

        // Two-op compositions
        let single_ops: Vec<StringDsl> = vec![StringDsl::Strip, StringDsl::Upper, StringDsl::Lower];
        for op1 in &single_ops {
            for op2 in &single_ops {
                candidates.push(vec![op1.clone(), op2.clone()]);
            }
        }
        // Split then join compositions
        for &delim in &common_delimiters {
            for &new_delim in &common_delimiters {
                candidates.push(vec![StringDsl::Join(new_delim.to_string())]);
                candidates.push(vec![
                    StringDsl::Split(delim.to_string(), 0),
                    StringDsl::Strip,
                ]);
            }
        }

        for ops in candidates {
            let program = FlashFillProgram { ops };
            if Self::verify(&program, examples) {
                return Some(program);
            }
        }
        None
    }
}

impl Default for FlashFillSolver {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// 6. NeuralProgramInducer
// ---------------------------------------------------------------------------

/// Opcodes for a simple register machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpCode {
    Add,
    Sub,
    Mul,
    Div,
    Copy,
    Max,
    Min,
    And,
    Or,
    Not,
}

/// A single instruction in the register machine.
#[derive(Debug, Clone)]
pub struct Instruction {
    pub op: OpCode,
    /// Argument indices into the register file.
    pub args: Vec<usize>,
}

/// Differentiable interpreter with soft-gated execution.
pub struct DifferentiableInterpreter {
    pub registers: Vec<f32>,
    pub program: Vec<Instruction>,
}

impl DifferentiableInterpreter {
    pub fn new(n_registers: usize, program: Vec<Instruction>) -> Self {
        Self {
            registers: vec![0.0; n_registers],
            program,
        }
    }

    /// Execute the program with soft (differentiable) semantics.
    ///
    /// Inputs are loaded into registers 0..input.len(). The function
    /// returns the final register file after execution.
    pub fn execute_soft(&mut self, input: &[f32]) -> Vec<f32> {
        // Load input into registers
        for (i, &v) in input.iter().enumerate() {
            if i < self.registers.len() {
                self.registers[i] = v;
            }
        }

        let n_regs = self.registers.len();
        let clamp = |v: f32| v.clamp(-1e6, 1e6);

        for instr in &self.program {
            let get = |idx: usize| -> f32 {
                if idx < n_regs {
                    self.registers[idx]
                } else {
                    0.0
                }
            };
            let a0 = instr.args.first().copied().unwrap_or(0);
            let a1 = instr.args.get(1).copied().unwrap_or(0);
            let a2 = instr.args.get(2).copied().unwrap_or(0);
            let result = match instr.op {
                OpCode::Add => clamp(get(a0) + get(a1)),
                OpCode::Sub => clamp(get(a0) - get(a1)),
                OpCode::Mul => {
                    // Soft mul: use tanh gate to prevent explosion
                    let product = get(a0) * get(a1);
                    clamp(product.tanh() * product.abs().sqrt())
                }
                OpCode::Div => {
                    let denom = get(a1);
                    if denom.abs() < 1e-7 {
                        0.0
                    } else {
                        clamp(get(a0) / denom)
                    }
                }
                OpCode::Copy => get(a0),
                OpCode::Max => get(a0).max(get(a1)),
                OpCode::Min => get(a0).min(get(a1)),
                OpCode::And => {
                    // Soft AND: product of sigmoids
                    let s0 = sigmoid_f32(get(a0));
                    let s1 = sigmoid_f32(get(a1));
                    s0 * s1
                }
                OpCode::Or => {
                    // Soft OR: 1 - (1-s0)*(1-s1)
                    let s0 = sigmoid_f32(get(a0));
                    let s1 = sigmoid_f32(get(a1));
                    1.0 - (1.0 - s0) * (1.0 - s1)
                }
                OpCode::Not => {
                    // Soft NOT: 1 - sigmoid
                    1.0 - sigmoid_f32(get(a0))
                }
            };
            if a2 < n_regs {
                self.registers[a2] = result;
            }
        }
        self.registers.clone()
    }
}

pub(crate) fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// 7. CodeSummarizer
// ---------------------------------------------------------------------------

/// Configuration for the pointer-generator summarizer.
#[derive(Debug, Clone)]
pub struct PointerGeneratorConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub attn_dim: usize,
}

/// Pointer-generator network for code summarization.
pub struct CodeSummarizer {
    config: PointerGeneratorConfig,
    /// Encoder embedding table [vocab_size x hidden_dim]
    encoder_embed: Vec<Vec<f32>>,
    /// Encoder GRU weights (simplified: W_in [hidden x hidden], W_rec [hidden x hidden])
    enc_w_in: Vec<Vec<f32>>,
    enc_w_rec: Vec<Vec<f32>>,
    /// Decoder embedding [vocab_size x hidden_dim]
    decoder_embed: Vec<Vec<f32>>,
    /// Decoder step weights
    dec_w_in: Vec<Vec<f32>>,
    dec_w_rec: Vec<Vec<f32>>,
    /// Attention: W_enc [attn x hidden], W_dec [attn x hidden], V [1 x attn]
    w_enc: Vec<Vec<f32>>,
    w_dec: Vec<Vec<f32>>,
    v_attn: Vec<f32>,
    /// Vocabulary distribution projection [vocab x hidden]
    w_vocab: Vec<Vec<f32>>,
    /// Copy gate weights [1 x hidden]
    w_copy_gate: Vec<f32>,
}

impl CodeSummarizer {
    pub fn new(config: PointerGeneratorConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(77);
        let h = config.hidden_dim;
        let v = config.vocab_size;
        let a = config.attn_dim;
        let scale = |d: usize| (1.0 / d as f32).sqrt();
        let mat = |rows: usize, cols: usize, rng: &mut StdRng| -> Vec<Vec<f32>> {
            let s = scale(rows);
            (0..rows)
                .map(|_| {
                    (0..cols)
                        .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * s)
                        .collect()
                })
                .collect()
        };
        let vec_init = |n: usize, rng: &mut StdRng| -> Vec<f32> {
            (0..n)
                .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale(n))
                .collect()
        };
        Self {
            encoder_embed: mat(v, h, &mut rng),
            enc_w_in: mat(h, h, &mut rng),
            enc_w_rec: mat(h, h, &mut rng),
            decoder_embed: mat(v, h, &mut rng),
            dec_w_in: mat(h, h, &mut rng),
            dec_w_rec: mat(h, h, &mut rng),
            w_enc: mat(a, h, &mut rng),
            w_dec: mat(a, h, &mut rng),
            v_attn: vec_init(a, &mut rng),
            w_vocab: mat(v, h, &mut rng),
            w_copy_gate: vec_init(h, &mut rng),
            config,
        }
    }

    fn mat_vec(mat: &[Vec<f32>], v: &[f32]) -> Vec<f32> {
        mat.iter()
            .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum::<f32>())
            .collect()
    }

    fn gru_step(x: &[f32], h: &[f32], w_in: &[Vec<f32>], w_rec: &[Vec<f32>]) -> Vec<f32> {
        let dim = h.len();
        let ax = Self::mat_vec(w_in, x);
        let ah = Self::mat_vec(w_rec, h);
        (0..dim)
            .map(|i| (ax.get(i).copied().unwrap_or(0.0) + ah.get(i).copied().unwrap_or(0.0)).tanh())
            .collect()
    }

    /// Encode source tokens into hidden states.
    pub fn encode_source(&self, tokens: &[usize]) -> Vec<Vec<f32>> {
        let h = self.config.hidden_dim;
        let v = self.config.vocab_size;
        let mut hidden = vec![0.0f32; h];
        let mut states = Vec::with_capacity(tokens.len());
        for &tid in tokens {
            let idx = tid % v;
            let embed = &self.encoder_embed[idx];
            hidden = Self::gru_step(embed, &hidden, &self.enc_w_in, &self.enc_w_rec);
            states.push(hidden.clone());
        }
        states
    }

    /// One decode step: returns (vocab_distribution, copy_distribution).
    pub fn decode_step(
        &self,
        prev_token: usize,
        hidden: &[f32],
        encoder_states: &[Vec<f32>],
    ) -> (Vec<f32>, Vec<f32>) {
        let v = self.config.vocab_size;
        let a = self.config.attn_dim;
        let idx = prev_token % v;
        let embed = &self.decoder_embed[idx];
        let new_hidden = Self::gru_step(embed, hidden, &self.dec_w_in, &self.dec_w_rec);

        // Attention
        let dec_proj = Self::mat_vec(&self.w_dec, &new_hidden);
        let mut attn_scores: Vec<f32> = encoder_states
            .iter()
            .map(|es| {
                let enc_proj = Self::mat_vec(&self.w_enc, es);
                let combined: Vec<f32> = enc_proj
                    .iter()
                    .zip(dec_proj.iter())
                    .map(|(e, d)| (e + d).tanh())
                    .collect();
                self.v_attn
                    .iter()
                    .zip(combined.iter())
                    .map(|(v, c)| v * c)
                    .sum::<f32>()
            })
            .collect();
        // Softmax
        let max_s = attn_scores
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = attn_scores.iter().map(|&s| (s - max_s).exp()).collect();
        let sum: f32 = exp.iter().sum::<f32>().max(1e-8);
        attn_scores = exp.iter().map(|e| e / sum).collect();

        // Context vector
        let mut ctx = vec![0.0f32; self.config.hidden_dim];
        for (weight, es) in attn_scores.iter().zip(encoder_states.iter()) {
            for (c, e) in ctx.iter_mut().zip(es.iter()) {
                *c += weight * e;
            }
        }

        // Vocab distribution
        let combined: Vec<f32> = new_hidden
            .iter()
            .zip(ctx.iter())
            .map(|(h, c)| h + c)
            .collect();
        let logits = Self::mat_vec(&self.w_vocab, &combined);
        let vocab_dist = softmax_vec(&logits);

        // Copy distribution (just attention weights padded to vocab_size)
        let src_count = encoder_states.len();
        let mut copy_dist = vec![0.0f32; src_count];
        copy_dist.copy_from_slice(&attn_scores);

        (vocab_dist, copy_dist)
    }

    /// Scatter copy-attention weights into vocab-space.
    pub fn copy_mechanism(
        &self,
        attn_weights: &[f32],
        src_tokens: &[usize],
        vocab_size: usize,
    ) -> Vec<f32> {
        let mut copy_vocab = vec![0.0f32; vocab_size];
        for (&weight, &tid) in attn_weights.iter().zip(src_tokens.iter()) {
            let idx = tid % vocab_size;
            copy_vocab[idx] += weight;
        }
        copy_vocab
    }
}

pub(crate) fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f32 = exp.iter().sum::<f32>().max(1e-8);
    exp.iter().map(|e| e / sum).collect()
}
