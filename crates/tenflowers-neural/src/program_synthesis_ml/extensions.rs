//! Extensions: bug localisation, test case generation, and code metrics.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;

use super::{Token, TokenKind, sigmoid_f32};

// ---------------------------------------------------------------------------
// 8. BugLocalizerGnn
// ---------------------------------------------------------------------------

/// A node in a control-flow graph.
#[derive(Debug, Clone)]
pub struct CfgNode {
    /// Token IDs for the code at this node.
    pub tokens: Vec<usize>,
    /// Indices of successor nodes.
    pub successors: Vec<usize>,
}

/// A control-flow graph over basic blocks.
#[derive(Debug, Clone)]
pub struct ControlFlowGraph {
    pub nodes: Vec<CfgNode>,
}

/// Graph neural network for bug localisation: scores each CFG node.
pub struct BugLocalizerGnn {
    pub hidden_dim: usize,
    pub n_layers: usize,
    vocab_size: usize,
    node_embed: Vec<Vec<f32>>,
    /// Message-passing weight matrices [hidden x hidden] per layer
    msg_weights: Vec<Vec<Vec<f32>>>,
    /// Per-node update GRU weights
    update_w_in: Vec<Vec<f32>>,
    update_w_rec: Vec<Vec<f32>>,
    /// Scoring head [1 x hidden]
    score_head: Vec<f32>,
}

impl BugLocalizerGnn {
    pub fn new(vocab_size: usize, hidden_dim: usize, n_layers: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(999);
        let scale = (1.0 / hidden_dim as f32).sqrt();
        let mat = |rng: &mut StdRng| -> Vec<Vec<f32>> {
            (0..hidden_dim)
                .map(|_| {
                    (0..hidden_dim)
                        .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
                        .collect()
                })
                .collect()
        };
        let node_embed: Vec<Vec<f32>> = (0..vocab_size)
            .map(|_| {
                (0..hidden_dim)
                    .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
                    .collect()
            })
            .collect();
        let msg_weights = (0..n_layers).map(|_| mat(&mut rng)).collect();
        let update_w_in = mat(&mut rng);
        let update_w_rec = mat(&mut rng);
        let score_head: Vec<f32> = (0..hidden_dim)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        Self {
            hidden_dim,
            n_layers,
            vocab_size,
            node_embed,
            msg_weights,
            update_w_in,
            update_w_rec,
            score_head,
        }
    }

    fn mat_vec(mat: &[Vec<f32>], v: &[f32]) -> Vec<f32> {
        mat.iter()
            .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum::<f32>())
            .collect()
    }

    /// Compute per-node suspiciousness scores via message-passing GNN.
    pub fn forward(&self, cfg: &ControlFlowGraph) -> Vec<f32> {
        let n = cfg.nodes.len();
        if n == 0 {
            return Vec::new();
        }
        let h = self.hidden_dim;

        // Initialise node hidden states from token mean-pool embeddings
        let mut node_states: Vec<Vec<f32>> = cfg
            .nodes
            .iter()
            .map(|node| {
                if node.tokens.is_empty() {
                    return vec![0.0; h];
                }
                let mut sum = vec![0.0f32; h];
                for &tid in &node.tokens {
                    let idx = tid % self.vocab_size;
                    for (s, e) in sum.iter_mut().zip(self.node_embed[idx].iter()) {
                        *s += e;
                    }
                }
                let n_t = node.tokens.len() as f32;
                sum.iter_mut().for_each(|v| *v /= n_t);
                sum
            })
            .collect();

        // Message-passing layers
        for layer_idx in 0..self.n_layers {
            let w = &self.msg_weights[layer_idx];
            let new_states: Vec<Vec<f32>> = (0..n)
                .map(|i| {
                    // Aggregate messages from successors
                    let mut agg = vec![0.0f32; h];
                    let deg = cfg.nodes[i].successors.len();
                    for &j in &cfg.nodes[i].successors {
                        if j < n {
                            let msg = Self::mat_vec(w, &node_states[j]);
                            for (a, m) in agg.iter_mut().zip(msg.iter()) {
                                *a += m;
                            }
                        }
                    }
                    if deg > 0 {
                        agg.iter_mut().for_each(|v| *v /= deg as f32);
                    }
                    // GRU update
                    let xi: Vec<f32> = agg
                        .iter()
                        .zip(node_states[i].iter())
                        .map(|(a, b)| a + b)
                        .collect();
                    let gate_in = Self::mat_vec(&self.update_w_in, &xi);
                    let gate_rec = Self::mat_vec(&self.update_w_rec, &node_states[i]);
                    gate_in
                        .iter()
                        .zip(gate_rec.iter())
                        .map(|(a, b)| (a + b).tanh())
                        .collect()
                })
                .collect();
            node_states = new_states;
        }

        // Score each node
        node_states
            .iter()
            .map(|state| {
                let logit: f32 = self
                    .score_head
                    .iter()
                    .zip(state.iter())
                    .map(|(w, s)| w * s)
                    .sum();
                sigmoid_f32(logit)
            })
            .collect()
    }

    /// Return the top-k (node_index, score) pairs sorted by descending score.
    pub fn top_k_suspicious(&self, scores: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }
}

// ---------------------------------------------------------------------------
// 9. TestCaseGenerator
// ---------------------------------------------------------------------------

/// Mutation operator applied to a test case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationOperator {
    NegateCondition,
    OffByOne,
    NullInput,
    BoundaryValue,
}

/// A test case with floating-point inputs and expected outputs.
#[derive(Debug, Clone)]
pub struct TestCase {
    pub input: Vec<f32>,
    pub expected_output: Vec<f32>,
}

/// Mutation-based test case generator.
pub struct TestCaseGenerator;

impl TestCaseGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Generate boundary tests for `n_inputs` inputs each with bounds `(lo, hi)`.
    ///
    /// Produces 2^n test cases (one for each corner of the hyper-rectangle).
    /// Capped at 64 test cases when `n_inputs > 6`.
    pub fn generate_boundary_tests(n_inputs: usize, bounds: &[(f32, f32)]) -> Vec<TestCase> {
        if n_inputs == 0 || bounds.is_empty() {
            return Vec::new();
        }
        let effective = n_inputs.min(bounds.len());
        let n_corners = 1usize << effective.min(6);
        let mut tests = Vec::with_capacity(n_corners);
        for mask in 0..n_corners {
            let input: Vec<f32> = (0..effective)
                .map(|bit| {
                    let (lo, hi) = bounds[bit];
                    if (mask >> bit) & 1 == 1 {
                        hi
                    } else {
                        lo
                    }
                })
                .collect();
            // Padding if n_inputs > effective
            let mut full_input = input.clone();
            for idx in effective..n_inputs {
                let (lo, _) = bounds.get(idx).copied().unwrap_or((0.0, 1.0));
                full_input.push(lo);
            }
            // Expected output: simple identity (to be overridden by user)
            let expected_output = full_input.clone();
            tests.push(TestCase {
                input: full_input,
                expected_output,
            });
        }
        tests
    }

    /// Apply a mutation operator to produce a new test case.
    pub fn mutate_test(tc: &TestCase, op: MutationOperator, rng: &mut StdRng) -> TestCase {
        let mut new_input = tc.input.clone();
        let mut new_output = tc.expected_output.clone();

        match op {
            MutationOperator::NegateCondition => {
                // Negate (flip sign of) a randomly-chosen input element
                if !new_input.is_empty() {
                    let idx =
                        (rng.random::<f32>() * new_input.len() as f32) as usize % new_input.len();
                    new_input[idx] = -new_input[idx];
                }
                if !new_output.is_empty() {
                    let idx =
                        (rng.random::<f32>() * new_output.len() as f32) as usize % new_output.len();
                    new_output[idx] = -new_output[idx];
                }
            }
            MutationOperator::OffByOne => {
                // Increment or decrement a random input by 1.0
                if !new_input.is_empty() {
                    let idx =
                        (rng.random::<f32>() * new_input.len() as f32) as usize % new_input.len();
                    let delta = if rng.random::<f32>() > 0.5 {
                        1.0f32
                    } else {
                        -1.0f32
                    };
                    new_input[idx] += delta;
                }
            }
            MutationOperator::NullInput => {
                // Zero out a random input
                if !new_input.is_empty() {
                    let idx =
                        (rng.random::<f32>() * new_input.len() as f32) as usize % new_input.len();
                    new_input[idx] = 0.0;
                }
            }
            MutationOperator::BoundaryValue => {
                // Set a random input to f32::MAX or f32::MIN
                if !new_input.is_empty() {
                    let idx =
                        (rng.random::<f32>() * new_input.len() as f32) as usize % new_input.len();
                    new_input[idx] = if rng.random::<f32>() > 0.5 {
                        f32::MAX
                    } else {
                        f32::MIN
                    };
                }
            }
        }
        TestCase {
            input: new_input,
            expected_output: new_output,
        }
    }
}

impl Default for TestCaseGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// 10. CodeMetrics
// ---------------------------------------------------------------------------

/// Halstead complexity metrics.
#[derive(Debug, Clone)]
pub struct HalsteadMetrics {
    /// n1 + n2
    pub vocabulary: usize,
    /// N1 + N2
    pub length: usize,
    /// V = length * log2(vocabulary)
    pub volume: f64,
    /// D = (n1/2) * (N2/n2)
    pub difficulty: f64,
    /// E = D * V
    pub effort: f64,
}

/// Static code-quality metrics.
pub struct CodeMetrics;

impl CodeMetrics {
    /// McCabe's cyclomatic complexity: V(G) = E - N + 2P.
    ///
    /// Uses signed arithmetic internally to handle the case E < N correctly.
    pub fn cyclomatic_complexity(n_edges: usize, n_nodes: usize, n_components: usize) -> usize {
        let v = n_edges as i64 - n_nodes as i64 + 2 * n_components as i64;
        v.max(1) as usize
    }

    /// Compute Halstead metrics from operator/operand counts.
    ///
    /// - `n_operators`: total number of operators (N1)
    /// - `n_operands`: total number of operands (N2)
    /// - `unique_ops`: distinct operators (n1)
    /// - `unique_operands`: distinct operands (n2)
    pub fn halstead_metrics(
        n_operators: usize,
        n_operands: usize,
        unique_ops: usize,
        unique_operands: usize,
    ) -> HalsteadMetrics {
        let vocabulary = unique_ops + unique_operands;
        let length = n_operators + n_operands;
        let volume = if vocabulary > 1 {
            length as f64 * (vocabulary as f64).log2()
        } else {
            0.0
        };
        let difficulty = if unique_operands > 0 {
            (unique_ops as f64 / 2.0) * (n_operands as f64 / unique_operands as f64)
        } else {
            0.0
        };
        let effort = difficulty * volume;
        HalsteadMetrics {
            vocabulary,
            length,
            volume,
            difficulty,
            effort,
        }
    }

    /// SEI Maintainability Index:
    /// MI = 171 - 5.2 * ln(V) - 0.23 * CC - 16.2 * ln(SLOC)
    ///
    /// Clamped to [0, 100].
    pub fn maintainability_index(halstead_volume: f64, cyclomatic: f64, sloc: usize) -> f64 {
        let ln_v = if halstead_volume > 0.0 {
            halstead_volume.ln()
        } else {
            0.0
        };
        let ln_sloc = if sloc > 0 { (sloc as f64).ln() } else { 0.0 };
        let mi = 171.0 - 5.2 * ln_v - 0.23 * cyclomatic - 16.2 * ln_sloc;
        mi.clamp(0.0, 100.0)
    }

    /// Count physical lines of code (non-blank, non-comment lines).
    pub fn count_sloc(source: &str) -> usize {
        source
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                !trimmed.is_empty() && !trimmed.starts_with("//") && !trimmed.starts_with('#')
            })
            .count()
    }

    /// Estimate operator/operand counts from a token stream.
    pub fn count_halstead_tokens(tokens: &[Token]) -> (usize, usize, usize, usize) {
        let mut operators: HashMap<String, usize> = HashMap::new();
        let mut operands: HashMap<String, usize> = HashMap::new();
        for token in tokens {
            match token.kind {
                TokenKind::Operator | TokenKind::Keyword => {
                    *operators.entry(token.text.clone()).or_insert(0) += 1;
                }
                TokenKind::Identifier | TokenKind::Literal => {
                    *operands.entry(token.text.clone()).or_insert(0) += 1;
                }
                _ => {}
            }
        }
        let n_operators: usize = operators.values().sum();
        let n_operands: usize = operands.values().sum();
        let unique_ops = operators.len();
        let unique_operands = operands.len();
        (n_operators, n_operands, unique_ops, unique_operands)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
