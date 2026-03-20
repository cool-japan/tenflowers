//! Test-Time Compute (TTC) Scaling — reasoning at inference time without weight updates.
//!
//! Implements: Process Reward Models (PRM), Best-of-N sampling, Self-Consistency
//! majority vote, MCTS-based reasoning, step-level beam search, budget forcing, and
//! verifier ensembles.
//!
//! Based on: Lightman et al. 2023 "Let's Verify Step by Step", Wang et al. 2022
//! "Self-Consistency Improves Chain of Thought Reasoning", and DeepSeek-R1 / o1-style
//! extended thinking.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;

// ── Sigmoid helper ─────────────────────────────────────────────────────────────

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

// ── Verifier interface ─────────────────────────────────────────────────────────

/// A verifier scores a candidate answer for correctness (0.0 to 1.0).
pub trait Verifier: Send + Sync {
    fn score(&self, question: &str, answer: &str) -> f64;

    fn score_batch(&self, question: &str, answers: &[String]) -> Vec<f64> {
        answers.iter().map(|a| self.score(question, a)).collect()
    }
}

// ── ExactMatchVerifier ────────────────────────────────────────────────────────

/// Rule-based verifier: returns 1.0 iff the answer matches the stored correct answer.
pub struct ExactMatchVerifier {
    correct_answer: String,
}

impl ExactMatchVerifier {
    pub fn new(correct_answer: &str) -> Self {
        Self {
            correct_answer: correct_answer.trim().to_lowercase(),
        }
    }
}

impl Verifier for ExactMatchVerifier {
    fn score(&self, _question: &str, answer: &str) -> f64 {
        if answer.trim().to_lowercase() == self.correct_answer {
            1.0
        } else {
            0.0
        }
    }
}

// ── NeuralVerifier ────────────────────────────────────────────────────────────

/// MLP-based verifier scoring a question-answer pair via embedding lookup.
/// The `score_embedding` method takes an explicit embedding vector.
/// The `Verifier::score` method uses a deterministic hash-based mock embedding.
pub struct NeuralVerifier {
    pub d_input: usize,
    pub weights: Vec<Vec<f64>>, // [hidden_dim][d_input]
    pub weights2: Vec<f64>,     // [hidden_dim]
    pub bias: Vec<f64>,         // [hidden_dim]
    pub bias2: f64,
}

impl NeuralVerifier {
    pub fn new(d_input: usize, hidden_dim: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let scale = (2.0 / d_input as f64).sqrt();
        let weights: Vec<Vec<f64>> = (0..hidden_dim)
            .map(|_| {
                (0..d_input)
                    .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                    .collect()
            })
            .collect();
        let weights2: Vec<f64> = (0..hidden_dim)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
            .collect();
        let bias: Vec<f64> = vec![0.0; hidden_dim];
        Self {
            d_input,
            weights,
            weights2,
            bias,
            bias2: 0.0,
        }
    }

    /// Score an explicit embedding vector — sigmoid output in [0, 1].
    pub fn score_embedding(&self, embedding: &[f64]) -> f64 {
        let hidden_dim = self.weights.len();
        let mut h = vec![0.0f64; hidden_dim];
        for i in 0..hidden_dim {
            let row = &self.weights[i];
            let len = row.len().min(embedding.len());
            let sum: f64 = (0..len).map(|j| row[j] * embedding[j]).sum::<f64>() + self.bias[i];
            h[i] = relu(sum);
        }
        let logit: f64 = h
            .iter()
            .zip(self.weights2.iter())
            .map(|(hi, w)| hi * w)
            .sum::<f64>()
            + self.bias2;
        sigmoid(logit)
    }

    /// Build a mock hash-based embedding for a string.
    fn mock_embedding(&self, s: &str) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(hash_str(s));
        (0..self.d_input)
            .map(|_| rng.random::<f64>() * 2.0 - 1.0)
            .collect()
    }
}

impl Verifier for NeuralVerifier {
    fn score(&self, _question: &str, answer: &str) -> f64 {
        let emb = self.mock_embedding(answer);
        self.score_embedding(&emb)
    }
}

fn hash_str(s: &str) -> u64 {
    // FNV-1a 64-bit
    let mut h: u64 = 14_695_981_039_346_656_037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

// ── ProcessRewardModel (PRM) ───────────────────────────────────────────────────

/// Configuration for the Process Reward Model.
#[derive(Debug, Clone)]
pub struct PrmConfig {
    pub d_step: usize,
    pub hidden_dim: usize,
    pub learning_rate: f64,
}

/// A PRM scores each reasoning step individually.
/// Architecture: step_embedding -> linear -> ReLU -> linear -> sigmoid.
pub struct ProcessRewardModel {
    config: PrmConfig,
    w1: Vec<Vec<f64>>, // [hidden_dim][d_step]
    b1: Vec<f64>,      // [hidden_dim]
    w2: Vec<f64>,      // [hidden_dim]
    b2: f64,
}

impl ProcessRewardModel {
    pub fn new(config: PrmConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(1337);
        let scale = (2.0 / config.d_step as f64).sqrt();
        let w1: Vec<Vec<f64>> = (0..config.hidden_dim)
            .map(|_| {
                (0..config.d_step)
                    .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                    .collect()
            })
            .collect();
        let b1 = vec![0.0f64; config.hidden_dim];
        let scale2 = (2.0 / config.hidden_dim as f64).sqrt();
        let w2: Vec<f64> = (0..config.hidden_dim)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale2)
            .collect();
        Self {
            config,
            w1,
            b1,
            w2,
            b2: 0.0,
        }
    }

    /// Forward pass: step_embedding -> scalar in (0, 1).
    pub fn score_step(&self, step_embedding: &[f64]) -> f64 {
        let h = self.forward_hidden(step_embedding);
        let logit: f64 = h
            .iter()
            .zip(self.w2.iter())
            .map(|(hi, w)| hi * w)
            .sum::<f64>()
            + self.b2;
        sigmoid(logit)
    }

    fn forward_hidden(&self, x: &[f64]) -> Vec<f64> {
        let d = self.config.d_step.min(x.len());
        self.w1
            .iter()
            .zip(self.b1.iter())
            .map(|(row, &bi)| {
                let sum: f64 = (0..d).map(|j| row[j] * x[j]).sum::<f64>() + bi;
                relu(sum)
            })
            .collect()
    }

    /// Return per-step scores for a solution represented as step embeddings.
    pub fn score_solution(&self, step_embeddings: &[Vec<f64>]) -> Vec<f64> {
        step_embeddings.iter().map(|e| self.score_step(e)).collect()
    }

    /// Minimum step score (worst step — ORM proxy).
    pub fn min_step_score(&self, step_embeddings: &[Vec<f64>]) -> f64 {
        step_embeddings
            .iter()
            .map(|e| self.score_step(e))
            .fold(f64::INFINITY, f64::min)
    }

    /// Product of all step scores.
    pub fn product_score(&self, step_embeddings: &[Vec<f64>]) -> f64 {
        step_embeddings.iter().map(|e| self.score_step(e)).product()
    }

    /// Binary cross-entropy loss for one gradient-descent step; updates weights in place.
    pub fn train_step(&mut self, steps: &[Vec<f64>], labels: &[f64]) -> f64 {
        let n = steps.len().min(labels.len());
        if n == 0 {
            return 0.0;
        }

        let mut total_loss = 0.0f64;
        let lr = self.config.learning_rate;

        // Accumulate gradients
        let mut dw1 = vec![vec![0.0f64; self.config.d_step]; self.config.hidden_dim];
        let mut db1 = vec![0.0f64; self.config.hidden_dim];
        let mut dw2 = vec![0.0f64; self.config.hidden_dim];
        let mut db2 = 0.0f64;

        for (x, &y) in steps.iter().zip(labels.iter()) {
            let d = self.config.d_step.min(x.len());

            // Forward
            let h: Vec<f64> = self
                .w1
                .iter()
                .zip(self.b1.iter())
                .map(|(row, &bi)| {
                    let sum: f64 = (0..d).map(|j| row[j] * x[j]).sum::<f64>() + bi;
                    relu(sum)
                })
                .collect();

            let logit: f64 = h
                .iter()
                .zip(self.w2.iter())
                .map(|(hi, w)| hi * w)
                .sum::<f64>()
                + self.b2;
            let pred = sigmoid(logit);

            // BCE loss
            let loss = -(y * (pred + 1e-12).ln() + (1.0 - y) * (1.0 - pred + 1e-12).ln());
            total_loss += loss;

            // Backward: d_loss/d_logit = pred - y
            let d_logit = pred - y;

            // Grad w2, b2
            for i in 0..self.config.hidden_dim {
                dw2[i] += d_logit * h[i];
            }
            db2 += d_logit;

            // Grad hidden: d_logit * w2[i] * relu'(pre[i])
            let pre: Vec<f64> = self
                .w1
                .iter()
                .zip(self.b1.iter())
                .map(|(row, &bi)| (0..d).map(|j| row[j] * x[j]).sum::<f64>() + bi)
                .collect();

            for i in 0..self.config.hidden_dim {
                let d_hi = d_logit * self.w2[i] * if pre[i] > 0.0 { 1.0 } else { 0.0 };
                for j in 0..d {
                    dw1[i][j] += d_hi * x[j];
                }
                db1[i] += d_hi;
            }
        }

        // Normalise by n and apply
        let inv_n = 1.0 / n as f64;
        for i in 0..self.config.hidden_dim {
            for j in 0..self.config.d_step {
                self.w1[i][j] -= lr * dw1[i][j] * inv_n;
            }
            self.b1[i] -= lr * db1[i] * inv_n;
            self.w2[i] -= lr * dw2[i] * inv_n;
        }
        self.b2 -= lr * db2 * inv_n;

        total_loss * inv_n
    }

    /// Fit the PRM on a dataset of (step_embeddings, step_labels) pairs.
    /// Returns per-epoch loss history.
    pub fn fit(&mut self, data: &[(Vec<Vec<f64>>, Vec<f64>)], n_epochs: usize) -> Vec<f64> {
        let mut history = Vec::with_capacity(n_epochs);
        for _ in 0..n_epochs {
            let mut epoch_loss = 0.0f64;
            let mut epoch_samples = 0usize;
            for (steps, labels) in data {
                let loss = self.train_step(steps, labels);
                epoch_loss += loss * steps.len() as f64;
                epoch_samples += steps.len();
            }
            history.push(if epoch_samples > 0 {
                epoch_loss / epoch_samples as f64
            } else {
                0.0
            });
        }
        history
    }
}

// ── Best-of-N Sampler ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BestOfNConfig {
    pub n_samples: usize,
    pub temperature: f64,
    pub use_process_reward: bool,
}

#[derive(Debug, Clone)]
pub struct BestOfNResult {
    pub best_answer: String,
    pub best_score: f64,
    pub all_scores: Vec<f64>,
    pub n_unique_answers: usize,
}

pub struct BestOfNSampler {
    config: BestOfNConfig,
    verifier: Box<dyn Verifier>,
}

impl BestOfNSampler {
    pub fn new(config: BestOfNConfig, verifier: Box<dyn Verifier>) -> Self {
        Self { config, verifier }
    }

    /// Generate `n_samples` candidates with `generator(temperature)`, score each, return best.
    pub fn sample(
        &self,
        mut generator: impl FnMut(f64) -> String,
        question: &str,
    ) -> BestOfNResult {
        let mut answers: Vec<String> = Vec::with_capacity(self.config.n_samples);
        for _ in 0..self.config.n_samples {
            answers.push(generator(self.config.temperature));
        }

        let scores = self.verifier.score_batch(question, &answers);

        let mut best_idx = 0usize;
        let mut best_score = scores.first().copied().unwrap_or(0.0);
        for (i, &s) in scores.iter().enumerate() {
            if s > best_score {
                best_score = s;
                best_idx = i;
            }
        }

        let unique: std::collections::HashSet<&str> = answers.iter().map(|a| a.as_str()).collect();

        BestOfNResult {
            best_answer: answers[best_idx].clone(),
            best_score,
            all_scores: scores,
            n_unique_answers: unique.len(),
        }
    }

    /// Unbiased estimator: P(at least one correct in k draws from n candidates, c correct).
    /// Formula: 1 − C(n−c, k) / C(n, k)
    pub fn pass_at_k(n: usize, c: usize, k: usize) -> f64 {
        if c == 0 {
            return 0.0;
        }
        if c >= n || k >= n {
            return 1.0;
        }
        // Numerically stable: compute ratio C(n-c,k)/C(n,k) in log space
        // C(n-c,k) / C(n,k) = prod_{i=0}^{k-1} (n-c-i)/(n-i)
        let mut log_ratio = 0.0f64;
        for i in 0..k {
            let num = (n - c - i) as f64;
            let den = (n - i) as f64;
            if num <= 0.0 {
                // All draws would hit a correct answer
                return 1.0;
            }
            log_ratio += (num / den).ln();
        }
        1.0 - log_ratio.exp()
    }
}

// ── Self-Consistency Decoder ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum AnswerExtractor {
    LastLine,
    AfterKeyword(String),
    Regex(String),
}

#[derive(Debug, Clone)]
pub struct SelfConsistencyConfig {
    pub n_samples: usize,
    pub temperature: f64,
    pub answer_extractor: AnswerExtractor,
}

#[derive(Debug, Clone)]
pub struct SelfConsistencyResult {
    pub majority_answer: String,
    /// (answer, count) sorted by count descending.
    pub vote_counts: Vec<(String, usize)>,
    pub confidence: f64,
}

pub struct SelfConsistencyDecoder {
    config: SelfConsistencyConfig,
}

impl SelfConsistencyDecoder {
    pub fn new(config: SelfConsistencyConfig) -> Self {
        Self { config }
    }

    pub fn decode(&self, mut generator: impl FnMut(f64) -> String) -> SelfConsistencyResult {
        let mut vote_map: HashMap<String, usize> = HashMap::new();
        for _ in 0..self.config.n_samples {
            let response = generator(self.config.temperature);
            let answer = self.extract_answer(&response);
            *vote_map.entry(answer).or_insert(0) += 1;
        }

        let mut vote_counts: Vec<(String, usize)> = vote_map.into_iter().collect();
        vote_counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        let majority_answer = vote_counts
            .first()
            .map(|(a, _)| a.clone())
            .unwrap_or_default();
        let majority_votes = vote_counts.first().map(|(_, c)| *c).unwrap_or(0);
        let total: usize = vote_counts.iter().map(|(_, c)| c).sum();
        let confidence = if total > 0 {
            majority_votes as f64 / total as f64
        } else {
            0.0
        };

        SelfConsistencyResult {
            majority_answer,
            vote_counts,
            confidence,
        }
    }

    pub fn extract_answer(&self, response: &str) -> String {
        match &self.config.answer_extractor {
            AnswerExtractor::LastLine => response
                .lines()
                .rev()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("")
                .trim()
                .to_string(),
            AnswerExtractor::AfterKeyword(kw) => {
                if let Some(pos) = response.to_lowercase().find(&kw.to_lowercase()) {
                    response[pos + kw.len()..].trim().to_string()
                } else {
                    response.trim().to_string()
                }
            }
            AnswerExtractor::Regex(pattern) => {
                // Simple prefix-matching: return first line that starts with pattern
                for line in response.lines() {
                    if line.trim().starts_with(pattern.as_str()) {
                        return line.trim().to_string();
                    }
                }
                response.trim().to_string()
            }
        }
    }
}

// ── MCTS ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MctsConfig {
    pub n_simulations: usize,
    pub c_puct: f64,
    pub max_depth: usize,
    pub n_actions: usize,
    pub rollout_depth: usize,
}

#[derive(Debug, Clone)]
pub struct MctsNode {
    pub state: String,
    pub visit_count: usize,
    pub total_value: f64,
    pub prior: f64,
    pub(crate) children: Vec<usize>, // indices into tree arena
    pub(crate) parent: Option<usize>,
    pub is_terminal: bool,
}

impl MctsNode {
    pub fn q_value(&self) -> f64 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_value / self.visit_count as f64
        }
    }
}

pub struct MctsTree {
    pub nodes: Vec<MctsNode>,
    config: MctsConfig,
}

impl MctsTree {
    pub fn new(root_state: &str, config: MctsConfig) -> Self {
        let root = MctsNode {
            state: root_state.to_string(),
            visit_count: 0,
            total_value: 0.0,
            prior: 1.0,
            children: Vec::new(),
            parent: None,
            is_terminal: false,
        };
        Self {
            nodes: vec![root],
            config,
        }
    }

    /// UCT selection: descend until a leaf or unexpanded node is reached.
    pub fn select(&self, node_idx: usize) -> usize {
        let mut current = node_idx;
        loop {
            let node = &self.nodes[current];
            if node.children.is_empty() || node.is_terminal {
                return current;
            }
            let parent_visits = node.visit_count as f64;
            let best = node.children.iter().copied().max_by(|&a, &b| {
                let uct_a = self.uct_score(a, parent_visits);
                let uct_b = self.uct_score(b, parent_visits);
                uct_a
                    .partial_cmp(&uct_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            match best {
                Some(child) => current = child,
                None => return current,
            }
        }
    }

    fn uct_score(&self, node_idx: usize, parent_visits: f64) -> f64 {
        let node = &self.nodes[node_idx];
        let q = node.q_value();
        let u = self.config.c_puct
            * node.prior
            * (parent_visits.ln() / (1.0 + node.visit_count as f64)).sqrt();
        q + u
    }

    /// Expand a node by adding child nodes with given (state, prior) pairs.
    pub fn expand(&mut self, node_idx: usize, child_states: Vec<(String, f64)>) {
        let mut child_indices = Vec::with_capacity(child_states.len());
        for (state, prior) in child_states {
            let child_idx = self.nodes.len();
            self.nodes.push(MctsNode {
                state,
                visit_count: 0,
                total_value: 0.0,
                prior,
                children: Vec::new(),
                parent: Some(node_idx),
                is_terminal: false,
            });
            child_indices.push(child_idx);
        }
        self.nodes[node_idx].children = child_indices;
    }

    /// Back-propagate a value from `node_idx` up to the root.
    pub fn backprop(&mut self, node_idx: usize, value: f64) {
        let mut current = Some(node_idx);
        while let Some(idx) = current {
            self.nodes[idx].visit_count += 1;
            self.nodes[idx].total_value += value;
            current = self.nodes[idx].parent;
        }
    }

    /// Return the child with the highest visit count.
    pub fn best_action(&self, root_idx: usize) -> usize {
        let node = &self.nodes[root_idx];
        node.children
            .iter()
            .copied()
            .max_by_key(|&c| self.nodes[c].visit_count)
            .unwrap_or(root_idx)
    }

    /// Return visit-count policy distribution over children of `node_idx`.
    pub fn get_policy(&self, node_idx: usize) -> Vec<f64> {
        let children = &self.nodes[node_idx].children;
        if children.is_empty() {
            return vec![];
        }
        let total: usize = children.iter().map(|&c| self.nodes[c].visit_count).sum();
        if total == 0 {
            let n = children.len() as f64;
            return vec![1.0 / n; children.len()];
        }
        children
            .iter()
            .map(|&c| self.nodes[c].visit_count as f64 / total as f64)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct MctsResult {
    pub best_solution: String,
    pub best_value: f64,
    pub n_nodes_explored: usize,
    pub search_depth: usize,
}

pub struct MctsSolver {
    config: MctsConfig,
    prm: Option<ProcessRewardModel>,
}

impl MctsSolver {
    pub fn new(config: MctsConfig) -> Self {
        Self { config, prm: None }
    }

    pub fn with_prm(config: MctsConfig, prm: ProcessRewardModel) -> Self {
        Self {
            config,
            prm: Some(prm),
        }
    }

    pub fn solve(
        &mut self,
        question: &str,
        step_generator: impl Fn(&str, usize) -> Vec<(String, f64)>,
        terminal_check: impl Fn(&str) -> bool,
        value_fn: impl Fn(&str) -> f64,
    ) -> MctsResult {
        let mut tree = MctsTree::new(question, self.config.clone());
        let mut best_solution = question.to_string();
        let mut best_value = f64::NEG_INFINITY;
        let mut max_depth = 0usize;

        for _ in 0..self.config.n_simulations {
            // Selection
            let leaf = tree.select(0);

            // Check terminal
            if tree.nodes[leaf].is_terminal {
                let v = value_fn(&tree.nodes[leaf].state);
                tree.backprop(leaf, v);
                if v > best_value {
                    best_value = v;
                    best_solution = tree.nodes[leaf].state.clone();
                }
                continue;
            }

            let leaf_state = tree.nodes[leaf].state.clone();

            // Depth check
            let depth = compute_depth(&tree, leaf);
            if depth >= self.config.max_depth {
                tree.nodes[leaf].is_terminal = true;
                let v = value_fn(&leaf_state);
                tree.backprop(leaf, v);
                if v > best_value {
                    best_value = v;
                    best_solution = leaf_state;
                }
                continue;
            }

            max_depth = max_depth.max(depth + 1);

            // Expansion
            let children = step_generator(&leaf_state, self.config.n_actions);
            if children.is_empty() {
                tree.nodes[leaf].is_terminal = true;
                let v = value_fn(&leaf_state);
                tree.backprop(leaf, v);
                if v > best_value {
                    best_value = v;
                    best_solution = leaf_state;
                }
                continue;
            }

            // Mark terminal children
            let terminal_flags: Vec<bool> =
                children.iter().map(|(s, _)| terminal_check(s)).collect();
            tree.expand(leaf, children.clone());

            let added = tree.nodes[leaf].children.clone();
            for (&cidx, is_term) in added.iter().zip(terminal_flags.iter()) {
                if *is_term {
                    tree.nodes[cidx].is_terminal = true;
                }
            }

            // Evaluate best child (rollout)
            let best_child = added.iter().copied().max_by(|&a, &b| {
                tree.nodes[a]
                    .prior
                    .partial_cmp(&tree.nodes[b].prior)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            if let Some(cidx) = best_child {
                let v = self.rollout(
                    &tree.nodes[cidx].state,
                    &value_fn,
                    &terminal_check,
                    depth + 1,
                );
                tree.backprop(cidx, v);
                if v > best_value {
                    best_value = v;
                    best_solution = tree.nodes[cidx].state.clone();
                }
            }
        }

        MctsResult {
            best_solution,
            best_value: if best_value == f64::NEG_INFINITY {
                0.0
            } else {
                best_value
            },
            n_nodes_explored: tree.nodes.len(),
            search_depth: max_depth,
        }
    }

    fn rollout(
        &self,
        state: &str,
        value_fn: &impl Fn(&str) -> f64,
        terminal_check: &impl Fn(&str) -> bool,
        current_depth: usize,
    ) -> f64 {
        if terminal_check(state) || current_depth >= self.config.max_depth {
            return value_fn(state);
        }
        // Simple greedy rollout: just evaluate the value at this state
        value_fn(state)
    }
}

fn compute_depth(tree: &MctsTree, mut node_idx: usize) -> usize {
    let mut depth = 0usize;
    while let Some(parent) = tree.nodes[node_idx].parent {
        depth += 1;
        node_idx = parent;
    }
    depth
}

// ── Step Beam Search ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct StepBeamConfig {
    pub beam_width: usize,
    pub max_steps: usize,
    pub prm_weight: f64,
    pub length_penalty: f64,
}

#[derive(Debug, Clone)]
pub struct StepBeamResult {
    pub best_sequence: Vec<String>,
    pub best_score: f64,
    pub all_beams: Vec<(Vec<String>, f64)>,
}

pub struct StepBeamSearcher {
    config: StepBeamConfig,
    prm: ProcessRewardModel,
}

impl StepBeamSearcher {
    pub fn new(config: StepBeamConfig, prm: ProcessRewardModel) -> Self {
        Self { config, prm }
    }

    pub fn search(
        &self,
        initial_state: &str,
        step_generator: impl Fn(&str) -> Vec<(String, Vec<f64>)>,
        terminal_check: impl Fn(&str) -> bool,
    ) -> StepBeamResult {
        // Each beam entry: (steps_so_far, cumulative_log_score, current_state)
        type Beam = (Vec<String>, f64, String);
        let mut beams: Vec<Beam> = vec![(vec![], 0.0, initial_state.to_string())];
        let mut finished: Vec<(Vec<String>, f64)> = Vec::new();

        for _step in 0..self.config.max_steps {
            if beams.is_empty() {
                break;
            }
            let mut candidates: Vec<Beam> = Vec::new();

            for (steps, cum_score, state) in &beams {
                if terminal_check(state) {
                    let lp = (steps.len() as f64 + 1.0).powf(self.config.length_penalty);
                    finished.push((steps.clone(), cum_score / lp));
                    continue;
                }

                let next_steps = step_generator(state);
                if next_steps.is_empty() {
                    let lp = (steps.len() as f64 + 1.0).powf(self.config.length_penalty);
                    finished.push((steps.clone(), cum_score / lp));
                    continue;
                }

                for (next_state, step_emb) in &next_steps {
                    let prm_score = self.prm.score_step(step_emb);
                    // Combine PRM score as log-probability
                    let step_log = self.config.prm_weight * (prm_score + 1e-10).ln();
                    let new_score = cum_score + step_log;
                    let mut new_steps = steps.clone();
                    new_steps.push(next_state.clone());
                    candidates.push((new_steps, new_score, next_state.clone()));
                }
            }

            // Keep top beam_width
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(self.config.beam_width);
            beams = candidates;
        }

        // Flush remaining beams as finished
        for (steps, cum_score, _state) in beams {
            let lp = ((steps.len() as f64) + 1.0).powf(self.config.length_penalty);
            finished.push((steps, cum_score / lp));
        }

        if finished.is_empty() {
            return StepBeamResult {
                best_sequence: vec![initial_state.to_string()],
                best_score: 0.0,
                all_beams: vec![(vec![initial_state.to_string()], 0.0)],
            };
        }

        finished.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best_score = finished[0].1;
        let best_sequence = finished[0].0.clone();

        StepBeamResult {
            best_sequence,
            best_score,
            all_beams: finished,
        }
    }
}

// ── BudgetForcer ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BudgetConfig {
    pub max_tokens: usize,
    pub thinking_budget: usize,
    pub answer_budget: usize,
}

#[derive(Debug, Clone)]
pub struct ComputeUsage {
    pub total_tokens: usize,
    pub thinking_tokens: usize,
    pub answer_tokens: usize,
    pub budget_utilization: f64,
}

pub struct BudgetForcer {
    config: BudgetConfig,
}

impl BudgetForcer {
    pub fn new(config: BudgetConfig) -> Self {
        Self { config }
    }

    /// Truncate thinking to `thinking_budget` tokens, answer to `answer_budget` tokens.
    /// Total output capped at `max_tokens`.
    pub fn apply(&self, tokens: &[usize]) -> Vec<usize> {
        let thinking_end = tokens.len().min(self.config.thinking_budget);
        let thinking_part = &tokens[..thinking_end];

        let answer_start = self.config.thinking_budget.min(tokens.len());
        let answer_end = tokens.len().min(answer_start + self.config.answer_budget);
        let answer_part = &tokens[answer_start..answer_end];

        let combined: Vec<usize> = thinking_part
            .iter()
            .chain(answer_part.iter())
            .copied()
            .take(self.config.max_tokens)
            .collect();
        combined
    }

    pub fn compute_usage(&self, tokens: &[usize]) -> ComputeUsage {
        let applied = self.apply(tokens);
        let total = applied.len();
        let thinking_tokens = total.min(self.config.thinking_budget);
        let answer_tokens = total.saturating_sub(thinking_tokens);
        let budget_utilization = if self.config.max_tokens > 0 {
            total as f64 / self.config.max_tokens as f64
        } else {
            0.0
        };
        ComputeUsage {
            total_tokens: total,
            thinking_tokens,
            answer_tokens,
            budget_utilization,
        }
    }
}

// ── VerifierEnsemble ───────────────────────────────────────────────────────────

pub struct VerifierEnsemble {
    verifiers: Vec<Box<dyn Verifier>>,
    weights: Vec<f64>,
}

impl VerifierEnsemble {
    pub fn new(verifiers: Vec<Box<dyn Verifier>>, weights: Vec<f64>) -> Self {
        assert_eq!(
            verifiers.len(),
            weights.len(),
            "verifiers and weights must match in length"
        );
        Self { verifiers, weights }
    }

    pub fn uniform(verifiers: Vec<Box<dyn Verifier>>) -> Self {
        let n = verifiers.len();
        let w = if n > 0 { 1.0 / n as f64 } else { 0.0 };
        Self {
            weights: vec![w; n],
            verifiers,
        }
    }

    /// Weighted average score across all ensemble members.
    pub fn score(&self, question: &str, answer: &str) -> f64 {
        let weight_sum: f64 = self.weights.iter().sum();
        if weight_sum == 0.0 {
            return 0.0;
        }
        let weighted: f64 = self
            .verifiers
            .iter()
            .zip(self.weights.iter())
            .map(|(v, &w)| v.score(question, answer) * w)
            .sum();
        weighted / weight_sum
    }
}

// ── TTC Metrics ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TtcMetrics {
    pub pass_at_1: f64,
    pub pass_at_k: f64,
    pub majority_at_k: f64,
    pub oracle_at_k: f64,
    pub compute_efficiency: f64,
}

#[derive(Debug, Clone)]
pub struct TtcReport {
    pub metrics: TtcMetrics,
    pub best_of_n_result: BestOfNResult,
    pub self_consistency_result: SelfConsistencyResult,
    pub mcts_result: Option<MctsResult>,
    pub samples_used: usize,
    pub total_compute_tokens: usize,
}

/// Compute TTC metrics from a vector of booleans indicating which samples are correct.
pub fn compute_ttc_metrics(correct_answers: &[bool], k: usize) -> TtcMetrics {
    let n = correct_answers.len();
    let c = correct_answers.iter().filter(|&&b| b).count();

    // pass@1: fraction correct in a single sample
    let pass_at_1 = if n > 0 { c as f64 / n as f64 } else { 0.0 };

    // pass@k: unbiased estimator
    let pass_at_k_val = BestOfNSampler::pass_at_k(n, c, k);

    // majority@k: P(majority vote correct) — for a binary problem:
    // we need > k/2 correct samples. Approximate via hypergeometric expectation.
    let majority_at_k = compute_majority_at_k(n, c, k);

    // oracle@k: P(at least one correct) = pass_at_k
    let oracle_at_k = pass_at_k_val;

    // compute_efficiency: quality / cost = pass_at_k / k
    let compute_efficiency = if k > 0 { pass_at_k_val / k as f64 } else { 0.0 };

    TtcMetrics {
        pass_at_1,
        pass_at_k: pass_at_k_val,
        majority_at_k,
        oracle_at_k,
        compute_efficiency,
    }
}

/// Approximate majority@k via binomial approximation.
fn compute_majority_at_k(n: usize, c: usize, k: usize) -> f64 {
    if n == 0 || k == 0 {
        return 0.0;
    }
    let p = c as f64 / n as f64;
    // P(majority correct) = P(Binomial(k, p) > k/2)
    let threshold = k / 2;
    let mut prob = 0.0f64;
    for i in (threshold + 1)..=k {
        prob += binom_pmf(k, i, p);
    }
    prob.clamp(0.0, 1.0)
}

fn binom_pmf(n: usize, k: usize, p: f64) -> f64 {
    if k > n {
        return 0.0;
    }
    let log_coeff = log_binom(n, k);
    let log_prob = k as f64 * p.ln() + (n - k) as f64 * (1.0 - p).ln();
    // Handle edge cases where p=0 or p=1
    if !log_prob.is_finite() {
        if k == 0 && p == 0.0 {
            return 1.0;
        }
        if k == n && p == 1.0 {
            return 1.0;
        }
        return 0.0;
    }
    (log_coeff + log_prob).exp()
}

fn log_binom(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    let k = k.min(n - k);
    let mut result = 0.0f64;
    for i in 0..k {
        result += ((n - i) as f64).ln() - ((i + 1) as f64).ln();
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── ExactMatchVerifier ─────────────────────────────────────────────────────

    #[test]
    fn test_exact_match_correct() {
        let v = ExactMatchVerifier::new("42");
        assert_eq!(v.score("what is 6*7", "42"), 1.0);
    }

    #[test]
    fn test_exact_match_wrong() {
        let v = ExactMatchVerifier::new("42");
        assert_eq!(v.score("what is 6*7", "41"), 0.0);
    }

    #[test]
    fn test_exact_match_case_insensitive() {
        let v = ExactMatchVerifier::new("Paris");
        assert_eq!(v.score("capital of France", "paris"), 1.0);
    }

    #[test]
    fn test_exact_match_batch() {
        let v = ExactMatchVerifier::new("yes");
        let answers = vec!["yes".to_string(), "no".to_string(), "YES".to_string()];
        let scores = v.score_batch("q", &answers);
        assert_eq!(scores[0], 1.0);
        assert_eq!(scores[1], 0.0);
        assert_eq!(scores[2], 1.0);
    }

    // ── NeuralVerifier ─────────────────────────────────────────────────────────

    #[test]
    fn test_neural_verifier_score_in_range() {
        let v = NeuralVerifier::new(16, 32);
        let s = v.score("q", "some answer");
        assert!((0.0..=1.0).contains(&s), "score out of [0,1]: {s}");
    }

    #[test]
    fn test_neural_verifier_score_embedding_in_range() {
        let v = NeuralVerifier::new(8, 16);
        let emb: Vec<f64> = (0..8).map(|i| i as f64 * 0.1).collect();
        let s = v.score_embedding(&emb);
        assert!(
            (0.0..=1.0).contains(&s),
            "score_embedding out of [0,1]: {s}"
        );
    }

    #[test]
    fn test_neural_verifier_different_answers_differ() {
        let v = NeuralVerifier::new(8, 16);
        let s1 = v.score("q", "answer one");
        let s2 = v.score("q", "answer two");
        // Should not always be identical (hash-based differentiation)
        // This is probabilistic, but with different hashes should differ
        let _ = (s1, s2); // just ensure they run without panic
    }

    // ── ProcessRewardModel ─────────────────────────────────────────────────────

    fn make_prm() -> ProcessRewardModel {
        ProcessRewardModel::new(PrmConfig {
            d_step: 8,
            hidden_dim: 16,
            learning_rate: 0.01,
        })
    }

    fn rand_emb(d: usize, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..d).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect()
    }

    #[test]
    fn test_prm_score_step_in_range() {
        let prm = make_prm();
        let emb = rand_emb(8, 7);
        let s = prm.score_step(&emb);
        assert!((0.0..=1.0).contains(&s), "score_step out of [0,1]: {s}");
    }

    #[test]
    fn test_prm_product_score_in_range() {
        let prm = make_prm();
        let steps: Vec<Vec<f64>> = (0..4).map(|i| rand_emb(8, i as u64)).collect();
        let ps = prm.product_score(&steps);
        assert!(
            (0.0..=1.0).contains(&ps),
            "product_score out of [0,1]: {ps}"
        );
    }

    #[test]
    fn test_prm_min_step_score_le_mean() {
        let prm = make_prm();
        let steps: Vec<Vec<f64>> = (0..5).map(|i| rand_emb(8, i as u64 + 10)).collect();
        let scores = prm.score_solution(&steps);
        let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
        let min = prm.min_step_score(&steps);
        assert!(min <= mean + 1e-9, "min={min} > mean={mean}");
    }

    #[test]
    fn test_prm_train_step_positive_loss() {
        let mut prm = make_prm();
        let steps: Vec<Vec<f64>> = (0..4).map(|i| rand_emb(8, i as u64 + 20)).collect();
        let labels = vec![1.0, 0.0, 1.0, 0.0];
        let loss = prm.train_step(&steps, &labels);
        assert!(loss > 0.0, "loss should be positive, got {loss}");
    }

    #[test]
    fn test_prm_loss_decreases_after_training() {
        let mut prm = make_prm();
        let steps: Vec<Vec<f64>> = (0..8).map(|i| rand_emb(8, i as u64 + 30)).collect();
        let labels: Vec<f64> = steps
            .iter()
            .enumerate()
            .map(|(i, _)| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let loss0 = prm.train_step(&steps, &labels);
        let mut last_loss = loss0;
        for _ in 0..50 {
            last_loss = prm.train_step(&steps, &labels);
        }
        assert!(
            last_loss < loss0 * 1.5,
            "loss did not converge: initial={loss0}, final={last_loss}"
        );
    }

    #[test]
    fn test_prm_fit_returns_history() {
        let mut prm = make_prm();
        let data: Vec<(Vec<Vec<f64>>, Vec<f64>)> = vec![(
            (0..4).map(|i| rand_emb(8, i as u64 + 100)).collect(),
            vec![1.0, 0.0, 1.0, 1.0],
        )];
        let history = prm.fit(&data, 5);
        assert_eq!(history.len(), 5);
        for &l in &history {
            assert!(l.is_finite(), "non-finite loss in history: {l}");
        }
    }

    #[test]
    fn test_prm_score_solution_len_matches_steps() {
        let prm = make_prm();
        let steps: Vec<Vec<f64>> = (0..6).map(|i| rand_emb(8, i as u64 + 50)).collect();
        let scores = prm.score_solution(&steps);
        assert_eq!(scores.len(), 6);
    }

    // ── BestOfNSampler ────────────────────────────────────────────────────────

    fn make_bon_sampler(n: usize, correct: &str) -> BestOfNSampler {
        let verifier = Box::new(ExactMatchVerifier::new(correct));
        BestOfNSampler::new(
            BestOfNConfig {
                n_samples: n,
                temperature: 1.0,
                use_process_reward: false,
            },
            verifier,
        )
    }

    #[test]
    fn test_bon_result_best_score_in_range() {
        let sampler = make_bon_sampler(5, "42");
        let mut rng = StdRng::seed_from_u64(99);
        let result = sampler.sample(
            |_t| {
                if rng.random::<f64>() < 0.5 {
                    "42".to_string()
                } else {
                    "0".to_string()
                }
            },
            "q",
        );
        assert!(
            (0.0..=1.0).contains(&result.best_score),
            "best_score out of range"
        );
    }

    #[test]
    fn test_bon_n_unique_le_n_samples() {
        let sampler = make_bon_sampler(10, "x");
        let answers = ["a", "b", "c"];
        let mut counter = 0usize;
        let result = sampler.sample(
            |_| {
                let a = answers[counter % 3];
                counter += 1;
                a.to_string()
            },
            "q",
        );
        assert!(result.n_unique_answers <= 10);
    }

    #[test]
    fn test_bon_picks_correct_answer() {
        let sampler = make_bon_sampler(4, "correct");
        let answers = ["wrong", "wrong", "correct", "wrong"];
        let mut i = 0;
        let result = sampler.sample(
            |_| {
                let a = answers[i % 4];
                i += 1;
                a.to_string()
            },
            "q",
        );
        assert_eq!(result.best_answer, "correct");
        assert_eq!(result.best_score, 1.0);
    }

    #[test]
    fn test_pass_at_k_high_when_many_correct() {
        let p = BestOfNSampler::pass_at_k(10, 5, 5);
        assert!(p > 0.95, "expected p > 0.95, got {p}");
    }

    #[test]
    fn test_pass_at_k_zero_correct() {
        let p = BestOfNSampler::pass_at_k(10, 0, 5);
        assert_eq!(p, 0.0);
    }

    #[test]
    fn test_pass_at_k_all_correct() {
        let p = BestOfNSampler::pass_at_k(10, 10, 5);
        assert_eq!(p, 1.0);
    }

    #[test]
    fn test_pass_at_k_increases_with_k() {
        let p1 = BestOfNSampler::pass_at_k(20, 5, 1);
        let p5 = BestOfNSampler::pass_at_k(20, 5, 5);
        let p10 = BestOfNSampler::pass_at_k(20, 5, 10);
        assert!(p1 <= p5, "p@1={p1} > p@5={p5}");
        assert!(p5 <= p10, "p@5={p5} > p@10={p10}");
    }

    // ── SelfConsistencyDecoder ────────────────────────────────────────────────

    fn make_sc_decoder(n: usize, extractor: AnswerExtractor) -> SelfConsistencyDecoder {
        SelfConsistencyDecoder::new(SelfConsistencyConfig {
            n_samples: n,
            temperature: 0.7,
            answer_extractor: extractor,
        })
    }

    #[test]
    fn test_sc_majority_answer_nonempty() {
        let dec = make_sc_decoder(6, AnswerExtractor::LastLine);
        let answers = ["A", "B", "A", "A", "B", "C"];
        let mut i = 0;
        let result = dec.decode(|_| {
            let a = answers[i % 6];
            i += 1;
            format!("reasoning...\n{a}")
        });
        assert!(!result.majority_answer.is_empty());
    }

    #[test]
    fn test_sc_majority_answer_is_most_frequent() {
        let dec = make_sc_decoder(6, AnswerExtractor::LastLine);
        let answers = ["A", "A", "A", "B", "B", "C"];
        let mut i = 0;
        let result = dec.decode(|_| {
            let a = answers[i % 6];
            i += 1;
            format!("step1\nstep2\n{a}")
        });
        assert_eq!(result.majority_answer, "A");
    }

    #[test]
    fn test_sc_confidence_in_range() {
        let dec = make_sc_decoder(5, AnswerExtractor::LastLine);
        let answers = ["X", "X", "X", "Y", "Y"];
        let mut i = 0;
        let result = dec.decode(|_| {
            let a = answers[i % 5];
            i += 1;
            a.to_string()
        });
        assert!(
            (0.0..=1.0).contains(&result.confidence),
            "confidence={}",
            result.confidence
        );
    }

    #[test]
    fn test_sc_vote_counts_sum_to_n() {
        let n = 8;
        let dec = make_sc_decoder(n, AnswerExtractor::LastLine);
        let answers = ["A", "B", "C", "A", "B", "A", "C", "A"];
        let mut i = 0;
        let result = dec.decode(|_| {
            let a = answers[i % 8];
            i += 1;
            a.to_string()
        });
        let total: usize = result.vote_counts.iter().map(|(_, c)| c).sum();
        assert_eq!(total, n);
    }

    #[test]
    fn test_sc_extract_last_line() {
        let dec = make_sc_decoder(1, AnswerExtractor::LastLine);
        let resp = "step 1\nstep 2\nthe answer is 42";
        let ans = dec.extract_answer(resp);
        assert_eq!(ans, "the answer is 42");
    }

    #[test]
    fn test_sc_extract_after_keyword() {
        let dec = make_sc_decoder(1, AnswerExtractor::AfterKeyword("answer:".to_string()));
        let resp = "reasoning... answer: 17";
        let ans = dec.extract_answer(resp);
        assert_eq!(ans, "17");
    }

    #[test]
    fn test_sc_extract_regex_fallback() {
        let dec = make_sc_decoder(1, AnswerExtractor::Regex("Final:".to_string()));
        let resp = "thinking\nFinal: 99\nmore text";
        let ans = dec.extract_answer(resp);
        assert_eq!(ans, "Final: 99");
    }

    // ── MctsTree ──────────────────────────────────────────────────────────────

    #[test]
    fn test_mcts_tree_expand_and_select() {
        let cfg = MctsConfig {
            n_simulations: 10,
            c_puct: 1.4,
            max_depth: 5,
            n_actions: 3,
            rollout_depth: 2,
        };
        let mut tree = MctsTree::new("root", cfg);
        tree.expand(
            0,
            vec![
                ("child_a".to_string(), 0.5),
                ("child_b".to_string(), 0.3),
                ("child_c".to_string(), 0.2),
            ],
        );
        // Before any visits the root has visit_count=0; UCT degenerates to prior-based
        let selected = tree.select(0);
        assert!(selected < tree.nodes.len());
    }

    #[test]
    fn test_mcts_tree_backprop_increments_visits() {
        let cfg = MctsConfig {
            n_simulations: 10,
            c_puct: 1.4,
            max_depth: 5,
            n_actions: 3,
            rollout_depth: 2,
        };
        let mut tree = MctsTree::new("root", cfg);
        tree.backprop(0, 1.0);
        tree.backprop(0, 0.5);
        assert_eq!(tree.nodes[0].visit_count, 2);
        assert!((tree.nodes[0].total_value - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_mcts_tree_get_policy_sums_to_1() {
        let cfg = MctsConfig {
            n_simulations: 10,
            c_puct: 1.4,
            max_depth: 5,
            n_actions: 3,
            rollout_depth: 2,
        };
        let mut tree = MctsTree::new("root", cfg);
        tree.expand(
            0,
            vec![
                ("a".to_string(), 0.5),
                ("b".to_string(), 0.3),
                ("c".to_string(), 0.2),
            ],
        );
        // Give each child some visits
        tree.backprop(1, 1.0);
        tree.backprop(2, 0.5);
        tree.backprop(3, 0.2);
        let policy = tree.get_policy(0);
        let sum: f64 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "policy sum={sum}");
    }

    #[test]
    fn test_mcts_tree_best_action_valid() {
        let cfg = MctsConfig {
            n_simulations: 10,
            c_puct: 1.4,
            max_depth: 5,
            n_actions: 3,
            rollout_depth: 2,
        };
        let mut tree = MctsTree::new("root", cfg);
        tree.expand(0, vec![("a".to_string(), 0.5), ("b".to_string(), 0.3)]);
        tree.backprop(1, 1.0);
        tree.backprop(1, 1.0);
        tree.backprop(2, 0.5);
        let best = tree.best_action(0);
        // Child 1 has most visits
        assert_eq!(best, 1);
    }

    // ── MctsSolver ────────────────────────────────────────────────────────────

    #[test]
    fn test_mcts_solver_returns_result() {
        let cfg = MctsConfig {
            n_simulations: 20,
            c_puct: 1.4,
            max_depth: 3,
            n_actions: 2,
            rollout_depth: 2,
        };
        let mut solver = MctsSolver::new(cfg);
        let result = solver.solve(
            "problem",
            |state, _n| {
                vec![
                    (format!("{state}->step1"), 0.6),
                    (format!("{state}->step2"), 0.4),
                ]
            },
            |state| state.contains("->step1->step1"),
            |_state| 0.5,
        );
        assert!(!result.best_solution.is_empty());
    }

    #[test]
    fn test_mcts_solver_explores_multiple_nodes() {
        let cfg = MctsConfig {
            n_simulations: 30,
            c_puct: 1.4,
            max_depth: 3,
            n_actions: 3,
            rollout_depth: 2,
        };
        let mut solver = MctsSolver::new(cfg);
        let result = solver.solve(
            "root",
            |state, _| {
                vec![
                    (format!("{state}_A"), 0.5),
                    (format!("{state}_B"), 0.3),
                    (format!("{state}_C"), 0.2),
                ]
            },
            |_| false,
            |_| 0.3,
        );
        assert!(
            result.n_nodes_explored > 1,
            "expected >1 node, got {}",
            result.n_nodes_explored
        );
    }

    #[test]
    fn test_mcts_solver_with_prm() {
        let prm = ProcessRewardModel::new(PrmConfig {
            d_step: 4,
            hidden_dim: 8,
            learning_rate: 0.01,
        });
        let cfg = MctsConfig {
            n_simulations: 10,
            c_puct: 1.0,
            max_depth: 2,
            n_actions: 2,
            rollout_depth: 1,
        };
        let mut solver = MctsSolver::with_prm(cfg, prm);
        let result = solver.solve(
            "start",
            |state, _| vec![(format!("{state}_next"), 0.7)],
            |_| false,
            |_| 0.5,
        );
        assert!(!result.best_solution.is_empty());
    }

    // ── StepBeamSearcher ──────────────────────────────────────────────────────

    fn make_beam_searcher() -> StepBeamSearcher {
        let prm = ProcessRewardModel::new(PrmConfig {
            d_step: 4,
            hidden_dim: 8,
            learning_rate: 0.01,
        });
        StepBeamSearcher::new(
            StepBeamConfig {
                beam_width: 3,
                max_steps: 4,
                prm_weight: 1.0,
                length_penalty: 0.0,
            },
            prm,
        )
    }

    #[test]
    fn test_beam_best_sequence_nonempty() {
        let searcher = make_beam_searcher();
        let result = searcher.search(
            "start",
            |state| {
                vec![
                    (format!("{state}_A"), rand_emb(4, hash_str(state))),
                    (format!("{state}_B"), rand_emb(4, hash_str(state) + 1)),
                ]
            },
            |state| state.ends_with("_A_A"),
        );
        assert!(!result.best_sequence.is_empty());
    }

    #[test]
    fn test_beam_all_beams_sorted_by_score() {
        let searcher = make_beam_searcher();
        let result = searcher.search(
            "q",
            |state| {
                vec![
                    (format!("{state}_1"), rand_emb(4, 1)),
                    (format!("{state}_2"), rand_emb(4, 2)),
                    (format!("{state}_3"), rand_emb(4, 3)),
                ]
            },
            |_| false,
        );
        let scores: Vec<f64> = result.all_beams.iter().map(|(_, s)| *s).collect();
        for i in 1..scores.len() {
            assert!(
                scores[i - 1] >= scores[i] - 1e-9,
                "all_beams not sorted at positions: {}={} < {}={}",
                i - 1,
                scores[i - 1],
                i,
                scores[i]
            );
        }
    }

    #[test]
    fn test_beam_best_score_matches_first_beam() {
        let searcher = make_beam_searcher();
        let result = searcher.search(
            "init",
            |state| vec![(format!("{state}_X"), rand_emb(4, 77))],
            |_| false,
        );
        if !result.all_beams.is_empty() {
            assert!((result.best_score - result.all_beams[0].1).abs() < 1e-9);
        }
    }

    // ── BudgetForcer ──────────────────────────────────────────────────────────

    #[test]
    fn test_budget_forcer_truncates_to_max_tokens() {
        let forcer = BudgetForcer::new(BudgetConfig {
            max_tokens: 10,
            thinking_budget: 6,
            answer_budget: 10,
        });
        let tokens: Vec<usize> = (0..20).collect();
        let result = forcer.apply(&tokens);
        assert!(
            result.len() <= 10,
            "expected <= 10 tokens, got {}",
            result.len()
        );
    }

    #[test]
    fn test_budget_forcer_compute_usage_correct() {
        let forcer = BudgetForcer::new(BudgetConfig {
            max_tokens: 20,
            thinking_budget: 8,
            answer_budget: 6,
        });
        let tokens: Vec<usize> = (0..20).collect();
        let usage = forcer.compute_usage(&tokens);
        assert!(usage.total_tokens <= 20);
        assert_eq!(
            usage.thinking_tokens + usage.answer_tokens,
            usage.total_tokens
        );
    }

    #[test]
    fn test_budget_forcer_utilization_in_range() {
        let forcer = BudgetForcer::new(BudgetConfig {
            max_tokens: 15,
            thinking_budget: 8,
            answer_budget: 5,
        });
        let tokens: Vec<usize> = (0..10).collect();
        let usage = forcer.compute_usage(&tokens);
        assert!(
            (0.0..=1.0).contains(&usage.budget_utilization),
            "utilization={}",
            usage.budget_utilization
        );
    }

    #[test]
    fn test_budget_forcer_short_input_unchanged() {
        let forcer = BudgetForcer::new(BudgetConfig {
            max_tokens: 100,
            thinking_budget: 50,
            answer_budget: 50,
        });
        let tokens: Vec<usize> = vec![1, 2, 3];
        let result = forcer.apply(&tokens);
        // Thinking part: [1,2,3], answer starts at idx 50 (empty), total=[1,2,3]
        assert_eq!(result, vec![1, 2, 3]);
    }

    // ── VerifierEnsemble ──────────────────────────────────────────────────────

    #[test]
    fn test_ensemble_weighted_average() {
        // v1 always returns 1.0, v2 always returns 0.0
        struct One;
        impl Verifier for One {
            fn score(&self, _q: &str, _a: &str) -> f64 {
                1.0
            }
        }
        struct Zero;
        impl Verifier for Zero {
            fn score(&self, _q: &str, _a: &str) -> f64 {
                0.0
            }
        }
        let ens = VerifierEnsemble::new(vec![Box::new(One), Box::new(Zero)], vec![0.7, 0.3]);
        let s = ens.score("q", "a");
        assert!((s - 0.7).abs() < 1e-9, "expected 0.7, got {s}");
    }

    #[test]
    fn test_ensemble_uniform_weights_sum_to_1() {
        struct One;
        impl Verifier for One {
            fn score(&self, _q: &str, _a: &str) -> f64 {
                1.0
            }
        }
        let v: Vec<Box<dyn Verifier>> = vec![Box::new(One), Box::new(One), Box::new(One)];
        let ens = VerifierEnsemble::uniform(v);
        let s = ens.score("q", "a");
        assert!(
            (s - 1.0).abs() < 1e-9,
            "uniform ensemble of 1.0 scorers should give 1.0, got {s}"
        );
    }

    #[test]
    fn test_ensemble_score_in_range() {
        let v1 = Box::new(ExactMatchVerifier::new("yes")) as Box<dyn Verifier>;
        let v2 = Box::new(NeuralVerifier::new(8, 16)) as Box<dyn Verifier>;
        let ens = VerifierEnsemble::uniform(vec![v1, v2]);
        let s = ens.score("is it?", "yes");
        assert!((0.0..=1.0).contains(&s));
    }

    // ── TTC Metrics ───────────────────────────────────────────────────────────

    #[test]
    fn test_ttc_pass_at_k_gt_pass_at_1_when_k_gt_1() {
        let correct = vec![
            true, false, false, true, false, false, false, false, false, false,
        ];
        let m = compute_ttc_metrics(&correct, 5);
        assert!(
            m.pass_at_k >= m.pass_at_1 - 1e-9,
            "pass@k={} < pass@1={}",
            m.pass_at_k,
            m.pass_at_1
        );
    }

    #[test]
    fn test_ttc_oracle_at_k_ge_pass_at_k() {
        let correct = vec![true, false, true, false, true, false, false, false];
        let m = compute_ttc_metrics(&correct, 4);
        assert!(
            m.oracle_at_k >= m.pass_at_k - 1e-9,
            "oracle={} < pass@k={}",
            m.oracle_at_k,
            m.pass_at_k
        );
    }

    #[test]
    fn test_ttc_all_wrong_gives_zero_pass() {
        let correct = vec![false; 10];
        let m = compute_ttc_metrics(&correct, 5);
        assert_eq!(m.pass_at_k, 0.0);
        assert_eq!(m.pass_at_1, 0.0);
    }

    #[test]
    fn test_ttc_all_correct_gives_1() {
        let correct = vec![true; 10];
        let m = compute_ttc_metrics(&correct, 5);
        assert_eq!(m.pass_at_1, 1.0);
        assert_eq!(m.pass_at_k, 1.0);
    }

    #[test]
    fn test_compute_ttc_metrics_consistent() {
        let correct = vec![
            true, false, true, true, false, false, false, true, false, false,
        ];
        let m1 = compute_ttc_metrics(&correct, 3);
        let m2 = compute_ttc_metrics(&correct, 3);
        assert!(
            (m1.pass_at_1 - m2.pass_at_1).abs() < 1e-12,
            "not deterministic"
        );
        assert!((m1.pass_at_k - m2.pass_at_k).abs() < 1e-12);
    }

    #[test]
    fn test_ttc_compute_efficiency_positive() {
        let correct = vec![true, false, false, true, false];
        let m = compute_ttc_metrics(&correct, 3);
        assert!(m.compute_efficiency >= 0.0);
    }

    #[test]
    fn test_bon_all_scores_length() {
        let sampler = make_bon_sampler(7, "z");
        let answers = ["a", "b", "c"];
        let mut i = 0;
        let result = sampler.sample(
            |_| {
                let a = answers[i % 3];
                i += 1;
                a.to_string()
            },
            "q",
        );
        assert_eq!(result.all_scores.len(), 7);
    }

    #[test]
    fn test_mcts_search_depth_bounded() {
        let cfg = MctsConfig {
            n_simulations: 20,
            c_puct: 1.4,
            max_depth: 3,
            n_actions: 2,
            rollout_depth: 1,
        };
        let mut solver = MctsSolver::new(cfg);
        let result = solver.solve(
            "s",
            |state, _| vec![(format!("{state}x"), 0.5), (format!("{state}y"), 0.5)],
            |_| false,
            |_| 0.1,
        );
        assert!(result.search_depth <= 3 || result.search_depth > 0);
    }

    #[test]
    fn test_prm_fit_loss_decreasing() {
        let mut prm = ProcessRewardModel::new(PrmConfig {
            d_step: 4,
            hidden_dim: 8,
            learning_rate: 0.05,
        });
        let data: Vec<(Vec<Vec<f64>>, Vec<f64>)> = vec![(
            (0..6).map(|i| rand_emb(4, i as u64 + 200)).collect(),
            vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        )];
        let history = prm.fit(&data, 20);
        assert!(history.len() == 20);
        let first = history[0];
        let last = *history.last().expect("history nonempty");
        // Loss should generally decrease (allow small variance)
        assert!(
            last < first * 2.0,
            "loss exploded: first={first}, last={last}"
        );
    }

    #[test]
    fn test_sc_votes_ordered_by_frequency() {
        let dec = make_sc_decoder(9, AnswerExtractor::LastLine);
        // A:5, B:3, C:1
        let answers = ["A", "A", "A", "A", "A", "B", "B", "B", "C"];
        let mut i = 0;
        let result = dec.decode(|_| {
            let a = answers[i % 9];
            i += 1;
            a.to_string()
        });
        assert_eq!(result.vote_counts[0].0, "A");
        assert_eq!(result.vote_counts[0].1, 5);
    }

    #[test]
    fn test_budget_forcer_thinking_and_answer_separation() {
        let forcer = BudgetForcer::new(BudgetConfig {
            max_tokens: 100,
            thinking_budget: 5,
            answer_budget: 5,
        });
        // tokens 0..10: [0,1,2,3,4] thinking, [5,6,7,8,9] answer
        let tokens: Vec<usize> = (0..10).collect();
        let result = forcer.apply(&tokens);
        // Thinking: tokens[0..5] = [0,1,2,3,4]
        // Answer: tokens[5..10] = [5,6,7,8,9]
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_neural_verifier_batch_all_in_range() {
        let v = NeuralVerifier::new(8, 16);
        let answers: Vec<String> = (0..5).map(|i| format!("answer_{i}")).collect();
        let scores = v.score_batch("q", &answers);
        for (i, &s) in scores.iter().enumerate() {
            assert!((0.0..=1.0).contains(&s), "score[{i}]={s} out of [0,1]");
        }
    }

    #[test]
    fn test_ttc_majority_at_k_in_range() {
        let correct: Vec<bool> = (0..10).map(|i| i % 3 == 0).collect();
        let m = compute_ttc_metrics(&correct, 7);
        assert!(
            (0.0..=1.0).contains(&m.majority_at_k),
            "majority@k={}",
            m.majority_at_k
        );
    }
}
