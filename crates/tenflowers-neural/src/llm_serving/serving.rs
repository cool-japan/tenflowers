//! LLM serving infrastructure: RLHF, batching, warmup, token classification, metrics, batch processing.

use std::collections::{HashMap, VecDeque};
use tenflowers_core::{Result, TensorError};

use super::core::{ls_sigmoid, ls_softmax, LsLcg, LsLinear};

// ─────────────────────────────────────────────────────────────────────────────
// §4  RLHFRewardModel — Bradley-Terry preference model
// ─────────────────────────────────────────────────────────────────────────────

/// Reward head: linear projection from hidden states to scalar reward.
#[derive(Debug, Clone)]
pub struct LsRewardHead {
    pub linear: LsLinear,
}

impl LsRewardHead {
    pub fn new(hidden_dim: usize) -> Self {
        Self {
            linear: LsLinear::new(hidden_dim, 1),
        }
    }

    /// Compute scalar reward from hidden state.
    pub fn forward(&self, hidden: &[f64]) -> f64 {
        let out = self.linear.forward(hidden);
        out.first().copied().unwrap_or(0.0)
    }
}

/// Running reward statistics for normalization.
#[derive(Debug, Clone)]
pub struct LsRewardNormalizer {
    pub mean: f64,
    pub var: f64,
    pub count: u64,
    pub epsilon: f64,
}

impl LsRewardNormalizer {
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            var: 1.0,
            count: 0,
            epsilon: 1e-8,
        }
    }

    /// Update running statistics with a new reward value (Welford's algorithm).
    pub fn update(&mut self, reward: f64) {
        self.count += 1;
        let delta = reward - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = reward - self.mean;
        self.var += (delta * delta2 - self.var) / self.count as f64;
    }

    /// Normalize a reward using running mean/std.
    pub fn normalize(&self, reward: f64) -> f64 {
        let std = (self.var + self.epsilon).sqrt();
        (reward - self.mean) / std
    }
}

impl Default for LsRewardNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

/// RLHF Reward Model with Bradley-Terry pairwise preference learning.
///
/// P(a > b) = sigmoid(r(a) - r(b))
#[derive(Debug)]
pub struct LsRlhfRewardModel {
    pub reward_head: LsRewardHead,
    pub normalizer: LsRewardNormalizer,
    pub hidden_dim: usize,
    /// Margin for margin-based loss variant.
    pub margin: f64,
    /// Learning rate.
    pub lr: f64,
    /// Training loss history.
    pub loss_history: Vec<f64>,
}

impl LsRlhfRewardModel {
    pub fn new(hidden_dim: usize, margin: f64, lr: f64) -> Self {
        Self {
            reward_head: LsRewardHead::new(hidden_dim),
            normalizer: LsRewardNormalizer::new(),
            hidden_dim,
            margin,
            lr,
            loss_history: Vec::new(),
        }
    }

    /// Compute reward for a sequence's hidden state.
    pub fn compute_reward(&mut self, hidden: &[f64]) -> f64 {
        let raw = self.reward_head.forward(hidden);
        self.normalizer.update(raw);
        raw
    }

    /// Compute reward for a batch of sequences.
    pub fn compute_rewards(&mut self, batch: &[Vec<f64>]) -> Vec<f64> {
        batch.iter().map(|h| self.compute_reward(h)).collect()
    }

    /// Bradley-Terry cross-entropy loss on a preference pair.
    /// chosen_hidden and rejected_hidden are the last-token hidden states.
    pub fn preference_loss(&self, chosen_hidden: &[f64], rejected_hidden: &[f64]) -> f64 {
        let r_chosen = self.reward_head.forward(chosen_hidden);
        let r_rejected = self.reward_head.forward(rejected_hidden);
        let diff = r_chosen - r_rejected;
        // -log(sigmoid(diff))
        -ls_sigmoid(diff).max(1e-30).ln()
    }

    /// Margin-based loss: max(0, margin - (r_chosen - r_rejected)) + CE loss.
    pub fn margin_loss(&self, chosen_hidden: &[f64], rejected_hidden: &[f64]) -> f64 {
        let r_chosen = self.reward_head.forward(chosen_hidden);
        let r_rejected = self.reward_head.forward(rejected_hidden);
        let diff = r_chosen - r_rejected;
        let ce = -ls_sigmoid(diff).max(1e-30).ln();
        let margin_penalty = (self.margin - diff).max(0.0);
        ce + margin_penalty
    }

    /// Train on a batch of preference pairs using SGD.
    /// Returns average loss.
    pub fn train_step(
        &mut self,
        chosen_batch: &[Vec<f64>],
        rejected_batch: &[Vec<f64>],
    ) -> Result<f64> {
        if chosen_batch.len() != rejected_batch.len() {
            return Err(TensorError::compute_error_simple(
                "Chosen and rejected batch sizes must match".to_string(),
            ));
        }
        if chosen_batch.is_empty() {
            return Ok(0.0);
        }

        let mut total_loss = 0.0;

        for (chosen, rejected) in chosen_batch.iter().zip(rejected_batch.iter()) {
            let loss = self.preference_loss(chosen, rejected);
            total_loss += loss;

            // SGD update via finite differences
            let r_c = self.reward_head.forward(chosen);
            let r_r = self.reward_head.forward(rejected);
            let diff = r_c - r_r;
            let grad_scale = ls_sigmoid(diff) - 1.0; // -sigmoid(-diff) = sigmoid(diff) - 1

            // Update weights: grad w.r.t. reward_head weights
            let mut grad_w = vec![vec![0.0; self.hidden_dim]; 1];
            let mut grad_b = vec![0.0; 1];

            // d(loss)/d(w) = grad_scale * (chosen - rejected)
            for j in 0..self.hidden_dim {
                let c_j = chosen.get(j).copied().unwrap_or(0.0);
                let r_j = rejected.get(j).copied().unwrap_or(0.0);
                grad_w[0][j] = grad_scale * (c_j - r_j);
            }
            grad_b[0] = grad_scale;

            self.reward_head.linear.update(&grad_w, &grad_b, self.lr);
        }

        let avg_loss = total_loss / chosen_batch.len() as f64;
        self.loss_history.push(avg_loss);
        Ok(avg_loss)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  DynamicBatcher — request batching for serving
// ─────────────────────────────────────────────────────────────────────────────

/// Priority of a serving request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LsRequestPriority {
    Low = 0,
    Normal = 1,
    High = 2,
}

/// A single serving request.
#[derive(Debug, Clone)]
pub struct LsServingRequest {
    pub request_id: u64,
    pub token_ids: Vec<usize>,
    pub max_new_tokens: usize,
    pub priority: LsRequestPriority,
    /// Arrival timestamp (monotonic counter).
    pub arrival_time: u64,
}

/// Batch of requests, padded to uniform length.
#[derive(Debug, Clone)]
pub struct LsBatch {
    pub request_ids: Vec<u64>,
    /// Padded token matrix: batch_size x max_seq_len.
    pub padded_tokens: Vec<Vec<usize>>,
    /// Attention mask: 1 = real token, 0 = padding.
    pub attention_mask: Vec<Vec<u8>>,
    /// Maximum sequence length in this batch.
    pub max_seq_len: usize,
}

/// Metrics for a batch.
#[derive(Debug, Clone)]
pub struct LsBatchMetrics {
    pub batch_size: usize,
    pub max_seq_len: usize,
    pub avg_seq_len: f64,
    pub padding_ratio: f64,
    pub utilization: f64,
}

/// Dynamic batcher with continuous batching and priority scheduling.
#[derive(Debug)]
pub struct LsDynamicBatcher {
    pub max_batch_size: usize,
    pub max_wait_time_ms: u64,
    pub pad_token: usize,
    /// Priority queue of pending requests.
    queue: VecDeque<LsServingRequest>,
    /// Current time counter.
    time_counter: u64,
    /// Historical batch metrics.
    pub metrics_history: Vec<LsBatchMetrics>,
}

impl LsDynamicBatcher {
    pub fn new(max_batch_size: usize, max_wait_time_ms: u64, pad_token: usize) -> Self {
        Self {
            max_batch_size,
            max_wait_time_ms,
            pad_token,
            queue: VecDeque::new(),
            time_counter: 0,
            metrics_history: Vec::new(),
        }
    }

    /// Add a request to the queue.
    pub fn add_request(&mut self, mut request: LsServingRequest) {
        self.time_counter += 1;
        request.arrival_time = self.time_counter;
        self.queue.push_back(request);
    }

    /// Number of pending requests.
    pub fn queue_depth(&self) -> usize {
        self.queue.len()
    }

    /// Form the next batch from the queue, sorted by priority then arrival time.
    pub fn form_batch(&mut self) -> Result<Option<LsBatch>> {
        if self.queue.is_empty() {
            return Ok(None);
        }

        // Sort by priority (desc) then arrival (asc)
        let mut candidates: Vec<LsServingRequest> = self.queue.drain(..).collect();
        candidates.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then(a.arrival_time.cmp(&b.arrival_time))
        });

        let take = candidates.len().min(self.max_batch_size);
        let batch_requests: Vec<LsServingRequest> = candidates.drain(..take).collect();

        // Put remaining back in queue
        for req in candidates {
            self.queue.push_back(req);
        }

        // Find max length for left-padding
        let max_len = batch_requests
            .iter()
            .map(|r| r.token_ids.len())
            .max()
            .unwrap_or(0);

        let batch_size = batch_requests.len();
        let mut padded_tokens = Vec::with_capacity(batch_size);
        let mut attention_mask = Vec::with_capacity(batch_size);
        let mut request_ids = Vec::with_capacity(batch_size);
        let mut total_real_tokens = 0usize;

        for req in &batch_requests {
            let seq_len = req.token_ids.len();
            let pad_len = max_len - seq_len;

            // Left-pad
            let mut tokens = vec![self.pad_token; pad_len];
            tokens.extend_from_slice(&req.token_ids);

            let mut mask = vec![0u8; pad_len];
            mask.extend(vec![1u8; seq_len]);

            total_real_tokens += seq_len;
            padded_tokens.push(tokens);
            attention_mask.push(mask);
            request_ids.push(req.request_id);
        }

        let total_slots = batch_size * max_len;
        let padding_ratio = if total_slots > 0 {
            1.0 - (total_real_tokens as f64 / total_slots as f64)
        } else {
            0.0
        };
        let utilization = batch_size as f64 / self.max_batch_size as f64;
        let avg_seq_len = if batch_size > 0 {
            total_real_tokens as f64 / batch_size as f64
        } else {
            0.0
        };

        let metrics = LsBatchMetrics {
            batch_size,
            max_seq_len: max_len,
            avg_seq_len,
            padding_ratio,
            utilization,
        };
        self.metrics_history.push(metrics);

        Ok(Some(LsBatch {
            request_ids,
            padded_tokens,
            attention_mask,
            max_seq_len: max_len,
        }))
    }

    /// Remove a completed request from the batch tracking.
    pub fn remove_request(&mut self, request_id: u64) -> bool {
        let before = self.queue.len();
        self.queue.retain(|r| r.request_id != request_id);
        self.queue.len() < before
    }

    /// Get average batch metrics.
    pub fn average_metrics(&self) -> LsBatchMetrics {
        if self.metrics_history.is_empty() {
            return LsBatchMetrics {
                batch_size: 0,
                max_seq_len: 0,
                avg_seq_len: 0.0,
                padding_ratio: 0.0,
                utilization: 0.0,
            };
        }
        let n = self.metrics_history.len() as f64;
        let avg_batch_size = self
            .metrics_history
            .iter()
            .map(|m| m.batch_size)
            .sum::<usize>() as f64
            / n;
        let avg_max_seq = self
            .metrics_history
            .iter()
            .map(|m| m.max_seq_len)
            .sum::<usize>() as f64
            / n;
        let avg_pad = self
            .metrics_history
            .iter()
            .map(|m| m.padding_ratio)
            .sum::<f64>()
            / n;
        let avg_util = self
            .metrics_history
            .iter()
            .map(|m| m.utilization)
            .sum::<f64>()
            / n;
        LsBatchMetrics {
            batch_size: avg_batch_size as usize,
            max_seq_len: avg_max_seq as usize,
            avg_seq_len: self
                .metrics_history
                .iter()
                .map(|m| m.avg_seq_len)
                .sum::<f64>()
                / n,
            padding_ratio: avg_pad,
            utilization: avg_util,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  ModelWarmup — profiling and calibration
// ─────────────────────────────────────────────────────────────────────────────

/// Per-layer timing result.
#[derive(Debug, Clone)]
pub struct LsLayerProfile {
    pub layer_name: String,
    pub latency_us: f64,
    pub param_count: usize,
}

/// Warmup and profiling report.
#[derive(Debug, Clone)]
pub struct LsWarmupReport {
    pub layer_profiles: Vec<LsLayerProfile>,
    pub total_latency_us: f64,
    pub peak_memory_bytes: usize,
    pub recommended_batch_size: usize,
    pub warmup_iterations: usize,
}

/// Model warmup and profiling engine.
///
/// Performs synthetic forward passes to warm up JIT/caches, measures per-layer
/// latency, and binary-searches for the optimal batch size.
#[derive(Debug)]
pub struct LsModelWarmup {
    pub num_warmup_iters: usize,
    pub max_batch_size: usize,
    /// Simulated memory limit in bytes.
    pub memory_limit_bytes: usize,
    /// Simulated bytes per element.
    pub bytes_per_element: usize,
}

impl LsModelWarmup {
    pub fn new(num_warmup_iters: usize, max_batch_size: usize, memory_limit_bytes: usize) -> Self {
        Self {
            num_warmup_iters,
            max_batch_size,
            memory_limit_bytes,
            bytes_per_element: 4, // f32
        }
    }

    /// Profile a list of layers with synthetic data.
    /// `layer_sizes` = [(name, in_dim, out_dim), ...].
    pub fn profile_layers(&self, layer_sizes: &[(&str, usize, usize)]) -> LsWarmupReport {
        let mut profiles = Vec::new();
        let mut total_latency = 0.0;
        let mut peak_mem = 0usize;

        for (name, in_d, out_d) in layer_sizes {
            let param_count = in_d * out_d + out_d;
            // Simulate latency proportional to FLOPs (2 * in * out per forward)
            let flops = 2.0 * (*in_d as f64) * (*out_d as f64);
            // Assume ~1 GFLOP/s throughput for simulation
            let latency_us = flops / 1000.0;

            let layer_mem = param_count * self.bytes_per_element;
            peak_mem = peak_mem.max(layer_mem);

            profiles.push(LsLayerProfile {
                layer_name: (*name).to_string(),
                latency_us,
                param_count,
            });
            total_latency += latency_us;
        }

        let total_params: usize = profiles.iter().map(|p| p.param_count).sum();
        let recommended = self.find_optimal_batch_size(total_params);

        LsWarmupReport {
            layer_profiles: profiles,
            total_latency_us: total_latency,
            peak_memory_bytes: peak_mem,
            recommended_batch_size: recommended,
            warmup_iterations: self.num_warmup_iters,
        }
    }

    /// Binary search for optimal batch size given memory constraints.
    pub(crate) fn find_optimal_batch_size(&self, total_params: usize) -> usize {
        let param_bytes = total_params * self.bytes_per_element;
        if param_bytes >= self.memory_limit_bytes {
            return 1;
        }
        let available = self.memory_limit_bytes - param_bytes;
        // Assume activation memory ~ batch_size * total_params * bytes_per_element / 4
        let activation_per_batch = (total_params * self.bytes_per_element) / 4;
        if activation_per_batch == 0 {
            return self.max_batch_size;
        }

        let mut lo = 1usize;
        let mut hi = self.max_batch_size;
        let mut best = 1usize;

        while lo <= hi {
            let mid = lo + (hi - lo) / 2;
            let mem_needed = mid * activation_per_batch;
            if mem_needed <= available {
                best = mid;
                if mid == hi {
                    break;
                }
                lo = mid + 1;
            } else {
                if mid == 0 {
                    break;
                }
                hi = mid - 1;
            }
        }
        best
    }

    /// Run warmup iterations with synthetic data (returns total latency estimate).
    pub fn run_warmup(&self, layer_sizes: &[(&str, usize, usize)]) -> f64 {
        let mut total = 0.0;
        for _ in 0..self.num_warmup_iters {
            for (_name, in_d, out_d) in layer_sizes {
                let flops = 2.0 * (*in_d as f64) * (*out_d as f64);
                total += flops / 1000.0;
            }
        }
        total
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  TokenClassifier — BIO/BILOU sequence labeling with CRF
// ─────────────────────────────────────────────────────────────────────────────

/// Tagging scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LsTaggingScheme {
    Bio,
    Bilou,
}

/// Entity extracted from BIO/BILOU tags.
#[derive(Debug, Clone, PartialEq)]
pub struct LsEntity {
    pub entity_type: String,
    pub start: usize,
    pub end: usize,
    pub text_span: String,
}

/// CRF transition matrix for structured prediction.
#[derive(Debug, Clone)]
pub struct LsCrfTransitions {
    /// Transition scores: transitions\[i\]\[j\] = score of transitioning from tag i to tag j.
    pub scores: Vec<Vec<f64>>,
    /// Start transition scores.
    pub start_scores: Vec<f64>,
    /// End transition scores.
    pub end_scores: Vec<f64>,
    pub num_tags: usize,
}

impl LsCrfTransitions {
    pub fn new(num_tags: usize) -> Self {
        let mut lcg = LsLcg::new(0xdeadbeef ^ (num_tags as u64));
        let scores: Vec<Vec<f64>> = (0..num_tags)
            .map(|_| (0..num_tags).map(|_| lcg.next_range(-0.1, 0.1)).collect())
            .collect();
        let start_scores: Vec<f64> = (0..num_tags).map(|_| lcg.next_range(-0.1, 0.1)).collect();
        let end_scores: Vec<f64> = (0..num_tags).map(|_| lcg.next_range(-0.1, 0.1)).collect();
        Self {
            scores,
            start_scores,
            end_scores,
            num_tags,
        }
    }
}

/// Token classifier with optional CRF layer for sequence labeling.
#[derive(Debug)]
pub struct LsTokenClassifier {
    pub linear: LsLinear,
    pub crf: Option<LsCrfTransitions>,
    pub num_labels: usize,
    pub tagging_scheme: LsTaggingScheme,
    /// Label names for entity extraction.
    pub label_names: Vec<String>,
}

impl LsTokenClassifier {
    pub fn new(
        hidden_dim: usize,
        num_labels: usize,
        use_crf: bool,
        scheme: LsTaggingScheme,
        label_names: Vec<String>,
    ) -> Result<Self> {
        if num_labels == 0 {
            return Err(TensorError::compute_error_simple(
                "num_labels must be > 0".to_string(),
            ));
        }
        Ok(Self {
            linear: LsLinear::new(hidden_dim, num_labels),
            crf: if use_crf {
                Some(LsCrfTransitions::new(num_labels))
            } else {
                None
            },
            num_labels,
            tagging_scheme: scheme,
            label_names,
        })
    }

    /// Forward pass: hidden_states\[t\] -> logits\[t\] per token.
    pub fn forward(&self, hidden_states: &[Vec<f64>]) -> Vec<Vec<f64>> {
        hidden_states
            .iter()
            .map(|h| self.linear.forward(h))
            .collect()
    }

    /// Greedy decode: argmax per position.
    pub fn decode_greedy(&self, logits: &[Vec<f64>]) -> Vec<usize> {
        logits
            .iter()
            .map(|l| {
                l.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Viterbi decoding using CRF transition matrix.
    pub fn decode_viterbi(&self, logits: &[Vec<f64>]) -> Result<Vec<usize>> {
        let crf = self
            .crf
            .as_ref()
            .ok_or_else(|| TensorError::compute_error_simple("CRF not initialized".to_string()))?;
        let n = logits.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let k = crf.num_tags;

        // dp[t][j] = best score ending at tag j at position t
        let mut dp = vec![vec![f64::NEG_INFINITY; k]; n];
        let mut backptr = vec![vec![0usize; k]; n];

        // Init: t=0
        for j in 0..k {
            let emit = logits[0].get(j).copied().unwrap_or(0.0);
            dp[0][j] = crf.start_scores[j] + emit;
        }

        // Forward pass
        for t in 1..n {
            for j in 0..k {
                let emit = logits[t].get(j).copied().unwrap_or(0.0);
                for i in 0..k {
                    let score = dp[t - 1][i] + crf.scores[i][j] + emit;
                    if score > dp[t][j] {
                        dp[t][j] = score;
                        backptr[t][j] = i;
                    }
                }
            }
        }

        // Add end scores and find best final tag
        let mut best_tag = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for j in 0..k {
            let score = dp[n - 1][j] + crf.end_scores[j];
            if score > best_score {
                best_score = score;
                best_tag = j;
            }
        }

        // Backtrack
        let mut path = vec![0usize; n];
        path[n - 1] = best_tag;
        for t in (0..n - 1).rev() {
            path[t] = backptr[t + 1][path[t + 1]];
        }
        Ok(path)
    }

    /// Decode logits using either CRF Viterbi or greedy.
    pub fn decode(&self, logits: &[Vec<f64>]) -> Result<Vec<usize>> {
        if self.crf.is_some() {
            self.decode_viterbi(logits)
        } else {
            Ok(self.decode_greedy(logits))
        }
    }

    /// Extract entities from BIO tag sequence.
    pub fn extract_entities(&self, tag_indices: &[usize], tokens: &[String]) -> Vec<LsEntity> {
        let mut entities = Vec::new();
        let mut current_entity: Option<(String, usize)> = None;

        for (i, &tag_idx) in tag_indices.iter().enumerate() {
            let label = self
                .label_names
                .get(tag_idx)
                .cloned()
                .unwrap_or_else(|| format!("TAG_{}", tag_idx));

            match self.tagging_scheme {
                LsTaggingScheme::Bio => {
                    if let Some(rest) = label.strip_prefix("B-") {
                        // Close previous entity
                        if let Some((etype, start)) = current_entity.take() {
                            let text = tokens[start..i].join(" ");
                            entities.push(LsEntity {
                                entity_type: etype,
                                start,
                                end: i,
                                text_span: text,
                            });
                        }
                        let etype = rest.to_string();
                        current_entity = Some((etype, i));
                    } else if label.starts_with("I-") {
                        // Continue only if matching type
                        if let Some((ref etype, _)) = current_entity {
                            let expected = format!("I-{}", etype);
                            if label != expected {
                                // Mismatch: close and start new
                                let (etype_old, start) =
                                    current_entity.take().unwrap_or_else(|| (String::new(), i));
                                let text = tokens[start..i].join(" ");
                                entities.push(LsEntity {
                                    entity_type: etype_old,
                                    start,
                                    end: i,
                                    text_span: text,
                                });
                            }
                        }
                    } else {
                        // O tag: close current entity
                        if let Some((etype, start)) = current_entity.take() {
                            let text = tokens[start..i].join(" ");
                            entities.push(LsEntity {
                                entity_type: etype,
                                start,
                                end: i,
                                text_span: text,
                            });
                        }
                    }
                }
                LsTaggingScheme::Bilou => {
                    if let Some(rest) = label.strip_prefix("U-") {
                        // Close previous
                        if let Some((etype, start)) = current_entity.take() {
                            let text = tokens[start..i].join(" ");
                            entities.push(LsEntity {
                                entity_type: etype,
                                start,
                                end: i,
                                text_span: text,
                            });
                        }
                        // Single-token entity
                        let etype = rest.to_string();
                        let text = tokens.get(i).cloned().unwrap_or_default();
                        entities.push(LsEntity {
                            entity_type: etype,
                            start: i,
                            end: i + 1,
                            text_span: text,
                        });
                    } else if let Some(rest) = label.strip_prefix("B-") {
                        if let Some((etype, start)) = current_entity.take() {
                            let text = tokens[start..i].join(" ");
                            entities.push(LsEntity {
                                entity_type: etype,
                                start,
                                end: i,
                                text_span: text,
                            });
                        }
                        let etype = rest.to_string();
                        current_entity = Some((etype, i));
                    } else if label.starts_with("L-") {
                        if let Some((etype, start)) = current_entity.take() {
                            let text = tokens[start..=i].join(" ");
                            entities.push(LsEntity {
                                entity_type: etype,
                                start,
                                end: i + 1,
                                text_span: text,
                            });
                        }
                    } else if !label.starts_with("I-") {
                        // O tag
                        if let Some((etype, start)) = current_entity.take() {
                            let text = tokens[start..i].join(" ");
                            entities.push(LsEntity {
                                entity_type: etype,
                                start,
                                end: i,
                                text_span: text,
                            });
                        }
                    }
                }
            }
        }

        // Close trailing entity
        if let Some((etype, start)) = current_entity.take() {
            let end = tag_indices.len();
            let text = tokens[start..end].join(" ");
            entities.push(LsEntity {
                entity_type: etype,
                start,
                end,
                text_span: text,
            });
        }

        entities
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  ServingMetrics — LLM serving performance metrics
// ─────────────────────────────────────────────────────────────────────────────

/// LLM serving performance metrics tracker.
#[derive(Debug, Clone)]
pub struct LsServingMetrics {
    /// Time to First Token samples (microseconds).
    pub ttft_samples: Vec<f64>,
    /// Tokens per second samples.
    pub tps_samples: Vec<f64>,
    /// Per-request total latency samples (microseconds).
    pub latency_samples: Vec<f64>,
    /// Queue wait time samples (microseconds).
    pub queue_wait_samples: Vec<f64>,
    /// Cache hit/miss counts.
    pub cache_hits: u64,
    pub cache_misses: u64,
    /// Queue depth snapshots.
    pub queue_depth_snapshots: Vec<usize>,
}

impl LsServingMetrics {
    pub fn new() -> Self {
        Self {
            ttft_samples: Vec::new(),
            tps_samples: Vec::new(),
            latency_samples: Vec::new(),
            queue_wait_samples: Vec::new(),
            cache_hits: 0,
            cache_misses: 0,
            queue_depth_snapshots: Vec::new(),
        }
    }

    pub fn record_ttft(&mut self, ttft_us: f64) {
        self.ttft_samples.push(ttft_us);
    }

    pub fn record_tps(&mut self, tps: f64) {
        self.tps_samples.push(tps);
    }

    pub fn record_latency(&mut self, latency_us: f64) {
        self.latency_samples.push(latency_us);
    }

    pub fn record_queue_wait(&mut self, wait_us: f64) {
        self.queue_wait_samples.push(wait_us);
    }

    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }

    pub fn record_queue_depth(&mut self, depth: usize) {
        self.queue_depth_snapshots.push(depth);
    }

    /// Compute a percentile from sorted samples.
    fn percentile(sorted: &[f64], p: f64) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
        let idx = idx.min(sorted.len() - 1);
        sorted[idx]
    }

    /// Cache hit rate.
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    /// Average queue depth.
    pub fn avg_queue_depth(&self) -> f64 {
        if self.queue_depth_snapshots.is_empty() {
            0.0
        } else {
            self.queue_depth_snapshots.iter().sum::<usize>() as f64
                / self.queue_depth_snapshots.len() as f64
        }
    }

    /// Generate a full serving report.
    pub fn report(&self) -> LsServingReport {
        let mut ttft_sorted = self.ttft_samples.clone();
        ttft_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut latency_sorted = self.latency_samples.clone();
        latency_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let avg_tps = if self.tps_samples.is_empty() {
            0.0
        } else {
            self.tps_samples.iter().sum::<f64>() / self.tps_samples.len() as f64
        };

        let avg_ttft = if ttft_sorted.is_empty() {
            0.0
        } else {
            ttft_sorted.iter().sum::<f64>() / ttft_sorted.len() as f64
        };

        LsServingReport {
            avg_ttft_us: avg_ttft,
            p50_ttft_us: Self::percentile(&ttft_sorted, 50.0),
            p95_ttft_us: Self::percentile(&ttft_sorted, 95.0),
            p99_ttft_us: Self::percentile(&ttft_sorted, 99.0),
            avg_tps,
            p50_latency_us: Self::percentile(&latency_sorted, 50.0),
            p95_latency_us: Self::percentile(&latency_sorted, 95.0),
            p99_latency_us: Self::percentile(&latency_sorted, 99.0),
            cache_hit_rate: self.cache_hit_rate(),
            avg_queue_depth: self.avg_queue_depth(),
            total_requests: self.latency_samples.len() as u64,
        }
    }
}

impl Default for LsServingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated serving report.
#[derive(Debug, Clone)]
pub struct LsServingReport {
    pub avg_ttft_us: f64,
    pub p50_ttft_us: f64,
    pub p95_ttft_us: f64,
    pub p99_ttft_us: f64,
    pub avg_tps: f64,
    pub p50_latency_us: f64,
    pub p95_latency_us: f64,
    pub p99_latency_us: f64,
    pub cache_hit_rate: f64,
    pub avg_queue_depth: f64,
    pub total_requests: u64,
}

impl LsServingReport {
    /// Check if metrics meet SLA requirements.
    pub fn check_sla(
        &self,
        max_p99_latency_us: f64,
        min_tps: f64,
        max_ttft_p95_us: f64,
    ) -> LsSlaCfompliance {
        LsSlaCfompliance {
            latency_ok: self.p99_latency_us <= max_p99_latency_us,
            tps_ok: self.avg_tps >= min_tps,
            ttft_ok: self.p95_ttft_us <= max_ttft_p95_us,
            overall_ok: self.p99_latency_us <= max_p99_latency_us
                && self.avg_tps >= min_tps
                && self.p95_ttft_us <= max_ttft_p95_us,
        }
    }
}

/// SLA compliance result.
#[derive(Debug, Clone)]
pub struct LsSlaCfompliance {
    pub latency_ok: bool,
    pub tps_ok: bool,
    pub ttft_ok: bool,
    pub overall_ok: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  ContinuousBatchProcessor — iteration-level batch engine
// ─────────────────────────────────────────────────────────────────────────────

/// State of a sequence in the continuous batch processor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LsSequenceState {
    /// Waiting for initial prefill.
    Prefill,
    /// Actively generating tokens.
    Decode,
    /// Generation complete.
    Complete,
    /// Paused due to memory pressure.
    Preempted,
}

/// A managed sequence in the continuous batch processor.
#[derive(Debug, Clone)]
pub struct LsManagedSequence {
    pub seq_id: u64,
    pub state: LsSequenceState,
    pub token_ids: Vec<usize>,
    pub generated_count: usize,
    pub max_new_tokens: usize,
    pub priority: LsRequestPriority,
    /// Memory cost estimate (in blocks).
    pub memory_blocks: usize,
}

/// Continuous batching engine with iteration-level management.
///
/// Manages sequences through prefill -> decode -> complete states,
/// with memory-aware admission control and preemption.
#[derive(Debug)]
pub struct LsContinuousBatchProcessor {
    /// All managed sequences.
    pub sequences: HashMap<u64, LsManagedSequence>,
    /// Maximum number of concurrent sequences.
    pub max_concurrent: usize,
    /// Maximum total memory blocks across all sequences.
    pub max_memory_blocks: usize,
    /// Current total memory blocks in use.
    pub used_memory_blocks: usize,
    /// Iteration counter.
    pub iteration: u64,
    /// Completed sequence IDs (for retrieval).
    pub completed: Vec<u64>,
    /// Blocks per token estimate.
    pub blocks_per_token: usize,
}

impl LsContinuousBatchProcessor {
    pub fn new(
        max_concurrent: usize,
        max_memory_blocks: usize,
        blocks_per_token: usize,
    ) -> Result<Self> {
        if max_concurrent == 0 || max_memory_blocks == 0 {
            return Err(TensorError::compute_error_simple(
                "max_concurrent and max_memory_blocks must be > 0".to_string(),
            ));
        }
        Ok(Self {
            sequences: HashMap::new(),
            max_concurrent,
            max_memory_blocks,
            used_memory_blocks: 0,
            iteration: 0,
            completed: Vec::new(),
            blocks_per_token: blocks_per_token.max(1),
        })
    }

    /// Check if a new sequence can be admitted.
    pub fn can_admit(&self, initial_tokens: usize) -> bool {
        let active_count = self
            .sequences
            .values()
            .filter(|s| {
                s.state != LsSequenceState::Complete && s.state != LsSequenceState::Preempted
            })
            .count();
        let needed_blocks = initial_tokens * self.blocks_per_token;
        active_count < self.max_concurrent
            && self.used_memory_blocks + needed_blocks <= self.max_memory_blocks
    }

    /// Add a new sequence for processing.
    pub fn add_sequence(
        &mut self,
        seq_id: u64,
        token_ids: Vec<usize>,
        max_new_tokens: usize,
        priority: LsRequestPriority,
    ) -> Result<()> {
        let needed_blocks = token_ids.len() * self.blocks_per_token;
        if !self.can_admit(token_ids.len()) {
            // Try preemption
            self.try_preempt(needed_blocks)?;
        }

        let mem_blocks = needed_blocks;
        self.used_memory_blocks += mem_blocks;
        self.sequences.insert(
            seq_id,
            LsManagedSequence {
                seq_id,
                state: LsSequenceState::Prefill,
                token_ids,
                generated_count: 0,
                max_new_tokens,
                priority,
                memory_blocks: mem_blocks,
            },
        );
        Ok(())
    }

    /// Try to preempt low-priority sequences to free memory.
    fn try_preempt(&mut self, needed_blocks: usize) -> Result<()> {
        let mut freed = 0usize;
        let mut to_preempt: Vec<u64> = Vec::new();

        // Sort sequences by priority (lowest first) for preemption
        let mut candidates: Vec<(u64, LsRequestPriority, usize)> = self
            .sequences
            .values()
            .filter(|s| s.state == LsSequenceState::Decode)
            .map(|s| (s.seq_id, s.priority, s.memory_blocks))
            .collect();
        candidates.sort_by_key(|(_, p, _)| *p);

        for (sid, _, mem) in &candidates {
            if freed >= needed_blocks {
                break;
            }
            to_preempt.push(*sid);
            freed += mem;
        }

        for sid in &to_preempt {
            if let Some(seq) = self.sequences.get_mut(sid) {
                seq.state = LsSequenceState::Preempted;
                self.used_memory_blocks = self.used_memory_blocks.saturating_sub(seq.memory_blocks);
            }
        }

        if freed < needed_blocks && !self.can_admit(0) {
            return Err(TensorError::compute_error_simple(
                "Cannot admit sequence: insufficient memory even after preemption".to_string(),
            ));
        }
        Ok(())
    }

    /// Process one iteration: advance all active sequences by one token.
    /// Returns the set of sequence IDs that were processed.
    pub fn step(&mut self) -> Result<Vec<u64>> {
        self.iteration += 1;
        let mut processed = Vec::new();

        // Collect IDs to process
        let active_ids: Vec<u64> = self
            .sequences
            .values()
            .filter(|s| s.state == LsSequenceState::Prefill || s.state == LsSequenceState::Decode)
            .map(|s| s.seq_id)
            .collect();

        for sid in active_ids {
            if let Some(seq) = self.sequences.get_mut(&sid) {
                match seq.state {
                    LsSequenceState::Prefill => {
                        // Transition to decode after prefill
                        seq.state = LsSequenceState::Decode;
                        processed.push(sid);
                    }
                    LsSequenceState::Decode => {
                        // Simulate generating one token
                        seq.generated_count += 1;
                        let new_blocks = self.blocks_per_token;
                        seq.memory_blocks += new_blocks;
                        self.used_memory_blocks += new_blocks;

                        if seq.generated_count >= seq.max_new_tokens {
                            seq.state = LsSequenceState::Complete;
                            self.used_memory_blocks =
                                self.used_memory_blocks.saturating_sub(seq.memory_blocks);
                            self.completed.push(sid);
                        }
                        processed.push(sid);
                    }
                    _ => {}
                }
            }
        }
        Ok(processed)
    }

    /// Resume a preempted sequence.
    pub fn resume_sequence(&mut self, seq_id: u64) -> Result<()> {
        let seq = self.sequences.get_mut(&seq_id).ok_or_else(|| {
            TensorError::compute_error_simple(format!("Sequence {} not found", seq_id))
        })?;
        if seq.state != LsSequenceState::Preempted {
            return Err(TensorError::compute_error_simple(format!(
                "Sequence {} is not preempted (state: {:?})",
                seq_id, seq.state
            )));
        }
        // Check memory availability
        if self.used_memory_blocks + seq.memory_blocks > self.max_memory_blocks {
            return Err(TensorError::compute_error_simple(
                "Insufficient memory to resume sequence".to_string(),
            ));
        }
        self.used_memory_blocks += seq.memory_blocks;
        seq.state = LsSequenceState::Decode;
        Ok(())
    }

    /// Get count of active (prefill + decode) sequences.
    pub fn active_count(&self) -> usize {
        self.sequences
            .values()
            .filter(|s| s.state == LsSequenceState::Prefill || s.state == LsSequenceState::Decode)
            .count()
    }

    /// Get count of completed sequences.
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// Memory utilization ratio.
    pub fn memory_utilization(&self) -> f64 {
        if self.max_memory_blocks == 0 {
            return 0.0;
        }
        self.used_memory_blocks as f64 / self.max_memory_blocks as f64
    }
}
