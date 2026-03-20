//! Core LLM serving components: utilities, KV caches, speculative decoding, instruction tuning.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::{HashMap, VecDeque};
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// §0  Shared utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically stable softmax over f64 slice.
pub(crate) fn ls_softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / logits.len() as f64; logits.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

/// Numerically stable log-softmax.
pub(crate) fn ls_log_softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let log_sum_exp = max_val
        + logits
            .iter()
            .map(|&x| (x - max_val).exp())
            .sum::<f64>()
            .ln();
    logits.iter().map(|&x| x - log_sum_exp).collect()
}

/// Stable sigmoid.
#[inline]
pub(crate) fn ls_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Simple LCG-based deterministic pseudo-random for reproducibility.
pub(crate) struct LsLcg {
    state: u64,
}

impl LsLcg {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.state >> 33) as u32;
        bits as f64 / u32::MAX as f64
    }

    pub(crate) fn next_range(&mut self, low: f64, high: f64) -> f64 {
        low + self.next_f64() * (high - low)
    }
}

/// Xavier-initialized linear layer for internal use.
#[derive(Debug, Clone)]
pub struct LsLinear {
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl LsLinear {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let limit = (6.0_f64 / (in_dim + out_dim) as f64).sqrt();
        let mut lcg = LsLcg::new(0x853c_49e6_748f_ea9b ^ (in_dim as u64 * 31 + out_dim as u64));
        let mut weights = vec![vec![0.0; in_dim]; out_dim];
        for row in weights.iter_mut() {
            for val in row.iter_mut() {
                *val = lcg.next_range(-limit, limit);
            }
        }
        Self {
            weights,
            bias: vec![0.0; out_dim],
            in_dim,
            out_dim,
        }
    }

    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(self.bias.iter())
            .map(|(row, &b)| {
                row.iter()
                    .zip(x.iter())
                    .map(|(&w, &xi)| w * xi)
                    .sum::<f64>()
                    + b
            })
            .collect()
    }

    pub fn update(&mut self, grad_w: &[Vec<f64>], grad_b: &[f64], lr: f64) {
        for (i, row) in self.weights.iter_mut().enumerate() {
            if let Some(grow) = grad_w.get(i) {
                for (j, wij) in row.iter_mut().enumerate() {
                    if let Some(&gw) = grow.get(j) {
                        *wij -= lr * gw;
                    }
                }
            }
        }
        for (bi, gi) in self.bias.iter_mut().zip(grad_b.iter()) {
            *bi -= lr * gi;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  KVCacheManager — block-based key-value cache
// ─────────────────────────────────────────────────────────────────────────────

/// A single block in the KV cache holding `block_size` token slots.
#[derive(Debug, Clone)]
pub struct LsKvBlock {
    /// Key vectors: shape [block_size, head_dim].
    pub keys: Vec<Vec<f64>>,
    /// Value vectors: shape [block_size, head_dim].
    pub values: Vec<Vec<f64>>,
    /// Number of slots currently filled.
    pub used: usize,
    /// Block capacity.
    pub capacity: usize,
    /// Last access timestamp for LRU eviction.
    pub last_access: u64,
}

impl LsKvBlock {
    pub fn new(block_size: usize, head_dim: usize) -> Self {
        Self {
            keys: vec![vec![0.0; head_dim]; block_size],
            values: vec![vec![0.0; head_dim]; block_size],
            used: 0,
            capacity: block_size,
            last_access: 0,
        }
    }

    pub fn is_full(&self) -> bool {
        self.used >= self.capacity
    }

    pub fn remaining(&self) -> usize {
        self.capacity - self.used
    }
}

/// Block-based KV cache manager with LRU eviction.
///
/// Organizes cache into fixed-size blocks for efficient memory management.
/// Each layer has its own set of blocks, enabling per-layer cache operations.
#[derive(Debug)]
pub struct LsKvCacheManager {
    /// block_size tokens per block.
    pub block_size: usize,
    /// Dimension of each key/value vector.
    pub head_dim: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Maximum number of blocks per layer.
    pub max_blocks_per_layer: usize,
    /// Per-layer block storage: blocks[layer_idx][block_idx].
    blocks: Vec<Vec<LsKvBlock>>,
    /// Global timestamp counter for LRU.
    timestamp: u64,
}

impl LsKvCacheManager {
    pub fn new(
        block_size: usize,
        head_dim: usize,
        num_layers: usize,
        max_blocks_per_layer: usize,
    ) -> Result<Self> {
        if block_size == 0 {
            return Err(TensorError::compute_error_simple(
                "block_size must be > 0".to_string(),
            ));
        }
        if head_dim == 0 {
            return Err(TensorError::compute_error_simple(
                "head_dim must be > 0".to_string(),
            ));
        }
        Ok(Self {
            block_size,
            head_dim,
            num_layers,
            max_blocks_per_layer,
            blocks: (0..num_layers).map(|_| Vec::new()).collect(),
            timestamp: 0,
        })
    }

    /// Allocate enough blocks for `seq_len` tokens at the given layer.
    /// Returns the indices of allocated blocks.
    pub fn allocate_blocks(&mut self, layer_idx: usize, seq_len: usize) -> Result<Vec<usize>> {
        if layer_idx >= self.num_layers {
            return Err(TensorError::compute_error_simple(format!(
                "layer_idx {} >= num_layers {}",
                layer_idx, self.num_layers
            )));
        }
        let needed = (seq_len + self.block_size - 1) / self.block_size;
        let layer_blocks = &mut self.blocks[layer_idx];
        let mut indices = Vec::with_capacity(needed);

        for _ in 0..needed {
            if layer_blocks.len() >= self.max_blocks_per_layer {
                // LRU eviction: find block with oldest last_access
                let evict_idx = layer_blocks
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, b)| b.last_access)
                    .map(|(i, _)| i)
                    .ok_or_else(|| {
                        TensorError::compute_error_simple("No blocks to evict".to_string())
                    })?;
                layer_blocks[evict_idx] = LsKvBlock::new(self.block_size, self.head_dim);
                self.timestamp += 1;
                layer_blocks[evict_idx].last_access = self.timestamp;
                indices.push(evict_idx);
            } else {
                let idx = layer_blocks.len();
                let mut block = LsKvBlock::new(self.block_size, self.head_dim);
                self.timestamp += 1;
                block.last_access = self.timestamp;
                layer_blocks.push(block);
                indices.push(idx);
            }
        }
        Ok(indices)
    }

    /// Append a key-value pair to the cache for a given layer.
    pub fn append_kv(&mut self, layer_idx: usize, key: &[f64], value: &[f64]) -> Result<()> {
        if layer_idx >= self.num_layers {
            return Err(TensorError::compute_error_simple(format!(
                "layer_idx {} out of range",
                layer_idx
            )));
        }
        if key.len() != self.head_dim || value.len() != self.head_dim {
            return Err(TensorError::compute_error_simple(format!(
                "key/value dim mismatch: expected {}, got key={} value={}",
                self.head_dim,
                key.len(),
                value.len()
            )));
        }

        let layer_blocks = &mut self.blocks[layer_idx];

        // Find a block with remaining capacity, or allocate new
        let block_idx = if let Some(idx) = layer_blocks.iter().position(|b| !b.is_full()) {
            idx
        } else {
            if layer_blocks.len() >= self.max_blocks_per_layer {
                // LRU evict
                let evict_idx = layer_blocks
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, b)| b.last_access)
                    .map(|(i, _)| i)
                    .ok_or_else(|| {
                        TensorError::compute_error_simple("No blocks to evict".to_string())
                    })?;
                layer_blocks[evict_idx] = LsKvBlock::new(self.block_size, self.head_dim);
                evict_idx
            } else {
                layer_blocks.push(LsKvBlock::new(self.block_size, self.head_dim));
                layer_blocks.len() - 1
            }
        };

        let block = &mut layer_blocks[block_idx];
        let slot = block.used;
        block.keys[slot] = key.to_vec();
        block.values[slot] = value.to_vec();
        block.used += 1;
        self.timestamp += 1;
        block.last_access = self.timestamp;
        Ok(())
    }

    /// Retrieve cached key-value pairs for a range of positions at a given layer.
    pub fn get_kv(
        &mut self,
        layer_idx: usize,
        start: usize,
        end: usize,
    ) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        if layer_idx >= self.num_layers {
            return Err(TensorError::compute_error_simple(format!(
                "layer_idx {} out of range",
                layer_idx
            )));
        }
        let layer_blocks = &mut self.blocks[layer_idx];
        let mut keys = Vec::new();
        let mut values = Vec::new();
        let mut pos = 0;

        for block in layer_blocks.iter_mut() {
            for slot in 0..block.used {
                if pos >= start && pos < end {
                    keys.push(block.keys[slot].clone());
                    values.push(block.values[slot].clone());
                }
                pos += 1;
            }
            if pos >= end {
                break;
            }
            self.timestamp += 1;
            block.last_access = self.timestamp;
        }
        Ok((keys, values))
    }

    /// Total number of cached tokens across all blocks for a layer.
    pub fn cached_tokens(&self, layer_idx: usize) -> usize {
        if layer_idx >= self.num_layers {
            return 0;
        }
        self.blocks[layer_idx].iter().map(|b| b.used).sum()
    }

    /// Clear all blocks for a given layer.
    pub fn clear_layer(&mut self, layer_idx: usize) {
        if layer_idx < self.num_layers {
            self.blocks[layer_idx].clear();
        }
    }

    /// Clear all layers.
    pub fn clear_all(&mut self) {
        for layer in self.blocks.iter_mut() {
            layer.clear();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §1b  PagedKVCache — virtual memory paged attention
// ─────────────────────────────────────────────────────────────────────────────

/// Block table entry mapping logical block -> physical block.
#[derive(Debug, Clone, Copy)]
pub struct LsBlockTableEntry {
    pub logical_block: usize,
    pub physical_block: usize,
}

/// Virtual-memory-style paged KV cache (inspired by vLLM PagedAttention).
///
/// Maps logical blocks per sequence to physical blocks in a shared pool,
/// enabling efficient memory sharing and copy-on-write semantics.
#[derive(Debug)]
pub struct LsPagedKvCache {
    pub block_size: usize,
    pub head_dim: usize,
    pub num_physical_blocks: usize,
    /// Physical block pool: pool[phys_idx] = KvBlock.
    pool: Vec<LsKvBlock>,
    /// Free physical block indices.
    free_list: VecDeque<usize>,
    /// Per-sequence block tables: seq_id -> [(logical, physical)].
    block_tables: HashMap<u64, Vec<LsBlockTableEntry>>,
    /// Reference counts per physical block (for CoW).
    ref_counts: Vec<usize>,
}

impl LsPagedKvCache {
    pub fn new(block_size: usize, head_dim: usize, num_physical_blocks: usize) -> Result<Self> {
        if block_size == 0 || head_dim == 0 || num_physical_blocks == 0 {
            return Err(TensorError::compute_error_simple(
                "All paged cache params must be > 0".to_string(),
            ));
        }
        let pool: Vec<LsKvBlock> = (0..num_physical_blocks)
            .map(|_| LsKvBlock::new(block_size, head_dim))
            .collect();
        let free_list: VecDeque<usize> = (0..num_physical_blocks).collect();
        Ok(Self {
            block_size,
            head_dim,
            num_physical_blocks,
            pool,
            free_list,
            block_tables: HashMap::new(),
            ref_counts: vec![0; num_physical_blocks],
        })
    }

    /// Allocate a physical block for a sequence. Returns physical block index.
    pub fn allocate_block(&mut self, seq_id: u64) -> Result<usize> {
        let phys_idx = self.free_list.pop_front().ok_or_else(|| {
            TensorError::compute_error_simple("No free physical blocks available".to_string())
        })?;
        self.ref_counts[phys_idx] = 1;
        let logical = self.block_tables.get(&seq_id).map_or(0, |t| t.len());
        self.block_tables
            .entry(seq_id)
            .or_default()
            .push(LsBlockTableEntry {
                logical_block: logical,
                physical_block: phys_idx,
            });
        Ok(phys_idx)
    }

    /// Append a KV pair into the last block of a sequence, allocating if needed.
    pub fn append_token(&mut self, seq_id: u64, key: &[f64], value: &[f64]) -> Result<()> {
        if key.len() != self.head_dim || value.len() != self.head_dim {
            return Err(TensorError::compute_error_simple(format!(
                "KV dim mismatch: expected {}",
                self.head_dim
            )));
        }

        // Check if last block has space, otherwise allocate new
        let need_new = {
            let table = self.block_tables.get(&seq_id);
            match table {
                None => true,
                Some(entries) => {
                    if entries.is_empty() {
                        true
                    } else {
                        let last_phys = entries.last().map(|e| e.physical_block);
                        match last_phys {
                            Some(idx) => self.pool[idx].is_full(),
                            None => true,
                        }
                    }
                }
            }
        };

        if need_new {
            self.allocate_block(seq_id)?;
        }

        let table = self.block_tables.get(&seq_id).ok_or_else(|| {
            TensorError::compute_error_simple("Sequence not found after allocation".to_string())
        })?;
        let phys_idx = table
            .last()
            .map(|e| e.physical_block)
            .ok_or_else(|| TensorError::compute_error_simple("Empty block table".to_string()))?;

        let block = &mut self.pool[phys_idx];
        let slot = block.used;
        block.keys[slot] = key.to_vec();
        block.values[slot] = value.to_vec();
        block.used += 1;
        Ok(())
    }

    /// Get all cached KV pairs for a sequence.
    pub fn get_sequence_kv(&self, seq_id: u64) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let table = self.block_tables.get(&seq_id).ok_or_else(|| {
            TensorError::compute_error_simple(format!("Sequence {} not found", seq_id))
        })?;
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for entry in table {
            let block = &self.pool[entry.physical_block];
            for slot in 0..block.used {
                keys.push(block.keys[slot].clone());
                values.push(block.values[slot].clone());
            }
        }
        Ok((keys, values))
    }

    /// Free all blocks belonging to a sequence.
    pub fn free_sequence(&mut self, seq_id: u64) -> Result<()> {
        let table = self.block_tables.remove(&seq_id).ok_or_else(|| {
            TensorError::compute_error_simple(format!("Sequence {} not found", seq_id))
        })?;
        for entry in &table {
            let phys = entry.physical_block;
            self.ref_counts[phys] = self.ref_counts[phys].saturating_sub(1);
            if self.ref_counts[phys] == 0 {
                self.pool[phys] = LsKvBlock::new(self.block_size, self.head_dim);
                self.free_list.push_back(phys);
            }
        }
        Ok(())
    }

    /// Number of free physical blocks remaining.
    pub fn free_blocks(&self) -> usize {
        self.free_list.len()
    }

    /// Number of active sequences.
    pub fn active_sequences(&self) -> usize {
        self.block_tables.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  SpeculativeDecoder — speculative decoding with adaptive K
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a speculative decoding step.
#[derive(Debug, Clone)]
pub struct LsSpeculativeResult {
    /// Accepted token indices.
    pub accepted_tokens: Vec<usize>,
    /// Number of accepted tokens.
    pub n_accepted: usize,
    /// Acceptance rate for this step.
    pub acceptance_rate: f64,
}

/// Token tree node for tree-based speculative verification.
#[derive(Debug, Clone)]
pub struct LsTokenTreeNode {
    pub token_id: usize,
    pub log_prob: f64,
    pub children: Vec<LsTokenTreeNode>,
}

impl LsTokenTreeNode {
    pub fn new(token_id: usize, log_prob: f64) -> Self {
        Self {
            token_id,
            log_prob,
            children: Vec::new(),
        }
    }

    pub fn add_child(&mut self, child: LsTokenTreeNode) {
        self.children.push(child);
    }
}

/// Speculative decoding engine (Leviathan et al. 2023).
///
/// A draft model proposes K candidate tokens; the target model verifies them
/// in a single forward pass using acceptance-rejection sampling.
#[derive(Debug)]
pub struct LsSpeculativeDecoder {
    /// Current speculation depth.
    pub k: usize,
    /// Minimum K.
    pub k_min: usize,
    /// Maximum K.
    pub k_max: usize,
    /// Exponential moving average of acceptance rate.
    pub ema_acceptance: f64,
    /// EMA decay factor.
    pub ema_alpha: f64,
    /// Increase K threshold.
    pub increase_threshold: f64,
    /// Decrease K threshold.
    pub decrease_threshold: f64,
    /// Total tokens accepted across all steps.
    pub total_accepted: u64,
    /// Total tokens proposed across all steps.
    pub total_proposed: u64,
    /// RNG seed.
    rng: StdRng,
}

impl LsSpeculativeDecoder {
    pub fn new(k: usize, k_min: usize, k_max: usize, seed: u64) -> Result<Self> {
        if k_min == 0 || k_max < k_min || k < k_min || k > k_max {
            return Err(TensorError::compute_error_simple(
                "Invalid k parameters: need 0 < k_min <= k <= k_max".to_string(),
            ));
        }
        Ok(Self {
            k,
            k_min,
            k_max,
            ema_acceptance: 0.8,
            ema_alpha: 0.1,
            increase_threshold: 0.8,
            decrease_threshold: 0.4,
            total_accepted: 0,
            total_proposed: 0,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// Core speculative decode: given draft and target log-probabilities for K positions,
    /// perform acceptance-rejection and return accepted tokens.
    ///
    /// `draft_logits[i]` and `target_logits[i]` are logit vectors at position i.
    /// `draft_tokens[i]` is the token chosen by the draft model at position i.
    pub fn decode(
        &mut self,
        draft_tokens: &[usize],
        draft_logits: &[Vec<f64>],
        target_logits: &[Vec<f64>],
    ) -> Result<LsSpeculativeResult> {
        let n = draft_tokens
            .len()
            .min(draft_logits.len())
            .min(target_logits.len());
        if n == 0 {
            return Ok(LsSpeculativeResult {
                accepted_tokens: Vec::new(),
                n_accepted: 0,
                acceptance_rate: 0.0,
            });
        }

        let mut accepted = Vec::new();

        for i in 0..n {
            let draft_probs = ls_softmax(&draft_logits[i]);
            let target_probs = ls_softmax(&target_logits[i]);
            let token = draft_tokens[i];

            if token >= draft_probs.len() || token >= target_probs.len() {
                break;
            }

            let p_draft = draft_probs[token].max(1e-30);
            let p_target = target_probs[token];

            // Accept if p_target >= p_draft, else accept with prob p_target/p_draft
            let accept = if p_target >= p_draft {
                true
            } else {
                let ratio = p_target / p_draft;
                let u: f64 = self.rng.random_range(0.0..1.0);
                u < ratio
            };

            if accept {
                accepted.push(token);
            } else {
                break; // Stop at first rejection
            }
        }

        let n_accepted = accepted.len();
        let acceptance_rate = if n > 0 {
            n_accepted as f64 / n as f64
        } else {
            0.0
        };

        // Update EMA and adaptive K
        self.ema_acceptance =
            self.ema_alpha * acceptance_rate + (1.0 - self.ema_alpha) * self.ema_acceptance;
        self.total_accepted += n_accepted as u64;
        self.total_proposed += n as u64;

        self.adapt_k();

        Ok(LsSpeculativeResult {
            accepted_tokens: accepted,
            n_accepted,
            acceptance_rate,
        })
    }

    /// Adapt K based on EMA acceptance rate.
    pub(crate) fn adapt_k(&mut self) {
        if self.ema_acceptance > self.increase_threshold && self.k < self.k_max {
            self.k += 1;
        } else if self.ema_acceptance < self.decrease_threshold && self.k > self.k_min {
            self.k -= 1;
        }
    }

    /// Build a token tree from draft model candidates (multiple candidates per position).
    pub fn build_token_tree(
        &self,
        candidates_per_pos: &[Vec<(usize, f64)>],
    ) -> Result<Vec<LsTokenTreeNode>> {
        if candidates_per_pos.is_empty() {
            return Ok(Vec::new());
        }
        let roots: Vec<LsTokenTreeNode> = candidates_per_pos[0]
            .iter()
            .map(|&(tok, lp)| {
                let mut node = LsTokenTreeNode::new(tok, lp);
                Self::build_subtree(&mut node, candidates_per_pos, 1);
                node
            })
            .collect();
        Ok(roots)
    }

    fn build_subtree(node: &mut LsTokenTreeNode, candidates: &[Vec<(usize, f64)>], depth: usize) {
        if depth >= candidates.len() {
            return;
        }
        for &(tok, lp) in &candidates[depth] {
            let mut child = LsTokenTreeNode::new(tok, lp);
            Self::build_subtree(&mut child, candidates, depth + 1);
            node.add_child(child);
        }
    }

    /// Verify a token tree against target probabilities. Returns the best accepted path.
    pub fn verify_tree(
        &mut self,
        roots: &[LsTokenTreeNode],
        target_logits: &[Vec<f64>],
    ) -> Result<Vec<usize>> {
        let mut best_path: Vec<usize> = Vec::new();
        let mut best_score = f64::NEG_INFINITY;

        for root in roots {
            let mut path = vec![root.token_id];
            let score = self.score_tree_path(root, target_logits, 0, &mut path);
            if score > best_score {
                best_score = score;
                best_path = path;
            }
        }
        Ok(best_path)
    }

    fn score_tree_path(
        &self,
        node: &LsTokenTreeNode,
        target_logits: &[Vec<f64>],
        depth: usize,
        path: &mut Vec<usize>,
    ) -> f64 {
        let mut score = if depth < target_logits.len() {
            let probs = ls_softmax(&target_logits[depth]);
            if node.token_id < probs.len() {
                probs[node.token_id].ln().max(-30.0)
            } else {
                -30.0
            }
        } else {
            0.0
        };

        if !node.children.is_empty() && depth + 1 < target_logits.len() {
            let mut best_child_score = f64::NEG_INFINITY;
            let mut best_child_tok = None;
            for child in &node.children {
                let mut child_path = path.clone();
                child_path.push(child.token_id);
                let cs = self.score_tree_path(child, target_logits, depth + 1, &mut child_path);
                if cs > best_child_score {
                    best_child_score = cs;
                    best_child_tok = Some(child.token_id);
                }
            }
            if let Some(tok) = best_child_tok {
                path.push(tok);
                score += best_child_score;
            }
        }
        score
    }

    /// Overall acceptance rate.
    pub fn overall_acceptance_rate(&self) -> f64 {
        if self.total_proposed == 0 {
            0.0
        } else {
            self.total_accepted as f64 / self.total_proposed as f64
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  InstructionTuner — instruction fine-tuning framework
// ─────────────────────────────────────────────────────────────────────────────

/// A single instruction example with system, user, and assistant fields.
#[derive(Debug, Clone)]
pub struct LsInstructionExample {
    pub system: String,
    pub user: String,
    pub assistant: String,
    /// Optional task label for multi-task mixing.
    pub task: Option<String>,
}

/// Formatted and tokenized instruction for training.
#[derive(Debug, Clone)]
pub struct LsFormattedInstruction {
    /// Full token sequence.
    pub token_ids: Vec<usize>,
    /// Loss mask: true = compute loss, false = ignore. Only assistant tokens have true.
    pub loss_mask: Vec<bool>,
    /// Segment boundaries.
    pub system_end: usize,
    pub user_end: usize,
    pub assistant_start: usize,
}

/// Instruction dataset with template formatting and multi-task mixing.
#[derive(Debug)]
pub struct LsInstructionDataset {
    pub examples: Vec<LsInstructionExample>,
    /// Simple char-level tokenizer vocabulary.
    pub vocab: HashMap<char, usize>,
    pub vocab_size: usize,
    /// Context length for packing.
    pub context_length: usize,
    /// Special token IDs.
    pub system_token: usize,
    pub user_token: usize,
    pub assistant_token: usize,
    pub eos_token: usize,
    pub pad_token: usize,
}

impl LsInstructionDataset {
    pub fn new(context_length: usize) -> Self {
        // Build a basic ASCII vocab + special tokens
        let mut vocab = HashMap::new();
        let mut idx = 0usize;
        for c in ' '..='~' {
            vocab.insert(c, idx);
            idx += 1;
        }
        let system_token = idx;
        let user_token = idx + 1;
        let assistant_token = idx + 2;
        let eos_token = idx + 3;
        let pad_token = idx + 4;
        let vocab_size = idx + 5;

        Self {
            examples: Vec::new(),
            vocab,
            vocab_size,
            context_length,
            system_token,
            user_token,
            assistant_token,
            eos_token,
            pad_token,
        }
    }

    /// Add an instruction example.
    pub fn add_example(&mut self, example: LsInstructionExample) {
        self.examples.push(example);
    }

    /// Tokenize a string using the char-level vocab.
    fn tokenize(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| self.vocab.get(&c).copied().unwrap_or(self.pad_token))
            .collect()
    }

    /// Format and tokenize an example with loss masking.
    pub fn format_example(&self, idx: usize) -> Result<LsFormattedInstruction> {
        let ex = self.examples.get(idx).ok_or_else(|| {
            TensorError::compute_error_simple(format!("Example index {} out of range", idx))
        })?;

        let mut tokens = Vec::new();
        let mut loss_mask = Vec::new();

        // <|system|> ... tokens
        tokens.push(self.system_token);
        loss_mask.push(false);
        let sys_toks = self.tokenize(&ex.system);
        for &t in &sys_toks {
            tokens.push(t);
            loss_mask.push(false);
        }
        let system_end = tokens.len();

        // <|user|> ... tokens
        tokens.push(self.user_token);
        loss_mask.push(false);
        let user_toks = self.tokenize(&ex.user);
        for &t in &user_toks {
            tokens.push(t);
            loss_mask.push(false);
        }
        let user_end = tokens.len();

        // <|assistant|> ... tokens (only these get loss)
        tokens.push(self.assistant_token);
        loss_mask.push(false);
        let assistant_start = tokens.len();
        let asst_toks = self.tokenize(&ex.assistant);
        for &t in &asst_toks {
            tokens.push(t);
            loss_mask.push(true);
        }

        // EOS
        tokens.push(self.eos_token);
        loss_mask.push(true);

        // Truncate to context_length
        if tokens.len() > self.context_length {
            tokens.truncate(self.context_length);
            loss_mask.truncate(self.context_length);
        }

        Ok(LsFormattedInstruction {
            token_ids: tokens,
            loss_mask,
            system_end,
            user_end,
            assistant_start,
        })
    }

    /// Pack multiple examples into a single context-length sequence.
    /// Returns packed token_ids and loss_mask.
    pub fn pack_examples(&self, indices: &[usize]) -> Result<(Vec<usize>, Vec<bool>)> {
        let mut packed_tokens = Vec::with_capacity(self.context_length);
        let mut packed_mask = Vec::with_capacity(self.context_length);

        for &idx in indices {
            let formatted = self.format_example(idx)?;
            let remaining = self.context_length.saturating_sub(packed_tokens.len());
            if remaining == 0 {
                break;
            }
            let take = formatted.token_ids.len().min(remaining);
            packed_tokens.extend_from_slice(&formatted.token_ids[..take]);
            packed_mask.extend_from_slice(&formatted.loss_mask[..take]);
        }

        // Pad to context_length
        while packed_tokens.len() < self.context_length {
            packed_tokens.push(self.pad_token);
            packed_mask.push(false);
        }

        Ok((packed_tokens, packed_mask))
    }

    /// Sample a batch with task-proportional mixing.
    pub fn sample_batch(&self, batch_size: usize, rng: &mut StdRng) -> Result<Vec<usize>> {
        if self.examples.is_empty() {
            return Err(TensorError::compute_error_simple(
                "No examples in dataset".to_string(),
            ));
        }

        // Group by task
        let mut task_indices: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, ex) in self.examples.iter().enumerate() {
            let task = ex.task.clone().unwrap_or_else(|| "default".to_string());
            task_indices.entry(task).or_default().push(i);
        }

        let num_tasks = task_indices.len();
        let per_task = (batch_size + num_tasks - 1) / num_tasks;

        let mut batch = Vec::with_capacity(batch_size);
        for indices in task_indices.values() {
            for _ in 0..per_task.min(batch_size.saturating_sub(batch.len())) {
                let idx = rng.random_range(0..indices.len());
                batch.push(indices[idx]);
            }
        }
        batch.truncate(batch_size);
        Ok(batch)
    }
}

/// Instruction tuner with training loop.
#[derive(Debug)]
pub struct LsInstructionTuner {
    pub dataset: LsInstructionDataset,
    pub learning_rate: f64,
    pub num_epochs: usize,
    pub batch_size: usize,
    /// Training loss history.
    pub loss_history: Vec<f64>,
}

impl LsInstructionTuner {
    pub fn new(
        dataset: LsInstructionDataset,
        learning_rate: f64,
        num_epochs: usize,
        batch_size: usize,
    ) -> Self {
        Self {
            dataset,
            learning_rate,
            num_epochs,
            batch_size,
            loss_history: Vec::new(),
        }
    }

    /// Compute masked cross-entropy loss on a formatted example.
    /// `logits[t]` is the logit distribution at time step t.
    pub fn compute_masked_loss(
        &self,
        logits: &[Vec<f64>],
        token_ids: &[usize],
        loss_mask: &[bool],
    ) -> Result<f64> {
        let n = logits.len().min(token_ids.len()).min(loss_mask.len());
        if n == 0 {
            return Ok(0.0);
        }

        let mut total_loss = 0.0;
        let mut count = 0usize;

        for i in 0..n.saturating_sub(1) {
            if !loss_mask[i + 1] {
                continue;
            }
            let log_probs = ls_log_softmax(&logits[i]);
            let target = token_ids[i + 1];
            if target < log_probs.len() {
                total_loss -= log_probs[target];
                count += 1;
            }
        }

        if count == 0 {
            Ok(0.0)
        } else {
            Ok(total_loss / count as f64)
        }
    }
}
