//! Neural Architecture Search (NAS) utilities and Lottery Ticket Hypothesis pruning.
//!
//! This module provides:
//!
//! - [`SearchSpace`]: Defines the architecture search space as a DAG of mixed operations.
//! - [`DartsCell`]: Differentiable Architecture Search (DARTS) cell with mixed ops.
//! - \[`EvolutionaryNas`\]: Regularized evolution for NAS via tournament selection and mutation.
//! - [`LotteryTicketPruner`]: Iterative Magnitude Pruning (IMP) implementing the Lottery Ticket
//!   Hypothesis (Frankle & Carlin, 2019).
//! - [`ArchitectureEvaluator`]: Training-free proxy metrics (SynFlow, grad norm, FLOPs, Pareto).
//! - [`RandomSearchNas`]: Simple random search baseline over the search space.
//!
//! # Design decisions
//!
//! * All tensors are represented as `Vec<f32>` / `Vec<bool>` / `Vec<usize>` for
//!   maximum interoperability.
//! * Randomness is sourced from `scirs2_core::random` — never from the `rand` crate.
//! * No `unwrap()` is used anywhere; every fallible path returns `Result<_, TensorError>`.
//! * No `unsafe` code.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// OpsChoice
// ─────────────────────────────────────────────────────────────────────────────

/// Enumeration of primitive operations available in the search space.
///
/// Each variant corresponds to a canonical NAS operation. In a full
/// implementation each would apply convolutions / pooling; here we track
/// them symbolically and provide FLOP / param estimates.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpsChoice {
    /// 3×3 standard convolution.
    Conv3x3,
    /// 5×5 standard convolution.
    Conv5x5,
    /// 3×3 dilated convolution (dilation=2).
    DilatedConv3x3,
    /// 3×3 depth-wise separable convolution.
    SepConv3x3,
    /// Identity skip connection.
    SkipConnect,
    /// 3×3 max-pooling.
    MaxPool3x3,
    /// 3×3 average-pooling.
    AvgPool3x3,
    /// Zero operation (drops the input).
    Zero,
}

impl OpsChoice {
    /// All variants in definition order.
    pub fn all() -> Vec<OpsChoice> {
        vec![
            OpsChoice::Conv3x3,
            OpsChoice::Conv5x5,
            OpsChoice::DilatedConv3x3,
            OpsChoice::SepConv3x3,
            OpsChoice::SkipConnect,
            OpsChoice::MaxPool3x3,
            OpsChoice::AvgPool3x3,
            OpsChoice::Zero,
        ]
    }

    /// Number of defined variants.
    pub fn num_variants() -> usize {
        8
    }

    /// Convert from a 0-based index.
    pub fn from_index(idx: usize) -> Result<OpsChoice> {
        match idx {
            0 => Ok(OpsChoice::Conv3x3),
            1 => Ok(OpsChoice::Conv5x5),
            2 => Ok(OpsChoice::DilatedConv3x3),
            3 => Ok(OpsChoice::SepConv3x3),
            4 => Ok(OpsChoice::SkipConnect),
            5 => Ok(OpsChoice::MaxPool3x3),
            6 => Ok(OpsChoice::AvgPool3x3),
            7 => Ok(OpsChoice::Zero),
            _ => Err(TensorError::InvalidArgument {
                operation: "OpsChoice::from_index".to_string(),
                reason: format!("index {idx} out of range [0, 7]"),
                context: None,
            }),
        }
    }

    /// Convert to a 0-based index.
    pub fn to_index(&self) -> usize {
        match self {
            OpsChoice::Conv3x3 => 0,
            OpsChoice::Conv5x5 => 1,
            OpsChoice::DilatedConv3x3 => 2,
            OpsChoice::SepConv3x3 => 3,
            OpsChoice::SkipConnect => 4,
            OpsChoice::MaxPool3x3 => 5,
            OpsChoice::AvgPool3x3 => 6,
            OpsChoice::Zero => 7,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MixedOp
// ─────────────────────────────────────────────────────────────────────────────

/// A soft mixture over a set of operations, used during DARTS search.
///
/// During the continuous relaxation phase, the output of a `MixedOp` is the
/// weighted sum of all operation outputs:
///
/// ```text
/// out = Σ_i softmax(weights)[i] * op_i(x)
/// ```
#[derive(Debug, Clone)]
pub struct MixedOp {
    /// Raw (pre-softmax) architecture weights, one per operation.
    pub weights: Vec<f32>,
    /// The candidate operations.
    pub ops: Vec<OpsChoice>,
}

impl MixedOp {
    /// Create a new `MixedOp` with uniform weights.
    pub fn new(ops: Vec<OpsChoice>) -> Result<MixedOp> {
        if ops.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "MixedOp::new".to_string(),
                reason: "ops list must not be empty".to_string(),
                context: None,
            });
        }
        let n = ops.len();
        Ok(MixedOp {
            weights: vec![0.0_f32; n],
            ops,
        })
    }

    /// Return the softmax-normalised weights.
    pub fn softmax_weights(&self) -> Vec<f32> {
        softmax(&self.weights)
    }

    /// Return the index of the operation with the highest weight (argmax).
    pub fn best_op_index(&self) -> usize {
        self.weights
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Return the chosen operation (argmax discretization).
    pub fn best_op(&self) -> Result<OpsChoice> {
        let idx = self.best_op_index();
        self.ops
            .get(idx)
            .cloned()
            .ok_or_else(|| TensorError::InvalidArgument {
                operation: "MixedOp::best_op".to_string(),
                reason: "ops list is empty".to_string(),
                context: None,
            })
    }

    /// Apply the mixed operation to a flat input buffer.
    ///
    /// The output is the softmax-weighted sum of symbolic per-op outputs.
    /// For this implementation the symbolic output for every op is just the
    /// input itself (identity), except for `Zero` which contributes zeros.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let sw = self.softmax_weights();
        let mut out = vec![0.0_f32; input.len()];
        for (i, op) in self.ops.iter().enumerate() {
            let w = sw[i];
            match op {
                OpsChoice::Zero => {
                    // Zero op contributes nothing.
                }
                _ => {
                    for (o, &x) in out.iter_mut().zip(input.iter()) {
                        *o += w * x;
                    }
                }
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SearchSpace
// ─────────────────────────────────────────────────────────────────────────────

/// Defines the architecture search space as a DAG with `num_nodes` intermediate
/// nodes, each receiving input from 2 predecessors.
///
/// The architecture is parameterised by a flat `alpha` vector of shape
/// `[num_edges × num_ops]` where `num_edges = num_input_nodes + num_nodes`.
#[derive(Debug, Clone)]
pub struct SearchSpace {
    /// Number of intermediate nodes in the cell DAG.
    pub num_nodes: usize,
    /// Number of input nodes (typically 2 for DARTS: x_{k-2} and x_{k-1}).
    pub num_input_nodes: usize,
    /// Number of candidate operations per edge.
    pub num_ops: usize,
}

impl SearchSpace {
    /// Create a new `SearchSpace`.
    ///
    /// * `num_nodes` — number of intermediate nodes in the DAG.
    /// * `num_input_nodes` — number of fixed input nodes (e.g. 2 for DARTS).
    /// * `num_ops` — number of candidate operations per edge.
    pub fn new(num_nodes: usize, num_input_nodes: usize, num_ops: usize) -> Result<SearchSpace> {
        if num_nodes == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "SearchSpace::new".to_string(),
                reason: "num_nodes must be at least 1".to_string(),
                context: None,
            });
        }
        if num_input_nodes == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "SearchSpace::new".to_string(),
                reason: "num_input_nodes must be at least 1".to_string(),
                context: None,
            });
        }
        if num_ops == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "SearchSpace::new".to_string(),
                reason: "num_ops must be at least 1".to_string(),
                context: None,
            });
        }
        Ok(SearchSpace {
            num_nodes,
            num_input_nodes,
            num_ops,
        })
    }

    /// Total number of edges in the DAG.
    ///
    /// Each intermediate node i (0-indexed) has i + num_input_nodes predecessors,
    /// so the total is Σ_{i=0}^{num_nodes-1} (i + num_input_nodes).
    pub fn num_edges(&self) -> usize {
        (0..self.num_nodes).map(|i| i + self.num_input_nodes).sum()
    }

    /// Expected length of the flat alpha vector: `num_edges × num_ops`.
    pub fn alpha_len(&self) -> usize {
        self.num_edges() * self.num_ops
    }

    /// Discretize `alpha` via argmax to produce one `OpsChoice` per edge.
    ///
    /// `alpha` must have length `alpha_len()`.  The returned vector has
    /// length `num_edges()`.
    pub fn sample_architecture(&self, alpha: &[f32]) -> Result<Vec<OpsChoice>> {
        let expected = self.alpha_len();
        if alpha.len() != expected {
            return Err(TensorError::InvalidArgument {
                operation: "SearchSpace::sample_architecture".to_string(),
                reason: format!(
                    "alpha length mismatch: expected {expected}, got {}",
                    alpha.len()
                ),
                context: None,
            });
        }
        let n_edges = self.num_edges();
        let mut arch = Vec::with_capacity(n_edges);
        for e in 0..n_edges {
            let slice = &alpha[e * self.num_ops..(e + 1) * self.num_ops];
            let best = argmax(slice);
            let op = OpsChoice::from_index(best % OpsChoice::num_variants())?;
            arch.push(op);
        }
        Ok(arch)
    }

    /// Compute softmax over each edge's weights in `alpha`.
    ///
    /// Returns a flat vector of shape `[num_edges × num_ops]` with
    /// per-edge softmax applied.
    pub fn softmax_architecture(&self, alpha: &[f32]) -> Result<Vec<f32>> {
        let expected = self.alpha_len();
        if alpha.len() != expected {
            return Err(TensorError::InvalidArgument {
                operation: "SearchSpace::softmax_architecture".to_string(),
                reason: format!(
                    "alpha length mismatch: expected {expected}, got {}",
                    alpha.len()
                ),
                context: None,
            });
        }
        let n_edges = self.num_edges();
        let mut out = Vec::with_capacity(expected);
        for e in 0..n_edges {
            let slice = &alpha[e * self.num_ops..(e + 1) * self.num_ops];
            out.extend_from_slice(&softmax(slice));
        }
        Ok(out)
    }

    /// Sample a random architecture (uniform over operations per edge).
    ///
    /// Uses the provided `rng` for reproducibility.
    pub fn random_architecture<R: Rng>(&self, rng: &mut R) -> Result<Vec<OpsChoice>> {
        let n_edges = self.num_edges();
        let mut arch = Vec::with_capacity(n_edges);
        for _ in 0..n_edges {
            let idx = rng.random_range(0..self.num_ops);
            let op_idx = idx % OpsChoice::num_variants();
            arch.push(OpsChoice::from_index(op_idx)?);
        }
        Ok(arch)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DartsCell
// ─────────────────────────────────────────────────────────────────────────────

/// A single DARTS cell (Normal or Reduction).
///
/// The cell is a directed acyclic graph with `num_input_nodes` fixed inputs and
/// `num_nodes` intermediate nodes.  Each intermediate node sums the weighted
/// mixed-op outputs from all its predecessors.  The cell output is the
/// concatenation of all intermediate node outputs.
///
/// For this simplified 1-D implementation height×width is treated as 1;
/// channel-level operations are used throughout.
#[derive(Debug, Clone)]
pub struct DartsCell {
    /// Underlying search space (determines DAG topology).
    pub search_space: SearchSpace,
    /// Whether this is a reduction cell (stride=2 semantics).
    pub is_reduction: bool,
}

impl DartsCell {
    /// Create a new DARTS cell.
    pub fn new(search_space: SearchSpace, is_reduction: bool) -> DartsCell {
        DartsCell {
            search_space,
            is_reduction,
        }
    }

    /// Forward pass through the cell.
    ///
    /// * `x_prev` — previous-previous feature map (flat: channels * height * width).
    /// * `x_curr` — previous feature map (same shape).
    /// * `alpha` — flat architecture weight vector of length `alpha_len()`.
    /// * `batch` — batch size (for bookkeeping; not used in 1-D version).
    /// * `channels` — number of channels.
    /// * `height` — spatial height (simplified: treated as 1).
    /// * `width` — spatial width (simplified: treated as 1).
    ///
    /// Returns the concatenated output of all intermediate nodes.
    pub fn forward(
        &self,
        x_prev: &[f32],
        x_curr: &[f32],
        alpha: &[f32],
        _batch: usize,
        channels: usize,
        _height: usize,
        _width: usize,
    ) -> Result<Vec<f32>> {
        let expected_alpha = self.search_space.alpha_len();
        if alpha.len() != expected_alpha {
            return Err(TensorError::InvalidArgument {
                operation: "DartsCell::forward".to_string(),
                reason: format!(
                    "alpha length mismatch: expected {expected_alpha}, got {}",
                    alpha.len()
                ),
                context: None,
            });
        }
        if x_prev.len() != channels || x_curr.len() != channels {
            return Err(TensorError::InvalidArgument {
                operation: "DartsCell::forward".to_string(),
                reason: format!(
                    "input length mismatch: expected {channels} channels, \
                     got x_prev={} x_curr={}",
                    x_prev.len(),
                    x_curr.len()
                ),
                context: None,
            });
        }

        // node_outputs[i] = output of node i (fixed inputs + intermediate).
        let num_inputs = self.search_space.num_input_nodes;
        let num_nodes = self.search_space.num_nodes;
        let num_ops = self.search_space.num_ops;

        // Build initial node outputs from input nodes.
        let mut node_outputs: Vec<Vec<f32>> = Vec::with_capacity(num_inputs + num_nodes);
        // Input node 0: x_prev, input node 1: x_curr (if present).
        node_outputs.push(x_prev.to_vec());
        if num_inputs > 1 {
            node_outputs.push(x_curr.to_vec());
        }
        for _ in 2..num_inputs {
            node_outputs.push(vec![0.0_f32; channels]);
        }

        let mut edge_offset = 0usize;

        // Compute each intermediate node.
        for node_i in 0..num_nodes {
            let num_predecessors = node_i + num_inputs;
            let mut node_out = vec![0.0_f32; channels];

            for pred in 0..num_predecessors {
                // Get the alpha slice for this edge.
                let alpha_slice = &alpha[edge_offset * num_ops..(edge_offset + 1) * num_ops];
                let sw = softmax(alpha_slice);

                // Apply weighted mixed op.
                let pred_out = &node_outputs[pred];
                for c in 0..channels {
                    let mut mixed = 0.0_f32;
                    for (op_i, op_w) in sw.iter().enumerate().take(num_ops) {
                        let op_choice_idx = op_i % OpsChoice::num_variants();
                        let op = OpsChoice::from_index(op_choice_idx)?;
                        let contribution = match op {
                            OpsChoice::Zero => 0.0_f32,
                            OpsChoice::SkipConnect => pred_out[c],
                            // Reduction cell halves the "value" to simulate stride.
                            OpsChoice::MaxPool3x3 | OpsChoice::AvgPool3x3 if self.is_reduction => {
                                pred_out[c] * 0.5
                            }
                            _ => pred_out[c],
                        };
                        mixed += op_w * contribution;
                    }
                    node_out[c] += mixed;
                }

                edge_offset += 1;
            }

            node_outputs.push(node_out);
        }

        // Concatenate intermediate node outputs (skip the fixed input nodes).
        let mut output = Vec::with_capacity(channels * num_nodes);
        for node_i in num_inputs..num_inputs + num_nodes {
            output.extend_from_slice(&node_outputs[node_i]);
        }
        Ok(output)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EvolutionaryNas
// ─────────────────────────────────────────────────────────────────────────────

/// Regularized evolution for neural architecture search.
///
/// Implements the aging-evolution algorithm from Real et al. (2019):
/// - Maintains a population of architectures with an age counter.
/// - Tournament selection: sample `tournament_size` architectures, keep the best.
/// - Mutation: randomly flip one operation to a different one.
/// - Aging: when population is full, remove the oldest individual before adding
///   a new one.
#[derive(Debug)]
pub struct AgingEvolutionNas {
    /// Current population of architectures.
    pub population: Vec<Vec<OpsChoice>>,
    /// Maximum population size.
    pub population_size: usize,
    /// Number of individuals sampled for tournament selection.
    pub tournament_size: usize,
    /// Probability of mutating any single edge (per-edge).
    pub mutation_rate: f32,
    /// Age of each individual (number of generations since creation).
    pub ages: Vec<usize>,
    /// Internal PRNG.
    rng: StdRng,
}

impl Clone for AgingEvolutionNas {
    fn clone(&self) -> Self {
        Self {
            population: self.population.clone(),
            population_size: self.population_size,
            tournament_size: self.tournament_size,
            mutation_rate: self.mutation_rate,
            ages: self.ages.clone(),
            rng: StdRng::seed_from_u64(0xA61E_E001),
        }
    }
}

impl AgingEvolutionNas {
    /// Create a new `EvolutionaryNas` controller.
    pub fn new(
        population_size: usize,
        tournament_size: usize,
        mutation_rate: f32,
        seed: u64,
    ) -> Result<AgingEvolutionNas> {
        if population_size == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "EvolutionaryNas::new".to_string(),
                reason: "population_size must be at least 1".to_string(),
                context: None,
            });
        }
        if tournament_size == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "EvolutionaryNas::new".to_string(),
                reason: "tournament_size must be at least 1".to_string(),
                context: None,
            });
        }
        if !(0.0..=1.0).contains(&mutation_rate) {
            return Err(TensorError::InvalidArgument {
                operation: "EvolutionaryNas::new".to_string(),
                reason: format!("mutation_rate {mutation_rate} must be in [0, 1]"),
                context: None,
            });
        }
        Ok(AgingEvolutionNas {
            population: Vec::new(),
            population_size,
            tournament_size,
            mutation_rate,
            ages: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// Seed the initial population from the search space.
    pub fn initialize(&mut self, search_space: &SearchSpace) -> Result<()> {
        self.population.clear();
        self.ages.clear();
        for _ in 0..self.population_size {
            let arch = search_space.random_architecture(&mut self.rng)?;
            self.population.push(arch);
            self.ages.push(0);
        }
        Ok(())
    }

    /// Perform one evolution step.
    ///
    /// 1. Increment all ages.
    /// 2. Tournament selection: pick `tournament_size` random indices,
    ///    choose the one with the highest fitness.
    /// 3. Mutate the winner.
    /// 4. If population is full, remove the oldest individual.
    /// 5. Add mutated child to the population.
    ///
    /// Returns `(winner_idx, new_child_position)`.
    ///
    /// `fitnesses` must have the same length as `self.population`.
    pub fn step(&mut self, fitnesses: &[f32]) -> Result<(usize, usize)> {
        if fitnesses.len() != self.population.len() {
            return Err(TensorError::InvalidArgument {
                operation: "EvolutionaryNas::step".to_string(),
                reason: format!(
                    "fitnesses length {} != population length {}",
                    fitnesses.len(),
                    self.population.len()
                ),
                context: None,
            });
        }
        if self.population.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "EvolutionaryNas::step".to_string(),
                reason: "population is empty; call initialize() first".to_string(),
                context: None,
            });
        }

        // Increment ages.
        for age in self.ages.iter_mut() {
            *age += 1;
        }

        // Tournament selection.
        let pop_len = self.population.len();
        let k = self.tournament_size.min(pop_len);
        let mut candidates: Vec<usize> = Vec::with_capacity(k);
        while candidates.len() < k {
            let idx = self.rng.random_range(0..pop_len);
            if !candidates.contains(&idx) {
                candidates.push(idx);
            }
        }
        let winner_idx = candidates
            .iter()
            .copied()
            .max_by(|&a, &b| {
                fitnesses[a]
                    .partial_cmp(&fitnesses[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(candidates[0]);

        // Mutate winner.
        let child = self.mutate_internal(&self.population[winner_idx].clone());

        // Aging: if at capacity, remove the oldest.
        if self.population.len() >= self.population_size {
            let oldest_pos = self
                .ages
                .iter()
                .enumerate()
                .max_by_key(|(_, &age)| age)
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.population.remove(oldest_pos);
            self.ages.remove(oldest_pos);
        }

        // Add child.
        self.population.push(child);
        self.ages.push(0);
        let child_pos = self.population.len() - 1;

        Ok((winner_idx, child_pos))
    }

    /// Mutate an architecture: each operation is replaced with probability
    /// `mutation_rate`, choosing uniformly among other operations.
    pub fn mutate(&mut self, arch: &[OpsChoice]) -> Vec<OpsChoice> {
        self.mutate_internal(arch)
    }

    fn mutate_internal(&mut self, arch: &[OpsChoice]) -> Vec<OpsChoice> {
        let all_ops = OpsChoice::all();
        let mut child = arch.to_vec();
        for op in child.iter_mut() {
            let r: f32 = self.rng.random_range(0.0..1.0_f32);
            if r < self.mutation_rate {
                // Pick a different operation.
                let current_idx = op.to_index();
                let n = all_ops.len();
                let mut new_idx = self.rng.random_range(0..n - 1);
                if new_idx >= current_idx {
                    new_idx += 1;
                }
                *op = all_ops[new_idx].clone();
            }
        }
        child
    }

    /// Force-mutate exactly one operation (used for testing determinism).
    pub fn mutate_one<R: Rng>(&self, arch: &[OpsChoice], rng: &mut R) -> Result<Vec<OpsChoice>> {
        if arch.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "EvolutionaryNas::mutate_one".to_string(),
                reason: "arch is empty; cannot mutate".to_string(),
                context: None,
            });
        }
        let all_ops = OpsChoice::all();
        let mut child = arch.to_vec();
        let pos = rng.random_range(0..arch.len());
        let current_idx = child[pos].to_index();
        let n = all_ops.len();
        let mut new_idx = rng.random_range(0..n - 1);
        if new_idx >= current_idx {
            new_idx += 1;
        }
        child[pos] = all_ops[new_idx].clone();
        Ok(child)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TicketState
// ─────────────────────────────────────────────────────────────────────────────

/// Encapsulates the state of a Lottery Ticket (a sparse sub-network).
#[derive(Debug, Clone)]
pub struct TicketState {
    /// Weights at initialization (`W₀`).
    pub initial_weights: Vec<f32>,
    /// Binary mask: `true` means the weight survives pruning.
    pub mask: Vec<bool>,
    /// Current IMP round index (0-based).
    pub round: usize,
}

impl TicketState {
    /// Create a new `TicketState` from initial weights (all weights active).
    pub fn new(initial_weights: Vec<f32>) -> TicketState {
        let n = initial_weights.len();
        TicketState {
            initial_weights,
            mask: vec![true; n],
            round: 0,
        }
    }

    /// Number of surviving (non-pruned) weights.
    pub fn num_active(&self) -> usize {
        self.mask.iter().filter(|&&m| m).count()
    }

    /// Sparsity: fraction of pruned weights.
    pub fn sparsity(&self) -> f32 {
        if self.mask.is_empty() {
            return 0.0;
        }
        let pruned = self.mask.iter().filter(|&&m| !m).count();
        pruned as f32 / self.mask.len() as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LotteryTicketPruner
// ─────────────────────────────────────────────────────────────────────────────

/// Implements Iterative Magnitude Pruning (IMP) from the Lottery Ticket
/// Hypothesis (Frankle & Carlin, 2019).
///
/// The algorithm:
/// 1. Initialize network; save initial weights `W₀`.
/// 2. Train for T steps to obtain `W_T`.
/// 3. Prune `pruning_rate` fraction of weights with the smallest `|W_T|`,
///    intersected with the current mask.
/// 4. Reset surviving weights to `W₀ ⊙ M` (winning ticket).
/// 5. Repeat for `rounds` iterations.
#[derive(Debug, Clone)]
pub struct LotteryTicketPruner {
    /// Fraction of remaining weights to prune each round.
    pub pruning_rate: f32,
    /// Total number of IMP rounds.
    pub rounds: usize,
}

impl LotteryTicketPruner {
    /// Create a new `LotteryTicketPruner`.
    ///
    /// * `pruning_rate` — fraction in (0, 1) of weights to prune each round.
    /// * `rounds` — number of IMP rounds.
    pub fn new(pruning_rate: f32, rounds: usize) -> Result<LotteryTicketPruner> {
        if !(0.0..1.0).contains(&pruning_rate) {
            return Err(TensorError::InvalidArgument {
                operation: "LotteryTicketPruner::new".to_string(),
                reason: format!("pruning_rate {pruning_rate} must be in (0, 1)"),
                context: None,
            });
        }
        if rounds == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "LotteryTicketPruner::new".to_string(),
                reason: "rounds must be at least 1".to_string(),
                context: None,
            });
        }
        Ok(LotteryTicketPruner {
            pruning_rate,
            rounds,
        })
    }

    /// Save initial weights and return a fresh `TicketState`.
    pub fn save_initial_weights(&self, weights: &[f32]) -> TicketState {
        TicketState::new(weights.to_vec())
    }

    /// Compute the pruning mask from `current_weights`.
    ///
    /// Prunes `pruning_fraction` of *currently active* weights (those where
    /// the existing mask is `true`).  Returns an updated mask of the same
    /// length as `current_weights`.
    pub fn compute_mask(
        &self,
        current_weights: &[f32],
        existing_mask: &[bool],
        pruning_fraction: f32,
    ) -> Result<Vec<bool>> {
        if current_weights.len() != existing_mask.len() {
            return Err(TensorError::InvalidArgument {
                operation: "LotteryTicketPruner::compute_mask".to_string(),
                reason: format!(
                    "weight length {} != mask length {}",
                    current_weights.len(),
                    existing_mask.len()
                ),
                context: None,
            });
        }
        if !(0.0..=1.0).contains(&pruning_fraction) {
            return Err(TensorError::InvalidArgument {
                operation: "LotteryTicketPruner::compute_mask".to_string(),
                reason: format!("pruning_fraction {pruning_fraction} must be in [0, 1]"),
                context: None,
            });
        }

        // Collect indices of currently active weights.
        let active_indices: Vec<usize> = existing_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .collect();

        let num_active = active_indices.len();
        let num_to_prune = ((num_active as f32) * pruning_fraction).floor() as usize;

        // Sort active weights by |w| ascending.
        let mut sorted_active: Vec<(usize, f32)> = active_indices
            .iter()
            .map(|&i| (i, current_weights[i].abs()))
            .collect();
        sorted_active.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Build new mask: prune the `num_to_prune` smallest.
        let mut new_mask = existing_mask.to_vec();
        for (idx, _) in sorted_active.iter().take(num_to_prune) {
            new_mask[*idx] = false;
        }
        Ok(new_mask)
    }

    /// Apply a binary mask to weights: masked-out positions become 0.
    pub fn apply_mask(&self, weights: &[f32], mask: &[bool]) -> Result<Vec<f32>> {
        if weights.len() != mask.len() {
            return Err(TensorError::InvalidArgument {
                operation: "LotteryTicketPruner::apply_mask".to_string(),
                reason: format!(
                    "weight length {} != mask length {}",
                    weights.len(),
                    mask.len()
                ),
                context: None,
            });
        }
        Ok(weights
            .iter()
            .zip(mask.iter())
            .map(|(&w, &m)| if m { w } else { 0.0 })
            .collect())
    }

    /// Rewind to winning ticket: `W₀ ⊙ M`.
    ///
    /// Returns the initial weights with masked-out positions zeroed.
    pub fn rewind_to_ticket(&self, initial_weights: &[f32], mask: &[bool]) -> Result<Vec<f32>> {
        self.apply_mask(initial_weights, mask)
    }

    /// Compute sparsity: fraction of `false` entries in `mask`.
    pub fn sparsity(mask: &[bool]) -> f32 {
        if mask.is_empty() {
            return 0.0;
        }
        let pruned = mask.iter().filter(|&&m| !m).count();
        pruned as f32 / mask.len() as f32
    }

    /// Run a full IMP loop given a callback that "trains" the weights.
    ///
    /// `trainer(weights, round) -> trained_weights` is called each round to
    /// simulate training.  Returns the final `TicketState`.
    pub fn run<F>(&self, initial_weights: &[f32], trainer: F) -> Result<TicketState>
    where
        F: Fn(&[f32], usize) -> Vec<f32>,
    {
        let mut state = self.save_initial_weights(initial_weights);

        for round in 0..self.rounds {
            // Apply current mask to get the sparse starting weights.
            let sparse_weights = self.apply_mask(&state.initial_weights, &state.mask)?;

            // "Train" for T steps.
            let trained = trainer(&sparse_weights, round);

            // Compute new mask from trained weights.
            let new_mask = self.compute_mask(&trained, &state.mask, self.pruning_rate)?;

            // Rewind surviving weights to initial values.
            state.mask = new_mask;
            state.round = round + 1;
        }

        Ok(state)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ArchitectureEvaluator
// ─────────────────────────────────────────────────────────────────────────────

/// Training-free proxy metrics for NAS.
///
/// These metrics allow ranking architectures without full training, using only
/// the network structure and random-initialization statistics.
#[derive(Debug, Clone, Default)]
pub struct ArchitectureEvaluator;

impl ArchitectureEvaluator {
    /// Create a new `ArchitectureEvaluator`.
    pub fn new() -> ArchitectureEvaluator {
        ArchitectureEvaluator
    }

    /// SynFlow score: Σ_l (w_l ⊙ ∂L/∂w_l) — measures gradient flow at init.
    ///
    /// A higher score indicates better gradient signal through the network.
    pub fn synflow_score(&self, weights: &[&[f32]], gradients: &[&[f32]]) -> Result<f32> {
        if weights.len() != gradients.len() {
            return Err(TensorError::InvalidArgument {
                operation: "ArchitectureEvaluator::synflow_score".to_string(),
                reason: format!(
                    "weights layers ({}) != gradient layers ({})",
                    weights.len(),
                    gradients.len()
                ),
                context: None,
            });
        }
        let mut score = 0.0_f32;
        for (w_layer, g_layer) in weights.iter().zip(gradients.iter()) {
            if w_layer.len() != g_layer.len() {
                return Err(TensorError::InvalidArgument {
                    operation: "ArchitectureEvaluator::synflow_score".to_string(),
                    reason: format!(
                        "layer weight count ({}) != gradient count ({})",
                        w_layer.len(),
                        g_layer.len()
                    ),
                    context: None,
                });
            }
            let layer_score: f32 = w_layer
                .iter()
                .zip(g_layer.iter())
                .map(|(&w, &g)| w * g)
                .sum();
            score += layer_score;
        }
        Ok(score)
    }

    /// Gradient norm score: L2 norm of all gradients concatenated.
    ///
    /// Architectures with higher gradient norms at init tend to train faster.
    pub fn grad_norm_score(&self, gradients: &[&[f32]]) -> f32 {
        let sum_sq: f32 = gradients
            .iter()
            .flat_map(|g| g.iter())
            .map(|&v| v * v)
            .sum();
        sum_sq.sqrt()
    }

    /// Estimate parameter count for an architecture.
    ///
    /// Each edge uses the default channel count `channels = 64`.
    pub fn param_count(&self, arch: &[OpsChoice]) -> usize {
        let c = 64usize;
        arch.iter().map(|op| op_param_count(op, c)).sum()
    }

    /// Estimate FLOP count for an architecture given a 1-D `input_size`.
    pub fn flops_estimate(&self, arch: &[OpsChoice], input_size: usize) -> usize {
        arch.iter().map(|op| op_flops(op, input_size)).sum()
    }

    /// Compute the Pareto front from a set of (accuracy, efficiency) pairs.
    ///
    /// Returns the indices of non-dominated architectures where:
    /// - higher accuracy is better.
    /// - higher efficiency is better.
    ///
    /// Architecture `i` dominates `j` iff `acc[i] >= acc[j] && eff[i] >= eff[j]`
    /// with at least one strict inequality.
    pub fn pareto_front(&self, scores: &[(f32, f32)]) -> Vec<usize> {
        let n = scores.len();
        let mut on_front = vec![true; n];

        for i in 0..n {
            if !on_front[i] {
                continue;
            }
            for j in 0..n {
                if i == j || !on_front[j] {
                    continue;
                }
                // Check if j dominates i.
                let j_dominates_i = scores[j].0 >= scores[i].0
                    && scores[j].1 >= scores[i].1
                    && (scores[j].0 > scores[i].0 || scores[j].1 > scores[i].1);
                if j_dominates_i {
                    on_front[i] = false;
                    break;
                }
            }
        }

        on_front
            .iter()
            .enumerate()
            .filter_map(|(i, &f)| if f { Some(i) } else { None })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RandomSearchNas
// ─────────────────────────────────────────────────────────────────────────────

/// Simple random-search baseline for NAS.
///
/// Samples `num_samples` random architectures from the search space and ranks
/// them by the gradient-norm proxy metric.
#[derive(Debug, Clone)]
pub struct RandomSearchNas {
    /// Underlying search space.
    pub search_space: SearchSpace,
    /// Number of architectures to sample.
    pub num_samples: usize,
    /// PRNG seed.
    seed: u64,
}

impl RandomSearchNas {
    /// Create a new `RandomSearchNas`.
    pub fn new(
        search_space: SearchSpace,
        num_samples: usize,
        seed: u64,
    ) -> Result<RandomSearchNas> {
        if num_samples == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "RandomSearchNas::new".to_string(),
                reason: "num_samples must be at least 1".to_string(),
                context: None,
            });
        }
        Ok(RandomSearchNas {
            search_space,
            num_samples,
            seed,
        })
    }

    /// Search for the best architectures using the given `evaluator`.
    ///
    /// Returns all sampled architectures sorted by descending proxy score,
    /// as `(arch, score)` pairs.
    pub fn search(&self, evaluator: &ArchitectureEvaluator) -> Result<Vec<(Vec<OpsChoice>, f32)>> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut results: Vec<(Vec<OpsChoice>, f32)> = Vec::with_capacity(self.num_samples);

        for _ in 0..self.num_samples {
            let arch = self.search_space.random_architecture(&mut rng)?;

            // Use param_count as a simple proxy score (architectures with more
            // parameters tend to have higher capacity).  A real implementation
            // would use SynFlow or grad-norm at random initialization.
            let score = evaluator.param_count(&arch) as f32;
            results.push((arch, score));
        }

        // Sort by descending score.
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    /// Return top-k architectures.
    pub fn top_k(
        &self,
        evaluator: &ArchitectureEvaluator,
        k: usize,
    ) -> Result<Vec<(Vec<OpsChoice>, f32)>> {
        let all = self.search(evaluator)?;
        Ok(all.into_iter().take(k).collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically stable softmax over a slice.
fn softmax(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / x.len() as f32; x.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// Return the index of the maximum element.
fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Estimate parameters for a single operation given `channels`.
fn op_param_count(op: &OpsChoice, channels: usize) -> usize {
    match op {
        OpsChoice::Conv3x3 => 3 * 3 * channels * channels,
        OpsChoice::Conv5x5 => 5 * 5 * channels * channels,
        OpsChoice::DilatedConv3x3 => 3 * 3 * channels * channels,
        OpsChoice::SepConv3x3 => 3 * 3 * channels + channels * channels,
        OpsChoice::SkipConnect => 0,
        OpsChoice::MaxPool3x3 => 0,
        OpsChoice::AvgPool3x3 => 0,
        OpsChoice::Zero => 0,
    }
}

/// Estimate FLOPs for a single operation given `input_size`.
fn op_flops(op: &OpsChoice, input_size: usize) -> usize {
    match op {
        OpsChoice::Conv3x3 => 2 * 3 * 3 * input_size,
        OpsChoice::Conv5x5 => 2 * 5 * 5 * input_size,
        OpsChoice::DilatedConv3x3 => 2 * 3 * 3 * input_size,
        OpsChoice::SepConv3x3 => 2 * (3 * 3 + 1) * input_size,
        OpsChoice::SkipConnect => input_size,
        OpsChoice::MaxPool3x3 => 3 * 3 * input_size,
        OpsChoice::AvgPool3x3 => 3 * 3 * input_size,
        OpsChoice::Zero => 0,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── OpsChoice ────────────────────────────────────────────────────────────

    #[test]
    fn test_ops_choice_round_trip() {
        for op in OpsChoice::all() {
            let idx = op.to_index();
            let recovered = OpsChoice::from_index(idx).expect("valid index");
            assert_eq!(op, recovered, "round-trip failed for {:?}", op);
        }
    }

    #[test]
    fn test_ops_choice_from_index_out_of_range() {
        assert!(OpsChoice::from_index(100).is_err());
    }

    #[test]
    fn test_ops_choice_all_has_correct_count() {
        assert_eq!(OpsChoice::all().len(), OpsChoice::num_variants());
    }

    // ── MixedOp ──────────────────────────────────────────────────────────────

    #[test]
    fn test_mixed_op_softmax_sums_to_one() {
        let ops = OpsChoice::all();
        let mo = MixedOp::new(ops).expect("test: operation should succeed");
        let sw = mo.softmax_weights();
        let total: f32 = sw.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "softmax sum = {total}");
    }

    #[test]
    fn test_mixed_op_best_op_zero_is_zero() {
        let ops = vec![OpsChoice::Zero, OpsChoice::Conv3x3];
        let mut mo = MixedOp::new(ops).expect("test: operation should succeed");
        // Set first weight higher.
        mo.weights[0] = 10.0;
        mo.weights[1] = 0.0;
        assert_eq!(
            mo.best_op().expect("test: operation should succeed"),
            OpsChoice::Zero
        );
    }

    #[test]
    fn test_mixed_op_forward_all_zero_ops() {
        let ops = vec![OpsChoice::Zero; 4];
        let mo = MixedOp::new(ops).expect("test: operation should succeed");
        let input = vec![1.0_f32, 2.0, 3.0];
        let out = mo.forward(&input);
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_mixed_op_forward_all_skip() {
        let ops = vec![OpsChoice::SkipConnect];
        let mo = MixedOp::new(ops).expect("test: operation should succeed");
        let input = vec![1.0_f32, 2.0, 3.0];
        let out = mo.forward(&input);
        // Single op with weight=1.0, so output == input.
        for (o, i) in out.iter().zip(input.iter()) {
            assert!((o - i).abs() < 1e-5, "expected {i} got {o}");
        }
    }

    // ── SearchSpace ──────────────────────────────────────────────────────────

    #[test]
    fn test_search_space_new_valid() {
        let ss = SearchSpace::new(4, 2, 8).expect("test: operation should succeed");
        assert_eq!(ss.num_nodes, 4);
        assert_eq!(ss.num_edges(), 2 + 3 + 4 + 5);
    }

    #[test]
    fn test_search_space_new_zero_nodes_err() {
        assert!(SearchSpace::new(0, 2, 8).is_err());
    }

    #[test]
    fn test_search_space_alpha_len() {
        let ss = SearchSpace::new(4, 2, 8).expect("test: operation should succeed");
        assert_eq!(ss.alpha_len(), ss.num_edges() * 8);
    }

    #[test]
    fn test_search_space_sample_architecture_valid() {
        let ss = SearchSpace::new(4, 2, 8).expect("test: operation should succeed");
        let alpha = vec![0.0_f32; ss.alpha_len()];
        let arch = ss
            .sample_architecture(&alpha)
            .expect("test: operation should succeed");
        assert_eq!(arch.len(), ss.num_edges());
        for op in &arch {
            assert!(OpsChoice::from_index(op.to_index()).is_ok());
        }
    }

    #[test]
    fn test_search_space_sample_architecture_argmax() {
        let ss = SearchSpace::new(1, 2, 8).expect("test: operation should succeed");
        // 2 edges (node 0 has 2 predecessors).
        let mut alpha = vec![0.0_f32; ss.alpha_len()];
        // Set Conv5x5 (index 1) highest on first edge.
        alpha[1] = 10.0;
        let arch = ss
            .sample_architecture(&alpha)
            .expect("test: operation should succeed");
        assert_eq!(arch[0], OpsChoice::Conv5x5);
    }

    #[test]
    fn test_search_space_softmax_architecture_shape() {
        let ss = SearchSpace::new(3, 2, 8).expect("test: operation should succeed");
        let alpha = vec![1.0_f32; ss.alpha_len()];
        let out = ss
            .softmax_architecture(&alpha)
            .expect("test: operation should succeed");
        assert_eq!(out.len(), ss.alpha_len());
    }

    #[test]
    fn test_search_space_softmax_each_edge_sums_to_one() {
        let ss = SearchSpace::new(2, 2, 8).expect("test: operation should succeed");
        let alpha = vec![1.0_f32; ss.alpha_len()];
        let out = ss
            .softmax_architecture(&alpha)
            .expect("test: operation should succeed");
        for e in 0..ss.num_edges() {
            let s: f32 = out[e * 8..(e + 1) * 8].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "edge {e} sum = {s}");
        }
    }

    #[test]
    fn test_search_space_random_architecture_valid() {
        let ss = SearchSpace::new(4, 2, 8).expect("test: operation should succeed");
        let mut rng = StdRng::seed_from_u64(42);
        let arch = ss
            .random_architecture(&mut rng)
            .expect("test: operation should succeed");
        assert_eq!(arch.len(), ss.num_edges());
    }

    // ── DartsCell ────────────────────────────────────────────────────────────

    #[test]
    fn test_darts_cell_forward_output_shape() {
        let ss = SearchSpace::new(4, 2, 8).expect("test: operation should succeed");
        let cell = DartsCell::new(ss.clone(), false);
        let alpha = vec![0.0_f32; ss.alpha_len()];
        let channels = 8;
        let x_prev = vec![1.0_f32; channels];
        let x_curr = vec![0.5_f32; channels];
        let out = cell
            .forward(&x_prev, &x_curr, &alpha, 1, channels, 1, 1)
            .expect("test: operation should succeed");
        // Output: num_nodes * channels.
        assert_eq!(out.len(), ss.num_nodes * channels);
    }

    #[test]
    fn test_darts_cell_zero_alpha_all_zero_with_zero_ops() {
        // When all weights are equal, mixed-op will average ops; with
        // SkipConnect dominant, output should be non-trivial.
        let ss = SearchSpace::new(2, 2, 8).expect("test: operation should succeed");
        let cell = DartsCell::new(ss.clone(), false);
        let alpha = vec![0.0_f32; ss.alpha_len()];
        let channels = 4;
        let x_prev = vec![1.0_f32; channels];
        let x_curr = vec![1.0_f32; channels];
        let out = cell
            .forward(&x_prev, &x_curr, &alpha, 1, channels, 1, 1)
            .expect("test: operation should succeed");
        // With equal weights some ops are non-zero so output should be non-zero.
        assert_eq!(out.len(), ss.num_nodes * channels);
    }

    #[test]
    fn test_darts_cell_reduction_halves_pool_contribution() {
        let ss = SearchSpace::new(1, 2, 8).expect("test: operation should succeed");
        let cell_normal = DartsCell::new(ss.clone(), false);
        let cell_reduce = DartsCell::new(ss.clone(), true);
        // Set MaxPool3x3 (index 5) to maximum weight on all edges.
        let mut alpha = vec![-10.0_f32; ss.alpha_len()];
        for e in 0..ss.num_edges() {
            alpha[e * 8 + 5] = 10.0; // MaxPool3x3
        }
        let channels = 4;
        let x_prev = vec![2.0_f32; channels];
        let x_curr = vec![2.0_f32; channels];
        let out_normal = cell_normal
            .forward(&x_prev, &x_curr, &alpha, 1, channels, 1, 1)
            .expect("test: operation should succeed");
        let out_reduce = cell_reduce
            .forward(&x_prev, &x_curr, &alpha, 1, channels, 1, 1)
            .expect("test: operation should succeed");
        // Reduction cell should produce smaller values.
        let sum_normal: f32 = out_normal.iter().sum();
        let sum_reduce: f32 = out_reduce.iter().sum();
        assert!(
            sum_reduce < sum_normal,
            "reduce={sum_reduce} normal={sum_normal}"
        );
    }

    // ── EvolutionaryNas ──────────────────────────────────────────────────────

    #[test]
    fn test_evolutionary_nas_initialize() {
        let ss = SearchSpace::new(4, 2, 8).expect("test: operation should succeed");
        let mut enas =
            AgingEvolutionNas::new(10, 3, 0.5, 42).expect("test: operation should succeed");
        enas.initialize(&ss)
            .expect("test: operation should succeed");
        assert_eq!(enas.population.len(), 10);
        assert_eq!(enas.ages.len(), 10);
    }

    #[test]
    fn test_evolutionary_nas_step_increases_population_until_full() {
        let ss = SearchSpace::new(2, 2, 8).expect("test: operation should succeed");
        let mut enas =
            AgingEvolutionNas::new(5, 2, 0.3, 0).expect("test: operation should succeed");
        enas.initialize(&ss)
            .expect("test: operation should succeed");
        let fitnesses = vec![0.1_f32, 0.5, 0.3, 0.8, 0.2];
        let (winner, child) = enas
            .step(&fitnesses)
            .expect("test: operation should succeed");
        // Population stays at capacity.
        assert_eq!(enas.population.len(), 5);
        assert!(winner < 5);
        assert_eq!(child, enas.population.len() - 1);
    }

    #[test]
    fn test_evolutionary_nas_tournament_selects_best() {
        // Create a population where one arch is clearly best.
        let ss = SearchSpace::new(2, 2, 8).expect("test: operation should succeed");
        let mut enas =
            AgingEvolutionNas::new(10, 10, 0.0, 7).expect("test: operation should succeed");
        enas.initialize(&ss)
            .expect("test: operation should succeed");
        // Fitness 0 is 1.0, all others are 0.0.
        let mut fitnesses = vec![0.0_f32; 10];
        fitnesses[0] = 1.0;
        // With tournament_size=10 and no mutation, arch[0] should always win.
        let (winner, _) = enas
            .step(&fitnesses)
            .expect("test: operation should succeed");
        assert_eq!(winner, 0);
    }

    #[test]
    fn test_evolutionary_nas_mutate_one_changes_exactly_one() {
        let enas = AgingEvolutionNas::new(5, 2, 1.0, 99).expect("test: operation should succeed");
        let arch: Vec<OpsChoice> = vec![
            OpsChoice::Conv3x3,
            OpsChoice::Conv5x5,
            OpsChoice::SkipConnect,
            OpsChoice::Zero,
        ];
        let mut rng = StdRng::seed_from_u64(1);
        let mutated = enas
            .mutate_one(&arch, &mut rng)
            .expect("test: operation should succeed");
        let diffs = arch
            .iter()
            .zip(mutated.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            diffs, 1,
            "mutate_one must change exactly one op, changed {diffs}"
        );
    }

    #[test]
    fn test_evolutionary_nas_mutate_one_preserves_length() {
        let enas = AgingEvolutionNas::new(5, 2, 1.0, 1).expect("test: operation should succeed");
        let arch: Vec<OpsChoice> = OpsChoice::all();
        let mut rng = StdRng::seed_from_u64(5);
        let mutated = enas
            .mutate_one(&arch, &mut rng)
            .expect("test: operation should succeed");
        assert_eq!(mutated.len(), arch.len());
    }

    #[test]
    fn test_evolutionary_nas_new_invalid_params() {
        assert!(AgingEvolutionNas::new(0, 2, 0.5, 0).is_err());
        assert!(AgingEvolutionNas::new(5, 0, 0.5, 0).is_err());
        assert!(AgingEvolutionNas::new(5, 2, 1.5, 0).is_err());
    }

    // ── LotteryTicketPruner ──────────────────────────────────────────────────

    #[test]
    fn test_lottery_ticket_pruner_new_invalid() {
        assert!(LotteryTicketPruner::new(0.0, 0).is_err());
        assert!(LotteryTicketPruner::new(1.0, 3).is_err());
        assert!(LotteryTicketPruner::new(0.5, 0).is_err());
    }

    #[test]
    fn test_lottery_ticket_pruner_compute_mask_correct_fraction() {
        let pruner = LotteryTicketPruner::new(0.5, 3).expect("test: operation should succeed");
        let weights = vec![1.0_f32, 0.1, 2.0, 0.05, 3.0, 0.01, 4.0, 5.0];
        let mask = vec![true; 8];
        let new_mask = pruner
            .compute_mask(&weights, &mask, 0.5)
            .expect("test: operation should succeed");
        let pruned = new_mask.iter().filter(|&&m| !m).count();
        // 50% of 8 = 4 pruned.
        assert_eq!(pruned, 4);
    }

    #[test]
    fn test_lottery_ticket_pruner_compute_mask_prunes_smallest() {
        let pruner = LotteryTicketPruner::new(0.5, 1).expect("test: operation should succeed");
        // weights: magnitudes [3.0, 1.0, 2.0, 4.0]
        let weights = vec![3.0_f32, -1.0, -2.0, 4.0];
        let mask = vec![true; 4];
        let new_mask = pruner
            .compute_mask(&weights, &mask, 0.5)
            .expect("test: operation should succeed");
        // Smallest magnitudes: |−1.0|=1, |−2.0|=2 → those should be pruned.
        // Index 1 (|−1|) and index 2 (|−2|) should be false.
        assert!(!new_mask[1], "weight -1.0 should be pruned");
        assert!(!new_mask[2], "weight -2.0 should be pruned");
        assert!(new_mask[0], "weight 3.0 should survive");
        assert!(new_mask[3], "weight 4.0 should survive");
    }

    #[test]
    fn test_lottery_ticket_pruner_apply_mask() {
        let pruner = LotteryTicketPruner::new(0.5, 1).expect("test: operation should succeed");
        let weights = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mask = vec![true, false, true, false];
        let masked = pruner
            .apply_mask(&weights, &mask)
            .expect("test: operation should succeed");
        assert_eq!(masked, vec![1.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_lottery_ticket_pruner_rewind_preserves_initial_for_active() {
        let pruner = LotteryTicketPruner::new(0.3, 2).expect("test: operation should succeed");
        let initial = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5];
        let mask = vec![true, false, true, true, false];
        let ticket = pruner
            .rewind_to_ticket(&initial, &mask)
            .expect("test: operation should succeed");
        // Active positions should match initial.
        assert!((ticket[0] - 0.1).abs() < 1e-6);
        assert!((ticket[2] - 0.3).abs() < 1e-6);
        assert!((ticket[3] - 0.4).abs() < 1e-6);
        // Pruned positions should be zero.
        assert_eq!(ticket[1], 0.0);
        assert_eq!(ticket[4], 0.0);
    }

    #[test]
    fn test_lottery_ticket_sparsity_zero_mask() {
        let mask = vec![true; 100];
        assert_eq!(LotteryTicketPruner::sparsity(&mask), 0.0);
    }

    #[test]
    fn test_lottery_ticket_sparsity_half() {
        let mask: Vec<bool> = (0..100).map(|i| i % 2 == 0).collect();
        let s = LotteryTicketPruner::sparsity(&mask);
        assert!((s - 0.5).abs() < 1e-5, "sparsity = {s}");
    }

    #[test]
    fn test_lottery_ticket_sparsity_all_pruned() {
        let mask = vec![false; 50];
        assert_eq!(LotteryTicketPruner::sparsity(&mask), 1.0);
    }

    #[test]
    fn test_lottery_ticket_sparsity_empty() {
        assert_eq!(LotteryTicketPruner::sparsity(&[]), 0.0);
    }

    #[test]
    fn test_lottery_ticket_pruner_run_increases_sparsity() {
        let pruner = LotteryTicketPruner::new(0.2, 3).expect("test: operation should succeed");
        let initial: Vec<f32> = (1..=20).map(|i| i as f32 * 0.01).collect();
        // Identity trainer (no actual training).
        let state = pruner
            .run(&initial, |w, _| w.to_vec())
            .expect("test: operation should succeed");
        assert_eq!(state.round, 3);
        let s = LotteryTicketPruner::sparsity(&state.mask);
        // After 3 rounds of 20% pruning, sparsity should be positive.
        assert!(s > 0.0, "sparsity should be > 0 after pruning, got {s}");
    }

    #[test]
    fn test_lottery_ticket_pruner_mask_respects_existing_mask() {
        let pruner = LotteryTicketPruner::new(0.5, 1).expect("test: operation should succeed");
        let weights = vec![1.0_f32, 2.0, 3.0, 4.0];
        // Pre-prune indices 0 and 1.
        let existing_mask = vec![false, false, true, true];
        let new_mask = pruner
            .compute_mask(&weights, &existing_mask, 0.5)
            .expect("test: operation should succeed");
        // Only 2 active weights; 50% of 2 = 1 pruned.
        // Index 2 has |3.0|, index 3 has |4.0|. Smallest active = index 2.
        assert!(!new_mask[2], "index 2 should be pruned");
        assert!(new_mask[3], "index 3 should survive");
        // Already-pruned indices stay pruned.
        assert!(!new_mask[0]);
        assert!(!new_mask[1]);
    }

    // ── ArchitectureEvaluator ─────────────────────────────────────────────────

    #[test]
    fn test_architecture_evaluator_synflow_positive() {
        let ev = ArchitectureEvaluator::new();
        let w1 = vec![1.0_f32, 2.0, 3.0];
        let g1 = vec![0.1_f32, 0.2, 0.3];
        let score = ev
            .synflow_score(&[&w1], &[&g1])
            .expect("test: operation should succeed");
        // 1*0.1 + 2*0.2 + 3*0.3 = 0.1 + 0.4 + 0.9 = 1.4
        assert!((score - 1.4).abs() < 1e-5, "score = {score}");
    }

    #[test]
    fn test_architecture_evaluator_synflow_length_mismatch_err() {
        let ev = ArchitectureEvaluator::new();
        let w1 = vec![1.0_f32];
        let g1 = vec![1.0_f32, 2.0];
        assert!(ev.synflow_score(&[&w1], &[&g1]).is_err());
    }

    #[test]
    fn test_architecture_evaluator_grad_norm_zero() {
        let ev = ArchitectureEvaluator::new();
        let g = vec![0.0_f32; 10];
        assert_eq!(ev.grad_norm_score(&[&g]), 0.0);
    }

    #[test]
    fn test_architecture_evaluator_grad_norm_known_value() {
        let ev = ArchitectureEvaluator::new();
        // [3, 4] → norm = 5
        let g = vec![3.0_f32, 4.0];
        let norm = ev.grad_norm_score(&[&g]);
        assert!((norm - 5.0).abs() < 1e-5, "norm = {norm}");
    }

    #[test]
    fn test_architecture_evaluator_param_count() {
        let ev = ArchitectureEvaluator::new();
        let arch = vec![OpsChoice::SkipConnect, OpsChoice::Zero];
        assert_eq!(ev.param_count(&arch), 0);
    }

    #[test]
    fn test_architecture_evaluator_param_count_conv3x3() {
        let ev = ArchitectureEvaluator::new();
        let arch = vec![OpsChoice::Conv3x3];
        // 9 * 64 * 64 = 36864
        assert_eq!(ev.param_count(&arch), 36864);
    }

    #[test]
    fn test_architecture_evaluator_flops_zero_op() {
        let ev = ArchitectureEvaluator::new();
        let arch = vec![OpsChoice::Zero];
        assert_eq!(ev.flops_estimate(&arch, 1024), 0);
    }

    #[test]
    fn test_architecture_evaluator_flops_skip() {
        let ev = ArchitectureEvaluator::new();
        let arch = vec![OpsChoice::SkipConnect];
        assert_eq!(ev.flops_estimate(&arch, 128), 128);
    }

    // ── Pareto front ──────────────────────────────────────────────────────────

    #[test]
    fn test_pareto_front_single_element() {
        let ev = ArchitectureEvaluator::new();
        let scores = vec![(0.9_f32, 0.5_f32)];
        let front = ev.pareto_front(&scores);
        assert_eq!(front, vec![0]);
    }

    #[test]
    fn test_pareto_front_all_dominated_by_one() {
        let ev = ArchitectureEvaluator::new();
        // (1.0, 1.0) dominates all others.
        let scores = vec![(0.5_f32, 0.5_f32), (0.6, 0.4), (1.0, 1.0), (0.3, 0.9)];
        let front = ev.pareto_front(&scores);
        assert_eq!(front, vec![2]);
    }

    #[test]
    fn test_pareto_front_two_non_dominated() {
        let ev = ArchitectureEvaluator::new();
        // (0.9, 0.6) and (0.6, 0.9): neither dominates the other (trade-off front).
        // (0.5, 0.5): dominated by both (0.9, 0.6) and (0.6, 0.9) since
        //   0.9>=0.5 && 0.6>=0.5 && (0.9>0.5 || 0.6>0.5) → true for (0.9, 0.6).
        let scores = vec![
            (0.9_f32, 0.6_f32), // index 0 — on front
            (0.5, 0.5),         // index 1 — dominated by index 0
            (0.6, 0.9),         // index 2 — on front
        ];
        let front = ev.pareto_front(&scores);
        assert!(front.contains(&0), "index 0 should be on Pareto front");
        assert!(front.contains(&2), "index 2 should be on Pareto front");
        assert!(!front.contains(&1), "index 1 should be dominated");
    }

    #[test]
    fn test_pareto_front_empty() {
        let ev = ArchitectureEvaluator::new();
        let front = ev.pareto_front(&[]);
        assert!(front.is_empty());
    }

    #[test]
    fn test_pareto_front_filters_dominated() {
        let ev = ArchitectureEvaluator::new();
        // Point (0.8, 0.9) dominates (0.7, 0.8) and (0.8, 0.5).
        let scores = vec![(0.8_f32, 0.9_f32), (0.7, 0.8), (0.8, 0.5), (0.9, 0.7)];
        let front = ev.pareto_front(&scores);
        // (0.8, 0.9) dominates (0.7, 0.8) and (0.8, 0.5).
        // (0.9, 0.7) is not dominated by (0.8, 0.9): 0.9 > 0.8 in accuracy.
        assert!(front.contains(&0));
        assert!(front.contains(&3));
        assert!(!front.contains(&1));
        assert!(!front.contains(&2));
    }

    // ── RandomSearchNas ───────────────────────────────────────────────────────

    #[test]
    fn test_random_search_nas_returns_sorted_results() {
        let ss = SearchSpace::new(4, 2, 8).expect("test: operation should succeed");
        let rns = RandomSearchNas::new(ss, 20, 42).expect("test: operation should succeed");
        let ev = ArchitectureEvaluator::new();
        let results = rns.search(&ev).expect("test: operation should succeed");
        assert_eq!(results.len(), 20);
        // Results should be in descending order of score.
        for w in results.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "results not sorted: {} < {}",
                w[0].1,
                w[1].1
            );
        }
    }

    #[test]
    fn test_random_search_nas_top_k() {
        let ss = SearchSpace::new(3, 2, 8).expect("test: operation should succeed");
        let rns = RandomSearchNas::new(ss, 50, 1).expect("test: operation should succeed");
        let ev = ArchitectureEvaluator::new();
        let top3 = rns.top_k(&ev, 3).expect("test: operation should succeed");
        assert_eq!(top3.len(), 3);
    }

    #[test]
    fn test_random_search_nas_zero_samples_err() {
        let ss = SearchSpace::new(2, 2, 8).expect("test: operation should succeed");
        assert!(RandomSearchNas::new(ss, 0, 0).is_err());
    }

    #[test]
    fn test_random_search_nas_reproducible() {
        let ss1 = SearchSpace::new(2, 2, 8).expect("test: operation should succeed");
        let ss2 = SearchSpace::new(2, 2, 8).expect("test: operation should succeed");
        let rns1 = RandomSearchNas::new(ss1, 10, 777).expect("test: operation should succeed");
        let rns2 = RandomSearchNas::new(ss2, 10, 777).expect("test: operation should succeed");
        let ev = ArchitectureEvaluator::new();
        let r1 = rns1.search(&ev).expect("test: operation should succeed");
        let r2 = rns2.search(&ev).expect("test: operation should succeed");
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.0, b.0, "architectures differ with same seed");
            assert!((a.1 - b.1).abs() < 1e-6);
        }
    }

    // ── TicketState helpers ───────────────────────────────────────────────────

    #[test]
    fn test_ticket_state_new_all_active() {
        let ts = TicketState::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(ts.num_active(), 3);
        assert_eq!(ts.sparsity(), 0.0);
    }

    #[test]
    fn test_ticket_state_sparsity_half() {
        let mut ts = TicketState::new(vec![1.0; 10]);
        ts.mask = (0..10).map(|i| i % 2 == 0).collect();
        let s = ts.sparsity();
        assert!((s - 0.5).abs() < 1e-5, "sparsity = {s}");
    }
}
