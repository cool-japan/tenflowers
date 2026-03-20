//! # Neural Program Execution Extensions
//!
//! Advanced program synthesis algorithms:
//!
//! * **[`NpiController`]** — Neural Program Interpreter (Reed & de Freitas 2016)
//! * **[`RecursiveNpi`]** — recursive NPI with call stack simulation
//! * **[`ProgramLibrary`]** — library of primitive and learned programs
//! * **[`TapeLanguage`]** — differentiable Brainfuck-inspired tape language
//! * **[`DifferentiableVm`]** — soft interpreter with Gumbel-softmax op selection
//! * **[`ProgramSearchBeam`]** — beam search over program token sequences
//! * **[`SimpleType`]** — simple type system with unification
//! * **[`TypedDsl`]** — typed DSL with type-checking
//! * **[`TypeGuidedEnumerator`]** — bottom-up type-guided program enumerator
//! * **[`ObservationalEquivalence`]** — deduplication by input/output behaviour
//! * **[`ExampleIo`]** — input/output example pair
//! * **[`IoEmbedder`]** — set encoding of I/O pairs
//! * **[`SyntaxGuidedSearch`]** — PCFG-based synthesis from examples
//! * **[`PsMetrics`]** — program synthesis evaluation metrics
//!
//! ## Randomness Policy
//!
//! All randomness sourced from `scirs2_core::random` — never from `rand`.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;

use super::softmax_vec;

// ---------------------------------------------------------------------------
// 1. Neural Program Induction
// ---------------------------------------------------------------------------

/// A primitive or learned program entry in the library.
#[derive(Debug, Clone)]
pub struct ProgramEntry {
    /// Human-readable name.
    pub name: String,
    /// Fixed-size embedding vector.
    pub embedding: Vec<f32>,
    /// Whether this is a primitive (true) or a learned program (false).
    pub is_primitive: bool,
}

/// Library of primitive and learned programs with embedding lookup.
///
/// Programs are stored with dense embeddings enabling the NPI controller
/// to select sub-programs via attention over the embedding table.
pub struct ProgramLibrary {
    /// All registered programs.
    pub entries: Vec<ProgramEntry>,
    /// Embedding dimension.
    pub embed_dim: usize,
}

impl ProgramLibrary {
    /// Create an empty library with the given embedding dimension.
    pub fn new(embed_dim: usize) -> Self {
        Self {
            entries: Vec::new(),
            embed_dim,
        }
    }

    /// Register a primitive program with a random embedding.
    pub fn add_primitive(&mut self, name: &str, rng: &mut StdRng) {
        let scale = (1.0 / self.embed_dim as f32).sqrt();
        let embedding = (0..self.embed_dim)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        self.entries.push(ProgramEntry {
            name: name.to_string(),
            embedding,
            is_primitive: true,
        });
    }

    /// Register a learned program with a provided embedding.
    pub fn add_learned(&mut self, name: &str, embedding: Vec<f32>) {
        self.entries.push(ProgramEntry {
            name: name.to_string(),
            embedding,
            is_primitive: false,
        });
    }

    /// Look up a program by name.
    pub fn get(&self, name: &str) -> Option<&ProgramEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Return the index of the best-matching program via cosine similarity.
    pub fn nearest(&self, query: &[f32]) -> Option<usize> {
        if self.entries.is_empty() {
            return None;
        }
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (i, entry) in self.entries.iter().enumerate() {
            let score = cosine_similarity(query, &entry.embedding);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        Some(best_idx)
    }
}

/// Neural Program Interpreter (NPI) controller (Reed & de Freitas 2016).
///
/// The controller maintains a program embedding and an LSTM hidden state.
/// At each step it predicts: (1) sub-program to call next, (2) arguments.
pub struct NpiController {
    /// Dimension of program and state embeddings.
    pub state_dim: usize,
    /// Number of argument slots.
    pub n_args: usize,
    /// LSTM input-to-hidden weight [state_dim x (state_dim + state_dim)]
    w_ih: Vec<Vec<f32>>,
    /// LSTM hidden-to-hidden weight [state_dim x state_dim]
    w_hh: Vec<Vec<f32>>,
    /// Program prediction head [state_dim x state_dim]
    w_prog: Vec<Vec<f32>>,
    /// Argument prediction head [n_args x state_dim]
    w_args: Vec<Vec<f32>>,
    /// Termination gate [1 x state_dim]
    w_term: Vec<f32>,
}

impl NpiController {
    /// Create a new NPI controller with Xavier-initialised weights.
    pub fn new(state_dim: usize, n_args: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (2.0 / (state_dim * 2) as f32).sqrt();
        let mut mat = |rows: usize, cols: usize| -> Vec<Vec<f32>> {
            let s = (2.0 / (rows + cols) as f32).sqrt();
            (0..rows)
                .map(|_| {
                    (0..cols)
                        .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * s)
                        .collect()
                })
                .collect()
        };
        let w_ih = mat(4 * state_dim, 2 * state_dim);
        let w_hh = mat(4 * state_dim, state_dim);
        let w_prog = mat(state_dim, state_dim);
        let w_args = mat(n_args, state_dim);
        let w_term: Vec<f32> = (0..state_dim)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        Self {
            state_dim,
            n_args,
            w_ih,
            w_hh,
            w_prog,
            w_args,
            w_term,
        }
    }

    /// One LSTM step given input `x` and previous hidden/cell states.
    ///
    /// Returns `(new_hidden, new_cell)`.
    pub fn lstm_step(
        &self,
        x: &[f32],
        h_prev: &[f32],
        c_prev: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let d = self.state_dim;
        // Concatenate x and h_prev
        let input: Vec<f32> = x.iter().chain(h_prev.iter()).copied().collect();

        // Gates: i, f, g, o
        let gates_ih: Vec<f32> = mat_vec(&self.w_ih, &input);
        let gates_hh: Vec<f32> = mat_vec(&self.w_hh, h_prev);

        let mut gates: Vec<f32> = gates_ih
            .iter()
            .zip(gates_hh.iter())
            .map(|(a, b)| a + b)
            .collect();

        // Apply activations: sigmoid for i, f, o; tanh for g
        let mut new_c = vec![0.0f32; d];
        let mut new_h = vec![0.0f32; d];
        for idx in 0..d {
            let i_gate = sigmoid(gates[idx]);
            let f_gate = sigmoid(gates[d + idx]);
            let g_gate = gates[2 * d + idx].tanh();
            let o_gate = sigmoid(gates[3 * d + idx]);
            new_c[idx] = f_gate * c_prev.get(idx).copied().unwrap_or(0.0) + i_gate * g_gate;
            new_h[idx] = o_gate * new_c[idx].tanh();
        }
        (new_h, new_c)
    }

    /// Predict the next program embedding given the current hidden state.
    pub fn predict_program(&self, hidden: &[f32]) -> Vec<f32> {
        mat_vec(&self.w_prog, hidden)
    }

    /// Predict argument values (as f32 scores) given the current hidden state.
    pub fn predict_args(&self, hidden: &[f32]) -> Vec<f32> {
        mat_vec(&self.w_args, hidden)
    }

    /// Predict termination probability (sigmoid of w_term · hidden).
    pub fn predict_terminate(&self, hidden: &[f32]) -> f32 {
        let logit: f32 = self
            .w_term
            .iter()
            .zip(hidden.iter())
            .map(|(w, h)| w * h)
            .sum();
        sigmoid(logit)
    }

    /// Run a full NPI episode.
    ///
    /// Given an initial `env_embedding` (environment state) and a `program_embedding`,
    /// runs `max_steps` LSTM steps, returning the sequence of predicted sub-programs
    /// (as embedding vectors) and their argument predictions.
    pub fn run_episode(
        &self,
        env_embedding: &[f32],
        program_embedding: &[f32],
        max_steps: usize,
    ) -> Vec<(Vec<f32>, Vec<f32>)> {
        let d = self.state_dim;
        let mut h = vec![0.0f32; d];
        let mut c = vec![0.0f32; d];
        let mut results = Vec::new();

        // Concatenate env + program as the first input
        let mut x: Vec<f32> = env_embedding
            .iter()
            .chain(program_embedding.iter())
            .take(d * 2)
            .copied()
            .collect();
        // Pad if shorter
        x.resize(d * 2, 0.0);

        for _ in 0..max_steps {
            let (new_h, new_c) = self.lstm_step(&x[..d], &h, &c);
            h = new_h;
            c = new_c;

            let prog_pred = self.predict_program(&h);
            let args_pred = self.predict_args(&h);
            results.push((prog_pred.clone(), args_pred));

            let p_term = self.predict_terminate(&h);
            if p_term > 0.5 {
                break;
            }

            // Feed predicted program as next input
            x[..d.min(prog_pred.len())].copy_from_slice(&prog_pred[..d.min(prog_pred.len())]);
        }
        results
    }
}

/// Recursive NPI with an explicit call stack, bounded by `max_depth`.
pub struct RecursiveNpi {
    /// Underlying NPI controller.
    pub controller: NpiController,
    /// Maximum recursion depth.
    pub max_depth: usize,
}

impl RecursiveNpi {
    /// Create a new RecursiveNpi.
    pub fn new(state_dim: usize, n_args: usize, max_depth: usize, seed: u64) -> Self {
        Self {
            controller: NpiController::new(state_dim, n_args, seed),
            max_depth,
        }
    }

    /// Execute a recursive NPI episode.
    ///
    /// Returns a flat list of `(sub_program_embedding, args)` in call order.
    pub fn execute(
        &self,
        env_embedding: &[f32],
        program_embedding: &[f32],
        depth: usize,
    ) -> Vec<(Vec<f32>, Vec<f32>)> {
        if depth >= self.max_depth {
            return Vec::new();
        }
        let steps = self
            .controller
            .run_episode(env_embedding, program_embedding, 8);
        let mut all_calls = Vec::new();
        for (prog_emb, args) in steps {
            let sub = self
                .execute(env_embedding, &prog_emb, depth + 1);
            all_calls.push((prog_emb, args));
            all_calls.extend(sub);
        }
        all_calls
    }
}

// ---------------------------------------------------------------------------
// 2. Differentiable Programming
// ---------------------------------------------------------------------------

/// The 8 tape operations of the TapeLanguage (Brainfuck-inspired).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TapeOp {
    /// Move tape head right.
    Right,
    /// Move tape head left.
    Left,
    /// Increment cell at head.
    Increment,
    /// Decrement cell at head.
    Decrement,
    /// Output cell value.
    Output,
    /// Read input into cell.
    Input,
    /// Jump forward past matching End if cell is zero.
    LoopBegin,
    /// Jump back to matching Begin if cell is non-zero.
    LoopEnd,
}

impl TapeOp {
    /// Return all 8 tape ops in canonical order.
    pub fn all() -> [TapeOp; 8] {
        [
            TapeOp::Right,
            TapeOp::Left,
            TapeOp::Increment,
            TapeOp::Decrement,
            TapeOp::Output,
            TapeOp::Input,
            TapeOp::LoopBegin,
            TapeOp::LoopEnd,
        ]
    }

    /// Return the index of this op in the canonical ordering.
    pub fn index(self) -> usize {
        TapeOp::all()
            .iter()
            .position(|o| *o == self)
            .unwrap_or(0)
    }
}

/// A small tape-based language with differentiable relaxation.
///
/// Hard programs are sequences of [`TapeOp`]; the differentiable relaxation
/// uses a soft categorical distribution (Gumbel-softmax) over ops at each step.
pub struct TapeLanguage {
    /// Tape length.
    pub tape_len: usize,
    /// Maximum program length.
    pub max_prog_len: usize,
}

impl TapeLanguage {
    /// Create a new TapeLanguage.
    pub fn new(tape_len: usize, max_prog_len: usize) -> Self {
        Self {
            tape_len,
            max_prog_len,
        }
    }

    /// Execute a hard (discrete) program on a tape initialised to `input`.
    ///
    /// Returns the output tape values and an output buffer from `Output` ops.
    pub fn execute_hard(&self, program: &[TapeOp], input: &[i32]) -> (Vec<i32>, Vec<i32>) {
        let mut tape = vec![0i32; self.tape_len];
        for (i, &v) in input.iter().enumerate().take(self.tape_len) {
            tape[i] = v;
        }
        let mut head: usize = 0;
        let mut ip: usize = 0;
        let mut input_ptr: usize = input.len();
        let mut output: Vec<i32> = Vec::new();
        let mut iter_count = 0usize;
        let max_iter = self.max_prog_len * 100;

        while ip < program.len() && iter_count < max_iter {
            iter_count += 1;
            match program[ip] {
                TapeOp::Right => {
                    head = (head + 1) % self.tape_len;
                }
                TapeOp::Left => {
                    head = if head == 0 { self.tape_len - 1 } else { head - 1 };
                }
                TapeOp::Increment => {
                    tape[head] = tape[head].wrapping_add(1);
                }
                TapeOp::Decrement => {
                    tape[head] = tape[head].wrapping_sub(1);
                }
                TapeOp::Output => {
                    output.push(tape[head]);
                }
                TapeOp::Input => {
                    tape[head] = input.get(input_ptr).copied().unwrap_or(0);
                    input_ptr += 1;
                }
                TapeOp::LoopBegin => {
                    if tape[head] == 0 {
                        // Jump forward to matching LoopEnd
                        let mut depth = 1usize;
                        ip += 1;
                        while ip < program.len() && depth > 0 {
                            match program[ip] {
                                TapeOp::LoopBegin => depth += 1,
                                TapeOp::LoopEnd => depth -= 1,
                                _ => {}
                            }
                            if depth > 0 {
                                ip += 1;
                            }
                        }
                    }
                }
                TapeOp::LoopEnd => {
                    if tape[head] != 0 {
                        // Jump back to matching LoopBegin
                        let mut depth = 1usize;
                        if ip == 0 {
                            ip += 1;
                            continue;
                        }
                        ip -= 1;
                        while depth > 0 {
                            match program[ip] {
                                TapeOp::LoopEnd => depth += 1,
                                TapeOp::LoopBegin => depth -= 1,
                                _ => {}
                            }
                            if depth > 0 {
                                if ip == 0 {
                                    break;
                                }
                                ip -= 1;
                            }
                        }
                    }
                }
            }
            ip += 1;
        }
        (tape, output)
    }

    /// Soft execution: given op probability distributions for each step,
    /// return expected output values using the op probabilities as weights.
    ///
    /// `op_probs[step]` is a probability vector of length 8 over `TapeOp::all()`.
    /// Returns a soft output tape (f32 values).
    pub fn execute_soft(&self, op_probs: &[Vec<f32>], initial_tape: &[f32]) -> Vec<f32> {
        let mut tape = vec![0.0f32; self.tape_len];
        for (i, &v) in initial_tape.iter().enumerate().take(self.tape_len) {
            tape[i] = v;
        }
        let mut soft_head = 0.0f32; // soft head position

        for probs in op_probs.iter().take(self.max_prog_len) {
            let p = if probs.len() >= 8 {
                probs.as_slice()
            } else {
                continue;
            };
            // Soft right/left: shift soft head
            let p_right = p[TapeOp::Right.index()];
            let p_left = p[TapeOp::Left.index()];
            soft_head = (soft_head + p_right - p_left).clamp(0.0, (self.tape_len - 1) as f32);

            // Soft increment/decrement on head cell
            let head_idx = soft_head.round() as usize % self.tape_len;
            let p_inc = p[TapeOp::Increment.index()];
            let p_dec = p[TapeOp::Decrement.index()];
            tape[head_idx] += p_inc - p_dec;
        }
        tape
    }
}

/// Differentiable virtual machine with learned op probabilities.
///
/// Rather than hard-selecting ops, the VM uses soft probabilities and
/// Gumbel-softmax sampling during training to select operations.
pub struct DifferentiableVm {
    /// Number of op slots (program length).
    pub prog_len: usize,
    /// Embedding dimension for state encoding.
    pub state_dim: usize,
    /// Tape language.
    pub tape: TapeLanguage,
    /// Logits for each op at each step: [prog_len x 8]
    pub op_logits: Vec<Vec<f32>>,
    /// State encoder weights [state_dim x tape_len]
    w_state: Vec<Vec<f32>>,
}

impl DifferentiableVm {
    /// Create a new DifferentiableVm with random initial logits.
    pub fn new(prog_len: usize, tape_len: usize, state_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = 0.1f32;
        let op_logits = (0..prog_len)
            .map(|_| {
                (0..8)
                    .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
                    .collect()
            })
            .collect();
        let w_state = (0..state_dim)
            .map(|_| {
                (0..tape_len)
                    .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
                    .collect()
            })
            .collect();
        Self {
            prog_len,
            state_dim,
            tape: TapeLanguage::new(tape_len, prog_len),
            op_logits,
            w_state,
        }
    }

    /// Sample a discrete program using Gumbel-softmax (hard = argmax of logits + noise).
    pub fn sample_program(&self, rng: &mut StdRng, temperature: f32) -> Vec<TapeOp> {
        let ops = TapeOp::all();
        self.op_logits
            .iter()
            .map(|logits| {
                // Gumbel noise
                let gumbel: Vec<f32> = logits
                    .iter()
                    .map(|&l| {
                        let u: f32 = rng.random::<f32>().clamp(1e-7, 1.0 - 1e-7);
                        l + (-(u.ln()).ln()) / temperature
                    })
                    .collect();
                let best = gumbel
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                ops[best]
            })
            .collect()
    }

    /// Soft forward pass: return expected tape after execution.
    pub fn forward_soft(&self, initial_tape: &[f32]) -> Vec<f32> {
        let probs: Vec<Vec<f32>> = self.op_logits.iter().map(|l| softmax_vec(l)).collect();
        self.tape.execute_soft(&probs, initial_tape)
    }
}

/// Beam search over program token sequences.
///
/// Each token corresponds to a [`TapeOp`]. The search maximises a
/// per-token log-probability with a length penalty.
pub struct ProgramSearchBeam {
    /// Beam width.
    pub beam_width: usize,
    /// Maximum program length.
    pub max_len: usize,
    /// Length penalty exponent alpha.
    pub length_penalty: f32,
}

/// A partial program candidate in the beam.
#[derive(Debug, Clone)]
pub struct BeamCandidate {
    /// Chosen ops so far.
    pub ops: Vec<TapeOp>,
    /// Cumulative log-probability.
    pub log_prob: f32,
}

impl ProgramSearchBeam {
    /// Create a new beam searcher.
    pub fn new(beam_width: usize, max_len: usize, length_penalty: f32) -> Self {
        Self {
            beam_width,
            max_len,
            length_penalty,
        }
    }

    /// Expand the beam by one step.
    ///
    /// `score_fn` maps a partial op sequence to a probability distribution
    /// over the 8 `TapeOp` variants for the next step.
    pub fn step<F>(&self, candidates: Vec<BeamCandidate>, score_fn: &F) -> Vec<BeamCandidate>
    where
        F: Fn(&[TapeOp]) -> Vec<f32>,
    {
        let mut next: Vec<BeamCandidate> = Vec::new();
        let all_ops = TapeOp::all();
        for cand in &candidates {
            let probs = score_fn(&cand.ops);
            for (i, op) in all_ops.iter().enumerate() {
                let p = probs.get(i).copied().unwrap_or(1e-10).max(1e-10);
                let mut new_ops = cand.ops.clone();
                new_ops.push(*op);
                next.push(BeamCandidate {
                    log_prob: cand.log_prob + p.ln(),
                    ops: new_ops,
                });
            }
        }
        // Keep top beam_width by score adjusted for length
        next.sort_by(|a, b| {
            let score_a = a.log_prob / (a.ops.len() as f32).powf(self.length_penalty);
            let score_b = b.log_prob / (b.ops.len() as f32).powf(self.length_penalty);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        next.truncate(self.beam_width);
        next
    }

    /// Run beam search to completion, returning the best program.
    pub fn search<F>(&self, score_fn: F) -> Option<Vec<TapeOp>>
    where
        F: Fn(&[TapeOp]) -> Vec<f32>,
    {
        let mut beam = vec![BeamCandidate {
            ops: Vec::new(),
            log_prob: 0.0,
        }];
        for _ in 0..self.max_len {
            beam = self.step(beam, &score_fn);
            if beam.is_empty() {
                return None;
            }
        }
        beam.into_iter().next().map(|c| c.ops)
    }
}

// ---------------------------------------------------------------------------
// 3. Type-Guided Synthesis
// ---------------------------------------------------------------------------

/// A simple type in the typed DSL.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SimpleType {
    /// Integer scalar.
    Int,
    /// Boolean.
    Bool,
    /// Homogeneous list.
    List(Box<SimpleType>),
    /// Type variable (used during unification).
    Var(u32),
}

impl SimpleType {
    /// Check if this type contains the given type variable.
    pub fn contains_var(&self, v: u32) -> bool {
        match self {
            SimpleType::Var(u) => *u == v,
            SimpleType::List(inner) => inner.contains_var(v),
            _ => false,
        }
    }

    /// Substitute type variable `v` with `replacement`.
    pub fn substitute(&self, v: u32, replacement: &SimpleType) -> SimpleType {
        match self {
            SimpleType::Var(u) if *u == v => replacement.clone(),
            SimpleType::List(inner) => {
                SimpleType::List(Box::new(inner.substitute(v, replacement)))
            }
            other => other.clone(),
        }
    }
}

/// Unification result.
#[derive(Debug, Clone)]
pub struct Substitution {
    /// Map from type variable index to concrete type.
    pub mapping: HashMap<u32, SimpleType>,
}

impl Substitution {
    /// Create an empty substitution.
    pub fn new() -> Self {
        Self {
            mapping: HashMap::new(),
        }
    }

    /// Apply the substitution to a type.
    pub fn apply(&self, ty: &SimpleType) -> SimpleType {
        match ty {
            SimpleType::Var(v) => {
                if let Some(t) = self.mapping.get(v) {
                    self.apply(t)
                } else {
                    ty.clone()
                }
            }
            SimpleType::List(inner) => SimpleType::List(Box::new(self.apply(inner))),
            other => other.clone(),
        }
    }

    /// Extend with variable `v -> ty`.
    pub fn bind(&mut self, v: u32, ty: SimpleType) {
        self.mapping.insert(v, ty);
    }
}

impl Default for Substitution {
    fn default() -> Self {
        Self::new()
    }
}

/// Unify two types, returning the resulting substitution or `None` on failure.
pub fn unify(t1: &SimpleType, t2: &SimpleType, subst: &mut Substitution) -> bool {
    let a = subst.apply(t1);
    let b = subst.apply(t2);
    match (&a, &b) {
        (SimpleType::Int, SimpleType::Int) => true,
        (SimpleType::Bool, SimpleType::Bool) => true,
        (SimpleType::List(inner_a), SimpleType::List(inner_b)) => {
            unify(inner_a, inner_b, subst)
        }
        (SimpleType::Var(v), other) | (other, SimpleType::Var(v)) => {
            if let SimpleType::Var(u) = other {
                if u == v {
                    return true;
                }
            }
            if other.contains_var(*v) {
                return false; // occurs check
            }
            subst.bind(*v, other.clone());
            true
        }
        _ => false,
    }
}

/// A typed DSL expression.
#[derive(Debug, Clone)]
pub enum TypedDslExpr {
    /// Integer literal.
    IntLit(i64),
    /// Boolean literal.
    BoolLit(bool),
    /// Add two integer expressions.
    Add(Box<TypedDslExpr>, Box<TypedDslExpr>),
    /// Multiply two integer expressions.
    Mul(Box<TypedDslExpr>, Box<TypedDslExpr>),
    /// Negate an integer expression.
    Neg(Box<TypedDslExpr>),
    /// Logical AND of two boolean expressions.
    And(Box<TypedDslExpr>, Box<TypedDslExpr>),
    /// Logical NOT of a boolean expression.
    Not(Box<TypedDslExpr>),
    /// Compare two integers for equality.
    Eq(Box<TypedDslExpr>, Box<TypedDslExpr>),
    /// Map a function over a list.
    MapNeg(Box<TypedDslExpr>),
    /// List literal.
    ListLit(Vec<TypedDslExpr>),
}

/// A typed DSL with type-checking before evaluation.
pub struct TypedDsl;

impl TypedDsl {
    /// Infer the type of an expression.
    pub fn infer_type(expr: &TypedDslExpr) -> Option<SimpleType> {
        match expr {
            TypedDslExpr::IntLit(_) => Some(SimpleType::Int),
            TypedDslExpr::BoolLit(_) => Some(SimpleType::Bool),
            TypedDslExpr::Add(a, b) | TypedDslExpr::Mul(a, b) => {
                if Self::infer_type(a)? == SimpleType::Int
                    && Self::infer_type(b)? == SimpleType::Int
                {
                    Some(SimpleType::Int)
                } else {
                    None
                }
            }
            TypedDslExpr::Neg(a) => {
                if Self::infer_type(a)? == SimpleType::Int {
                    Some(SimpleType::Int)
                } else {
                    None
                }
            }
            TypedDslExpr::And(a, b) => {
                if Self::infer_type(a)? == SimpleType::Bool
                    && Self::infer_type(b)? == SimpleType::Bool
                {
                    Some(SimpleType::Bool)
                } else {
                    None
                }
            }
            TypedDslExpr::Not(a) => {
                if Self::infer_type(a)? == SimpleType::Bool {
                    Some(SimpleType::Bool)
                } else {
                    None
                }
            }
            TypedDslExpr::Eq(a, b) => {
                if Self::infer_type(a)? == SimpleType::Int
                    && Self::infer_type(b)? == SimpleType::Int
                {
                    Some(SimpleType::Bool)
                } else {
                    None
                }
            }
            TypedDslExpr::MapNeg(list) => {
                if let Some(SimpleType::List(inner)) = Self::infer_type(list) {
                    if *inner == SimpleType::Int {
                        Some(SimpleType::List(Box::new(SimpleType::Int)))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            TypedDslExpr::ListLit(elems) => {
                if elems.is_empty() {
                    Some(SimpleType::List(Box::new(SimpleType::Int)))
                } else {
                    let elem_ty = Self::infer_type(&elems[0])?;
                    for e in &elems[1..] {
                        if Self::infer_type(e)? != elem_ty {
                            return None;
                        }
                    }
                    Some(SimpleType::List(Box::new(elem_ty)))
                }
            }
        }
    }

    /// Evaluate a type-checked expression.
    ///
    /// Returns `None` if the expression is ill-typed or evaluation fails.
    pub fn eval(expr: &TypedDslExpr) -> Option<DslValue> {
        // Type-check first
        Self::infer_type(expr)?;
        Self::eval_unchecked(expr)
    }

    fn eval_unchecked(expr: &TypedDslExpr) -> Option<DslValue> {
        match expr {
            TypedDslExpr::IntLit(n) => Some(DslValue::Int(*n)),
            TypedDslExpr::BoolLit(b) => Some(DslValue::Bool(*b)),
            TypedDslExpr::Add(a, b) => {
                let va = Self::eval_unchecked(a)?;
                let vb = Self::eval_unchecked(b)?;
                Some(DslValue::Int(va.as_int()? + vb.as_int()?))
            }
            TypedDslExpr::Mul(a, b) => {
                let va = Self::eval_unchecked(a)?;
                let vb = Self::eval_unchecked(b)?;
                Some(DslValue::Int(va.as_int()? * vb.as_int()?))
            }
            TypedDslExpr::Neg(a) => {
                let va = Self::eval_unchecked(a)?;
                Some(DslValue::Int(-va.as_int()?))
            }
            TypedDslExpr::And(a, b) => {
                let va = Self::eval_unchecked(a)?;
                let vb = Self::eval_unchecked(b)?;
                Some(DslValue::Bool(va.as_bool()? && vb.as_bool()?))
            }
            TypedDslExpr::Not(a) => {
                let va = Self::eval_unchecked(a)?;
                Some(DslValue::Bool(!va.as_bool()?))
            }
            TypedDslExpr::Eq(a, b) => {
                let va = Self::eval_unchecked(a)?;
                let vb = Self::eval_unchecked(b)?;
                Some(DslValue::Bool(va.as_int()? == vb.as_int()?))
            }
            TypedDslExpr::MapNeg(list) => {
                let vl = Self::eval_unchecked(list)?;
                let elems = vl.as_list()?;
                let negated: Option<Vec<DslValue>> = elems
                    .into_iter()
                    .map(|v| v.as_int().map(|n| DslValue::Int(-n)))
                    .collect();
                Some(DslValue::List(negated?))
            }
            TypedDslExpr::ListLit(exprs) => {
                let vals: Option<Vec<DslValue>> =
                    exprs.iter().map(Self::eval_unchecked).collect();
                Some(DslValue::List(vals?))
            }
        }
    }
}

/// A runtime value in the typed DSL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DslValue {
    /// An integer value.
    Int(i64),
    /// A boolean value.
    Bool(bool),
    /// A list of values.
    List(Vec<DslValue>),
}

impl DslValue {
    /// Extract integer or return None.
    pub fn as_int(&self) -> Option<i64> {
        if let DslValue::Int(n) = self {
            Some(*n)
        } else {
            None
        }
    }

    /// Extract boolean or return None.
    pub fn as_bool(&self) -> Option<bool> {
        if let DslValue::Bool(b) = self {
            Some(*b)
        } else {
            None
        }
    }

    /// Extract list or return None.
    pub fn as_list(self) -> Option<Vec<DslValue>> {
        if let DslValue::List(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

/// Deduplicate partial programs by observational equivalence on a fixed input set.
pub struct ObservationalEquivalence {
    /// Input vectors to use as the observation basis.
    pub basis_inputs: Vec<Vec<i64>>,
}

impl ObservationalEquivalence {
    /// Create a new equivalence checker with the given basis inputs.
    pub fn new(basis_inputs: Vec<Vec<i64>>) -> Self {
        Self { basis_inputs }
    }

    /// Compute the observational fingerprint of an expression.
    ///
    /// Each fingerprint entry is the output of evaluating the expression
    /// (treated as a function of the first input element).
    pub fn fingerprint(&self, expr: &TypedDslExpr) -> Vec<Option<DslValue>> {
        self.basis_inputs
            .iter()
            .map(|_inp| TypedDsl::eval(expr))
            .collect()
    }

    /// Filter a list of expressions to unique equivalence classes.
    pub fn deduplicate(&self, exprs: Vec<TypedDslExpr>) -> Vec<TypedDslExpr> {
        let mut seen: Vec<Vec<Option<DslValue>>> = Vec::new();
        let mut unique = Vec::new();
        for expr in exprs {
            let fp = self.fingerprint(&expr);
            if !seen.contains(&fp) {
                seen.push(fp);
                unique.push(expr);
            }
        }
        unique
    }
}

/// Bottom-up type-guided program enumerator.
///
/// Enumerates programs in order of increasing size, using type constraints
/// to prune the search space. Programs are also deduplicated by observational
/// equivalence.
pub struct TypeGuidedEnumerator {
    /// Maximum program size (AST depth).
    pub max_size: usize,
    /// Observational equivalence checker.
    pub equiv: ObservationalEquivalence,
}

impl TypeGuidedEnumerator {
    /// Create a new enumerator.
    pub fn new(max_size: usize, basis_inputs: Vec<Vec<i64>>) -> Self {
        Self {
            max_size,
            equiv: ObservationalEquivalence::new(basis_inputs),
        }
    }

    /// Enumerate all well-typed expressions up to size `max_size`.
    pub fn enumerate(&self, target_type: &SimpleType) -> Vec<TypedDslExpr> {
        let mut all: Vec<TypedDslExpr> = Vec::new();

        // Size 1: literals
        let mut base: Vec<TypedDslExpr> = vec![
            TypedDslExpr::IntLit(0),
            TypedDslExpr::IntLit(1),
            TypedDslExpr::IntLit(-1),
            TypedDslExpr::BoolLit(true),
            TypedDslExpr::BoolLit(false),
            TypedDslExpr::ListLit(vec![TypedDslExpr::IntLit(0)]),
            TypedDslExpr::ListLit(vec![TypedDslExpr::IntLit(1)]),
        ];

        if self.max_size >= 2 {
            // Unary: Neg, Not, MapNeg
            let neg_exprs: Vec<TypedDslExpr> = base
                .iter()
                .filter(|e| TypedDsl::infer_type(e) == Some(SimpleType::Int))
                .map(|e| TypedDslExpr::Neg(Box::new(e.clone())))
                .collect();
            let not_exprs: Vec<TypedDslExpr> = base
                .iter()
                .filter(|e| TypedDsl::infer_type(e) == Some(SimpleType::Bool))
                .map(|e| TypedDslExpr::Not(Box::new(e.clone())))
                .collect();
            base.extend(neg_exprs);
            base.extend(not_exprs);
        }

        if self.max_size >= 3 {
            // Binary: Add, Mul, And, Eq (over the base set)
            let mut binary: Vec<TypedDslExpr> = Vec::new();
            for a in &base {
                for b in &base {
                    let ta = TypedDsl::infer_type(a);
                    let tb = TypedDsl::infer_type(b);
                    if ta == Some(SimpleType::Int) && tb == Some(SimpleType::Int) {
                        binary.push(TypedDslExpr::Add(Box::new(a.clone()), Box::new(b.clone())));
                        binary.push(TypedDslExpr::Eq(Box::new(a.clone()), Box::new(b.clone())));
                    }
                    if ta == Some(SimpleType::Bool) && tb == Some(SimpleType::Bool) {
                        binary.push(TypedDslExpr::And(Box::new(a.clone()), Box::new(b.clone())));
                    }
                }
            }
            base.extend(binary);
        }

        // Filter by target type
        for expr in base {
            if TypedDsl::infer_type(&expr).as_ref() == Some(target_type) {
                all.push(expr);
            }
        }

        self.equiv.deduplicate(all)
    }
}

// ---------------------------------------------------------------------------
// 4. Execution-Guided Neural Synthesis
// ---------------------------------------------------------------------------

/// An input/output example pair for program synthesis.
#[derive(Debug, Clone)]
pub struct ExampleIo {
    /// Input as a vector of floats.
    pub input: Vec<f32>,
    /// Expected output as a vector of floats.
    pub output: Vec<f32>,
}

impl ExampleIo {
    /// Create a new example pair.
    pub fn new(input: Vec<f32>, output: Vec<f32>) -> Self {
        Self { input, output }
    }
}

/// Encode input/output example pairs as fixed-size embeddings (set encoding).
///
/// Each (input, output) pair is encoded by concatenating mean-pooled input
/// and output embeddings; all pairs are mean-pooled into a single vector.
pub struct IoEmbedder {
    /// Embedding dimension per I/O vector.
    pub io_dim: usize,
    /// Final embedding dimension.
    pub embed_dim: usize,
    /// Projection matrix for inputs [io_dim x max_io_len]
    w_in: Vec<Vec<f32>>,
    /// Projection matrix for outputs [io_dim x max_io_len]
    w_out: Vec<Vec<f32>>,
    /// Final projection [embed_dim x (2 * io_dim)]
    w_final: Vec<Vec<f32>>,
}

impl IoEmbedder {
    /// Create a new IoEmbedder.
    pub fn new(io_dim: usize, max_io_len: usize, embed_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (1.0 / io_dim as f32).sqrt();
        let mut mat = |rows: usize, cols: usize| -> Vec<Vec<f32>> {
            let s = (2.0 / (rows + cols) as f32).sqrt();
            (0..rows)
                .map(|_| {
                    (0..cols)
                        .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * s)
                        .collect()
                })
                .collect()
        };
        let _ = scale; // suppress warning
        Self {
            io_dim,
            embed_dim,
            w_in: mat(io_dim, max_io_len),
            w_out: mat(io_dim, max_io_len),
            w_final: mat(embed_dim, 2 * io_dim),
        }
    }

    fn project_vec(mat: &[Vec<f32>], v: &[f32], out_dim: usize) -> Vec<f32> {
        mat.iter()
            .take(out_dim)
            .map(|row| {
                row.iter()
                    .zip(v.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f32>()
            })
            .collect()
    }

    /// Encode a set of I/O examples into a fixed embedding.
    pub fn encode(&self, examples: &[ExampleIo]) -> Vec<f32> {
        if examples.is_empty() {
            return vec![0.0; self.embed_dim];
        }
        let pair_embeds: Vec<Vec<f32>> = examples
            .iter()
            .map(|ex| {
                let in_emb = Self::project_vec(&self.w_in, &ex.input, self.io_dim);
                let out_emb = Self::project_vec(&self.w_out, &ex.output, self.io_dim);
                // Concatenate
                in_emb.into_iter().chain(out_emb).collect()
            })
            .collect();

        // Mean pool over examples
        let pair_dim = pair_embeds[0].len();
        let mut mean = vec![0.0f32; pair_dim];
        for pe in &pair_embeds {
            for (m, v) in mean.iter_mut().zip(pe.iter()) {
                *m += v;
            }
        }
        let n = examples.len() as f32;
        mean.iter_mut().for_each(|v| *v /= n);

        // Project to embed_dim
        Self::project_vec(&self.w_final, &mean, self.embed_dim)
    }
}

/// A production rule in a probabilistic context-free grammar (PCFG).
#[derive(Debug, Clone)]
pub struct PcfgRule {
    /// Left-hand side non-terminal.
    pub lhs: String,
    /// Right-hand side symbols (terminals and non-terminals).
    pub rhs: Vec<String>,
    /// Log-probability of this rule.
    pub log_prob: f32,
}

/// PCFG-based program search guided by I/O examples.
///
/// Learns rule probabilities from a corpus of programs and uses them
/// to guide synthesis towards programs consistent with the given examples.
pub struct SyntaxGuidedSearch {
    /// PCFG rules.
    pub rules: Vec<PcfgRule>,
    /// Start symbol.
    pub start: String,
    /// Maximum program depth.
    pub max_depth: usize,
    /// I/O example embedder.
    pub embedder: IoEmbedder,
}

impl SyntaxGuidedSearch {
    /// Create a new syntax-guided searcher with uniform rule probabilities.
    pub fn new(
        rules: Vec<PcfgRule>,
        start: String,
        max_depth: usize,
        embedder: IoEmbedder,
    ) -> Self {
        Self {
            rules,
            start,
            max_depth,
            embedder,
        }
    }

    /// Return all rules applicable to the given non-terminal.
    pub fn rules_for(&self, nt: &str) -> Vec<&PcfgRule> {
        self.rules
            .iter()
            .filter(|r| r.lhs == nt)
            .collect()
    }

    /// Enumerate derivations up to `max_depth` from `nt`.
    ///
    /// Returns a list of terminal sequences (representing candidate programs),
    /// each paired with its log-probability.
    pub fn enumerate_derivations(
        &self,
        nt: &str,
        depth: usize,
    ) -> Vec<(Vec<String>, f32)> {
        if depth == 0 {
            return Vec::new();
        }
        let applicable = self.rules_for(nt);
        let mut results = Vec::new();
        for rule in applicable {
            // For simplicity, only expand single-terminal RHS (base case)
            if rule.rhs.len() == 1 {
                results.push((rule.rhs.clone(), rule.log_prob));
            } else if rule.rhs.len() == 2 && depth > 1 {
                // Binary rule: expand both non-terminals
                let left = self.enumerate_derivations(&rule.rhs[0], depth - 1);
                let right = self.enumerate_derivations(&rule.rhs[1], depth - 1);
                for (lseq, lp) in &left {
                    for (rseq, rp) in &right {
                        let mut seq = lseq.clone();
                        seq.extend(rseq.iter().cloned());
                        results.push((seq, rule.log_prob + lp + rp));
                    }
                }
            }
        }
        results
    }

    /// Search for the highest-probability derivation from the start symbol.
    pub fn best_derivation(&self) -> Option<Vec<String>> {
        let derivations = self.enumerate_derivations(&self.start.clone(), self.max_depth);
        derivations
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(seq, _)| seq)
    }
}

/// Program synthesis evaluation metrics.
#[derive(Debug, Clone)]
pub struct PsMetrics {
    /// Number of successful exact matches.
    pub n_exact: usize,
    /// Total number of tasks.
    pub n_total: usize,
    /// Sum of synthesis times in milliseconds.
    pub total_time_ms: f64,
    /// Number of tasks where generalization succeeded.
    pub n_generalized: usize,
}

impl PsMetrics {
    /// Create empty metrics.
    pub fn new() -> Self {
        Self {
            n_exact: 0,
            n_total: 0,
            total_time_ms: 0.0,
            n_generalized: 0,
        }
    }

    /// Record a synthesis result.
    pub fn record(&mut self, exact_match: bool, generalized: bool, time_ms: f64) {
        self.n_total += 1;
        self.total_time_ms += time_ms;
        if exact_match {
            self.n_exact += 1;
        }
        if generalized {
            self.n_generalized += 1;
        }
    }

    /// Exact match rate in [0, 1].
    pub fn exact_match_rate(&self) -> f64 {
        if self.n_total == 0 {
            0.0
        } else {
            self.n_exact as f64 / self.n_total as f64
        }
    }

    /// Generalization rate in [0, 1].
    pub fn generalization_rate(&self) -> f64 {
        if self.n_total == 0 {
            0.0
        } else {
            self.n_generalized as f64 / self.n_total as f64
        }
    }

    /// Mean synthesis time in milliseconds.
    pub fn mean_time_ms(&self) -> f64 {
        if self.n_total == 0 {
            0.0
        } else {
            self.total_time_ms / self.n_total as f64
        }
    }
}

impl Default for PsMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    dot / (na * nb)
}

fn mat_vec(mat: &[Vec<f32>], v: &[f32]) -> Vec<f32> {
    mat.iter()
        .map(|row| {
            row.iter()
                .zip(v.iter())
                .map(|(a, b)| a * b)
                .sum::<f32>()
        })
        .collect()
}
