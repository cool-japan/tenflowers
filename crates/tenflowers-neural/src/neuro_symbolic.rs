//! Neuro-Symbolic AI & Logic Learning.
//!
//! Integrates symbolic reasoning with neural learning:
//! - Differentiable fuzzy logic (Łukasiewicz, Product t-norms), LTN groundings.
//! - Neural program synthesis with beam search and RNN embeddings.
//! - Enhanced KG embeddings: TransR, DistMult, analogy, path queries, AMIE rules.
//! - Constraint satisfaction: AC-3, GNN-guided CSP, differentiable SAT loss.
//! - Causal reasoning: SCMs, do-calculus, counterfactuals, PC algorithm, backdoor adjustment.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::{HashMap, HashSet, VecDeque};
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: Differentiable Logic
// ─────────────────────────────────────────────────────────────────────────────

/// Łukasiewicz fuzzy logic: t-norm=max(0,a+b−1), t-conorm=min(1,a+b), neg=1−a.
#[derive(Debug, Clone)]
pub struct FuzzyLogic;

impl FuzzyLogic {
    /// Łukasiewicz t-norm: `max(0, a+b−1)`.
    #[inline]
    pub fn tnorm(a: f64, b: f64) -> f64 {
        (a + b - 1.0).max(0.0)
    }
    /// Łukasiewicz t-conorm: `min(1, a+b)`.
    #[inline]
    pub fn tconorm(a: f64, b: f64) -> f64 {
        (a + b).min(1.0)
    }
    /// Łukasiewicz negation: `1−a`.
    #[inline]
    pub fn negation(a: f64) -> f64 {
        1.0 - a
    }
    /// Łukasiewicz implication: `min(1, 1−a+b)`.
    #[inline]
    pub fn implication(a: f64, b: f64) -> f64 {
        (1.0 - a + b).min(1.0)
    }
}

/// Product fuzzy logic: t-norm=a*b, t-conorm=a+b-ab, implication=min(1,b/a).
#[derive(Debug, Clone)]
pub struct ProductFuzzy;

impl ProductFuzzy {
    /// Product t-norm: `a * b`.
    #[inline]
    pub fn tnorm(a: f64, b: f64) -> f64 {
        a * b
    }
    /// Product t-conorm: `a + b − a*b`.
    #[inline]
    pub fn tconorm(a: f64, b: f64) -> f64 {
        a + b - a * b
    }
    /// Product implication: `min(1, b/a)` with guard for a≈0.
    #[inline]
    pub fn implication(a: f64, b: f64) -> f64 {
        if a < 1e-12 {
            1.0
        } else {
            (b / a).min(1.0)
        }
    }
}

/// Neural theorem prover: soft unification via cosine similarity.
#[derive(Debug, Clone)]
pub struct NeuralTheoremProver {
    /// Predicate embedding dimension.
    pub dim: usize,
    /// Registered predicate embeddings: id → unit vector.
    pub embeddings: HashMap<usize, Vec<f64>>,
}

impl NeuralTheoremProver {
    pub fn new(dim: usize) -> Self {
        NeuralTheoremProver {
            dim,
            embeddings: HashMap::new(),
        }
    }

    pub fn register(&mut self, id: usize, embedding: Vec<f64>) -> Result<()> {
        if embedding.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "register",
                &format!("expected dim {}, got {}", self.dim, embedding.len()),
            ));
        }
        self.embeddings.insert(id, normalise_vec(&embedding));
        Ok(())
    }

    /// Soft unification via cosine similarity mapped to \[0,1\].
    pub fn unify(p: &[f64], q: &[f64]) -> f64 {
        0.5 * (cosine_similarity(p, q) + 1.0)
    }

    /// Prove a chain of predicate-pair rules via min (Gödel) composition.
    pub fn prove_chain(&self, rule_chain: &[(usize, usize)]) -> f64 {
        rule_chain.iter().fold(1.0_f64, |acc, (pid, qid)| {
            let p = self
                .embeddings
                .get(pid)
                .cloned()
                .unwrap_or_else(|| vec![0.0; self.dim]);
            let q = self
                .embeddings
                .get(qid)
                .cloned()
                .unwrap_or_else(|| vec![0.0; self.dim]);
            acc.min(Self::unify(&p, &q))
        })
    }
}

/// Differentiable forward chaining over soft rules using weighted Łukasiewicz t-norms.
#[derive(Debug, Clone)]
pub struct DifferentiableForwardChaining {
    pub max_steps: usize,
}

impl DifferentiableForwardChaining {
    pub fn new(max_steps: usize) -> Self {
        DifferentiableForwardChaining { max_steps }
    }

    /// Apply one rule: weighted sum of Łukasiewicz t-norms over antecedent pairs.
    pub fn apply_rule(antecedents: &[f64], rule_weights: &[f64]) -> f64 {
        if antecedents.is_empty() || rule_weights.is_empty() {
            return 0.0;
        }
        let max_w = rule_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = rule_weights.iter().map(|w| (w - max_w).exp()).collect();
        let sum_exp: f64 = exps.iter().sum();
        let n = antecedents.len().min(rule_weights.len());
        (0..n).fold(0.0_f64, |acc, i| {
            let next = *antecedents.get(i + 1).unwrap_or(&antecedents[i]);
            acc + (exps[i] / sum_exp) * FuzzyLogic::tnorm(antecedents[i], next)
        })
    }

    /// Run multi-step forward chaining, returning truth at each step.
    pub fn chain(&self, initial: &[f64], rule_weights: &[f64]) -> Vec<f64> {
        let mut states = vec![Self::apply_rule(initial, rule_weights)];
        for _step in 1..self.max_steps {
            let last = *states.last().unwrap_or(&0.0);
            let pseudo: Vec<f64> = initial
                .iter()
                .map(|&a| FuzzyLogic::tnorm(a, last))
                .collect();
            states.push(Self::apply_rule(&pseudo, rule_weights));
        }
        states
    }
}

/// Logic Tensor Network: ground atoms as sigmoid(pred · arg) truth values.
#[derive(Debug, Clone)]
pub struct LogicTensorNetwork {
    pub dim: usize,
}

impl LogicTensorNetwork {
    pub fn new(dim: usize) -> Self {
        LogicTensorNetwork { dim }
    }

    /// Compute truth values for predicate applied to each argument vector.
    pub fn grounding(predicate: &[f64], args: &[Vec<f64>]) -> Vec<f64> {
        args.iter()
            .map(|arg| sigmoid(predicate.iter().zip(arg.iter()).map(|(p, a)| p * a).sum()))
            .collect()
    }

    /// Mean satisfiability over all argument groundings.
    pub fn satisfiability(&self, predicate: &[f64], args: &[Vec<f64>]) -> f64 {
        let ts = Self::grounding(predicate, args);
        if ts.is_empty() {
            0.0
        } else {
            ts.iter().sum::<f64>() / ts.len() as f64
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Neural Program Synthesis
// ─────────────────────────────────────────────────────────────────────────────

/// Token type for a simple differentiable program language.
#[derive(Debug, Clone, PartialEq)]
pub enum ProgramToken {
    If,
    Then,
    Else,
    Loop(usize),
    Add,
    Sub,
    Mul,
    Eq,
    Lt,
    Var(usize),
    Const(f64),
}

/// Recursive program AST node.
#[derive(Debug, Clone)]
pub enum ProgramAst {
    Constant(f64),
    Variable(usize),
    BinOp(ProgramToken, Box<ProgramAst>, Box<ProgramAst>),
    IfThenElse(Box<ProgramAst>, Box<ProgramAst>, Box<ProgramAst>),
    Sequence(Vec<ProgramAst>),
    LoopNode(usize, Box<ProgramAst>),
}

impl ProgramAst {
    /// Evaluate the AST, mutating environment registers.
    pub fn evaluate(&self, env: &mut Vec<f64>) -> f64 {
        match self {
            ProgramAst::Constant(v) => *v,
            ProgramAst::Variable(idx) => env.get(*idx).copied().unwrap_or(0.0),
            ProgramAst::BinOp(op, l, r) => {
                let lv = l.evaluate(env);
                let rv = r.evaluate(env);
                match op {
                    ProgramToken::Add => lv + rv,
                    ProgramToken::Sub => lv - rv,
                    ProgramToken::Mul => lv * rv,
                    ProgramToken::Eq
                        if (lv - rv).abs() < 1e-9 => {
                            1.0
                        }
                    ProgramToken::Lt
                        if lv < rv => {
                            1.0
                        }
                    _ => 0.0,
                }
            }
            ProgramAst::IfThenElse(cond, then_b, else_b) => {
                if cond.evaluate(env) > 0.5 {
                    then_b.evaluate(env)
                } else {
                    else_b.evaluate(env)
                }
            }
            ProgramAst::Sequence(stmts) => stmts.iter().fold(0.0, |_, s| s.evaluate(env)),
            ProgramAst::LoopNode(n, body) => (0..*n).fold(0.0, |_, _| body.evaluate(env)),
        }
    }
}

/// Neural-guided program searcher using beam search over token sequences.
#[derive(Debug, Clone)]
pub struct NeuralProgramSearcher {
    pub vocabulary: Vec<ProgramToken>,
    pub embed_dim: usize,
    pub token_embeddings: Vec<Vec<f64>>,
}

impl NeuralProgramSearcher {
    pub fn new(embed_dim: usize, seed: u64) -> Self {
        let vocabulary = vec![
            ProgramToken::Add,
            ProgramToken::Sub,
            ProgramToken::Mul,
            ProgramToken::Eq,
            ProgramToken::Lt,
            ProgramToken::Var(0),
            ProgramToken::Var(1),
            ProgramToken::Const(0.0),
            ProgramToken::Const(1.0),
            ProgramToken::If,
            ProgramToken::Then,
            ProgramToken::Else,
        ];
        let mut rng = StdRng::seed_from_u64(seed);
        let token_embeddings = vocabulary
            .iter()
            .map(|_| {
                (0..embed_dim)
                    .map(|_| rng.random::<f64>() * 2.0 - 1.0)
                    .collect()
            })
            .collect();
        NeuralProgramSearcher {
            vocabulary,
            embed_dim,
            token_embeddings,
        }
    }

    fn tokens_to_ast(&self, tokens: &[ProgramToken]) -> ProgramAst {
        if tokens.is_empty() {
            return ProgramAst::Constant(0.0);
        }
        let mut nodes: Vec<ProgramAst> = tokens
            .iter()
            .map(|tok| match tok {
                ProgramToken::Const(v) => ProgramAst::Constant(*v),
                ProgramToken::Var(i) => ProgramAst::Variable(*i),
                _ => ProgramAst::Constant(0.0),
            })
            .collect();
        let ops: Vec<&ProgramToken> = tokens
            .iter()
            .filter(|t| {
                matches!(
                    t,
                    ProgramToken::Add
                        | ProgramToken::Sub
                        | ProgramToken::Mul
                        | ProgramToken::Eq
                        | ProgramToken::Lt
                )
            })
            .collect();
        if ops.is_empty() || nodes.len() < 2 {
            return nodes
                .into_iter()
                .next()
                .unwrap_or(ProgramAst::Constant(0.0));
        }
        let lhs = nodes.remove(0);
        let rhs = nodes
            .into_iter()
            .next()
            .unwrap_or(ProgramAst::Constant(0.0));
        ProgramAst::BinOp(ops[0].clone(), Box::new(lhs), Box::new(rhs))
    }

    fn score_program(&self, tokens: &[ProgramToken], pairs: &[(&[f64], f64)]) -> f64 {
        if tokens.is_empty() || pairs.is_empty() {
            return f64::MAX;
        }
        let ast = self.tokens_to_ast(tokens);
        pairs
            .iter()
            .map(|(inp, tgt)| {
                let mut env = inp.to_vec();
                (ast.evaluate(&mut env) - tgt).powi(2)
            })
            .sum::<f64>()
            / pairs.len() as f64
    }

    /// Beam search: returns best token sequence within max_depth / beam_width.
    pub fn search(
        &self,
        input_output_pairs: &[(&[f64], f64)],
        max_depth: usize,
        beam_width: usize,
        rng: &mut StdRng,
    ) -> Vec<ProgramToken> {
        let mut beam: Vec<(f64, Vec<ProgramToken>)> = vec![(f64::MAX, vec![])];
        for _depth in 0..max_depth {
            let mut cands: Vec<(f64, Vec<ProgramToken>)> = beam
                .iter()
                .flat_map(|(_, seq)| {
                    self.vocabulary.iter().map(move |tok| {
                        let mut ns = seq.clone();
                        ns.push(tok.clone());
                        let s = self.score_program(&ns, input_output_pairs);
                        (s, ns)
                    })
                })
                .collect();
            cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            cands.truncate(beam_width);
            // Fisher-Yates shuffle ties for diversity
            if cands.len() > 1 {
                let tie = cands[0].0;
                let te = cands
                    .iter()
                    .position(|(s, _)| (*s - tie).abs() > 1e-12)
                    .unwrap_or(cands.len());
                for i in (1..te.min(cands.len())).rev() {
                    let j = rng.random_range(0..=i);
                    cands.swap(i, j);
                }
            }
            beam = cands;
        }
        beam.into_iter().next().map(|(_, s)| s).unwrap_or_default()
    }
}

/// Elman-RNN program embedder: token ids → fixed-size embedding via tanh RNN.
#[derive(Debug, Clone)]
pub struct ProgramEmbedding {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub w_in: Vec<Vec<f64>>,
    pub w_rec: Vec<Vec<f64>>,
    pub w_out: Vec<Vec<f64>>,
}

impl ProgramEmbedding {
    pub fn new(vocab_size: usize, hidden_size: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (2.0_f64 / hidden_size as f64).sqrt();
        let rw = |rng: &mut StdRng, r, c| -> Vec<Vec<f64>> {
            (0..r)
                .map(|_| {
                    (0..c)
                        .map(|_| rng.random::<f64>() * scale - scale / 2.0)
                        .collect()
                })
                .collect()
        };
        let w_in = rw(&mut rng, vocab_size, hidden_size);
        let w_rec = rw(&mut rng, hidden_size, hidden_size);
        let w_out = rw(&mut rng, hidden_size, hidden_size);
        ProgramEmbedding {
            vocab_size,
            hidden_size,
            w_in,
            w_rec,
            w_out,
        }
    }

    /// Embed a token-index sequence → final hidden state after RNN.
    pub fn embed(&self, token_ids: &[usize]) -> Vec<f64> {
        let mut hidden = vec![0.0_f64; self.hidden_size];
        for &tok_id in token_ids {
            let ti = tok_id.min(self.vocab_size.saturating_sub(1));
            hidden = (0..self.hidden_size)
                .map(|j| {
                    let inp = self.w_in[ti][j];
                    let rec: f64 = self
                        .w_rec
                        .iter()
                        .enumerate()
                        .map(|(k, row)| row[j] * hidden[k])
                        .sum();
                    (inp + rec).tanh()
                })
                .collect();
        }
        (0..self.hidden_size)
            .map(|j| {
                self.w_out
                    .iter()
                    .enumerate()
                    .map(|(k, row)| row[j] * hidden[k])
                    .sum::<f64>()
                    .tanh()
            })
            .collect()
    }
}

/// Execution tracer: records (token, value) pairs during program evaluation.
#[derive(Debug, Clone)]
pub struct ExecutionTracing;

impl ExecutionTracing {
    pub fn trace(program: &[ProgramToken], input: &[f64]) -> Vec<(ProgramToken, f64)> {
        let mut env = input.to_vec();
        let mut acc = 0.0_f64;
        program
            .iter()
            .map(|tok| {
                let val = match tok {
                    ProgramToken::Const(v) => {
                        acc = *v;
                        acc
                    }
                    ProgramToken::Var(idx) => {
                        acc = env.get(*idx).copied().unwrap_or(0.0);
                        acc
                    }
                    ProgramToken::Add => {
                        let b = env.pop().unwrap_or(0.0);
                        let a = env.pop().unwrap_or(0.0);
                        acc = a + b;
                        env.push(acc);
                        acc
                    }
                    ProgramToken::Sub => {
                        let b = env.pop().unwrap_or(0.0);
                        let a = env.pop().unwrap_or(0.0);
                        acc = a - b;
                        env.push(acc);
                        acc
                    }
                    ProgramToken::Mul => {
                        let b = env.pop().unwrap_or(0.0);
                        let a = env.pop().unwrap_or(0.0);
                        acc = a * b;
                        env.push(acc);
                        acc
                    }
                    ProgramToken::Eq => {
                        let b = env.last().copied().unwrap_or(0.0);
                        acc = if (acc - b).abs() < 1e-9 { 1.0 } else { 0.0 };
                        acc
                    }
                    ProgramToken::Lt => {
                        let b = env.last().copied().unwrap_or(0.0);
                        acc = if acc < b { 1.0 } else { 0.0 };
                        acc
                    }
                    ProgramToken::If | ProgramToken::Then | ProgramToken::Else => acc,
                    ProgramToken::Loop(n) => {
                        acc *= *n as f64;
                        acc
                    }
                };
                (tok.clone(), val)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Knowledge Graph Embedding Enhanced
// ─────────────────────────────────────────────────────────────────────────────

/// TransR: projects entity embeddings into relation-specific subspace via M_r.
///
/// Score = −||M_r*h + r − M_r*t||_2.
#[derive(Debug, Clone)]
pub struct TransRModel {
    pub entity_dim: usize,
    pub relation_dim: usize,
}

impl TransRModel {
    pub fn new(entity_dim: usize, relation_dim: usize) -> Self {
        TransRModel {
            entity_dim,
            relation_dim,
        }
    }

    pub fn score(&self, h: &[f64], r: &[f64], t: &[f64], m_r: &[Vec<f64>]) -> Result<f64> {
        if h.len() != self.entity_dim {
            return Err(TensorError::invalid_argument_op(
                "TransRModel::score",
                &format!("expected h.len()={}, got {}", self.entity_dim, h.len()),
            ));
        }
        if r.len() != self.relation_dim {
            return Err(TensorError::invalid_argument_op(
                "TransRModel::score",
                &format!("expected r.len()={}, got {}", self.relation_dim, r.len()),
            ));
        }
        if m_r.len() != self.relation_dim {
            return Err(TensorError::invalid_argument_op(
                "TransRModel::score",
                "M_r rows must equal relation_dim",
            ));
        }
        let proj = |e: &[f64]| -> Vec<f64> {
            m_r.iter()
                .map(|row| row.iter().zip(e.iter()).map(|(m, ei)| m * ei).sum())
                .collect()
        };
        let ph = proj(h);
        let pt = proj(t);
        let dist = ph
            .iter()
            .zip(r.iter())
            .zip(pt.iter())
            .map(|((p, ri), q)| (p + ri - q).powi(2))
            .sum::<f64>()
            .sqrt();
        Ok(-dist)
    }
}

/// DistMult: score = Σ h_i * r_i * t_i (symmetric bilinear form).
#[derive(Debug, Clone)]
pub struct DistMultModel {
    pub dim: usize,
}

impl DistMultModel {
    pub fn new(dim: usize) -> Self {
        DistMultModel { dim }
    }

    pub fn score(&self, h: &[f64], r: &[f64], t: &[f64]) -> Result<f64> {
        if h.len() != self.dim || r.len() != self.dim || t.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "DistMultModel::score",
                "all lengths must equal dim",
            ));
        }
        Ok(h.iter()
            .zip(r.iter())
            .zip(t.iter())
            .map(|((hi, ri), ti)| hi * ri * ti)
            .sum())
    }
}

/// Analogy reasoning in embedding space: A:B = C:D → find nearest D.
#[derive(Debug, Clone)]
pub struct AnalygyReasoning;

impl AnalygyReasoning {
    /// Solve A:B = C:? by computing target = B−A+C and nearest cosine neighbor.
    pub fn solve_analogy(a: &[f64], b: &[f64], c: &[f64], embeddings: &[Vec<f64>]) -> usize {
        let target = normalise_vec(
            &b.iter()
                .zip(a.iter())
                .zip(c.iter())
                .map(|((bi, ai), ci)| bi - ai + ci)
                .collect::<Vec<_>>(),
        );
        embeddings
            .iter()
            .enumerate()
            .map(|(i, e)| (i, cosine_similarity(&target, e)))
            .max_by(|(_, s1), (_, s2)| s1.partial_cmp(s2).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// Multi-hop path query embedding via TransE-style relation sum + normalisation.
#[derive(Debug, Clone)]
pub struct PathQueryEmbedding {
    pub dim: usize,
}

impl PathQueryEmbedding {
    pub fn new(dim: usize) -> Self {
        PathQueryEmbedding { dim }
    }

    pub fn embed_path(&self, relations: &[Vec<f64>]) -> Result<Vec<f64>> {
        if relations.is_empty() {
            return Ok(vec![0.0; self.dim]);
        }
        for (i, rel) in relations.iter().enumerate() {
            if rel.len() != self.dim {
                return Err(TensorError::invalid_argument_op(
                    "PathQueryEmbedding::embed_path",
                    &format!("relation[{}] len {} != dim {}", i, rel.len(), self.dim),
                ));
            }
        }
        let mut result = vec![0.0_f64; self.dim];
        for rel in relations {
            result.iter_mut().zip(rel.iter()).for_each(|(r, v)| *r += v);
        }
        Ok(normalise_vec(&result))
    }
}

/// Induced rule from knowledge graph triples.
#[derive(Debug, Clone)]
pub struct Rule {
    pub head: usize,
    pub body: Vec<usize>,
    pub confidence: f64,
    pub support: usize,
}

/// AMIE-style rule induction: mine closed depth-1 and depth-2 rules.
#[derive(Debug, Clone)]
pub struct RuleInduction;

impl RuleInduction {
    pub fn mine_rules(triples: &[(usize, usize, usize)], min_conf: f64) -> Vec<Rule> {
        let mut rel_pairs: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for &(h, r, t) in triples {
            rel_pairs.entry(r).or_default().push((h, t));
        }
        let relations: Vec<usize> = rel_pairs.keys().cloned().collect();
        let mut rules = Vec::new();

        // Depth-1 rules: r_body(X,Y) → r_head(X,Y)
        for &r_head in &relations {
            let head_set: HashSet<(usize, usize)> = rel_pairs[&r_head].iter().cloned().collect();
            for &r_body in &relations {
                if r_body == r_head {
                    continue;
                }
                let bp = &rel_pairs[&r_body];
                let sup = bp.iter().filter(|p| head_set.contains(p)).count();
                if sup == 0 {
                    continue;
                }
                let conf = sup as f64 / bp.len() as f64;
                if conf >= min_conf {
                    rules.push(Rule {
                        head: r_head,
                        body: vec![r_body],
                        confidence: conf,
                        support: sup,
                    });
                }
            }
        }

        // Depth-2 chain rules: r1(X,Z) ∧ r2(Z,Y) → r_head(X,Y)
        for &r_head in &relations {
            let head_set: HashSet<(usize, usize)> = rel_pairs[&r_head].iter().cloned().collect();
            for &r1 in &relations {
                let mut r2_from: HashMap<usize, Vec<usize>> = HashMap::new();
                for &r2 in &relations {
                    if r2 == r_head && r1 == r_head {
                        continue;
                    }
                    r2_from.clear();
                    if let Some(r2l) = rel_pairs.get(&r2) {
                        for &(h2, t2) in r2l {
                            r2_from.entry(h2).or_default().push(t2);
                        }
                    }
                    let mut join: HashSet<(usize, usize)> = HashSet::new();
                    if let Some(r1l) = rel_pairs.get(&r1) {
                        for &(h1, mid) in r1l {
                            if let Some(tails) = r2_from.get(&mid) {
                                for &t2 in tails {
                                    join.insert((h1, t2));
                                }
                            }
                        }
                    }
                    if join.is_empty() {
                        continue;
                    }
                    let sup = join.iter().filter(|p| head_set.contains(p)).count();
                    if sup == 0 {
                        continue;
                    }
                    let conf = sup as f64 / join.len() as f64;
                    if conf >= min_conf {
                        rules.push(Rule {
                            head: r_head,
                            body: vec![r1, r2],
                            confidence: conf,
                            support: sup,
                        });
                    }
                }
            }
        }
        rules.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rules
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: Constraint Satisfaction & Optimization
// ─────────────────────────────────────────────────────────────────────────────

/// CSP variable with integer domain.
#[derive(Debug, Clone)]
pub struct CspVariable {
    pub name: String,
    pub domain: Vec<i32>,
}

impl CspVariable {
    pub fn new(name: impl Into<String>, domain: Vec<i32>) -> Self {
        CspVariable {
            name: name.into(),
            domain,
        }
    }
    pub fn is_assigned(&self) -> bool {
        self.domain.len() == 1
    }
    pub fn assigned_value(&self) -> Option<i32> {
        if self.is_assigned() {
            Some(self.domain[0])
        } else {
            None
        }
    }
}

/// Kind of binary constraint.
#[derive(Debug, Clone, PartialEq)]
pub enum BinaryConstraintType {
    NotEqual,
    LessThan,
    LessEqual,
    Equal,
    MinGap(i32),
}

/// Binary constraint between two CSP variables.
#[derive(Debug, Clone)]
pub struct BinaryCspConstraint {
    pub var1: usize,
    pub var2: usize,
    pub constraint_type: BinaryConstraintType,
}

impl BinaryCspConstraint {
    pub fn new(var1: usize, var2: usize, constraint_type: BinaryConstraintType) -> Self {
        BinaryCspConstraint {
            var1,
            var2,
            constraint_type,
        }
    }
    pub fn check(&self, v1: i32, v2: i32) -> bool {
        match &self.constraint_type {
            BinaryConstraintType::NotEqual => v1 != v2,
            BinaryConstraintType::LessThan => v1 < v2,
            BinaryConstraintType::LessEqual => v1 <= v2,
            BinaryConstraintType::Equal => v1 == v2,
            BinaryConstraintType::MinGap(g) => (v1 - v2).abs() >= *g,
        }
    }
}

/// AC-3 arc consistency enforcer for binary CSPs.
#[derive(Debug, Clone)]
pub struct ArcConsistency;

impl ArcConsistency {
    /// Run AC-3, pruning domains in place. Returns false if any domain becomes empty.
    pub fn enforce(variables: &mut [CspVariable], constraints: &[BinaryCspConstraint]) -> bool {
        let mut queue: VecDeque<(usize, usize)> = constraints
            .iter()
            .flat_map(|c| [(c.var1, c.var2), (c.var2, c.var1)])
            .collect();
        while let Some((xi, xj)) = queue.pop_front() {
            if Self::revise(variables, constraints, xi, xj) {
                if variables[xi].domain.is_empty() {
                    return false;
                }
                for c in constraints {
                    let nb = if c.var1 == xi {
                        Some(c.var2)
                    } else if c.var2 == xi {
                        Some(c.var1)
                    } else {
                        None
                    };
                    if let Some(xk) = nb {
                        if xk != xj {
                            queue.push_back((xk, xi));
                        }
                    }
                }
            }
        }
        true
    }

    fn revise(
        variables: &mut [CspVariable],
        constraints: &[BinaryCspConstraint],
        xi: usize,
        xj: usize,
    ) -> bool {
        let relevant: Vec<&BinaryCspConstraint> = constraints
            .iter()
            .filter(|c| (c.var1 == xi && c.var2 == xj) || (c.var1 == xj && c.var2 == xi))
            .collect();
        if relevant.is_empty() {
            return false;
        }
        let dom_j: Vec<i32> = variables[xj].domain.clone();
        let init = variables[xi].domain.len();
        variables[xi].domain.retain(|&vi| {
            relevant.iter().all(|c| {
                dom_j.iter().any(|&vj| {
                    if c.var1 == xi {
                        c.check(vi, vj)
                    } else {
                        c.check(vj, vi)
                    }
                })
            })
        });
        variables[xi].domain.len() < init
    }
}

/// GNN-guided CSP solver: message-passing scores variable-value assignments.
#[derive(Debug, Clone)]
pub struct GnnCspSolver {
    pub num_rounds: usize,
    pub hidden_dim: usize,
}

impl GnnCspSolver {
    pub fn new(num_rounds: usize, hidden_dim: usize) -> Self {
        GnnCspSolver {
            num_rounds,
            hidden_dim,
        }
    }

    /// Forward pass: returns softmax assignment matrix (n_vars × domain_size).
    pub fn forward(
        &self,
        var_feats: &[Vec<f64>],
        constraint_feats: &[Vec<f64>],
        domain_size: usize,
    ) -> Vec<Vec<f64>> {
        let n_vars = var_feats.len();
        if n_vars == 0 {
            return Vec::new();
        }
        let feat_dim = var_feats[0].len().max(1);
        let mut node_emb: Vec<Vec<f64>> = var_feats
            .iter()
            .map(|f| {
                let scale = 1.0 / feat_dim as f64;
                (0..self.hidden_dim)
                    .map(|h| f.get(h % feat_dim).copied().unwrap_or(0.0) * scale)
                    .collect()
            })
            .collect();
        for _round in 0..self.num_rounds {
            let cf_dim = constraint_feats
                .first()
                .map(|f| f.len())
                .unwrap_or(1)
                .max(1);
            let cm: Vec<f64> = if constraint_feats.is_empty() {
                vec![0.0; self.hidden_dim]
            } else {
                (0..self.hidden_dim)
                    .map(|h| {
                        constraint_feats
                            .iter()
                            .map(|f| f.get(h % cf_dim).copied().unwrap_or(0.0))
                            .sum::<f64>()
                            / constraint_feats.len() as f64
                    })
                    .collect()
            };
            for emb in &mut node_emb {
                emb.iter_mut()
                    .zip(cm.iter())
                    .for_each(|(e, c)| *e = (*e + c * 0.1).tanh());
            }
        }
        node_emb
            .iter()
            .map(|emb| {
                let logits: Vec<f64> = (0..domain_size)
                    .map(|d| {
                        emb.iter()
                            .enumerate()
                            .map(|(i, &e)| e * ((i * d + 1) as f64).cos())
                            .sum::<f64>()
                    })
                    .collect();
                softmax(&logits)
            })
            .collect()
    }
}

/// Differentiable SAT loss: soft constraint satisfaction via Łukasiewicz t-conorm.
#[derive(Debug, Clone)]
pub struct SatisfiabilityLoss;

impl SatisfiabilityLoss {
    /// Mean hinge loss over unsatisfied clauses.
    pub fn compute(assignment: &[f64], clauses: &[Vec<(usize, bool)>]) -> f64 {
        if clauses.is_empty() {
            return 0.0;
        }
        clauses
            .iter()
            .map(|clause| {
                let truth = clause.iter().fold(0.0_f64, |acc, &(var, pos)| {
                    let lit = if var < assignment.len() {
                        if pos {
                            assignment[var]
                        } else {
                            1.0 - assignment[var]
                        }
                    } else {
                        0.0
                    };
                    FuzzyLogic::tconorm(acc, lit)
                });
                (1.0 - truth).max(0.0)
            })
            .sum::<f64>()
            / clauses.len() as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: Causal Reasoning
// ─────────────────────────────────────────────────────────────────────────────

type StructuralEquation = Box<dyn Fn(&HashMap<String, f64>) -> f64 + Send + Sync>;

/// Structural Causal Model (SCM) with do-calculus intervention support.
pub struct StructuralCausalModel {
    pub variables: Vec<String>,
    pub parents: HashMap<String, Vec<String>>,
    equations: HashMap<String, StructuralEquation>,
    interventions: HashMap<String, f64>,
}

impl std::fmt::Debug for StructuralCausalModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StructuralCausalModel")
            .field("variables", &self.variables)
            .field("parents", &self.parents)
            .field("interventions", &self.interventions)
            .finish()
    }
}

impl StructuralCausalModel {
    pub fn new() -> Self {
        StructuralCausalModel {
            variables: Vec::new(),
            parents: HashMap::new(),
            equations: HashMap::new(),
            interventions: HashMap::new(),
        }
    }

    pub fn add_variable(
        &mut self,
        name: impl Into<String>,
        parents: Vec<String>,
        equation: StructuralEquation,
    ) {
        let name = name.into();
        self.parents.insert(name.clone(), parents);
        self.equations.insert(name.clone(), equation);
        if !self.variables.contains(&name) {
            self.variables.push(name);
        }
    }

    /// Set hard intervention: do(variable = value).
    pub fn intervene(&mut self, variable: impl Into<String>, value: f64) {
        self.interventions.insert(variable.into(), value);
    }

    pub fn remove_intervention(&mut self, variable: &str) {
        self.interventions.remove(variable);
    }

    /// Forward-evaluate all variables in topological order.
    pub fn evaluate(&self, evidence: &HashMap<String, f64>) -> HashMap<String, f64> {
        let mut values: HashMap<String, f64> = evidence.clone();
        for var in &self.variables {
            if let Some(&v) = self.interventions.get(var) {
                values.insert(var.clone(), v);
                continue;
            }
            if evidence.contains_key(var) {
                continue;
            }
            if let Some(eq) = self.equations.get(var) {
                values.insert(var.clone(), eq(&values));
            }
        }
        values
    }

    /// Query expected value of target given evidence.
    pub fn query(&self, target: &str, evidence: &HashMap<String, f64>) -> f64 {
        self.evaluate(evidence).get(target).copied().unwrap_or(0.0)
    }
}

impl Default for StructuralCausalModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Do-calculus intervention layer: override specific activations with fixed values.
#[derive(Debug, Clone)]
pub struct InterventionLayer {
    pub overrides: HashMap<usize, f64>,
}

impl InterventionLayer {
    pub fn new() -> Self {
        InterventionLayer {
            overrides: HashMap::new(),
        }
    }
    pub fn set_intervention(&mut self, index: usize, value: f64) {
        self.overrides.insert(index, value);
    }
    pub fn apply(&self, activations: &[f64]) -> Vec<f64> {
        activations
            .iter()
            .enumerate()
            .map(|(i, &a)| self.overrides.get(&i).copied().unwrap_or(a))
            .collect()
    }
}

impl Default for InterventionLayer {
    fn default() -> Self {
        Self::new()
    }
}

/// Counterfactual estimator via twin-network (abduction-action-prediction).
#[derive(Debug, Clone)]
pub struct CounterfactualEstimator {
    pub noise_variance: f64,
}

impl CounterfactualEstimator {
    pub fn new(noise_variance: f64) -> Self {
        CounterfactualEstimator { noise_variance }
    }

    /// Linear SCM counterfactual: abduct U = Y − X, predict Y_cf = X_cf + U.
    pub fn estimate(&self, x_actual: f64, x_counter: f64, observed: f64) -> f64 {
        let u = observed - x_actual; // beta=1 linear assumption
        x_counter + u
    }
}

/// Constraint-based causal discovery (PC algorithm skeleton via Fisher Z-test).
#[derive(Debug, Clone)]
pub struct CausalDiscovery {
    pub alpha: f64,
}

impl CausalDiscovery {
    pub fn new(alpha: f64) -> Self {
        CausalDiscovery { alpha }
    }

    /// Discover skeleton edges: (i,j) pairs where marginal dependence is detected.
    pub fn discover(&self, data: &[Vec<f64>], alpha: f64) -> Vec<(usize, usize)> {
        if data.is_empty() {
            return Vec::new();
        }
        let n_vars = data[0].len();
        let n = data.len();
        let means: Vec<f64> = (0..n_vars)
            .map(|j| data.iter().map(|row| row[j]).sum::<f64>() / n as f64)
            .collect();
        let mut skeleton: HashSet<(usize, usize)> = HashSet::new();
        for i in 0..n_vars {
            for j in (i + 1)..n_vars {
                let r = pearson_correlation(data, i, j, &means);
                let z = fisher_z(r, n);
                let crit = normal_quantile_approx(1.0 - alpha / 2.0);
                if z.abs() >= crit {
                    skeleton.insert((i, j));
                }
            }
        }
        skeleton.into_iter().collect()
    }
}

/// Neural backdoor adjustment via IPW with learned propensity score.
#[derive(Debug, Clone)]
pub struct NeuralBackdoorAdjustment {
    pub confounder_dim: usize,
    pub treatment_dim: usize,
    pub weights: Vec<f64>,
}

impl NeuralBackdoorAdjustment {
    pub fn new(confounder_dim: usize, treatment_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let weights = (0..confounder_dim)
            .map(|_| rng.random::<f64>() * 0.1)
            .collect();
        NeuralBackdoorAdjustment {
            confounder_dim,
            treatment_dim,
            weights,
        }
    }

    pub fn propensity(&self, z: &[f64]) -> f64 {
        sigmoid(
            self.weights
                .iter()
                .zip(z.iter())
                .map(|(w, zi)| w * zi)
                .sum(),
        )
    }

    pub fn adjust(&self, x: f64, z: &[f64], y_pred: f64) -> f64 {
        let ps = self.propensity(z).clamp(1e-6, 1.0 - 1e-6);
        if x > 0.5 {
            y_pred / ps
        } else {
            y_pred / (1.0 - ps)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility functions
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        0.0
    } else {
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}

#[inline]
fn normalise_vec(v: &[f64]) -> Vec<f64> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-12 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum < 1e-12 {
        vec![1.0 / logits.len() as f64; logits.len()]
    } else {
        exps.iter().map(|e| e / sum).collect()
    }
}

fn pearson_correlation(data: &[Vec<f64>], i: usize, j: usize, means: &[f64]) -> f64 {
    let (mi, mj) = (means[i], means[j]);
    let (mut num, mut di2, mut dj2) = (0.0_f64, 0.0_f64, 0.0_f64);
    for row in data {
        let (di, dj) = (row[i] - mi, row[j] - mj);
        num += di * dj;
        di2 += di * di;
        dj2 += dj * dj;
    }
    let denom = (di2 * dj2).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        (num / denom).clamp(-1.0, 1.0)
    }
}

fn fisher_z(r: f64, n: usize) -> f64 {
    let r = r.clamp(-0.9999, 0.9999);
    0.5 * ((1.0 + r) / (1.0 - r)).ln() * ((n as f64 - 3.0).max(1.0)).sqrt()
}

/// Rational approximation of the probit function (Beasley-Springer-Moro).
fn normal_quantile_approx(p: f64) -> f64 {
    let p = p.clamp(1e-10, 1.0 - 1e-10);
    // Coefficients for central region
    let a = [
        -3.969_683_028_665_376e1_f64,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    let b = [
        -5.447_609_879_822_406e1_f64,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    let c = [
        -7.784_894_002_430_293e-3_f64,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    let d = [
        7.784_695_709_041_462e-3_f64,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let (p_low, p_high) = (0.02425, 1.0 - 0.02425);
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::SeedableRng;

    // ── Differentiable Logic ──────────────────────────────────────────────────

    #[test]
    fn test_fuzzy_tnorm_bounds() {
        for &(a, b) in &[
            (0.3_f64, 0.9),
            (0.7, 0.7),
            (0.1, 0.1),
            (1.0, 1.0),
            (0.0, 0.0),
        ] {
            let v = FuzzyLogic::tnorm(a, b);
            assert!((0.0..=1.0).contains(&v), "tnorm out of [0,1]: {v}");
        }
        assert!((FuzzyLogic::tnorm(0.8, 0.9) - 0.7).abs() < 1e-12);
    }

    #[test]
    fn test_fuzzy_tconorm_bounds() {
        for &(a, b) in &[(0.3_f64, 0.4), (0.8, 0.6), (0.0, 0.0), (1.0, 1.0)] {
            let v = FuzzyLogic::tconorm(a, b);
            assert!((0.0..=1.0).contains(&v), "tconorm out of [0,1]: {v}");
        }
    }

    #[test]
    fn test_fuzzy_negation() {
        assert!((FuzzyLogic::negation(0.0) - 1.0).abs() < 1e-12);
        assert!((FuzzyLogic::negation(1.0)).abs() < 1e-12);
        assert!((FuzzyLogic::negation(0.4) - 0.6).abs() < 1e-12);
    }

    #[test]
    fn test_fuzzy_implication() {
        assert!((FuzzyLogic::implication(0.3, 0.7) - 1.0).abs() < 1e-12);
        let v = FuzzyLogic::implication(0.8, 0.3);
        assert!((0.0..1.0).contains(&v));
    }

    #[test]
    fn test_product_tnorm() {
        assert!((ProductFuzzy::tnorm(0.4, 0.5) - 0.2).abs() < 1e-12);
        assert!((ProductFuzzy::tconorm(0.4, 0.5) - 0.7).abs() < 1e-12);
    }

    #[test]
    fn test_product_implication_zero_denom() {
        let v = ProductFuzzy::implication(0.0, 0.5);
        assert!((v - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_unify_identical() {
        let p = vec![1.0, 0.0, 0.0];
        assert!((NeuralTheoremProver::unify(&p, &p) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_unify_orthogonal() {
        let p = vec![1.0, 0.0];
        let q = vec![0.0, 1.0];
        assert!((NeuralTheoremProver::unify(&p, &q) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_differentiable_forward_chain() {
        let r = DifferentiableForwardChaining::apply_rule(&[0.8, 0.9], &[1.0, 1.0]);
        assert!((0.0..=1.0).contains(&r), "result out of [0,1]: {r}");
    }

    #[test]
    fn test_forward_chain_empty() {
        assert_eq!(DifferentiableForwardChaining::apply_rule(&[], &[]), 0.0);
    }

    #[test]
    fn test_ltn_grounding_shape() {
        let pred = vec![1.0, -0.5, 0.3];
        let args = vec![
            vec![0.2, 0.8, -0.1],
            vec![-0.5, 0.3, 0.7],
            vec![1.0, 0.0, 0.0],
        ];
        let ts = LogicTensorNetwork::grounding(&pred, &args);
        assert_eq!(ts.len(), 3);
        for t in &ts {
            assert!(*t > 0.0 && *t < 1.0, "LTN truth not in (0,1): {t}");
        }
    }

    #[test]
    fn test_ltn_grounding_empty() {
        assert!(LogicTensorNetwork::grounding(&[1.0], &[]).is_empty());
    }

    // ── Neural Program Synthesis ──────────────────────────────────────────────

    #[test]
    fn test_program_token_eval() {
        let ast = ProgramAst::BinOp(
            ProgramToken::Add,
            Box::new(ProgramAst::Constant(3.0)),
            Box::new(ProgramAst::Constant(2.0)),
        );
        assert!((ast.evaluate(&mut vec![]) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_program_token_var() {
        let ast = ProgramAst::Variable(0);
        assert!((ast.evaluate(&mut vec![42.0]) - 42.0).abs() < 1e-12);
    }

    #[test]
    fn test_program_if_then_else() {
        let ast = ProgramAst::IfThenElse(
            Box::new(ProgramAst::Constant(1.0)),
            Box::new(ProgramAst::Constant(10.0)),
            Box::new(ProgramAst::Constant(20.0)),
        );
        assert!((ast.evaluate(&mut vec![]) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_program_embedding_shape() {
        let pe = ProgramEmbedding::new(12, 16, 42);
        let emb = pe.embed(&[0usize, 2, 4, 1]);
        assert_eq!(emb.len(), 16);
        assert!(emb.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_program_embedding_deterministic() {
        let pe = ProgramEmbedding::new(10, 8, 7);
        assert_eq!(pe.embed(&[0, 1, 2]), pe.embed(&[0, 1, 2]));
    }

    #[test]
    fn test_execution_trace_length() {
        let prog = vec![
            ProgramToken::Const(3.0),
            ProgramToken::Const(4.0),
            ProgramToken::Add,
        ];
        assert_eq!(ExecutionTracing::trace(&prog, &[]).len(), 3);
    }

    #[test]
    fn test_execution_trace_values() {
        let prog = vec![ProgramToken::Var(0), ProgramToken::Const(2.0)];
        let trace = ExecutionTracing::trace(&prog, &[7.0]);
        assert!((trace[0].1 - 7.0).abs() < 1e-12);
        assert!((trace[1].1 - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_neural_searcher_construction() {
        let s = NeuralProgramSearcher::new(8, 42);
        assert!(!s.vocabulary.is_empty());
        assert_eq!(s.token_embeddings.len(), s.vocabulary.len());
        assert_eq!(s.token_embeddings[0].len(), 8);
    }

    #[test]
    fn test_neural_searcher_returns_tokens() {
        let searcher = NeuralProgramSearcher::new(4, 99);
        let pairs: Vec<(&[f64], f64)> = vec![(&[1.0], 2.0), (&[2.0], 3.0)];
        let mut rng = StdRng::seed_from_u64(1);
        assert!(!searcher.search(&pairs, 3, 4, &mut rng).is_empty());
    }

    // ── Knowledge Graph Embeddings ────────────────────────────────────────────

    #[test]
    fn test_transr_score_diff_positive() {
        let model = TransRModel::new(4, 3);
        let h = vec![1.0, 0.0, 0.0, 0.0];
        let r = vec![0.1, 0.0, 0.0];
        let t = vec![1.0, 0.0, 0.0, 0.0];
        let m_r = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        assert!(model.score(&h, &r, &t, &m_r).expect("score failed") <= 0.0);
    }

    #[test]
    fn test_transr_score_far_pair() {
        let model = TransRModel::new(2, 2);
        let h = vec![1.0, 0.0];
        let t = vec![0.0, 1.0];
        let r = vec![0.0, 0.0];
        let m_r = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!(model.score(&h, &r, &h, &m_r).expect("score failed") > model.score(&h, &r, &t, &m_r).expect("score failed"));
    }

    #[test]
    fn test_distmult_symmetric() {
        let model = DistMultModel::new(3);
        let h = vec![0.5, -0.3, 0.8];
        let r = vec![1.0, 1.0, 1.0];
        let t = vec![0.2, 0.7, -0.1];
        let s1 = model.score(&h, &r, &t).expect("score failed");
        let s2 = model.score(&t, &r, &h).expect("score failed");
        assert!((s1 - s2).abs() < 1e-12, "DistMult should be symmetric");
    }

    #[test]
    fn test_distmult_dimension_mismatch() {
        assert!(DistMultModel::new(3)
            .score(&[1.0, 2.0], &[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0])
            .is_err());
    }

    #[test]
    fn test_analogy_reasoning() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let c = vec![1.0, 0.0];
        let cands = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
        assert_eq!(AnalygyReasoning::solve_analogy(&a, &b, &c, &cands), 1);
    }

    #[test]
    fn test_path_query_shape() {
        let pqe = PathQueryEmbedding::new(4);
        let rels = vec![vec![1.0, 0.0, -1.0, 0.5], vec![0.5, -0.5, 0.0, 1.0]];
        let emb = pqe.embed_path(&rels).expect("embed_path failed");
        assert_eq!(emb.len(), 4);
    }

    #[test]
    fn test_path_query_empty() {
        assert_eq!(
            PathQueryEmbedding::new(3).embed_path(&[]).expect("embed_path failed"),
            vec![0.0; 3]
        );
    }

    #[test]
    fn test_rule_induction_confidence() {
        let triples = vec![
            (0, 0, 1),
            (1, 0, 2),
            (2, 0, 3),
            (0, 1, 1),
            (1, 1, 2),
            (2, 1, 3),
        ];
        let rules = RuleInduction::mine_rules(&triples, 0.5);
        assert!(!rules.is_empty());
        for rule in &rules {
            assert!(rule.confidence >= 0.5 && rule.confidence <= 1.0);
        }
    }

    #[test]
    fn test_rule_induction_depth2() {
        let triples = vec![
            (0, 0, 1),
            (1, 0, 2),
            (1, 1, 3),
            (2, 1, 4),
            (0, 2, 3),
            (1, 2, 4),
        ];
        let rules = RuleInduction::mine_rules(&triples, 0.4);
        assert!(!rules.is_empty());
    }

    // ── CSP ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_csp_variable_domain() {
        let var = CspVariable::new("x", vec![1, 2, 3, 4]);
        assert_eq!(var.domain.len(), 4);
        assert!(!var.is_assigned());
        let a = CspVariable::new("y", vec![5]);
        assert!(a.is_assigned());
        assert_eq!(a.assigned_value(), Some(5));
    }

    #[test]
    fn test_binary_constraint() {
        let c = BinaryCspConstraint::new(0, 1, BinaryConstraintType::NotEqual);
        assert!(c.check(1, 2));
        assert!(!c.check(1, 1));
        let lt = BinaryCspConstraint::new(0, 1, BinaryConstraintType::LessThan);
        assert!(lt.check(1, 5));
        assert!(!lt.check(5, 1));
    }

    #[test]
    fn test_arc_consistency_prune() {
        let mut vars = vec![
            CspVariable::new("x", vec![1, 2, 3]),
            CspVariable::new("y", vec![1, 2, 3]),
        ];
        let cs = vec![BinaryCspConstraint::new(
            0,
            1,
            BinaryConstraintType::LessThan,
        )];
        assert!(ArcConsistency::enforce(&mut vars, &cs));
        assert!(!vars[0].domain.contains(&3));
        assert!(!vars[1].domain.contains(&1));
    }

    #[test]
    fn test_arc_consistency_unsatisfiable() {
        let mut vars = vec![
            CspVariable::new("x", vec![5]),
            CspVariable::new("y", vec![5]),
        ];
        let cs = vec![BinaryCspConstraint::new(
            0,
            1,
            BinaryConstraintType::LessThan,
        )];
        assert!(!ArcConsistency::enforce(&mut vars, &cs));
    }

    #[test]
    fn test_gnn_csp_shape() {
        let solver = GnnCspSolver::new(2, 8);
        let vf = vec![vec![1.0_f64, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
        let cf = vec![vec![0.1_f64, 0.9]];
        let assignments = solver.forward(&vf, &cf, 4);
        assert_eq!(assignments.len(), 3);
        for row in &assignments {
            assert_eq!(row.len(), 4);
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_satisfiability_loss_bounds() {
        let a = vec![0.8, 0.2, 0.9];
        let clauses = vec![vec![(0usize, true), (1usize, false)]];
        let loss = SatisfiabilityLoss::compute(&a, &clauses);
        assert!((0.0..=1.0).contains(&loss), "SAT loss out of [0,1]: {loss}");
    }

    #[test]
    fn test_satisfiability_loss_fully_satisfied() {
        let loss = SatisfiabilityLoss::compute(&[1.0, 0.0], &[vec![(0usize, true)]]);
        assert!(loss < 1e-12);
    }

    #[test]
    fn test_satisfiability_loss_fully_violated() {
        let loss = SatisfiabilityLoss::compute(&[0.0], &[vec![(0usize, true)]]);
        assert!((loss - 1.0).abs() < 1e-12);
    }

    // ── Causal Reasoning ─────────────────────────────────────────────────────

    #[test]
    fn test_scm_query() {
        let mut scm = StructuralCausalModel::new();
        scm.add_variable("X", vec![], Box::new(|_| 2.0));
        scm.add_variable(
            "Y",
            vec!["X".to_string()],
            Box::new(|e| e.get("X").copied().unwrap_or(0.0) * 3.0),
        );
        assert!((scm.query("Y", &HashMap::new()) - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_scm_intervention() {
        let mut scm = StructuralCausalModel::new();
        scm.add_variable("X", vec![], Box::new(|_| 2.0));
        scm.add_variable(
            "Y",
            vec!["X".to_string()],
            Box::new(|e| e.get("X").copied().unwrap_or(0.0) * 3.0),
        );
        scm.intervene("X", 10.0);
        assert!((scm.query("Y", &HashMap::new()) - 30.0).abs() < 1e-12);
    }

    #[test]
    fn test_scm_evidence_override() {
        let mut scm = StructuralCausalModel::new();
        scm.add_variable("X", vec![], Box::new(|_| 5.0));
        let mut ev = HashMap::new();
        ev.insert("X".to_string(), 99.0);
        assert!((scm.query("X", &ev) - 99.0).abs() < 1e-12);
    }

    #[test]
    fn test_counterfactual_estimator() {
        // Factual: X=1, Y=3. CF: X=2 → U=2, Y_cf=2+2=4
        let cf = CounterfactualEstimator::new(1.0).estimate(1.0, 2.0, 3.0);
        assert!((cf - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_counterfactual_linearity() {
        let cf = CounterfactualEstimator::new(0.5).estimate(2.0, 5.0, 7.0);
        assert!((cf - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_causal_discovery_edges() {
        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<Vec<f64>> = (0..200)
            .map(|_| {
                let x = rng.random::<f64>() * 2.0 - 1.0;
                let y = x * 0.9 + rng.random::<f64>() * 0.1;
                let z = rng.random::<f64>() * 2.0 - 1.0;
                vec![x, y, z]
            })
            .collect();
        let edges = CausalDiscovery::new(0.05).discover(&data, 0.05);
        assert!(
            edges.contains(&(0, 1)) || edges.contains(&(1, 0)),
            "X-Y edge should be discovered"
        );
    }

    #[test]
    fn test_causal_discovery_no_data() {
        assert!(CausalDiscovery::new(0.05).discover(&[], 0.05).is_empty());
    }

    #[test]
    fn test_backdoor_adjustment() {
        let adj = NeuralBackdoorAdjustment::new(3, 1, 42);
        let r = adj.adjust(1.0, &[0.5, -0.2, 0.1], 0.8);
        assert!(r.is_finite() && r > 0.0);
    }

    #[test]
    fn test_backdoor_adjustment_treated_vs_control() {
        let adj = NeuralBackdoorAdjustment::new(2, 1, 7);
        let z = vec![0.3, 0.7];
        assert!((adj.adjust(1.0, &z, 0.5) - adj.adjust(0.0, &z, 0.5)).abs() > 1e-9);
    }

    #[test]
    fn test_intervention_layer_apply() {
        let mut layer = InterventionLayer::new();
        layer.set_intervention(1, 99.0);
        let result = layer.apply(&[1.0, 2.0, 3.0]);
        assert!((result[0] - 1.0).abs() < 1e-12);
        assert!((result[1] - 99.0).abs() < 1e-12);
        assert!((result[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_intervention_layer_empty() {
        let layer = InterventionLayer::new();
        let acts = vec![5.0, 6.0];
        assert_eq!(layer.apply(&acts), acts);
    }

    // ── Additional edge-case tests ────────────────────────────────────────────

    #[test]
    fn test_forward_chaining_chain_steps() {
        let fc = DifferentiableForwardChaining::new(3);
        let states = fc.chain(&[0.9, 0.8], &[0.5, 0.5]);
        assert_eq!(states.len(), 3);
        assert!(states.iter().all(|&s| (0.0..=1.0).contains(&s)));
    }

    #[test]
    fn test_neural_prover_register_prove() {
        let mut prover = NeuralTheoremProver::new(3);
        prover.register(0, vec![1.0, 0.0, 0.0]).expect("register failed");
        prover.register(1, vec![1.0, 0.0, 0.0]).expect("register failed");
        assert!((prover.prove_chain(&[(0, 1)]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_program_loop_node() {
        let ast = ProgramAst::LoopNode(3, Box::new(ProgramAst::Constant(5.0)));
        assert!((ast.evaluate(&mut vec![]) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_program_sequence() {
        let ast = ProgramAst::Sequence(vec![ProgramAst::Constant(1.0), ProgramAst::Constant(7.0)]);
        assert!((ast.evaluate(&mut vec![]) - 7.0).abs() < 1e-12);
    }
}
