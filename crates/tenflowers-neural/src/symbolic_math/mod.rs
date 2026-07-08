//! Symbolic Mathematics and Neural-Symbolic Computation
//!
//! This module provides a comprehensive suite of symbolic math tools:
//! - Expression trees with eval/simplify/differentiate/integrate
//! - Neural expression synthesizer (beam search over token sequences)
//! - Gradient-based symbolic regressor (sparse LASSO over expression library)
//! - Genetic-programming formula search
//! - Equation balancer via integer null-space
//! - Dimensional analysis (Buckingham Pi theorem)
//! - Matrix calculus automatic derivative rules
//! - Exact polynomial arithmetic (division, GCD, companion matrix roots)
//! - Pattern-based symbolic integrator
//! - Automated theorem proving (ProofState, TacticEngine, NeuralTacticSelector)
//! - Equation discovery (SrSymbolicRegressor, EquationDatabase)

pub mod extensions;
pub use extensions::*;
pub mod advanced;
pub use advanced::*;
#[cfg(test)]
pub mod tests;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// 1. SymbolicExpr — expression tree
// ────────────────────────────────────────────────────────────────────────────

/// An expression tree node.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Numeric constant
    Const(f64),
    /// Named variable
    Var(String),
    /// Addition
    Add(Box<Expr>, Box<Expr>),
    /// Subtraction
    Sub(Box<Expr>, Box<Expr>),
    /// Multiplication
    Mul(Box<Expr>, Box<Expr>),
    /// Division
    Div(Box<Expr>, Box<Expr>),
    /// Exponentiation
    Pow(Box<Expr>, Box<Expr>),
    /// Negation
    Neg(Box<Expr>),
    /// Sine
    Sin(Box<Expr>),
    /// Cosine
    Cos(Box<Expr>),
    /// Natural exponential
    Exp(Box<Expr>),
    /// Natural logarithm
    Ln(Box<Expr>),
    /// Square root
    Sqrt(Box<Expr>),
}

impl Expr {
    fn is_const(&self, c: f64) -> bool {
        matches!(self, Expr::Const(v) if (*v - c).abs() < 1e-12)
    }

    fn const_val(&self) -> Option<f64> {
        if let Expr::Const(v) = self {
            Some(*v)
        } else {
            None
        }
    }
}

/// Evaluate an expression with variable bindings.
pub fn eval(expr: &Expr, vars: &HashMap<String, f64>) -> Result<f64, String> {
    match expr {
        Expr::Const(c) => Ok(*c),
        Expr::Var(name) => vars
            .get(name)
            .copied()
            .ok_or_else(|| format!("undefined variable: {name}")),
        Expr::Add(l, r) => Ok(eval(l, vars)? + eval(r, vars)?),
        Expr::Sub(l, r) => Ok(eval(l, vars)? - eval(r, vars)?),
        Expr::Mul(l, r) => Ok(eval(l, vars)? * eval(r, vars)?),
        Expr::Div(l, r) => {
            let d = eval(r, vars)?;
            if d.abs() < 1e-300 {
                Err("division by zero".to_string())
            } else {
                Ok(eval(l, vars)? / d)
            }
        }
        Expr::Pow(base, exp) => Ok(eval(base, vars)?.powf(eval(exp, vars)?)),
        Expr::Neg(e) => Ok(-eval(e, vars)?),
        Expr::Sin(e) => Ok(eval(e, vars)?.sin()),
        Expr::Cos(e) => Ok(eval(e, vars)?.cos()),
        Expr::Exp(e) => Ok(eval(e, vars)?.exp()),
        Expr::Ln(e) => {
            let v = eval(e, vars)?;
            if v <= 0.0 {
                Err(format!("ln of non-positive value: {v}"))
            } else {
                Ok(v.ln())
            }
        }
        Expr::Sqrt(e) => {
            let v = eval(e, vars)?;
            if v < 0.0 {
                Err(format!("sqrt of negative value: {v}"))
            } else {
                Ok(v.sqrt())
            }
        }
    }
}

/// Constant-folding simplification with algebraic identities.
pub fn simplify(expr: Expr) -> Expr {
    match expr {
        Expr::Const(c) => Expr::Const(c),
        Expr::Var(v) => Expr::Var(v),
        Expr::Neg(inner) => {
            let s = simplify(*inner);
            if let Expr::Const(c) = &s {
                Expr::Const(-c)
            } else {
                Expr::Neg(Box::new(s))
            }
        }
        Expr::Add(l, r) => {
            let ls = simplify(*l);
            let rs = simplify(*r);
            match (&ls, &rs) {
                (Expr::Const(a), Expr::Const(b)) => Expr::Const(a + b),
                _ if ls.is_const(0.0) => rs,
                _ if rs.is_const(0.0) => ls,
                _ => Expr::Add(Box::new(ls), Box::new(rs)),
            }
        }
        Expr::Sub(l, r) => {
            let ls = simplify(*l);
            let rs = simplify(*r);
            match (&ls, &rs) {
                (Expr::Const(a), Expr::Const(b)) => Expr::Const(a - b),
                _ if rs.is_const(0.0) => ls,
                _ if ls == rs => Expr::Const(0.0),
                _ => Expr::Sub(Box::new(ls), Box::new(rs)),
            }
        }
        Expr::Mul(l, r) => {
            let ls = simplify(*l);
            let rs = simplify(*r);
            match (&ls, &rs) {
                (Expr::Const(a), Expr::Const(b)) => Expr::Const(a * b),
                _ if ls.is_const(0.0) || rs.is_const(0.0) => Expr::Const(0.0),
                _ if ls.is_const(1.0) => rs,
                _ if rs.is_const(1.0) => ls,
                _ => Expr::Mul(Box::new(ls), Box::new(rs)),
            }
        }
        Expr::Div(l, r) => {
            let ls = simplify(*l);
            let rs = simplify(*r);
            match (&ls, &rs) {
                (Expr::Const(a), Expr::Const(b)) if b.abs() > 1e-300 => Expr::Const(a / b),
                _ if rs.is_const(1.0) => ls,
                _ if ls == rs => Expr::Const(1.0),
                _ => Expr::Div(Box::new(ls), Box::new(rs)),
            }
        }
        Expr::Pow(base, exp) => {
            let bs = simplify(*base);
            let es = simplify(*exp);
            match (&bs, &es) {
                (Expr::Const(a), Expr::Const(b)) => Expr::Const(a.powf(*b)),
                _ if es.is_const(0.0) => Expr::Const(1.0),
                _ if es.is_const(1.0) => bs,
                _ if bs.is_const(1.0) => Expr::Const(1.0),
                _ => Expr::Pow(Box::new(bs), Box::new(es)),
            }
        }
        Expr::Sin(e) => {
            let s = simplify(*e);
            if let Expr::Const(c) = &s {
                Expr::Const(c.sin())
            } else {
                Expr::Sin(Box::new(s))
            }
        }
        Expr::Cos(e) => {
            let s = simplify(*e);
            if let Expr::Const(c) = &s {
                Expr::Const(c.cos())
            } else {
                Expr::Cos(Box::new(s))
            }
        }
        Expr::Exp(e) => {
            let s = simplify(*e);
            if let Expr::Const(c) = &s {
                Expr::Const(c.exp())
            } else {
                Expr::Exp(Box::new(s))
            }
        }
        Expr::Ln(e) => {
            let s = simplify(*e);
            if let Expr::Const(c) = &s {
                Expr::Const(c.ln())
            } else {
                Expr::Ln(Box::new(s))
            }
        }
        Expr::Sqrt(e) => {
            let s = simplify(*e);
            if let Expr::Const(c) = &s {
                Expr::Const(c.sqrt())
            } else {
                Expr::Sqrt(Box::new(s))
            }
        }
    }
}

/// Symbolic differentiation using the chain rule.
pub fn derivative(expr: &Expr, var: &str) -> Expr {
    match expr {
        Expr::Const(_) => Expr::Const(0.0),
        Expr::Var(v) => {
            if v == var {
                Expr::Const(1.0)
            } else {
                Expr::Const(0.0)
            }
        }
        Expr::Neg(e) => Expr::Neg(Box::new(derivative(e, var))),
        Expr::Add(l, r) => Expr::Add(Box::new(derivative(l, var)), Box::new(derivative(r, var))),
        Expr::Sub(l, r) => Expr::Sub(Box::new(derivative(l, var)), Box::new(derivative(r, var))),
        // Product rule: (f*g)' = f'*g + f*g'
        Expr::Mul(f, g) => {
            let df = derivative(f, var);
            let dg = derivative(g, var);
            Expr::Add(
                Box::new(Expr::Mul(Box::new(df), g.clone())),
                Box::new(Expr::Mul(f.clone(), Box::new(dg))),
            )
        }
        // Quotient rule: (f/g)' = (f'g - fg') / g^2
        Expr::Div(f, g) => {
            let df = derivative(f, var);
            let dg = derivative(g, var);
            let num = Expr::Sub(
                Box::new(Expr::Mul(Box::new(df), g.clone())),
                Box::new(Expr::Mul(f.clone(), Box::new(dg))),
            );
            let den = Expr::Pow(g.clone(), Box::new(Expr::Const(2.0)));
            Expr::Div(Box::new(num), Box::new(den))
        }
        // Power rule: d/dx [f^g] where g is constant = g * f^(g-1) * f'
        Expr::Pow(base, exp) => {
            if let Some(n) = exp.const_val() {
                let df = derivative(base, var);
                let new_exp = Expr::Const(n - 1.0);
                simplify(Expr::Mul(
                    Box::new(Expr::Mul(
                        Box::new(Expr::Const(n)),
                        Box::new(Expr::Pow(base.clone(), Box::new(new_exp))),
                    )),
                    Box::new(df),
                ))
            } else {
                // General: d/dx [f^g] = f^g * (g' * ln(f) + g * f'/f)
                let ln_f = Expr::Ln(base.clone());
                let dg = derivative(exp, var);
                let df = derivative(base, var);
                let term1 = Expr::Mul(Box::new(dg), Box::new(ln_f));
                let term2 = Expr::Mul(exp.clone(), Box::new(Expr::Div(Box::new(df), base.clone())));
                let sum = Expr::Add(Box::new(term1), Box::new(term2));
                simplify(Expr::Mul(Box::new(expr.clone()), Box::new(sum)))
            }
        }
        // Chain rule for unary ops
        Expr::Sin(e) => {
            let inner = derivative(e, var);
            simplify(Expr::Mul(Box::new(Expr::Cos(e.clone())), Box::new(inner)))
        }
        Expr::Cos(e) => {
            let inner = derivative(e, var);
            simplify(Expr::Mul(
                Box::new(Expr::Neg(Box::new(Expr::Sin(e.clone())))),
                Box::new(inner),
            ))
        }
        Expr::Exp(e) => {
            let inner = derivative(e, var);
            simplify(Expr::Mul(Box::new(expr.clone()), Box::new(inner)))
        }
        Expr::Ln(e) => {
            let inner = derivative(e, var);
            simplify(Expr::Div(Box::new(inner), e.clone()))
        }
        Expr::Sqrt(e) => {
            // d/dx sqrt(f) = f' / (2*sqrt(f))
            let inner = derivative(e, var);
            simplify(Expr::Div(
                Box::new(inner),
                Box::new(Expr::Mul(
                    Box::new(Expr::Const(2.0)),
                    Box::new(expr.clone()),
                )),
            ))
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 2. ExpressionHasher — canonical structural hashing
// ────────────────────────────────────────────────────────────────────────────

/// FNV-1a structural hasher for `Expr`.
pub struct ExpressionHasher;

const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

fn fnv_mix(hash: u64, byte: u8) -> u64 {
    (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
}

fn fnv_mix_u64(hash: u64, v: u64) -> u64 {
    let bytes = v.to_le_bytes();
    let mut h = hash;
    for b in bytes {
        h = fnv_mix(h, b);
    }
    h
}

fn fnv_mix_f64(hash: u64, v: f64) -> u64 {
    fnv_mix_u64(hash, v.to_bits())
}

impl ExpressionHasher {
    /// Compute a canonical FNV-1a hash of the expression structure.
    pub fn hash_expr(expr: &Expr) -> u64 {
        let mut h = FNV_OFFSET;
        Self::hash_rec(expr, &mut h);
        h
    }

    fn hash_rec(expr: &Expr, h: &mut u64) {
        // Use a tag byte per variant
        match expr {
            Expr::Const(c) => {
                *h = fnv_mix(*h, 0x01);
                *h = fnv_mix_f64(*h, *c);
            }
            Expr::Var(v) => {
                *h = fnv_mix(*h, 0x02);
                for b in v.bytes() {
                    *h = fnv_mix(*h, b);
                }
            }
            Expr::Add(l, r) => {
                *h = fnv_mix(*h, 0x03);
                Self::hash_rec(l, h);
                Self::hash_rec(r, h);
            }
            Expr::Sub(l, r) => {
                *h = fnv_mix(*h, 0x04);
                Self::hash_rec(l, h);
                Self::hash_rec(r, h);
            }
            Expr::Mul(l, r) => {
                *h = fnv_mix(*h, 0x05);
                Self::hash_rec(l, h);
                Self::hash_rec(r, h);
            }
            Expr::Div(l, r) => {
                *h = fnv_mix(*h, 0x06);
                Self::hash_rec(l, h);
                Self::hash_rec(r, h);
            }
            Expr::Pow(b, e) => {
                *h = fnv_mix(*h, 0x07);
                Self::hash_rec(b, h);
                Self::hash_rec(e, h);
            }
            Expr::Neg(e) => {
                *h = fnv_mix(*h, 0x08);
                Self::hash_rec(e, h);
            }
            Expr::Sin(e) => {
                *h = fnv_mix(*h, 0x09);
                Self::hash_rec(e, h);
            }
            Expr::Cos(e) => {
                *h = fnv_mix(*h, 0x0A);
                Self::hash_rec(e, h);
            }
            Expr::Exp(e) => {
                *h = fnv_mix(*h, 0x0B);
                Self::hash_rec(e, h);
            }
            Expr::Ln(e) => {
                *h = fnv_mix(*h, 0x0C);
                Self::hash_rec(e, h);
            }
            Expr::Sqrt(e) => {
                *h = fnv_mix(*h, 0x0D);
                Self::hash_rec(e, h);
            }
        }
    }

    /// Structural equality (mirrors `PartialEq` but callable on references).
    pub fn exprs_equal(a: &Expr, b: &Expr) -> bool {
        a == b
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 3. NeuralExpressionSynthesizer
// ────────────────────────────────────────────────────────────────────────────

/// Tokens for expression serialization (postfix-like).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExprToken {
    /// Numeric constant (encoded as index into constant table)
    Const(u32),
    /// Variable index
    Var(u32),
    Add,
    Sub,
    Mul,
    Div,
    Sin,
    Cos,
    Exp,
    Ln,
    Sqrt,
    LParen,
    RParen,
    End,
}

/// Configuration for the expression synthesizer.
#[derive(Debug, Clone)]
pub struct SynthesizerConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub hidden_dim: usize,
    pub max_len: usize,
}

impl Default for SynthesizerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 20,
            embed_dim: 64,
            hidden_dim: 128,
            max_len: 32,
        }
    }
}

/// Beam-search hypothesis for expression synthesis.
#[derive(Clone)]
struct BeamHypothesis {
    tokens: Vec<ExprToken>,
    log_prob: f64,
}

/// Neural expression synthesizer with beam-search decoder.
pub struct NeuralExpressionSynthesizer {
    pub config: SynthesizerConfig,
    /// Embedding matrix [vocab_size × embed_dim] stored row-major.
    embeddings: Vec<f64>,
    /// Simple linear projection [embed_dim → vocab_size].
    proj: Vec<f64>,
    /// Constant table used during token generation.
    const_table: Vec<f64>,
    /// Variable names.
    var_names: Vec<String>,
}

impl NeuralExpressionSynthesizer {
    /// Create a new synthesizer with random weights.
    pub fn new(config: SynthesizerConfig, var_names: Vec<String>, rng: &mut StdRng) -> Self {
        let vocab_size = config.vocab_size;
        let embed_dim = config.embed_dim;
        let embeddings: Vec<f64> = (0..vocab_size * embed_dim)
            .map(|_| (rng.random::<f64>() - 0.5) * 0.1)
            .collect();
        let proj: Vec<f64> = (0..embed_dim * vocab_size)
            .map(|_| (rng.random::<f64>() - 0.5) * 0.1)
            .collect();
        let const_table = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0];
        Self {
            config,
            embeddings,
            proj,
            const_table,
            var_names,
        }
    }

    /// Score a token index given a context embedding.
    fn score(&self, ctx: &[f64], token_idx: usize) -> f64 {
        let embed_dim = self.config.embed_dim;
        let vocab_size = self.config.vocab_size;
        let offset = token_idx * embed_dim;
        if offset + embed_dim > self.proj.len() {
            return f64::NEG_INFINITY;
        }
        let mut s = 0.0_f64;
        for i in 0..embed_dim.min(ctx.len()) {
            s += ctx[i] * self.proj[offset + i];
        }
        // softmax is applied externally; return raw logit
        let _ = vocab_size;
        s
    }

    /// Compute context embedding from token sequence (mean-pool embeddings).
    fn context_embed(&self, tokens: &[ExprToken]) -> Vec<f64> {
        let embed_dim = self.config.embed_dim;
        let vocab_size = self.config.vocab_size;
        let mut ctx = vec![0.0f64; embed_dim];
        if tokens.is_empty() {
            return ctx;
        }
        for tok in tokens {
            let idx = self.token_to_idx(tok).min(vocab_size - 1);
            let base = idx * embed_dim;
            if base + embed_dim <= self.embeddings.len() {
                for i in 0..embed_dim {
                    ctx[i] += self.embeddings[base + i];
                }
            }
        }
        let n = tokens.len() as f64;
        for v in &mut ctx {
            *v /= n;
        }
        ctx
    }

    fn token_to_idx(&self, tok: &ExprToken) -> usize {
        match tok {
            ExprToken::Const(i) => (*i as usize).min(6),
            ExprToken::Var(i) => 7 + (*i as usize).min(3),
            ExprToken::Add => 11,
            ExprToken::Sub => 12,
            ExprToken::Mul => 13,
            ExprToken::Div => 14,
            ExprToken::Sin => 15,
            ExprToken::Cos => 16,
            ExprToken::Exp => 17,
            ExprToken::Ln => 18,
            ExprToken::Sqrt => 19,
            ExprToken::LParen => 10,
            ExprToken::RParen => 9,
            ExprToken::End => 8,
        }
    }

    /// Beam-search decode a token sequence of length ≤ max_len.
    pub fn beam_decode(&self, beam_width: usize) -> Vec<ExprToken> {
        let max_len = self.config.max_len;
        let mut beams: Vec<BeamHypothesis> = vec![BeamHypothesis {
            tokens: vec![],
            log_prob: 0.0,
        }];

        for _step in 0..max_len {
            let mut candidates: Vec<BeamHypothesis> = Vec::new();
            for hyp in &beams {
                if hyp.tokens.last() == Some(&ExprToken::End) {
                    candidates.push(hyp.clone());
                    continue;
                }
                let ctx = self.context_embed(&hyp.tokens);
                // Score all vocab positions and take top beam_width
                let vocab_size = self.config.vocab_size;
                let mut scores: Vec<(usize, f64)> =
                    (0..vocab_size).map(|i| (i, self.score(&ctx, i))).collect();
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                // log-softmax
                let max_s = scores[0].1;
                let sum_exp: f64 = scores.iter().map(|(_, s)| (s - max_s).exp()).sum();
                let log_z = max_s + sum_exp.ln();
                for &(tok_idx, logit) in scores.iter().take(beam_width) {
                    let log_prob = hyp.log_prob + (logit - log_z);
                    let tok = self.idx_to_token(tok_idx);
                    let mut new_tokens = hyp.tokens.clone();
                    new_tokens.push(tok);
                    candidates.push(BeamHypothesis {
                        tokens: new_tokens,
                        log_prob,
                    });
                }
            }
            candidates.sort_by(|a, b| {
                b.log_prob
                    .partial_cmp(&a.log_prob)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            beams = candidates.into_iter().take(beam_width).collect();
        }
        beams
            .into_iter()
            .next()
            .map(|h| h.tokens)
            .unwrap_or_default()
    }

    fn idx_to_token(&self, idx: usize) -> ExprToken {
        match idx {
            0..=6 => ExprToken::Const(idx as u32),
            7..=10 => ExprToken::Var((idx - 7) as u32),
            11 => ExprToken::Add,
            12 => ExprToken::Sub,
            13 => ExprToken::Mul,
            14 => ExprToken::Div,
            15 => ExprToken::Sin,
            16 => ExprToken::Cos,
            17 => ExprToken::Exp,
            18 => ExprToken::Ln,
            19 => ExprToken::Sqrt,
            _ => ExprToken::End,
        }
    }
}

/// Parse a postfix token sequence into an `Expr`.
pub fn parse_token_sequence(
    tokens: &[ExprToken],
    const_table: &[f64],
    var_names: &[String],
) -> Result<Expr, String> {
    let mut stack: Vec<Expr> = Vec::new();
    for tok in tokens {
        match tok {
            ExprToken::End => break,
            ExprToken::Const(i) => {
                let idx = *i as usize;
                let c = const_table.get(idx).copied().unwrap_or(0.0);
                stack.push(Expr::Const(c));
            }
            ExprToken::Var(i) => {
                let idx = *i as usize;
                let name = var_names
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| format!("x{idx}"));
                stack.push(Expr::Var(name));
            }
            ExprToken::Add => {
                let r = stack.pop().ok_or("stack underflow: Add")?;
                let l = stack.pop().ok_or("stack underflow: Add")?;
                stack.push(Expr::Add(Box::new(l), Box::new(r)));
            }
            ExprToken::Sub => {
                let r = stack.pop().ok_or("stack underflow: Sub")?;
                let l = stack.pop().ok_or("stack underflow: Sub")?;
                stack.push(Expr::Sub(Box::new(l), Box::new(r)));
            }
            ExprToken::Mul => {
                let r = stack.pop().ok_or("stack underflow: Mul")?;
                let l = stack.pop().ok_or("stack underflow: Mul")?;
                stack.push(Expr::Mul(Box::new(l), Box::new(r)));
            }
            ExprToken::Div => {
                let r = stack.pop().ok_or("stack underflow: Div")?;
                let l = stack.pop().ok_or("stack underflow: Div")?;
                stack.push(Expr::Div(Box::new(l), Box::new(r)));
            }
            ExprToken::Sin => {
                let e = stack.pop().ok_or("stack underflow: Sin")?;
                stack.push(Expr::Sin(Box::new(e)));
            }
            ExprToken::Cos => {
                let e = stack.pop().ok_or("stack underflow: Cos")?;
                stack.push(Expr::Cos(Box::new(e)));
            }
            ExprToken::Exp => {
                let e = stack.pop().ok_or("stack underflow: Exp")?;
                stack.push(Expr::Exp(Box::new(e)));
            }
            ExprToken::Ln => {
                let e = stack.pop().ok_or("stack underflow: Ln")?;
                stack.push(Expr::Ln(Box::new(e)));
            }
            ExprToken::Sqrt => {
                let e = stack.pop().ok_or("stack underflow: Sqrt")?;
                stack.push(Expr::Sqrt(Box::new(e)));
            }
            ExprToken::LParen | ExprToken::RParen => {}
        }
    }
    if stack.len() != 1 {
        return Err(format!("parse error: stack has {} elements", stack.len()));
    }
    stack.pop().ok_or_else(|| "empty stack".to_string())
}

// ────────────────────────────────────────────────────────────────────────────
// 4. GradientSymbolicRegressor
// ────────────────────────────────────────────────────────────────────────────

/// Library of candidate symbolic basis expressions.
pub struct ExprBasis {
    pub exprs: Vec<Expr>,
    pub names: Vec<String>,
}

impl ExprBasis {
    pub fn new(exprs: Vec<Expr>, names: Vec<String>) -> Self {
        assert_eq!(exprs.len(), names.len());
        Self { exprs, names }
    }

    /// Standard library: {1, x0, x1, x0^2, x1^2, x0*x1, sin(x0), cos(x0), exp(x0), ln|x0|}
    pub fn standard(n_vars: usize) -> Self {
        // One constant term plus four basis functions per variable.
        let capacity = 1 + 4 * n_vars;
        let mut exprs = Vec::with_capacity(capacity);
        let mut names = Vec::with_capacity(capacity);
        exprs.push(Expr::Const(1.0));
        names.push("1".to_string());
        for i in 0..n_vars {
            let var = Expr::Var(format!("x{i}"));
            names.push(format!("x{i}"));
            exprs.push(var.clone());
            let sq = Expr::Pow(Box::new(var.clone()), Box::new(Expr::Const(2.0)));
            names.push(format!("x{i}^2"));
            exprs.push(sq);
            let sin_v = Expr::Sin(Box::new(var.clone()));
            names.push(format!("sin(x{i})"));
            exprs.push(sin_v);
            let exp_v = Expr::Exp(Box::new(var));
            names.push(format!("exp(x{i})"));
            exprs.push(exp_v);
        }
        Self { exprs, names }
    }
}

/// Sparse regression over symbolic basis functions (LASSO-like iterative thresholding).
pub struct GradientSymbolicRegressor {
    basis: ExprBasis,
    coefficients: Vec<f64>,
    max_iter: usize,
    learning_rate: f64,
}

impl GradientSymbolicRegressor {
    pub fn new(basis: ExprBasis) -> Self {
        let n = basis.exprs.len();
        Self {
            basis,
            coefficients: vec![0.0; n],
            max_iter: 200,
            learning_rate: 0.01,
        }
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Evaluate basis on a single sample.
    fn eval_basis(&self, x: &[f64]) -> Vec<f64> {
        let mut vars = HashMap::new();
        for (i, &v) in x.iter().enumerate() {
            vars.insert(format!("x{i}"), v);
        }
        self.basis
            .exprs
            .iter()
            .map(|e| {
                let v = eval(e, &vars).unwrap_or(0.0);
                // Clamp to prevent gradient explosion from unbounded basis functions (e.g. exp)
                if v.is_finite() {
                    v.clamp(-1e6, 1e6)
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Fit via gradient descent + iterative thresholding (LASSO-like).
    pub fn fit(&mut self, x_data: &[Vec<f64>], y_data: &[f64], sparsity: f64) -> Vec<f64> {
        let n_basis = self.basis.exprs.len();
        let n_samples = x_data.len().min(y_data.len());
        self.coefficients = vec![0.0; n_basis];

        // Pre-compute basis matrix [n_samples × n_basis]
        let phi: Vec<Vec<f64>> = x_data
            .iter()
            .take(n_samples)
            .map(|x| self.eval_basis(x))
            .collect();

        for _iter in 0..self.max_iter {
            // Compute predictions and residuals
            let mut grad = vec![0.0; n_basis];
            for s in 0..n_samples {
                let y_hat: f64 = phi[s]
                    .iter()
                    .zip(&self.coefficients)
                    .map(|(b, c)| b * c)
                    .sum();
                let resid = if y_hat.is_finite() {
                    y_hat - y_data[s]
                } else {
                    0.0
                };
                for j in 0..n_basis {
                    let g = 2.0 * resid * phi[s][j] / n_samples as f64;
                    if g.is_finite() {
                        grad[j] += g;
                    }
                }
            }
            // Gradient step with NaN guard
            for j in 0..n_basis {
                if grad[j].is_finite() {
                    self.coefficients[j] -= self.learning_rate * grad[j];
                    // Clamp to prevent runaway
                    if !self.coefficients[j].is_finite() {
                        self.coefficients[j] = 0.0;
                    }
                }
            }
            // Soft-threshold (sparsity = λ)
            for c in &mut self.coefficients {
                if c.abs() < sparsity {
                    *c = 0.0;
                } else {
                    *c -= c.signum() * sparsity * self.learning_rate;
                }
            }
        }
        self.coefficients.clone()
    }

    /// Predict using fitted coefficients.
    pub fn predict(&self, x: &[f64]) -> f64 {
        let basis_vals = self.eval_basis(x);
        basis_vals
            .iter()
            .zip(&self.coefficients)
            .map(|(b, c)| b * c)
            .sum()
    }

    /// Return active terms (non-zero coefficients).
    pub fn active_terms(&self) -> Vec<(String, f64)> {
        self.basis
            .names
            .iter()
            .zip(&self.coefficients)
            .filter(|(_, c)| c.abs() > 1e-10)
            .map(|(name, &c)| (name.clone(), c))
            .collect()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 5. FormulaSearch — genetic programming
// ────────────────────────────────────────────────────────────────────────────

/// Individual in a genetic programming population.
#[derive(Debug, Clone)]
pub struct Individual {
    pub expr: Expr,
    pub fitness: f64,
}

/// Count nodes in an expression tree.
fn count_nodes(expr: &Expr) -> usize {
    match expr {
        Expr::Const(_) | Expr::Var(_) => 1,
        Expr::Neg(e) | Expr::Sin(e) | Expr::Cos(e) | Expr::Exp(e) | Expr::Ln(e) | Expr::Sqrt(e) => {
            1 + count_nodes(e)
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            1 + count_nodes(l) + count_nodes(r)
        }
    }
}

/// Get node at position `target` (DFS order), returns the sub-expression.
fn get_node(expr: &Expr, pos: usize, counter: &mut usize) -> Option<Expr> {
    if *counter == pos {
        return Some(expr.clone());
    }
    *counter += 1;
    match expr {
        Expr::Const(_) | Expr::Var(_) => None,
        Expr::Neg(e) | Expr::Sin(e) | Expr::Cos(e) | Expr::Exp(e) | Expr::Ln(e) | Expr::Sqrt(e) => {
            get_node(e, pos, counter)
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            if let Some(found) = get_node(l, pos, counter) {
                return Some(found);
            }
            get_node(r, pos, counter)
        }
    }
}

/// Replace node at position `target` with `replacement`.
fn replace_node(expr: Expr, pos: usize, replacement: &Expr, counter: &mut usize) -> Expr {
    if *counter == pos {
        *counter += 1;
        return replacement.clone();
    }
    *counter += 1;
    match expr {
        Expr::Const(_) | Expr::Var(_) => expr,
        Expr::Neg(e) => Expr::Neg(Box::new(replace_node(*e, pos, replacement, counter))),
        Expr::Sin(e) => Expr::Sin(Box::new(replace_node(*e, pos, replacement, counter))),
        Expr::Cos(e) => Expr::Cos(Box::new(replace_node(*e, pos, replacement, counter))),
        Expr::Exp(e) => Expr::Exp(Box::new(replace_node(*e, pos, replacement, counter))),
        Expr::Ln(e) => Expr::Ln(Box::new(replace_node(*e, pos, replacement, counter))),
        Expr::Sqrt(e) => Expr::Sqrt(Box::new(replace_node(*e, pos, replacement, counter))),
        Expr::Add(l, r) => {
            let nl = replace_node(*l, pos, replacement, counter);
            let nr = replace_node(*r, pos, replacement, counter);
            Expr::Add(Box::new(nl), Box::new(nr))
        }
        Expr::Sub(l, r) => {
            let nl = replace_node(*l, pos, replacement, counter);
            let nr = replace_node(*r, pos, replacement, counter);
            Expr::Sub(Box::new(nl), Box::new(nr))
        }
        Expr::Mul(l, r) => {
            let nl = replace_node(*l, pos, replacement, counter);
            let nr = replace_node(*r, pos, replacement, counter);
            Expr::Mul(Box::new(nl), Box::new(nr))
        }
        Expr::Div(l, r) => {
            let nl = replace_node(*l, pos, replacement, counter);
            let nr = replace_node(*r, pos, replacement, counter);
            Expr::Div(Box::new(nl), Box::new(nr))
        }
        Expr::Pow(l, r) => {
            let nl = replace_node(*l, pos, replacement, counter);
            let nr = replace_node(*r, pos, replacement, counter);
            Expr::Pow(Box::new(nl), Box::new(nr))
        }
    }
}

/// Crossover two expressions by swapping a random subtree.
pub fn crossover(a: &Expr, b: &Expr, rng: &mut StdRng) -> Expr {
    let n_a = count_nodes(a);
    let n_b = count_nodes(b);
    if n_a == 0 || n_b == 0 {
        return a.clone();
    }
    let pos_b = rng.random_range(0..n_b);
    let mut ctr = 0usize;
    let subtree = match get_node(b, pos_b, &mut ctr) {
        Some(s) => s,
        None => b.clone(),
    };
    let pos_a = rng.random_range(0..n_a);
    let mut ctr2 = 0usize;
    replace_node(a.clone(), pos_a, &subtree, &mut ctr2)
}

fn random_expr(rng: &mut StdRng, var_names: &[String], depth: usize) -> Expr {
    if depth == 0 || rng.random::<f64>() < 0.3 {
        // Terminal
        if var_names.is_empty() || rng.random::<f64>() < 0.4 {
            Expr::Const((rng.random::<f64>() - 0.5) * 4.0)
        } else {
            let idx = rng.random_range(0..var_names.len());
            Expr::Var(var_names[idx].clone())
        }
    } else {
        let op = rng.random_range(0u8..10u8);
        match op {
            0 => Expr::Add(
                Box::new(random_expr(rng, var_names, depth - 1)),
                Box::new(random_expr(rng, var_names, depth - 1)),
            ),
            1 => Expr::Sub(
                Box::new(random_expr(rng, var_names, depth - 1)),
                Box::new(random_expr(rng, var_names, depth - 1)),
            ),
            2 => Expr::Mul(
                Box::new(random_expr(rng, var_names, depth - 1)),
                Box::new(random_expr(rng, var_names, depth - 1)),
            ),
            3 => Expr::Div(
                Box::new(random_expr(rng, var_names, depth - 1)),
                Box::new(random_expr(rng, var_names, depth - 1)),
            ),
            4 => Expr::Sin(Box::new(random_expr(rng, var_names, depth - 1))),
            5 => Expr::Cos(Box::new(random_expr(rng, var_names, depth - 1))),
            6 => Expr::Exp(Box::new(random_expr(rng, var_names, depth - 1))),
            7 => Expr::Neg(Box::new(random_expr(rng, var_names, depth - 1))),
            8 => Expr::Pow(
                Box::new(random_expr(rng, var_names, depth - 1)),
                Box::new(Expr::Const(rng.random_range(2u8..4u8) as f64)),
            ),
            _ => Expr::Sqrt(Box::new(Expr::Pow(
                Box::new(random_expr(rng, var_names, depth - 1)),
                Box::new(Expr::Const(2.0)),
            ))),
        }
    }
}

/// Mutate an expression: replace a random node with a random sub-expression.
pub fn mutate(expr: &Expr, rng: &mut StdRng, var_names: &[String]) -> Expr {
    let n = count_nodes(expr);
    if n == 0 {
        return random_expr(rng, var_names, 2);
    }
    let pos = rng.random_range(0..n);
    let replacement = random_expr(rng, var_names, 2);
    let mut ctr = 0usize;
    replace_node(expr.clone(), pos, &replacement, &mut ctr)
}

/// Tournament selection: pick the best from `k` random candidates.
pub fn tournament_select<'a>(pop: &'a [Individual], k: usize, rng: &mut StdRng) -> &'a Individual {
    let n = pop.len();
    let mut best_idx = rng.random_range(0..n);
    for _ in 1..k {
        let idx = rng.random_range(0..n);
        if pop[idx].fitness < pop[best_idx].fitness {
            best_idx = idx;
        }
    }
    &pop[best_idx]
}

fn mse_fitness(expr: &Expr, x: &[Vec<f64>], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return f64::MAX;
    }
    let mut total = 0.0f64;
    for i in 0..n {
        let mut vars = HashMap::new();
        for (j, v) in x[i].iter().enumerate() {
            vars.insert(format!("x{j}"), *v);
        }
        let pred = eval(expr, &vars).unwrap_or(f64::NAN);
        if pred.is_nan() || pred.is_infinite() {
            return f64::MAX;
        }
        total += (pred - y[i]).powi(2);
    }
    (total / n as f64).sqrt()
}
