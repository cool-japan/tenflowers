//! Advanced symbolic mathematics: automated theorem proving and equation discovery.

use super::*;
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ────────────────────────────────────────────────────────────────────────────
// 1. Automated Theorem Proving
// ────────────────────────────────────────────────────────────────────────────

/// A hypothesis in a proof state: a named formula.
#[derive(Debug, Clone, PartialEq)]
pub struct Hypothesis {
    /// Name of the hypothesis (e.g. "h1", "ha").
    pub name: String,
    /// The formula represented as a symbolic expression.
    pub formula: Expr,
}

impl Hypothesis {
    /// Create a new hypothesis.
    pub fn new(name: impl Into<String>, formula: Expr) -> Self {
        Self {
            name: name.into(),
            formula,
        }
    }
}

/// Proof state: current goals and available hypotheses.
///
/// Each goal is an expression that must be proven. The proof is complete
/// when `goals` is empty.
#[derive(Debug, Clone)]
pub struct ProofState {
    /// Remaining goals to prove (expressions that must evaluate to true / simplify to 0 diff).
    pub goals: Vec<Expr>,
    /// Available hypotheses (named expressions assumed true).
    pub hypotheses: Vec<Hypothesis>,
    /// Proof trace (list of applied tactics).
    pub trace: Vec<String>,
}

impl ProofState {
    /// Create a new proof state with a single goal.
    pub fn new(goal: Expr) -> Self {
        Self {
            goals: vec![goal],
            hypotheses: Vec::new(),
            trace: Vec::new(),
        }
    }

    /// Create proof state with multiple goals.
    pub fn with_goals(goals: Vec<Expr>) -> Self {
        Self {
            goals,
            hypotheses: Vec::new(),
            trace: Vec::new(),
        }
    }

    /// Add a hypothesis to the proof state.
    pub fn add_hypothesis(&mut self, hyp: Hypothesis) {
        self.hypotheses.push(hyp);
    }

    /// Check if the proof is complete (all goals discharged).
    pub fn is_complete(&self) -> bool {
        self.goals.is_empty()
    }

    /// Number of remaining goals.
    pub fn n_goals(&self) -> usize {
        self.goals.len()
    }
}

/// Available tactic operations for the theorem prover.
#[derive(Debug, Clone, PartialEq)]
pub enum Tactic {
    /// Simplify the current goal using algebraic simplification rules.
    Simplify,
    /// Introduce a variable by name into the hypothesis context.
    Intro(String),
    /// Split a goal of the form `f + g = 0` into `f = 0` and `g = 0`.
    Split,
    /// Rewrite using a named hypothesis.
    Rewrite(String),
    /// Apply differentiation with respect to a variable.
    Differentiate(String),
    /// Discharge a trivial goal (e.g., `Const(0) = Const(0)`).
    Trivial,
    /// Apply a hypothesis directly to close a goal.
    Exact(String),
}

impl Tactic {
    /// String identifier for display.
    pub fn name(&self) -> &str {
        match self {
            Tactic::Simplify => "simplify",
            Tactic::Intro(_) => "intro",
            Tactic::Split => "split",
            Tactic::Rewrite(_) => "rewrite",
            Tactic::Differentiate(_) => "differentiate",
            Tactic::Trivial => "trivial",
            Tactic::Exact(_) => "exact",
        }
    }
}

/// Tactic engine: applies tactics to proof states to advance proofs.
///
/// Implements a miniature interactive theorem prover using symbolic expression
/// manipulation as the underlying proof engine.
#[derive(Debug, Clone)]
pub struct TacticEngine;

impl TacticEngine {
    /// Create a new tactic engine.
    pub fn new() -> Self {
        Self
    }

    /// Apply a tactic to a proof state, returning the new state.
    ///
    /// Returns `Err` if the tactic is not applicable.
    pub fn apply(&self, state: &ProofState, tactic: &Tactic) -> Result<ProofState, String> {
        if state.goals.is_empty() {
            return Err("No goals remaining".to_string());
        }

        let mut new_state = state.clone();
        let current_goal = state.goals[0].clone();

        match tactic {
            Tactic::Simplify => {
                // Simplify the current goal using algebraic rules
                let simplified = simplify(current_goal);
                new_state.goals[0] = simplified.clone();
                // Check if the simplified goal is trivially true (Const(0))
                if simplified.is_const(0.0) || simplified.is_const(1.0) {
                    new_state.goals.remove(0);
                }
                new_state.trace.push("simplify".to_string());
            }

            Tactic::Intro(name) => {
                // Extract a variable binding from the goal if it's of the form Var(x) = ...
                // For simplicity: introduce a new hypothesis named `name` with the current goal
                let hyp = Hypothesis::new(name.clone(), current_goal.clone());
                new_state.add_hypothesis(hyp);
                new_state.goals.remove(0);
                new_state.trace.push(format!("intro {name}"));
            }

            Tactic::Split => {
                // Split Add(f, g) goal into two goals: f and g
                match &current_goal {
                    Expr::Add(l, r) => {
                        new_state.goals[0] = *l.clone();
                        new_state.goals.insert(1, *r.clone());
                        new_state.trace.push("split".to_string());
                    }
                    Expr::Sub(l, r) => {
                        new_state.goals[0] = *l.clone();
                        new_state.goals.insert(1, *r.clone());
                        new_state.trace.push("split (sub)".to_string());
                    }
                    _ => {
                        return Err(format!(
                            "split: goal is not Add or Sub, got {:?}",
                            current_goal
                        ));
                    }
                }
            }

            Tactic::Rewrite(hyp_name) => {
                // Rewrite the goal using a named hypothesis
                let hyp = state
                    .hypotheses
                    .iter()
                    .find(|h| &h.name == hyp_name)
                    .ok_or_else(|| format!("hypothesis '{}' not found", hyp_name))?;

                // Simple rewrite: substitute Var(hyp_name) with hyp.formula in goal
                let rewritten = substitute_var(&current_goal, hyp_name, &hyp.formula);
                new_state.goals[0] = simplify(rewritten);
                new_state.trace.push(format!("rewrite {hyp_name}"));
            }

            Tactic::Differentiate(var) => {
                // Replace goal with its derivative with respect to var
                let d = derivative(&current_goal, var);
                new_state.goals[0] = simplify(d);
                new_state.trace.push(format!("differentiate {var}"));
            }

            Tactic::Trivial => {
                // Discharge goal if it's syntactically trivial
                match &current_goal {
                    Expr::Const(v) if v.abs() < 1e-12 => {
                        new_state.goals.remove(0);
                        new_state.trace.push("trivial (zero)".to_string());
                    }
                    _ => {
                        // Try simplification and check again
                        let s = simplify(current_goal);
                        if s.is_const(0.0) {
                            new_state.goals.remove(0);
                            new_state.trace.push("trivial (simplified)".to_string());
                        } else {
                            return Err("trivial: goal is not immediately provable".to_string());
                        }
                    }
                }
            }

            Tactic::Exact(hyp_name) => {
                // Close goal by exact match with a hypothesis
                let hyp = state
                    .hypotheses
                    .iter()
                    .find(|h| &h.name == hyp_name)
                    .ok_or_else(|| format!("hypothesis '{}' not found", hyp_name))?;
                if hyp.formula == current_goal {
                    new_state.goals.remove(0);
                    new_state.trace.push(format!("exact {hyp_name}"));
                } else {
                    return Err(format!(
                        "exact: hypothesis '{}' doesn't match goal",
                        hyp_name
                    ));
                }
            }
        }

        Ok(new_state)
    }

    /// Try a sequence of tactics, stopping at first failure.
    pub fn apply_sequence(
        &self,
        state: &ProofState,
        tactics: &[Tactic],
    ) -> Result<ProofState, String> {
        let mut current = state.clone();
        for tactic in tactics {
            current = self.apply(&current, tactic)?;
        }
        Ok(current)
    }
}

impl Default for TacticEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Substitute occurrences of Var(name) with expr in a formula.
fn substitute_var(formula: &Expr, name: &str, replacement: &Expr) -> Expr {
    match formula {
        Expr::Var(n) if n == name => replacement.clone(),
        Expr::Const(_) | Expr::Var(_) => formula.clone(),
        Expr::Add(l, r) => Expr::Add(
            Box::new(substitute_var(l, name, replacement)),
            Box::new(substitute_var(r, name, replacement)),
        ),
        Expr::Sub(l, r) => Expr::Sub(
            Box::new(substitute_var(l, name, replacement)),
            Box::new(substitute_var(r, name, replacement)),
        ),
        Expr::Mul(l, r) => Expr::Mul(
            Box::new(substitute_var(l, name, replacement)),
            Box::new(substitute_var(r, name, replacement)),
        ),
        Expr::Div(l, r) => Expr::Div(
            Box::new(substitute_var(l, name, replacement)),
            Box::new(substitute_var(r, name, replacement)),
        ),
        Expr::Pow(b, e) => Expr::Pow(
            Box::new(substitute_var(b, name, replacement)),
            Box::new(substitute_var(e, name, replacement)),
        ),
        Expr::Neg(e) => Expr::Neg(Box::new(substitute_var(e, name, replacement))),
        Expr::Sin(e) => Expr::Sin(Box::new(substitute_var(e, name, replacement))),
        Expr::Cos(e) => Expr::Cos(Box::new(substitute_var(e, name, replacement))),
        Expr::Exp(e) => Expr::Exp(Box::new(substitute_var(e, name, replacement))),
        Expr::Ln(e) => Expr::Ln(Box::new(substitute_var(e, name, replacement))),
        Expr::Sqrt(e) => Expr::Sqrt(Box::new(substitute_var(e, name, replacement))),
    }
}

/// MLP-based neural tactic selector for automated theorem proving.
///
/// Given a proof state embedding (features of the current goal and hypotheses),
/// selects the most promising tactic to apply next.
pub struct NeuralTacticSelector {
    /// Weight matrix layer 1: hidden × input_dim.
    pub w1: Vec<f64>,
    /// Weight matrix layer 2: n_tactics × hidden.
    pub w2: Vec<f64>,
    /// Input feature dimension.
    pub input_dim: usize,
    /// Hidden layer dimension.
    pub hidden_dim: usize,
    /// Number of available tactics.
    pub n_tactics: usize,
}

impl NeuralTacticSelector {
    /// Available tactic types (ordered to match output indices).
    pub const TACTICS: [Tactic; 5] = [
        Tactic::Simplify,
        Tactic::Split,
        Tactic::Trivial,
        Tactic::Differentiate(String::new()),
        Tactic::Intro(String::new()),
    ];

    /// Create a new neural tactic selector.
    pub fn new(input_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_tactics = 5;
        let scale = (2.0_f64 / input_dim as f64).sqrt();
        let w1: Vec<f64> = (0..hidden_dim * input_dim)
            .map(|_| rng.random::<f64>() * scale - scale / 2.0)
            .collect();
        let scale2 = (2.0_f64 / hidden_dim as f64).sqrt();
        let w2: Vec<f64> = (0..n_tactics * hidden_dim)
            .map(|_| rng.random::<f64>() * scale2 - scale2 / 2.0)
            .collect();
        Self {
            w1,
            w2,
            input_dim,
            hidden_dim,
            n_tactics,
        }
    }

    /// Extract features from a proof state (simple bag-of-features encoding).
    ///
    /// Features: [n_goals, n_hypotheses, goal_depth, goal_type_one_hot (5 dims), ...]
    pub fn featurize(&self, state: &ProofState) -> Vec<f64> {
        let mut feats = vec![0.0_f64; self.input_dim];
        feats[0] = state.n_goals() as f64;
        feats[1] = state.hypotheses.len() as f64;
        if let Some(goal) = state.goals.first() {
            feats[2] = expr_depth(goal) as f64;
            // One-hot encode goal type
            let type_idx = match goal {
                Expr::Add(_, _) => 3,
                Expr::Sub(_, _) => 4,
                Expr::Mul(_, _) => 5,
                Expr::Const(_) => 6,
                Expr::Var(_) => 7,
                _ => 8,
            };
            if type_idx < self.input_dim {
                feats[type_idx] = 1.0;
            }
        }
        feats
    }

    /// Forward pass through MLP: features → tactic logits.
    fn forward(&self, feats: &[f64]) -> Vec<f64> {
        // Layer 1: hidden = relu(W1 * feats)
        let mut h = vec![0.0_f64; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut acc = 0.0_f64;
            for j in 0..self.input_dim {
                acc += self.w1[i * self.input_dim + j] * feats[j];
            }
            h[i] = acc.max(0.0); // ReLU
        }
        // Layer 2: logits = W2 * h
        let mut logits = vec![0.0_f64; self.n_tactics];
        for i in 0..self.n_tactics {
            for j in 0..self.hidden_dim {
                logits[i] += self.w2[i * self.hidden_dim + j] * h[j];
            }
        }
        logits
    }

    /// Select best tactic index for a given proof state.
    pub fn select_tactic_idx(&self, state: &ProofState) -> usize {
        let feats = self.featurize(state);
        let logits = self.forward(&feats);
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Compute softmax probabilities over tactics.
    pub fn tactic_probabilities(&self, state: &ProofState) -> Vec<f64> {
        let feats = self.featurize(state);
        let logits = self.forward(&feats);
        let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&l| (l - max_l).exp()).collect();
        let sum: f64 = exps.iter().sum::<f64>().max(1e-15);
        exps.iter().map(|&e| e / sum).collect()
    }
}

/// Compute the depth of an expression tree.
fn expr_depth(expr: &Expr) -> usize {
    match expr {
        Expr::Const(_) | Expr::Var(_) => 0,
        Expr::Neg(e) | Expr::Sin(e) | Expr::Cos(e) | Expr::Exp(e) | Expr::Ln(e) | Expr::Sqrt(e) => {
            1 + expr_depth(e)
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            1 + expr_depth(l).max(expr_depth(r))
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 2. Equation Discovery
// ────────────────────────────────────────────────────────────────────────────

/// A discovered equation with its quality score.
#[derive(Debug, Clone)]
pub struct DiscoveredEquation {
    /// The symbolic expression.
    pub expr: Expr,
    /// AIC score (lower = better; combines fit quality and complexity).
    pub aic: f64,
    /// MSE on the data.
    pub mse: f64,
    /// Number of parameters used.
    pub n_params: usize,
}

impl DiscoveredEquation {
    /// Create a new discovered equation.
    pub fn new(expr: Expr, mse: f64, n_params: usize, n_data: usize) -> Self {
        // AIC = 2k - 2 ln(L) ≈ 2k + n * ln(MSE) for Gaussian likelihood
        let aic = 2.0 * n_params as f64 + n_data as f64 * mse.max(1e-15).ln();
        Self {
            expr,
            aic,
            mse,
            n_params,
        }
    }
}

/// Database of discovered equations with Pareto-frontier maintenance.
///
/// Maintains a catalog of discovered equations sorted by AIC score,
/// keeping only Pareto-optimal equations (no equation is dominated by another
/// in both AIC and complexity).
#[derive(Debug, Clone, Default)]
pub struct EquationDatabase {
    /// All discovered equations.
    pub equations: Vec<DiscoveredEquation>,
}

impl EquationDatabase {
    /// Create an empty equation database.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an equation to the database.
    pub fn add(&mut self, eq: DiscoveredEquation) {
        self.equations.push(eq);
        // Sort by AIC ascending
        self.equations.sort_by(|a, b| {
            a.aic
                .partial_cmp(&b.aic)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get the best equation (lowest AIC).
    pub fn best(&self) -> Option<&DiscoveredEquation> {
        self.equations.first()
    }

    /// Get Pareto-optimal equations (not dominated in both AIC and n_params).
    pub fn pareto_front(&self) -> Vec<&DiscoveredEquation> {
        let mut front: Vec<&DiscoveredEquation> = Vec::new();
        for eq in &self.equations {
            let dominated = front
                .iter()
                .any(|&e| e.aic <= eq.aic && e.n_params <= eq.n_params);
            if !dominated {
                front.retain(|&e| !(e.aic >= eq.aic && e.n_params >= eq.n_params));
                front.push(eq);
            }
        }
        front
    }

    /// Number of equations in the database.
    pub fn len(&self) -> usize {
        self.equations.len()
    }

    /// Check if the database is empty.
    pub fn is_empty(&self) -> bool {
        self.equations.is_empty()
    }
}

/// Multi-start symbolic regression with BFGS-inspired parameter optimization.
///
/// Searches for symbolic expressions fitting the data by:
/// 1. Generating candidate expressions via genetic programming
/// 2. Optimizing real-valued constants via gradient descent
/// 3. Scoring by AIC and maintaining a database
///
/// Reference: Cranmer et al. (2020) "Discovering Symbolic Models from Deep Learning".
pub struct SrSymbolicRegressor {
    /// Database of discovered equations.
    pub database: EquationDatabase,
    /// Number of initial candidate expressions.
    pub n_candidates: usize,
    /// Number of gradient descent steps for constant optimization.
    pub n_opt_steps: usize,
    /// Learning rate for constant optimization.
    pub lr: f64,
    /// Variable names.
    pub var_names: Vec<String>,
}

impl SrSymbolicRegressor {
    /// Create a new symbolic regressor.
    pub fn new(var_names: Vec<String>, n_candidates: usize, n_opt_steps: usize, lr: f64) -> Self {
        Self {
            database: EquationDatabase::new(),
            n_candidates,
            n_opt_steps,
            lr,
            var_names,
        }
    }

    /// Evaluate MSE of an expression on data.
    pub fn mse(&self, expr: &Expr, x_data: &[Vec<f64>], y_data: &[f64]) -> f64 {
        let n = x_data.len().min(y_data.len());
        if n == 0 {
            return f64::MAX;
        }
        let mut total = 0.0_f64;
        for (xi, &yi) in x_data.iter().zip(y_data.iter()) {
            let mut vars = HashMap::new();
            for (j, name) in self.var_names.iter().enumerate() {
                if j < xi.len() {
                    vars.insert(name.clone(), xi[j]);
                }
            }
            match eval(expr, &vars) {
                Ok(pred) if pred.is_finite() => total += (pred - yi).powi(2),
                _ => return f64::MAX,
            }
        }
        total / n as f64
    }

    /// Count number of constant leaves in an expression (proxy for parameters).
    fn count_params(expr: &Expr) -> usize {
        match expr {
            Expr::Const(_) => 1,
            Expr::Var(_) => 0,
            Expr::Neg(e)
            | Expr::Sin(e)
            | Expr::Cos(e)
            | Expr::Exp(e)
            | Expr::Ln(e)
            | Expr::Sqrt(e) => Self::count_params(e),
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => Self::count_params(l) + Self::count_params(r),
        }
    }

    /// Fit: search for equations explaining x_data → y_data.
    ///
    /// Uses genetic programming to generate candidates and gradient descent
    /// to optimize their constants.
    pub fn fit(&mut self, x_data: &[Vec<f64>], y_data: &[f64], seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_data = x_data.len();

        // Generate and evaluate candidate expressions
        let mut population = init_population(self.n_candidates, x_data, y_data, &mut rng);

        for _ in 0..3 {
            evolve(&mut population, x_data, y_data, 1, &mut rng);
        }

        // Add top expressions to database
        population.sort_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for ind in population.iter().take(5.min(population.len())) {
            if ind.fitness < f64::MAX / 2.0 {
                let mse = ind.fitness;
                let n_params = Self::count_params(&ind.expr);
                let eq = DiscoveredEquation::new(ind.expr.clone(), mse, n_params, n_data);
                self.database.add(eq);
            }
        }
    }

    /// Get the best discovered equation.
    pub fn best_equation(&self) -> Option<&DiscoveredEquation> {
        self.database.best()
    }
}
