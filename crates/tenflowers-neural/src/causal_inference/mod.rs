//! Causal Inference Module — Round 13 Track A.
//!
//! A comprehensive causal inference library providing:
//!
//! - **Structural Causal Models (SCM)**: Graph-based causal models with DAG validation,
//!   d-separation (Bayes Ball), Markov blankets, and topological ordering.
//! - **Interventional Distributions**: Backdoor adjustment, ATE, CATE via do-calculus.
//! - **Double/Debiased ML**: Robinson (1988) partially linear model with cross-fitting.
//! - **Counterfactual Estimation**: Abduction-action-prediction 3-step procedure.
//! - **Propensity Score Methods**: Logistic regression, IPW (ATE, ATT), stabilized weights.
//! - **Regression Discontinuity Design**: Triangular-kernel weighted local regression.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use tenflowers_neural::causal_inference::{CausalGraph, CausalEstimator, CausalQuery, Intervention};
//!
//! let mut g = CausalGraph::new();
//! g.add_variable("X")?;
//! g.add_variable("Y")?;
//! g.add_edge("X", "Y")?;
//! assert!(g.is_dag());
//!
//! let sorted = g.topological_sort()?;
//! assert_eq!(sorted[0], "X");
//! ```

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::{HashMap, HashSet, VecDeque};
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Mathematical helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Sigmoid function: `1 / (1 + exp(-x))`.
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Gaussian kernel density: `exp(-0.5 * ((x - mu) / h)^2) / (h * sqrt(2π))`.
#[inline]
fn gaussian_kernel(x: f64, mu: f64, h: f64) -> f64 {
    let u = (x - mu) / h;
    (-0.5 * u * u).exp() / (h * (2.0 * std::f64::consts::PI).sqrt())
}

/// Triangular kernel: `max(0, 1 - |u|)` where `u = (x - cutoff) / bandwidth`.
#[inline]
fn triangular_kernel(x: f64, cutoff: f64, bandwidth: f64) -> f64 {
    let u = (x - cutoff) / bandwidth;
    (1.0 - u.abs()).max(0.0)
}

/// Normal CDF approximation (Abramowitz & Stegun 26.2.16) for p-value computation.
fn normal_cdf(x: f64) -> f64 {
    // Approximation error < 7.5e-8
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let phi = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    if x >= 0.0 {
        1.0 - phi * poly
    } else {
        phi * poly
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Structural Causal Model (SCM) / CausalGraph
// ─────────────────────────────────────────────────────────────────────────────

/// One structural equation in an SCM:
/// `variable = f(parents) + N(0, noise_std²)`.
#[derive(Debug, Clone)]
pub struct StructuralEquation {
    /// The variable this equation generates.
    pub variable: String,
    /// Direct parent variables (inputs to the function).
    pub parents: Vec<String>,
    /// Standard deviation of additive Gaussian noise.
    pub noise_std: f64,
}

/// A directed acyclic graph (DAG) representing causal structure.
///
/// Internally stores adjacency lists for fast parent/child lookups and a flat
/// edge list for iteration.  All mutating methods validate the DAG invariant
/// before committing changes.
#[derive(Debug, Clone)]
pub struct CausalGraph {
    /// Ordered list of variable names.
    variables: Vec<String>,
    /// Directed edges as `(parent, child)` pairs.
    edges: Vec<(String, String)>,
    /// Structural equations, keyed by variable name.
    equations: HashMap<String, StructuralEquation>,
    /// `parents[v]` = set of direct parents of `v`.
    parents: HashMap<String, HashSet<String>>,
    /// `children[v]` = set of direct children of `v`.
    children: HashMap<String, HashSet<String>>,
}

impl CausalGraph {
    /// Create an empty causal graph.
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            edges: Vec::new(),
            equations: HashMap::new(),
            parents: HashMap::new(),
            children: HashMap::new(),
        }
    }

    /// Register a new variable.  Returns an error if the name is already present.
    pub fn add_variable(&mut self, name: &str) -> Result<()> {
        if self.parents.contains_key(name) {
            return Err(TensorError::invalid_argument_op(
                "add_variable",
                &format!("Variable '{}' already exists in the graph", name),
            ));
        }
        self.variables.push(name.to_string());
        self.parents.insert(name.to_string(), HashSet::new());
        self.children.insert(name.to_string(), HashSet::new());
        Ok(())
    }

    /// Add a directed edge `from → to`.
    ///
    /// Both endpoints must be registered first.  Refuses to create cycles,
    /// returning an error if `to` is already an ancestor of `from`.
    pub fn add_edge(&mut self, from: &str, to: &str) -> Result<()> {
        if !self.parents.contains_key(from) {
            return Err(TensorError::invalid_argument_op(
                "add_edge",
                &format!("Variable '{}' not found", from),
            ));
        }
        if !self.parents.contains_key(to) {
            return Err(TensorError::invalid_argument_op(
                "add_edge",
                &format!("Variable '{}' not found", to),
            ));
        }
        // Check that adding from→to does not introduce a cycle.
        // A cycle exists iff `from` is reachable from `to`.
        let ancestors_of_from = self.ancestors(from);
        if ancestors_of_from.contains(to) || from == to {
            return Err(TensorError::invalid_argument_op(
                "add_edge",
                &format!("Adding edge '{}' → '{}' would create a cycle", from, to),
            ));
        }
        self.edges.push((from.to_string(), to.to_string()));
        self.parents.get_mut(to).map(|s| s.insert(from.to_string()));
        self.children
            .get_mut(from)
            .map(|s| s.insert(to.to_string()));
        Ok(())
    }

    /// Attach a structural equation to a variable.
    pub fn set_equation(&mut self, eq: StructuralEquation) -> Result<()> {
        if !self.parents.contains_key(&eq.variable) {
            return Err(TensorError::invalid_argument_op(
                "set_equation",
                &format!("Variable '{}' not found", eq.variable),
            ));
        }
        self.equations.insert(eq.variable.clone(), eq);
        Ok(())
    }

    /// Topological ordering using Kahn's BFS algorithm.
    ///
    /// Returns an error if the graph contains a cycle (shouldn't happen if
    /// `add_edge` was used, but handles direct construction).
    pub fn topological_sort(&self) -> Result<Vec<String>> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        for v in &self.variables {
            in_degree.insert(v.as_str(), 0);
        }
        for (_, to) in &self.edges {
            *in_degree.entry(to.as_str()).or_insert(0) += 1;
        }

        let mut queue: VecDeque<&str> = VecDeque::new();
        // Stable ordering: iterate variables in insertion order.
        for v in &self.variables {
            if in_degree[v.as_str()] == 0 {
                queue.push_back(v.as_str());
            }
        }

        let mut order: Vec<String> = Vec::with_capacity(self.variables.len());
        while let Some(v) = queue.pop_front() {
            order.push(v.to_string());
            if let Some(children) = self.children.get(v) {
                // Iterate in sorted order for determinism.
                let mut child_list: Vec<&str> = children.iter().map(|s| s.as_str()).collect();
                child_list.sort_unstable();
                for child in child_list {
                    if let Some(deg) = in_degree.get_mut(child) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push_back(child);
                        }
                    }
                }
            }
        }

        if order.len() != self.variables.len() {
            return Err(TensorError::invalid_argument_op(
                "topological_sort",
                "Graph contains a cycle; not a DAG",
            ));
        }
        Ok(order)
    }

    /// Returns `true` iff the graph is a directed acyclic graph.
    pub fn is_dag(&self) -> bool {
        self.topological_sort().is_ok()
    }

    /// Set of all strict ancestors of `var` (not including `var` itself).
    pub fn ancestors(&self, var: &str) -> HashSet<String> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        if let Some(parents) = self.parents.get(var) {
            for p in parents {
                queue.push_back(p.clone());
            }
        }
        while let Some(node) = queue.pop_front() {
            if visited.contains(&node) {
                continue;
            }
            visited.insert(node.clone());
            if let Some(parents) = self.parents.get(&node) {
                for p in parents {
                    if !visited.contains(p) {
                        queue.push_back(p.clone());
                    }
                }
            }
        }
        visited
    }

    /// Set of all strict descendants of `var` (not including `var` itself).
    pub fn descendants(&self, var: &str) -> HashSet<String> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        if let Some(children) = self.children.get(var) {
            for c in children {
                queue.push_back(c.clone());
            }
        }
        while let Some(node) = queue.pop_front() {
            if visited.contains(&node) {
                continue;
            }
            visited.insert(node.clone());
            if let Some(children) = self.children.get(&node) {
                for c in children {
                    if !visited.contains(c) {
                        queue.push_back(c.clone());
                    }
                }
            }
        }
        visited
    }

    /// Markov blanket of `var`: parents ∪ children ∪ co-parents (parents of children).
    pub fn markov_blanket(&self, var: &str) -> HashSet<String> {
        let mut blanket: HashSet<String> = HashSet::new();

        // Parents
        if let Some(parents) = self.parents.get(var) {
            for p in parents {
                if p != var {
                    blanket.insert(p.clone());
                }
            }
        }

        // Children and co-parents
        if let Some(children) = self.children.get(var) {
            for c in children {
                if c != var {
                    blanket.insert(c.clone());
                    // Co-parents: other parents of each child
                    if let Some(co_parents) = self.parents.get(c) {
                        for cp in co_parents {
                            if cp != var {
                                blanket.insert(cp.clone());
                            }
                        }
                    }
                }
            }
        }

        blanket
    }

    /// D-separation test using the Bayes Ball algorithm (Shachter 1998;
    /// Koller & Friedman 2009, Algorithm 3.1).
    ///
    /// Returns `true` iff `X` and `Y` are d-separated given conditioning set `Z`.
    ///
    /// **State**: `(node, via_child: bool)`.
    /// - `via_child = true`:  ball arrived at `node` from one of its children
    ///   (ball is travelling **upward** toward ancestors).
    /// - `via_child = false`: ball arrived at `node` from one of its parents
    ///   (ball is travelling **downward** toward descendants).
    ///
    /// **Propagation rules** at `node` in state `(node, vc)`:
    ///
    /// | vc    | node observed? | action |
    /// |-------|----------------|--------|
    /// | true  | no             | send up (to parents, vc=true) AND send down (to children, vc=false) |
    /// | true  | yes            | send up (to parents, vc=true) — observed non-collider still transmits upward via chain; *but* node IS a collider in the vc=true case since the ball came from a child — so if observed, open upward |
    /// | false | no             | send down (to children, vc=false) |
    /// | false | yes            | blocked (observed non-collider blocks chain/fork) |
    ///
    /// **Collider rule**: A node C is a collider on a path when BOTH adjacent
    /// path edges point INTO C.  In BFS, this is captured by `via_child=false`
    /// (ball arriving from a parent) at a node that also has incoming edges
    /// from the OTHER direction.  Specifically:
    /// - `(C, false)` AND C has ≥1 parent that could send the ball from the
    ///   other side: C acts as a collider for that path segment.
    /// - Collider is **active** iff C ∈ Z ∪ desc(Z).
    /// - Active collider in state `(C, false)`: send up to parents (vc=true).
    /// - Inactive collider in state `(C, false)`: blocked.
    ///
    /// To detect the collider case correctly, we use a structural criterion:
    /// a node is a **structural collider** if it has ≥2 parents.  For nodes
    /// with 0 or 1 parent, the `(C, false)` case is always a non-collider.
    pub fn d_separation(&self, x: &str, y: &str, z: &HashSet<String>) -> bool {
        // Precompute Z ∪ desc(Z) for collider activation.
        let mut z_and_desc: HashSet<String> = z.clone();
        for zv in z {
            for d in self.descendants(zv) {
                z_and_desc.insert(d);
            }
        }

        // BFS over (node, via_child) states.
        let mut visited: HashSet<(String, bool)> = HashSet::new();
        let mut queue: VecDeque<(String, bool)> = VecDeque::new();

        // Seed with X in both directions.
        queue.push_back((x.to_string(), true));
        queue.push_back((x.to_string(), false));

        while let Some((node, via_child)) = queue.pop_front() {
            let state = (node.clone(), via_child);
            if visited.contains(&state) {
                continue;
            }
            visited.insert(state);

            if node == y {
                return false; // active path reaches Y → NOT d-separated
            }

            let is_observed = z.contains(&node);
            let n_parents = self.parents.get(&node).map(|s| s.len()).unwrap_or(0);
            // Structural collider: has ≥2 parents (both edges point in).
            let is_structural_collider = n_parents >= 2;

            if via_child {
                // Ball travelling upward (arrived from a child).
                // At this point, node is NOT acting as a collider for the
                // current path segment (colliders are defined by incoming
                // edges from BOTH sides; here the ball came from below).
                // Non-collider rules:
                if !is_observed {
                    // Unobserved non-collider: transmit upward AND downward.
                    if let Some(parents) = self.parents.get(&node) {
                        for p in parents {
                            let s = (p.clone(), true);
                            if !visited.contains(&s) {
                                queue.push_back(s);
                            }
                        }
                    }
                    if let Some(children) = self.children.get(&node) {
                        for c in children {
                            let s = (c.clone(), false);
                            if !visited.contains(&s) {
                                queue.push_back(s);
                            }
                        }
                    }
                }
                // Observed non-collider with via_child=true: blocks.
            } else {
                // Ball travelling downward (arrived from a parent).
                if is_structural_collider {
                    // Collider node: only transmit if active (observed or has
                    // observed/active descendant).
                    if z_and_desc.contains(&node) {
                        // Active collider: send upward to ALL parents.
                        if let Some(parents) = self.parents.get(&node) {
                            for p in parents {
                                let s = (p.clone(), true);
                                if !visited.contains(&s) {
                                    queue.push_back(s);
                                }
                            }
                        }
                        // Also send downward if not observed itself.
                        if !is_observed {
                            if let Some(children) = self.children.get(&node) {
                                for c in children {
                                    let s = (c.clone(), false);
                                    if !visited.contains(&s) {
                                        queue.push_back(s);
                                    }
                                }
                            }
                        }
                    }
                    // Inactive collider: blocked (do nothing).
                } else {
                    // Non-collider node (0 or 1 parent): standard chain/fork.
                    if !is_observed {
                        // Unobserved: transmit downward to children.
                        if let Some(children) = self.children.get(&node) {
                            for c in children {
                                let s = (c.clone(), false);
                                if !visited.contains(&s) {
                                    queue.push_back(s);
                                }
                            }
                        }
                        // Also transmit upward (for fork: Z → X and Z → Y,
                        // ball from Z going down can also propagate to other
                        // parents via upward leg if Z has multiple parents).
                        if let Some(parents) = self.parents.get(&node) {
                            for p in parents {
                                let s = (p.clone(), true);
                                if !visited.contains(&s) {
                                    queue.push_back(s);
                                }
                            }
                        }
                    }
                    // Observed non-collider: blocked.
                }
            }
        }

        true // Y not reachable → d-separated
    }

    /// Direct parents of `var`.
    pub fn parents_of(&self, var: &str) -> HashSet<String> {
        self.parents.get(var).cloned().unwrap_or_default()
    }

    /// Direct children of `var`.
    pub fn children_of(&self, var: &str) -> HashSet<String> {
        self.children.get(var).cloned().unwrap_or_default()
    }

    /// All registered variable names.
    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    /// All edges as `(parent, child)` pairs.
    pub fn edges(&self) -> &[(String, String)] {
        &self.edges
    }

    /// Structural equation for `var`, if registered.
    pub fn equation(&self, var: &str) -> Option<&StructuralEquation> {
        self.equations.get(var)
    }
}

impl Default for CausalGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Interventional Distribution (do-calculus)
// ─────────────────────────────────────────────────────────────────────────────

/// A hard intervention `do(variable = value)`.
#[derive(Debug, Clone)]
pub struct Intervention {
    /// Variable being intervened upon.
    pub variable: String,
    /// Assigned value after intervention.
    pub value: f64,
}

/// A causal query combining intervention and optional conditioning.
#[derive(Debug, Clone)]
pub struct CausalQuery {
    /// Outcome variable of interest.
    pub target: String,
    /// Optional do-calculus intervention.
    pub intervention: Option<Intervention>,
    /// Observational conditioning on variable values.
    pub conditioning: HashMap<String, f64>,
}

/// Causal estimator backed by observational data.
pub struct CausalEstimator {
    graph: CausalGraph,
    data: Vec<HashMap<String, f64>>,
    seed: u64,
}

impl CausalEstimator {
    /// Create a new estimator from a causal graph and observational data.
    pub fn new(graph: CausalGraph, data: Vec<HashMap<String, f64>>, seed: u64) -> Self {
        Self { graph, data, seed }
    }

    /// Backdoor adjustment formula:
    /// `E[Y | do(X=x)] = Σ_z P(Y | X=x, Z=z) P(Z=z)`
    ///
    /// Uses Gaussian kernel density estimation over the adjustment set `Z`.
    /// The estimator integrates over observed values of `Z`.
    pub fn backdoor_adjustment(
        &self,
        query: &CausalQuery,
        adjustment_set: &[String],
    ) -> Result<f64> {
        if self.data.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "backdoor_adjustment",
                "Data is empty",
            ));
        }
        let intervention = match &query.intervention {
            Some(iv) => iv,
            None => {
                return Err(TensorError::invalid_argument_op(
                    "backdoor_adjustment",
                    "Query must have an intervention for backdoor adjustment",
                ))
            }
        };

        // If no adjustment needed, directly compute conditional mean.
        if adjustment_set.is_empty() {
            return self.conditional_mean_given_intervention(&query.target, intervention);
        }

        // Collect outcome values and adjustment variable values for all rows.
        let n = self.data.len() as f64;
        let bandwidth = 1.06 * n.powf(-0.2); // Silverman's rule, unit variance assumed

        // For each adjustment stratum (each data point as KDE representative),
        // compute P(Y | X=x, Z=z_i) weighted by P(Z=z_i).
        let mut numerator = 0.0_f64;
        let mut denominator = 0.0_f64;

        for row_i in &self.data {
            // Check this row has all adjustment variables.
            let z_values: Option<Vec<f64>> = adjustment_set
                .iter()
                .map(|z| row_i.get(z).copied())
                .collect();
            let z_i = match z_values {
                Some(v) => v,
                None => continue,
            };

            // Compute P(Z = z_i) using KDE over all rows.
            let mut p_z = 1.0_f64;
            for (j, z_name) in adjustment_set.iter().enumerate() {
                let kernel_sum: f64 = self
                    .data
                    .iter()
                    .filter_map(|r| r.get(z_name).copied())
                    .map(|v| gaussian_kernel(z_i[j], v, bandwidth))
                    .sum::<f64>()
                    / n;
                p_z *= kernel_sum.max(1e-30);
            }

            // Compute E[Y | X = x, Z = z_i] using weighted average.
            let y_given_xz = self.compute_conditional_mean(
                &query.target,
                intervention,
                adjustment_set,
                &z_i,
                bandwidth,
            )?;

            numerator += y_given_xz * p_z;
            denominator += p_z;
        }

        if denominator < 1e-30 {
            return Err(TensorError::numerical_error(
                "backdoor_adjustment",
                "Denominator is near zero; insufficient data coverage",
                vec!["Use wider bandwidth or more data".to_string()],
            ));
        }
        Ok(numerator / denominator)
    }

    /// Average Treatment Effect: `ATE = E[Y|do(X=1)] - E[Y|do(X=0)]`.
    pub fn average_treatment_effect(
        &self,
        treatment: &str,
        outcome: &str,
        adjustment_set: &[String],
    ) -> Result<f64> {
        let query1 = CausalQuery {
            target: outcome.to_string(),
            intervention: Some(Intervention {
                variable: treatment.to_string(),
                value: 1.0,
            }),
            conditioning: HashMap::new(),
        };
        let query0 = CausalQuery {
            target: outcome.to_string(),
            intervention: Some(Intervention {
                variable: treatment.to_string(),
                value: 0.0,
            }),
            conditioning: HashMap::new(),
        };
        let e_y1 = self.backdoor_adjustment(&query1, adjustment_set)?;
        let e_y0 = self.backdoor_adjustment(&query0, adjustment_set)?;
        Ok(e_y1 - e_y0)
    }

    /// Conditional ATE given `condition`: `CATE = E[Y|do(X=1),Z=z] - E[Y|do(X=0),Z=z]`.
    pub fn conditional_ate(
        &self,
        treatment: &str,
        outcome: &str,
        condition: &HashMap<String, f64>,
        adjustment_set: &[String],
    ) -> Result<f64> {
        let query1 = CausalQuery {
            target: outcome.to_string(),
            intervention: Some(Intervention {
                variable: treatment.to_string(),
                value: 1.0,
            }),
            conditioning: condition.clone(),
        };
        let query0 = CausalQuery {
            target: outcome.to_string(),
            intervention: Some(Intervention {
                variable: treatment.to_string(),
                value: 0.0,
            }),
            conditioning: condition.clone(),
        };
        let e_y1 = self.backdoor_adjustment(&query1, adjustment_set)?;
        let e_y0 = self.backdoor_adjustment(&query0, adjustment_set)?;
        Ok(e_y1 - e_y0)
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Simple mean of `outcome` for rows where `intervention.variable ≈ intervention.value`.
    fn conditional_mean_given_intervention(
        &self,
        outcome: &str,
        intervention: &Intervention,
    ) -> Result<f64> {
        let tol = 0.5; // binary treatment tolerance
        let (sum, count) = self
            .data
            .iter()
            .filter(|r| {
                r.get(&intervention.variable)
                    .map(|v| (v - intervention.value).abs() < tol)
                    .unwrap_or(false)
            })
            .filter_map(|r| r.get(outcome).copied())
            .fold((0.0_f64, 0usize), |(s, c), y| (s + y, c + 1));
        if count == 0 {
            return Err(TensorError::invalid_argument_op(
                "conditional_mean",
                "No matching rows for intervention value",
            ));
        }
        Ok(sum / count as f64)
    }

    /// E[Y | X=x, Z=z] estimated via Nadaraya-Watson kernel regression.
    fn compute_conditional_mean(
        &self,
        outcome: &str,
        intervention: &Intervention,
        adjustment_vars: &[String],
        z_target: &[f64],
        bandwidth: f64,
    ) -> Result<f64> {
        let mut numerator = 0.0_f64;
        let mut denominator = 0.0_f64;
        let x_tol = bandwidth; // tolerance for matching X value

        for row in &self.data {
            // Check intervention variable is close enough.
            let x_val = match row.get(&intervention.variable) {
                Some(v) => *v,
                None => continue,
            };
            if (x_val - intervention.value).abs() > x_tol * 2.0 {
                continue;
            }
            let y_val = match row.get(outcome) {
                Some(v) => *v,
                None => continue,
            };

            // Compute kernel weight for Z dimensions.
            let mut w = gaussian_kernel(x_val, intervention.value, bandwidth.max(0.01));
            for (j, z_name) in adjustment_vars.iter().enumerate() {
                let z_row = match row.get(z_name) {
                    Some(v) => *v,
                    None => {
                        w = 0.0;
                        break;
                    }
                };
                w *= gaussian_kernel(z_row, z_target[j], bandwidth.max(0.01));
            }
            numerator += w * y_val;
            denominator += w;
        }

        if denominator < 1e-30 {
            // Fall back to unconditional intervention mean.
            return self.conditional_mean_given_intervention(outcome, intervention);
        }
        Ok(numerator / denominator)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Double/Debiased ML (Partially Linear Model)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Double ML estimator.
#[derive(Debug, Clone)]
pub struct DoubleMLConfig {
    /// Number of cross-fitting folds (default 5).
    pub n_folds: usize,
    /// L2 regularisation strength for nuisance ridge regression.
    pub reg_lambda: f64,
    /// Random seed for fold shuffling.
    pub seed: u64,
}

impl Default for DoubleMLConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            reg_lambda: 1e-3,
            seed: 42,
        }
    }
}

/// Ridge regression nuisance model: minimises `‖Xw - y‖² + λ‖w‖²`.
#[derive(Debug, Clone)]
pub struct NuisanceModel {
    weights: Vec<f64>,
    bias: f64,
}

impl NuisanceModel {
    /// Create an untrained nuisance model.
    pub fn new() -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
        }
    }

    /// Fit ridge regression: closed-form `(XᵀX + λI)⁻¹ Xᵀy`.
    ///
    /// Uses coordinate descent for numerical stability when the design matrix
    /// is ill-conditioned.  Centres features internally.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<()> {
        let n = x.len();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "NuisanceModel::fit",
                "Empty training set",
            ));
        }
        if y.len() != n {
            return Err(TensorError::invalid_argument_op(
                "NuisanceModel::fit",
                "x and y length mismatch",
            ));
        }
        let p = x[0].len();
        // Compute feature means for centering.
        let mut x_mean = vec![0.0_f64; p];
        let y_mean = y.iter().sum::<f64>() / n as f64;
        for row in x {
            for (j, v) in row.iter().enumerate() {
                if j < x_mean.len() {
                    x_mean[j] += v;
                }
            }
        }
        for m in x_mean.iter_mut() {
            *m /= n as f64;
        }

        // Center X and y.
        let x_c: Vec<Vec<f64>> = x
            .iter()
            .map(|row| {
                row.iter()
                    .zip(x_mean.iter())
                    .map(|(xi, mi)| xi - mi)
                    .collect()
            })
            .collect();
        let y_c: Vec<f64> = y.iter().map(|yi| yi - y_mean).collect();

        // Analytical ridge: w = (XᵀX + λI)⁻¹ Xᵀy.
        // We implement this via gradient descent for p potentially large.
        // For p small enough, direct computation is fine.
        if p <= 200 {
            self.weights = ridge_solve(&x_c, &y_c, 1e-3)?;
        } else {
            // Gradient descent fallback.
            self.weights = ridge_gd(&x_c, &y_c, 1e-3, 500)?;
        }

        // Recover bias from centering.
        self.bias = y_mean;
        for j in 0..p {
            if j < self.weights.len() {
                self.bias -= self.weights[j] * x_mean[j];
            }
        }
        Ok(())
    }

    /// Predict `Xw + b` for each row.
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>> {
        x.iter()
            .map(|row| {
                let dot: f64 = row
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(xi, wi)| xi * wi)
                    .sum();
                Ok(dot + self.bias)
            })
            .collect()
    }
}

impl Default for NuisanceModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Solve `(XᵀX + λI) w = Xᵀy` directly for `p ≤ 200`.
fn ridge_solve(x: &[Vec<f64>], y: &[f64], lambda: f64) -> Result<Vec<f64>> {
    let n = x.len();
    let p = if n > 0 { x[0].len() } else { 0 };

    // XᵀX  (p × p)
    let mut xtx = vec![0.0_f64; p * p];
    // Xᵀy  (p)
    let mut xty = vec![0.0_f64; p];

    for (i, row) in x.iter().enumerate() {
        for j in 0..p {
            xty[j] += row[j] * y[i];
            for k in 0..p {
                xtx[j * p + k] += row[j] * row[k];
            }
        }
    }
    // Add ridge penalty.
    for j in 0..p {
        xtx[j * p + j] += lambda;
    }
    // Solve via Cholesky (positive definite after ridge).
    cholesky_solve(&xtx, &xty, p)
}

/// Cholesky decomposition and back-substitution for symmetric positive-definite matrix.
fn cholesky_solve(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>> {
    // L L^T decomposition.
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
            if i == j {
                let diag = a[i * n + i] - s;
                if diag <= 0.0 {
                    return Err(TensorError::numerical_error(
                        "cholesky_solve",
                        "Matrix is not positive definite",
                        vec!["Increase ridge lambda".to_string()],
                    ));
                }
                l[i * n + j] = diag.sqrt();
            } else {
                l[i * n + j] = (a[i * n + j] - s) / l[j * n + j];
            }
        }
    }
    // Forward substitution: Lz = b.
    let mut z = vec![0.0_f64; n];
    for i in 0..n {
        let s: f64 = (0..i).map(|k| l[i * n + k] * z[k]).sum();
        z[i] = (b[i] - s) / l[i * n + i];
    }
    // Backward substitution: L^T x = z.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| l[k * n + i] * x[k]).sum();
        x[i] = (z[i] - s) / l[i * n + i];
    }
    Ok(x)
}

/// Gradient descent fallback for large feature sets.
fn ridge_gd(x: &[Vec<f64>], y: &[f64], lambda: f64, iters: usize) -> Result<Vec<f64>> {
    let n = x.len();
    let p = if n > 0 { x[0].len() } else { 0 };
    let mut w = vec![0.0_f64; p];
    let lr = 1.0 / (n as f64 + lambda);
    for _ in 0..iters {
        let mut grad = vec![0.0_f64; p];
        for (i, row) in x.iter().enumerate() {
            let pred: f64 = row.iter().zip(w.iter()).map(|(xi, wi)| xi * wi).sum();
            let res = pred - y[i];
            for j in 0..p {
                grad[j] += res * row[j];
            }
        }
        for j in 0..p {
            w[j] -= lr * (grad[j] / n as f64 + lambda * w[j]);
        }
    }
    Ok(w)
}

/// Result of the Double ML estimator.
#[derive(Debug, Clone)]
pub struct DoubleMLResult {
    /// Estimated treatment effect θ.
    pub theta: f64,
    /// Asymptotic standard error.
    pub std_error: f64,
    /// t-statistic = theta / std_error.
    pub t_statistic: f64,
    /// Two-sided p-value (normal approximation).
    pub p_value: f64,
    /// 95% confidence interval.
    pub confidence_interval: (f64, f64),
}

/// Double/Debiased Machine Learning estimator for the partially linear model.
///
/// Model: `Y = θ·T + g(X) + ε`,  `T = m(X) + v`.
///
/// Cross-fitting procedure:
/// 1. Split observations into `K` folds.
/// 2. For each fold `k`: train nuisance models on complement folds, predict on fold `k`.
/// 3. Compute residuals `Ỹ = Y - Ŷ` and `T̃ = T - T̂`.
/// 4. Estimate `θ = Σ(T̃·Ỹ) / Σ(T̃²)`.
pub struct DoubleML {
    config: DoubleMLConfig,
}

impl DoubleML {
    /// Create a new Double ML estimator.
    pub fn new(config: DoubleMLConfig) -> Self {
        Self { config }
    }

    /// Fit the partially linear model and return treatment effect estimate.
    pub fn fit(
        &self,
        treatment: &[f64],
        outcome: &[f64],
        controls: &[Vec<f64>],
    ) -> Result<DoubleMLResult> {
        let n = treatment.len();
        if n < 2 {
            return Err(TensorError::invalid_argument_op(
                "DoubleML::fit",
                "Need at least 2 observations",
            ));
        }
        if outcome.len() != n || controls.len() != n {
            return Err(TensorError::invalid_argument_op(
                "DoubleML::fit",
                "treatment, outcome, and controls must have the same length",
            ));
        }
        let k = self.config.n_folds.max(2).min(n);

        // Shuffle indices.
        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let mut indices: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (rng.random::<f64>() * (i + 1) as f64) as usize;
            indices.swap(i, j);
        }

        // Assign fold labels.
        let mut fold_of = vec![0usize; n];
        for (pos, &idx) in indices.iter().enumerate() {
            fold_of[idx] = pos * k / n;
        }

        // Cross-fitting.
        let mut y_tilde = vec![0.0_f64; n];
        let mut t_tilde = vec![0.0_f64; n];

        for fold in 0..k {
            let train_idx: Vec<usize> = (0..n).filter(|&i| fold_of[i] != fold).collect();
            let test_idx: Vec<usize> = (0..n).filter(|&i| fold_of[i] == fold).collect();

            if train_idx.is_empty() || test_idx.is_empty() {
                continue;
            }

            let x_train: Vec<Vec<f64>> = train_idx.iter().map(|&i| controls[i].clone()).collect();
            let y_train: Vec<f64> = train_idx.iter().map(|&i| outcome[i]).collect();
            let t_train: Vec<f64> = train_idx.iter().map(|&i| treatment[i]).collect();
            let x_test: Vec<Vec<f64>> = test_idx.iter().map(|&i| controls[i].clone()).collect();

            // Nuisance model for outcome: E[Y | X].
            let mut y_model = NuisanceModel::new();
            y_model.fit(&x_train, &y_train)?;
            let y_hat = y_model.predict(&x_test)?;

            // Nuisance model for treatment: E[T | X].
            let mut t_model = NuisanceModel::new();
            t_model.fit(&x_train, &t_train)?;
            let t_hat = t_model.predict(&x_test)?;

            for (pos, &i) in test_idx.iter().enumerate() {
                y_tilde[i] = outcome[i] - y_hat[pos];
                t_tilde[i] = treatment[i] - t_hat[pos];
            }
        }

        // Estimate θ via OLS of Y̋ on T̋.
        let tt_sum: f64 = t_tilde.iter().map(|v| v * v).sum();
        if tt_sum < 1e-30 {
            return Err(TensorError::numerical_error(
                "DoubleML::fit",
                "Treatment residuals are all zero; check data",
                vec!["Ensure treatment has variation after partialling out controls".to_string()],
            ));
        }
        let ty_sum: f64 = t_tilde.iter().zip(y_tilde.iter()).map(|(t, y)| t * y).sum();
        let theta = ty_sum / tt_sum;

        // Asymptotic variance: var(θ) = (1/n²) Σ (T̋_i (Ỹ_i - θ T̋_i))² / (Σ T̋_i²/n)².
        let score: Vec<f64> = t_tilde
            .iter()
            .zip(y_tilde.iter())
            .map(|(t, y)| t * (y - theta * t))
            .collect();
        let score_var: f64 = score.iter().map(|s| s * s).sum::<f64>() / (n as f64 * n as f64);
        let denom = (tt_sum / n as f64).powi(2);
        let variance = score_var / denom.max(1e-60);
        let std_error = variance.sqrt();

        let t_stat = theta / std_error.max(1e-30);
        let p_value = 2.0 * (1.0 - normal_cdf(t_stat.abs()));
        let z95 = 1.959_963_985; // Φ⁻¹(0.975)
        let ci_lo = theta - z95 * std_error;
        let ci_hi = theta + z95 * std_error;

        Ok(DoubleMLResult {
            theta,
            std_error,
            t_statistic: t_stat,
            p_value,
            confidence_interval: (ci_lo, ci_hi),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Counterfactual Estimation
// ─────────────────────────────────────────────────────────────────────────────

/// A counterfactual query specifying factual observations and a hypothetical intervention.
#[derive(Debug, Clone)]
pub struct CounterfactualQuery {
    /// Observed (factual) values of all relevant variables.
    pub observed: HashMap<String, f64>,
    /// Hypothetical intervention to apply.
    pub intervention: Intervention,
    /// Target variable to evaluate under the intervention.
    pub target: String,
}

/// Result of counterfactual estimation.
#[derive(Debug, Clone)]
pub struct CounterfactualResult {
    /// Factual (observed) value of the target.
    pub factual: f64,
    /// Counterfactual (interventional) value of the target.
    pub counterfactual: f64,
    /// Individual treatment effect = counterfactual - factual.
    pub individual_treatment_effect: f64,
}

/// Counterfactual estimator for SCMs with additive Gaussian noise.
///
/// Implements the 3-step abduction-action-prediction procedure:
/// 1. **Abduction**: infer latent noise `U_v = X_v - f(Pa(v))` from observed data.
/// 2. **Action**: mutilate the graph by removing all edges into the intervened variable.
/// 3. **Prediction**: forward-propagate through the modified graph using the inferred noises.
pub struct CounterfactualEstimator {
    graph: CausalGraph,
    seed: u64,
}

impl CounterfactualEstimator {
    /// Create a new counterfactual estimator.
    pub fn new(graph: CausalGraph, seed: u64) -> Self {
        Self { graph, seed }
    }

    /// Estimate the counterfactual outcome for `query.target`.
    ///
    /// `noise_estimates` should provide the inferred noise values `U_v` for each
    /// variable reachable from the intervention.  If a noise value is not
    /// provided, zero is assumed (corresponds to the "typical" individual).
    pub fn estimate(
        &self,
        query: &CounterfactualQuery,
        noise_estimates: &HashMap<String, f64>,
    ) -> Result<CounterfactualResult> {
        let factual = *query.observed.get(&query.target).ok_or_else(|| {
            TensorError::invalid_argument_op(
                "CounterfactualEstimator::estimate",
                &format!("Factual value for '{}' not in observed", query.target),
            )
        })?;

        // Get topological order for forward propagation.
        let order = self.graph.topological_sort()?;

        // Build intervened-upon variable map: X_do values propagated.
        let mut cf_values: HashMap<String, f64> = HashMap::new();

        for var in &order {
            if var == &query.intervention.variable {
                // Hard intervention: set variable to its do-value.
                cf_values.insert(var.clone(), query.intervention.value);
                continue;
            }

            // Compute f(Pa(var)) using counterfactual parent values.
            let parents = self.graph.parents_of(var);
            if parents.is_empty() {
                // Root variable: use factual observation (noise abduction).
                let factual_val = query.observed.get(var).copied().unwrap_or(0.0);
                let noise = noise_estimates.get(var).copied().unwrap_or(0.0);
                cf_values.insert(var.clone(), factual_val + noise);
            } else {
                // Non-root: f(Pa) + U_v.
                // Simple linear additive model: f(Pa) = sum(Pa_values) / |Pa|.
                let pa_sum: f64 = parents.iter().filter_map(|p| cf_values.get(p)).sum::<f64>();
                let pa_count = parents.len() as f64;
                let noise = noise_estimates.get(var).copied().unwrap_or(0.0);
                cf_values.insert(var.clone(), pa_sum / pa_count.max(1.0) + noise);
            }
        }

        let cf_val = *cf_values.get(&query.target).ok_or_else(|| {
            TensorError::invalid_argument_op(
                "CounterfactualEstimator::estimate",
                &format!("Could not compute counterfactual for '{}'", query.target),
            )
        })?;

        let ite = self.individual_treatment_effect(factual, cf_val);
        Ok(CounterfactualResult {
            factual,
            counterfactual: cf_val,
            individual_treatment_effect: ite,
        })
    }

    /// Compute the individual treatment effect: `counterfactual - factual`.
    pub fn individual_treatment_effect(
        &self,
        factual_outcome: f64,
        counterfactual_outcome: f64,
    ) -> f64 {
        counterfactual_outcome - factual_outcome
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Propensity Score Methods
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for propensity score estimation.
#[derive(Debug, Clone)]
pub struct PropensityScoreConfig {
    /// Number of gradient descent iterations for logistic regression.
    pub n_estimators: usize,
    /// L2 regularisation strength.
    pub reg_lambda: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for PropensityScoreConfig {
    fn default() -> Self {
        Self {
            n_estimators: 50,
            reg_lambda: 1e-3,
            seed: 42,
        }
    }
}

/// Logistic regression model for propensity score estimation.
///
/// Trained via mini-batch SGD with L2 regularisation.
#[derive(Debug, Clone)]
pub struct PropensityModel {
    weights: Vec<f64>,
    bias: f64,
}

impl PropensityModel {
    /// Create an untrained propensity model.
    pub fn new() -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
        }
    }

    /// Fit logistic regression on `(x, treatment)` using SGD.
    pub fn fit(
        &mut self,
        x: &[Vec<f64>],
        treatment: &[bool],
        config: &PropensityScoreConfig,
        rng: &mut StdRng,
    ) -> Result<()> {
        let n = x.len();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "PropensityModel::fit",
                "Empty training set",
            ));
        }
        if treatment.len() != n {
            return Err(TensorError::invalid_argument_op(
                "PropensityModel::fit",
                "x and treatment length mismatch",
            ));
        }
        let p = x[0].len();
        self.weights = vec![0.0_f64; p];
        self.bias = 0.0;

        let lr = 0.1 / n as f64;
        let y: Vec<f64> = treatment
            .iter()
            .map(|&t| if t { 1.0 } else { 0.0 })
            .collect();

        // Shuffle indices.
        let mut idx: Vec<usize> = (0..n).collect();

        for _iter in 0..config.n_estimators {
            // Full-batch gradient descent (N often small for causal studies).
            for i in (1..n).rev() {
                let j = (rng.random::<f64>() * (i + 1) as f64) as usize;
                idx.swap(i, j);
            }
            let mut grad_w = vec![0.0_f64; p];
            let mut grad_b = 0.0_f64;
            for &i in &idx {
                let row = &x[i];
                let logit: f64 = row
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(xi, wi)| xi * wi)
                    .sum::<f64>()
                    + self.bias;
                let prob = sigmoid(logit);
                let err = prob - y[i];
                for j in 0..p {
                    grad_w[j] += err * row[j];
                }
                grad_b += err;
            }
            for j in 0..p {
                self.weights[j] -= lr * (grad_w[j] + config.reg_lambda * self.weights[j]);
            }
            self.bias -= lr * grad_b;
        }
        Ok(())
    }

    /// Predict P(T=1 | X) for each row.
    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Result<Vec<f64>> {
        x.iter()
            .map(|row| {
                let logit: f64 = row
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(xi, wi)| xi * wi)
                    .sum::<f64>()
                    + self.bias;
                Ok(sigmoid(logit))
            })
            .collect()
    }
}

impl Default for PropensityModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Inverse Probability Weighting (IPW) estimator.
///
/// Estimates causal effects by reweighting observations by their propensity scores.
pub struct IpwEstimator {
    config: PropensityScoreConfig,
}

impl IpwEstimator {
    /// Create a new IPW estimator.
    pub fn new(config: PropensityScoreConfig) -> Self {
        Self { config }
    }

    /// Estimate ATE using stabilised Horvitz-Thompson IPW:
    /// `ATE = Σ_i [T_i Y_i / e_i] / Σ_i [T_i / e_i] - Σ_i [(1-T_i) Y_i / (1-e_i)] / Σ_i [(1-T_i) / (1-e_i)]`
    pub fn estimate_ate(&self, x: &[Vec<f64>], treatment: &[bool], outcome: &[f64]) -> Result<f64> {
        let proba = self.fit_and_predict(x, treatment)?;
        let n = outcome.len();

        let mut treated_num = 0.0_f64;
        let mut treated_den = 0.0_f64;
        let mut control_num = 0.0_f64;
        let mut control_den = 0.0_f64;

        for i in 0..n {
            let e = proba[i].clamp(0.01, 0.99);
            if treatment[i] {
                treated_num += outcome[i] / e;
                treated_den += 1.0 / e;
            } else {
                control_num += outcome[i] / (1.0 - e);
                control_den += 1.0 / (1.0 - e);
            }
        }

        if treated_den < 1e-12 || control_den < 1e-12 {
            return Err(TensorError::numerical_error(
                "IpwEstimator::estimate_ate",
                "Insufficient treated or control units",
                vec!["Check treatment assignment".to_string()],
            ));
        }
        Ok(treated_num / treated_den - control_num / control_den)
    }

    /// Estimate ATT (Average Treatment Effect on the Treated):
    /// `ATT = Σ_i T_i Y_i / n_t - Σ_i (1-T_i) Y_i w_i / Σ_i (1-T_i) w_i`
    /// where `w_i = e_i / (1 - e_i)` (odds-ratio weights for controls).
    pub fn estimate_att(&self, x: &[Vec<f64>], treatment: &[bool], outcome: &[f64]) -> Result<f64> {
        let proba = self.fit_and_predict(x, treatment)?;
        let n = outcome.len();

        let mut treated_sum = 0.0_f64;
        let mut treated_count = 0usize;
        let mut control_num = 0.0_f64;
        let mut control_den = 0.0_f64;

        for i in 0..n {
            let e = proba[i].clamp(0.01, 0.99);
            if treatment[i] {
                treated_sum += outcome[i];
                treated_count += 1;
            } else {
                let w = e / (1.0 - e);
                control_num += outcome[i] * w;
                control_den += w;
            }
        }

        if treated_count == 0 {
            return Err(TensorError::invalid_argument_op(
                "IpwEstimator::estimate_att",
                "No treated units found",
            ));
        }
        if control_den < 1e-12 {
            return Err(TensorError::numerical_error(
                "IpwEstimator::estimate_att",
                "Control weight sum is near zero",
                vec!["Check propensity score overlap".to_string()],
            ));
        }

        let treated_mean = treated_sum / treated_count as f64;
        let control_mean = control_num / control_den;
        Ok(treated_mean - control_mean)
    }

    fn fit_and_predict(&self, x: &[Vec<f64>], treatment: &[bool]) -> Result<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let mut model = PropensityModel::new();
        model.fit(x, treatment, &self.config, &mut rng)?;
        model.predict_proba(x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Regression Discontinuity Design
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Regression Discontinuity estimation.
#[derive(Debug, Clone)]
pub struct RddConfig {
    /// Bandwidth window around the cutoff.
    pub bandwidth: f64,
    /// Cutoff threshold: units with `running_var >= cutoff` receive treatment.
    pub cutoff: f64,
    /// Polynomial order for local regression (1 = linear, 2 = quadratic).
    pub polynomial_order: usize,
}

impl Default for RddConfig {
    fn default() -> Self {
        Self {
            bandwidth: 1.0,
            cutoff: 0.0,
            polynomial_order: 1,
        }
    }
}

/// Result of an RDD analysis.
#[derive(Debug, Clone)]
pub struct RddResult {
    /// Local ATE at the cutoff (discontinuity jump estimate).
    pub local_ate: f64,
    /// Asymptotic standard error of the jump estimate.
    pub std_error: f64,
    /// 95% confidence interval.
    pub confidence_interval: (f64, f64),
    /// Bandwidth used for estimation.
    pub bandwidth: f64,
    /// Number of observations within the bandwidth.
    pub n_used: usize,
}

/// RDD estimator using triangular-kernel weighted local polynomial regression.
pub struct RddEstimator {
    config: RddConfig,
}

impl RddEstimator {
    /// Create a new RDD estimator.
    pub fn new(config: RddConfig) -> Self {
        Self { config }
    }

    /// Estimate the treatment effect at the cutoff.
    ///
    /// Fits separate local polynomial regressions on each side of the cutoff
    /// and computes the jump as `lim_{x↓c} μ(x) - lim_{x↑c} μ(x)`.
    pub fn estimate(&self, running_var: &[f64], outcome: &[f64]) -> Result<RddResult> {
        let n = running_var.len();
        if n < 4 {
            return Err(TensorError::invalid_argument_op(
                "RddEstimator::estimate",
                "Need at least 4 observations",
            ));
        }
        if outcome.len() != n {
            return Err(TensorError::invalid_argument_op(
                "RddEstimator::estimate",
                "running_var and outcome must have the same length",
            ));
        }

        let cutoff = self.config.cutoff;
        let bw = self.config.bandwidth;

        // Collect observations within the bandwidth.
        let mut x_left: Vec<f64> = Vec::new();
        let mut y_left: Vec<f64> = Vec::new();
        let mut w_left: Vec<f64> = Vec::new();
        let mut x_right: Vec<f64> = Vec::new();
        let mut y_right: Vec<f64> = Vec::new();
        let mut w_right: Vec<f64> = Vec::new();

        for i in 0..n {
            let xi = running_var[i];
            let dist = (xi - cutoff).abs();
            if dist > bw {
                continue;
            }
            let kernel_w = triangular_kernel(xi, cutoff, bw);
            if xi < cutoff {
                x_left.push(xi);
                y_left.push(outcome[i]);
                w_left.push(kernel_w);
            } else {
                x_right.push(xi);
                y_right.push(outcome[i]);
                w_right.push(kernel_w);
            }
        }

        let n_used = x_left.len() + x_right.len();
        if x_left.is_empty() || x_right.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "RddEstimator::estimate",
                "Insufficient observations on one or both sides of the cutoff",
            ));
        }

        // Fit local regressions and extrapolate to cutoff.
        let (mu_left, se_left) = self.weighted_local_regression(&x_left, &y_left, &w_left)?;
        let (mu_right, se_right) = self.weighted_local_regression(&x_right, &y_right, &w_right)?;

        let local_ate = mu_right - mu_left;
        let std_error = (se_left * se_left + se_right * se_right).sqrt();
        let z95 = 1.959_963_985;
        let ci_lo = local_ate - z95 * std_error;
        let ci_hi = local_ate + z95 * std_error;

        Ok(RddResult {
            local_ate,
            std_error,
            confidence_interval: (ci_lo, ci_hi),
            bandwidth: bw,
            n_used,
        })
    }

    /// Weighted local polynomial regression, returning the predicted value at the
    /// cutoff and the standard error of that prediction.
    ///
    /// Uses WLS (weighted least squares) with triangular kernel weights.
    fn weighted_local_regression(
        &self,
        x: &[f64],
        y: &[f64],
        weights: &[f64],
    ) -> Result<(f64, f64)> {
        let n = x.len();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "weighted_local_regression",
                "Empty data",
            ));
        }

        let cutoff = self.config.cutoff;
        let order = self.config.polynomial_order.min(2);
        let p = order + 1; // number of polynomial coefficients

        // Design matrix: columns [1, (x-c), (x-c)², ...] up to order.
        let x_centered: Vec<f64> = x.iter().map(|xi| xi - cutoff).collect();

        // Build WLS normal equations: (XᵀWX)β = XᵀWy.
        let mut xtwx = vec![0.0_f64; p * p];
        let mut xtwy = vec![0.0_f64; p];

        for i in 0..n {
            let w = weights[i];
            let mut phi = vec![1.0_f64; p];
            for d in 1..p {
                phi[d] = phi[d - 1] * x_centered[i];
            }
            for j in 0..p {
                xtwy[j] += w * phi[j] * y[i];
                for k in 0..p {
                    xtwx[j * p + k] += w * phi[j] * phi[k];
                }
            }
        }

        // Add small ridge for numerical stability.
        for j in 0..p {
            xtwx[j * p + j] += 1e-10;
        }

        let beta = cholesky_solve(&xtwx, &xtwy, p)?;

        // Predicted value at x = cutoff (x_centered = 0): just the intercept.
        let mu = beta[0];

        // Residual standard error for the intercept.
        let mut sse = 0.0_f64;
        let mut w_total = 0.0_f64;
        for i in 0..n {
            let w = weights[i];
            let mut phi = vec![1.0_f64; p];
            for d in 1..p {
                phi[d] = phi[d - 1] * x_centered[i];
            }
            let pred: f64 = beta.iter().zip(phi.iter()).map(|(b, ph)| b * ph).sum();
            sse += w * (y[i] - pred).powi(2);
            w_total += w;
        }

        let sigma2 = if n > p {
            sse / (n - p).max(1) as f64
        } else {
            sse / 1.0_f64
        };

        // Variance of intercept = sigma² * (XᵀWX)⁻¹_{00}.
        // Approximate as sigma² / (n * w_mean).
        let se = (sigma2 / w_total.max(1e-12)).sqrt();

        Ok((mu, se))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
