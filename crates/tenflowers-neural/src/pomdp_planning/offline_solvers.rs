//! Offline POMDP solvers: PbviSolver, PerseusAlgorithm, QmdpApproximation, FibAlgorithm

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

use super::types::{AlphaVector, BeliefState, PomdpAction, PomdpModel};

// ─── 4. PbviSolver ────────────────────────────────────────────────────────────

/// Point-Based Value Iteration (Pineau et al. 2003).
///
/// Maintains a finite set of belief points and performs backup operations
/// to produce alpha vectors that approximate the value function.
pub struct PbviSolver {
    model: PomdpModel,
    /// Belief points used during backups.
    pub belief_points: Vec<BeliefState>,
    /// Current set of alpha vectors.
    pub alpha_vectors: Vec<AlphaVector>,
}

impl PbviSolver {
    /// Create a new PBVI solver.
    pub fn new(model: PomdpModel) -> Self {
        let n_states = model.n_states;
        // Initialize with point-mass alpha vectors (pessimistic init)
        let alpha_vectors: Vec<AlphaVector> = (0..model.n_actions)
            .map(|a| {
                let min_r = model
                    .reward
                    .iter()
                    .map(|rs| rs[a])
                    .fold(f64::INFINITY, f64::min);
                let v = min_r / (1.0 - model.gamma.max(0.001));
                AlphaVector::new(a, vec![v; n_states])
            })
            .collect();
        Self {
            model,
            belief_points: Vec::new(),
            alpha_vectors,
        }
    }

    /// Add a belief point to the set.
    pub fn add_belief_point(&mut self, b: BeliefState) {
        self.belief_points.push(b);
    }

    /// Simulate random trajectories to collect diverse belief points.
    pub fn sample_belief_points(&mut self, n_points: usize, horizon: usize, rng: &mut StdRng) {
        let model = &self.model;
        for _ in 0..n_points {
            let mut b = BeliefState::uniform(model.n_states);
            self.belief_points.push(b.clone());
            for _ in 0..horizon {
                let a = rng.random_range(0..model.n_actions);
                let s = b.sample(rng);
                // Simulate transition
                let cumsum: Vec<f64> = model.transition[s][a]
                    .iter()
                    .scan(0.0, |acc, &p| {
                        *acc += p;
                        Some(*acc)
                    })
                    .collect();
                let u: f64 = rng.random();
                let sp = cumsum
                    .iter()
                    .position(|&c| u <= c)
                    .unwrap_or(model.n_states - 1);
                // Sample observation
                let cumsum_o: Vec<f64> = model.observation[a][sp]
                    .iter()
                    .scan(0.0, |acc, &p| {
                        *acc += p;
                        Some(*acc)
                    })
                    .collect();
                let uo: f64 = rng.random();
                let o = cumsum_o
                    .iter()
                    .position(|&c| uo <= c)
                    .unwrap_or(model.n_obs - 1);
                b = b.update(a, o, model);
                self.belief_points.push(b.clone());
            }
        }
    }

    /// Perform a single PBVI backup over all belief points.
    /// Returns the new set of alpha vectors.
    fn backup(&self) -> Vec<AlphaVector> {
        let model = &self.model;
        let n_s = model.n_states;
        let gamma = model.gamma;

        // For each belief point, compute the best alpha vector via cross-product backup
        let mut new_alphas: Vec<AlphaVector> = Vec::new();

        for b in &self.belief_points {
            let best = self.backup_belief(b, &self.alpha_vectors, n_s, gamma);
            // Add if not already represented (check by action and closeness)
            let dominated = new_alphas.iter().any(|existing: &AlphaVector| {
                existing.action == best.action
                    && existing
                        .coeffs
                        .iter()
                        .zip(best.coeffs.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-10)
            });
            if !dominated {
                new_alphas.push(best);
            }
        }

        if new_alphas.is_empty() {
            self.alpha_vectors.clone()
        } else {
            new_alphas
        }
    }

    /// Backup a single belief point: compute best alpha for this belief.
    pub fn backup_belief(
        &self,
        b: &BeliefState,
        alphas: &[AlphaVector],
        n_s: usize,
        gamma: f64,
    ) -> AlphaVector {
        let model = &self.model;
        let mut best_val = f64::NEG_INFINITY;
        let mut best_alpha = AlphaVector::new(0, vec![0.0; n_s]);

        for a in 0..model.n_actions {
            // For each observation o, find the best alpha vector for b^{a,o}
            // alpha_ao[s'] = Σ_o Z(a,s',o) * max_alpha (alpha · b^{a,o})_s'
            let mut alpha_a = vec![0.0f64; n_s];

            for s in 0..n_s {
                alpha_a[s] += model.reward[s][a];
            }

            for o in 0..model.n_obs {
                // Compute the belief b' = τ(b, a, o) (unnormalized intermediate)
                // For backup we need: best alpha for each (a,o) weighted belief
                let b_ao_prob: Vec<f64> = (0..n_s)
                    .map(|sp| {
                        let predicted: f64 = (0..n_s)
                            .map(|s| b.prob[s] * model.transition[s][a][sp])
                            .sum();
                        predicted * model.observation[a][sp][o]
                    })
                    .collect();
                let b_ao_sum: f64 = b_ao_prob.iter().sum();

                if b_ao_sum < 1e-15 {
                    continue;
                }
                let b_ao = BeliefState::from_vec(b_ao_prob.clone());

                // Find best alpha for this belief
                let (_, best_idx) = AlphaVector::max_value(alphas, &b_ao);
                let best_alpha_ao = &alphas[best_idx];

                // Accumulate: alpha_a[s] += gamma * sum_{s'} T(s,a,s') Z(a,s',o) alpha*(s')
                for s in 0..n_s {
                    for sp in 0..n_s {
                        alpha_a[s] += gamma
                            * model.transition[s][a][sp]
                            * model.observation[a][sp][o]
                            * best_alpha_ao.coeffs[sp];
                    }
                }
            }

            let val: f64 = alpha_a.iter().zip(b.prob.iter()).map(|(c, p)| c * p).sum();
            if val > best_val {
                best_val = val;
                best_alpha = AlphaVector::new(a, alpha_a);
            }
        }

        best_alpha
    }

    /// Run PBVI for `max_iters` iterations or until improvement < `tolerance`.
    pub fn solve(&mut self, max_iters: usize, tolerance: f64) -> &[AlphaVector] {
        if self.belief_points.is_empty() {
            let mut rng = StdRng::seed_from_u64(42);
            self.sample_belief_points(10, 5, &mut rng);
        }

        for _ in 0..max_iters {
            let new_alphas = self.backup();
            // Check convergence: max change in value at belief points
            let max_change = self
                .belief_points
                .iter()
                .map(|b| {
                    let old_val = AlphaVector::max_value(&self.alpha_vectors, b).0;
                    let new_val = AlphaVector::max_value(&new_alphas, b).0;
                    (new_val - old_val).abs()
                })
                .fold(0.0_f64, f64::max);

            self.alpha_vectors = new_alphas;
            if max_change < tolerance {
                break;
            }
        }
        &self.alpha_vectors
    }

    /// Extract the best action for a given belief.
    pub fn best_action(&self, b: &BeliefState) -> PomdpAction {
        if self.alpha_vectors.is_empty() {
            return 0;
        }
        let (_, idx) = AlphaVector::max_value(&self.alpha_vectors, b);
        self.alpha_vectors[idx].action
    }
}

// ─── Helper: PbviSolverHelper ─────────────────────────────────────────────────

/// Helper struct for single-belief backup, used by Perseus and SARSOP.
pub(super) struct PbviSolverHelper<'a> {
    pub model: &'a PomdpModel,
}

impl<'a> PbviSolverHelper<'a> {
    pub fn new(model: &'a PomdpModel) -> Self {
        Self { model }
    }

    pub fn backup_single(&self, b: &BeliefState, alphas: &[AlphaVector]) -> AlphaVector {
        let model = self.model;
        let n_s = model.n_states;
        let gamma = model.gamma;
        let mut best_val = f64::NEG_INFINITY;
        let mut best_alpha = AlphaVector::new(0, vec![0.0; n_s]);

        for a in 0..model.n_actions {
            let mut alpha_a = vec![0.0f64; n_s];
            for s in 0..n_s {
                alpha_a[s] += model.reward[s][a];
            }
            for o in 0..model.n_obs {
                let b_ao_prob: Vec<f64> = (0..n_s)
                    .map(|sp| {
                        let pred: f64 = (0..n_s)
                            .map(|s| b.prob[s] * model.transition[s][a][sp])
                            .sum();
                        pred * model.observation[a][sp][o]
                    })
                    .collect();
                let b_ao_sum: f64 = b_ao_prob.iter().sum();
                if b_ao_sum < 1e-15 {
                    continue;
                }
                let b_ao = BeliefState::from_vec(b_ao_prob);
                let (_, best_idx) = AlphaVector::max_value(alphas, &b_ao);
                let best_alpha_ao = &alphas[best_idx];
                for s in 0..n_s {
                    for sp in 0..n_s {
                        alpha_a[s] += gamma
                            * model.transition[s][a][sp]
                            * model.observation[a][sp][o]
                            * best_alpha_ao.coeffs[sp];
                    }
                }
            }
            let val: f64 = alpha_a.iter().zip(b.prob.iter()).map(|(c, p)| c * p).sum();
            if val > best_val {
                best_val = val;
                best_alpha = AlphaVector::new(a, alpha_a);
            }
        }
        best_alpha
    }
}

// ─── 5. PerseusAlgorithm ──────────────────────────────────────────────────────

/// Perseus randomized point-based POMDP solver (Spaan & Vlassis 2005).
///
/// Selects unimproved belief points randomly and adds new alpha vectors
/// only when they improve the value for at least one belief point.
pub struct PerseusAlgorithm {
    model: PomdpModel,
    /// Belief points
    pub belief_points: Vec<BeliefState>,
    /// Current alpha vector set
    pub alpha_vectors: Vec<AlphaVector>,
}

impl PerseusAlgorithm {
    /// Create a new Perseus solver.
    pub fn new(model: PomdpModel) -> Self {
        let n_states = model.n_states;
        let alpha_vectors: Vec<AlphaVector> = (0..model.n_actions)
            .map(|a| {
                let min_r = model
                    .reward
                    .iter()
                    .map(|rs| rs[a])
                    .fold(f64::INFINITY, f64::min);
                let v = min_r / (1.0 - model.gamma.max(0.001));
                AlphaVector::new(a, vec![v; n_states])
            })
            .collect();
        Self {
            model,
            belief_points: Vec::new(),
            alpha_vectors,
        }
    }

    /// Add a belief point.
    pub fn add_belief_point(&mut self, b: BeliefState) {
        self.belief_points.push(b);
    }

    /// Sample belief points via random simulation.
    pub fn sample_beliefs(&mut self, n: usize, rng: &mut StdRng) {
        let model = &self.model;
        for _ in 0..n {
            let mut b = BeliefState::uniform(model.n_states);
            self.belief_points.push(b.clone());
            for _ in 0..8 {
                let a = rng.random_range(0..model.n_actions);
                let s = b.sample(rng);
                let u: f64 = rng.random();
                let mut cum = 0.0;
                let mut sp = model.n_states - 1;
                for (spp, &p) in model.transition[s][a].iter().enumerate() {
                    cum += p;
                    if u <= cum {
                        sp = spp;
                        break;
                    }
                }
                let uo: f64 = rng.random();
                let mut cum_o = 0.0;
                let mut o = model.n_obs - 1;
                for (oo, &p) in model.observation[a][sp].iter().enumerate() {
                    cum_o += p;
                    if uo <= cum_o {
                        o = oo;
                        break;
                    }
                }
                b = b.update(a, o, model);
                self.belief_points.push(b.clone());
            }
        }
    }

    /// One Perseus iteration: randomized backup.
    fn iterate(&mut self, rng: &mut StdRng) {
        // Compute current values at all belief points
        let old_vals: Vec<f64> = self
            .belief_points
            .iter()
            .map(|b| AlphaVector::max_value(&self.alpha_vectors, b).0)
            .collect();

        let mut improved = vec![false; self.belief_points.len()];
        let mut new_alphas = self.alpha_vectors.clone();

        // Shuffle indices for randomized selection
        let mut indices: Vec<usize> = (0..self.belief_points.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = rng.random_range(0..=i);
            indices.swap(i, j);
        }

        for &idx in &indices {
            if improved[idx] {
                continue;
            }
            let b = &self.belief_points[idx].clone();
            let helper = PbviSolverHelper::new(&self.model);
            let candidate = helper.backup_single(b, &self.alpha_vectors);

            let candidate_val = candidate.value(b);
            // Only add if it improves the value for this belief
            if candidate_val >= old_vals[idx] - 1e-10 {
                new_alphas.push(candidate);
                // Mark all beliefs improved by this new vector
                for (i, bp) in self.belief_points.iter().enumerate() {
                    let new_v = AlphaVector::max_value(&new_alphas, bp).0;
                    if new_v >= old_vals[i] - 1e-10 {
                        improved[i] = true;
                    }
                }
            }
        }

        self.alpha_vectors = new_alphas;
    }

    /// Run Perseus for `max_iters` iterations.
    pub fn solve(&mut self, max_iters: usize, rng: &mut StdRng) -> &[AlphaVector] {
        if self.belief_points.is_empty() {
            self.sample_beliefs(10, rng);
        }
        for _ in 0..max_iters {
            self.iterate(rng);
        }
        &self.alpha_vectors
    }

    /// Best action for a belief.
    pub fn best_action(&self, b: &BeliefState) -> PomdpAction {
        if self.alpha_vectors.is_empty() {
            return 0;
        }
        let (_, idx) = AlphaVector::max_value(&self.alpha_vectors, b);
        self.alpha_vectors[idx].action
    }
}

// ─── 6. QmdpApproximation ─────────────────────────────────────────────────────

/// QMDP approximation: solve MDP as if fully observable, use Q-values as alpha vectors.
///
/// Fast but ignores information gathering — optimal when state becomes fully observable.
pub struct QmdpApproximation {
    model: PomdpModel,
    /// Computed Q-values: Q\[s\]\[a\]
    pub q_values: Vec<Vec<f64>>,
}

impl QmdpApproximation {
    /// Create a new QMDP solver and immediately compute Q-values.
    pub fn new(model: PomdpModel) -> Self {
        let n_s = model.n_states;
        let n_a = model.n_actions;
        let q_values = vec![vec![0.0; n_a]; n_s];
        let mut solver = Self { model, q_values };
        solver.compute_qmdp_internal(1000, 1e-6);
        solver
    }

    fn compute_qmdp_internal(&mut self, max_iters: usize, tol: f64) {
        let n_s = self.model.n_states;
        let n_a = self.model.n_actions;
        let gamma = self.model.gamma;
        let mut v = vec![0.0f64; n_s];

        for _ in 0..max_iters {
            let v_old = v.clone();
            for s in 0..n_s {
                let best_q = (0..n_a)
                    .map(|a| {
                        self.model.reward[s][a]
                            + gamma
                                * (0..n_s)
                                    .map(|sp| self.model.transition[s][a][sp] * v_old[sp])
                                    .sum::<f64>()
                    })
                    .fold(f64::NEG_INFINITY, f64::max);
                v[s] = best_q;
            }
            let max_change = v
                .iter()
                .zip(v_old.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            if max_change < tol {
                break;
            }
        }

        // Compute Q values
        for s in 0..n_s {
            for a in 0..n_a {
                self.q_values[s][a] = self.model.reward[s][a]
                    + gamma
                        * (0..n_s)
                            .map(|sp| self.model.transition[s][a][sp] * v[sp])
                            .sum::<f64>();
            }
        }
    }

    /// Compute QMDP alpha vectors: one per action with Q\[s\]\[a\] as coefficients.
    pub fn compute_qmdp(model: PomdpModel) -> Vec<AlphaVector> {
        let solver = Self::new(model);
        let n_a = solver.model.n_actions;
        let n_s = solver.model.n_states;
        (0..n_a)
            .map(|a| {
                let coeffs: Vec<f64> = (0..n_s).map(|s| solver.q_values[s][a]).collect();
                AlphaVector::new(a, coeffs)
            })
            .collect()
    }

    /// Best action for a belief.
    pub fn best_action(&self, b: &BeliefState) -> PomdpAction {
        let n_a = self.model.n_actions;
        (0..n_a)
            .map(|a| {
                let val: f64 = b
                    .prob
                    .iter()
                    .enumerate()
                    .map(|(s, &p)| p * self.q_values[s][a])
                    .sum();
                (val, a)
            })
            .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, a)| a)
            .unwrap_or(0)
    }
}

// ─── 7. FibAlgorithm ──────────────────────────────────────────────────────────

/// Fast Informed Bound (Hauskrecht 2000).
///
/// Computes a tighter upper bound than QMDP by incorporating observation likelihoods.
pub struct FibAlgorithm {
    model: PomdpModel,
    /// Alpha vectors from FIB
    pub alpha_vectors: Vec<AlphaVector>,
}

impl FibAlgorithm {
    /// Create a new FIB solver.
    pub fn new(model: PomdpModel) -> Self {
        let n_states = model.n_states;
        let n_actions = model.n_actions;
        // Initialize with QMDP as starting point
        let q_alphas = QmdpApproximation::compute_qmdp(model.clone());
        let alpha_vectors = if q_alphas.is_empty() {
            (0..n_actions)
                .map(|a| AlphaVector::new(a, vec![0.0; n_states]))
                .collect()
        } else {
            q_alphas
        };
        Self {
            model,
            alpha_vectors,
        }
    }

    /// Compute FIB alpha vectors via value iteration.
    pub fn compute_fib(model: PomdpModel, max_iters: usize) -> Vec<AlphaVector> {
        let mut fib = Self::new(model);
        for _ in 0..max_iters {
            fib.fib_backup();
        }
        fib.alpha_vectors
    }

    fn fib_backup(&mut self) {
        let model = &self.model;
        let n_s = model.n_states;
        let n_a = model.n_actions;
        let n_o = model.n_obs;
        let gamma = model.gamma;

        let old_alphas = self.alpha_vectors.clone();
        let mut new_alphas: Vec<AlphaVector> = Vec::new();

        for a in 0..n_a {
            let mut alpha_a = vec![0.0f64; n_s];
            for s in 0..n_s {
                alpha_a[s] += model.reward[s][a];
                for o in 0..n_o {
                    // For each (s, a, o): find best alpha for next belief
                    // FIB uses: max over alpha in old set, weighted by obs
                    let mut best_contribution = f64::NEG_INFINITY;
                    for prev_alpha in &old_alphas {
                        let contrib: f64 = (0..n_s)
                            .map(|sp| {
                                model.transition[s][a][sp]
                                    * model.observation[a][sp][o]
                                    * prev_alpha.coeffs[sp]
                            })
                            .sum();
                        if contrib > best_contribution {
                            best_contribution = contrib;
                        }
                    }
                    if best_contribution > f64::NEG_INFINITY {
                        alpha_a[s] += gamma * best_contribution;
                    }
                }
            }
            new_alphas.push(AlphaVector::new(a, alpha_a));
        }
        self.alpha_vectors = new_alphas;
    }

    /// Best action for a given belief.
    pub fn best_action(&self, b: &BeliefState) -> PomdpAction {
        if self.alpha_vectors.is_empty() {
            return 0;
        }
        let (_, idx) = AlphaVector::max_value(&self.alpha_vectors, b);
        self.alpha_vectors[idx].action
    }
}
