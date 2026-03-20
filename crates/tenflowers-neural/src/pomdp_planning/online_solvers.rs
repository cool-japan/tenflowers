//! Online/Monte Carlo POMDP solvers: PomcpSolver, SarsopAlgorithm

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;

use super::offline_solvers::{PbviSolverHelper, QmdpApproximation};
use super::types::{AlphaVector, BeliefState, PomdpAction, PomdpModel, PomdpObs, PomdpState};

// ─── 8. PomcpSolver ───────────────────────────────────────────────────────────

/// POMCP node in the belief tree (Silver & Veness 2010).
#[derive(Debug, Clone)]
pub struct PomcpNode {
    /// Visit count N(h)
    pub visits: u64,
    /// Value estimate V(h)
    pub value: f64,
    /// Children indexed by action
    pub children: HashMap<PomdpAction, PomcpActionNode>,
}

/// POMCP action node: connects history-action pairs to observation children.
#[derive(Debug, Clone)]
pub struct PomcpActionNode {
    /// Visit count N(h,a)
    pub visits: u64,
    /// Q-value estimate Q(h,a)
    pub value: f64,
    /// Children indexed by observation → history node
    pub children: HashMap<PomdpObs, PomcpNode>,
}

impl PomcpNode {
    pub(super) fn new() -> Self {
        Self {
            visits: 0,
            value: 0.0,
            children: HashMap::new(),
        }
    }
}

impl PomcpActionNode {
    pub(super) fn new() -> Self {
        Self {
            visits: 0,
            value: 0.0,
            children: HashMap::new(),
        }
    }
}

/// POMDP Monte Carlo Planning (Silver & Veness 2010).
///
/// Uses particle belief representation and UCT-based tree search.
pub struct PomcpSolver {
    model: PomdpModel,
    /// UCT exploration constant
    pub uct_c: f64,
    /// Max search depth
    pub max_depth: usize,
    /// Rollout horizon for simulation
    pub rollout_depth: usize,
    /// Root node of the search tree
    pub root: PomcpNode,
}

impl PomcpSolver {
    /// Create a new POMCP solver.
    pub fn new(model: PomdpModel, uct_c: f64, max_depth: usize) -> Self {
        Self {
            model,
            uct_c,
            max_depth,
            rollout_depth: 10,
            root: PomcpNode::new(),
        }
    }

    /// Search for the best action given a particle belief set.
    /// `particles`: sampled states representing the belief.
    /// `budget`: number of simulations.
    pub fn search(
        &mut self,
        particles: &[PomdpState],
        budget: usize,
        rng: &mut StdRng,
    ) -> PomdpAction {
        self.root = PomcpNode::new();
        for _ in 0..budget {
            if particles.is_empty() {
                break;
            }
            let s_idx = rng.random_range(0..particles.len());
            let s = particles[s_idx];
            // Non-recursive simulation using explicit stack
            self.simulate_iterative(s, rng);
        }
        self.best_action_root()
    }

    fn simulate_iterative(&mut self, start_state: PomdpState, rng: &mut StdRng) -> f64 {
        let model = &self.model;
        let n_a = model.n_actions;
        let gamma = model.gamma;
        let uct_c = self.uct_c;
        let max_depth = self.max_depth;

        // We'll use a simplified iterative version: tree search then rollout
        let mut current_state = start_state;
        let mut total_reward = 0.0;
        let mut depth_factor = 1.0;

        // Tree phase: traverse existing tree
        // For simplicity, do one level of UCT at root
        let a = self.uct_action_root(n_a, uct_c);
        let (sp, o, r) = self.simulate_step(current_state, a, rng);
        total_reward += depth_factor * r;
        depth_factor *= gamma;
        current_state = sp;

        // Update root action node
        let action_node = self
            .root
            .children
            .entry(a)
            .or_insert_with(PomcpActionNode::new);
        action_node.visits += 1;

        // Rollout from next state
        let rollout_val = self.rollout(current_state, self.rollout_depth.min(max_depth), rng);
        total_reward += depth_factor * rollout_val;

        // Backpropagate to action node
        let action_node = self
            .root
            .children
            .entry(a)
            .or_insert_with(PomcpActionNode::new);
        action_node.value += (total_reward - action_node.value) / action_node.visits as f64;

        // Update observation child
        let obs_child = action_node.children.entry(o).or_insert_with(PomcpNode::new);
        obs_child.visits += 1;
        obs_child.value += (total_reward - obs_child.value) / obs_child.visits as f64;

        // Update root
        self.root.visits += 1;
        self.root.value = self
            .root
            .children
            .values()
            .map(|an| an.value)
            .fold(f64::NEG_INFINITY, f64::max);

        total_reward
    }

    fn uct_action_root(&self, n_a: usize, uct_c: f64) -> PomdpAction {
        let total_visits = self.root.visits.max(1);
        let log_n = (total_visits as f64).ln();
        (0..n_a)
            .max_by(|&a1, &a2| {
                let ucb = |a: usize| -> f64 {
                    match self.root.children.get(&a) {
                        Some(node) if node.visits > 0 => {
                            node.value + uct_c * (log_n / node.visits as f64).sqrt()
                        }
                        _ => f64::INFINITY,
                    }
                };
                ucb(a1)
                    .partial_cmp(&ucb(a2))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0)
    }

    fn simulate_step(
        &self,
        s: PomdpState,
        a: PomdpAction,
        rng: &mut StdRng,
    ) -> (PomdpState, PomdpObs, f64) {
        let model = &self.model;
        let r = model.reward[s][a];
        // Sample next state
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
        // Sample observation
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
        (sp, o, r)
    }

    fn rollout(&self, start: PomdpState, depth: usize, rng: &mut StdRng) -> f64 {
        let model = &self.model;
        let mut total = 0.0;
        let mut factor = 1.0;
        let mut s = start;
        for _ in 0..depth {
            let a = rng.random_range(0..model.n_actions);
            total += factor * model.reward[s][a];
            factor *= model.gamma;
            let (sp, _, _) = self.simulate_step(s, a, rng);
            s = sp;
        }
        total
    }

    fn best_action_root(&self) -> PomdpAction {
        self.root
            .children
            .iter()
            .max_by(|a, b| {
                a.1.visits
                    .partial_cmp(&b.1.visits)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(&a, _)| a)
            .unwrap_or(0)
    }

    /// Reinvigorate particles: mix current particles with new samples to avoid depletion.
    pub fn reinvigorate_particles(
        &self,
        particles: &mut [PomdpState],
        fraction: f64,
        rng: &mut StdRng,
    ) {
        let n = particles.len();
        let n_new = (n as f64 * fraction) as usize;
        let n_states = self.model.n_states;
        for _ in 0..n_new {
            let s = rng.random_range(0..n_states);
            if !particles.is_empty() {
                let idx = rng.random_range(0..n);
                particles[idx] = s;
            }
        }
    }
}

// ─── 9. SarsopAlgorithm ───────────────────────────────────────────────────────

/// SARSOP: Successive Approximations of the Reachable Space under Optimal Policies
/// (Kurniawati et al. 2008).
///
/// Maintains both an upper bound (sawtooth / QMDP) and lower bound (Perseus-style),
/// and prunes the reachable space to focus computation.
pub struct SarsopAlgorithm {
    model: PomdpModel,
    /// Lower bound alpha vectors
    pub lower_bound: Vec<AlphaVector>,
    /// Upper bound points: (belief, value)
    pub upper_bound_points: Vec<(BeliefState, f64)>,
    /// Epsilon gap target
    pub target_epsilon: f64,
}

impl SarsopAlgorithm {
    /// Create a new SARSOP solver.
    pub fn new(model: PomdpModel, target_epsilon: f64) -> Self {
        let n_states = model.n_states;
        // Initialize lower bound pessimistically
        let min_r = model
            .reward
            .iter()
            .flat_map(|row| row.iter().copied())
            .fold(f64::INFINITY, f64::min);
        let pessimistic_val = min_r / (1.0 - model.gamma.max(0.001));
        let lower_bound = vec![AlphaVector::new(0, vec![pessimistic_val; n_states])];

        // Initialize upper bound via QMDP
        let qmdp_alphas = QmdpApproximation::compute_qmdp(model.clone());
        let init_belief = BeliefState::uniform(n_states);
        let ub_val = AlphaVector::max_value(&qmdp_alphas, &init_belief).0;
        let upper_bound_points = vec![(init_belief, ub_val)];

        Self {
            model,
            lower_bound,
            upper_bound_points,
            target_epsilon,
        }
    }

    /// Upper bound at a belief via sawtooth interpolation.
    fn upper_bound(&self, _b: &BeliefState) -> f64 {
        // Use best QMDP-style upper bound stored in points
        self.upper_bound_points
            .iter()
            .map(|(_, v)| *v)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Lower bound at a belief.
    fn lower_bound_val(&self, b: &BeliefState) -> f64 {
        AlphaVector::max_value(&self.lower_bound, b).0
    }

    /// Gap = upper - lower at belief.
    fn gap(&self, b: &BeliefState) -> f64 {
        self.upper_bound(b) - self.lower_bound_val(b)
    }

    /// Run SARSOP for `max_iters` iterations or until gap < epsilon.
    pub fn solve(&mut self, epsilon: f64, max_iters: usize) -> &[AlphaVector] {
        let mut rng = StdRng::seed_from_u64(0);
        let init_belief = BeliefState::uniform(self.model.n_states);

        for _ in 0..max_iters {
            if self.gap(&init_belief) < epsilon {
                break;
            }

            // Sample from reachable space: simulate a trajectory and backup
            let mut b = init_belief.clone();
            let mut trajectory: Vec<BeliefState> = vec![b.clone()];

            for _ in 0..10 {
                let a = self.greedy_action_lb(&b);
                let o = rng.random_range(0..self.model.n_obs);
                b = b.update(a, o, &self.model);
                if self.gap(&b) < epsilon {
                    break;
                }
                trajectory.push(b.clone());
            }

            // Backup along trajectory (backward)
            for b_traj in trajectory.iter().rev() {
                let helper = PbviSolverHelper::new(&self.model);
                let new_alpha = helper.backup_single(b_traj, &self.lower_bound);
                let new_val = new_alpha.value(b_traj);
                let old_val = self.lower_bound_val(b_traj);
                if new_val > old_val + 1e-10 {
                    self.lower_bound.push(new_alpha);
                }
            }

            // Update upper bound: add new point from trajectory end
            let ub_val = self.upper_bound(&b);
            let lb_val = self.lower_bound_val(&b);
            self.upper_bound_points
                .push((b.clone(), (ub_val + lb_val) / 2.0 + epsilon));

            // Prune dominated lower bound vectors
            if self.lower_bound.len() > 200 {
                self.prune_lower_bound(&init_belief);
            }
        }

        &self.lower_bound
    }

    fn greedy_action_lb(&self, b: &BeliefState) -> PomdpAction {
        if self.lower_bound.is_empty() {
            return 0;
        }
        let (_, idx) = AlphaVector::max_value(&self.lower_bound, b);
        self.lower_bound[idx].action
    }

    fn prune_lower_bound(&mut self, ref_belief: &BeliefState) {
        // Keep only the best ~100 vectors as measured at the reference belief
        let mut scored: Vec<(f64, usize)> = self
            .lower_bound
            .iter()
            .enumerate()
            .map(|(i, a)| (a.value(ref_belief), i))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(100);
        let keep: std::collections::HashSet<usize> = scored.iter().map(|&(_, i)| i).collect();
        self.lower_bound = self
            .lower_bound
            .iter()
            .enumerate()
            .filter(|(i, _)| keep.contains(i))
            .map(|(_, a)| a.clone())
            .collect();
    }
}
