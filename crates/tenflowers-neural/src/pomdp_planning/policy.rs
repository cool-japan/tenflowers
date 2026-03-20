//! Policy representations: PomdpPolicyGraph, BeliefMdpSolver, OnlineBeliefTreeSearch

use scirs2_core::random::{rngs::StdRng, Rng};
use scirs2_core::RngExt;

use super::offline_solvers::QmdpApproximation;
use super::types::{AlphaVector, BeliefState, PomdpAction, PomdpModel, PomdpObs};

// ─── 10. PomdpPolicyGraph ─────────────────────────────────────────────────────

/// Finite-State Controller (FSC) node for policy graph.
#[derive(Debug, Clone)]
pub struct FscNode {
    /// Action taken at this node.
    pub action: PomdpAction,
    /// Observation → next FSC node index.
    pub transitions: Vec<usize>,
}

/// Policy graph (finite-state controller) for a POMDP.
///
/// A finite-state controller maintains internal memory nodes and selects
/// actions based on the current node.
pub struct PomdpPolicyGraph {
    model: PomdpModel,
    /// FSC nodes
    pub nodes: Vec<FscNode>,
    /// Value function for each (fsc_node, state) pair
    pub node_values: Vec<Vec<f64>>,
}

impl PomdpPolicyGraph {
    /// Create a new policy graph with `n_nodes` nodes, initialized with random actions.
    pub fn new(model: PomdpModel, n_nodes: usize, rng: &mut StdRng) -> Self {
        let n_a = model.n_actions;
        let n_o = model.n_obs;
        let n_s = model.n_states;
        let nodes: Vec<FscNode> = (0..n_nodes)
            .map(|_| FscNode {
                action: rng.random_range(0..n_a),
                transitions: (0..n_o).map(|_| rng.random_range(0..n_nodes)).collect(),
            })
            .collect();
        let node_values = vec![vec![0.0f64; n_s]; n_nodes];
        Self {
            model,
            nodes,
            node_values,
        }
    }

    /// Evaluate the policy graph via value iteration on the FSC × State product MDP.
    pub fn evaluate_policy_graph(&mut self, max_iters: usize, tol: f64) {
        let n_nodes = self.nodes.len();
        let n_s = self.model.n_states;
        let gamma = self.model.gamma;

        for _ in 0..max_iters {
            let old_vals = self.node_values.clone();
            let mut max_change = 0.0f64;

            for q in 0..n_nodes {
                let a = self.nodes[q].action;
                for s in 0..n_s {
                    let mut val = self.model.reward[s][a];
                    for sp in 0..n_s {
                        let t = self.model.transition[s][a][sp];
                        if t < 1e-15 {
                            continue;
                        }
                        let sum_o: f64 = (0..self.model.n_obs)
                            .map(|o| {
                                let z = self.model.observation[a][sp][o];
                                let qp = self.nodes[q].transitions[o];
                                z * old_vals[qp][sp]
                            })
                            .sum();
                        val += gamma * t * sum_o;
                    }
                    let change = (val - self.node_values[q][s]).abs();
                    if change > max_change {
                        max_change = change;
                    }
                    self.node_values[q][s] = val;
                }
            }

            if max_change < tol {
                break;
            }
        }
    }

    /// Improve the policy graph by node splitting: add a new node with best action.
    pub fn improve_policy_graph(&mut self, rng: &mut StdRng) {
        let n_nodes = self.nodes.len();
        let n_a = self.model.n_actions;
        let n_o = self.model.n_obs;
        let n_s = self.model.n_states;

        // Find the node with the lowest average value
        let worst_node = (0..n_nodes)
            .min_by(|&q1, &q2| {
                let v1: f64 = self.node_values[q1].iter().sum::<f64>() / n_s as f64;
                let v2: f64 = self.node_values[q2].iter().sum::<f64>() / n_s as f64;
                v1.partial_cmp(&v2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        // Try all actions and pick the one with best expected value
        let best_action = (0..n_a)
            .max_by(|&a1, &a2| {
                let val =
                    |a: usize| -> f64 { (0..n_s).map(|s| self.model.reward[s][a]).sum::<f64>() };
                val(a1)
                    .partial_cmp(&val(a2))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        // Split: replace worst node with improved version
        self.nodes[worst_node].action = best_action;
        for o in 0..n_o {
            self.nodes[worst_node].transitions[o] = rng.random_range(0..n_nodes);
        }
    }

    /// Get action for a given FSC node.
    pub fn action_at(&self, node: usize) -> PomdpAction {
        if node < self.nodes.len() {
            self.nodes[node].action
        } else {
            0
        }
    }

    /// Get next FSC node after observation `o` at node `q`.
    pub fn next_node(&self, q: usize, o: PomdpObs) -> usize {
        if q < self.nodes.len() && o < self.nodes[q].transitions.len() {
            self.nodes[q].transitions[o]
        } else {
            0
        }
    }
}

// ─── 11. BeliefMdpSolver ──────────────────────────────────────────────────────

/// Belief-MDP solver: enumerate belief states and apply tabular value iteration.
///
/// Exact for finite belief sets (small POMDPs); uses simulation to enumerate beliefs.
pub struct BeliefMdpSolver {
    model: PomdpModel,
    /// Enumerated belief states
    pub beliefs: Vec<BeliefState>,
    /// Value at each belief state
    pub values: Vec<f64>,
    /// Best action at each belief state
    pub policy: Vec<PomdpAction>,
}

impl BeliefMdpSolver {
    /// Create a new Belief-MDP solver.
    pub fn new(model: PomdpModel) -> Self {
        Self {
            model,
            beliefs: Vec::new(),
            values: Vec::new(),
            policy: Vec::new(),
        }
    }

    /// Enumerate belief states via simulation.
    pub fn enumerate_beliefs(&mut self, n_trajectories: usize, horizon: usize, rng: &mut StdRng) {
        let model = &self.model;
        let mut beliefs = vec![BeliefState::uniform(model.n_states)];

        for _ in 0..n_trajectories {
            let mut b = BeliefState::uniform(model.n_states);
            for _ in 0..horizon {
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
                // Add if distinct enough from existing
                let is_new = beliefs.iter().all(|existing| b.distance(existing) > 0.05);
                if is_new {
                    beliefs.push(b.clone());
                }
            }
        }

        self.beliefs = beliefs;
        self.values = vec![0.0; self.beliefs.len()];
        self.policy = vec![0; self.beliefs.len()];
    }

    /// Solve via value iteration over enumerated belief states.
    pub fn solve(&mut self, max_iters: usize, tol: f64) {
        let n = self.beliefs.len();
        let gamma = self.model.gamma;

        for _ in 0..max_iters {
            let old_vals = self.values.clone();
            let mut max_change = 0.0f64;

            for i in 0..n {
                let b = &self.beliefs[i].clone();
                let mut best_val = f64::NEG_INFINITY;
                let mut best_a = 0;

                for a in 0..self.model.n_actions {
                    // Immediate expected reward
                    let r: f64 = b
                        .prob
                        .iter()
                        .enumerate()
                        .map(|(s, &p)| p * self.model.reward[s][a])
                        .sum();

                    // Expected future value over (o, b')
                    let mut future_val = 0.0;
                    for o in 0..self.model.n_obs {
                        let b_next = b.update(a, o, &self.model);
                        // P(o|b,a) = Σ_s b(s) Σ_s' T(s,a,s') Z(a,s',o)
                        let prob_o: f64 = b
                            .prob
                            .iter()
                            .enumerate()
                            .map(|(s, &bs)| {
                                (0..self.model.n_states)
                                    .map(|sp| {
                                        self.model.transition[s][a][sp]
                                            * self.model.observation[a][sp][o]
                                    })
                                    .sum::<f64>()
                                    * bs
                            })
                            .sum();

                        if prob_o < 1e-15 {
                            continue;
                        }

                        // Find closest enumerated belief
                        let best_idx = (0..self.beliefs.len())
                            .min_by(|&j1, &j2| {
                                let d1 = b_next.distance(&self.beliefs[j1]);
                                let d2 = b_next.distance(&self.beliefs[j2]);
                                d1.partial_cmp(&d2).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .unwrap_or(0);

                        future_val += prob_o * old_vals[best_idx];
                    }

                    let total = r + gamma * future_val;
                    if total > best_val {
                        best_val = total;
                        best_a = a;
                    }
                }

                let change = (best_val - self.values[i]).abs();
                if change > max_change {
                    max_change = change;
                }
                self.values[i] = best_val;
                self.policy[i] = best_a;
            }

            if max_change < tol {
                break;
            }
        }
    }

    /// Best action for a belief (finds nearest enumerated belief).
    pub fn best_action(&self, b: &BeliefState) -> PomdpAction {
        if self.beliefs.is_empty() {
            return 0;
        }
        let best_idx = (0..self.beliefs.len())
            .min_by(|&j1, &j2| {
                let d1 = b.distance(&self.beliefs[j1]);
                let d2 = b.distance(&self.beliefs[j2]);
                d1.partial_cmp(&d2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        self.policy[best_idx]
    }
}

// ─── 12. OnlineBeliefTreeSearch ───────────────────────────────────────────────

/// Node in the AO* belief tree.
#[derive(Debug, Clone)]
pub struct BtNode {
    /// Belief at this node
    pub belief: BeliefState,
    /// Depth in tree
    pub depth: usize,
    /// Best value estimate (upper bound / heuristic)
    pub value: f64,
    /// Whether this node has been expanded
    pub expanded: bool,
    /// Children: (action, obs, child_idx) tuples
    pub children: Vec<(PomdpAction, PomdpObs, usize)>,
}

/// Online AO* belief tree search for POMDP planning.
///
/// Builds a belief tree using heuristic values and backtracks to find best action.
pub struct OnlineBeliefTreeSearch {
    model: PomdpModel,
    /// All tree nodes
    pub nodes: Vec<BtNode>,
    /// Max depth for expansion
    pub max_depth: usize,
    /// Heuristic alpha vectors (e.g., QMDP)
    heuristic_alphas: Vec<AlphaVector>,
}

impl OnlineBeliefTreeSearch {
    /// Create a new AO* belief tree searcher.
    pub fn new(model: PomdpModel, max_depth: usize) -> Self {
        let heuristic_alphas = QmdpApproximation::compute_qmdp(model.clone());
        Self {
            model,
            nodes: Vec::new(),
            max_depth,
            heuristic_alphas,
        }
    }

    /// Run AO* search from a given belief.
    pub fn search(&mut self, belief: BeliefState, budget: usize) -> PomdpAction {
        self.nodes.clear();
        let root_val = self.heuristic_value(&belief);
        self.nodes.push(BtNode {
            belief,
            depth: 0,
            value: root_val,
            expanded: false,
            children: Vec::new(),
        });

        for _ in 0..budget {
            // Find best unexpanded node via heuristic
            let node_idx = self.select_node();
            if node_idx >= self.nodes.len() {
                break;
            }
            let depth = self.nodes[node_idx].depth;
            if depth >= self.max_depth {
                continue;
            }
            self.expand_node(node_idx);
            self.backpropagate_value(node_idx);
        }

        self.best_action(0)
    }

    /// Compute the heuristic value at a belief (exposed for testing).
    pub fn heuristic_value(&self, b: &BeliefState) -> f64 {
        if self.heuristic_alphas.is_empty() {
            return 0.0;
        }
        AlphaVector::max_value(&self.heuristic_alphas, b).0
    }

    fn select_node(&self) -> usize {
        // Select unexpanded node with highest value
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| !n.expanded && n.depth < self.max_depth)
            .max_by(|a, b| {
                a.1.value
                    .partial_cmp(&b.1.value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(self.nodes.len())
    }

    /// Expand a node: create children for all (action, observation) pairs.
    pub fn expand_node(&mut self, node_idx: usize) {
        if node_idx >= self.nodes.len() {
            return;
        }
        self.nodes[node_idx].expanded = true;
        let depth = self.nodes[node_idx].depth;
        let belief = self.nodes[node_idx].belief.clone();

        let n_a = self.model.n_actions;
        let n_o = self.model.n_obs;
        let gamma = self.model.gamma;

        let mut new_children: Vec<(PomdpAction, PomdpObs, usize)> = Vec::new();

        for a in 0..n_a {
            let r: f64 = belief
                .prob
                .iter()
                .enumerate()
                .map(|(s, &p)| p * self.model.reward[s][a])
                .sum();

            for o in 0..n_o {
                let b_next = belief.update(a, o, &self.model);
                let prob_o: f64 = belief
                    .prob
                    .iter()
                    .enumerate()
                    .map(|(s, &bs)| {
                        (0..self.model.n_states)
                            .map(|sp| {
                                self.model.transition[s][a][sp] * self.model.observation[a][sp][o]
                            })
                            .sum::<f64>()
                            * bs
                    })
                    .sum();

                if prob_o < 1e-15 {
                    continue;
                }

                let h_val = self.heuristic_value(&b_next);
                let child_val = r + gamma * h_val;
                let child_idx = self.nodes.len();
                self.nodes.push(BtNode {
                    belief: b_next,
                    depth: depth + 1,
                    value: child_val,
                    expanded: false,
                    children: Vec::new(),
                });
                new_children.push((a, o, child_idx));
            }
        }

        self.nodes[node_idx].children = new_children;
    }

    /// Backpropagate values up from a node.
    pub fn backpropagate_value(&mut self, node_idx: usize) {
        if node_idx >= self.nodes.len() {
            return;
        }
        if self.nodes[node_idx].children.is_empty() {
            return;
        }

        let n_a = self.model.n_actions;
        let n_o = self.model.n_obs;
        let gamma = self.model.gamma;
        let belief = self.nodes[node_idx].belief.clone();

        // Compute best action value
        let mut best_val = f64::NEG_INFINITY;
        for a in 0..n_a {
            let r: f64 = belief
                .prob
                .iter()
                .enumerate()
                .map(|(s, &p)| p * self.model.reward[s][a])
                .sum();

            let mut future_val = 0.0;
            for o in 0..n_o {
                // Find child for (a, o)
                let child_val = self.nodes[node_idx]
                    .children
                    .iter()
                    .find(|&&(ca, co, _)| ca == a && co == o)
                    .map(|&(_, _, ci)| {
                        if ci < self.nodes.len() {
                            self.nodes[ci].value
                        } else {
                            0.0
                        }
                    })
                    .unwrap_or(0.0);

                let prob_o: f64 = belief
                    .prob
                    .iter()
                    .enumerate()
                    .map(|(s, &bs)| {
                        (0..self.model.n_states)
                            .map(|sp| {
                                self.model.transition[s][a][sp] * self.model.observation[a][sp][o]
                            })
                            .sum::<f64>()
                            * bs
                    })
                    .sum();

                future_val += prob_o * child_val;
            }

            let val = r + gamma * future_val;
            if val > best_val {
                best_val = val;
            }
        }

        self.nodes[node_idx].value = best_val;
    }

    /// Get the best action at a node.
    pub fn best_action(&self, node_idx: usize) -> PomdpAction {
        if node_idx >= self.nodes.len() || self.nodes[node_idx].children.is_empty() {
            return 0;
        }
        let belief = &self.nodes[node_idx].belief;
        let n_a = self.model.n_actions;
        let gamma = self.model.gamma;

        (0..n_a)
            .max_by(|&a1, &a2| {
                let action_val = |a: usize| -> f64 {
                    let r: f64 = belief
                        .prob
                        .iter()
                        .enumerate()
                        .map(|(s, &p)| p * self.model.reward[s][a])
                        .sum();
                    let future: f64 = self.nodes[node_idx]
                        .children
                        .iter()
                        .filter(|&&(ca, _, _)| ca == a)
                        .map(|&(_, _, ci)| {
                            if ci < self.nodes.len() {
                                self.nodes[ci].value
                            } else {
                                0.0
                            }
                        })
                        .sum();
                    r + gamma * future
                };
                action_val(a1)
                    .partial_cmp(&action_val(a2))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0)
    }
}
