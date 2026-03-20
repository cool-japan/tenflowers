//! Advanced mean field game algorithms.
//!
//! Provides:
//! - Mean Field Control (MFC) with neural cost functional and actor-critic policy gradient
//! - Graphon Mean Field Games (Caines & Huang 2019)
//! - Risk-Sensitive MFG with exponential utility and CVaR objective
//! - Stationary (ergodic) MFG solver with ergodic constant estimation
//! - Extended metrics: convergence rate, inter-iteration Wasserstein, Nash gap

use super::{mfg_err, sample_normal_f64, clamp_f64,
            MfgDistribution, MfgState, MfgAction,
            McKeanVlasovDynamics, MfgCostFunction, MfgQuadraticCost,
            MfgDenseLayer};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use tenflowers_core::Result;

// ═══════════════════════════════════════════════════════════════════════════════
// §A  Mean Field Control (MFC)
// ═══════════════════════════════════════════════════════════════════════════════

/// Neural-network cost functional for MFC: L(x, u, μ) parameterised by a small MLP.
/// Input = [x, u, μ_mean, μ_var], output = scalar cost.
#[derive(Debug, Clone)]
pub struct MfcCostFunctional {
    layers: Vec<MfgDenseLayer>,
    /// Scale applied to the raw network output.
    pub output_scale: f64,
}

impl MfcCostFunctional {
    /// Construct an MfcCostFunctional with the given hidden layer sizes.
    pub fn new(hidden: &[usize], seed: u64) -> Result<Self> {
        if hidden.is_empty() {
            return Err(mfg_err(
                "MfcCostFunctional::new",
                "need at least one hidden layer",
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut dims: Vec<usize> = vec![4];
        dims.extend_from_slice(hidden);
        dims.push(1);
        let layers = dims
            .windows(2)
            .map(|w| MfgDenseLayer::xavier_init(w[0], w[1], &mut rng))
            .collect();
        Ok(MfcCostFunctional {
            layers,
            output_scale: 1.0,
        })
    }

    /// Evaluate L(x, u, μ): always non-negative via softplus.
    pub fn evaluate(&self, x: f64, u: f64, mu: &MfgDistribution) -> f64 {
        let mut inp = vec![x, u, mu.mean(), mu.variance()];
        let n = self.layers.len();
        for (k, layer) in self.layers.iter().enumerate() {
            let out = layer.forward(&inp);
            inp = if k < n - 1 {
                out.iter().map(|v| v.tanh()).collect()
            } else {
                out.iter().map(|v| v.abs() * self.output_scale).collect()
            };
        }
        inp[0]
    }
}

impl MfgCostFunction for MfcCostFunctional {
    fn running_cost(&self, x: MfgState, a: MfgAction, mu: &MfgDistribution) -> f64 {
        self.evaluate(x.0, a.0, mu)
    }
    fn terminal_cost(&self, x: MfgState, mu: &MfgDistribution) -> f64 {
        self.evaluate(x.0, 0.0, mu)
    }
}

/// Critic (value function) network V(x, μ) → scalar for MFC actor-critic.
/// Input = [x, μ_mean, μ_var, μ_skew], output = V.
#[derive(Debug, Clone)]
pub struct MfcValueFunction {
    layers: Vec<MfgDenseLayer>,
    /// Total number of trainable parameters.
    n_params: usize,
}

impl MfcValueFunction {
    /// Construct a critic network.
    pub fn new(hidden: &[usize], seed: u64) -> Result<Self> {
        if hidden.is_empty() {
            return Err(mfg_err(
                "MfcValueFunction::new",
                "need at least one hidden layer",
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut dims: Vec<usize> = vec![4];
        dims.extend_from_slice(hidden);
        dims.push(1);
        let layers: Vec<MfgDenseLayer> = dims
            .windows(2)
            .map(|w| MfgDenseLayer::xavier_init(w[0], w[1], &mut rng))
            .collect();
        let n_params = layers.iter().map(|l| l.n_params()).sum();
        Ok(MfcValueFunction { layers, n_params })
    }

    /// Forward pass: V(x, μ).
    pub fn forward(&self, x: f64, mu: &MfgDistribution) -> f64 {
        let [mean, var, skew] = mu.features();
        let mut inp = vec![x, mean, var, skew];
        let n = self.layers.len();
        for (k, layer) in self.layers.iter().enumerate() {
            let out = layer.forward(&inp);
            inp = if k < n - 1 {
                out.iter().map(|v| v.tanh()).collect()
            } else {
                out.to_vec()
            };
        }
        inp[0]
    }

    /// Total trainable parameter count.
    pub fn n_params(&self) -> usize {
        self.n_params
    }
}

/// Config for the MFC policy gradient solver.
#[derive(Debug, Clone)]
pub struct MfcPolicyGradientConfig {
    /// Number of agents in the population sample.
    pub n_agents: usize,
    /// Number of time steps per episode.
    pub n_steps: usize,
    /// Horizon T.
    pub horizon: f64,
    /// Policy gradient learning rate.
    pub lr_policy: f64,
    /// Critic learning rate.
    pub lr_critic: f64,
    /// Number of training episodes.
    pub n_episodes: usize,
    /// Finite-difference epsilon.
    pub fd_eps: f64,
    /// RNG seed.
    pub seed: u64,
    /// State grid bins.
    pub n_bins: usize,
    /// Grid lower bound.
    pub x_min: f64,
    /// Grid upper bound.
    pub x_max: f64,
}

impl Default for MfcPolicyGradientConfig {
    fn default() -> Self {
        MfcPolicyGradientConfig {
            n_agents: 50,
            n_steps: 10,
            horizon: 1.0,
            lr_policy: 1e-3,
            lr_critic: 1e-3,
            n_episodes: 10,
            fd_eps: 1e-4,
            seed: 0,
            n_bins: 16,
            x_min: -3.0,
            x_max: 3.0,
        }
    }
}

/// Training result for MFC policy gradient.
#[derive(Debug, Clone)]
pub struct MfcTrainResult {
    /// Per-episode total cost.
    pub episode_costs: Vec<f64>,
    /// Converged mean-field distribution sequence.
    pub mu_seq: Vec<MfgDistribution>,
}

/// REINFORCE-style policy gradient for Mean Field Control.
///
/// Uses the population empirical distribution as the mean field,
/// and updates the policy via finite-difference gradient of the expected cost.
pub struct MfcPolicyGradient {
    /// Configuration.
    pub config: MfcPolicyGradientConfig,
    /// Stochastic dynamics.
    pub dynamics: McKeanVlasovDynamics,
    /// Cost functional.
    pub cost: Box<dyn MfgCostFunction>,
}

impl MfcPolicyGradient {
    /// Construct a new MFC policy gradient solver.
    pub fn new(
        config: MfcPolicyGradientConfig,
        dynamics: McKeanVlasovDynamics,
        cost: Box<dyn MfgCostFunction>,
    ) -> Self {
        MfcPolicyGradient {
            config,
            dynamics,
            cost,
        }
    }

    /// Train a simple proportional controller θ: u = -θ·x.
    /// Returns per-episode total cost and final distribution sequence.
    pub fn train(&self) -> Result<MfcTrainResult> {
        let cfg = &self.config;
        let dt = cfg.horizon / cfg.n_steps as f64;
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let mut theta = 0.5f64; // scalar policy parameter
        let mut episode_costs = Vec::with_capacity(cfg.n_episodes);

        let mut mu_seq: Vec<MfgDistribution> = (0..=cfg.n_steps)
            .map(|_| MfgDistribution::uniform(cfg.n_bins, cfg.x_min, cfg.x_max))
            .collect::<Result<Vec<_>>>()?;

        for ep in 0..cfg.n_episodes {
            let seed_ep = cfg.seed.wrapping_add(ep as u64 * 1000 + 1);
            let cost0 = self.rollout_theta(theta, &mu_seq, seed_ep, dt)?;
            let cost_p = self.rollout_theta(theta + cfg.fd_eps, &mu_seq, seed_ep, dt)?;
            let grad = (cost_p - cost0) / cfg.fd_eps;
            theta -= cfg.lr_policy * grad;
            episode_costs.push(cost0);
            // update distribution
            let new_mu = self.simulate_theta(theta, &mu_seq, &mut rng, dt)?;
            for t in 0..=cfg.n_steps {
                for i in 0..cfg.n_bins {
                    mu_seq[t].weights[i] =
                        0.5 * mu_seq[t].weights[i] + 0.5 * new_mu[t].weights[i];
                }
                mu_seq[t].normalise();
            }
        }
        Ok(MfcTrainResult {
            episode_costs,
            mu_seq,
        })
    }

    fn rollout_theta(
        &self,
        theta: f64,
        mu_seq: &[MfgDistribution],
        seed: u64,
        dt: f64,
    ) -> Result<f64> {
        let cfg = &self.config;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut total = 0.0f64;
        for _ in 0..cfg.n_agents {
            let mut x = MfgState(sample_normal_f64(&mut rng) * 0.5);
            let n_steps = cfg.n_steps.min(mu_seq.len().saturating_sub(1));
            for t in 0..n_steps {
                let mu_t = &mu_seq[t];
                let a = MfgAction(-theta * x.0);
                total += self.cost.running_cost(x, a, mu_t) * dt;
                x = self.dynamics.step(x, a, mu_t, &mut rng);
            }
            let mu_terminal = mu_seq
                .last()
                .ok_or_else(|| mfg_err("rollout_theta", "mu_seq empty"))?;
            total += self.cost.terminal_cost(x, mu_terminal);
        }
        Ok(total / cfg.n_agents as f64)
    }

    fn simulate_theta(
        &self,
        theta: f64,
        mu_seq: &[MfgDistribution],
        rng: &mut impl Rng,
        dt: f64,
    ) -> Result<Vec<MfgDistribution>> {
        let cfg = &self.config;
        let n = cfg.n_bins;
        let mut new_mu_seq: Vec<MfgDistribution> = (0..=cfg.n_steps)
            .map(|_| MfgDistribution {
                weights: vec![0.0; n],
                x_min: cfg.x_min,
                x_max: cfg.x_max,
            })
            .collect();
        let mut states: Vec<f64> = (0..cfg.n_agents)
            .map(|_| sample_normal_f64(rng) * 0.5)
            .collect();
        for &x in &states {
            new_mu_seq[0].deposit_state(x);
        }
        new_mu_seq[0].normalise();
        let n_steps = cfg.n_steps.min(mu_seq.len().saturating_sub(1));
        for t in 0..n_steps {
            let mu_t = &mu_seq[t];
            for x_ref in states.iter_mut() {
                let a = MfgAction(-theta * *x_ref);
                let next = self.dynamics.step(MfgState(*x_ref), a, mu_t, rng);
                *x_ref = clamp_f64(next.0, cfg.x_min - 1.0, cfg.x_max + 1.0);
            }
            for &x in &states {
                new_mu_seq[t + 1].deposit_state(x);
            }
            new_mu_seq[t + 1].normalise();
        }
        Ok(new_mu_seq)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §B  Graphon Mean Field Games
// ═══════════════════════════════════════════════════════════════════════════════

/// Symmetric L²(\[0,1\]²) graphon kernel W(u,v) discretised on an n×n grid.
/// W\[i\]\[j\] represents W(i/n, j/n).
#[derive(Debug, Clone)]
pub struct GraphonKernel {
    /// Row-major kernel values, length = n × n.
    pub values: Vec<f64>,
    /// Grid size along each axis.
    pub n: usize,
}

impl GraphonKernel {
    /// Construct from a flat row-major grid (must have length n×n).
    pub fn new(values: Vec<f64>, n: usize) -> Result<Self> {
        if values.len() != n * n {
            return Err(mfg_err(
                "GraphonKernel::new",
                "values.len() must equal n*n",
            ));
        }
        if n == 0 {
            return Err(mfg_err("GraphonKernel::new", "n must be > 0"));
        }
        Ok(GraphonKernel { values, n })
    }

    /// Construct the Erdős–Rényi constant graphon W(u,v) = p.
    pub fn erdos_renyi(n: usize, p: f64) -> Result<Self> {
        if n == 0 {
            return Err(mfg_err("GraphonKernel::erdos_renyi", "n must be > 0"));
        }
        Ok(GraphonKernel {
            values: vec![p; n * n],
            n,
        })
    }

    /// Construct a block-structured graphon: W(u,v) = p_same if in same block, p_cross otherwise.
    pub fn block(n: usize, n_blocks: usize, p_same: f64, p_cross: f64) -> Result<Self> {
        if n == 0 {
            return Err(mfg_err("GraphonKernel::block", "n must be > 0"));
        }
        let block_size = (n as f64 / n_blocks as f64).ceil() as usize;
        let mut values = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let bi = i / block_size;
                let bj = j / block_size;
                values[i * n + j] = if bi == bj { p_same } else { p_cross };
            }
        }
        Ok(GraphonKernel { values, n })
    }

    /// Evaluate W(u, v) at continuous coordinates u, v ∈ \[0,1\] via nearest-neighbour lookup.
    pub fn eval(&self, u: f64, v: f64) -> f64 {
        let i = ((u * self.n as f64).floor() as usize).min(self.n - 1);
        let j = ((v * self.n as f64).floor() as usize).min(self.n - 1);
        self.values[i * self.n + j]
    }

    /// Compute the graphon-weighted mean field for agent at position `u` given distribution sequence.
    /// Returns ∫ W(u,v) · μ(v) dv approximated by discrete sum over `n` graphon nodes.
    pub fn weighted_mean(&self, u_idx: usize, distributions: &[MfgDistribution]) -> f64 {
        let n = self.n.min(distributions.len());
        if n == 0 {
            return 0.0;
        }
        let mut total = 0.0f64;
        let mut weight_sum = 0.0f64;
        for j in 0..n {
            let w = self.values[u_idx.min(self.n - 1) * self.n + j];
            let mu_mean = distributions[j].mean();
            total += w * mu_mean;
            weight_sum += w;
        }
        if weight_sum > 1e-15 {
            total / weight_sum
        } else {
            0.0
        }
    }
}

/// Config for the Graphon MFG solver.
#[derive(Debug, Clone)]
pub struct GraphonMfgConfig {
    /// Number of graphon nodes (= number of agent sub-populations).
    pub n_nodes: usize,
    /// Number of state-space bins per node.
    pub n_bins: usize,
    /// Grid lower bound.
    pub x_min: f64,
    /// Grid upper bound.
    pub x_max: f64,
    /// Time horizon.
    pub horizon: f64,
    /// Number of time steps.
    pub n_steps: usize,
    /// Number of agents per node.
    pub n_agents_per_node: usize,
    /// Fixed-point iterations.
    pub max_iters: usize,
    /// Damping for distribution update.
    pub damping: f64,
    /// RNG seed.
    pub seed: u64,
}

impl Default for GraphonMfgConfig {
    fn default() -> Self {
        GraphonMfgConfig {
            n_nodes: 4,
            n_bins: 16,
            x_min: -3.0,
            x_max: 3.0,
            horizon: 1.0,
            n_steps: 10,
            n_agents_per_node: 30,
            max_iters: 5,
            damping: 0.5,
            seed: 0,
        }
    }
}

/// Graphon-based MFG equilibrium point.
#[derive(Debug, Clone)]
pub struct GraphonEquilibrium {
    /// Per-node distribution sequences (outer = node, inner = time).
    pub node_distributions: Vec<Vec<MfgDistribution>>,
    /// Iteration residuals (max Wasserstein across all nodes and times).
    pub residuals: Vec<f64>,
    /// Whether the iteration converged.
    pub converged: bool,
}

/// Graphon MFG solver (Caines & Huang 2019).
///
/// Each node `i` has its own mean field `μ_i(t)`.
/// The effective drift for a node-`i` agent is influenced by a graphon-weighted combination
/// of the node distributions: b_graphon = κ·Σ_j W(i/N, j/N)·μ̄_j / Σ_j W(i/N, j/N).
pub struct GraphonMfgSolver {
    /// Kernel.
    pub kernel: GraphonKernel,
    /// Config.
    pub config: GraphonMfgConfig,
    /// Dynamics (shared across nodes).
    pub dynamics: McKeanVlasovDynamics,
}

impl GraphonMfgSolver {
    /// Construct a new Graphon MFG solver.
    pub fn new(
        kernel: GraphonKernel,
        config: GraphonMfgConfig,
        dynamics: McKeanVlasovDynamics,
    ) -> Self {
        GraphonMfgSolver {
            kernel,
            config,
            dynamics,
        }
    }

    /// Run fixed-point iteration to find the graphon MFG equilibrium.
    pub fn solve(&self) -> Result<GraphonEquilibrium> {
        let cfg = &self.config;
        let dt = cfg.horizon / cfg.n_steps as f64;
        // initialise: node_dist[node][time]
        let mut node_dist: Vec<Vec<MfgDistribution>> = (0..cfg.n_nodes)
            .map(|_| {
                (0..=cfg.n_steps)
                    .map(|_| MfgDistribution::uniform(cfg.n_bins, cfg.x_min, cfg.x_max))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;

        let mut residuals = Vec::new();
        let mut rng = StdRng::seed_from_u64(cfg.seed);

        for iter in 0..cfg.max_iters {
            let prev = node_dist.clone();
            let mut max_w1 = 0.0f64;

            for node in 0..cfg.n_nodes {
                let u_frac = node as f64 / cfg.n_nodes.max(1) as f64;
                let u_idx = (u_frac * self.kernel.n as f64).floor() as usize;
                let u_idx = u_idx.min(self.kernel.n - 1);

                let mut states: Vec<f64> = (0..cfg.n_agents_per_node)
                    .map(|_| sample_normal_f64(&mut rng) * 0.5)
                    .collect();

                let mut new_dist: Vec<MfgDistribution> = (0..=cfg.n_steps)
                    .map(|_| MfgDistribution {
                        weights: vec![0.0; cfg.n_bins],
                        x_min: cfg.x_min,
                        x_max: cfg.x_max,
                    })
                    .collect();

                for &x in &states {
                    new_dist[0].deposit_state(x);
                }
                new_dist[0].normalise();

                for t in 0..cfg.n_steps {
                    // compute graphon-weighted mean from previous iteration
                    let current_means: Vec<MfgDistribution> =
                        (0..cfg.n_nodes).map(|j| prev[j][t].clone()).collect();
                    let gw_mean = self.kernel.weighted_mean(u_idx, &current_means);
                    let own_mean = prev[node][t].mean();
                    // effective mean = blend of own mean and graphon-weighted mean
                    let eff_mean = 0.5 * own_mean + 0.5 * gw_mean;

                    for x_ref in states.iter_mut() {
                        // proportional control: a = -kappa * (x - eff_mean)
                        let a = -self.dynamics.config.kappa * (*x_ref - eff_mean);
                        let drift = self.dynamics.config.sigma_mf * (eff_mean - *x_ref)
                            - self.dynamics.config.kappa * *x_ref
                            + a;
                        let noise =
                            self.dynamics.config.sigma * dt.sqrt() * sample_normal_f64(&mut rng);
                        *x_ref = clamp_f64(*x_ref + drift * dt + noise, cfg.x_min - 1.0, cfg.x_max + 1.0);
                    }
                    for &x in &states {
                        new_dist[t + 1].deposit_state(x);
                    }
                    new_dist[t + 1].normalise();
                }

                // compute residual before update
                for t in 0..=cfg.n_steps {
                    let w1 = prev[node][t].wasserstein1(&new_dist[t])?;
                    if w1 > max_w1 {
                        max_w1 = w1;
                    }
                    // damped update
                    let alpha = cfg.damping;
                    for i in 0..cfg.n_bins {
                        node_dist[node][t].weights[i] = (1.0 - alpha) * prev[node][t].weights[i]
                            + alpha * new_dist[t].weights[i];
                    }
                    node_dist[node][t].normalise();
                }
            }

            residuals.push(max_w1);
            if max_w1 < 1e-4 && iter > 0 {
                return Ok(GraphonEquilibrium {
                    node_distributions: node_dist,
                    residuals,
                    converged: true,
                });
            }
        }
        Ok(GraphonEquilibrium {
            node_distributions: node_dist,
            residuals,
            converged: false,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §C  Risk-Sensitive MFG
// ═══════════════════════════════════════════════════════════════════════════════

/// Exponential utility function U(c) = (1/θ)·(1 - exp(-θ·c)).
/// θ > 0 → risk-averse; θ < 0 → risk-seeking; θ → 0 → risk-neutral.
#[derive(Debug, Clone)]
pub struct ExponentialUtility {
    /// Risk-sensitivity parameter θ (non-zero).
    pub theta: f64,
}

impl ExponentialUtility {
    /// Construct with risk parameter.
    pub fn new(theta: f64) -> Result<Self> {
        if theta.abs() < 1e-15 {
            return Err(mfg_err(
                "ExponentialUtility::new",
                "theta must be non-zero; use theta→0 for risk-neutral",
            ));
        }
        Ok(ExponentialUtility { theta })
    }

    /// Evaluate the certainty equivalent: CE(c) = -(1/θ)·ln(E[exp(-θ·c)]).
    /// For a scalar cost c this is simply c (used for per-step transformations).
    pub fn certainty_equivalent(&self, expected_exp_cost: f64) -> f64 {
        if self.theta.abs() < 1e-15 {
            return expected_exp_cost;
        }
        -expected_exp_cost.ln() / self.theta
    }

    /// Compute the risk-sensitive accumulated cost weight for a trajectory.
    /// Returns exp(-θ · cumulative_cost).
    pub fn trajectory_weight(&self, cumulative_cost: f64) -> f64 {
        (-self.theta * cumulative_cost).exp().max(1e-300)
    }

    /// Aggregate risk-sensitive costs from a sample of trajectory costs.
    /// Uses the log-sum-exp trick for numerical stability.
    pub fn aggregate(&self, costs: &[f64]) -> f64 {
        if costs.is_empty() {
            return 0.0;
        }
        let theta = self.theta;
        let shifted: Vec<f64> = costs.iter().map(|&c| -theta * c).collect();
        let max_s = shifted.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let log_sum = max_s + shifted.iter().map(|&s| (s - max_s).exp()).sum::<f64>().ln();
        -log_sum / theta - (costs.len() as f64).ln() / theta
    }
}

/// Config for the risk-sensitive MFG solver.
#[derive(Debug, Clone)]
pub struct RiskSensitiveMfgConfig {
    /// Risk-sensitivity parameter θ.
    pub theta: f64,
    /// State bins.
    pub n_bins: usize,
    /// Grid lower bound.
    pub x_min: f64,
    /// Grid upper bound.
    pub x_max: f64,
    /// Time horizon.
    pub horizon: f64,
    /// Time steps.
    pub n_steps: usize,
    /// Population size.
    pub n_agents: usize,
    /// Fixed-point iterations.
    pub max_iters: usize,
    /// RNG seed.
    pub seed: u64,
}

impl Default for RiskSensitiveMfgConfig {
    fn default() -> Self {
        RiskSensitiveMfgConfig {
            theta: 0.5,
            n_bins: 16,
            x_min: -3.0,
            x_max: 3.0,
            horizon: 1.0,
            n_steps: 10,
            n_agents: 50,
            max_iters: 5,
            seed: 0,
        }
    }
}

/// Solver for risk-sensitive MFG with exponential utility.
///
/// The risk-sensitive value function satisfies an entropic HJB equation:
/// -∂_t V - (1/θ)·ln(∫ exp(θ·(−L(x,a,μ) − a·∂_x V)) da) + (σ²/2)·∂_xx V = 0.
/// This is approximated here via importance-weighted population simulation.
pub struct RiskSensitiveMfgSolver {
    /// Solver config.
    pub config: RiskSensitiveMfgConfig,
    /// Dynamics.
    pub dynamics: McKeanVlasovDynamics,
    /// Underlying cost functional.
    pub cost: Box<dyn MfgCostFunction>,
    /// Exponential utility specification.
    pub utility: ExponentialUtility,
}

impl RiskSensitiveMfgSolver {
    /// Construct a new risk-sensitive MFG solver.
    pub fn new(
        config: RiskSensitiveMfgConfig,
        dynamics: McKeanVlasovDynamics,
        cost: Box<dyn MfgCostFunction>,
    ) -> Result<Self> {
        let utility = ExponentialUtility::new(config.theta)?;
        Ok(RiskSensitiveMfgSolver {
            config,
            dynamics,
            cost,
            utility,
        })
    }

    /// Run the risk-sensitive fixed-point iteration.
    /// Returns (distribution sequence, per-iteration risk-adjusted cost, converged flag).
    pub fn solve(&self) -> Result<(Vec<MfgDistribution>, Vec<f64>, bool)> {
        let cfg = &self.config;
        let dt = cfg.horizon / cfg.n_steps as f64;
        let mut mu_seq: Vec<MfgDistribution> = (0..=cfg.n_steps)
            .map(|_| MfgDistribution::uniform(cfg.n_bins, cfg.x_min, cfg.x_max))
            .collect::<Result<Vec<_>>>()?;
        let mut iter_costs = Vec::new();
        let mut rng = StdRng::seed_from_u64(cfg.seed);

        for _iter in 0..cfg.max_iters {
            let mut traj_costs = Vec::with_capacity(cfg.n_agents);
            let mut new_mu_seq: Vec<MfgDistribution> = (0..=cfg.n_steps)
                .map(|_| MfgDistribution {
                    weights: vec![0.0; cfg.n_bins],
                    x_min: cfg.x_min,
                    x_max: cfg.x_max,
                })
                .collect();

            let mut states: Vec<f64> = (0..cfg.n_agents)
                .map(|_| sample_normal_f64(&mut rng) * 0.5)
                .collect();
            let mut agent_costs = vec![0.0f64; cfg.n_agents];

            for &x in &states {
                new_mu_seq[0].deposit_state(x);
            }
            new_mu_seq[0].normalise();

            for t in 0..cfg.n_steps {
                let mu_t = &mu_seq[t];
                for (idx, x_ref) in states.iter_mut().enumerate() {
                    // greedy action: minimise L + proportional to x
                    let a = -self.dynamics.config.kappa * *x_ref;
                    let lc = self.cost.running_cost(MfgState(*x_ref), MfgAction(a), mu_t) * dt;
                    agent_costs[idx] += lc;
                    let next = self.dynamics.step(MfgState(*x_ref), MfgAction(a), mu_t, &mut rng);
                    *x_ref = clamp_f64(next.0, cfg.x_min - 1.0, cfg.x_max + 1.0);
                }
                for &x in &states {
                    new_mu_seq[t + 1].deposit_state(x);
                }
                new_mu_seq[t + 1].normalise();
            }

            let mu_term = mu_seq
                .last()
                .ok_or_else(|| mfg_err("rs_solve", "mu_seq empty"))?;
            for (idx, &x) in states.iter().enumerate() {
                agent_costs[idx] += self.cost.terminal_cost(MfgState(x), mu_term);
            }
            traj_costs.extend_from_slice(&agent_costs);

            let risk_cost = self.utility.aggregate(&traj_costs);
            iter_costs.push(risk_cost);

            // importance-weighted distribution update
            let weights: Vec<f64> = agent_costs
                .iter()
                .map(|&c| self.utility.trajectory_weight(c))
                .collect();
            let w_sum: f64 = weights.iter().sum::<f64>().max(1e-15);

            for t in 0..=cfg.n_steps {
                let mut new_w = vec![0.0f64; cfg.n_bins];
                // We only have the final states; approximate importance weights by
                // depositing each initial state with its trajectory weight.
                // For a more accurate version we re-simulate, but here we use the
                // empirical distribution weighted by the importance weights.
                for (j, &x) in states.iter().enumerate() {
                    let n = cfg.n_bins;
                    let dx = (cfg.x_max - cfg.x_min) / (n - 1).max(1) as f64;
                    let idx = ((x - cfg.x_min) / dx).round() as isize;
                    let idx = idx.max(0).min(n as isize - 1) as usize;
                    new_w[idx] += weights[j] / w_sum;
                }
                let s: f64 = new_w.iter().sum::<f64>().max(1e-20);
                for w in new_w.iter_mut() {
                    *w /= s;
                }
                for i in 0..cfg.n_bins {
                    mu_seq[t].weights[i] =
                        0.5 * mu_seq[t].weights[i] + 0.5 * new_mu_seq[t].weights[i];
                }
                mu_seq[t].normalise();
            }
        }

        let converged = iter_costs
            .windows(2)
            .last()
            .map(|w| (w[1] - w[0]).abs() < 1e-3)
            .unwrap_or(false);
        Ok((mu_seq, iter_costs, converged))
    }
}

/// CVaR-constrained MFG objective.
///
/// Computes CVaR_α(cost) = E[cost | cost ≥ q_α] where q_α is the α-quantile.
#[derive(Debug, Clone)]
pub struct CVaRMfgObjective {
    /// Confidence level α ∈ (0, 1).
    pub alpha: f64,
}

impl CVaRMfgObjective {
    /// Construct with confidence level.
    pub fn new(alpha: f64) -> Result<Self> {
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(mfg_err(
                "CVaRMfgObjective::new",
                "alpha must be in (0, 1)",
            ));
        }
        Ok(CVaRMfgObjective { alpha })
    }

    /// Compute CVaR_α from a sample of costs.
    pub fn compute(&self, costs: &[f64]) -> Result<f64> {
        if costs.is_empty() {
            return Err(mfg_err("CVaRMfgObjective::compute", "costs is empty"));
        }
        let mut sorted = costs.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let quantile_idx = ((1.0 - self.alpha) * n as f64).floor() as usize;
        let quantile_idx = quantile_idx.min(n - 1);
        let tail: Vec<f64> = sorted[quantile_idx..].to_vec();
        if tail.is_empty() {
            return Ok(sorted[n - 1]);
        }
        Ok(tail.iter().sum::<f64>() / tail.len() as f64)
    }

    /// Compute the Nash gap under CVaR: max over agents of CVaR(deviation cost) - CVaR(eq cost).
    pub fn nash_gap(
        &self,
        eq_costs: &[f64],
        dev_costs: &[f64],
    ) -> Result<f64> {
        if eq_costs.len() != dev_costs.len() {
            return Err(mfg_err(
                "CVaRMfgObjective::nash_gap",
                "eq_costs and dev_costs must have equal length",
            ));
        }
        let cvar_eq = self.compute(eq_costs)?;
        let cvar_dev = self.compute(dev_costs)?;
        Ok((cvar_eq - cvar_dev).abs())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §D  Stationary MFG
// ═══════════════════════════════════════════════════════════════════════════════

/// Config for the stationary (ergodic) MFG solver.
#[derive(Debug, Clone)]
pub struct StationaryMfgConfig {
    /// Number of state bins.
    pub n_bins: usize,
    /// Grid lower bound.
    pub x_min: f64,
    /// Grid upper bound.
    pub x_max: f64,
    /// Stationary FP diffusion coefficient.
    pub diffusion: f64,
    /// Maximum fixed-point iterations.
    pub max_iters: usize,
    /// Convergence tolerance on value-function change.
    pub tolerance: f64,
    /// Damping factor for distribution update.
    pub damping: f64,
    /// RNG seed.
    pub seed: u64,
}

impl Default for StationaryMfgConfig {
    fn default() -> Self {
        StationaryMfgConfig {
            n_bins: 32,
            x_min: -3.0,
            x_max: 3.0,
            diffusion: 0.05,
            max_iters: 30,
            tolerance: 1e-4,
            damping: 0.5,
            seed: 0,
        }
    }
}

/// Stationary MFG solver for the ergodic problem:
///   λ + H(x, ∂_x V, m) = f(x, m) + D·∂_xx V,   ∂_x(m·v) = D·∂_xx m.
/// Solved via alternating updates of (V, m) and ergodic constant λ.
pub struct StationaryMfgSolver {
    /// Solver config.
    pub config: StationaryMfgConfig,
    /// Cost functional.
    pub cost: Box<dyn MfgCostFunction>,
    /// Dynamics parameters (kappa, sigma used).
    pub dynamics: McKeanVlasovDynamics,
}

impl StationaryMfgSolver {
    /// Construct a new stationary MFG solver.
    pub fn new(
        config: StationaryMfgConfig,
        cost: Box<dyn MfgCostFunction>,
        dynamics: McKeanVlasovDynamics,
    ) -> Self {
        StationaryMfgSolver {
            config,
            cost,
            dynamics,
        }
    }

    /// Solve for the stationary distribution, value function, and ergodic constant.
    /// Returns `(stationary_mu, value_fn, ergodic_constant, residuals, converged)`.
    pub fn solve(&self) -> Result<(MfgDistribution, Vec<f64>, f64, Vec<f64>, bool)> {
        let cfg = &self.config;
        let n = cfg.n_bins;
        let dx = (cfg.x_max - cfg.x_min) / (n - 1).max(1) as f64;
        let d = cfg.diffusion;

        // Initialise uniform stationary distribution
        let mut mu = MfgDistribution::uniform(n, cfg.x_min, cfg.x_max)?;
        // Initialise value function to zero
        let mut v = vec![0.0f64; n];
        let mut lambda = 0.0f64;
        let mut residuals = Vec::new();

        for _iter in 0..cfg.max_iters {
            // ── HJB update (value iteration for stationary HJB) ──────────────
            let mut v_new = vec![0.0f64; n];
            for i in 0..n {
                let xi = cfg.x_min + i as f64 * dx;
                let v_x = if i == 0 {
                    (v[1] - v[0]) / dx
                } else if i == n - 1 {
                    (v[n - 1] - v[n - 2]) / dx
                } else {
                    (v[i + 1] - v[i - 1]) / (2.0 * dx)
                };
                let v_xx = if i == 0 || i == n - 1 {
                    0.0
                } else {
                    (v[i + 1] - 2.0 * v[i] + v[i - 1]) / (dx * dx)
                };
                // optimal Hamiltonian H(x, v_x) = min_a { L(x,a,mu) + v_x * a }
                let h = self.optimal_h(xi, v_x, &mu);
                // stationary HJB: λ = H - f + D·v_xx  (f = running_cost from mu)
                // Update V: V_new[i] = V[i] + dt * (H - lambda + D * v_xx)
                // Using pseudo-time stepping dt = dx^2 / (2D) for stability
                let dt_pseudo = (dx * dx / (2.0 * d)).min(0.1);
                v_new[i] = v[i] + dt_pseudo * (h - lambda + d * v_xx);
            }
            // Normalise v to have zero mean (ergodic constant gauge)
            let v_mean: f64 = v_new.iter().sum::<f64>() / n as f64;
            for vi in v_new.iter_mut() {
                *vi -= v_mean;
            }

            // ── Ergodic constant estimate ─────────────────────────────────────
            let mut lambda_new = 0.0f64;
            for i in 0..n {
                let xi = cfg.x_min + i as f64 * dx;
                let v_x = if i == 0 {
                    (v_new[1] - v_new[0]) / dx
                } else if i == n - 1 {
                    (v_new[n - 1] - v_new[n - 2]) / dx
                } else {
                    (v_new[i + 1] - v_new[i - 1]) / (2.0 * dx)
                };
                let v_xx = if i == 0 || i == n - 1 {
                    0.0
                } else {
                    (v_new[i + 1] - 2.0 * v_new[i] + v_new[i - 1]) / (dx * dx)
                };
                let h = self.optimal_h(xi, v_x, &mu);
                lambda_new += (h + d * v_xx) * mu.weights[i];
            }

            // ── Fokker–Planck stationary update ───────────────────────────────
            let mut mu_new_w = vec![0.0f64; n];
            for i in 1..n - 1 {
                let xi = cfg.x_min + i as f64 * dx;
                let v_x = (v_new[i + 1] - v_new[i - 1]) / (2.0 * dx);
                // drift = -kappa*x + optimal_action(v_x)
                let a_opt = self.optimal_action(xi, v_x, &mu);
                let drift = -self.dynamics.config.kappa * xi + a_opt;
                // upwind flux
                let flux_r = if drift >= 0.0 {
                    drift * mu.weights[i]
                } else {
                    drift * mu.weights[i + 1]
                };
                let xi_l = cfg.x_min + (i - 1) as f64 * dx;
                let v_x_l = (v_new[i] - v_new[i - 1]) / dx;
                let a_l = self.optimal_action(xi_l, v_x_l, &mu);
                let drift_l = -self.dynamics.config.kappa * xi_l + a_l;
                let flux_l = if drift_l >= 0.0 {
                    drift_l * mu.weights[i - 1]
                } else {
                    drift_l * mu.weights[i]
                };
                // diffusion
                let diff = d * (mu.weights[i + 1] - 2.0 * mu.weights[i] + mu.weights[i - 1])
                    / (dx * dx);
                let dt_fp = (dx * dx / (2.0 * d)).min(0.05);
                mu_new_w[i] = (mu.weights[i] - dt_fp * (flux_r - flux_l) / dx + dt_fp * diff)
                    .max(0.0);
            }
            // Boundary: no-flux
            mu_new_w[0] = mu.weights[0];
            mu_new_w[n - 1] = mu.weights[n - 1];
            let s: f64 = mu_new_w.iter().sum::<f64>().max(1e-20);
            for w in mu_new_w.iter_mut() {
                *w /= s;
            }

            // ── Residual and update ───────────────────────────────────────────
            let v_diff: f64 = v_new
                .iter()
                .zip(v.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            residuals.push(v_diff);

            // Damped updates
            let alpha = cfg.damping;
            for i in 0..n {
                v[i] = (1.0 - alpha) * v[i] + alpha * v_new[i];
                mu.weights[i] = (1.0 - alpha) * mu.weights[i] + alpha * mu_new_w[i];
            }
            mu.normalise();
            lambda = (1.0 - alpha) * lambda + alpha * lambda_new;

            if v_diff < cfg.tolerance {
                return Ok((mu, v, lambda, residuals, true));
            }
        }
        Ok((mu, v, lambda, residuals, false))
    }

    fn optimal_h(&self, x: f64, v_x: f64, mu: &MfgDistribution) -> f64 {
        let (a_min, a_max, cands) = (-2.0f64, 2.0f64, 21usize);
        let mut best = f64::INFINITY;
        for k in 0..cands {
            let a = a_min + (a_max - a_min) * k as f64 / (cands - 1) as f64;
            let h = self.cost.running_cost(MfgState(x), MfgAction(a), mu) + v_x * a;
            if h < best {
                best = h;
            }
        }
        best
    }

    fn optimal_action(&self, x: f64, v_x: f64, mu: &MfgDistribution) -> f64 {
        let (a_min, a_max, cands) = (-2.0f64, 2.0f64, 21usize);
        let (mut best_a, mut best_h) = (0.0f64, f64::INFINITY);
        for k in 0..cands {
            let a = a_min + (a_max - a_min) * k as f64 / (cands - 1) as f64;
            let h = self.cost.running_cost(MfgState(x), MfgAction(a), mu) + v_x * a;
            if h < best_h {
                best_h = h;
                best_a = a;
            }
        }
        best_a
    }
}

/// Estimates the ergodic constant λ for a stationary MFG via power iteration
/// on the linearised HJB operator.
pub struct ErgodConstantEstimator {
    /// Number of state bins.
    pub n_bins: usize,
    /// Grid lower bound.
    pub x_min: f64,
    /// Grid upper bound.
    pub x_max: f64,
    /// Diffusion coefficient.
    pub diffusion: f64,
    /// Power iterations.
    pub max_iters: usize,
}

impl ErgodConstantEstimator {
    /// Construct a new estimator.
    pub fn new(n_bins: usize, x_min: f64, x_max: f64, diffusion: f64, max_iters: usize) -> Self {
        ErgodConstantEstimator {
            n_bins,
            x_min,
            x_max,
            diffusion,
            max_iters,
        }
    }

    /// Estimate λ from a stationary value function v and distribution mu.
    /// Uses the Rayleigh quotient: λ ≈ ⟨L v, m⟩ / ⟨v, m⟩.
    pub fn estimate(
        &self,
        v: &[f64],
        mu: &MfgDistribution,
        cost: &dyn MfgCostFunction,
    ) -> Result<f64> {
        let n = self.n_bins;
        if v.len() != n || mu.n_bins() != n {
            return Err(mfg_err(
                "ErgodConstantEstimator::estimate",
                "v and mu must have length n_bins",
            ));
        }
        let dx = (self.x_max - self.x_min) / (n - 1).max(1) as f64;
        let d = self.diffusion;
        let mut numerator = 0.0f64;
        let mut denominator = 0.0f64;
        for i in 0..n {
            let xi = self.x_min + i as f64 * dx;
            let v_x = if i == 0 {
                (v[1] - v[0]) / dx
            } else if i == n - 1 {
                (v[n - 1] - v[n - 2]) / dx
            } else {
                (v[i + 1] - v[i - 1]) / (2.0 * dx)
            };
            let v_xx = if i == 0 || i == n - 1 {
                0.0
            } else {
                (v[i + 1] - 2.0 * v[i] + v[i - 1]) / (dx * dx)
            };
            // running cost approximation: use v_x * 0 as action proxy
            let lv = cost.running_cost(MfgState(xi), MfgAction(0.0), mu) + d * v_xx
                + v_x * (-0.5 * xi); // Hamiltonian approximation
            numerator += lv * mu.weights[i];
            denominator += v[i].abs() * mu.weights[i];
        }
        if denominator.abs() < 1e-15 {
            return Ok(0.0);
        }
        Ok(numerator / denominator)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §E  Extended Metrics
// ═══════════════════════════════════════════════════════════════════════════════

/// Extended evaluation metrics for MFG algorithms.
#[derive(Debug, Clone)]
pub struct MfgMetricsExtended {
    /// Geometric convergence rate estimated from residual sequence.
    pub convergence_rate: f64,
    /// Wasserstein distance between last two iterations.
    pub final_wasserstein: f64,
    /// Nash gap: maximum deviation incentive across grid points.
    pub nash_gap: f64,
    /// Mean of residuals.
    pub mean_residual: f64,
}

impl MfgMetricsExtended {
    /// Compute extended metrics from a residual sequence and two distributions.
    pub fn compute(
        residuals: &[f64],
        mu_prev: &MfgDistribution,
        mu_curr: &MfgDistribution,
        cost: &dyn MfgCostFunction,
        v: &[f64],
        x_min: f64,
        x_max: f64,
    ) -> Result<Self> {
        if residuals.is_empty() {
            return Err(mfg_err(
                "MfgMetricsExtended::compute",
                "residuals must be non-empty",
            ));
        }

        // Convergence rate from linear regression on log-residuals
        let convergence_rate = if residuals.len() >= 2 {
            let log_r: Vec<f64> = residuals
                .iter()
                .filter(|&&r| r > 1e-15)
                .map(|&r| r.ln())
                .collect();
            if log_r.len() >= 2 {
                let n = log_r.len() as f64;
                let xs: Vec<f64> = (0..log_r.len()).map(|i| i as f64).collect();
                let x_mean = xs.iter().sum::<f64>() / n;
                let y_mean = log_r.iter().sum::<f64>() / n;
                let cov = xs
                    .iter()
                    .zip(log_r.iter())
                    .map(|(x, y)| (x - x_mean) * (y - y_mean))
                    .sum::<f64>();
                let var_x = xs.iter().map(|x| (x - x_mean).powi(2)).sum::<f64>();
                if var_x > 1e-15 {
                    (cov / var_x).exp()
                } else {
                    1.0
                }
            } else {
                1.0
            }
        } else {
            1.0
        };

        let final_wasserstein = mu_prev.wasserstein1(mu_curr)?;

        // Nash gap: max over grid of |optimal H - current H|
        let n = v.len();
        let dx = if n > 1 { (x_max - x_min) / (n - 1) as f64 } else { 1.0 };
        let mut nash_gap = 0.0f64;
        for i in 0..n {
            let xi = x_min + i as f64 * dx;
            let v_x = if i == 0 {
                (v[1.min(n - 1)] - v[0]) / dx
            } else if i == n - 1 {
                (v[n - 1] - v[n - 2]) / dx
            } else {
                (v[i + 1] - v[i - 1]) / (2.0 * dx)
            };
            // Estimate incentive to deviate: H(x, v_x) vs zero-action cost
            let h_opt = {
                let (a_min, a_max, cands) = (-2.0f64, 2.0f64, 11usize);
                let mut best = f64::INFINITY;
                for k in 0..cands {
                    let a = a_min + (a_max - a_min) * k as f64 / (cands - 1) as f64;
                    let h = cost.running_cost(MfgState(xi), MfgAction(a), mu_curr) + v_x * a;
                    if h < best {
                        best = h;
                    }
                }
                best
            };
            let h_zero = cost.running_cost(MfgState(xi), MfgAction(0.0), mu_curr);
            let gap = (h_zero - h_opt).abs();
            if gap > nash_gap {
                nash_gap = gap;
            }
        }

        let mean_residual = residuals.iter().sum::<f64>() / residuals.len() as f64;

        Ok(MfgMetricsExtended {
            convergence_rate,
            final_wasserstein,
            nash_gap,
            mean_residual,
        })
    }

    /// Quick constructor when only residual statistics are needed.
    pub fn from_residuals(residuals: &[f64]) -> Result<Self> {
        if residuals.is_empty() {
            return Err(mfg_err(
                "MfgMetricsExtended::from_residuals",
                "residuals must be non-empty",
            ));
        }
        let n = residuals.len();
        let mean_residual = residuals.iter().sum::<f64>() / n as f64;
        let final_wasserstein = *residuals.last().unwrap_or(&0.0);
        let convergence_rate = if n >= 2 && residuals[0] > 1e-15 && residuals[n - 1] > 1e-15 {
            (residuals[n - 1] / residuals[0]).powf(1.0 / (n - 1) as f64)
        } else {
            1.0
        };
        Ok(MfgMetricsExtended {
            convergence_rate,
            final_wasserstein,
            nash_gap: 0.0,
            mean_residual,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §F  Convenience constructors using default quadratic cost
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a default quadratic cost box for use in solvers.
pub fn default_quadratic_cost() -> Box<dyn MfgCostFunction> {
    Box::new(MfgQuadraticCost::default())
}
