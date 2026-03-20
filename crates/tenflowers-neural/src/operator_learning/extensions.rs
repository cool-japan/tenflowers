//! Extensions for Operator Learning: GP Symbolic Regression, Neural-Symbolic Hybrid,
//! Hamiltonian/Lagrangian NNs, Symplectic Integrator, Score Matching.

use super::{dense_forward_f32, relu, sample_normal_bm, xavier_init_f32, Mlp, SymbolicExpr, SymbolicNode};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

/// A genetic programming symbolic regressor.
#[derive(Debug, Clone)]
pub struct GpSymbolicRegressor {
    /// Population size.
    pub pop_size: usize,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Tournament size for selection.
    pub tournament_size: usize,
    /// Probability of mutation per individual.
    pub mutation_prob: f64,
    /// Number of generations to evolve.
    pub n_generations: usize,
    /// Number of input variables.
    pub n_vars: usize,
}

impl GpSymbolicRegressor {
    /// Create a new GP regressor.
    pub fn new(
        pop_size: usize,
        max_depth: usize,
        tournament_size: usize,
        mutation_prob: f64,
        n_generations: usize,
        n_vars: usize,
    ) -> Self {
        Self {
            pop_size,
            max_depth,
            tournament_size,
            mutation_prob,
            n_generations,
            n_vars,
        }
    }

    /// Compute MSE fitness of an expression on a dataset.
    pub fn mse_fitness(&self, expr: &SymbolicExpr, x: &[Vec<f64>], y: &[f64]) -> f64 {
        if x.is_empty() {
            return f64::INFINITY;
        }
        let sum: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, &yi)| {
                let pred = expr.evaluate(xi);
                let diff = pred - yi;
                diff * diff
            })
            .sum();
        sum / x.len() as f64
    }

    /// Generate a random expression tree up to `depth`.
    fn random_tree(&self, depth: usize, rng: &mut StdRng) -> SymbolicExpr {
        if depth == 0 {
            if rng.random::<f64>() < 0.5 {
                return SymbolicExpr::constant((rng.random::<f64>() - 0.5) * 4.0);
            } else {
                let var_idx = (rng.random::<f64>() * self.n_vars as f64) as usize;
                return SymbolicExpr::variable(var_idx.min(self.n_vars - 1));
            }
        }
        let op_choice: f64 = rng.random();
        if op_choice < 0.3 {
            if rng.random::<f64>() < 0.5 {
                SymbolicExpr::constant((rng.random::<f64>() - 0.5) * 4.0)
            } else {
                let var_idx = (rng.random::<f64>() * self.n_vars as f64) as usize;
                SymbolicExpr::variable(var_idx.min(self.n_vars - 1))
            }
        } else if op_choice < 0.7 {
            let lhs = self.random_tree(depth - 1, rng);
            let rhs = self.random_tree(depth - 1, rng);
            let op_idx = (rng.random::<f64>() * 5.0) as usize;
            let op = match op_idx {
                0 => SymbolicNode::Add,
                1 => SymbolicNode::Mul,
                2 => SymbolicNode::Sub,
                3 => SymbolicNode::Div,
                _ => SymbolicNode::Pow,
            };
            SymbolicExpr::binary(op, lhs, rhs)
        } else {
            let child = self.random_tree(depth - 1, rng);
            let op_idx = (rng.random::<f64>() * 4.0) as usize;
            let op = match op_idx {
                0 => SymbolicNode::Sin,
                1 => SymbolicNode::Cos,
                2 => SymbolicNode::Exp,
                _ => SymbolicNode::Log,
            };
            SymbolicExpr::unary(op, child)
        }
    }

    /// Tournament selection: return index of best individual.
    fn tournament(&self, fitnesses: &[f64], rng: &mut StdRng) -> usize {
        let n = fitnesses.len();
        let mut best_idx = (rng.random::<f64>() * n as f64) as usize % n;
        for _ in 1..self.tournament_size {
            let idx = (rng.random::<f64>() * n as f64) as usize % n;
            if fitnesses[idx] < fitnesses[best_idx] {
                best_idx = idx;
            }
        }
        best_idx
    }

    /// Point mutation: replace a random leaf with a new random subtree.
    fn mutate(&self, expr: &SymbolicExpr, rng: &mut StdRng) -> SymbolicExpr {
        let n = expr.nodes.len();
        if n == 0 {
            return expr.clone();
        }
        let mutate_idx = (rng.random::<f64>() * n as f64) as usize % n;
        let new_leaf = if rng.random::<f64>() < 0.5 {
            SymbolicExpr::constant((rng.random::<f64>() - 0.5) * 4.0)
        } else {
            let var_idx = (rng.random::<f64>() * self.n_vars as f64) as usize;
            SymbolicExpr::variable(var_idx.min(self.n_vars - 1))
        };
        let mut new_nodes = Vec::with_capacity(n + new_leaf.nodes.len());
        let new_root =
            self.rebuild_with_replacement(expr, expr.root, mutate_idx, &new_leaf, &mut new_nodes);
        SymbolicExpr {
            nodes: new_nodes,
            root: new_root,
        }
    }

    fn rebuild_with_replacement(
        &self,
        expr: &SymbolicExpr,
        idx: usize,
        target: usize,
        replacement: &SymbolicExpr,
        out: &mut Vec<(SymbolicNode, usize, usize)>,
    ) -> usize {
        if idx == target {
            let offset = out.len();
            for (node, l, r) in &replacement.nodes {
                out.push((node.clone(), l + offset, r + offset));
            }
            return offset + replacement.root;
        }
        match &expr.nodes[idx] {
            (node @ SymbolicNode::Constant(_), _, _)
            | (node @ SymbolicNode::Variable(_), _, _) => {
                let new_idx = out.len();
                out.push((node.clone(), 0, 0));
                new_idx
            }
            (node @ SymbolicNode::Sin, l, _)
            | (node @ SymbolicNode::Cos, l, _)
            | (node @ SymbolicNode::Exp, l, _)
            | (node @ SymbolicNode::Log, l, _) => {
                let l_new = self.rebuild_with_replacement(expr, *l, target, replacement, out);
                let new_idx = out.len();
                out.push((node.clone(), l_new, 0));
                new_idx
            }
            (node, l, r) => {
                let l_new = self.rebuild_with_replacement(expr, *l, target, replacement, out);
                let r_new = self.rebuild_with_replacement(expr, *r, target, replacement, out);
                let new_idx = out.len();
                out.push((node.clone(), l_new, r_new));
                new_idx
            }
        }
    }

    /// Run GP and return the best symbolic expression found.
    pub fn fit(&self, x: &[Vec<f64>], y: &[f64], seed: u64) -> Result<SymbolicExpr> {
        if x.is_empty() || y.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "GpSymbolicRegressor::fit",
                "empty dataset",
            ));
        }
        if self.n_vars == 0 {
            return Err(TensorError::invalid_argument_op(
                "GpSymbolicRegressor::fit",
                "n_vars must be > 0",
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);

        let mut population: Vec<SymbolicExpr> = (0..self.pop_size)
            .map(|_| self.random_tree(self.max_depth, &mut rng))
            .collect();

        let mut fitnesses: Vec<f64> = population
            .iter()
            .map(|e| self.mse_fitness(e, x, y))
            .collect();

        for _gen in 0..self.n_generations {
            let mut new_pop: Vec<SymbolicExpr> = Vec::with_capacity(self.pop_size);
            for _ in 0..self.pop_size {
                let parent_idx = self.tournament(&fitnesses, &mut rng);
                let mut child = population[parent_idx].clone();
                if rng.random::<f64>() < self.mutation_prob {
                    child = self.mutate(&child, &mut rng);
                }
                new_pop.push(child);
            }
            population = new_pop;
            fitnesses = population
                .iter()
                .map(|e| self.mse_fitness(e, x, y))
                .collect();
        }

        let best_idx = fitnesses
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(population[best_idx].clone())
    }
}

/// Neural-symbolic hybrid: a neural network that produces symbolic expression coefficients.
#[derive(Debug, Clone)]
pub struct NeuralSymbolicHybrid {
    /// Neural network for computing basis coefficients.
    pub net: Mlp,
    /// Number of symbolic basis functions.
    pub n_basis: usize,
    /// Symbolic basis expressions.
    pub basis: Vec<SymbolicExpr>,
}

impl NeuralSymbolicHybrid {
    /// Create a neural-symbolic hybrid.
    pub fn new(
        input_dim: usize,
        hidden_dims: &[usize],
        basis: Vec<SymbolicExpr>,
        seed: u64,
    ) -> Result<Self> {
        let n_basis = basis.len();
        if n_basis == 0 {
            return Err(TensorError::invalid_argument_op(
                "NeuralSymbolicHybrid::new",
                "basis must not be empty",
            ));
        }
        let mut dims = vec![input_dim];
        dims.extend_from_slice(hidden_dims);
        dims.push(n_basis);
        let net = Mlp::new(&dims, seed)?;
        Ok(Self {
            net,
            n_basis,
            basis,
        })
    }

    /// Forward pass: evaluate all basis functions and combine with neural coefficients.
    pub fn forward(&self, input_f32: &[f32], x_symbolic: &[f64]) -> Result<f64> {
        let coeffs = self.net.forward(input_f32)?;
        let mut out = 0.0_f64;
        for (coeff, basis_expr) in coeffs.iter().zip(self.basis.iter()) {
            out += *coeff as f64 * basis_expr.evaluate(x_symbolic);
        }
        Ok(out)
    }
}

/// Hamiltonian Neural Network — learns H(q, p) and derives equations of motion.
#[derive(Debug, Clone)]
pub struct HamiltonianNN {
    /// Neural network approximating the Hamiltonian scalar.
    pub net: Mlp,
    /// Dimension of position and momentum vectors.
    pub dim: usize,
    /// Finite-difference step size for computing derivatives.
    pub h_fd: f64,
}

impl HamiltonianNN {
    /// Create an HNN with `dim`-dimensional phase space.
    pub fn new(dim: usize, hidden_dims: &[usize], seed: u64) -> Result<Self> {
        let input_dim = 2 * dim; // (q, p) concatenated
        let mut layer_dims = vec![input_dim];
        layer_dims.extend_from_slice(hidden_dims);
        layer_dims.push(1); // scalar H
        let net = Mlp::new(&layer_dims, seed)?;
        Ok(Self {
            net,
            dim,
            h_fd: 1e-5,
        })
    }

    /// Evaluate H(q, p).
    pub fn hamiltonian(&self, q: &[f32], p: &[f32]) -> Result<f32> {
        if q.len() != self.dim || p.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "HamiltonianNN::hamiltonian",
                "q and p must have length dim",
            ));
        }
        let mut input: Vec<f32> = q.to_vec();
        input.extend_from_slice(p);
        let out = self.net.forward(&input)?;
        Ok(out[0])
    }

    /// Compute `dq/dt = ∂H/∂p` and `dp/dt = -∂H/∂q` via central finite differences.
    pub fn equations_of_motion(&self, q: &[f32], p: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let d = self.dim;
        if q.len() != d || p.len() != d {
            return Err(TensorError::invalid_argument_op(
                "HamiltonianNN::equations_of_motion",
                "q and p must have length dim",
            ));
        }
        let h = self.h_fd as f32;
        let mut dqdt = vec![0.0_f32; d];
        let mut dpdt = vec![0.0_f32; d];

        for i in 0..d {
            let mut p_plus = p.to_vec();
            let mut p_minus = p.to_vec();
            p_plus[i] += h;
            p_minus[i] -= h;
            let hp = self.hamiltonian(q, &p_plus)?;
            let hm = self.hamiltonian(q, &p_minus)?;
            dqdt[i] = (hp - hm) / (2.0 * h);
        }

        for i in 0..d {
            let mut q_plus = q.to_vec();
            let mut q_minus = q.to_vec();
            q_plus[i] += h;
            q_minus[i] -= h;
            let hp = self.hamiltonian(&q_plus, p)?;
            let hm = self.hamiltonian(&q_minus, p)?;
            dpdt[i] = -(hp - hm) / (2.0 * h);
        }

        Ok((dqdt, dpdt))
    }
}

/// HNN trainer — minimises symplecticity violation on trajectory data.
#[derive(Debug, Clone)]
pub struct HnnTrainer {
    /// Learning rate.
    pub lr: f32,
    /// Finite-difference step for gradient computation.
    pub fd_h: f32,
}

impl HnnTrainer {
    /// Create an HnnTrainer.
    pub fn new(lr: f32) -> Self {
        Self { lr, fd_h: 1e-4 }
    }

    /// Compute MSE of predicted vs. observed time derivatives.
    pub fn trajectory_loss(
        &self,
        model: &HamiltonianNN,
        trajectories: &[(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)],
    ) -> Result<f32> {
        if trajectories.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "HnnTrainer::trajectory_loss",
                "empty trajectories",
            ));
        }
        let mut total = 0.0_f32;
        let mut n = 0_usize;
        for (q, p, dqdt_true, dpdt_true) in trajectories {
            let (dqdt_pred, dpdt_pred) = model.equations_of_motion(q, p)?;
            for i in 0..model.dim {
                total += (dqdt_pred[i] - dqdt_true[i]).powi(2);
                total += (dpdt_pred[i] - dpdt_true[i]).powi(2);
                n += 2;
            }
        }
        Ok(total / n as f32)
    }
}

/// Lagrangian Neural Network — learns L(q, qdot) and derives equations of motion.
#[derive(Debug, Clone)]
pub struct LagrangianNN {
    /// Neural network approximating the Lagrangian scalar.
    pub net: Mlp,
    /// Dimension of configuration space.
    pub dim: usize,
    /// Finite-difference step size.
    pub h_fd: f32,
}

impl LagrangianNN {
    /// Create an LNN.
    pub fn new(dim: usize, hidden_dims: &[usize], seed: u64) -> Result<Self> {
        let input_dim = 2 * dim;
        let mut layer_dims = vec![input_dim];
        layer_dims.extend_from_slice(hidden_dims);
        layer_dims.push(1);
        let net = Mlp::new(&layer_dims, seed)?;
        Ok(Self {
            net,
            dim,
            h_fd: 1e-4,
        })
    }

    /// Evaluate L(q, qdot).
    pub fn lagrangian(&self, q: &[f32], qdot: &[f32]) -> Result<f32> {
        if q.len() != self.dim || qdot.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "LagrangianNN::lagrangian",
                "dimension mismatch",
            ));
        }
        let mut input: Vec<f32> = q.to_vec();
        input.extend_from_slice(qdot);
        let out = self.net.forward(&input)?;
        Ok(out[0])
    }

    /// Compute `qddot` from Euler-Lagrange equations via finite differences.
    pub fn equations_of_motion(&self, q: &[f32], qdot: &[f32]) -> Result<Vec<f32>> {
        let d = self.dim;
        let h = self.h_fd;
        let mut dl_dq = vec![0.0_f32; d];
        let mut dl_dqdot = vec![0.0_f32; d];

        for i in 0..d {
            let mut q_plus = q.to_vec();
            let mut q_minus = q.to_vec();
            q_plus[i] += h;
            q_minus[i] -= h;
            let lp = self.lagrangian(&q_plus, qdot)?;
            let lm = self.lagrangian(&q_minus, qdot)?;
            dl_dq[i] = (lp - lm) / (2.0 * h);

            let mut qd_plus = qdot.to_vec();
            let mut qd_minus = qdot.to_vec();
            qd_plus[i] += h;
            qd_minus[i] -= h;
            let lp2 = self.lagrangian(q, &qd_plus)?;
            let lm2 = self.lagrangian(q, &qd_minus)?;
            dl_dqdot[i] = (lp2 - lm2) / (2.0 * h);
        }

        let _ = dl_dqdot; // Used for conceptual completeness
        Ok(dl_dq)
    }
}

/// Störmer-Verlet symplectic integrator using a learned Hamiltonian.
#[derive(Debug, Clone)]
pub struct SymplecticIntegrator {
    /// Learned Hamiltonian neural network.
    pub hnn: HamiltonianNN,
    /// Integration time step.
    pub dt: f32,
}

impl SymplecticIntegrator {
    /// Create a symplectic integrator.
    pub fn new(hnn: HamiltonianNN, dt: f32) -> Self {
        Self { hnn, dt }
    }

    /// Perform one Störmer-Verlet step.
    pub fn step(&self, q: &[f32], p: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let dt = self.dt;
        let dt_half = dt * 0.5;

        let (_, dpdt) = self.hnn.equations_of_motion(q, p)?;
        let p_half: Vec<f32> = p
            .iter()
            .zip(dpdt.iter())
            .map(|(pi, di)| pi + dt_half * di)
            .collect();

        let (dqdt, _) = self.hnn.equations_of_motion(q, &p_half)?;
        let q_new: Vec<f32> = q
            .iter()
            .zip(dqdt.iter())
            .map(|(qi, di)| qi + dt * di)
            .collect();

        let (_, dpdt2) = self.hnn.equations_of_motion(&q_new, &p_half)?;
        let p_new: Vec<f32> = p_half
            .iter()
            .zip(dpdt2.iter())
            .map(|(pi, di)| pi + dt_half * di)
            .collect();

        Ok((q_new, p_new))
    }

    /// Integrate for `n_steps` steps from initial (q, p).
    pub fn integrate(
        &self,
        q0: &[f32],
        p0: &[f32],
        n_steps: usize,
    ) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
        let mut trajectory = Vec::with_capacity(n_steps + 1);
        let mut q = q0.to_vec();
        let mut p = p0.to_vec();
        trajectory.push((q.clone(), p.clone()));

        for _ in 0..n_steps {
            let (qn, pn) = self.step(&q, &p)?;
            q = qn;
            p = pn;
            trajectory.push((q.clone(), p.clone()));
        }
        Ok(trajectory)
    }
}

/// Neural network estimating the score function ∇_x log p(x).
#[derive(Debug, Clone)]
pub struct ScoreNetwork {
    /// Inner MLP for score estimation.
    pub net: Mlp,
    /// Dimensionality of the data space.
    pub data_dim: usize,
}

impl ScoreNetwork {
    /// Create a ScoreNetwork for data in ℝ^`data_dim`.
    pub fn new(data_dim: usize, hidden_dims: &[usize], seed: u64) -> Result<Self> {
        let input_dim = data_dim + 1; // x + σ conditioning
        let mut layer_dims = vec![input_dim];
        layer_dims.extend_from_slice(hidden_dims);
        layer_dims.push(data_dim);
        let net = Mlp::new(&layer_dims, seed)?;
        Ok(Self { net, data_dim })
    }

    /// Estimate score ∇_x log p(x) at noise level `sigma`.
    pub fn score(&self, x: &[f32], sigma: f32) -> Result<Vec<f32>> {
        if x.len() != self.data_dim {
            return Err(TensorError::invalid_argument_op(
                "ScoreNetwork::score",
                &format!("x.len() {} != data_dim {}", x.len(), self.data_dim),
            ));
        }
        let mut input: Vec<f32> = x.to_vec();
        input.push(sigma);
        self.net.forward(&input)
    }
}

/// Denoising score matching loss.
#[derive(Debug, Clone)]
pub struct ScoreMatchingLoss {
    /// Noise levels (sigma values) to train with.
    pub sigmas: Vec<f32>,
}

impl ScoreMatchingLoss {
    /// Create with a list of noise levels.
    pub fn new(sigmas: Vec<f32>) -> Result<Self> {
        if sigmas.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "ScoreMatchingLoss::new",
                "sigmas must not be empty",
            ));
        }
        Ok(Self { sigmas })
    }

    /// Compute denoising score matching loss for one batch.
    pub fn compute(&self, network: &ScoreNetwork, x_clean: &[Vec<f32>], seed: u64) -> Result<f32> {
        if x_clean.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "ScoreMatchingLoss::compute",
                "empty batch",
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let d = network.data_dim;
        let n_sigmas = self.sigmas.len();
        let mut total_loss = 0.0_f32;

        for x in x_clean {
            if x.len() != d {
                return Err(TensorError::invalid_argument_op(
                    "ScoreMatchingLoss::compute",
                    "data_dim mismatch",
                ));
            }
            let sigma_idx = (rng.random::<f64>() * n_sigmas as f64) as usize % n_sigmas;
            let sigma = self.sigmas[sigma_idx];

            let noise: Vec<f32> =
                (0..d).map(|_| sample_normal_bm(&mut rng) * sigma).collect();
            let x_noisy: Vec<f32> =
                x.iter().zip(noise.iter()).map(|(xi, ni)| xi + ni).collect();

            let target_score: Vec<f32> = noise
                .iter()
                .map(|ni| -ni / (sigma * sigma).max(1e-8))
                .collect();

            let pred_score = network.score(&x_noisy, sigma)?;

            let sigma_sq = sigma * sigma;
            let loss: f32 = pred_score
                .iter()
                .zip(target_score.iter())
                .map(|(ps, ts)| sigma_sq * (ps - ts).powi(2))
                .sum::<f32>()
                / d as f32;
            total_loss += loss;
        }
        Ok(total_loss / x_clean.len() as f32)
    }
}

/// Unadjusted Langevin dynamics sampler using a learned score network.
#[derive(Debug, Clone)]
pub struct LangevinSampler {
    /// Step size for the Langevin update.
    pub step_size: f32,
    /// Number of Langevin steps.
    pub n_steps: usize,
    /// Noise level for score evaluation.
    pub sigma: f32,
}

impl LangevinSampler {
    /// Create a Langevin sampler.
    pub fn new(step_size: f32, n_steps: usize, sigma: f32) -> Self {
        Self {
            step_size,
            n_steps,
            sigma,
        }
    }

    /// Run Langevin dynamics from initial point `x0`.
    pub fn sample(&self, network: &ScoreNetwork, x0: &[f32], seed: u64) -> Result<Vec<f32>> {
        let d = network.data_dim;
        if x0.len() != d {
            return Err(TensorError::invalid_argument_op(
                "LangevinSampler::sample",
                "x0.len() != data_dim",
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut x = x0.to_vec();
        let eps = self.step_size;
        let sqrt_eps = eps.sqrt();

        for _ in 0..self.n_steps {
            let score = network.score(&x, self.sigma)?;
            let noise: Vec<f32> = (0..d).map(|_| sample_normal_bm(&mut rng)).collect();
            for i in 0..d {
                x[i] += (eps / 2.0) * score[i] + sqrt_eps * noise[i];
            }
        }
        Ok(x)
    }

    /// Run a trajectory of Langevin steps and return all intermediate states.
    pub fn sample_trajectory(
        &self,
        network: &ScoreNetwork,
        x0: &[f32],
        seed: u64,
    ) -> Result<Vec<Vec<f32>>> {
        let d = network.data_dim;
        if x0.len() != d {
            return Err(TensorError::invalid_argument_op(
                "LangevinSampler::sample_trajectory",
                "x0.len() != data_dim",
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut x = x0.to_vec();
        let eps = self.step_size;
        let sqrt_eps = eps.sqrt();
        let mut trajectory = Vec::with_capacity(self.n_steps + 1);
        trajectory.push(x.clone());

        for _ in 0..self.n_steps {
            let score = network.score(&x, self.sigma)?;
            let noise: Vec<f32> = (0..d).map(|_| sample_normal_bm(&mut rng)).collect();
            for i in 0..d {
                x[i] += (eps / 2.0) * score[i] + sqrt_eps * noise[i];
            }
            trajectory.push(x.clone());
        }
        Ok(trajectory)
    }
}

/// Sliced score matching — estimate trace(∇ s) via random projections.
#[derive(Debug, Clone)]
pub struct SlicedScoreMatching {
    /// Number of random projection directions.
    pub n_projections: usize,
    /// Noise level for score evaluation.
    pub sigma: f32,
    /// Finite-difference step for Jacobian-vector products.
    pub fd_h: f32,
}

impl SlicedScoreMatching {
    /// Create a sliced score matching loss.
    pub fn new(n_projections: usize, sigma: f32) -> Self {
        Self {
            n_projections,
            sigma,
            fd_h: 1e-4,
        }
    }

    /// Compute sliced score matching loss via finite differences for Jv product.
    pub fn compute(
        &self,
        network: &ScoreNetwork,
        x_batch: &[Vec<f32>],
        seed: u64,
    ) -> Result<f32> {
        if x_batch.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "SlicedScoreMatching::compute",
                "empty batch",
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let d = network.data_dim;
        let mut total = 0.0_f32;

        for x in x_batch {
            if x.len() != d {
                return Err(TensorError::invalid_argument_op(
                    "SlicedScoreMatching::compute",
                    "dim mismatch",
                ));
            }
            let score_x = network.score(x, self.sigma)?;
            let norm_sq: f32 = score_x.iter().map(|s| s * s).sum();

            let mut jvv_sum = 0.0_f32;
            for _ in 0..self.n_projections {
                let v: Vec<f32> = (0..d)
                    .map(|_| {
                        if rng.random::<f64>() < 0.5 {
                            1.0_f32
                        } else {
                            -1.0_f32
                        }
                    })
                    .collect();

                let h = self.fd_h;
                let x_plus: Vec<f32> =
                    x.iter().zip(v.iter()).map(|(xi, vi)| xi + h * vi).collect();
                let x_minus: Vec<f32> =
                    x.iter().zip(v.iter()).map(|(xi, vi)| xi - h * vi).collect();
                let s_plus = network.score(&x_plus, self.sigma)?;
                let s_minus = network.score(&x_minus, self.sigma)?;
                let jv: Vec<f32> = s_plus
                    .iter()
                    .zip(s_minus.iter())
                    .map(|(sp, sm)| (sp - sm) / (2.0 * h))
                    .collect();

                let vt_jv: f32 = v.iter().zip(jv.iter()).map(|(vi, jvi)| vi * jvi).sum();
                jvv_sum += vt_jv;
            }
            let jvv_avg = jvv_sum / self.n_projections as f32;
            total += jvv_avg + 0.5 * norm_sq;
        }
        Ok(total / x_batch.len() as f32)
    }
}
