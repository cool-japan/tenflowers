//! # Evolutionary Computation — Advanced Algorithms (§6–§10)
//!
//! EcDifferentialEvolution (DE/rand/1/bin), EcParticleSwarm (PSO),
//! EcGeneticProgramming (symbolic regression), EcNsga3 (NSGA-III), EcMetrics.
//!
//! All randomness via the RNG helpers re-exported from the parent module.

use super::{ec_rand01, ec_rand_usize, ec_rand_usize_ne, ec_randn, EcError, EcGenome, EcSpecies};

// ── §6 EcDifferentialEvolution — DE/rand/1/bin ────────────────────────────────

/// Differential Evolution (Storn & Price 1997).
#[derive(Debug, Clone)]
pub struct EcDifferentialEvolution {
    pub population: Vec<Vec<f64>>,
    pub pop_size: usize,
    pub dim: usize,
    pub cr: f64,
    pub f: f64,
    pub bounds: Vec<(f64, f64)>,
    pub best_idx: usize,
    pub best_fitness: f64,
    pub generation: usize,
    pub fitness_history: Vec<f64>,
}

impl EcDifferentialEvolution {
    pub fn new(
        dim: usize,
        pop_size: usize,
        cr: f64,
        f: f64,
        bounds: Vec<(f64, f64)>,
    ) -> Result<Self, EcError> {
        if bounds.len() != dim {
            return Err(EcError::InvalidConfig(format!(
                "bounds len {} != dim {}",
                bounds.len(),
                dim
            )));
        }
        if pop_size < 4 {
            return Err(EcError::InvalidConfig(
                "pop_size must be >= 4 for DE/rand/1".to_string(),
            ));
        }
        Ok(EcDifferentialEvolution {
            population: Vec::new(),
            pop_size,
            dim,
            cr,
            f,
            bounds,
            best_idx: 0,
            best_fitness: f64::NEG_INFINITY,
            generation: 0,
            fitness_history: Vec::new(),
        })
    }

    /// Uniform initialization within bounds.
    pub fn initialize(&mut self, seed: &mut u64) {
        self.population = (0..self.pop_size)
            .map(|_| {
                (0..self.dim)
                    .map(|j| {
                        let (lo, hi) = self.bounds[j];
                        lo + ec_rand01(seed) * (hi - lo)
                    })
                    .collect()
            })
            .collect();
    }

    /// DE/rand/1 mutation: v = x_r1 + F * (x_r2 - x_r3).
    pub fn mutate_vector(&self, target_idx: usize, seed: &mut u64) -> Vec<f64> {
        let r1 = ec_rand_usize_ne(seed, self.pop_size, &[target_idx]);
        let r2 = ec_rand_usize_ne(seed, self.pop_size, &[target_idx, r1]);
        let r3 = ec_rand_usize_ne(seed, self.pop_size, &[target_idx, r1, r2]);
        (0..self.dim)
            .map(|j| {
                self.population[r1][j] + self.f * (self.population[r2][j] - self.population[r3][j])
            })
            .collect()
    }

    /// Binomial crossover.
    pub fn crossover(&self, target: &[f64], donor: &[f64], seed: &mut u64) -> Vec<f64> {
        let j_rand = ec_rand_usize(seed, self.dim);
        (0..self.dim)
            .map(|j| {
                if ec_rand01(seed) < self.cr || j == j_rand {
                    donor[j]
                } else {
                    target[j]
                }
            })
            .collect()
    }

    /// Clip solution to bounds.
    pub fn clip_to_bounds(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(j, &xi)| {
                let (lo, hi) = self.bounds[j];
                xi.clamp(lo, hi)
            })
            .collect()
    }

    /// One DE generation step.
    pub fn evolve_step(
        &mut self,
        fitness_fn: &dyn Fn(&[f64]) -> f64,
        seed: &mut u64,
        fitnesses: &mut Vec<f64>,
    ) {
        if self.population.is_empty() {
            return;
        }
        if fitnesses.len() != self.pop_size {
            *fitnesses = self.population.iter().map(|x| fitness_fn(x)).collect();
        }

        let mut new_population = self.population.clone();
        let mut new_fitnesses = fitnesses.clone();

        for i in 0..self.pop_size {
            let donor = self.mutate_vector(i, seed);
            let trial = self.crossover(&self.population[i], &donor, seed);
            let trial_clipped = self.clip_to_bounds(&trial);
            let trial_fitness = fitness_fn(&trial_clipped);
            if trial_fitness >= fitnesses[i] {
                new_population[i] = trial_clipped;
                new_fitnesses[i] = trial_fitness;
            }
        }

        self.population = new_population;
        *fitnesses = new_fitnesses;

        // Update best
        for (i, &fit) in fitnesses.iter().enumerate() {
            if fit > self.best_fitness {
                self.best_fitness = fit;
                self.best_idx = i;
            }
        }

        self.generation += 1;
        self.fitness_history.push(self.best_fitness);
    }

    pub fn get_best(&self) -> &[f64] {
        if self.population.is_empty() {
            &[]
        } else {
            &self.population[self.best_idx]
        }
    }
}

// ── §7 EcParticleSwarm ────────────────────────────────────────────────────────

/// A single particle in PSO.
#[derive(Debug, Clone)]
pub struct EcParticle {
    pub position: Vec<f64>,
    pub velocity: Vec<f64>,
    pub best_position: Vec<f64>,
    pub best_fitness: f64,
}

/// Particle Swarm Optimization (Kennedy & Eberhart 1995).
#[derive(Debug, Clone)]
pub struct EcParticleSwarm {
    pub particles: Vec<EcParticle>,
    pub global_best: Vec<f64>,
    pub global_best_fitness: f64,
    pub omega: f64,
    pub phi_p: f64,
    pub phi_g: f64,
    pub bounds: Vec<(f64, f64)>,
    pub generation: usize,
}

impl EcParticleSwarm {
    pub fn new(n_particles: usize, dim: usize, bounds: Vec<(f64, f64)>) -> Self {
        let placeholder = EcParticle {
            position: vec![0.0; dim],
            velocity: vec![0.0; dim],
            best_position: vec![0.0; dim],
            best_fitness: f64::NEG_INFINITY,
        };
        EcParticleSwarm {
            particles: vec![placeholder; n_particles],
            global_best: vec![0.0; dim],
            global_best_fitness: f64::NEG_INFINITY,
            omega: 0.7,
            phi_p: 1.5,
            phi_g: 1.5,
            bounds,
            generation: 0,
        }
    }

    pub fn initialize(&mut self, seed: &mut u64) {
        let dim = self.global_best.len();
        for particle in &mut self.particles {
            particle.position = (0..dim)
                .map(|j| {
                    let (lo, hi) = self.bounds[j];
                    lo + ec_rand01(seed) * (hi - lo)
                })
                .collect();
            particle.velocity = (0..dim)
                .map(|j| {
                    let (lo, hi) = self.bounds[j];
                    (ec_rand01(seed) - 0.5) * (hi - lo) * 0.1
                })
                .collect();
            particle.best_position = particle.position.clone();
            particle.best_fitness = f64::NEG_INFINITY;
        }
    }

    pub fn step(&mut self, fitness_fn: &dyn Fn(&[f64]) -> f64, seed: &mut u64) {
        let dim = self.global_best.len();

        for particle in &mut self.particles {
            // Update velocity and position
            let r_p: Vec<f64> = (0..dim).map(|_| ec_rand01(seed)).collect();
            let r_g: Vec<f64> = (0..dim).map(|_| ec_rand01(seed)).collect();

            for j in 0..dim {
                particle.velocity[j] = self.omega * particle.velocity[j]
                    + self.phi_p * r_p[j] * (particle.best_position[j] - particle.position[j])
                    + self.phi_g * r_g[j] * (self.global_best[j] - particle.position[j]);
                particle.position[j] = (particle.position[j] + particle.velocity[j])
                    .clamp(self.bounds[j].0, self.bounds[j].1);
            }

            let fitness = fitness_fn(&particle.position);
            if fitness > particle.best_fitness {
                particle.best_fitness = fitness;
                particle.best_position = particle.position.clone();
            }
            if fitness > self.global_best_fitness {
                self.global_best_fitness = fitness;
                self.global_best = particle.position.clone();
            }
        }
        self.generation += 1;
    }

    pub fn get_best(&self) -> (&[f64], f64) {
        (&self.global_best, self.global_best_fitness)
    }

    /// Mean pairwise distance between particles.
    pub fn swarm_diversity(&self) -> f64 {
        let n = self.particles.len();
        if n < 2 {
            return 0.0;
        }
        let mut total = 0.0f64;
        let mut count = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let dist: f64 = self.particles[i]
                    .position
                    .iter()
                    .zip(&self.particles[j].position)
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                total += dist;
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            total / count as f64
        }
    }
}

// ── §8 EcGeneticProgramming ───────────────────────────────────────────────────

/// Node in a GP expression tree.
#[derive(Debug, Clone)]
pub enum EcGpNode {
    Const(f64),
    Var(usize),
    Add,
    Sub,
    Mul,
    Div,
    Sin,
    Exp,
}

/// A GP expression tree stored as a flat Vec (prefix order).
#[derive(Debug, Clone)]
pub struct EcGpTree {
    pub nodes: Vec<EcGpNode>,
    pub n_vars: usize,
}

impl EcGpNode {
    fn arity(&self) -> usize {
        match self {
            EcGpNode::Const(_) | EcGpNode::Var(_) => 0,
            EcGpNode::Add | EcGpNode::Sub | EcGpNode::Mul | EcGpNode::Div => 2,
            EcGpNode::Sin | EcGpNode::Exp => 1,
        }
    }
}

/// Build a random GP tree (recursive, prefix-order) with given max depth.
fn gp_build_tree(
    nodes: &mut Vec<EcGpNode>,
    depth: usize,
    max_depth: usize,
    n_vars: usize,
    seed: &mut u64,
) {
    if depth >= max_depth || (depth > 0 && ec_rand01(seed) < 0.3) {
        // Terminal
        if n_vars > 0 && ec_rand01(seed) < 0.5 {
            let var = ec_rand_usize(seed, n_vars);
            nodes.push(EcGpNode::Var(var));
        } else {
            let c = ec_randn(seed);
            nodes.push(EcGpNode::Const(c));
        }
        return;
    }
    // Function node
    let r = ec_rand01(seed);
    let node = if r < 0.25 {
        EcGpNode::Add
    } else if r < 0.50 {
        EcGpNode::Mul
    } else if r < 0.625 {
        EcGpNode::Sub
    } else if r < 0.75 {
        EcGpNode::Div
    } else if r < 0.875 {
        EcGpNode::Sin
    } else {
        EcGpNode::Exp
    };
    let arity = node.arity();
    nodes.push(node);
    for _ in 0..arity {
        gp_build_tree(nodes, depth + 1, max_depth, n_vars, seed);
    }
}

/// Evaluate a prefix-encoded tree using a stack.
/// Returns (result, nodes_consumed).
fn gp_eval_tree(nodes: &[EcGpNode], x: &[f64], pos: usize) -> (f64, usize) {
    if pos >= nodes.len() {
        return (0.0, 1);
    }
    match &nodes[pos] {
        EcGpNode::Const(c) => (*c, 1),
        EcGpNode::Var(i) => {
            let val = if *i < x.len() { x[*i] } else { 0.0 };
            (val, 1)
        }
        EcGpNode::Add => {
            let (l, l_sz) = gp_eval_tree(nodes, x, pos + 1);
            let (r, r_sz) = gp_eval_tree(nodes, x, pos + 1 + l_sz);
            (l + r, 1 + l_sz + r_sz)
        }
        EcGpNode::Sub => {
            let (l, l_sz) = gp_eval_tree(nodes, x, pos + 1);
            let (r, r_sz) = gp_eval_tree(nodes, x, pos + 1 + l_sz);
            (l - r, 1 + l_sz + r_sz)
        }
        EcGpNode::Mul => {
            let (l, l_sz) = gp_eval_tree(nodes, x, pos + 1);
            let (r, r_sz) = gp_eval_tree(nodes, x, pos + 1 + l_sz);
            (l * r, 1 + l_sz + r_sz)
        }
        EcGpNode::Div => {
            let (l, l_sz) = gp_eval_tree(nodes, x, pos + 1);
            let (r, r_sz) = gp_eval_tree(nodes, x, pos + 1 + l_sz);
            let result = if r.abs() < 1e-9 { 1.0 } else { l / r };
            (result, 1 + l_sz + r_sz)
        }
        EcGpNode::Sin => {
            let (val, sz) = gp_eval_tree(nodes, x, pos + 1);
            (val.sin(), 1 + sz)
        }
        EcGpNode::Exp => {
            let (val, sz) = gp_eval_tree(nodes, x, pos + 1);
            (val.clamp(-50.0, 50.0).exp(), 1 + sz)
        }
    }
}

/// Compute the size (number of nodes) of a subtree rooted at `pos`.
fn gp_subtree_size(nodes: &[EcGpNode], pos: usize) -> usize {
    if pos >= nodes.len() {
        return 0;
    }
    let arity = nodes[pos].arity();
    let mut sz = 1;
    let mut cur = pos + 1;
    for _ in 0..arity {
        let child_sz = gp_subtree_size(nodes, cur);
        sz += child_sz;
        cur += child_sz;
    }
    sz
}

impl EcGpTree {
    pub fn random(max_depth: usize, n_vars: usize, seed: &mut u64) -> Self {
        let mut nodes = Vec::new();
        gp_build_tree(&mut nodes, 0, max_depth, n_vars, seed);
        if nodes.is_empty() {
            nodes.push(EcGpNode::Const(0.0));
        }
        EcGpTree { nodes, n_vars }
    }

    pub fn evaluate(&self, x: &[f64]) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let (val, _) = gp_eval_tree(&self.nodes, x, 0);
        if val.is_finite() {
            val
        } else {
            0.0
        }
    }

    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    pub fn depth(&self) -> usize {
        fn depth_at(nodes: &[EcGpNode], pos: usize) -> usize {
            if pos >= nodes.len() {
                return 0;
            }
            let arity = nodes[pos].arity();
            if arity == 0 {
                return 1;
            }
            let mut cur = pos + 1;
            let mut max_child_depth = 0;
            for _ in 0..arity {
                let child_depth = depth_at(nodes, cur);
                max_child_depth = max_child_depth.max(child_depth);
                cur += gp_subtree_size(nodes, cur);
            }
            1 + max_child_depth
        }
        depth_at(&self.nodes, 0)
    }

    /// Subtree crossover: pick random node in each tree, swap subtrees.
    pub fn crossover(&self, other: &Self, seed: &mut u64) -> Self {
        if self.nodes.is_empty() {
            return other.clone();
        }
        if other.nodes.is_empty() {
            return self.clone();
        }

        let cross_point = ec_rand_usize(seed, self.nodes.len());
        let other_point = ec_rand_usize(seed, other.nodes.len());

        let self_subtree_sz = gp_subtree_size(&self.nodes, cross_point);
        let other_subtree_sz = gp_subtree_size(&other.nodes, other_point);

        let mut new_nodes = Vec::new();
        // Prefix up to cross_point
        new_nodes.extend_from_slice(&self.nodes[..cross_point]);
        // Subtree from other
        new_nodes.extend_from_slice(&other.nodes[other_point..other_point + other_subtree_sz]);
        // Rest of self after cross_point subtree
        new_nodes.extend_from_slice(&self.nodes[cross_point + self_subtree_sz..]);

        if new_nodes.is_empty() {
            new_nodes.push(EcGpNode::Const(0.0));
        }

        EcGpTree {
            nodes: new_nodes,
            n_vars: self.n_vars,
        }
    }

    /// Random node replacement or constant perturbation.
    pub fn mutate(&self, seed: &mut u64) -> Self {
        if self.nodes.is_empty() {
            return self.clone();
        }
        let mut new_nodes = self.nodes.clone();
        let idx = ec_rand_usize(seed, new_nodes.len());

        if ec_rand01(seed) < 0.5 {
            // Perturb constant if it's a constant, else replace with constant
            match &new_nodes[idx] {
                EcGpNode::Const(c) => {
                    new_nodes[idx] = EcGpNode::Const(c + ec_randn(seed) * 0.1);
                }
                _ => {
                    new_nodes[idx] = EcGpNode::Const(ec_randn(seed));
                }
            }
        } else {
            // Replace with a random terminal
            if self.n_vars > 0 && ec_rand01(seed) < 0.5 {
                new_nodes[idx] = EcGpNode::Var(ec_rand_usize(seed, self.n_vars));
            } else {
                new_nodes[idx] = EcGpNode::Const(ec_randn(seed));
            }
        }

        EcGpTree {
            nodes: new_nodes,
            n_vars: self.n_vars,
        }
    }

    pub fn complexity_penalty(&self) -> f64 {
        self.size() as f64 * 0.001
    }
}

/// Genetic Programming for symbolic regression.
#[derive(Debug, Clone)]
pub struct EcGeneticProgramming {
    pub population: Vec<EcGpTree>,
    pub fitnesses: Vec<f64>,
    pub pop_size: usize,
    pub n_vars: usize,
    pub max_depth: usize,
    pub tournament_size: usize,
    pub mutation_rate: f64,
    pub generation: usize,
    pub best_fitness: f64,
}

impl EcGeneticProgramming {
    pub fn new(pop_size: usize, n_vars: usize, max_depth: usize) -> Self {
        EcGeneticProgramming {
            population: Vec::new(),
            fitnesses: Vec::new(),
            pop_size,
            n_vars,
            max_depth,
            tournament_size: 5,
            mutation_rate: 0.1,
            generation: 0,
            best_fitness: f64::NEG_INFINITY,
        }
    }

    pub fn evaluate_fitness(&mut self, x_data: &[Vec<f64>], y_data: &[f64]) {
        if self.population.is_empty() {
            return;
        }
        self.fitnesses = self
            .population
            .iter()
            .map(|tree| {
                if x_data.is_empty() {
                    return 0.0;
                }
                let mse: f64 = x_data
                    .iter()
                    .zip(y_data.iter())
                    .map(|(x, &y)| {
                        let pred = tree.evaluate(x);
                        (pred - y).powi(2)
                    })
                    .sum::<f64>()
                    / x_data.len() as f64;
                -mse - tree.complexity_penalty()
            })
            .collect();

        if let Some(&best) = self
            .fitnesses
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            if best > self.best_fitness {
                self.best_fitness = best;
            }
        }
    }

    /// Tournament selection: return index of winner.
    pub fn tournament_select(&self, seed: &mut u64) -> usize {
        let n = self.population.len();
        if n == 0 {
            return 0;
        }
        let t = self.tournament_size.min(n);
        let mut best_idx = ec_rand_usize(seed, n);
        let mut best_fit = if best_idx < self.fitnesses.len() {
            self.fitnesses[best_idx]
        } else {
            f64::NEG_INFINITY
        };
        for _ in 1..t {
            let idx = ec_rand_usize(seed, n);
            let fit = if idx < self.fitnesses.len() {
                self.fitnesses[idx]
            } else {
                f64::NEG_INFINITY
            };
            if fit > best_fit {
                best_fit = fit;
                best_idx = idx;
            }
        }
        best_idx
    }

    pub fn evolve_step(&mut self, x_data: &[Vec<f64>], y_data: &[f64], seed: &mut u64) {
        // Initialize if empty
        if self.population.is_empty() {
            self.population = (0..self.pop_size)
                .map(|_| EcGpTree::random(self.max_depth, self.n_vars, seed))
                .collect();
            self.evaluate_fitness(x_data, y_data);
        }

        let mut new_pop = Vec::with_capacity(self.pop_size);

        // Elitism: keep best
        if let Some(best_idx) = self
            .fitnesses
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
        {
            new_pop.push(self.population[best_idx].clone());
        }

        while new_pop.len() < self.pop_size {
            if ec_rand01(seed) < self.mutation_rate {
                let idx = self.tournament_select(seed);
                new_pop.push(self.population[idx].mutate(seed));
            } else {
                let p1 = self.tournament_select(seed);
                let p2 = self.tournament_select(seed);
                let child = self.population[p1].crossover(&self.population[p2], seed);
                new_pop.push(child);
            }
        }

        self.population = new_pop;
        self.evaluate_fitness(x_data, y_data);
        self.generation += 1;
    }

    pub fn best_tree(&self) -> &EcGpTree {
        if self.population.is_empty() || self.fitnesses.is_empty() {
            return &self.population[0];
        }
        let best_idx = self
            .fitnesses
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        &self.population[best_idx.min(self.population.len() - 1)]
    }
}

// ── §9 EcMultiObjectiveNsga3 — NSGA-III ──────────────────────────────────────

/// A solution in multi-objective optimization.
#[derive(Debug, Clone)]
pub struct EcSolution {
    pub params: Vec<f64>,
    pub objectives: Vec<f64>,
    pub rank: usize,
    pub distance: f64,
}

/// NSGA-III (Deb & Jain 2014): reference point-based many-objective optimization.
#[derive(Debug, Clone)]
pub struct EcNsga3 {
    pub population: Vec<EcSolution>,
    pub reference_points: Vec<Vec<f64>>,
    pub n_objectives: usize,
    pub pop_size: usize,
    pub dim: usize,
    pub cr: f64,
    pub eta_c: f64,
    pub eta_m: f64,
    pub bounds: Vec<(f64, f64)>,
    pub generation: usize,
}

impl EcNsga3 {
    pub fn new(pop_size: usize, dim: usize, n_objectives: usize, bounds: Vec<(f64, f64)>) -> Self {
        let n_divisions = if n_objectives <= 3 { 12 } else { 6 };
        let reference_points = Self::generate_reference_points(n_objectives, n_divisions);
        EcNsga3 {
            population: Vec::new(),
            reference_points,
            n_objectives,
            pop_size,
            dim,
            cr: 0.9,
            eta_c: 20.0,
            eta_m: 20.0,
            bounds,
            generation: 0,
        }
    }

    /// Generate structured Das-Dennis reference points on the unit simplex.
    pub fn generate_reference_points(n_objectives: usize, n_divisions: usize) -> Vec<Vec<f64>> {
        let mut result = Vec::new();
        let mut current = vec![0usize; n_objectives];
        Self::gen_ref_recursive(
            &mut result,
            &mut current,
            0,
            n_objectives,
            n_divisions,
            n_divisions,
        );
        result
    }

    fn gen_ref_recursive(
        result: &mut Vec<Vec<f64>>,
        current: &mut Vec<usize>,
        obj_idx: usize,
        n_objectives: usize,
        n_divisions: usize,
        remaining: usize,
    ) {
        if obj_idx == n_objectives - 1 {
            current[obj_idx] = remaining;
            let point: Vec<f64> = current
                .iter()
                .map(|&v| v as f64 / n_divisions as f64)
                .collect();
            result.push(point);
        } else {
            for i in 0..=remaining {
                current[obj_idx] = i;
                Self::gen_ref_recursive(
                    result,
                    current,
                    obj_idx + 1,
                    n_objectives,
                    n_divisions,
                    remaining - i,
                );
            }
        }
    }

    /// Non-dominated sort returning fronts.
    pub fn non_dominated_sort(solutions: &[EcSolution]) -> Vec<Vec<usize>> {
        let n = solutions.len();
        if n == 0 {
            return Vec::new();
        }

        let mut dominates: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut dominated_by: Vec<usize> = vec![0; n];
        let mut fronts: Vec<Vec<usize>> = Vec::new();
        let mut first_front = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dom_i_over_j = solutions[i]
                    .objectives
                    .iter()
                    .zip(&solutions[j].objectives)
                    .all(|(a, b)| a <= b)
                    && solutions[i]
                        .objectives
                        .iter()
                        .zip(&solutions[j].objectives)
                        .any(|(a, b)| a < b);
                if dom_i_over_j {
                    dominates[i].push(j);
                } else {
                    let dom_j_over_i = solutions[j]
                        .objectives
                        .iter()
                        .zip(&solutions[i].objectives)
                        .all(|(a, b)| a <= b)
                        && solutions[j]
                            .objectives
                            .iter()
                            .zip(&solutions[i].objectives)
                            .any(|(a, b)| a < b);
                    if dom_j_over_i {
                        dominated_by[i] += 1;
                    }
                }
            }
            if dominated_by[i] == 0 {
                first_front.push(i);
            }
        }

        fronts.push(first_front);
        let mut front_idx = 0;
        while !fronts[front_idx].is_empty() {
            let mut next_front = Vec::new();
            for &i in &fronts[front_idx].clone() {
                for &j in &dominates[i] {
                    dominated_by[j] -= 1;
                    if dominated_by[j] == 0 {
                        next_front.push(j);
                    }
                }
            }
            if next_front.is_empty() {
                break;
            }
            fronts.push(next_front);
            front_idx += 1;
        }

        fronts
    }

    /// Associate each solution to its nearest reference point (for niching).
    pub fn associate_to_refs(&self, population: &mut [EcSolution]) {
        // Normalize objectives
        if population.is_empty() {
            return;
        }
        let n_obj = self.n_objectives;
        let mut ideal = vec![f64::INFINITY; n_obj];
        for sol in population.iter() {
            for (j, &obj) in sol.objectives.iter().enumerate() {
                if j < n_obj && obj < ideal[j] {
                    ideal[j] = obj;
                }
            }
        }

        for sol in population.iter_mut() {
            let norm_obj: Vec<f64> = sol
                .objectives
                .iter()
                .enumerate()
                .map(|(j, &obj)| {
                    if j < ideal.len() {
                        (obj - ideal[j]).max(0.0)
                    } else {
                        obj
                    }
                })
                .collect();

            // Find nearest reference point
            let mut min_dist = f64::INFINITY;
            let mut nearest = 0;
            for (ri, ref_pt) in self.reference_points.iter().enumerate() {
                let dist: f64 = norm_obj
                    .iter()
                    .zip(ref_pt.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if dist < min_dist {
                    min_dist = dist;
                    nearest = ri;
                }
            }
            sol.distance = nearest as f64; // reuse field to store ref point idx
        }
    }

    /// Simulated Binary Crossover.
    pub fn sbx_crossover(&self, p1: &[f64], p2: &[f64], seed: &mut u64) -> (Vec<f64>, Vec<f64>) {
        let mut c1 = p1.to_vec();
        let mut c2 = p2.to_vec();

        if ec_rand01(seed) > self.cr {
            return (c1, c2);
        }

        for j in 0..self.dim.min(p1.len()).min(p2.len()) {
            if ec_rand01(seed) < 0.5 {
                let u = ec_rand01(seed).max(1e-10);
                let beta = if u <= 0.5 {
                    (2.0 * u).powf(1.0 / (self.eta_c + 1.0))
                } else {
                    (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (self.eta_c + 1.0))
                };
                c1[j] = 0.5 * ((1.0 + beta) * p1[j] + (1.0 - beta) * p2[j]);
                c2[j] = 0.5 * ((1.0 - beta) * p1[j] + (1.0 + beta) * p2[j]);
                let (lo, hi) = self.bounds[j];
                c1[j] = c1[j].clamp(lo, hi);
                c2[j] = c2[j].clamp(lo, hi);
            }
        }
        (c1, c2)
    }

    /// Polynomial mutation.
    pub fn polynomial_mutation(&self, x: &[f64], seed: &mut u64) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(j, &xi)| {
                if j >= self.bounds.len() {
                    return xi;
                }
                let (lo, hi) = self.bounds[j];
                let u = ec_rand01(seed);
                let delta = if u < 0.5 {
                    (2.0 * u).powf(1.0 / (self.eta_m + 1.0)) - 1.0
                } else {
                    1.0 - (2.0 * (1.0 - u)).powf(1.0 / (self.eta_m + 1.0))
                };
                let range = hi - lo;
                (xi + delta * range).clamp(lo, hi)
            })
            .collect()
    }

    /// One NSGA-III generation step.
    pub fn evolve_step(
        &mut self,
        objective_fn: &dyn Fn(&[f64]) -> Vec<f64>,
        seed: &mut u64,
    ) -> Result<(), EcError> {
        // Initialize population if empty
        if self.population.is_empty() {
            for _ in 0..self.pop_size {
                let params: Vec<f64> = (0..self.dim)
                    .map(|j| {
                        let (lo, hi) = self.bounds[j];
                        lo + ec_rand01(seed) * (hi - lo)
                    })
                    .collect();
                let objectives = objective_fn(&params);
                self.population.push(EcSolution {
                    params,
                    objectives,
                    rank: 0,
                    distance: 0.0,
                });
            }
        }

        // Generate offspring via SBX + mutation
        let mut offspring: Vec<EcSolution> = Vec::with_capacity(self.pop_size);
        let n = self.population.len();
        for _ in 0..(self.pop_size / 2) {
            let i = ec_rand_usize(seed, n);
            let j = ec_rand_usize_ne(seed, n, &[i]);
            let (c1_params, c2_params) =
                self.sbx_crossover(&self.population[i].params, &self.population[j].params, seed);
            let c1_mut = self.polynomial_mutation(&c1_params, seed);
            let c2_mut = self.polynomial_mutation(&c2_params, seed);
            offspring.push(EcSolution {
                objectives: objective_fn(&c1_mut),
                params: c1_mut,
                rank: 0,
                distance: 0.0,
            });
            offspring.push(EcSolution {
                objectives: objective_fn(&c2_mut),
                params: c2_mut,
                rank: 0,
                distance: 0.0,
            });
        }

        // Combine parent + offspring
        let mut combined = self.population.clone();
        combined.extend(offspring);

        // Non-dominated sort
        let fronts = Self::non_dominated_sort(&combined);

        // Select pop_size solutions
        let mut new_pop: Vec<EcSolution> = Vec::with_capacity(self.pop_size);
        let mut front_rank = 0;
        for front in &fronts {
            if new_pop.len() + front.len() <= self.pop_size {
                for &idx in front {
                    let mut sol = combined[idx].clone();
                    sol.rank = front_rank;
                    new_pop.push(sol);
                }
                front_rank += 1;
            } else {
                // Fill remaining slots using reference point niching
                let remaining = self.pop_size - new_pop.len();
                for &idx in front.iter().take(remaining) {
                    let mut sol = combined[idx].clone();
                    sol.rank = front_rank;
                    new_pop.push(sol);
                }
                break;
            }
        }

        self.associate_to_refs(&mut new_pop);
        self.population = new_pop;
        self.generation += 1;
        Ok(())
    }

    pub fn pareto_front(&self) -> Vec<&EcSolution> {
        self.population.iter().filter(|s| s.rank == 0).collect()
    }

    /// 2D hypervolume indicator (sweep line for 2 objectives, minimization).
    pub fn hypervolume_indicator(&self, reference_point: &[f64]) -> f64 {
        if self.n_objectives != 2 {
            // Approximate using dominated hyperbox for >2 objectives
            return self
                .pareto_front()
                .iter()
                .map(|s| {
                    s.objectives
                        .iter()
                        .zip(reference_point.iter())
                        .map(|(&obj, &ref_val)| (ref_val - obj).max(0.0))
                        .product::<f64>()
                })
                .sum::<f64>()
                .max(0.0);
        }

        let front = self.pareto_front();
        if front.is_empty() {
            return 0.0;
        }

        // Sort by first objective
        let mut pts: Vec<(f64, f64)> = front
            .iter()
            .filter(|s| s.objectives.len() >= 2)
            .map(|s| (s.objectives[0], s.objectives[1]))
            .collect();
        pts.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let ref_x = reference_point.first().copied().unwrap_or(1.0);
        let ref_y = reference_point.get(1).copied().unwrap_or(1.0);

        let mut hv = 0.0f64;
        let mut prev_y = ref_y;
        for &(x, y) in &pts {
            if x >= ref_x || y >= ref_y {
                continue;
            }
            let dy = prev_y - y;
            let dx = ref_x - x;
            if dy > 0.0 {
                hv += dx * dy;
                prev_y = y;
            }
        }
        hv.max(0.0)
    }
}

// ── §10 EcMetrics ─────────────────────────────────────────────────────────────

/// Metrics utilities for evolutionary computation.
pub struct EcMetrics;

impl EcMetrics {
    /// Mean pairwise Euclidean distance.
    pub fn population_diversity(population: &[Vec<f64>]) -> f64 {
        let n = population.len();
        if n < 2 {
            return 0.0;
        }
        let mut total = 0.0f64;
        let mut count = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let d: f64 = population[i]
                    .iter()
                    .zip(&population[j])
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                total += d;
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            total / count as f64
        }
    }

    /// (best_value, generation_where_achieved).
    pub fn best_fitness_history(history: &[f64]) -> (f64, usize) {
        if history.is_empty() {
            return (0.0, 0);
        }
        let mut best = history[0];
        let mut best_gen = 0;
        for (i, &v) in history.iter().enumerate() {
            if v > best {
                best = v;
                best_gen = i;
            }
        }
        (best, best_gen)
    }

    /// Linear regression slope of fitness over generations.
    pub fn convergence_rate(history: &[f64]) -> f64 {
        let n = history.len();
        if n < 2 {
            return 0.0;
        }
        let n_f = n as f64;
        let x_mean = (n_f - 1.0) / 2.0;
        let y_mean: f64 = history.iter().sum::<f64>() / n_f;
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (i, &y) in history.iter().enumerate() {
            let dx = i as f64 - x_mean;
            num += dx * (y - y_mean);
            den += dx * dx;
        }
        if den.abs() < 1e-12 {
            0.0
        } else {
            num / den
        }
    }

    /// Approximate hyperbox coverage fraction.
    pub fn coverage_metric(population: &[Vec<f64>], bounds: &[(f64, f64)]) -> f64 {
        if population.is_empty() || bounds.is_empty() {
            return 0.0;
        }
        let dim = bounds.len();
        // Compute range covered by population
        let min_vals: Vec<f64> = bounds.iter().map(|&(lo, _)| lo).collect();
        let max_vals: Vec<f64> = bounds.iter().map(|&(_, hi)| hi).collect();
        let mut pop_min: Vec<f64> = vec![f64::INFINITY; dim];
        let mut pop_max: Vec<f64> = vec![f64::NEG_INFINITY; dim];
        for x in population {
            for j in 0..dim.min(x.len()) {
                if x[j] < pop_min[j] {
                    pop_min[j] = x[j];
                }
                if x[j] > pop_max[j] {
                    pop_max[j] = x[j];
                }
            }
        }
        let mut coverage_vol = 1.0f64;
        let mut total_vol = 1.0f64;
        for j in 0..dim {
            let total_range = max_vals[j] - min_vals[j];
            let pop_range = (pop_max[j] - pop_min[j]).max(0.0);
            if total_range > 1e-12 {
                coverage_vol *= pop_range;
                total_vol *= total_range;
            }
        }
        if total_vol < 1e-12 {
            0.0
        } else {
            (coverage_vol / total_vol).clamp(0.0, 1.0)
        }
    }

    pub fn species_count(species: &[EcSpecies]) -> usize {
        species.len()
    }

    /// (mean_nodes, mean_connections) across population.
    pub fn neat_complexity(population: &[EcGenome]) -> (f64, f64) {
        if population.is_empty() {
            return (0.0, 0.0);
        }
        let n = population.len() as f64;
        let mean_nodes = population.iter().map(|g| g.nodes.len() as f64).sum::<f64>() / n;
        let mean_conns = population
            .iter()
            .map(|g| g.connections.len() as f64)
            .sum::<f64>()
            / n;
        (mean_nodes, mean_conns)
    }
}
