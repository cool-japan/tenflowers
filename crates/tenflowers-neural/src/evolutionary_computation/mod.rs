//! # Evolutionary Computation & Neuroevolution
//!
//! Advanced evolutionary algorithms for neural architecture search, quality-diversity
//! optimization, and symbolic regression. Implements NEAT, OpenAI ES, MAP-Elites,
//! Novelty Search, Differential Evolution, PSO, Genetic Programming, and NSGA-III.
//!
//! All randomness via internal LCG + Box-Muller. No ndarray. No unwrap().

use std::fmt;

#[cfg(test)]
mod tests;

pub mod ec_advanced;
pub use ec_advanced::*;

// ── Error Type ────────────────────────────────────────────────────────────────

/// Errors for evolutionary computation operations.
#[derive(Debug, Clone)]
pub enum EcError {
    InvalidConfig(String),
    NumericalError(String),
    EmptyPopulation,
}

impl fmt::Display for EcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EcError::InvalidConfig(s) => write!(f, "InvalidConfig: {}", s),
            EcError::NumericalError(s) => write!(f, "NumericalError: {}", s),
            EcError::EmptyPopulation => write!(f, "EmptyPopulation"),
        }
    }
}

impl std::error::Error for EcError {}

// ── RNG Helpers ───────────────────────────────────────────────────────────────

/// LCG-based pseudo-random number generator: uniform [0, 1).
/// Uses Xorshift64 + bit-manipulation to convert to f64.
#[inline]
pub fn ec_rand01(seed: &mut u64) -> f64 {
    // Xorshift64 for better randomness than pure LCG
    let mut x = *seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *seed = x;
    // Convert to f64 in [0, 1): take upper 53 bits, use double precision
    // Mantissa of f64 is 52 bits, exponent for [1, 2) is 0x3FF00000_00000000
    let mantissa = (x >> 12) & 0x000F_FFFF_FFFF_FFFFu64; // 52 bits
    let bits = 0x3FF0_0000_0000_0000u64 | mantissa;
    f64::from_bits(bits) - 1.0
}

/// Box-Muller transform for standard normal sample.
#[inline]
pub fn ec_randn(seed: &mut u64) -> f64 {
    let u1 = ec_rand01(seed).max(1e-300);
    let u2 = ec_rand01(seed);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    r * theta.cos()
}

/// Return a random integer in [0, n).
#[inline]
fn ec_rand_usize(seed: &mut u64, n: usize) -> usize {
    let v = (ec_rand01(seed) * n as f64) as usize;
    v.min(n - 1)
}

/// Return a random integer in [0, n) distinct from `exclude`.
fn ec_rand_usize_ne(seed: &mut u64, n: usize, exclude: &[usize]) -> usize {
    for _ in 0..1000 {
        let v = ec_rand_usize(seed, n);
        if !exclude.contains(&v) {
            return v;
        }
    }
    // fallback: first index not in exclude
    (0..n).find(|x| !exclude.contains(x)).unwrap_or(0)
}

// ── §1 EcGenome — Genome for NEAT ─────────────────────────────────────────────

/// Type of a node in a NEAT genome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EcNodeType {
    Input,
    Hidden,
    Output,
}

/// Activation function applied at a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EcActivation {
    Relu,
    Sigmoid,
    Tanh,
    Linear,
}

impl EcActivation {
    fn apply(&self, x: f64) -> f64 {
        match self {
            EcActivation::Relu => x.max(0.0),
            EcActivation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            EcActivation::Tanh => x.tanh(),
            EcActivation::Linear => x,
        }
    }
}

/// A node gene in a NEAT genome.
#[derive(Debug, Clone)]
pub struct EcNodeGene {
    pub id: usize,
    pub node_type: EcNodeType,
    pub activation: EcActivation,
}

/// A connection gene in a NEAT genome.
#[derive(Debug, Clone)]
pub struct EcConnectionGene {
    pub in_node: usize,
    pub out_node: usize,
    pub weight: f64,
    pub enabled: bool,
    pub innovation: usize,
}

/// A NEAT genome: collection of node and connection genes.
#[derive(Debug, Clone)]
pub struct EcGenome {
    pub nodes: Vec<EcNodeGene>,
    pub connections: Vec<EcConnectionGene>,
    pub fitness: f64,
}

impl EcGenome {
    /// Create a minimal fully-connected genome with n_inputs→n_outputs.
    pub fn new_minimal(n_inputs: usize, n_outputs: usize) -> Self {
        let mut nodes = Vec::with_capacity(n_inputs + n_outputs);
        for i in 0..n_inputs {
            nodes.push(EcNodeGene {
                id: i,
                node_type: EcNodeType::Input,
                activation: EcActivation::Linear,
            });
        }
        for i in 0..n_outputs {
            nodes.push(EcNodeGene {
                id: n_inputs + i,
                node_type: EcNodeType::Output,
                activation: EcActivation::Sigmoid,
            });
        }
        let mut connections = Vec::with_capacity(n_inputs * n_outputs);
        let mut innov = 0;
        for i in 0..n_inputs {
            for j in 0..n_outputs {
                connections.push(EcConnectionGene {
                    in_node: i,
                    out_node: n_inputs + j,
                    weight: 0.0,
                    enabled: true,
                    innovation: innov,
                });
                innov += 1;
            }
        }
        EcGenome {
            nodes,
            connections,
            fitness: 0.0,
        }
    }

    /// Feedforward activation using iterative relaxation (max 10 passes).
    pub fn activate(&self, inputs: &[f64]) -> Result<Vec<f64>, EcError> {
        let n_inputs = self
            .nodes
            .iter()
            .filter(|n| n.node_type == EcNodeType::Input)
            .count();
        if inputs.len() != n_inputs {
            return Err(EcError::InvalidConfig(format!(
                "Expected {} inputs, got {}",
                n_inputs,
                inputs.len()
            )));
        }

        // Build node value map
        let max_id = self.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;
        let mut values = vec![0.0f64; max_id];

        // Set input values
        for (idx, node) in self
            .nodes
            .iter()
            .filter(|n| n.node_type == EcNodeType::Input)
            .enumerate()
        {
            if idx < inputs.len() {
                values[node.id] = inputs[idx];
            }
        }

        // Iterative relaxation for feedforward (handles layered topology)
        for _ in 0..10 {
            for conn in &self.connections {
                if !conn.enabled {
                    continue;
                }
                if conn.in_node < values.len() && conn.out_node < values.len() {
                    values[conn.out_node] += values[conn.in_node] * conn.weight;
                }
            }
        }

        // Apply activations and collect outputs
        for node in &self.nodes {
            if node.node_type != EcNodeType::Input && node.id < values.len() {
                values[node.id] = node.activation.apply(values[node.id]);
            }
        }

        let outputs: Vec<f64> = self
            .nodes
            .iter()
            .filter(|n| n.node_type == EcNodeType::Output)
            .map(|n| {
                if n.id < values.len() {
                    values[n.id]
                } else {
                    0.0
                }
            })
            .collect();

        Ok(outputs)
    }

    /// Count enabled connections.
    pub fn n_parameters(&self) -> usize {
        self.connections.iter().filter(|c| c.enabled).count()
    }

    /// Deep clone of the genome.
    pub fn clone_genome(&self) -> EcGenome {
        self.clone()
    }
}

// ── §2 EcNeat ─────────────────────────────────────────────────────────────────

/// Configuration for NEAT.
#[derive(Debug, Clone)]
pub struct EcNeatConfig {
    pub population_size: usize,
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub weight_mutation_rate: f64,
    pub weight_perturbation: f64,
    pub add_node_rate: f64,
    pub add_connection_rate: f64,
    pub c1: f64,
    pub c2: f64,
    pub c3: f64,
    pub compatibility_threshold: f64,
    pub survival_threshold: f64,
}

impl Default for EcNeatConfig {
    fn default() -> Self {
        EcNeatConfig {
            population_size: 150,
            n_inputs: 2,
            n_outputs: 1,
            weight_mutation_rate: 0.8,
            weight_perturbation: 0.5,
            add_node_rate: 0.03,
            add_connection_rate: 0.05,
            c1: 1.0,
            c2: 1.0,
            c3: 0.4,
            compatibility_threshold: 3.0,
            survival_threshold: 0.2,
        }
    }
}

/// A species in NEAT: a group of genetically similar genomes.
#[derive(Debug, Clone)]
pub struct EcSpecies {
    pub representative: EcGenome,
    pub members: Vec<usize>,
    pub best_fitness: f64,
    pub staleness: usize,
}

/// NEAT (NeuroEvolution of Augmenting Topologies) algorithm.
#[derive(Debug, Clone)]
pub struct EcNeat {
    pub config: EcNeatConfig,
    pub population: Vec<EcGenome>,
    pub species: Vec<EcSpecies>,
    pub generation: usize,
    pub innovation_counter: usize,
    pub global_best: f64,
}

impl EcNeat {
    /// Initialize NEAT with given config; populate with minimal genomes.
    pub fn new(config: EcNeatConfig) -> Self {
        let n_initial_innovations = config.n_inputs * config.n_outputs;
        let population: Vec<EcGenome> = (0..config.population_size)
            .map(|_| EcGenome::new_minimal(config.n_inputs, config.n_outputs))
            .collect();
        EcNeat {
            population,
            species: Vec::new(),
            generation: 0,
            innovation_counter: n_initial_innovations,
            global_best: f64::NEG_INFINITY,
            config,
        }
    }

    /// Stanley 2002 compatibility distance between two genomes.
    pub fn compatibility_distance(&self, g1: &EcGenome, g2: &EcGenome) -> f64 {
        let max_innov1 = g1
            .connections
            .iter()
            .map(|c| c.innovation)
            .max()
            .unwrap_or(0);
        let max_innov2 = g2
            .connections
            .iter()
            .map(|c| c.innovation)
            .max()
            .unwrap_or(0);
        let max_innov = max_innov1.max(max_innov2);

        let n = g1.connections.len().max(g2.connections.len());
        if n == 0 {
            return 0.0;
        }

        let mut excess = 0usize;
        let mut disjoint = 0usize;
        let mut weight_diff_sum = 0.0f64;
        let mut matching = 0usize;

        // Build innovation maps
        let map1: std::collections::HashMap<usize, f64> = g1
            .connections
            .iter()
            .map(|c| (c.innovation, c.weight))
            .collect();
        let map2: std::collections::HashMap<usize, f64> = g2
            .connections
            .iter()
            .map(|c| (c.innovation, c.weight))
            .collect();

        let min_max_innov = max_innov1.min(max_innov2);

        for innov in 0..=max_innov {
            let in1 = map1.contains_key(&innov);
            let in2 = map2.contains_key(&innov);
            match (in1, in2) {
                (true, true) => {
                    weight_diff_sum += (map1[&innov] - map2[&innov]).abs();
                    matching += 1;
                }
                (true, false) | (false, true) => {
                    if innov > min_max_innov {
                        excess += 1;
                    } else {
                        disjoint += 1;
                    }
                }
                (false, false) => {}
            }
        }

        let big_n = n.max(1) as f64;
        let w_bar = if matching > 0 {
            weight_diff_sum / matching as f64
        } else {
            0.0
        };

        self.config.c1 * excess as f64 / big_n
            + self.config.c2 * disjoint as f64 / big_n
            + self.config.c3 * w_bar
    }

    /// Assign each genome to a species by genetic distance.
    pub fn speciate(&mut self) {
        // Clear member lists but keep representatives
        for sp in &mut self.species {
            sp.members.clear();
        }

        for (i, genome) in self.population.iter().enumerate() {
            let mut found = false;
            for sp in &mut self.species {
                let dist = {
                    // Compute distance inline to avoid borrow issues
                    let g1 = genome;
                    let g2 = &sp.representative;
                    let max_innov1 = g1
                        .connections
                        .iter()
                        .map(|c| c.innovation)
                        .max()
                        .unwrap_or(0);
                    let max_innov2 = g2
                        .connections
                        .iter()
                        .map(|c| c.innovation)
                        .max()
                        .unwrap_or(0);
                    let max_innov = max_innov1.max(max_innov2);
                    let n = g1.connections.len().max(g2.connections.len());
                    if n == 0 {
                        0.0
                    } else {
                        let map1: std::collections::HashMap<usize, f64> = g1
                            .connections
                            .iter()
                            .map(|c| (c.innovation, c.weight))
                            .collect();
                        let map2: std::collections::HashMap<usize, f64> = g2
                            .connections
                            .iter()
                            .map(|c| (c.innovation, c.weight))
                            .collect();
                        let min_max = max_innov1.min(max_innov2);
                        let mut excess = 0usize;
                        let mut disjoint = 0usize;
                        let mut wdiff = 0.0f64;
                        let mut match_count = 0usize;
                        for innov in 0..=max_innov {
                            match (map1.contains_key(&innov), map2.contains_key(&innov)) {
                                (true, true) => {
                                    wdiff += (map1[&innov] - map2[&innov]).abs();
                                    match_count += 1;
                                }
                                (true, false) | (false, true) => {
                                    if innov > min_max {
                                        excess += 1;
                                    } else {
                                        disjoint += 1;
                                    }
                                }
                                _ => {}
                            }
                        }
                        let big_n = n.max(1) as f64;
                        let w_bar = if match_count > 0 {
                            wdiff / match_count as f64
                        } else {
                            0.0
                        };
                        self.config.c1 * excess as f64 / big_n
                            + self.config.c2 * disjoint as f64 / big_n
                            + self.config.c3 * w_bar
                    }
                };
                if dist < self.config.compatibility_threshold {
                    sp.members.push(i);
                    found = true;
                    break;
                }
            }
            if !found {
                let new_sp = EcSpecies {
                    representative: genome.clone(),
                    members: vec![i],
                    best_fitness: genome.fitness,
                    staleness: 0,
                };
                self.species.push(new_sp);
            }
        }

        // Remove empty species
        self.species.retain(|sp| !sp.members.is_empty());
    }

    /// Crossover: align by innovation number, inherit matching from both parents.
    pub fn crossover(&self, parent1: &EcGenome, parent2: &EcGenome, seed: &mut u64) -> EcGenome {
        let (fitter, weaker) = if parent1.fitness >= parent2.fitness {
            (parent1, parent2)
        } else {
            (parent2, parent1)
        };

        let map_weaker: std::collections::HashMap<usize, &EcConnectionGene> = weaker
            .connections
            .iter()
            .map(|c| (c.innovation, c))
            .collect();

        let mut child_conns: Vec<EcConnectionGene> = Vec::new();
        for conn in &fitter.connections {
            if let Some(other_conn) = map_weaker.get(&conn.innovation) {
                // Matching gene: inherit from random parent
                if ec_rand01(seed) < 0.5 {
                    child_conns.push(conn.clone());
                } else {
                    child_conns.push((*other_conn).clone());
                }
            } else {
                // Excess/disjoint: inherit from fitter
                child_conns.push(conn.clone());
            }
        }

        EcGenome {
            nodes: fitter.nodes.clone(),
            connections: child_conns,
            fitness: 0.0,
        }
    }

    /// Mutate a genome: weight mutation and structural mutations.
    pub fn mutate(&mut self, genome: &mut EcGenome, seed: &mut u64) {
        // Weight mutation
        if ec_rand01(seed) < self.config.weight_mutation_rate {
            for conn in &mut genome.connections {
                if ec_rand01(seed) < 0.9 {
                    // Perturb
                    conn.weight += ec_randn(seed) * self.config.weight_perturbation;
                } else {
                    // Replace
                    conn.weight = ec_randn(seed);
                }
            }
        }

        // Add node mutation: split an existing connection
        if ec_rand01(seed) < self.config.add_node_rate && !genome.connections.is_empty() {
            let idx = ec_rand_usize(seed, genome.connections.len());
            if genome.connections[idx].enabled {
                genome.connections[idx].enabled = false;
                let old_in = genome.connections[idx].in_node;
                let old_out = genome.connections[idx].out_node;
                let old_weight = genome.connections[idx].weight;
                let new_node_id = genome.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;
                genome.nodes.push(EcNodeGene {
                    id: new_node_id,
                    node_type: EcNodeType::Hidden,
                    activation: EcActivation::Tanh,
                });
                genome.connections.push(EcConnectionGene {
                    in_node: old_in,
                    out_node: new_node_id,
                    weight: 1.0,
                    enabled: true,
                    innovation: self.innovation_counter,
                });
                self.innovation_counter += 1;
                genome.connections.push(EcConnectionGene {
                    in_node: new_node_id,
                    out_node: old_out,
                    weight: old_weight,
                    enabled: true,
                    innovation: self.innovation_counter,
                });
                self.innovation_counter += 1;
            }
        }

        // Add connection mutation
        if ec_rand01(seed) < self.config.add_connection_rate {
            let node_ids: Vec<usize> = genome.nodes.iter().map(|n| n.id).collect();
            if node_ids.len() >= 2 {
                let in_idx = ec_rand_usize(seed, node_ids.len());
                let out_idx = ec_rand_usize(seed, node_ids.len());
                if in_idx != out_idx {
                    let in_id = node_ids[in_idx];
                    let out_id = node_ids[out_idx];
                    let exists = genome
                        .connections
                        .iter()
                        .any(|c| c.in_node == in_id && c.out_node == out_id);
                    if !exists {
                        genome.connections.push(EcConnectionGene {
                            in_node: in_id,
                            out_node: out_id,
                            weight: ec_randn(seed),
                            enabled: true,
                            innovation: self.innovation_counter,
                        });
                        self.innovation_counter += 1;
                    }
                }
            }
        }
    }

    /// Run one generation: assign fitnesses, speciate, reproduce.
    pub fn evolve_generation(&mut self, fitnesses: &[f64], seed: &mut u64) -> Result<(), EcError> {
        if fitnesses.len() != self.population.len() {
            return Err(EcError::InvalidConfig(format!(
                "fitnesses len {} != population len {}",
                fitnesses.len(),
                self.population.len()
            )));
        }

        // Assign fitnesses
        for (genome, &fit) in self.population.iter_mut().zip(fitnesses.iter()) {
            genome.fitness = fit;
            if fit > self.global_best {
                self.global_best = fit;
            }
        }

        self.speciate();

        if self.species.is_empty() {
            return Err(EcError::EmptyPopulation);
        }

        // Compute species adjusted fitnesses (fitness sharing)
        let mut species_total: Vec<f64> = Vec::with_capacity(self.species.len());
        for sp in &self.species {
            let n = sp.members.len() as f64;
            let sum: f64 = sp
                .members
                .iter()
                .map(|&idx| fitnesses[idx].max(0.0) / n)
                .sum();
            species_total.push(sum);
        }

        let grand_total: f64 = species_total.iter().sum();

        // Determine offspring count per species
        let pop_size = self.config.population_size;
        let mut offspring_counts: Vec<usize> = species_total
            .iter()
            .map(|&s| {
                if grand_total > 0.0 {
                    ((s / grand_total) * pop_size as f64).round() as usize
                } else {
                    pop_size / self.species.len()
                }
            })
            .collect();

        // Adjust to ensure exactly pop_size offspring
        let total_offspring: usize = offspring_counts.iter().sum();
        if total_offspring < pop_size && !offspring_counts.is_empty() {
            offspring_counts[0] += pop_size - total_offspring;
        } else {
            let mut excess = total_offspring.saturating_sub(pop_size);
            for count in offspring_counts.iter_mut().rev() {
                if excess == 0 {
                    break;
                }
                let sub = (*count).min(excess);
                *count -= sub;
                excess -= sub;
            }
        }

        // Collect species reproduction plans without holding borrow on self.species
        struct SpeciesPlan {
            n_offspring: usize,
            survivors: Vec<usize>, // population indices of survivors
        }
        let species_plans: Vec<SpeciesPlan> = self
            .species
            .iter()
            .enumerate()
            .map(|(sp_idx, sp)| {
                let n_offspring = if sp_idx < offspring_counts.len() {
                    offspring_counts[sp_idx]
                } else {
                    0
                };
                let mut sorted_members = sp.members.clone();
                sorted_members.sort_unstable_by(|&a, &b| {
                    fitnesses[b]
                        .partial_cmp(&fitnesses[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let n_survivors = ((sorted_members.len() as f64 * self.config.survival_threshold)
                    .ceil() as usize)
                    .max(1);
                let survivors = sorted_members[..n_survivors.min(sorted_members.len())].to_vec();
                SpeciesPlan {
                    n_offspring,
                    survivors,
                }
            })
            .collect();

        // Reproduce within species
        let mut new_population: Vec<EcGenome> = Vec::with_capacity(pop_size);

        for plan in &species_plans {
            if plan.n_offspring == 0 || plan.survivors.is_empty() {
                continue;
            }
            let survivors = &plan.survivors;

            for i in 0..plan.n_offspring {
                if i == 0 {
                    // Elite: copy best genome
                    new_population.push(self.population[survivors[0]].clone_genome());
                } else {
                    // Select two parents from survivors
                    let p1_idx = survivors[ec_rand_usize(seed, survivors.len())];
                    let p2_idx = survivors[ec_rand_usize(seed, survivors.len())];
                    // Crossover: done by cloning parent data before mutating
                    let parent1 = self.population[p1_idx].clone_genome();
                    let parent2 = self.population[p2_idx].clone_genome();
                    let mut child = self.crossover(&parent1, &parent2, seed);
                    self.mutate(&mut child, seed);
                    new_population.push(child);
                }
            }
        }

        // Pad if needed
        while new_population.len() < pop_size {
            let mut g = EcGenome::new_minimal(self.config.n_inputs, self.config.n_outputs);
            self.mutate(&mut g, seed);
            new_population.push(g);
        }
        new_population.truncate(pop_size);

        self.population = new_population;
        self.generation += 1;

        Ok(())
    }

    /// Return the genome with the highest fitness.
    pub fn best_genome(&self) -> Option<&EcGenome> {
        self.population.iter().max_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

// ── §3 EcEvolutionStrategies — OpenAI ES ─────────────────────────────────────

/// OpenAI Evolution Strategies (Salimans et al. 2017).
#[derive(Debug, Clone)]
pub struct EcEvolutionStrategies {
    pub theta: Vec<f64>,
    pub sigma: f64,
    pub lr: f64,
    pub population_size: usize,
    pub dim: usize,
    pub generation: usize,
    pub best_fitness: f64,
    pub fitness_history: Vec<f64>,
}

impl EcEvolutionStrategies {
    pub fn new(dim: usize, sigma: f64, lr: f64, population_size: usize) -> Self {
        EcEvolutionStrategies {
            theta: vec![0.0; dim],
            sigma,
            lr,
            population_size,
            dim,
            generation: 0,
            best_fitness: f64::NEG_INFINITY,
            fitness_history: Vec::new(),
        }
    }

    /// Sample population_size/2 antithetic pairs: (theta+sigma*eps, theta-sigma*eps).
    pub fn ask(&self, seed: &mut u64) -> Vec<Vec<f64>> {
        let n_pairs = (self.population_size / 2).max(1);
        let mut candidates = Vec::with_capacity(n_pairs * 2);
        for _ in 0..n_pairs {
            let eps: Vec<f64> = (0..self.dim).map(|_| ec_randn(seed)).collect();
            let pos: Vec<f64> = self
                .theta
                .iter()
                .zip(&eps)
                .map(|(&t, &e)| t + self.sigma * e)
                .collect();
            let neg: Vec<f64> = self
                .theta
                .iter()
                .zip(&eps)
                .map(|(&t, &e)| t - self.sigma * e)
                .collect();
            candidates.push(pos);
            candidates.push(neg);
        }
        candidates
    }

    /// Centered rank normalization: rank / N - 0.5.
    pub fn rank_normalize(fitnesses: &[f64]) -> Vec<f64> {
        let n = fitnesses.len();
        if n == 0 {
            return Vec::new();
        }
        // Get sorted indices
        let mut indexed: Vec<(usize, f64)> = fitnesses.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut ranks = vec![0.0f64; n];
        for (rank, (orig_idx, _)) in indexed.iter().enumerate() {
            ranks[*orig_idx] = rank as f64 / (n as f64 - 1.0).max(1.0) - 0.5;
        }
        ranks
    }

    /// Compute ES gradient update given fitness values and the perturbations.
    pub fn tell(&mut self, fitness_values: &[f64], perturbations: &[Vec<f64>]) {
        if fitness_values.is_empty() || perturbations.is_empty() {
            return;
        }
        let n = fitness_values.len().min(perturbations.len());
        let ranked = Self::rank_normalize(&fitness_values[..n]);

        // Compute gradient: sum_i ranked_i * eps_i
        let mut grad = vec![0.0f64; self.dim];
        for i in 0..n {
            let eps_i = if i < perturbations.len() {
                &perturbations[i]
            } else {
                continue;
            };
            for j in 0..self.dim.min(eps_i.len()) {
                grad[j] += ranked[i] * eps_i[j];
            }
        }

        let scale = self.lr / (n as f64 * self.sigma);
        for j in 0..self.dim {
            self.theta[j] += scale * grad[j];
        }
    }

    /// Complete one generation: ask, evaluate, tell.
    pub fn step(&mut self, fitness_fn: &dyn Fn(&[f64]) -> f64, seed: &mut u64) -> f64 {
        let candidates = self.ask(seed);
        let fitnesses: Vec<f64> = candidates.iter().map(|c| fitness_fn(c)).collect();
        let best = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if best > self.best_fitness {
            self.best_fitness = best;
        }

        // Compute perturbations (eps = (candidate - theta) / sigma)
        let perturbations: Vec<Vec<f64>> = candidates
            .iter()
            .map(|c| {
                c.iter()
                    .zip(&self.theta)
                    .map(|(&ci, &ti)| (ci - ti) / self.sigma)
                    .collect()
            })
            .collect();

        self.tell(&fitnesses, &perturbations);
        self.generation += 1;
        self.fitness_history.push(best);
        best
    }

    pub fn get_solution(&self) -> &[f64] {
        &self.theta
    }
}

// ── §4 EcMapElites ────────────────────────────────────────────────────────────

/// Grid storage for MAP-Elites.
#[derive(Debug, Clone)]
pub struct EcMapElitesGrid {
    pub cells: Vec<Option<(Vec<f64>, f64)>>,
    pub n_dims: usize,
    pub n_bins: usize,
    pub filled: usize,
}

impl EcMapElitesGrid {
    fn new(n_dims: usize, n_bins: usize) -> Self {
        let total = n_bins.pow(n_dims as u32);
        EcMapElitesGrid {
            cells: vec![None; total],
            n_dims,
            n_bins,
            filled: 0,
        }
    }
}

/// MAP-Elites (Mouret & Clune 2015): quality-diversity optimization.
#[derive(Debug, Clone)]
pub struct EcMapElites {
    pub grid: EcMapElitesGrid,
    pub dim: usize,
    pub bd_dim: usize,
    pub sigma: f64,
    pub generation: usize,
}

impl EcMapElites {
    pub fn new(dim: usize, bd_dim: usize, n_bins: usize, sigma: f64) -> Self {
        EcMapElites {
            grid: EcMapElitesGrid::new(bd_dim, n_bins),
            dim,
            bd_dim,
            sigma,
            generation: 0,
        }
    }

    /// Sample random solution in [-1, 1]^dim.
    pub fn random_solution(&self, seed: &mut u64) -> Vec<f64> {
        (0..self.dim).map(|_| ec_rand01(seed) * 2.0 - 1.0).collect()
    }

    /// Gaussian perturbation.
    pub fn mutate_solution(&self, x: &[f64], seed: &mut u64) -> Vec<f64> {
        x.iter()
            .map(|&xi| xi + self.sigma * ec_randn(seed))
            .collect()
    }

    /// Map behavior descriptor to flat cell index.
    fn bd_to_cell(&self, bd: &[f64]) -> usize {
        let mut idx = 0usize;
        let nb = self.grid.n_bins;
        for &b in bd.iter().take(self.bd_dim) {
            let bin = ((b.clamp(0.0, 1.0 - 1e-9)) * nb as f64) as usize;
            idx = idx * nb + bin.min(nb - 1);
        }
        idx.min(self.grid.cells.len().saturating_sub(1))
    }

    /// Add solution to grid if cell is empty or has lower fitness. Returns true if grid updated.
    pub fn add_to_grid(&mut self, solution: Vec<f64>, fitness: f64, bd: &[f64]) -> bool {
        let cell_idx = self.bd_to_cell(bd);
        if cell_idx >= self.grid.cells.len() {
            return false;
        }
        match &self.grid.cells[cell_idx] {
            None => {
                self.grid.cells[cell_idx] = Some((solution, fitness));
                self.grid.filled += 1;
                true
            }
            Some((_, existing_fitness)) => {
                if fitness > *existing_fitness {
                    self.grid.cells[cell_idx] = Some((solution, fitness));
                    true
                } else {
                    false
                }
            }
        }
    }

    /// One MAP-Elites step: sample, mutate, evaluate, add.
    pub fn step(&mut self, fitness_fn: &dyn Fn(&[f64]) -> (f64, Vec<f64>), seed: &mut u64) {
        let x = if self.grid.filled > 0 && ec_rand01(seed) < 0.9 {
            // Sample from filled cells
            let filled_cells: Vec<usize> = self
                .grid
                .cells
                .iter()
                .enumerate()
                .filter_map(|(i, c)| c.as_ref().map(|_| i))
                .collect();
            if !filled_cells.is_empty() {
                let idx = ec_rand_usize(seed, filled_cells.len());
                let cell_idx = filled_cells[idx];
                if let Some((sol, _)) = &self.grid.cells[cell_idx] {
                    self.mutate_solution(sol, seed)
                } else {
                    self.random_solution(seed)
                }
            } else {
                self.random_solution(seed)
            }
        } else {
            self.random_solution(seed)
        };

        let x_mutated = self.mutate_solution(&x, seed);
        let (fitness, bd) = fitness_fn(&x_mutated);
        self.add_to_grid(x_mutated, fitness, &bd);
        self.generation += 1;
    }

    pub fn coverage(&self) -> f64 {
        let total = self.grid.cells.len();
        if total == 0 {
            return 0.0;
        }
        self.grid.filled as f64 / total as f64
    }

    pub fn max_fitness(&self) -> f64 {
        self.grid
            .cells
            .iter()
            .filter_map(|c| c.as_ref().map(|(_, f)| *f))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn mean_fitness(&self) -> f64 {
        let vals: Vec<f64> = self
            .grid
            .cells
            .iter()
            .filter_map(|c| c.as_ref().map(|(_, f)| *f))
            .collect();
        if vals.is_empty() {
            return 0.0;
        }
        vals.iter().sum::<f64>() / vals.len() as f64
    }

    pub fn qd_score(&self) -> f64 {
        self.grid
            .cells
            .iter()
            .filter_map(|c| c.as_ref().map(|(_, f)| f.max(0.0)))
            .sum()
    }
}

// ── §5 EcNoveltySearch ────────────────────────────────────────────────────────

/// Novelty Search (Lehman & Stanley 2011): reward behavioral novelty.
#[derive(Debug, Clone)]
pub struct EcNoveltySearch {
    pub archive: Vec<Vec<f64>>,
    pub k: usize,
    pub novelty_threshold: f64,
    pub archive_prob: f64,
    pub population: Vec<(Vec<f64>, Vec<f64>)>,
    pub pop_size: usize,
    pub dim: usize,
    pub bd_dim: usize,
    pub generation: usize,
}

impl EcNoveltySearch {
    pub fn new(pop_size: usize, dim: usize, bd_dim: usize, k: usize) -> Self {
        EcNoveltySearch {
            archive: Vec::new(),
            k,
            novelty_threshold: 0.01,
            archive_prob: 0.3,
            population: Vec::new(),
            pop_size,
            dim,
            bd_dim,
            generation: 0,
        }
    }

    /// Sorted Euclidean distances from bd to all behaviors in `all_bds`.
    pub fn knn_distances(&self, bd: &[f64], all_bds: &[Vec<f64>]) -> Vec<f64> {
        let mut dists: Vec<f64> = all_bds
            .iter()
            .map(|other| {
                bd.iter()
                    .zip(other.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        dists.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        dists
    }

    /// Average distance to k nearest neighbors in population + archive.
    pub fn novelty_score(&self, bd: &[f64]) -> f64 {
        let mut all_bds: Vec<Vec<f64>> = self.population.iter().map(|(_, b)| b.clone()).collect();
        all_bds.extend(self.archive.iter().cloned());

        if all_bds.is_empty() {
            return 0.0;
        }

        let dists = self.knn_distances(bd, &all_bds);
        let k_actual = self.k.min(dists.len());
        if k_actual == 0 {
            return 0.0;
        }
        dists[..k_actual].iter().sum::<f64>() / k_actual as f64
    }

    /// One evolution step: tournament selection by novelty, mutate, add novel to archive.
    pub fn evolve_step(
        &mut self,
        fitness_fn: &dyn Fn(&[f64]) -> (f64, Vec<f64>),
        seed: &mut u64,
    ) -> f64 {
        // Initialize population if empty
        if self.population.is_empty() {
            for _ in 0..self.pop_size {
                let x: Vec<f64> = (0..self.dim).map(|_| ec_rand01(seed) * 2.0 - 1.0).collect();
                let (_, bd) = fitness_fn(&x);
                self.population.push((x, bd));
            }
        }

        // Compute novelty scores for each individual
        let pop_bds: Vec<Vec<f64>> = self.population.iter().map(|(_, b)| b.clone()).collect();

        let novelties: Vec<f64> = pop_bds.iter().map(|bd| self.novelty_score(bd)).collect();

        let mut new_pop: Vec<(Vec<f64>, Vec<f64>)> = Vec::with_capacity(self.pop_size);

        for _ in 0..self.pop_size {
            // Tournament selection (size 3) by novelty
            let tournament_size = 3.min(self.population.len());
            let mut best_nov = f64::NEG_INFINITY;
            let mut best_idx = 0;
            for _ in 0..tournament_size {
                let idx = ec_rand_usize(seed, self.population.len());
                if novelties[idx] > best_nov {
                    best_nov = novelties[idx];
                    best_idx = idx;
                }
            }

            let parent = &self.population[best_idx].0;
            // Gaussian mutation
            let sigma = 0.1;
            let child: Vec<f64> = parent.iter().map(|&p| p + sigma * ec_randn(seed)).collect();
            let (_, bd) = fitness_fn(&child);

            let novelty = self.novelty_score(&bd);
            // Add to archive with some probability if novel enough
            if novelty > self.novelty_threshold || ec_rand01(seed) < self.archive_prob {
                self.archive.push(bd.clone());
            }
            new_pop.push((child, bd));
        }

        self.population = new_pop;
        self.generation += 1;

        // Return mean novelty
        let mean_novelty: f64 = novelties.iter().sum::<f64>() / novelties.len().max(1) as f64;
        mean_novelty
    }

    pub fn archive_size(&self) -> usize {
        self.archive.len()
    }
}

// §6–§10 (EcDifferentialEvolution, EcParticleSwarm, EcGeneticProgramming,
// EcNsga3, EcMetrics) live in ec_advanced.rs, re-exported via `pub use ec_advanced::*`.
