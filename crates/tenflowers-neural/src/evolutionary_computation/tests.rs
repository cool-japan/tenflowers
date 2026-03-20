//! Tests for evolutionary_computation module.

use super::*;

// ── RNG helpers ───────────────────────────────────────────────────────────────

#[test]
fn test_ec_rand01_range() {
    let mut seed = 12345u64;
    for _ in 0..1000 {
        let v = ec_rand01(&mut seed);
        assert!((0.0..1.0).contains(&v), "ec_rand01 out of range: {}", v);
    }
}

#[test]
fn test_ec_randn_finite() {
    let mut seed = 99999u64;
    for _ in 0..500 {
        let v = ec_randn(&mut seed);
        assert!(v.is_finite(), "ec_randn returned non-finite: {}", v);
    }
}

// ── §1 EcGenome ───────────────────────────────────────────────────────────────

#[test]
fn test_genome_new_minimal_nodes() {
    let g = EcGenome::new_minimal(3, 2);
    assert_eq!(g.nodes.len(), 5, "3 inputs + 2 outputs = 5 nodes");
    let n_inputs = g
        .nodes
        .iter()
        .filter(|n| n.node_type == EcNodeType::Input)
        .count();
    let n_outputs = g
        .nodes
        .iter()
        .filter(|n| n.node_type == EcNodeType::Output)
        .count();
    assert_eq!(n_inputs, 3);
    assert_eq!(n_outputs, 2);
}

#[test]
fn test_genome_new_minimal_connections() {
    let g = EcGenome::new_minimal(3, 2);
    assert_eq!(
        g.connections.len(),
        6,
        "3 inputs × 2 outputs = 6 connections"
    );
    for conn in &g.connections {
        assert!(conn.enabled);
    }
}

#[test]
fn test_genome_innovation_numbers() {
    let g = EcGenome::new_minimal(2, 3);
    let innovs: Vec<usize> = g.connections.iter().map(|c| c.innovation).collect();
    // Should be 0..=5
    assert_eq!(innovs.len(), 6);
    for (i, &innov) in innovs.iter().enumerate() {
        assert_eq!(innov, i);
    }
}

#[test]
fn test_genome_activate_shape() {
    let mut g = EcGenome::new_minimal(2, 1);
    // Set weights to known values for reproducible output
    for conn in &mut g.connections {
        conn.weight = 0.5;
    }
    let out = g.activate(&[1.0, 1.0]).expect("activate failed");
    assert_eq!(out.len(), 1, "should have 1 output");
    assert!(out[0].is_finite());
}

#[test]
fn test_genome_activate_wrong_input_count() {
    let g = EcGenome::new_minimal(3, 1);
    let result = g.activate(&[1.0, 2.0]); // wrong: need 3
    assert!(result.is_err());
}

#[test]
fn test_genome_n_parameters() {
    let g = EcGenome::new_minimal(4, 2);
    assert_eq!(g.n_parameters(), 8);
}

#[test]
fn test_genome_n_parameters_after_disable() {
    let mut g = EcGenome::new_minimal(2, 2);
    g.connections[0].enabled = false;
    assert_eq!(g.n_parameters(), 3);
}

#[test]
fn test_genome_clone() {
    let g = EcGenome::new_minimal(2, 2);
    let gc = g.clone_genome();
    assert_eq!(gc.nodes.len(), g.nodes.len());
    assert_eq!(gc.connections.len(), g.connections.len());
}

#[test]
fn test_genome_activate_sigmoid_output_range() {
    let mut g = EcGenome::new_minimal(2, 2);
    for conn in &mut g.connections {
        conn.weight = 1.0;
    }
    let out = g.activate(&[0.5, -0.5]).expect("activate");
    for &v in &out {
        assert!((0.0..=1.0).contains(&v), "sigmoid output out of [0,1]: {}", v);
    }
}

// ── §2 EcNeat ─────────────────────────────────────────────────────────────────

#[test]
fn test_neat_new_population_size() {
    let cfg = EcNeatConfig {
        population_size: 20,
        n_inputs: 2,
        n_outputs: 1,
        ..EcNeatConfig::default()
    };
    let neat = EcNeat::new(cfg);
    assert_eq!(neat.population.len(), 20);
}

#[test]
fn test_neat_compatibility_distance_same_genome() {
    let cfg = EcNeatConfig::default();
    let neat = EcNeat::new(cfg);
    let g = EcGenome::new_minimal(2, 1);
    let dist = neat.compatibility_distance(&g, &g);
    // Same genome: excess=0, disjoint=0, W_bar≈0 (same weights)
    assert!(
        dist.abs() < 1e-9,
        "same genome should have distance ≈0, got {}",
        dist
    );
}

#[test]
fn test_neat_compatibility_distance_nonneg() {
    let cfg = EcNeatConfig::default();
    let neat = EcNeat::new(cfg);
    let g1 = EcGenome::new_minimal(2, 1);
    let g2 = EcGenome::new_minimal(2, 1);
    let dist = neat.compatibility_distance(&g1, &g2);
    assert!(dist >= 0.0);
}

#[test]
fn test_neat_speciate_creates_species() {
    let cfg = EcNeatConfig {
        population_size: 10,
        n_inputs: 2,
        n_outputs: 1,
        ..EcNeatConfig::default()
    };
    let mut neat = EcNeat::new(cfg);
    // Assign different fitnesses
    for (i, g) in neat.population.iter_mut().enumerate() {
        g.fitness = i as f64;
    }
    neat.speciate();
    assert!(!neat.species.is_empty(), "should have at least one species");
}

#[test]
fn test_neat_speciate_all_members_assigned() {
    let cfg = EcNeatConfig {
        population_size: 15,
        n_inputs: 2,
        n_outputs: 1,
        ..EcNeatConfig::default()
    };
    let mut neat = EcNeat::new(cfg);
    neat.speciate();
    let total_members: usize = neat.species.iter().map(|s| s.members.len()).sum();
    assert_eq!(total_members, 15, "all individuals should be in a species");
}

#[test]
fn test_neat_crossover_valid_genome() {
    let cfg = EcNeatConfig::default();
    let neat = EcNeat::new(cfg);
    let mut p1 = EcGenome::new_minimal(2, 1);
    let mut p2 = EcGenome::new_minimal(2, 1);
    p1.fitness = 1.0;
    p2.fitness = 0.5;
    let mut seed = 42u64;
    let child = neat.crossover(&p1, &p2, &mut seed);
    assert!(!child.nodes.is_empty());
    assert!(!child.connections.is_empty());
}

#[test]
fn test_neat_mutate_no_crash() {
    let cfg = EcNeatConfig {
        weight_mutation_rate: 1.0,
        ..EcNeatConfig::default()
    };
    let mut neat = EcNeat::new(cfg);
    let mut g = EcGenome::new_minimal(2, 1);
    let mut seed = 1234u64;
    neat.mutate(&mut g, &mut seed);
    // After mutation, connections should still exist
    assert!(!g.nodes.is_empty());
}

#[test]
fn test_neat_evolve_generation_ok() {
    let cfg = EcNeatConfig {
        population_size: 10,
        n_inputs: 2,
        n_outputs: 1,
        ..EcNeatConfig::default()
    };
    let mut neat = EcNeat::new(cfg);
    let fitnesses: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let mut seed = 777u64;
    neat.evolve_generation(&fitnesses, &mut seed)
        .expect("evolve_generation failed");
    assert_eq!(
        neat.population.len(),
        10,
        "population size should be maintained"
    );
    assert_eq!(neat.generation, 1);
}

#[test]
fn test_neat_evolve_generation_wrong_fitnesses_size() {
    let cfg = EcNeatConfig {
        population_size: 10,
        n_inputs: 2,
        n_outputs: 1,
        ..EcNeatConfig::default()
    };
    let mut neat = EcNeat::new(cfg);
    let fitnesses = vec![1.0; 5]; // wrong size
    let mut seed = 1u64;
    let result = neat.evolve_generation(&fitnesses, &mut seed);
    assert!(result.is_err());
}

#[test]
fn test_neat_best_genome_returns_some() {
    let cfg = EcNeatConfig {
        population_size: 5,
        n_inputs: 2,
        n_outputs: 1,
        ..EcNeatConfig::default()
    };
    let mut neat = EcNeat::new(cfg);
    neat.population[0].fitness = 10.0;
    neat.population[2].fitness = 5.0;
    let best = neat.best_genome();
    assert!(best.is_some());
    assert_eq!(best.expect("best_genome should return Some").fitness, 10.0);
}

// ── §3 EcEvolutionStrategies ──────────────────────────────────────────────────

#[test]
fn test_es_ask_returns_population_size() {
    let es = EcEvolutionStrategies::new(10, 0.1, 0.01, 20);
    let mut seed = 42u64;
    let candidates = es.ask(&mut seed);
    assert_eq!(
        candidates.len(),
        20,
        "should return population_size candidates"
    );
}

#[test]
fn test_es_ask_antithetic_pairs() {
    let es = EcEvolutionStrategies::new(4, 0.1, 0.01, 6);
    let mut seed = 100u64;
    let candidates = es.ask(&mut seed);
    // antithetic: first and second should be symmetric around theta
    let theta = &es.theta;
    let mid: Vec<f64> = candidates[0]
        .iter()
        .zip(&candidates[1])
        .map(|(&a, &b)| (a + b) / 2.0)
        .collect();
    for (m, t) in mid.iter().zip(theta.iter()) {
        assert!(
            (m - t).abs() < 1e-10,
            "antithetic pairs should average to theta"
        );
    }
}

#[test]
fn test_es_tell_updates_theta() {
    let mut es = EcEvolutionStrategies::new(4, 0.1, 0.1, 10);
    let mut seed = 55u64;
    let candidates = es.ask(&mut seed);
    let fitnesses: Vec<f64> = candidates.iter().map(|c| c[0]).collect();
    let perturbations: Vec<Vec<f64>> = candidates
        .iter()
        .map(|c| {
            c.iter()
                .zip(&es.theta)
                .map(|(&ci, &ti)| (ci - ti) / es.sigma)
                .collect()
        })
        .collect();
    let theta_before = es.theta.clone();
    es.tell(&fitnesses, &perturbations);
    let changed = theta_before
        .iter()
        .zip(&es.theta)
        .any(|(&a, &b)| (a - b).abs() > 1e-15);
    assert!(changed, "tell should update theta");
}

#[test]
fn test_es_rank_normalize_range() {
    let fitnesses = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let ranked = EcEvolutionStrategies::rank_normalize(&fitnesses);
    assert_eq!(ranked.len(), fitnesses.len());
    for &r in &ranked {
        assert!(
            (-0.5..=0.5).contains(&r),
            "rank should be in [-0.5, 0.5], got {}",
            r
        );
    }
}

#[test]
fn test_es_rank_normalize_empty() {
    let ranked = EcEvolutionStrategies::rank_normalize(&[]);
    assert!(ranked.is_empty());
}

#[test]
fn test_es_step_improves() {
    let mut es = EcEvolutionStrategies::new(2, 0.1, 0.05, 10);
    let mut seed = 42u64;
    // Fitness: negative sum of squares (maximize -> converge to 0)
    let fitness_fn = |x: &[f64]| -> f64 { -(x[0].powi(2) + x[1].powi(2)) };
    for _ in 0..20 {
        es.step(&fitness_fn, &mut seed);
    }
    assert!(es.best_fitness > f64::NEG_INFINITY);
    assert!(!es.fitness_history.is_empty());
}

#[test]
fn test_es_get_solution() {
    let es = EcEvolutionStrategies::new(5, 0.1, 0.01, 10);
    assert_eq!(es.get_solution().len(), 5);
}

// ── §4 EcMapElites ────────────────────────────────────────────────────────────

#[test]
fn test_mapelites_new() {
    let me = EcMapElites::new(4, 2, 5, 0.1);
    assert_eq!(me.grid.cells.len(), 25); // 5^2
    assert_eq!(me.grid.filled, 0);
}

#[test]
fn test_mapelites_add_to_grid_fills_cell() {
    let mut me = EcMapElites::new(4, 2, 5, 0.1);
    let added = me.add_to_grid(vec![0.1; 4], 1.0, &[0.1, 0.1]);
    assert!(added);
    assert_eq!(me.grid.filled, 1);
}

#[test]
fn test_mapelites_add_to_grid_improves() {
    let mut me = EcMapElites::new(4, 2, 5, 0.1);
    me.add_to_grid(vec![0.1; 4], 1.0, &[0.1, 0.1]);
    let improved = me.add_to_grid(vec![0.2; 4], 2.0, &[0.1, 0.1]);
    assert!(improved, "better fitness should replace cell");
    assert_eq!(
        me.grid.filled, 1,
        "filled count should not change for same cell"
    );
}

#[test]
fn test_mapelites_add_to_grid_no_improvement() {
    let mut me = EcMapElites::new(4, 2, 5, 0.1);
    me.add_to_grid(vec![0.1; 4], 5.0, &[0.1, 0.1]);
    let updated = me.add_to_grid(vec![0.2; 4], 1.0, &[0.1, 0.1]);
    assert!(!updated, "lower fitness should not update cell");
}

#[test]
fn test_mapelites_coverage_range() {
    let mut me = EcMapElites::new(4, 2, 4, 0.1);
    assert_eq!(me.coverage(), 0.0);
    me.add_to_grid(vec![0.0; 4], 1.0, &[0.0, 0.0]);
    let cov = me.coverage();
    assert!(cov > 0.0 && cov <= 1.0);
}

#[test]
fn test_mapelites_qd_score_nonneg() {
    let mut me = EcMapElites::new(4, 2, 5, 0.1);
    me.add_to_grid(vec![0.1; 4], 2.5, &[0.1, 0.1]);
    me.add_to_grid(vec![0.5; 4], 1.5, &[0.5, 0.5]);
    assert!(me.qd_score() >= 0.0);
}

#[test]
fn test_mapelites_step_increases_filled() {
    let mut me = EcMapElites::new(4, 2, 8, 0.5);
    let mut seed = 42u64;
    let fitness_fn = |x: &[f64]| -> (f64, Vec<f64>) {
        let fit = -(x[0].powi(2) + x[1].powi(2));
        let bd = vec![
            (x[0] * 0.5 + 0.5).clamp(0.0, 0.999),
            (x[1] * 0.5 + 0.5).clamp(0.0, 0.999),
        ];
        (fit, bd)
    };
    for _ in 0..30 {
        me.step(&fitness_fn, &mut seed);
    }
    assert!(me.grid.filled > 0, "should have filled some cells");
}

#[test]
fn test_mapelites_max_mean_fitness_after_adding() {
    let mut me = EcMapElites::new(4, 2, 5, 0.1);
    me.add_to_grid(vec![0.1; 4], 3.0, &[0.1, 0.1]);
    me.add_to_grid(vec![0.5; 4], 1.0, &[0.5, 0.5]);
    assert_eq!(me.max_fitness(), 3.0);
    assert!((me.mean_fitness() - 2.0).abs() < 1e-10);
}

// ── §5 EcNoveltySearch ────────────────────────────────────────────────────────

#[test]
fn test_novelty_search_new() {
    let ns = EcNoveltySearch::new(20, 4, 2, 5);
    assert_eq!(ns.pop_size, 20);
    assert_eq!(ns.k, 5);
    assert!(ns.archive.is_empty());
    assert!(ns.population.is_empty());
}

#[test]
fn test_novelty_score_empty_population() {
    let ns = EcNoveltySearch::new(10, 4, 2, 3);
    let score = ns.novelty_score(&[0.5, 0.5]);
    assert_eq!(score, 0.0);
}

#[test]
fn test_novelty_score_nonneg() {
    let mut ns = EcNoveltySearch::new(5, 2, 2, 2);
    ns.population = vec![
        (vec![0.0, 0.0], vec![0.0, 0.0]),
        (vec![1.0, 0.0], vec![1.0, 0.0]),
        (vec![0.0, 1.0], vec![0.0, 1.0]),
    ];
    let score = ns.novelty_score(&[0.5, 0.5]);
    assert!(score >= 0.0, "novelty score should be non-negative");
}

#[test]
fn test_novelty_knn_distances_sorted() {
    let ns = EcNoveltySearch::new(5, 2, 2, 2);
    let all_bds = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![2.0, 2.0],
    ];
    let dists = ns.knn_distances(&[0.5, 0.5], &all_bds);
    assert_eq!(dists.len(), 4);
    for i in 1..dists.len() {
        assert!(
            dists[i] >= dists[i - 1],
            "knn_distances should be sorted ascending"
        );
    }
}

#[test]
fn test_novelty_evolve_step_archive_grows() {
    let mut ns = EcNoveltySearch::new(8, 2, 2, 3);
    ns.novelty_threshold = 0.0; // always add to archive
    ns.archive_prob = 1.0;
    let mut seed = 42u64;
    let fitness_fn = |x: &[f64]| -> (f64, Vec<f64>) {
        let fit = -(x[0].powi(2) + x[1].powi(2));
        let bd = vec![x[0].abs().min(1.0), x[1].abs().min(1.0)];
        (fit, bd)
    };
    ns.evolve_step(&fitness_fn, &mut seed);
    assert!(
        ns.archive_size() > 0,
        "archive should grow after evolve_step"
    );
}

#[test]
fn test_novelty_evolve_step_population_size_preserved() {
    let mut ns = EcNoveltySearch::new(10, 2, 2, 3);
    let mut seed = 123u64;
    let fitness_fn = |x: &[f64]| -> (f64, Vec<f64>) {
        (x[0] + x[1], vec![x[0].abs().min(1.0), x[1].abs().min(1.0)])
    };
    ns.evolve_step(&fitness_fn, &mut seed);
    assert_eq!(ns.population.len(), 10);
}

// ── §6 EcDifferentialEvolution ────────────────────────────────────────────────

#[test]
fn test_de_new_valid() {
    let bounds = vec![(-5.0, 5.0); 4];
    let de = EcDifferentialEvolution::new(4, 20, 0.9, 0.8, bounds);
    assert!(de.is_ok());
}

#[test]
fn test_de_new_wrong_bounds_size() {
    let bounds = vec![(-5.0, 5.0); 3]; // wrong: dim=4
    let de = EcDifferentialEvolution::new(4, 20, 0.9, 0.8, bounds);
    assert!(de.is_err());
}

#[test]
fn test_de_new_too_small_pop() {
    let bounds = vec![(-5.0, 5.0); 2];
    let de = EcDifferentialEvolution::new(2, 3, 0.9, 0.8, bounds); // pop_size=3 < 4
    assert!(de.is_err());
}

#[test]
fn test_de_initialize_fills_population() {
    let bounds = vec![(-5.0, 5.0); 3];
    let mut de = EcDifferentialEvolution::new(3, 10, 0.9, 0.8, bounds).expect("DE creation should succeed");
    let mut seed = 42u64;
    de.initialize(&mut seed);
    assert_eq!(de.population.len(), 10);
    for ind in &de.population {
        assert_eq!(ind.len(), 3);
    }
}

#[test]
fn test_de_initialize_within_bounds() {
    let bounds = vec![(0.0, 1.0); 3];
    let mut de = EcDifferentialEvolution::new(3, 20, 0.9, 0.8, bounds).expect("DE creation should succeed");
    let mut seed = 77u64;
    de.initialize(&mut seed);
    for ind in &de.population {
        for &v in ind {
            assert!((0.0..=1.0).contains(&v), "value {} out of [0,1]", v);
        }
    }
}

#[test]
fn test_de_mutate_vector() {
    let bounds = vec![(-1.0, 1.0); 3];
    let mut de = EcDifferentialEvolution::new(3, 10, 0.9, 0.8, bounds).expect("DE creation should succeed");
    let mut seed = 55u64;
    de.initialize(&mut seed);
    let donor = de.mutate_vector(0, &mut seed);
    assert_eq!(donor.len(), 3);
    for &v in &donor {
        assert!(v.is_finite());
    }
}

#[test]
fn test_de_crossover_returns_correct_dim() {
    let bounds = vec![(-1.0, 1.0); 5];
    let mut de = EcDifferentialEvolution::new(5, 10, 0.5, 0.8, bounds).expect("DE creation should succeed");
    let mut seed = 99u64;
    de.initialize(&mut seed);
    let target = vec![0.0; 5];
    let donor = vec![1.0; 5];
    let trial = de.crossover(&target, &donor, &mut seed);
    assert_eq!(trial.len(), 5);
}

#[test]
fn test_de_evolve_step_works() {
    let bounds = vec![(-5.0, 5.0); 3];
    let mut de = EcDifferentialEvolution::new(3, 10, 0.9, 0.8, bounds).expect("DE creation should succeed");
    let mut seed = 42u64;
    de.initialize(&mut seed);
    assert_eq!(de.population.len(), 10);
    // All initialized values should be finite and within bounds
    for ind in &de.population {
        assert_eq!(ind.len(), 3);
        for &v in ind {
            assert!(
                v.is_finite() && (-5.0..=5.0).contains(&v),
                "init value {} out of range",
                v
            );
        }
    }
    let fitness_fn = |x: &[f64]| -> f64 { -x.iter().map(|v| v.powi(2)).sum::<f64>() };
    let mut fitnesses = Vec::new();
    de.evolve_step(&fitness_fn, &mut seed, &mut fitnesses);
    assert_eq!(de.generation, 1);
    assert_eq!(fitnesses.len(), 10);
    assert!(
        fitnesses.iter().all(|f| f.is_finite()),
        "all fitnesses should be finite"
    );
    // best_fitness should have been set to a finite value (negative since -sum_of_squares)
    assert!(
        de.best_fitness.is_finite(),
        "best_fitness should be finite, got {}",
        de.best_fitness
    );
    assert!(
        de.best_fitness <= 0.0,
        "best fitness from -sum_of_squares should be <= 0"
    );
}

#[test]
fn test_de_clip_to_bounds() {
    let bounds = vec![(0.0, 1.0); 3];
    let de = EcDifferentialEvolution::new(3, 10, 0.9, 0.8, bounds).expect("DE creation should succeed");
    let x = vec![-0.5, 0.5, 1.5];
    let clipped = de.clip_to_bounds(&x);
    assert_eq!(clipped, vec![0.0, 0.5, 1.0]);
}

// ── §7 EcParticleSwarm ────────────────────────────────────────────────────────

#[test]
fn test_pso_new() {
    let bounds = vec![(-1.0, 1.0); 3];
    let pso = EcParticleSwarm::new(10, 3, bounds);
    assert_eq!(pso.particles.len(), 10);
}

#[test]
fn test_pso_initialize_within_bounds() {
    let bounds = vec![(0.0, 2.0); 4];
    let mut pso = EcParticleSwarm::new(15, 4, bounds.clone());
    let mut seed = 42u64;
    pso.initialize(&mut seed);
    for particle in &pso.particles {
        for (j, &pos) in particle.position.iter().enumerate() {
            let (lo, hi) = bounds[j];
            assert!(
                pos >= lo && pos <= hi,
                "particle pos {} out of [{},{}]",
                pos,
                lo,
                hi
            );
        }
    }
}

#[test]
fn test_pso_step_updates_positions() {
    let bounds = vec![(-5.0, 5.0); 2];
    let mut pso = EcParticleSwarm::new(10, 2, bounds);
    let mut seed = 42u64;
    pso.initialize(&mut seed);
    let positions_before: Vec<Vec<f64>> =
        pso.particles.iter().map(|p| p.position.clone()).collect();
    let fitness_fn = |x: &[f64]| -> f64 { -(x[0].powi(2) + x[1].powi(2)) };
    pso.step(&fitness_fn, &mut seed);
    let any_changed = pso
        .particles
        .iter()
        .zip(&positions_before)
        .any(|(p, prev)| p.position != *prev);
    assert!(any_changed, "step should update at least some positions");
}

#[test]
fn test_pso_diversity_nonneg() {
    let bounds = vec![(-1.0, 1.0); 3];
    let mut pso = EcParticleSwarm::new(5, 3, bounds);
    let mut seed = 42u64;
    pso.initialize(&mut seed);
    assert!(pso.swarm_diversity() >= 0.0);
}

#[test]
fn test_pso_get_best_after_step() {
    let bounds = vec![(-5.0, 5.0); 2];
    let mut pso = EcParticleSwarm::new(10, 2, bounds);
    let mut seed = 42u64;
    pso.initialize(&mut seed);
    let fitness_fn = |x: &[f64]| -> f64 { -(x[0].powi(2) + x[1].powi(2)) };
    pso.step(&fitness_fn, &mut seed);
    let (best_pos, best_fit) = pso.get_best();
    assert_eq!(best_pos.len(), 2);
    assert!(best_fit > f64::NEG_INFINITY);
}

#[test]
fn test_pso_step_clamps_positions() {
    let bounds = vec![(0.0, 1.0); 2];
    let mut pso = EcParticleSwarm::new(5, 2, bounds.clone());
    let mut seed = 42u64;
    pso.initialize(&mut seed);
    let fitness_fn = |x: &[f64]| -> f64 { x[0] + x[1] };
    for _ in 0..5 {
        pso.step(&fitness_fn, &mut seed);
    }
    for particle in &pso.particles {
        for (j, &pos) in particle.position.iter().enumerate() {
            let (lo, hi) = bounds[j];
            assert!(pos >= lo && pos <= hi, "pos {} out of [{},{}]", pos, lo, hi);
        }
    }
}

// ── §8 EcGpTree ───────────────────────────────────────────────────────────────

#[test]
fn test_gp_tree_random_nonempty() {
    let mut seed = 42u64;
    let tree = EcGpTree::random(4, 2, &mut seed);
    assert!(!tree.nodes.is_empty());
}

#[test]
fn test_gp_tree_evaluate_finite() {
    let mut seed = 42u64;
    for _ in 0..20 {
        let tree = EcGpTree::random(4, 2, &mut seed);
        let v = tree.evaluate(&[1.0, 2.0]);
        assert!(v.is_finite(), "evaluate should return finite, got {}", v);
    }
}

#[test]
fn test_gp_tree_size() {
    let mut seed = 42u64;
    let tree = EcGpTree::random(3, 1, &mut seed);
    assert!(tree.size() >= 1);
}

#[test]
fn test_gp_tree_depth() {
    let mut seed = 100u64;
    let tree = EcGpTree::random(4, 2, &mut seed);
    assert!(tree.depth() >= 1);
}

#[test]
fn test_gp_tree_mutate_valid() {
    let mut seed = 42u64;
    let tree = EcGpTree::random(4, 2, &mut seed);
    let mutated = tree.mutate(&mut seed);
    assert!(!mutated.nodes.is_empty());
    let v = mutated.evaluate(&[0.5, -0.5]);
    assert!(v.is_finite());
}

#[test]
fn test_gp_tree_crossover_valid() {
    let mut seed = 42u64;
    let t1 = EcGpTree::random(4, 2, &mut seed);
    let t2 = EcGpTree::random(4, 2, &mut seed);
    let child = t1.crossover(&t2, &mut seed);
    assert!(!child.nodes.is_empty());
    let v = child.evaluate(&[1.0, -1.0]);
    assert!(v.is_finite());
}

#[test]
fn test_gp_tree_complexity_penalty() {
    let mut seed = 42u64;
    let tree = EcGpTree::random(4, 2, &mut seed);
    let penalty = tree.complexity_penalty();
    assert!(
        penalty >= 0.001,
        "penalty should be >= 0.001 (at least 1 node)"
    );
}

#[test]
fn test_gp_eval_const() {
    let tree = EcGpTree {
        nodes: vec![EcGpNode::Const(3.125)],
        n_vars: 0,
    };
    assert!((tree.evaluate(&[]) - 3.125).abs() < 1e-10);
}

#[test]
fn test_gp_eval_var() {
    let tree = EcGpTree {
        nodes: vec![EcGpNode::Var(1)],
        n_vars: 2,
    };
    assert!((tree.evaluate(&[0.0, 2.5]) - 2.5).abs() < 1e-10);
}

#[test]
fn test_gp_genetic_programming_evolve() {
    let mut gp = EcGeneticProgramming::new(10, 1, 3);
    let x_data: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.1]).collect();
    let y_data: Vec<f64> = x_data.iter().map(|x| x[0].powi(2)).collect();
    let mut seed = 42u64;
    gp.evolve_step(&x_data, &y_data, &mut seed);
    assert_eq!(gp.population.len(), 10);
    assert_eq!(gp.generation, 1);
}

// ── §9 EcNsga3 ────────────────────────────────────────────────────────────────

#[test]
fn test_nsga3_reference_points_sum_to_one() {
    let ref_pts = EcNsga3::generate_reference_points(3, 4);
    for pt in &ref_pts {
        let sum: f64 = pt.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "ref point sums to {}", sum);
    }
}

#[test]
fn test_nsga3_reference_points_2obj() {
    let ref_pts = EcNsga3::generate_reference_points(2, 4);
    // Should have 5 points: (0,1), (0.25,0.75), (0.5,0.5), (0.75,0.25), (1,0)
    assert_eq!(ref_pts.len(), 5);
}

#[test]
fn test_nsga3_non_dominated_sort_basic() {
    let solutions = vec![
        EcSolution {
            params: vec![],
            objectives: vec![1.0, 2.0],
            rank: 0,
            distance: 0.0,
        },
        EcSolution {
            params: vec![],
            objectives: vec![2.0, 1.0],
            rank: 0,
            distance: 0.0,
        },
        EcSolution {
            params: vec![],
            objectives: vec![3.0, 3.0],
            rank: 0,
            distance: 0.0,
        },
    ];
    let fronts = EcNsga3::non_dominated_sort(&solutions);
    assert!(!fronts.is_empty());
    // First two should be in front 0 (non-dominated), third in front 1
    let front0 = &fronts[0];
    assert!(front0.contains(&0) || front0.contains(&1));
    assert_eq!(front0.len(), 2, "first two solutions are non-dominated");
}

#[test]
fn test_nsga3_non_dominated_sort_dominated() {
    let solutions = vec![
        EcSolution {
            params: vec![],
            objectives: vec![1.0, 1.0],
            rank: 0,
            distance: 0.0,
        },
        EcSolution {
            params: vec![],
            objectives: vec![2.0, 2.0],
            rank: 0,
            distance: 0.0,
        }, // dominated by 0
    ];
    let fronts = EcNsga3::non_dominated_sort(&solutions);
    assert!(fronts[0].contains(&0));
    if fronts.len() > 1 {
        assert!(fronts[1].contains(&1));
    }
}

#[test]
fn test_nsga3_evolve_step_ok() {
    let bounds = vec![(0.0, 1.0); 3];
    let mut nsga3 = EcNsga3::new(10, 3, 2, bounds);
    let mut seed = 42u64;
    let obj_fn = |x: &[f64]| -> Vec<f64> { vec![x[0], 1.0 - x[0].sqrt()] };
    nsga3
        .evolve_step(&obj_fn, &mut seed)
        .expect("evolve_step failed");
    assert_eq!(nsga3.population.len(), 10);
    assert_eq!(nsga3.generation, 1);
}

#[test]
fn test_nsga3_hypervolume_nonneg() {
    let bounds = vec![(0.0, 1.0); 2];
    let mut nsga3 = EcNsga3::new(10, 2, 2, bounds);
    let mut seed = 42u64;
    let obj_fn = |x: &[f64]| -> Vec<f64> { vec![x[0], 1.0 - x[0]] };
    nsga3.evolve_step(&obj_fn, &mut seed).expect("evolve_step should succeed");
    // Force some solutions to rank 0
    for sol in &mut nsga3.population {
        sol.rank = 0;
    }
    let hv = nsga3.hypervolume_indicator(&[1.1, 1.1]);
    assert!(hv >= 0.0, "hypervolume should be non-negative, got {}", hv);
}

#[test]
fn test_nsga3_sbx_crossover() {
    let bounds = vec![(0.0, 1.0); 4];
    let nsga3 = EcNsga3::new(10, 4, 2, bounds);
    let p1 = vec![0.2, 0.4, 0.6, 0.8];
    let p2 = vec![0.8, 0.6, 0.4, 0.2];
    let mut seed = 42u64;
    let (c1, c2) = nsga3.sbx_crossover(&p1, &p2, &mut seed);
    assert_eq!(c1.len(), 4);
    assert_eq!(c2.len(), 4);
    for &v in c1.iter().chain(c2.iter()) {
        assert!((0.0..=1.0).contains(&v), "sbx child out of bounds: {}", v);
    }
}

#[test]
fn test_nsga3_polynomial_mutation() {
    let bounds = vec![(0.0, 1.0); 5];
    let nsga3 = EcNsga3::new(10, 5, 2, bounds);
    let x = vec![0.5; 5];
    let mut seed = 42u64;
    let mutated = nsga3.polynomial_mutation(&x, &mut seed);
    assert_eq!(mutated.len(), 5);
    for &v in &mutated {
        assert!((0.0..=1.0).contains(&v), "mutated value {} out of [0,1]", v);
    }
}

// ── §10 EcMetrics ─────────────────────────────────────────────────────────────

#[test]
fn test_metrics_diversity_nonneg() {
    let pop = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let d = EcMetrics::population_diversity(&pop);
    assert!(d >= 0.0, "diversity should be non-negative");
}

#[test]
fn test_metrics_diversity_zero_single() {
    let pop = vec![vec![1.0, 2.0]];
    let d = EcMetrics::population_diversity(&pop);
    assert_eq!(d, 0.0);
}

#[test]
fn test_metrics_diversity_identical() {
    let pop = vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]];
    let d = EcMetrics::population_diversity(&pop);
    assert!(d.abs() < 1e-10);
}

#[test]
fn test_metrics_best_fitness_history() {
    let history = vec![1.0, 3.0, 2.0, 5.0, 4.0];
    let (best, gen) = EcMetrics::best_fitness_history(&history);
    assert_eq!(best, 5.0);
    assert_eq!(gen, 3);
}

#[test]
fn test_metrics_convergence_rate_finite() {
    let history: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let rate = EcMetrics::convergence_rate(&history);
    assert!(rate.is_finite());
    assert!(
        rate > 0.0,
        "monotonically increasing should have positive rate"
    );
}

#[test]
fn test_metrics_convergence_rate_empty() {
    let rate = EcMetrics::convergence_rate(&[]);
    assert_eq!(rate, 0.0);
}

#[test]
fn test_metrics_species_count() {
    let g = EcGenome::new_minimal(2, 1);
    let species = vec![
        EcSpecies {
            representative: g.clone(),
            members: vec![0, 1],
            best_fitness: 1.0,
            staleness: 0,
        },
        EcSpecies {
            representative: g.clone(),
            members: vec![2],
            best_fitness: 0.5,
            staleness: 0,
        },
    ];
    assert_eq!(EcMetrics::species_count(&species), 2);
}

#[test]
fn test_metrics_neat_complexity() {
    let pop = vec![EcGenome::new_minimal(2, 2), EcGenome::new_minimal(3, 1)];
    let (mean_nodes, mean_conns) = EcMetrics::neat_complexity(&pop);
    // Genome(2,2): 4 nodes, 4 conns; Genome(3,1): 4 nodes, 3 conns
    assert!(
        (mean_nodes - 4.0).abs() < 1e-10,
        "mean nodes = {}",
        mean_nodes
    );
    assert!(
        (mean_conns - 3.5).abs() < 1e-10,
        "mean conns = {}",
        mean_conns
    );
}

#[test]
fn test_metrics_coverage_metric() {
    let pop = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
    let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    let cov = EcMetrics::coverage_metric(&pop, &bounds);
    assert!(
        (0.0..=1.0).contains(&cov),
        "coverage should be in [0,1], got {}",
        cov
    );
}

#[test]
fn test_ec_error_display() {
    let e1 = EcError::InvalidConfig("bad".to_string());
    let e2 = EcError::NumericalError("nan".to_string());
    let e3 = EcError::EmptyPopulation;
    assert!(format!("{}", e1).contains("InvalidConfig"));
    assert!(format!("{}", e2).contains("NumericalError"));
    assert!(format!("{}", e3).contains("EmptyPopulation"));
}

#[test]
fn test_de_multiple_evolve_steps() {
    let bounds = vec![(-5.0, 5.0); 2];
    let mut de = EcDifferentialEvolution::new(2, 8, 0.9, 0.8, bounds).expect("DE creation should succeed");
    let mut seed = 42u64;
    de.initialize(&mut seed);
    let fitness_fn = |x: &[f64]| -> f64 { -(x[0].powi(2) + x[1].powi(2)) };
    let mut fitnesses = Vec::new();
    for _ in 0..5 {
        de.evolve_step(&fitness_fn, &mut seed, &mut fitnesses);
    }
    assert_eq!(de.generation, 5);
    assert_eq!(de.fitness_history.len(), 5);
}

#[test]
fn test_neat_global_best_updates() {
    let cfg = EcNeatConfig {
        population_size: 10,
        n_inputs: 2,
        n_outputs: 1,
        ..EcNeatConfig::default()
    };
    let mut neat = EcNeat::new(cfg);
    let fitnesses: Vec<f64> = (0..10).map(|i| (i as f64) * 2.0).collect();
    let mut seed = 42u64;
    neat.evolve_generation(&fitnesses, &mut seed).expect("evolve_generation should succeed");
    assert_eq!(neat.global_best, 18.0, "global_best should be max fitness");
}

#[test]
fn test_es_multiple_steps() {
    let mut es = EcEvolutionStrategies::new(3, 0.05, 0.01, 8);
    let mut seed = 42u64;
    let fitness_fn = |x: &[f64]| -> f64 { -(x.iter().map(|v| v.powi(2)).sum::<f64>()) };
    for _ in 0..5 {
        es.step(&fitness_fn, &mut seed);
    }
    assert_eq!(es.generation, 5);
    assert_eq!(es.fitness_history.len(), 5);
}

#[test]
fn test_gp_best_tree_valid() {
    let mut gp = EcGeneticProgramming::new(5, 1, 3);
    let x_data: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
    let y_data: Vec<f64> = x_data.iter().map(|x| x[0]).collect();
    let mut seed = 42u64;
    gp.evolve_step(&x_data, &y_data, &mut seed);
    let best = gp.best_tree();
    let v = best.evaluate(&[1.0]);
    assert!(v.is_finite());
}
