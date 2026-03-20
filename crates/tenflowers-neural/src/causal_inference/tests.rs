use super::*;

// ── CausalGraph tests ────────────────────────────────────────────────────

#[test]
fn test_graph_empty() {
    let g = CausalGraph::new();
    assert!(g.variables().is_empty());
    assert!(g.is_dag());
}

#[test]
fn test_graph_add_variable() {
    let mut g = CausalGraph::new();
    g.add_variable("X")
        .expect("adding variable X should succeed");
    g.add_variable("Y")
        .expect("adding variable Y should succeed");
    assert_eq!(g.variables().len(), 2);
}

#[test]
fn test_graph_duplicate_variable_error() {
    let mut g = CausalGraph::new();
    g.add_variable("X")
        .expect("adding variable X should succeed");
    assert!(g.add_variable("X").is_err());
}

#[test]
fn test_graph_add_edge() {
    let mut g = CausalGraph::new();
    g.add_variable("X")
        .expect("adding variable X should succeed");
    g.add_variable("Y")
        .expect("adding variable Y should succeed");
    g.add_edge("X", "Y")
        .expect("adding edge X->Y should succeed");
    assert_eq!(g.edges().len(), 1);
}

#[test]
fn test_graph_cycle_detection() {
    let mut g = CausalGraph::new();
    g.add_variable("X")
        .expect("adding variable X should succeed");
    g.add_variable("Y")
        .expect("adding variable Y should succeed");
    g.add_edge("X", "Y")
        .expect("adding edge X->Y should succeed");
    let result = g.add_edge("Y", "X");
    assert!(result.is_err(), "Should detect cycle");
}

#[test]
fn test_graph_self_loop_error() {
    let mut g = CausalGraph::new();
    g.add_variable("X")
        .expect("adding variable X should succeed");
    assert!(g.add_edge("X", "X").is_err());
}

#[test]
fn test_topological_sort_chain() {
    let mut g = CausalGraph::new();
    for v in ["A", "B", "C", "D"] {
        g.add_variable(v).expect("adding variable should succeed");
    }
    g.add_edge("A", "B")
        .expect("adding edge A->B should succeed");
    g.add_edge("B", "C")
        .expect("adding edge B->C should succeed");
    g.add_edge("C", "D")
        .expect("adding edge C->D should succeed");
    let order = g
        .topological_sort()
        .expect("topological sort should succeed");
    let pos: HashMap<&str, usize> = order
        .iter()
        .enumerate()
        .map(|(i, v)| (v.as_str(), i))
        .collect();
    assert!(pos["A"] < pos["B"]);
    assert!(pos["B"] < pos["C"]);
    assert!(pos["C"] < pos["D"]);
}

#[test]
fn test_topological_sort_fork() {
    let mut g = CausalGraph::new();
    g.add_variable("X")
        .expect("adding variable X should succeed");
    g.add_variable("Y")
        .expect("adding variable Y should succeed");
    g.add_variable("Z")
        .expect("adding variable Z should succeed");
    g.add_edge("X", "Y")
        .expect("adding edge X->Y should succeed");
    g.add_edge("X", "Z")
        .expect("adding edge X->Z should succeed");
    let order = g
        .topological_sort()
        .expect("topological sort should succeed");
    assert_eq!(order[0], "X");
}

#[test]
fn test_is_dag_true() {
    let mut g = CausalGraph::new();
    g.add_variable("X")
        .expect("adding variable X should succeed");
    g.add_variable("Y")
        .expect("adding variable Y should succeed");
    g.add_edge("X", "Y")
        .expect("adding edge X->Y should succeed");
    assert!(g.is_dag());
}

#[test]
fn test_ancestors() {
    let mut g = CausalGraph::new();
    for v in ["A", "B", "C"] {
        g.add_variable(v).expect("adding variable should succeed");
    }
    g.add_edge("A", "B")
        .expect("adding edge A->B should succeed");
    g.add_edge("B", "C")
        .expect("adding edge B->C should succeed");
    let anc = g.ancestors("C");
    assert!(anc.contains("A"));
    assert!(anc.contains("B"));
    assert!(!anc.contains("C"));
}

#[test]
fn test_descendants() {
    let mut g = CausalGraph::new();
    for v in ["A", "B", "C"] {
        g.add_variable(v).expect("adding variable should succeed");
    }
    g.add_edge("A", "B")
        .expect("adding edge A->B should succeed");
    g.add_edge("B", "C")
        .expect("adding edge B->C should succeed");
    let desc = g.descendants("A");
    assert!(desc.contains("B"));
    assert!(desc.contains("C"));
    assert!(!desc.contains("A"));
}

#[test]
fn test_markov_blanket_simple() {
    let mut g = CausalGraph::new();
    for v in ["X", "Y", "Z", "W"] {
        g.add_variable(v).expect("adding variable should succeed");
    }
    // X → Y, Z → Y, Y → W
    g.add_edge("X", "Y")
        .expect("adding edge X->Y should succeed");
    g.add_edge("Z", "Y")
        .expect("adding edge Z->Y should succeed");
    g.add_edge("Y", "W")
        .expect("adding edge Y->W should succeed");
    let mb = g.markov_blanket("Y");
    // Parents of Y: X, Z. Children of Y: W. Co-parents of W (via Y): none.
    assert!(mb.contains("X"));
    assert!(mb.contains("Z"));
    assert!(mb.contains("W"));
    assert!(!mb.contains("Y"));
}

#[test]
fn test_d_separation_chain_blocked() {
    // Chain X → M → Y; conditioning on M blocks the path.
    let mut g = CausalGraph::new();
    for v in ["X", "M", "Y"] {
        g.add_variable(v).expect("adding variable should succeed");
    }
    g.add_edge("X", "M")
        .expect("adding edge X->M should succeed");
    g.add_edge("M", "Y")
        .expect("adding edge M->Y should succeed");
    let z: HashSet<String> = vec!["M".to_string()].into_iter().collect();
    assert!(g.d_separation("X", "Y", &z));
}

#[test]
fn test_d_separation_chain_open() {
    // Chain X → M → Y; without conditioning, path is open.
    let mut g = CausalGraph::new();
    for v in ["X", "M", "Y"] {
        g.add_variable(v).expect("adding variable should succeed");
    }
    g.add_edge("X", "M")
        .expect("adding edge X->M should succeed");
    g.add_edge("M", "Y")
        .expect("adding edge M->Y should succeed");
    let z: HashSet<String> = HashSet::new();
    assert!(!g.d_separation("X", "Y", &z));
}

#[test]
fn test_d_separation_fork_blocked() {
    // Fork X ← Z → Y; conditioning on Z blocks path.
    let mut g = CausalGraph::new();
    for v in ["X", "Z", "Y"] {
        g.add_variable(v).expect("adding variable should succeed");
    }
    g.add_edge("Z", "X")
        .expect("adding edge Z->X should succeed");
    g.add_edge("Z", "Y")
        .expect("adding edge Z->Y should succeed");
    let z_set: HashSet<String> = vec!["Z".to_string()].into_iter().collect();
    assert!(g.d_separation("X", "Y", &z_set));
}

#[test]
fn test_d_separation_fork_open() {
    // Fork X ← Z → Y; without conditioning, path is open.
    let mut g = CausalGraph::new();
    for v in ["X", "Z", "Y"] {
        g.add_variable(v).expect("adding variable should succeed");
    }
    g.add_edge("Z", "X")
        .expect("adding edge Z->X should succeed");
    g.add_edge("Z", "Y")
        .expect("adding edge Z->Y should succeed");
    let z_set: HashSet<String> = HashSet::new();
    assert!(!g.d_separation("X", "Y", &z_set));
}

#[test]
fn test_d_separation_collider_closed() {
    // Collider X → C ← Y; without conditioning on C, path is blocked.
    let mut g = CausalGraph::new();
    for v in ["X", "C", "Y"] {
        g.add_variable(v).expect("adding variable should succeed");
    }
    g.add_edge("X", "C")
        .expect("adding edge X->C should succeed");
    g.add_edge("Y", "C")
        .expect("adding edge Y->C should succeed");
    let z_set: HashSet<String> = HashSet::new();
    assert!(g.d_separation("X", "Y", &z_set));
}

#[test]
fn test_single_variable_graph() {
    let mut g = CausalGraph::new();
    g.add_variable("X")
        .expect("adding variable X should succeed");
    assert!(g.is_dag());
    let order = g
        .topological_sort()
        .expect("topological sort should succeed");
    assert_eq!(order, vec!["X"]);
}

#[test]
fn test_disconnected_components() {
    let mut g = CausalGraph::new();
    for v in ["A", "B", "C", "D"] {
        g.add_variable(v).expect("adding variable should succeed");
    }
    g.add_edge("A", "B")
        .expect("adding edge A->B should succeed");
    g.add_edge("C", "D")
        .expect("adding edge C->D should succeed");
    let order = g
        .topological_sort()
        .expect("topological sort should succeed");
    assert_eq!(order.len(), 4);
    assert!(g.is_dag());
}

// ── CausalEstimator tests ────────────────────────────────────────────────

fn make_simple_data(n: usize, seed: u64) -> Vec<HashMap<String, f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|i| {
            let z: f64 = rng.random::<f64>() * 2.0 - 1.0;
            let x: f64 = if i % 2 == 0 { 1.0 } else { 0.0 };
            let y: f64 = 2.0 * x + 0.5 * z + 0.1 * (rng.random::<f64>() - 0.5);
            let mut row = HashMap::new();
            row.insert("X".to_string(), x);
            row.insert("Y".to_string(), y);
            row.insert("Z".to_string(), z);
            row
        })
        .collect()
}

#[test]
fn test_ate_estimation_direction() {
    let mut g = CausalGraph::new();
    g.add_variable("Z")
        .expect("adding variable Z should succeed");
    g.add_variable("X")
        .expect("adding variable X should succeed");
    g.add_variable("Y")
        .expect("adding variable Y should succeed");
    g.add_edge("Z", "X")
        .expect("adding edge Z->X should succeed");
    g.add_edge("X", "Y")
        .expect("adding edge X->Y should succeed");
    g.add_edge("Z", "Y")
        .expect("adding edge Z->Y should succeed");

    let data = make_simple_data(400, 0);
    let est = CausalEstimator::new(g, data, 0);
    let ate = est
        .average_treatment_effect("X", "Y", &["Z".to_string()])
        .expect("ATE estimation should succeed");
    // True ATE ≈ 2.0; should be positive and > 1.
    assert!(ate > 0.5, "ATE should be clearly positive, got {}", ate);
}

#[test]
fn test_backdoor_empty_adjustment_set() {
    let mut g = CausalGraph::new();
    g.add_variable("X")
        .expect("adding variable X should succeed");
    g.add_variable("Y")
        .expect("adding variable Y should succeed");
    g.add_edge("X", "Y")
        .expect("adding edge X->Y should succeed");
    let data = make_simple_data(200, 1);
    let est = CausalEstimator::new(g, data, 1);
    let query = CausalQuery {
        target: "Y".to_string(),
        intervention: Some(Intervention {
            variable: "X".to_string(),
            value: 1.0,
        }),
        conditioning: HashMap::new(),
    };
    let result = est.backdoor_adjustment(&query, &[]);
    assert!(result.is_ok());
}

// ── DoubleML tests ───────────────────────────────────────────────────────

fn make_double_ml_data(
    n: usize,
    true_theta: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut treatment = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    let mut controls = Vec::with_capacity(n);
    for _ in 0..n {
        let x1: f64 = rng.random::<f64>() * 2.0 - 1.0;
        let x2: f64 = rng.random::<f64>() * 2.0 - 1.0;
        // Treatment: linear function of X + noise.
        let t = 0.5 * x1 - 0.3 * x2 + 0.3 * (rng.random::<f64>() * 2.0 - 1.0);
        // Outcome: theta * T + g(X) + noise.
        let y = true_theta * t + 0.8 * x1 + 0.2 * x2 + 0.1 * (rng.random::<f64>() * 2.0 - 1.0);
        treatment.push(t);
        outcome.push(y);
        controls.push(vec![x1, x2]);
    }
    (treatment, outcome, controls)
}

#[test]
fn test_double_ml_recovers_theta() {
    let true_theta = 2.0;
    let (t, y, x) = make_double_ml_data(500, true_theta, 42);
    let config = DoubleMLConfig {
        n_folds: 5,
        reg_lambda: 1e-3,
        seed: 42,
    };
    let dml = DoubleML::new(config);
    let result = dml.fit(&t, &y, &x).expect("DoubleML fit should succeed");
    assert!(
        (result.theta - true_theta).abs() < 0.5,
        "theta={} far from true {}",
        result.theta,
        true_theta
    );
}

#[test]
fn test_double_ml_confidence_interval_contains_truth() {
    let true_theta = 1.5;
    let (t, y, x) = make_double_ml_data(600, true_theta, 7);
    let config = DoubleMLConfig::default();
    let dml = DoubleML::new(config);
    let result = dml.fit(&t, &y, &x).expect("DoubleML fit should succeed");
    let (lo, hi) = result.confidence_interval;
    // CI should contain truth for this sample size.
    assert!(
        lo <= true_theta + 1.0 && hi >= true_theta - 1.0,
        "CI ({:.3}, {:.3}) misses true theta {:.3}",
        lo,
        hi,
        true_theta
    );
}

#[test]
fn test_double_ml_std_error_positive() {
    let (t, y, x) = make_double_ml_data(200, 1.0, 99);
    let dml = DoubleML::new(DoubleMLConfig::default());
    let result = dml.fit(&t, &y, &x).expect("DoubleML fit should succeed");
    assert!(result.std_error > 0.0);
}

#[test]
fn test_double_ml_p_value_range() {
    let (t, y, x) = make_double_ml_data(300, 3.0, 17);
    let dml = DoubleML::new(DoubleMLConfig::default());
    let result = dml.fit(&t, &y, &x).expect("DoubleML fit should succeed");
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
}

#[test]
fn test_double_ml_error_on_empty_data() {
    let dml = DoubleML::new(DoubleMLConfig::default());
    let result = dml.fit(&[], &[], &[]);
    assert!(result.is_err());
}

#[test]
fn test_nuisance_model_fit_predict() {
    let x = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let y = vec![2.0, 1.0, 3.0];
    let mut m = NuisanceModel::new();
    m.fit(&x, &y).expect("nuisance model fit should succeed");
    let pred = m
        .predict(&x)
        .expect("nuisance model predict should succeed");
    assert_eq!(pred.len(), 3);
    // Predictions should be in a reasonable range.
    for p in &pred {
        assert!(p.is_finite());
    }
}

#[test]
fn test_nuisance_model_empty_error() {
    let mut m = NuisanceModel::new();
    assert!(m.fit(&[], &[]).is_err());
}

// ── Counterfactual tests ─────────────────────────────────────────────────

#[test]
fn test_counterfactual_simple_additive() {
    // Graph: X → Y with Y = X + noise. Factual: X=0, Y=0.5. CF: X=1.
    let mut g = CausalGraph::new();
    g.add_variable("X")
        .expect("adding variable X should succeed");
    g.add_variable("Y")
        .expect("adding variable Y should succeed");
    g.add_edge("X", "Y")
        .expect("adding edge X->Y should succeed");

    let est = CounterfactualEstimator::new(g, 0);
    let mut observed = HashMap::new();
    observed.insert("X".to_string(), 0.0);
    observed.insert("Y".to_string(), 0.5);

    let query = CounterfactualQuery {
        observed,
        intervention: Intervention {
            variable: "X".to_string(),
            value: 1.0,
        },
        target: "Y".to_string(),
    };

    let mut noise = HashMap::new();
    noise.insert("Y".to_string(), 0.5); // Inferred noise

    let result = est
        .estimate(&query, &noise)
        .expect("counterfactual estimation should succeed");
    assert_eq!(result.factual, 0.5);
    // CF: f(Pa={X=1}) + U_Y = 1.0 + 0.5 = 1.5.
    assert!((result.counterfactual - 1.5).abs() < 1e-10);
    assert!((result.individual_treatment_effect - 1.0).abs() < 1e-10);
}

#[test]
fn test_counterfactual_missing_factual_error() {
    let mut g = CausalGraph::new();
    g.add_variable("X")
        .expect("adding variable X should succeed");
    g.add_variable("Y")
        .expect("adding variable Y should succeed");
    g.add_edge("X", "Y")
        .expect("adding edge X->Y should succeed");

    let est = CounterfactualEstimator::new(g, 0);
    let query = CounterfactualQuery {
        observed: HashMap::new(), // Y not observed
        intervention: Intervention {
            variable: "X".to_string(),
            value: 1.0,
        },
        target: "Y".to_string(),
    };
    assert!(est.estimate(&query, &HashMap::new()).is_err());
}

#[test]
fn test_counterfactual_ite_sign() {
    let mut g = CausalGraph::new();
    g.add_variable("X")
        .expect("adding variable X should succeed");
    g.add_variable("Y")
        .expect("adding variable Y should succeed");
    g.add_edge("X", "Y")
        .expect("adding edge X->Y should succeed");

    let est = CounterfactualEstimator::new(g, 0);
    let ite = est.individual_treatment_effect(2.0, 5.0);
    assert!((ite - 3.0).abs() < 1e-10);
}

#[test]
fn test_counterfactual_root_variable_intervention() {
    // When the intervention is on a root variable, downstream should propagate.
    let mut g = CausalGraph::new();
    g.add_variable("X")
        .expect("adding variable X should succeed");
    g.add_variable("Y")
        .expect("adding variable Y should succeed");
    g.add_variable("Z")
        .expect("adding variable Z should succeed");
    g.add_edge("X", "Y")
        .expect("adding edge X->Y should succeed");
    g.add_edge("Y", "Z")
        .expect("adding edge Y->Z should succeed");

    let est = CounterfactualEstimator::new(g, 0);
    let mut observed = HashMap::new();
    observed.insert("X".to_string(), 0.0);
    observed.insert("Y".to_string(), 0.0);
    observed.insert("Z".to_string(), 0.0);

    let query = CounterfactualQuery {
        observed,
        intervention: Intervention {
            variable: "X".to_string(),
            value: 2.0,
        },
        target: "Z".to_string(),
    };
    let result = est
        .estimate(&query, &HashMap::new())
        .expect("counterfactual estimation should succeed");
    // Z = f(Y) = Y = f(X=2) = 2.0.
    assert!((result.counterfactual - 2.0).abs() < 1e-10);
}

// ── PropensityModel / IPW tests ──────────────────────────────────────────

#[test]
fn test_propensity_model_fit_predict() {
    let mut rng = StdRng::seed_from_u64(0);
    let x = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![0.0, 0.0],
    ];
    let treatment = vec![true, false, true, false];
    let config = PropensityScoreConfig::default();
    let mut model = PropensityModel::new();
    model
        .fit(&x, &treatment, &config, &mut rng)
        .expect("propensity model fit should succeed");
    let proba = model
        .predict_proba(&x)
        .expect("propensity model predict should succeed");
    assert_eq!(proba.len(), 4);
    for p in &proba {
        assert!(*p > 0.0 && *p < 1.0, "Probability out of (0,1): {}", p);
    }
}

#[test]
fn test_ipw_ate_randomised_experiment() {
    // In a randomised experiment, IPW ATE should recover the true effect.
    let mut rng = StdRng::seed_from_u64(42);
    let n = 500;
    let mut x: Vec<Vec<f64>> = Vec::new();
    let mut treatment: Vec<bool> = Vec::new();
    let mut outcome: Vec<f64> = Vec::new();
    let true_effect = 2.0;
    for _ in 0..n {
        let xi = rng.random::<f64>();
        let t = rng.random::<f64>() > 0.5;
        let y = if t { true_effect } else { 0.0 } + 0.1 * xi + 0.05 * (rng.random::<f64>() - 0.5);
        x.push(vec![xi]);
        treatment.push(t);
        outcome.push(y);
    }
    let config = PropensityScoreConfig {
        n_estimators: 50,
        reg_lambda: 1e-2,
        seed: 42,
    };
    let est = IpwEstimator::new(config);
    let ate = est
        .estimate_ate(&x, &treatment, &outcome)
        .expect("IPW ATE estimation should succeed");
    assert!(
        (ate - true_effect).abs() < 0.5,
        "IPW ATE = {:.3}, expected ≈ {:.1}",
        ate,
        true_effect
    );
}

#[test]
fn test_ipw_att_positive() {
    let mut rng = StdRng::seed_from_u64(7);
    let n = 200;
    let mut x: Vec<Vec<f64>> = Vec::new();
    let mut treatment: Vec<bool> = Vec::new();
    let mut outcome: Vec<f64> = Vec::new();
    for _ in 0..n {
        let xi = rng.random::<f64>();
        let t = rng.random::<f64>() > 0.5;
        let y = if t { 3.0 } else { 0.0 } + 0.1 * (rng.random::<f64>() - 0.5);
        x.push(vec![xi]);
        treatment.push(t);
        outcome.push(y);
    }
    let config = PropensityScoreConfig::default();
    let est = IpwEstimator::new(config);
    let att = est
        .estimate_att(&x, &treatment, &outcome)
        .expect("IPW ATT estimation should succeed");
    assert!(att > 1.0, "ATT should be positive, got {}", att);
}

#[test]
fn test_ipw_empty_treatment_error() {
    let config = PropensityScoreConfig::default();
    let est = IpwEstimator::new(config);
    let result = est.estimate_ate(&[], &[], &[]);
    assert!(result.is_err());
}

// ── RDD tests ────────────────────────────────────────────────────────────

fn make_rdd_data(n: usize, jump: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut running = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    for _ in 0..n {
        let x: f64 = rng.random::<f64>() * 4.0 - 2.0; // U[-2, 2]
        let treatment = if x >= 0.0 { 1.0 } else { 0.0 };
        let y = 1.0 * x + jump * treatment + 0.1 * (rng.random::<f64>() * 2.0 - 1.0);
        running.push(x);
        outcome.push(y);
    }
    (running, outcome)
}

#[test]
fn test_rdd_sharp_discontinuity() {
    let jump = 2.0;
    let (running, outcome) = make_rdd_data(1000, jump, 0);
    let config = RddConfig {
        bandwidth: 0.5,
        cutoff: 0.0,
        polynomial_order: 1,
    };
    let est = RddEstimator::new(config);
    let result = est
        .estimate(&running, &outcome)
        .expect("RDD estimation should succeed");
    assert!(
        (result.local_ate - jump).abs() < 1.0,
        "RDD estimate = {:.3}, expected ≈ {:.1}",
        result.local_ate,
        jump
    );
}

#[test]
fn test_rdd_result_n_used() {
    let (running, outcome) = make_rdd_data(500, 1.0, 1);
    let config = RddConfig {
        bandwidth: 0.3,
        cutoff: 0.0,
        polynomial_order: 1,
    };
    let est = RddEstimator::new(config);
    let result = est
        .estimate(&running, &outcome)
        .expect("RDD estimation should succeed");
    assert!(result.n_used > 0);
    assert!(result.n_used <= 500);
}

#[test]
fn test_rdd_confidence_interval_width() {
    let (running, outcome) = make_rdd_data(500, 1.5, 5);
    let config = RddConfig::default();
    let est = RddEstimator::new(config);
    let result = est
        .estimate(&running, &outcome)
        .expect("RDD estimation should succeed");
    let width = result.confidence_interval.1 - result.confidence_interval.0;
    assert!(width > 0.0, "CI width should be positive");
}

#[test]
fn test_rdd_error_too_few_obs() {
    let running = vec![0.1, -0.1];
    let outcome = vec![1.0, 0.0];
    let config = RddConfig::default();
    let est = RddEstimator::new(config);
    assert!(est.estimate(&running, &outcome).is_err());
}

#[test]
fn test_rdd_error_length_mismatch() {
    let running = vec![0.1, -0.1, 0.5, -0.5];
    let outcome = vec![1.0, 0.0]; // length mismatch
    let config = RddConfig::default();
    let est = RddEstimator::new(config);
    assert!(est.estimate(&running, &outcome).is_err());
}

#[test]
fn test_rdd_quadratic_order() {
    let (running, outcome) = make_rdd_data(800, 2.0, 3);
    let config = RddConfig {
        bandwidth: 0.8,
        cutoff: 0.0,
        polynomial_order: 2,
    };
    let est = RddEstimator::new(config);
    let result = est
        .estimate(&running, &outcome)
        .expect("RDD quadratic estimation should succeed");
    assert!(result.local_ate.is_finite());
}

// ── Mathematical helper tests ────────────────────────────────────────────

#[test]
fn test_sigmoid_bounds() {
    assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
    assert!(sigmoid(100.0) > 0.99);
    assert!(sigmoid(-100.0) < 0.01);
}

#[test]
fn test_normal_cdf() {
    // Φ(0) ≈ 0.5
    assert!((normal_cdf(0.0) - 0.5).abs() < 1e-2);
    // Φ(1.96) ≈ 0.975
    assert!((normal_cdf(1.96) - 0.975).abs() < 0.01);
}

#[test]
fn test_gaussian_kernel_positive() {
    let k = gaussian_kernel(0.0, 0.0, 1.0);
    assert!(k > 0.0);
}

#[test]
fn test_triangular_kernel_at_cutoff() {
    let k = triangular_kernel(0.0, 0.0, 1.0);
    assert!((k - 1.0).abs() < 1e-10);
}

#[test]
fn test_triangular_kernel_at_boundary() {
    let k = triangular_kernel(1.0, 0.0, 1.0);
    assert!((k).abs() < 1e-10);
}

#[test]
fn test_cholesky_solve_identity() {
    // (I + 0) * x = b → x = b.
    let a = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
    let b = vec![3.0, 5.0];
    let x = cholesky_solve(&a, &b, 2).expect("cholesky solve should succeed");
    assert!((x[0] - 3.0).abs() < 1e-10);
    assert!((x[1] - 5.0).abs() < 1e-10);
}

#[test]
fn test_ridge_solve_basic() {
    let x = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let y = vec![2.0, 3.0, 5.0];
    let w = ridge_solve(&x, &y, 1e-6).expect("ridge solve should succeed");
    assert_eq!(w.len(), 2);
    // Should approximately solve Xw = y.
    for wj in &w {
        assert!(wj.is_finite());
    }
}
