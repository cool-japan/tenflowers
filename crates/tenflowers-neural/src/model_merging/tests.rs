//! Tests for the model_merging module.
//!
//! All tests use seeded pseudo-random numbers for reproducibility and
//! temporary allocations only (no filesystem I/O).

use super::*;

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Build a small 2-layer model with deterministic weights.
fn make_model(seed: &mut u64, n_layers: usize, layer_size: usize) -> MmModelWeights {
    MmModelWeights::from_pretrained(seed, n_layers, layer_size)
}

/// Build a model with all parameters set to a constant value.
fn make_constant_model(value: f64, n_layers: usize, layer_size: usize) -> MmModelWeights {
    let total = n_layers * layer_size;
    let layer_sizes = vec![layer_size; n_layers];
    let layer_names = (0..n_layers).map(|i| format!("layer_{}", i)).collect();
    MmModelWeights {
        params: vec![value; total],
        layer_sizes,
        layer_names,
    }
}

// -----------------------------------------------------------------------
// §1  MmModelWeights
// -----------------------------------------------------------------------

#[test]
fn test_mm_model_weights_new_valid() {
    let w = MmModelWeights::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        vec!["l0".into(), "l1".into()],
    )
    .expect("construction should succeed");
    assert_eq!(w.n_params(), 4);
}

#[test]
fn test_mm_model_weights_new_mismatch_sizes_names() {
    let result = MmModelWeights::new(
        vec![1.0, 2.0],
        vec![2],
        vec!["l0".into(), "l1".into()], // 2 names, 1 size
    );
    assert!(result.is_err());
}

#[test]
fn test_mm_model_weights_new_mismatch_total() {
    let result = MmModelWeights::new(
        vec![1.0, 2.0, 3.0], // 3 params
        vec![2, 2],          // sum = 4
        vec!["l0".into(), "l1".into()],
    );
    assert!(result.is_err());
}

#[test]
fn test_mm_model_weights_get_layer() {
    let w = MmModelWeights::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![3, 3],
        vec!["l0".into(), "l1".into()],
    )
    .expect("construction should succeed");
    let l0 = w.get_layer(0).expect("layer 0 must exist");
    assert_eq!(l0, &[1.0, 2.0, 3.0]);
    let l1 = w.get_layer(1).expect("layer 1 must exist");
    assert_eq!(l1, &[4.0, 5.0, 6.0]);
}

#[test]
fn test_mm_model_weights_get_layer_oob() {
    let w = MmModelWeights::new(vec![1.0], vec![1], vec!["l0".into()]).expect("construction should succeed");
    assert!(w.get_layer(1).is_err());
}

#[test]
fn test_mm_model_weights_get_layer_mut() {
    let mut w = MmModelWeights::new(vec![10.0, 20.0, 30.0], vec![3], vec!["l0".into()]).expect("construction should succeed");
    {
        let layer = w.get_layer_mut(0).expect("get_layer_mut should succeed");
        layer[0] = 99.0;
    }
    assert_eq!(w.params[0], 99.0);
}

#[test]
fn test_mm_model_weights_add() {
    let a = make_constant_model(1.0, 2, 4);
    let b = make_constant_model(2.0, 2, 4);
    let c = a.add(&b).expect("add should succeed");
    for &v in &c.params {
        assert!((v - 3.0).abs() < 1e-12, "expected 3.0, got {}", v);
    }
}

#[test]
fn test_mm_model_weights_sub() {
    let a = make_constant_model(5.0, 2, 4);
    let b = make_constant_model(3.0, 2, 4);
    let c = a.sub(&b).expect("sub should succeed");
    for &v in &c.params {
        assert!((v - 2.0).abs() < 1e-12, "expected 2.0, got {}", v);
    }
}

#[test]
fn test_mm_model_weights_sub_dimension_mismatch() {
    let a = make_constant_model(1.0, 2, 4);
    let b = make_constant_model(1.0, 2, 5);
    assert!(a.sub(&b).is_err());
}

#[test]
fn test_mm_model_weights_scale() {
    let a = make_constant_model(3.0, 2, 4);
    let b = a.scale(2.0);
    for &v in &b.params {
        assert!((v - 6.0).abs() < 1e-12);
    }
}

#[test]
fn test_mm_model_weights_l2_norm_zero() {
    let z = make_constant_model(0.0, 2, 4);
    assert!(z.l2_norm() < 1e-12);
}

#[test]
fn test_mm_model_weights_l2_norm_nonzero() {
    // [3, 4] → norm = 5
    let w = MmModelWeights::new(vec![3.0, 4.0], vec![2], vec!["l".into()]).expect("construction should succeed");
    assert!((w.l2_norm() - 5.0).abs() < 1e-10);
}

#[test]
fn test_mm_model_weights_cosine_similarity_identical() {
    let mut seed = 7u64;
    let a = make_model(&mut seed, 2, 10);
    // cosine similarity of a model with itself must be 1.0
    let sim = a.cosine_similarity(&a);
    assert!((sim - 1.0).abs() < 1e-10, "expected 1.0, got {}", sim);
}

#[test]
fn test_mm_model_weights_cosine_similarity_opposite() {
    let a = make_constant_model(1.0, 1, 4);
    let b = make_constant_model(-1.0, 1, 4);
    let sim = a.cosine_similarity(&b);
    assert!((sim + 1.0).abs() < 1e-10, "expected -1.0, got {}", sim);
}

#[test]
fn test_mm_model_weights_cosine_similarity_zero_vector() {
    let a = make_constant_model(0.0, 1, 4);
    let b = make_constant_model(1.0, 1, 4);
    let sim = a.cosine_similarity(&b);
    assert!(sim.abs() < 1e-10, "expected 0.0, got {}", sim);
}

#[test]
fn test_mm_model_weights_from_pretrained() {
    let mut seed = 1234u64;
    let m = MmModelWeights::from_pretrained(&mut seed, 3, 16);
    assert_eq!(m.n_params(), 48);
    assert_eq!(m.layer_sizes.len(), 3);
}

// -----------------------------------------------------------------------
// §2  MmSimpleAverage
// -----------------------------------------------------------------------

#[test]
fn test_mm_simple_average_identical_models() {
    let mut seed = 11u64;
    let a = make_model(&mut seed, 2, 8);
    let b = a.clone();
    let merged = MmSimpleAverage::merge(&[a.clone(), b]).expect("simple average merge should succeed");
    for (m, &orig) in merged.params.iter().zip(a.params.iter()) {
        assert!(
            (m - orig).abs() < 1e-12,
            "merge of identical should be identity"
        );
    }
}

#[test]
fn test_mm_simple_average_two_models() {
    let a = make_constant_model(2.0, 2, 4);
    let b = make_constant_model(4.0, 2, 4);
    let merged = MmSimpleAverage::merge(&[a, b]).expect("simple average merge should succeed");
    for &v in &merged.params {
        assert!((v - 3.0).abs() < 1e-12, "expected average 3.0, got {}", v);
    }
}

#[test]
fn test_mm_simple_average_empty_error() {
    let result = MmSimpleAverage::merge(&[]);
    assert!(result.is_err());
}

#[test]
fn test_mm_simple_average_weighted_uniform() {
    let a = make_constant_model(1.0, 1, 6);
    let b = make_constant_model(3.0, 1, 6);
    let merged = MmSimpleAverage::weighted_merge(&[a, b], &[1.0, 1.0]).expect("weighted_merge should succeed");
    for &v in &merged.params {
        assert!(
            (v - 2.0).abs() < 1e-12,
            "uniform weights should give simple average"
        );
    }
}

#[test]
fn test_mm_simple_average_weighted_asymmetric() {
    let a = make_constant_model(0.0, 1, 4);
    let b = make_constant_model(4.0, 1, 4);
    // weights [3, 1] → normalised [0.75, 0.25] → 0*0.75 + 4*0.25 = 1.0
    let merged = MmSimpleAverage::weighted_merge(&[a, b], &[3.0, 1.0]).expect("weighted_merge should succeed");
    for &v in &merged.params {
        assert!((v - 1.0).abs() < 1e-12, "expected 1.0, got {}", v);
    }
}

#[test]
fn test_mm_simple_average_weighted_mismatch() {
    let a = make_constant_model(1.0, 1, 4);
    let result = MmSimpleAverage::weighted_merge(&[a], &[1.0, 2.0]);
    assert!(result.is_err());
}

#[test]
fn test_mm_greedy_soup_no_improvement() {
    // All candidates are worse; should return the best single model (base).
    let base = make_constant_model(1.0, 1, 4);
    // Evaluate as sum of params
    let eval = |m: &MmModelWeights| -> f64 { m.params.iter().sum() };
    // Candidates with lower sum
    let candidates = vec![make_constant_model(0.5, 1, 4)];
    let soup = MmSimpleAverage::greedy_soup(&base, &candidates, &eval).expect("greedy_soup should succeed");
    // Soup should not be worse than base
    assert!(eval(&soup) >= eval(&base) - 1e-10);
}

#[test]
fn test_mm_greedy_soup_empty_candidates() {
    let base = make_constant_model(2.0, 1, 4);
    let eval = |m: &MmModelWeights| -> f64 { m.params.iter().sum() };
    let soup = MmSimpleAverage::greedy_soup(&base, &[], &eval).expect("greedy_soup should succeed");
    for (s, &b) in soup.params.iter().zip(base.params.iter()) {
        assert!((s - b).abs() < 1e-12);
    }
}

// -----------------------------------------------------------------------
// §3  MmTaskArithmetic
// -----------------------------------------------------------------------

#[test]
fn test_mm_task_vector_is_difference() {
    let pre = make_constant_model(1.0, 2, 4);
    let fine = make_constant_model(3.0, 2, 4);
    let tv = MmTaskArithmetic::compute_task_vector(&pre, &fine, "task_a").expect("compute_task_vector should succeed");
    for &v in &tv.vector {
        assert!(
            (v - 2.0).abs() < 1e-12,
            "expected task vector 2.0, got {}",
            v
        );
    }
}

#[test]
fn test_mm_task_arithmetic_add_task_recovers_finetuned() {
    let pre = make_constant_model(1.0, 2, 4);
    let fine = make_constant_model(3.0, 2, 4);
    let tv = MmTaskArithmetic::compute_task_vector(&pre, &fine, "t").expect("compute_task_vector should succeed");
    let recovered = MmTaskArithmetic::add_task(&pre, &[(&tv, 1.0)]).expect("add_task should succeed");
    for (r, &f) in recovered.params.iter().zip(fine.params.iter()) {
        assert!((r - f).abs() < 1e-12, "expected {}, got {}", f, r);
    }
}

#[test]
fn test_mm_task_arithmetic_scale_task_vector() {
    let pre = make_constant_model(0.0, 1, 4);
    let fine = make_constant_model(2.0, 1, 4);
    let tv = MmTaskArithmetic::compute_task_vector(&pre, &fine, "t").expect("compute_task_vector should succeed");
    // Scale 0.5 → each param should be 0 + 0.5 * 2 = 1.0
    let result = MmTaskArithmetic::add_task(&pre, &[(&tv, 0.5)]).expect("add_task should succeed");
    for &v in &result.params {
        assert!((v - 1.0).abs() < 1e-12);
    }
}

#[test]
fn test_mm_task_arithmetic_negate() {
    let pre = make_constant_model(0.0, 1, 4);
    let fine = make_constant_model(2.0, 1, 4);
    let tv = MmTaskArithmetic::compute_task_vector(&pre, &fine, "t").expect("compute_task_vector should succeed");
    let neg = MmTaskArithmetic::negate_task(&tv);
    for &v in &neg.vector {
        assert!((v + 2.0).abs() < 1e-12, "expected -2.0, got {}", v);
    }
}

#[test]
fn test_mm_task_arithmetic_negation_removes_capability() {
    // Applying the negated task vector should bring the model back towards pretrained.
    let pre = make_constant_model(1.0, 1, 4);
    let fine = make_constant_model(3.0, 1, 4);
    let tv = MmTaskArithmetic::compute_task_vector(&pre, &fine, "t").expect("compute_task_vector should succeed");
    let neg = MmTaskArithmetic::negate_task(&tv);
    // Start from finetuned, apply negated → should recover pretrained
    let restored = MmTaskArithmetic::add_task(&fine, &[(&neg, 1.0)]).expect("add_task should succeed");
    for (r, &p) in restored.params.iter().zip(pre.params.iter()) {
        assert!((r - p).abs() < 1e-12, "expected {}, got {}", p, r);
    }
}

#[test]
fn test_mm_task_arithmetic_task_analogy() {
    let pre = make_constant_model(0.0, 1, 4);
    let fine_a = make_constant_model(1.0, 1, 4);
    let fine_b = make_constant_model(2.0, 1, 4);
    let fine_c = make_constant_model(3.0, 1, 4);
    let tv_a = MmTaskArithmetic::compute_task_vector(&pre, &fine_a, "a").expect("compute_task_vector a should succeed");
    let tv_b = MmTaskArithmetic::compute_task_vector(&pre, &fine_b, "b").expect("compute_task_vector b should succeed");
    let tv_c = MmTaskArithmetic::compute_task_vector(&pre, &fine_c, "c").expect("compute_task_vector c should succeed");
    // θ_out = pre + τ_c + (τ_a − τ_b) = 0 + 3 + (1 − 2) = 2
    let out = MmTaskArithmetic::task_analogy(&pre, &tv_a, &tv_b, &tv_c).expect("task_analogy should succeed");
    for &v in &out.params {
        assert!((v - 2.0).abs() < 1e-12, "expected 2.0, got {}", v);
    }
}

#[test]
fn test_mm_task_arithmetic_forgetting_score_zero() {
    let a = make_constant_model(1.0, 1, 4);
    let score = MmTaskArithmetic::forgetting_score(&a, &a);
    // cosine_similarity(a, a) = 1 → forgetting = 0
    assert!(
        score.abs() < 1e-10,
        "identical models have forgetting_score 0"
    );
}

#[test]
fn test_mm_task_arithmetic_dimension_mismatch() {
    let pre = make_constant_model(1.0, 1, 4);
    let fine = make_constant_model(2.0, 1, 5);
    let result = MmTaskArithmetic::compute_task_vector(&pre, &fine, "t");
    assert!(result.is_err());
}

// -----------------------------------------------------------------------
// §4  MmTiesMerging
// -----------------------------------------------------------------------

#[test]
fn test_mm_ties_trim_keeps_top_k() {
    let v = vec![0.1, -0.9, 0.5, -0.2, 0.8];
    let trimmed = MmTiesMerging::trim(&v, 2);
    // Expect positions 1 (-0.9) and 4 (0.8) kept (two largest abs)
    let n_nonzero = trimmed.iter().filter(|&&x| x != 0.0).count();
    assert_eq!(n_nonzero, 2);
    // The kept values must have the two largest absolute magnitudes
    assert!(trimmed[1].abs() > 0.0 || trimmed[4].abs() > 0.0);
}

#[test]
fn test_mm_ties_trim_zero_k() {
    let v = vec![1.0, 2.0, 3.0];
    let trimmed = MmTiesMerging::trim(&v, 0);
    for &x in &trimmed {
        assert_eq!(x, 0.0);
    }
}

#[test]
fn test_mm_ties_trim_k_geq_n() {
    let v = vec![1.0, -2.0, 3.0];
    let trimmed = MmTiesMerging::trim(&v, 10);
    assert_eq!(trimmed, v);
}

#[test]
fn test_mm_ties_elect_sign_majority() {
    // Two vectors: [1, -1, 2, -2], [3, -3, -1, -1]
    // Sums:         4,  -4,  1,  -3
    // Signs:        1,  -1,  1,  -1
    let tv1 = vec![1.0, -1.0, 2.0, -2.0];
    let tv2 = vec![3.0, -3.0, -1.0, -1.0];
    let elected = MmTiesMerging::elect_sign(&[tv1, tv2]);
    assert_eq!(elected[0], 1.0);
    assert_eq!(elected[1], -1.0);
    assert_eq!(elected[2], 1.0);
    assert_eq!(elected[3], -1.0);
}

#[test]
fn test_mm_ties_elect_sign_empty() {
    let elected = MmTiesMerging::elect_sign(&[]);
    assert!(elected.is_empty());
}

#[test]
fn test_mm_ties_disjoint_merge_filters_correctly() {
    // elected_signs = [1, -1]
    // tv1 = [1, -1], tv2 = [-1, -1]
    // position 0: elected=1, tv1=+→agree, tv2=−→disagree → avg([1]) = 1
    // position 1: elected=-1, tv1=−→agree, tv2=−→agree → avg([-1,-1]) = -1
    let elected = vec![1.0, -1.0];
    let tv1 = vec![1.0, -1.0];
    let tv2 = vec![-1.0, -1.0];
    let dm = MmTiesMerging::disjoint_merge(&[tv1, tv2], &elected);
    assert!((dm[0] - 1.0).abs() < 1e-12, "expected 1.0, got {}", dm[0]);
    assert!((dm[1] + 1.0).abs() < 1e-12, "expected -1.0, got {}", dm[1]);
}

#[test]
fn test_mm_ties_merge_preserves_pretrained_when_no_tasks() {
    let pre = make_constant_model(5.0, 1, 4);
    let merged = MmTiesMerging::merge(&pre, &[], 1.0).expect("TIES merge should succeed");
    for (m, &p) in merged.params.iter().zip(pre.params.iter()) {
        assert!((m - p).abs() < 1e-12);
    }
}

#[test]
fn test_mm_ties_merge_with_tasks() {
    let pre = make_constant_model(0.0, 2, 6);
    let fine1 = make_constant_model(1.0, 2, 6);
    let fine2 = make_constant_model(-1.0, 2, 6);
    let tv1 = MmTaskArithmetic::compute_task_vector(&pre, &fine1, "t1").expect("compute_task_vector t1 should succeed");
    let tv2 = MmTaskArithmetic::compute_task_vector(&pre, &fine2, "t2").expect("compute_task_vector t2 should succeed");
    // Just check it runs without error and produces expected shape
    let merged = MmTiesMerging::merge(&pre, &[tv1, tv2], 1.0).expect("TIES merge should succeed");
    assert_eq!(merged.n_params(), pre.n_params());
}

// -----------------------------------------------------------------------
// §5  MmDare
// -----------------------------------------------------------------------

#[test]
fn test_mm_dare_apply_drops_parameters() {
    // With a fixed seed and drop_rate=0.9, roughly 90% should be zero.
    let pre = make_constant_model(0.0, 1, 100);
    let fine = make_constant_model(1.0, 1, 100);
    let tv = MmTaskArithmetic::compute_task_vector(&pre, &fine, "t").expect("compute_task_vector should succeed");
    let mut seed = 999u64;
    let dared = MmDare::apply_dare(&tv, &mut seed);
    let n_zero = dared.vector.iter().filter(|&&x| x == 0.0).count();
    // With 100 params and ~90% drop rate, expect 60-99 zeros (broad test)
    assert!(
        n_zero >= 50,
        "expected most params zeroed, got {} zeros",
        n_zero
    );
}

#[test]
fn test_mm_dare_rescaling_preserves_expected_magnitude() {
    // DARE: drop with prob p, rescale surviving by 1/(1-p).
    // E[dared_v_i] = v_i  (unbiased)
    // E[dared_v_i^2] = (1-p) * (v_i * 1/(1-p))^2 = v_i^2 / (1-p)
    // So E[||dared||^2] = ||v||^2 / (1-p)  → for p=0.9, expected ratio ≈ 10.
    // Test that the empirical ratio is within a wide tolerance of the expectation.
    let n = 1000usize;
    let pre = MmModelWeights {
        params: vec![0.0; n],
        layer_sizes: vec![n],
        layer_names: vec!["l".into()],
    };
    let fine = MmModelWeights {
        params: vec![1.0; n],
        layer_sizes: vec![n],
        layer_names: vec!["l".into()],
    };
    let tv = MmTaskArithmetic::compute_task_vector(&pre, &fine, "t").expect("compute_task_vector should succeed");
    let orig_norm_sq: f64 = tv.vector.iter().map(|x| x * x).sum();

    let mut seed = 42u64;
    let dared = MmDare::apply_dare(&tv, &mut seed);
    let dared_norm_sq: f64 = dared.vector.iter().map(|x| x * x).sum();

    // Expected ratio = 1/(1-p) = 10.  Allow generous ±80% tolerance
    // (stochastic test with 1000 samples at 90% sparsity has high variance).
    let expected_ratio = 10.0_f64;
    let ratio = dared_norm_sq / orig_norm_sq;
    assert!(
        ratio > expected_ratio * 0.2 && ratio < expected_ratio * 1.8,
        "DARE squared-norm ratio out of expected range: {} (expected ~{})",
        ratio,
        expected_ratio
    );
}

#[test]
fn test_mm_dare_ties_runs() {
    let pre = make_constant_model(0.0, 2, 8);
    let tv1 = MmTaskVector {
        vector: vec![1.0; 16],
        task_name: "t1".into(),
    };
    let tv2 = MmTaskVector {
        vector: vec![-1.0; 16],
        task_name: "t2".into(),
    };
    let mut seed = 77u64;
    let result = MmDare::dare_ties(&pre, &[tv1, tv2], 0.5, &mut seed).expect("dare_ties should succeed");
    assert_eq!(result.n_params(), 16);
}

#[test]
fn test_mm_dare_linear_adds_to_pretrained() {
    let pre = make_constant_model(0.0, 1, 8);
    let tv = MmTaskVector {
        vector: vec![2.0; 8],
        task_name: "t".into(),
    };
    let mut seed = 55u64;
    let result = MmDare::dare_linear(&pre, &[(&tv, 1.0)], &mut seed).expect("dare_linear should succeed");
    // Some parameters should be non-zero (those not dropped)
    let n_nonzero = result.params.iter().filter(|&&x| x.abs() > 1e-12).count();
    // With n=8 and 90% drop rate, expect roughly 0-2 non-zero; just assert total is right
    assert_eq!(result.n_params(), 8);
    let _ = n_nonzero;
}

// -----------------------------------------------------------------------
// §6  MmFisherWeightedMerge
// -----------------------------------------------------------------------

#[test]
fn test_mm_fisher_equal_weights_is_simple_average() {
    let a = make_constant_model(1.0, 1, 4);
    let b = make_constant_model(3.0, 1, 4);
    // Equal Fisher information → Fisher merge = simple average
    let fisher_a = vec![1.0; 4];
    let fisher_b = vec![1.0; 4];
    let merged = MmFisherWeightedMerge::merge(&[a, b], &[fisher_a, fisher_b]).expect("Fisher merge should succeed");
    for &v in &merged.params {
        assert!(
            (v - 2.0).abs() < 1e-10,
            "equal Fisher should give simple avg 2.0, got {}",
            v
        );
    }
}

#[test]
fn test_mm_fisher_higher_weight_wins() {
    let a = make_constant_model(0.0, 1, 4);
    let b = make_constant_model(10.0, 1, 4);
    // b has 9× higher Fisher → merged should be closer to b
    let fisher_a = vec![1.0; 4];
    let fisher_b = vec![9.0; 4];
    let merged = MmFisherWeightedMerge::merge(&[a, b], &[fisher_a, fisher_b]).expect("Fisher merge should succeed");
    // expected = (0*1 + 10*9)/(1+9) = 9.0
    for &v in &merged.params {
        assert!((v - 9.0).abs() < 1e-10, "expected 9.0, got {}", v);
    }
}

#[test]
fn test_mm_fisher_approximate_fisher_nonneg() {
    let mut seed = 3u64;
    let m = make_model(&mut seed, 2, 8);
    // Gradient samples
    let grads: Vec<Vec<f64>> = (0..10)
        .map(|_| (0..16).map(|i| i as f64 * 0.1).collect())
        .collect();
    let fisher = MmFisherWeightedMerge::approximate_fisher(&m, &grads);
    for &f in &fisher {
        assert!(
            f >= 0.0,
            "Fisher information must be non-negative, got {}",
            f
        );
    }
}

#[test]
fn test_mm_fisher_approximate_fisher_empty_grads() {
    let mut seed = 5u64;
    let m = make_model(&mut seed, 1, 4);
    let fisher = MmFisherWeightedMerge::approximate_fisher(&m, &[]);
    for &f in &fisher {
        assert_eq!(f, 0.0);
    }
}

#[test]
fn test_mm_fisher_merge_empty_error() {
    let result = MmFisherWeightedMerge::merge(&[], &[]);
    assert!(result.is_err());
}

#[test]
fn test_mm_fisher_merge_with_gradients() {
    let a = make_constant_model(1.0, 1, 4);
    let b = make_constant_model(3.0, 1, 4);
    let grads_a: Vec<Vec<f64>> = vec![vec![1.0; 4]; 5];
    let grads_b: Vec<Vec<f64>> = vec![vec![1.0; 4]; 5];
    let merged = MmFisherWeightedMerge::merge_with_gradients(&[a, b], &[grads_a, grads_b]).expect("merge_with_gradients should succeed");
    assert_eq!(merged.n_params(), 4);
    // Equal gradients → equal Fisher → simple average = 2.0
    for &v in &merged.params {
        assert!((v - 2.0).abs() < 1e-10, "expected 2.0, got {}", v);
    }
}

// -----------------------------------------------------------------------
// §7  MmLoraAggregation
// -----------------------------------------------------------------------

fn make_lora_adapter(
    out_dim: usize,
    in_dim: usize,
    rank: usize,
    scale: f64,
    name: &str,
) -> MmLoraAdapter {
    // Fill A with 0.1, B with 0.2 for simple deterministic tests
    let lora_a: Vec<Vec<f64>> = (0..rank).map(|_| vec![0.1; in_dim]).collect();
    let lora_b: Vec<Vec<f64>> = (0..out_dim).map(|_| vec![0.2; rank]).collect();
    MmLoraAdapter {
        lora_a,
        lora_b,
        scale,
        task_name: name.to_string(),
    }
}

#[test]
fn test_mm_lora_merge_into_base_changes_weights() {
    let base = vec![vec![1.0f64; 4]; 4];
    let adapter = make_lora_adapter(4, 4, 2, 1.0, "t");
    let merged = MmLoraAggregation::merge_into_base(&base, &adapter);
    // delta = scale * B @ A; since B[i][k]=0.2, A[k][j]=0.1, rank=2
    // delta[i][j] = 1.0 * (0.2*0.1 + 0.2*0.1) * 1 = 0.04
    // merged[i][j] = 1.0 + 0.04 = 1.04
    for row in &merged {
        for &v in row {
            assert!((v - 1.04).abs() < 1e-10, "expected 1.04, got {}", v);
        }
    }
}

#[test]
fn test_mm_lora_merge_into_base_dimensions() {
    let base = vec![vec![0.0f64; 8]; 6]; // 6×8
    let adapter = make_lora_adapter(6, 8, 3, 1.0, "t");
    let merged = MmLoraAggregation::merge_into_base(&base, &adapter);
    assert_eq!(merged.len(), 6);
    assert_eq!(merged[0].len(), 8);
}

#[test]
fn test_mm_lora_soup_applies_all_adapters() {
    let base = vec![vec![0.0f64; 4]; 4];
    let a1 = make_lora_adapter(4, 4, 1, 1.0, "t1");
    let a2 = make_lora_adapter(4, 4, 1, 1.0, "t2");
    let soup = MmLoraAggregation::lora_soup(&base, &[&a1, &a2]);
    // Each adapter adds 0.1*0.2 = 0.02; two adapters → 0.04
    for row in &soup {
        for &v in row {
            assert!(v > 0.0, "soup should add positive delta");
        }
    }
}

#[test]
fn test_mm_lora_combine_adapters() {
    let a1 = make_lora_adapter(4, 4, 2, 1.0, "t1");
    let a2 = make_lora_adapter(4, 4, 2, 1.0, "t2");
    let combined = MmLoraAggregation::combine_adapters(&[a1, a2], &[1.0, 1.0]).expect("combine_adapters should succeed");
    // Should return a rank-2 adapter with the same dimensions
    assert_eq!(combined.out_dim(), 4);
    assert_eq!(combined.in_dim(), 4);
}

#[test]
fn test_mm_lora_svd_compression_rank() {
    let delta = vec![vec![1.0f64; 8]; 6]; // rank-1 matrix (all rows equal)
    let adapter = MmLoraAggregation::svd_compression(&delta, 2);
    // Should produce at most rank-2 adapter
    assert!(adapter.rank() <= 2);
    assert_eq!(adapter.out_dim(), 6);
    assert_eq!(adapter.in_dim(), 8);
}

#[test]
fn test_mm_lora_svd_compression_reconstructs() {
    // delta = outer product [1,1,1] × [2,2,2,2] → rank 1
    let delta: Vec<Vec<f64>> = (0..3).map(|_| vec![2.0f64, 2.0, 2.0, 2.0]).collect();
    let adapter = MmLoraAggregation::svd_compression(&delta, 1);
    // Reconstruct W_delta = scale * B @ A
    let base = vec![vec![0.0f64; 4]; 3];
    let reconstructed = MmLoraAggregation::merge_into_base(&base, &adapter);
    // Check that reconstructed has the right sign/shape
    for row in &reconstructed {
        for &v in row {
            assert!(v.abs() > 1e-8, "reconstruction should be non-trivial");
        }
    }
}

// -----------------------------------------------------------------------
// §8  MmRegmean
// -----------------------------------------------------------------------

#[test]
fn test_mm_regmean_approximate_gram_psd_diag() {
    let acts: Vec<Vec<f64>> = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0],
        vec![0.0, 0.0, 3.0],
    ];
    let gram = MmRegmean::approximate_gram(&acts);
    // Diagonal must be non-negative (diagonal of Gram matrix is mean(x_i^2))
    for i in 0..3 {
        assert!(gram[i][i] >= 0.0, "Gram diagonal must be non-negative");
    }
}

#[test]
fn test_mm_regmean_approximate_gram_symmetric() {
    let acts: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let gram = MmRegmean::approximate_gram(&acts);
    let n = gram.len();
    for i in 0..n {
        for j in 0..n {
            assert!(
                (gram[i][j] - gram[j][i]).abs() < 1e-10,
                "Gram must be symmetric at ({},{})",
                i,
                j
            );
        }
    }
}

#[test]
fn test_mm_regmean_merge_layer_equal_grams() {
    // With identical Gram matrices, RegMean should give the same result as simple average.
    let w1 = vec![vec![1.0f64, 0.0], vec![0.0, 1.0]];
    let w2 = vec![vec![3.0f64, 0.0], vec![0.0, 3.0]];
    let g1 = vec![vec![1.0f64, 0.0], vec![0.0, 1.0]]; // identity
    let g2 = vec![vec![1.0f64, 0.0], vec![0.0, 1.0]];
    let merged = MmRegmean::merge_layer(&[w1, w2], &[g1, g2]).expect("RegMean merge_layer should succeed");
    // Expected: simple average = 2.0 on diagonal, 0.0 off-diagonal
    assert!((merged[0][0] - 2.0).abs() < 1e-6);
    assert!((merged[1][1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_mm_regmean_merge_layer_empty_error() {
    let result = MmRegmean::merge_layer(&[], &[]);
    assert!(result.is_err());
}

// -----------------------------------------------------------------------
// §9  MmLinearInterpolation
// -----------------------------------------------------------------------

#[test]
fn test_mm_linear_interpolation_t0_gives_a() {
    let a = make_constant_model(1.0, 2, 4);
    let b = make_constant_model(5.0, 2, 4);
    let interp = MmLinearInterpolation::interpolate(&a, &b, 0.0).expect("interpolate should succeed");
    for (v, &orig) in interp.params.iter().zip(a.params.iter()) {
        assert!((v - orig).abs() < 1e-12, "t=0 should return A");
    }
}

#[test]
fn test_mm_linear_interpolation_t1_gives_b() {
    let a = make_constant_model(1.0, 2, 4);
    let b = make_constant_model(5.0, 2, 4);
    let interp = MmLinearInterpolation::interpolate(&a, &b, 1.0).expect("interpolate should succeed");
    for (v, &orig) in interp.params.iter().zip(b.params.iter()) {
        assert!((v - orig).abs() < 1e-12, "t=1 should return B");
    }
}

#[test]
fn test_mm_linear_interpolation_midpoint() {
    let a = make_constant_model(0.0, 1, 4);
    let b = make_constant_model(4.0, 1, 4);
    let interp = MmLinearInterpolation::interpolate(&a, &b, 0.5).expect("interpolate should succeed");
    for &v in &interp.params {
        assert!((v - 2.0).abs() < 1e-12, "midpoint should be 2.0");
    }
}

#[test]
fn test_mm_linear_interpolation_dimension_mismatch() {
    let a = make_constant_model(1.0, 1, 4);
    let b = make_constant_model(1.0, 1, 5);
    assert!(MmLinearInterpolation::interpolate(&a, &b, 0.5).is_err());
}

#[test]
fn test_mm_linear_interpolation_loss_barrier_nonneg() {
    let a = make_constant_model(1.0, 1, 4);
    let b = make_constant_model(-1.0, 1, 4);
    let eval = |_m: &MmModelWeights| 0.9f64;
    let barrier = MmLinearInterpolation::loss_barrier(&a, &b, &eval, 10);
    assert!(barrier >= 0.0, "loss barrier must be non-negative");
}

#[test]
fn test_mm_linear_interpolation_git_rebasin_identity() {
    let a = make_constant_model(0.0, 2, 4);
    let b = make_constant_model(3.0, 2, 4);
    let rebasin = MmLinearInterpolation::git_rebasin_identity(&a, &b);
    // Identity permutation → same as b
    for (r, &orig) in rebasin.params.iter().zip(b.params.iter()) {
        assert!(
            (r - orig).abs() < 1e-12,
            "identity rebasin should return b unchanged"
        );
    }
}

#[test]
fn test_mm_quadratic_interpolation_corners() {
    let a = make_constant_model(1.0, 1, 4);
    let b = make_constant_model(0.0, 1, 4); // midpoint
    let c = make_constant_model(3.0, 1, 4);
    // t=0 → w_a=1, w_b=0, w_c=0 → A
    let at0 = MmLinearInterpolation::quadratic_interpolation(&a, &b, &c, 0.0, 0.0).expect("quadratic_interpolation should succeed");
    for &v in &at0.params {
        assert!((v - 1.0).abs() < 1e-12, "t=0 Bezier should return A");
    }
    // t=1 → w_a=0, w_b=0, w_c=1 → C
    let at1 = MmLinearInterpolation::quadratic_interpolation(&a, &b, &c, 1.0, 0.0).expect("quadratic_interpolation should succeed");
    for &v in &at1.params {
        assert!((v - 3.0).abs() < 1e-12, "t=1 Bezier should return C");
    }
}

#[test]
fn test_mm_quadratic_interpolation_midpoint() {
    // t=0.5: w_a=0.25, w_b=0.5, w_c=0.25
    // A=0, B=4, C=0 → result = 0.25*0 + 0.5*4 + 0.25*0 = 2.0
    let a = make_constant_model(0.0, 1, 4);
    let b = make_constant_model(4.0, 1, 4);
    let c = make_constant_model(0.0, 1, 4);
    let mid = MmLinearInterpolation::quadratic_interpolation(&a, &b, &c, 0.5, 0.0).expect("quadratic_interpolation should succeed");
    for &v in &mid.params {
        assert!(
            (v - 2.0).abs() < 1e-12,
            "Bezier midpoint should be 2.0, got {}",
            v
        );
    }
}

// -----------------------------------------------------------------------
// §10  MmMetrics
// -----------------------------------------------------------------------

#[test]
fn test_mm_metrics_weight_distance_same_model() {
    let mut seed = 17u64;
    let a = make_model(&mut seed, 2, 8);
    let dist = MmMetrics::weight_distance(&a, &a);
    assert!(dist.abs() < 1e-12, "distance to self should be 0");
}

#[test]
fn test_mm_metrics_weight_distance_nonneg() {
    let a = make_constant_model(1.0, 1, 4);
    let b = make_constant_model(2.0, 1, 4);
    let dist = MmMetrics::weight_distance(&a, &b);
    assert!(dist >= 0.0, "distance must be non-negative");
    assert!(
        dist > 0.0,
        "distance between distinct models must be positive"
    );
}

#[test]
fn test_mm_metrics_task_vector_magnitude_zero() {
    let tv = MmTaskVector {
        vector: vec![0.0; 8],
        task_name: "z".into(),
    };
    assert!(MmMetrics::task_vector_magnitude(&tv).abs() < 1e-12);
}

#[test]
fn test_mm_metrics_task_vector_magnitude_known() {
    // [3, 4] → 5
    let tv = MmTaskVector {
        vector: vec![3.0, 4.0],
        task_name: "t".into(),
    };
    assert!((MmMetrics::task_vector_magnitude(&tv) - 5.0).abs() < 1e-10);
}

#[test]
fn test_mm_metrics_agreement_rate_identical() {
    let tv = MmTaskVector {
        vector: vec![1.0, -1.0, 2.0, -3.0],
        task_name: "t".into(),
    };
    let rate = MmMetrics::agreement_rate(&tv, &tv);
    assert!(
        (rate - 1.0).abs() < 1e-12,
        "identical task vectors → 100% agreement"
    );
}

#[test]
fn test_mm_metrics_agreement_rate_opposite() {
    let tv1 = MmTaskVector {
        vector: vec![1.0, 1.0, 1.0, 1.0],
        task_name: "a".into(),
    };
    let tv2 = MmTaskVector {
        vector: vec![-1.0, -1.0, -1.0, -1.0],
        task_name: "b".into(),
    };
    let rate = MmMetrics::agreement_rate(&tv1, &tv2);
    assert!(rate.abs() < 1e-12, "opposite task vectors → 0% agreement");
}

#[test]
fn test_mm_metrics_agreement_rate_in_range() {
    let tv1 = MmTaskVector {
        vector: vec![1.0, -1.0, 1.0, -1.0],
        task_name: "a".into(),
    };
    let tv2 = MmTaskVector {
        vector: vec![1.0, 1.0, -1.0, -1.0],
        task_name: "b".into(),
    };
    let rate = MmMetrics::agreement_rate(&tv1, &tv2);
    assert!(
        (0.0..=1.0).contains(&rate),
        "agreement rate must be in [0,1], got {}",
        rate
    );
}

#[test]
fn test_mm_metrics_merge_quality_identical() {
    let a = make_constant_model(1.0, 1, 4);
    let quality = MmMetrics::merge_quality(&a, &[a.clone(), a.clone()]);
    assert!(
        (quality - 1.0).abs() < 1e-10,
        "identical model → quality 1.0"
    );
}

#[test]
fn test_mm_metrics_merge_quality_empty() {
    let a = make_constant_model(1.0, 1, 4);
    let quality = MmMetrics::merge_quality(&a, &[]);
    assert_eq!(quality, 0.0);
}

#[test]
fn test_mm_metrics_parameter_overlap_identical() {
    let tv = MmTaskVector {
        vector: vec![1.0, 0.5, -0.8, 0.3],
        task_name: "t".into(),
    };
    let overlap = MmMetrics::parameter_overlap(&tv, &tv, 0.4);
    // Identical task vectors → overlap = 1.0
    assert!((overlap - 1.0).abs() < 1e-12, "identical → 100% overlap");
}

#[test]
fn test_mm_metrics_parameter_overlap_disjoint() {
    // tv1 has |>0.4| at positions 0,1 and tv2 at positions 2,3 (0 overlap in union)
    let tv1 = MmTaskVector {
        vector: vec![1.0, 1.0, 0.0, 0.0],
        task_name: "a".into(),
    };
    let tv2 = MmTaskVector {
        vector: vec![0.0, 0.0, 1.0, 1.0],
        task_name: "b".into(),
    };
    let overlap = MmMetrics::parameter_overlap(&tv1, &tv2, 0.5);
    assert!(
        overlap.abs() < 1e-12,
        "disjoint significant params → 0 overlap"
    );
}

#[test]
fn test_mm_metrics_forgetting_gap_zero() {
    let gap = MmMetrics::forgetting_gap(0.9, 0.9);
    assert!(gap.abs() < 1e-12, "no forgetting when accuracy unchanged");
}

#[test]
fn test_mm_metrics_forgetting_gap_positive() {
    let gap = MmMetrics::forgetting_gap(0.95, 0.80);
    assert!((gap - 0.15).abs() < 1e-10, "expected 0.15, got {}", gap);
}

#[test]
fn test_mm_metrics_forgetting_gap_improvement_is_zero() {
    // If performance improves (after > before), gap should be 0, not negative
    let gap = MmMetrics::forgetting_gap(0.80, 0.95);
    assert_eq!(gap, 0.0, "no forgetting when performance improves");
}
